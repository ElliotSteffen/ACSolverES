"""Process the paired 124-presentation W&B export and create paper figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ac_unsolved124_matplotlib"),
)


AUTO_DUAL = "auto_dual"
TRIPLE_GS = "triple_gs"
METHOD_ORDER = (AUTO_DUAL, TRIPLE_GS)
METHOD_LABELS = {
    AUTO_DUAL: "Auto Dual GS",
    TRIPLE_GS: "Triple GS",
}
SOURCE_ALGORITHMS = {
    "standard_gs_sub_plus_auto_cov_every_pop_compact_cpp": AUTO_DUAL,
    "triple_gs_auto_cov_complete_cov_compact_cpp": TRIPLE_GS,
}
COLORS = {
    AUTO_DUAL: "#0B6E75",
    TRIPLE_GS: "#E76F51",
    "tie": "#8A8F98",
    "substitution": "#264653",
    "automorphism": "#2A9D8F",
    "non_auto": "#E9C46A",
}


def _require_columns(fieldnames: Sequence[str] | None, required: Iterable[str]) -> None:
    available = set(fieldnames or ())
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _as_int(row: Mapping[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in {"", "null", "None"}:
        raise ValueError(f"Missing integer field {key!r} for presentation {row.get('label')!r}")
    return int(float(value))


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _best_length(row: Mapping[str, str]) -> tuple[int, str]:
    minimum = row.get("minimum_total_length", "")
    if minimum not in {"", "null", "None"}:
        return int(float(minimum)), "minimum_total_length"
    return _as_int(row, "final_total_length"), "final_total_length"


def load_paired_results(input_csv: str | Path) -> list[Dict[str, Any]]:
    """Load, validate, and pair Auto Dual and Triple GS by presentation index."""

    input_csv = Path(input_csv).expanduser().resolve()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader.fieldnames,
            {
                "algorithm",
                "presentation_index",
                "label",
                "r1",
                "r2",
                "start_total_length",
                "final_total_length",
                "total_nodes_used",
                "solved",
                "termination_reason",
                "sub_pop_count",
                "cov_pop_count",
            },
        )
        raw_rows = list(reader)

    by_method: Dict[str, Dict[int, Dict[str, str]]] = {
        method: {} for method in METHOD_ORDER
    }
    unknown = Counter()
    for row in raw_rows:
        source_algorithm = row["algorithm"]
        method = SOURCE_ALGORITHMS.get(source_algorithm)
        if method is None:
            unknown[source_algorithm] += 1
            continue
        presentation_index = _as_int(row, "presentation_index")
        if presentation_index in by_method[method]:
            raise ValueError(
                f"Duplicate {METHOD_LABELS[method]} row for presentation {presentation_index}."
            )
        by_method[method][presentation_index] = row

    if unknown:
        raise ValueError(f"Unexpected algorithm rows in input CSV: {dict(unknown)}")

    index_sets = {method: set(rows) for method, rows in by_method.items()}
    if not index_sets[AUTO_DUAL] or index_sets[AUTO_DUAL] != index_sets[TRIPLE_GS]:
        raise ValueError(
            "Auto Dual and Triple GS must contain the same nonempty presentation indices."
        )
    if len(index_sets[AUTO_DUAL]) != 124:
        raise ValueError(
            f"Expected 124 paired presentations, found {len(index_sets[AUTO_DUAL])}."
        )

    paired: list[Dict[str, Any]] = []
    for presentation_index in sorted(index_sets[AUTO_DUAL]):
        dual = by_method[AUTO_DUAL][presentation_index]
        triple = by_method[TRIPLE_GS][presentation_index]
        for key in ("label", "r1", "r2", "start_total_length"):
            if dual[key] != triple[key]:
                raise ValueError(
                    f"Presentation {presentation_index} differs between runs in {key!r}."
                )

        dual_best, dual_best_source = _best_length(dual)
        triple_best, triple_best_source = _best_length(triple)
        start_length = _as_int(dual, "start_total_length")
        dual_nodes = _as_int(dual, "total_nodes_used")
        triple_nodes = _as_int(triple, "total_nodes_used")
        if dual_nodes != triple_nodes:
            raise ValueError(
                f"Presentation {presentation_index} used unequal node budgets: "
                f"{dual_nodes} versus {triple_nodes}."
            )

        triple_sub_pops = _as_int(triple, "sub_pop_count")
        triple_auto_pops = _as_int(triple, "cov_pop_count")
        explicit_complete = triple.get("complete_cov_queue_pop_count", "")
        if explicit_complete not in {"", "null", "None"}:
            triple_non_auto_pops = int(float(explicit_complete))
            non_auto_source = "complete_cov_queue_pop_count"
        else:
            triple_non_auto_pops = triple_nodes - triple_sub_pops - triple_auto_pops
            non_auto_source = "total_nodes_used - sub_pop_count - cov_pop_count"
        if triple_non_auto_pops < 0:
            raise ValueError(
                f"Negative inferred Non-Auto COV pop count for presentation {presentation_index}."
            )

        dual_reduction = start_length - dual_best
        triple_reduction = start_length - triple_best
        if dual_best < triple_best:
            paired_outcome = "Auto Dual better"
        elif triple_best < dual_best:
            paired_outcome = "Triple better"
        else:
            paired_outcome = "Tie"

        if dual_reduction > 0 and triple_reduction > 0:
            coverage_group = "Both reduced"
        elif dual_reduction > 0:
            coverage_group = "Auto Dual only"
        elif triple_reduction > 0:
            coverage_group = "Triple only"
        else:
            coverage_group = "Neither reduced"

        paired.append(
            {
                "presentation_index": presentation_index,
                "presentation_label": dual["label"],
                "r1": dual["r1"],
                "r2": dual["r2"],
                "start_total_length": start_length,
                "auto_dual_best_total_length": dual_best,
                "triple_best_total_length": triple_best,
                "auto_dual_best_length_source": dual_best_source,
                "triple_best_length_source": triple_best_source,
                "auto_dual_length_reduction": dual_reduction,
                "triple_length_reduction": triple_reduction,
                "auto_dual_reduced": int(dual_reduction > 0),
                "triple_reduced": int(triple_reduction > 0),
                "paired_outcome": paired_outcome,
                "reduction_coverage_group": coverage_group,
                "auto_dual_solved": int(_as_bool(dual["solved"])),
                "triple_solved": int(_as_bool(triple["solved"])),
                "auto_dual_termination_reason": dual["termination_reason"],
                "triple_termination_reason": triple["termination_reason"],
                "auto_dual_total_nodes": dual_nodes,
                "triple_total_nodes": triple_nodes,
                "triple_substitution_pop_count": triple_sub_pops,
                "triple_automorphism_pop_count": triple_auto_pops,
                "triple_non_auto_cov_pop_count": triple_non_auto_pops,
                "triple_non_auto_cov_pop_source": non_auto_source,
                "triple_substitution_pop_fraction": triple_sub_pops / triple_nodes,
                "triple_automorphism_pop_fraction": triple_auto_pops / triple_nodes,
                "triple_non_auto_cov_pop_fraction": triple_non_auto_pops / triple_nodes,
                "auto_dual_frontier_prune_count": _as_int(dual, "frontier_prune_count"),
                "triple_frontier_prune_count": _as_int(triple, "frontier_prune_count"),
                "auto_dual_frontier_states_pruned": _as_int(
                    dual, "frontier_states_pruned"
                ),
                "triple_frontier_states_pruned": _as_int(
                    triple, "frontier_states_pruned"
                ),
            }
        )
    return paired


def build_algorithm_summary(paired: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    summary = []
    for method in METHOD_ORDER:
        prefix = "auto_dual" if method == AUTO_DUAL else "triple"
        best = [int(row[f"{prefix}_best_total_length"]) for row in paired]
        reduction = [int(row[f"{prefix}_length_reduction"]) for row in paired]
        solved = [int(row[f"{prefix}_solved"]) for row in paired]
        nodes = [int(row[f"{prefix}_total_nodes"]) for row in paired]
        summary.append(
            {
                "algorithm": method,
                "algorithm_label": METHOD_LABELS[method],
                "presentations": len(paired),
                "node_budget_per_presentation": nodes[0],
                "solved_count": sum(solved),
                "reduced_count": sum(value > 0 for value in reduction),
                "unchanged_count": sum(value == 0 for value in reduction),
                "reduction_rate": sum(value > 0 for value in reduction) / len(paired),
                "mean_best_total_length": statistics.fmean(best),
                "median_best_total_length": statistics.median(best),
                "mean_length_reduction": statistics.fmean(reduction),
                "median_length_reduction": statistics.median(reduction),
                "maximum_length_reduction": max(reduction),
            }
        )
    return summary


def build_head_to_head_summary(paired: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    counts = Counter(str(row["paired_outcome"]) for row in paired)
    order = ("Auto Dual better", "Tie", "Triple better")
    return [
        {
            "outcome": outcome,
            "presentations": counts[outcome],
            "fraction": counts[outcome] / len(paired),
        }
        for outcome in order
    ]


def build_coverage_summary(paired: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    counts = Counter(str(row["reduction_coverage_group"]) for row in paired)
    order = ("Auto Dual only", "Both reduced", "Triple only", "Neither reduced")
    return [
        {
            "coverage_group": group,
            "presentations": counts[group],
            "fraction": counts[group] / len(paired),
        }
        for group in order
    ]


def build_triple_queue_summary(paired: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    fields = (
        ("Substitution", "triple_substitution_pop_count"),
        ("Automorphism", "triple_automorphism_pop_count"),
        ("Non-Auto COV", "triple_non_auto_cov_pop_count"),
    )
    total = sum(int(row["triple_total_nodes"]) for row in paired)
    return [
        {
            "move_family": label,
            "queue_pops": sum(int(row[field]) for row in paired),
            "queue_pop_fraction": sum(int(row[field]) for row in paired) / total,
        }
        for label, field in fields
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table to {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figure_style(mode: str) -> Dict[str, Any]:
    if mode == "slides":
        return {
            "figsize": (12.8, 7.2),
            "title_size": 25,
            "label_size": 18,
            "tick_size": 15,
            "legend_size": 15,
            "line_width": 2.6,
            "marker_scale": 1.35,
        }
    return {
        "figsize": (7.2, 4.6),
        "title_size": 15,
        "label_size": 11,
        "tick_size": 9,
        "legend_size": 9,
        "line_width": 1.7,
        "marker_scale": 0.8,
    }


def _prepare_axis(ax: Any, style: Mapping[str, Any]) -> None:
    ax.tick_params(labelsize=style["tick_size"], colors="#263238")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7B858B")
    ax.spines["bottom"].set_color("#7B858B")
    ax.grid(axis="y", color="#D9DEDF", linewidth=0.8, alpha=0.7, zorder=0)


def _save_figure(fig: Any, base_path: Path, mode: str) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    bbox = None if mode == "slides" else "tight"
    fig.savefig(png, dpi=220 if mode == "slides" else 320, bbox_inches=bbox)
    fig.savefig(pdf, bbox_inches=bbox)
    return [png, pdf]


def _plot_paired_best_length(
    paired: Sequence[Mapping[str, Any]], output_dir: Path, mode: str
) -> list[Path]:
    import matplotlib.pyplot as plt
    style = _figure_style(mode)
    fig, ax = plt.subplots(figsize=style["figsize"], facecolor="white")
    points = Counter(
        (
            int(row["auto_dual_best_total_length"]),
            int(row["triple_best_total_length"]),
        )
        for row in paired
    )
    for (dual_length, triple_length), count in sorted(points.items()):
        if triple_length < dual_length:
            color = COLORS[TRIPLE_GS]
        elif dual_length < triple_length:
            color = COLORS[AUTO_DUAL]
        else:
            color = COLORS["tie"]
        ax.scatter(
            dual_length,
            triple_length,
            s=(42 + 22 * count) * style["marker_scale"],
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.9,
            zorder=3,
        )

    lengths = [
        int(row[key])
        for row in paired
        for key in ("auto_dual_best_total_length", "triple_best_total_length")
    ]
    low, high = min(lengths) - 0.7, max(lengths) + 0.7
    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        color="#56646B",
        linewidth=style["line_width"],
        zorder=1,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        "Auto Dual GS best total length", fontsize=style["label_size"], color="#263238"
    )
    ax.set_ylabel(
        "Triple GS best total length", fontsize=style["label_size"], color="#263238"
    )
    title = (
        "Triple GS matched or improved every Auto Dual result"
        if mode == "slides"
        else "Paired best total length after one million nodes"
    )
    ax.set_title(title, fontsize=style["title_size"], weight="bold", color="#17333B", pad=14)
    outcomes = Counter(str(row["paired_outcome"]) for row in paired)
    ax.text(
        0.02,
        0.98,
        f"Triple better: {outcomes['Triple better']}\nTie: {outcomes['Tie']}\nAuto Dual better: {outcomes['Auto Dual better']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style["legend_size"],
        color="#263238",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D0D7D8"},
    )
    ax.text(
        0.98,
        0.03,
        "Orange: Triple GS shorter\nGray: tie\nLower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=style["legend_size"],
        color="#56646B",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#D0D7D8"},
    )
    _prepare_axis(ax, style)
    ax.grid(False)
    fig.tight_layout()
    paths = _save_figure(fig, output_dir / "paired_best_total_length", mode)
    plt.close(fig)
    return paths


def _plot_reduction_coverage(
    paired: Sequence[Mapping[str, Any]], output_dir: Path, mode: str
) -> list[Path]:
    import matplotlib.pyplot as plt

    style = _figure_style(mode)
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    values = [
        sum(int(row["auto_dual_reduced"]) for row in paired),
        sum(int(row["triple_reduced"]) for row in paired),
    ]
    triple_only = sum(
        str(row["reduction_coverage_group"]) == "Triple only" for row in paired
    )
    auto_dual_only = sum(
        str(row["reduction_coverage_group"]) == "Auto Dual only" for row in paired
    )
    fig, ax = plt.subplots(figsize=style["figsize"], facecolor="white")
    bars = ax.barh(
        labels,
        values,
        color=[COLORS[AUTO_DUAL], COLORS[TRIPLE_GS]],
        height=0.52,
        zorder=2,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, len(paired))
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value} of {len(paired)} ({value / len(paired):.1%})",
            va="center",
            fontsize=style["label_size"],
            color="#263238",
            weight="bold",
        )
    presentation_label = "presentation" if triple_only == 1 else "presentations"
    title = (
        f"Triple GS reduced {triple_only} additional hard {presentation_label}"
        if mode == "slides"
        else "Presentations reduced from their starting length"
    )
    ax.set_title(title, fontsize=style["title_size"], weight="bold", color="#17333B", pad=14)
    ax.set_xlabel("Number of presentations", fontsize=style["label_size"], color="#263238")
    if auto_dual_only == 0:
        caption = (
            f"All {values[0]} Auto Dual reductions were also achieved by Triple GS."
        )
    else:
        suffix = "reduction was" if auto_dual_only == 1 else "reductions were"
        caption = (
            f"{auto_dual_only} Auto Dual {suffix} not matched by Triple GS."
        )
    ax.text(
        0.99,
        -0.18,
        caption,
        transform=ax.transAxes,
        ha="right",
        fontsize=style["legend_size"],
        color="#56646B",
    )
    _prepare_axis(ax, style)
    fig.tight_layout()
    paths = _save_figure(fig, output_dir / "reduction_coverage", mode)
    plt.close(fig)
    return paths


def _plot_reduction_magnitude(
    paired: Sequence[Mapping[str, Any]], output_dir: Path, mode: str
) -> list[Path]:
    import matplotlib.pyplot as plt

    style = _figure_style(mode)
    dual_counts = Counter(int(row["auto_dual_length_reduction"]) for row in paired)
    triple_counts = Counter(int(row["triple_length_reduction"]) for row in paired)
    max_reduction = max((*dual_counts.keys(), *triple_counts.keys()), default=0)
    # Keep a meaningful x-axis even when no presentation was reduced.
    magnitudes = np.arange(1, max(1, max_reduction) + 1)
    width = 0.34
    fig, ax = plt.subplots(figsize=style["figsize"], facecolor="white")
    dual_bars = ax.bar(
        magnitudes - width / 2,
        [dual_counts[int(value)] for value in magnitudes],
        width,
        color=COLORS[AUTO_DUAL],
        label=METHOD_LABELS[AUTO_DUAL],
        zorder=2,
    )
    triple_bars = ax.bar(
        magnitudes + width / 2,
        [triple_counts[int(value)] for value in magnitudes],
        width,
        color=COLORS[TRIPLE_GS],
        label=METHOD_LABELS[TRIPLE_GS],
        zorder=2,
    )
    for bars in (dual_bars, triple_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=style["legend_size"],
                color="#263238",
            )
    ax.set_xticks(magnitudes)
    ax.set_xlabel("Symbols removed from total length", fontsize=style["label_size"], color="#263238")
    ax.set_ylabel("Presentations", fontsize=style["label_size"], color="#263238")
    ax.set_title(
        "Magnitude of the achieved length reductions",
        fontsize=style["title_size"],
        weight="bold",
        color="#17333B",
        pad=14,
    )
    ax.legend(frameon=False, fontsize=style["legend_size"], ncol=2, loc="upper right")
    maximum_count = max(
        max(dual_counts[value] for value in magnitudes),
        max(triple_counts[value] for value in magnitudes),
    )
    ax.set_ylim(0, maximum_count + 3)
    ax.text(
        0.01,
        0.98,
        f"Unchanged: Auto Dual {dual_counts[0]}, Triple {triple_counts[0]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style["legend_size"],
        color="#56646B",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    _prepare_axis(ax, style)
    fig.tight_layout()
    paths = _save_figure(fig, output_dir / "reduction_magnitude", mode)
    plt.close(fig)
    return paths


def _plot_reduced_cases_matrix(
    paired: Sequence[Mapping[str, Any]], output_dir: Path, mode: str
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    reduced = [
        row
        for row in paired
        if int(row["auto_dual_length_reduction"]) > 0
        or int(row["triple_length_reduction"]) > 0
    ]
    reduced.sort(
        key=lambda row: (
            -int(row["triple_length_reduction"]),
            -int(row["auto_dual_length_reduction"]),
            int(row["presentation_index"]),
        )
    )
    matrix = np.array(
        [
            [int(row["auto_dual_length_reduction"]) for row in reduced],
            [int(row["triple_length_reduction"]) for row in reduced],
        ]
    )
    style = _figure_style(mode)
    figsize = style["figsize"] if mode == "slides" else (10.4, 2.8)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    max_reduction = max(int(matrix.max()), 1)
    cmap = LinearSegmentedColormap.from_list(
        "reduction",
        ["#F2F1EC", "#F4C77C", "#E98B5F", "#C94C4C"],
        N=max_reduction + 1,
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max_reduction, aspect="auto")
    ax.set_yticks([0, 1], [METHOD_LABELS[AUTO_DUAL], METHOD_LABELS[TRIPLE_GS]])
    ax.set_xticks(
        np.arange(len(reduced)),
        [str(row["presentation_label"]) for row in reduced],
        rotation=55,
        ha="right",
    )
    ax.tick_params(axis="x", labelsize=11 if mode == "slides" else 7)
    ax.tick_params(axis="y", labelsize=style["label_size"])
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=12 if mode == "slides" else 8,
                color="white" if value >= 2 else "#263238",
                weight="bold",
            )
    ax.set_title(
        "Length reduction on every presentation improved by either method",
        fontsize=style["title_size"],
        weight="bold",
        color="#17333B",
        pad=14,
    )
    ax.set_xlabel("Presentation label", fontsize=style["label_size"], color="#263238")
    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
        ticks=range(max_reduction + 1),
    )
    colorbar.set_label("Symbols removed", fontsize=style["label_size"])
    colorbar.ax.tick_params(labelsize=style["tick_size"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    paths = _save_figure(fig, output_dir / "reduced_cases_matrix", mode)
    plt.close(fig)
    return paths


def _plot_triple_queue_mix(
    queue_summary: Sequence[Mapping[str, Any]], output_dir: Path, mode: str
) -> list[Path]:
    import matplotlib.pyplot as plt

    style = _figure_style(mode)
    labels = [str(row["move_family"]) for row in queue_summary]
    fractions = [float(row["queue_pop_fraction"]) for row in queue_summary]
    colors = [COLORS["substitution"], COLORS["automorphism"], COLORS["non_auto"]]
    fig, ax = plt.subplots(figsize=style["figsize"], facecolor="white")
    left = 0.0
    for label, fraction, color in zip(labels, fractions, colors):
        ax.barh([0], [fraction], left=left, height=0.42, color=color, label=label)
        if fraction >= 0.04:
            ax.text(
                left + fraction / 2,
                0,
                f"{fraction:.1%}",
                ha="center",
                va="center",
                fontsize=style["label_size"],
                color="white",
                weight="bold",
            )
        left += fraction
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Share of Triple GS queue expansions", fontsize=style["label_size"], color="#263238")
    ax.set_title(
        "How Triple GS used its one-million-node search budget",
        fontsize=style["title_size"],
        weight="bold",
        color="#17333B",
        pad=14,
    )
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        fontsize=style["legend_size"],
    )
    ax.annotate(
        f"Non-Auto COV: {fractions[2]:.1%}",
        xy=(1 - fractions[2] / 2, 0),
        xytext=(0.80, -0.32),
        arrowprops={"arrowstyle": "->", "color": "#8A6A24", "lw": style["line_width"]},
        ha="center",
        va="center",
        fontsize=style["legend_size"],
        color="#6F551B",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7B858B")
    ax.tick_params(axis="x", labelsize=style["tick_size"])
    fig.tight_layout()
    paths = _save_figure(fig, output_dir / "triple_queue_mix", mode)
    plt.close(fig)
    return paths


def build_artifacts(input_csv: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Build processed tables and all paper/slide figures."""

    output_dir = Path(output_dir).expanduser().resolve()
    analysis_root = output_dir.parent
    input_path = Path(input_csv).expanduser().resolve()

    def portable_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(analysis_root).as_posix()
        except ValueError:
            return path.name

    paired = load_paired_results(input_path)
    algorithm_summary = build_algorithm_summary(paired)
    head_to_head = build_head_to_head_summary(paired)
    coverage = build_coverage_summary(paired)
    queue_summary = build_triple_queue_summary(paired)

    paired_csv = output_dir / "processed_paired_results.csv"
    algorithm_csv = output_dir / "algorithm_summary.csv"
    head_to_head_csv = output_dir / "head_to_head_summary.csv"
    coverage_csv = output_dir / "reduction_coverage_summary.csv"
    queue_csv = output_dir / "triple_queue_summary.csv"
    _write_csv(paired_csv, paired)
    _write_csv(algorithm_csv, algorithm_summary)
    _write_csv(head_to_head_csv, head_to_head)
    _write_csv(coverage_csv, coverage)
    _write_csv(queue_csv, queue_summary)

    figures: list[Path] = []
    for mode in ("paper", "slides"):
        figure_dir = output_dir / "figures" / mode
        figures.extend(_plot_paired_best_length(paired, figure_dir, mode))
        figures.extend(_plot_reduction_coverage(paired, figure_dir, mode))
        figures.extend(_plot_reduction_magnitude(paired, figure_dir, mode))
        figures.extend(_plot_reduced_cases_matrix(paired, figure_dir, mode))
        figures.extend(_plot_triple_queue_mix(queue_summary, figure_dir, mode))

    manifest = {
        "input_csv": portable_path(input_path),
        "presentation_count": len(paired),
        "method_order": [METHOD_LABELS[method] for method in METHOD_ORDER],
        "tables": [
            portable_path(paired_csv),
            portable_path(algorithm_csv),
            portable_path(head_to_head_csv),
            portable_path(coverage_csv),
            portable_path(queue_csv),
        ],
        "figures": [portable_path(path) for path in figures],
        "key_findings": {
            "auto_dual_reduced_count": sum(int(row["auto_dual_reduced"]) for row in paired),
            "triple_reduced_count": sum(int(row["triple_reduced"]) for row in paired),
            "auto_dual_better_count": sum(row["paired_outcome"] == "Auto Dual better" for row in paired),
            "tie_count": sum(row["paired_outcome"] == "Tie" for row in paired),
            "triple_better_count": sum(row["paired_outcome"] == "Triple better" for row in paired),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    manifest["algorithm_summary"] = algorithm_summary
    manifest["head_to_head_summary"] = head_to_head
    manifest["coverage_summary"] = coverage
    manifest["triple_queue_summary"] = queue_summary
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "wandb_unsolved124_dual_triple.csv",
    )
    parser.add_argument("--output", type=Path, default=root / "outputs")
    args = parser.parse_args()
    artifacts = build_artifacts(args.input, args.output)
    print(json.dumps(artifacts["key_findings"], indent=2))
    print(f"Tables and figures saved under: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
