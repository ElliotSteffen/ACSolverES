"""Publication-oriented processing and plots for the 640-case comparison."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
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
    str(Path(tempfile.gettempdir()) / "ac_final_paper_matplotlib"),
)

from .algorithm_adapters import ALGORITHM_LABELS, ALGORITHM_ORDER


SUB_ONLY = "sub_only"
METHOD_ORDER = (SUB_ONLY,) + ALGORITHM_ORDER
METHOD_LABELS = {
    SUB_ONLY: "Sub-only GS",
    **ALGORITHM_LABELS,
}
METHOD_SHORT_LABELS = {
    SUB_ONLY: "Sub-only",
    "single_auto_cov_dual": "Single-Auto-COV Dual",
    "auto_dual": "Auto Dual",
    "auto_nonauto_cov_triple": "Auto + Non-Auto Triple",
}
COLORS = {
    SUB_ONLY: "#243447",
    "single_auto_cov_dual": "#56B4E9",
    "auto_dual": "#0072B2",
    "auto_nonauto_cov_triple": "#D55E00",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _latest_by_id(rows: Iterable[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    latest = {}
    for row in rows:
        latest[str(row["pres_id"])] = dict(row)
    return latest


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_matched_results(
    baseline_csv: Path | str,
    run_dir: Path | str,
    *,
    require_all_solved: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    baseline_path = Path(baseline_csv).resolve()
    run_path = Path(run_dir).resolve()
    baseline_rows = _read_csv(baseline_path)
    baseline = {}
    for row in baseline_rows:
        pres_id = str(row["pres_id"])
        baseline[pres_id] = {
            "algorithm": SUB_ONLY,
            "algorithm_label": METHOD_LABELS[SUB_ONLY],
            "pres_id": pres_id,
            "r1": row["r1"],
            "r2": row["r2"],
            "status": "ok",
            "solved": True,
            "nodes_explored": int(row["nodes_explored"]),
            "path_length": int(row["path_length"]),
            "sub_moves_in_path": int(row["path_length"]),
            "single_auto_cov_moves_in_path": 0,
            "automorphism_moves_in_path": 0,
            "non_auto_cov_moves_in_path": 0,
        }
    if len(baseline) != len(baseline_rows):
        raise ValueError("Baseline pres_id values must be unique.")

    matched: Dict[str, Dict[str, Dict[str, Any]]] = {SUB_ONLY: baseline}
    baseline_ids = set(baseline)
    for algorithm in ALGORITHM_ORDER:
        path = run_path / f"{algorithm}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        latest = _latest_by_id(_read_csv(path))
        failed = [
            pres_id
            for pres_id, row in latest.items()
            if row.get("status") != "ok"
        ]
        if failed:
            raise ValueError(
                f"{algorithm} has failed rows for pres_id values {failed[:10]}."
            )
        if set(latest) != baseline_ids:
            missing = sorted(baseline_ids.difference(latest), key=int)
            extra = sorted(set(latest).difference(baseline_ids), key=int)
            raise ValueError(
                f"{algorithm} is not matched to the baseline; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
        normalized = {}
        unsolved = []
        for pres_id, row in latest.items():
            solved = _parse_bool(row["solved"])
            if not solved:
                unsolved.append(pres_id)
            normalized[pres_id] = {
                **row,
                "solved": solved,
                "nodes_explored": int(row["nodes_explored"]),
                "path_length": int(row["path_length"]),
                "sub_moves_in_path": int(row.get("sub_moves_in_path") or 0),
                "single_auto_cov_moves_in_path": int(
                    row.get("single_auto_cov_moves_in_path") or 0
                ),
                "automorphism_moves_in_path": int(
                    row.get("automorphism_moves_in_path") or 0
                ),
                "non_auto_cov_moves_in_path": int(
                    row.get("non_auto_cov_moves_in_path") or 0
                ),
            }
        if require_all_solved and unsolved:
            raise ValueError(
                f"{algorithm} has {len(unsolved)} unsolved rows: {unsolved[:10]}"
            )
        matched[algorithm] = normalized
    return matched


def _geometric_mean(values: np.ndarray) -> float:
    if np.any(values <= 0):
        raise ValueError("Geometric means require positive node counts.")
    return float(np.exp(np.mean(np.log(values))))


def _ordered_ids(matched: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    return sorted(matched[SUB_ONLY], key=lambda value: int(value))


def _values(
    matched: Mapping[str, Mapping[str, Mapping[str, Any]]],
    algorithm: str,
    field: str,
    ids: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [matched[algorithm][pres_id][field] for pres_id in ids],
        dtype=float,
    )


def summary_rows(
    matched: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[Dict[str, Any]]:
    ids = _ordered_ids(matched)
    baseline_nodes = _values(matched, SUB_ONLY, "nodes_explored", ids)
    rows = []
    for algorithm in METHOD_ORDER:
        nodes = _values(matched, algorithm, "nodes_explored", ids)
        paths = _values(matched, algorithm, "path_length", ids)
        difference = nodes - baseline_nodes
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_label": METHOD_LABELS[algorithm],
                "presentations": len(ids),
                "mean_nodes": float(np.mean(nodes)),
                "median_nodes": float(np.median(nodes)),
                "geometric_mean_nodes": _geometric_mean(nodes),
                "total_nodes": int(np.sum(nodes)),
                "mean_path_length": float(np.mean(paths)),
                "median_path_length": float(np.median(paths)),
                "fewer_nodes_vs_sub_only": int(np.sum(difference < 0)),
                "equal_nodes_vs_sub_only": int(np.sum(difference == 0)),
                "more_nodes_vs_sub_only": int(np.sum(difference > 0)),
                "median_node_ratio_vs_sub_only": float(
                    np.median(nodes / baseline_nodes)
                ),
            }
        )
    return rows


def _combined_rows(
    matched: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[Dict[str, Any]]:
    ids = _ordered_ids(matched)
    rows = []
    for algorithm in METHOD_ORDER:
        for pres_id in ids:
            source = matched[algorithm][pres_id]
            rows.append(
                {
                    "algorithm": algorithm,
                    "algorithm_label": METHOD_LABELS[algorithm],
                    "pres_id": pres_id,
                    "r1": source["r1"],
                    "r2": source["r2"],
                    "nodes_explored": source["nodes_explored"],
                    "path_length": source["path_length"],
                    "sub_moves_in_path": source["sub_moves_in_path"],
                    "single_auto_cov_moves_in_path": source[
                        "single_auto_cov_moves_in_path"
                    ],
                    "automorphism_moves_in_path": source[
                        "automorphism_moves_in_path"
                    ],
                    "non_auto_cov_moves_in_path": source[
                        "non_auto_cov_moves_in_path"
                    ],
                }
            )
    return rows


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def _configure_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plots. Run the notebook setup cell "
            "or install it with: python -m pip install matplotlib"
        ) from exc
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def build_paper_artifacts(
    *,
    baseline_csv: Path | str,
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    require_all_solved: bool = True,
) -> Dict[str, Any]:
    """Create matched tables and all primary plots used in the paper."""

    plt = _configure_matplotlib()
    run_path = Path(run_dir).resolve()
    artifact_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else run_path / "paper_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    matched = load_matched_results(
        baseline_csv,
        run_path,
        require_all_solved=require_all_solved,
    )
    ids = _ordered_ids(matched)

    summary = summary_rows(matched)
    summary_path = artifact_dir / "algorithm_summary.csv"
    combined_path = artifact_dir / "matched_results_long.csv"
    _write_csv(summary_path, summary)
    _write_csv(combined_path, _combined_rows(matched))
    figure_paths: list[Path] = []

    # 1. Empirical cumulative distribution of nodes.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for algorithm in METHOD_ORDER:
        nodes = np.sort(_values(matched, algorithm, "nodes_explored", ids))
        cumulative = np.arange(1, len(nodes) + 1) / len(nodes)
        ax.step(
            nodes,
            cumulative,
            where="post",
            label=METHOD_SHORT_LABELS[algorithm],
            color=COLORS[algorithm],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Nodes explored (log scale)")
    ax.set_ylabel("Fraction of presentations solved")
    ax.set_title("Search-effort distribution across 640 solved presentations")
    ax.grid(axis="both", alpha=0.2)
    ax.legend(frameon=False)
    figure_paths.extend(_save_figure(fig, artifact_dir, "nodes_ecdf"))
    plt.close(fig)

    # 2. Paired node comparisons against Sub-only GS.
    baseline_nodes = _values(matched, SUB_ONLY, "nodes_explored", ids)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    positive_min = float(np.min(baseline_nodes))
    positive_max = float(np.max(baseline_nodes))
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        nodes = _values(matched, algorithm, "nodes_explored", ids)
        positive_min = min(positive_min, float(np.min(nodes)))
        positive_max = max(positive_max, float(np.max(nodes)))
        ax.scatter(
            baseline_nodes,
            nodes,
            s=13,
            alpha=0.58,
            color=COLORS[algorithm],
            edgecolors="none",
        )
        ax.set_title(METHOD_SHORT_LABELS[algorithm])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
    diagonal = [positive_min, positive_max]
    for ax in axes:
        ax.plot(diagonal, diagonal, linestyle="--", color="#555555", linewidth=1)
        ax.set_xlabel("Sub-only nodes")
    axes[0].set_ylabel("Comparison algorithm nodes")
    fig.suptitle("Paired node counts; points below the diagonal use fewer nodes")
    fig.tight_layout()
    figure_paths.extend(_save_figure(fig, artifact_dir, "nodes_paired_vs_sub_only"))
    plt.close(fig)

    # 3. Per-presentation node ratios ordered by baseline difficulty.
    order = np.argsort(baseline_nodes)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for algorithm in ALGORITHM_ORDER:
        nodes = _values(matched, algorithm, "nodes_explored", ids)
        ratio = (nodes / baseline_nodes)[order]
        ax.plot(
            np.arange(len(ids)),
            ratio,
            label=METHOD_SHORT_LABELS[algorithm],
            color=COLORS[algorithm],
            linewidth=1.35,
            alpha=0.9,
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Presentations ordered by Sub-only node count")
    ax.set_ylabel("Node ratio relative to Sub-only GS")
    ax.set_title("Instance-level search-effort ratio")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    figure_paths.extend(_save_figure(fig, artifact_dir, "node_ratio_by_instance"))
    plt.close(fig)

    # 4. Paired path-length comparisons.
    baseline_paths = _values(matched, SUB_ONLY, "path_length", ids)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    path_max = float(np.max(baseline_paths))
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        paths = _values(matched, algorithm, "path_length", ids)
        path_max = max(path_max, float(np.max(paths)))
        ax.scatter(
            baseline_paths,
            paths,
            s=13,
            alpha=0.58,
            color=COLORS[algorithm],
            edgecolors="none",
        )
        ax.set_title(METHOD_SHORT_LABELS[algorithm])
        ax.set_xlabel("Sub-only path length")
        ax.grid(alpha=0.18)
    for ax in axes:
        ax.plot([0, path_max], [0, path_max], linestyle="--", color="#555555")
    axes[0].set_ylabel("Comparison path length")
    fig.suptitle("Paired solution-path lengths")
    fig.tight_layout()
    figure_paths.extend(_save_figure(fig, artifact_dir, "path_length_paired"))
    plt.close(fig)

    # 5. Win/tie/loss counts versus Sub-only.
    comparison_rows = {row["algorithm"]: row for row in summary}
    algorithms = list(ALGORITHM_ORDER)
    fewer = [comparison_rows[name]["fewer_nodes_vs_sub_only"] for name in algorithms]
    equal = [comparison_rows[name]["equal_nodes_vs_sub_only"] for name in algorithms]
    more = [comparison_rows[name]["more_nodes_vs_sub_only"] for name in algorithms]
    y = np.arange(len(algorithms))
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.barh(y, fewer, color="#009E73", label="Fewer nodes")
    ax.barh(y, equal, left=fewer, color="#B7B7B7", label="Equal")
    ax.barh(
        y,
        more,
        left=np.asarray(fewer) + np.asarray(equal),
        color="#D55E00",
        label="More nodes",
    )
    ax.set_yticks(y, [METHOD_SHORT_LABELS[name] for name in algorithms])
    ax.set_xlabel("Number of presentations")
    ax.set_title("Node-count outcomes relative to Sub-only GS")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    figure_paths.extend(_save_figure(fig, artifact_dir, "nodes_win_tie_loss"))
    plt.close(fig)

    # 6. Mean move composition of recovered solution paths.
    move_fields = (
        "sub_moves_in_path",
        "single_auto_cov_moves_in_path",
        "automorphism_moves_in_path",
        "non_auto_cov_moves_in_path",
    )
    move_labels = ("Substitution", "Single-Auto-COV", "Automorphism", "Non-Auto COV")
    move_colors = ("#808080", "#56B4E9", "#0072B2", "#D55E00")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bottoms = np.zeros(len(METHOD_ORDER))
    x = np.arange(len(METHOD_ORDER))
    for field, label, color in zip(move_fields, move_labels, move_colors):
        means = [
            float(np.mean(_values(matched, algorithm, field, ids)))
            for algorithm in METHOD_ORDER
        ]
        ax.bar(x, means, bottom=bottoms, label=label, color=color)
        bottoms += np.asarray(means)
    ax.set_xticks(x, [METHOD_SHORT_LABELS[name] for name in METHOD_ORDER], rotation=15)
    ax.set_ylabel("Mean moves per solution")
    ax.set_title("Solution-path move composition")
    ax.legend(frameon=False, ncol=2)
    figure_paths.extend(_save_figure(fig, artifact_dir, "solution_move_composition"))
    plt.close(fig)

    return {
        "output_dir": artifact_dir,
        "summary_csv": summary_path,
        "combined_csv": combined_path,
        "figures": figure_paths,
        "summary": summary,
    }


__all__ = [
    "METHOD_LABELS",
    "METHOD_ORDER",
    "SUB_ONLY",
    "build_paper_artifacts",
    "load_matched_results",
    "summary_rows",
]
