"""Resumable local benchmark runner for the final-paper algorithm panel."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

from .algorithm_adapters import (
    ALGORITHM_LABELS,
    ALGORITHM_ORDER,
    run_algorithm,
    warm_up,
)


RESULT_FIELDS = (
    "algorithm",
    "algorithm_label",
    "pres_id",
    "r1",
    "r2",
    "status",
    "solved",
    "nodes_explored",
    "path_length",
    "termination_reason",
    "final_total_length",
    "minimum_total_length",
    "sub_queue_pops",
    "single_cov_queue_pops",
    "automorphism_queue_pops",
    "non_auto_cov_queue_pops",
    "sub_moves_in_path",
    "single_auto_cov_moves_in_path",
    "automorphism_moves_in_path",
    "non_auto_cov_moves_in_path",
    "path_json",
    "moves_json",
    "error",
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_presentations(
    csv_path: Path | str,
    *,
    limit: Optional[int] = None,
    pres_ids: Optional[Iterable[str | int]] = None,
) -> list[Dict[str, str]]:
    path = Path(csv_path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pres_id", "r1", "r2"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    selected_ids = None if pres_ids is None else {str(value) for value in pres_ids}
    selected = [
        {"pres_id": str(row["pres_id"]), "r1": row["r1"], "r2": row["r2"]}
        for row in rows
        if selected_ids is None or str(row["pres_id"]) in selected_ids
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive or None.")
        selected = selected[:limit]
    ids = [row["pres_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Presentation IDs must be unique.")
    if selected_ids is not None and set(ids) != selected_ids:
        missing_ids = sorted(selected_ids.difference(ids))
        raise ValueError(f"Requested presentation IDs not found: {missing_ids}")
    return selected


def _move_counts(moves: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {
        "sub_moves_in_path": 0,
        "single_auto_cov_moves_in_path": 0,
        "automorphism_moves_in_path": 0,
        "non_auto_cov_moves_in_path": 0,
    }
    for move in moves:
        move_type = move.get("move_type")
        if move_type == "standard_substitution":
            counts["sub_moves_in_path"] += 1
        elif move_type == "auto_cov":
            counts["automorphism_moves_in_path"] += 1
        elif move_type == "complete_cov" and move.get("cov_variant") == "single_cov":
            counts["single_auto_cov_moves_in_path"] += 1
        elif move_type == "complete_cov":
            counts["non_auto_cov_moves_in_path"] += 1
    return counts


def _result_row(
    algorithm: str,
    presentation: Mapping[str, str],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    moves = result.get("moves") or []
    move_counts = _move_counts(moves)
    return {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "pres_id": presentation["pres_id"],
        "r1": presentation["r1"],
        "r2": presentation["r2"],
        "status": "ok",
        "solved": bool(result.get("solved")),
        "nodes_explored": int(result["total_nodes_used"]),
        "path_length": int(result.get("path_length", 0)),
        "termination_reason": result.get("termination_reason", ""),
        "final_total_length": result.get("final_total_length", ""),
        "minimum_total_length": result.get("minimum_total_length", ""),
        "sub_queue_pops": result.get("sub_pop_count", 0),
        "single_cov_queue_pops": (
            result.get("cov_pop_count", 0) if algorithm == "single_auto_cov_dual" else 0
        ),
        "automorphism_queue_pops": result.get(
            "auto_cov_queue_pop_count",
            result.get("automorphism_queue_pop_count", 0),
        ),
        "non_auto_cov_queue_pops": result.get("complete_cov_queue_pop_count", 0),
        **move_counts,
        "path_json": json.dumps(_json_safe(result.get("path") or [])),
        "moves_json": json.dumps(_json_safe(moves)),
        "error": "",
    }


def _error_row(
    algorithm: str,
    presentation: Mapping[str, str],
    error: BaseException,
) -> Dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS[algorithm],
            "pres_id": presentation["pres_id"],
            "r1": presentation["r1"],
            "r2": presentation["r2"],
            "status": "error",
            "solved": False,
            "error": "".join(traceback.format_exception_only(type(error), error)).strip(),
        }
    )
    return row


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["pres_id"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        handle.flush()


def _experiment_manifest(
    input_csv: Path,
    presentations: Sequence[Mapping[str, str]],
    algorithm_parameters: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "input_csv": input_csv.name,
        "input_sha256": _sha256(input_csv),
        "presentation_ids": [row["pres_id"] for row in presentations],
        "algorithms": list(algorithm_parameters),
        "algorithm_parameters": _json_safe(algorithm_parameters),
        "metrics": ["nodes_explored", "path_length", "move_composition"],
        "runtime_recorded": False,
    }


def _prepare_run_directory(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "config.json"
    normalized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        existing = manifest_path.read_text(encoding="utf-8")
        if existing != normalized:
            raise ValueError(
                f"{run_dir} already contains a different configuration. "
                "Choose a new run_name or restore the original hyperparameters."
            )
    else:
        manifest_path.write_text(normalized, encoding="utf-8")


def run_experiment_suite(
    *,
    input_csv: Path | str,
    output_root: Path | str,
    run_name: str,
    algorithm_parameters: Mapping[str, Mapping[str, Any]],
    limit: Optional[int] = None,
    pres_ids: Optional[Iterable[str | int]] = None,
    resume: bool = True,
    fail_fast: bool = True,
    require_all_solved: bool = True,
    print_progress: bool = True,
) -> Path:
    """Run all configured algorithms and checkpoint every presentation locally."""

    unknown = set(algorithm_parameters).difference(ALGORITHM_ORDER)
    if unknown:
        raise ValueError(f"Unknown algorithms in configuration: {sorted(unknown)}")
    if not algorithm_parameters:
        raise ValueError("At least one algorithm must be configured.")

    input_path = Path(input_csv).resolve()
    presentations = load_presentations(input_path, limit=limit, pres_ids=pres_ids)
    run_dir = Path(output_root).resolve() / run_name
    manifest = _experiment_manifest(input_path, presentations, algorithm_parameters)
    _prepare_run_directory(run_dir, manifest)

    max_len = max(
        int(parameters.get("max_len", 50))
        for parameters in algorithm_parameters.values()
    )
    warm_up(max_len=max_len)

    unsolved = []
    for algorithm in ALGORITHM_ORDER:
        if algorithm not in algorithm_parameters:
            continue
        output_csv = run_dir / f"{algorithm}.csv"
        completed = _read_completed(output_csv) if resume else set()
        if output_csv.exists() and not resume and output_csv.stat().st_size:
            raise FileExistsError(
                f"{output_csv} already exists. Enable resume or choose a new run_name."
            )
        parameters = dict(algorithm_parameters[algorithm])
        for index, presentation in enumerate(presentations, start=1):
            pres_id = presentation["pres_id"]
            if pres_id in completed:
                continue
            if print_progress:
                print(
                    f"[{algorithm}] {index}/{len(presentations)} pres_id={pres_id}",
                    flush=True,
                )
            try:
                result = run_algorithm(
                    algorithm,
                    (presentation["r1"], presentation["r2"]),
                    parameters,
                )
                row = _result_row(algorithm, presentation, result)
                if not _parse_bool(row["solved"]):
                    unsolved.append((algorithm, pres_id, row["termination_reason"]))
            except Exception as error:
                row = _error_row(algorithm, presentation, error)
                _append_row(output_csv, row)
                if fail_fast:
                    raise
                continue
            _append_row(output_csv, row)

    if require_all_solved and unsolved:
        preview = ", ".join(
            f"{algorithm}:{pres_id} ({reason})"
            for algorithm, pres_id, reason in unsolved[:10]
        )
        raise RuntimeError(
            f"{len(unsolved)} algorithm/presentation runs were not solved: {preview}"
        )
    return run_dir


def result_coverage(run_dir: Path | str) -> list[Dict[str, Any]]:
    """Return a compact coverage summary for notebook display."""

    directory = Path(run_dir)
    coverage = []
    for algorithm in ALGORITHM_ORDER:
        path = directory / f"{algorithm}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        coverage.append(
            {
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "rows": len(rows),
                "solved": sum(_parse_bool(row["solved"]) for row in rows),
                "errors": sum(row["status"] == "error" for row in rows),
            }
        )
    return coverage


__all__ = [
    "RESULT_FIELDS",
    "load_presentations",
    "result_coverage",
    "run_experiment_suite",
]
