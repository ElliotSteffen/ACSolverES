"""Paper-facing adapters for the three non-baseline search algorithms.

This module gives the final-paper experiment a small, stable interface over
the packaged Dual GS and Triple GS implementations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Tuple


WordState = Tuple[str, str]
from .solvers.compact_dual_gs import (
    compact_native_available,
    run_compact_dual_single_cov_gs,
)
from .solvers.dual_auto_cov_gs import (
    run_compact_dual_auto_cov_gs,
)
from .solvers.standard_single_cov_gs import load_single_cov_namespace
from .solvers.triple_gs import (
    compact_triple_native_available,
    load_complete_cov_namespace,
    run_compact_triple_gs,
)


SINGLE_COV_DUAL = "single_auto_cov_dual"
AUTO_DUAL = "auto_dual"
AUTO_NONAUTO_TRIPLE = "auto_nonauto_cov_triple"

ALGORITHM_ORDER = (
    SINGLE_COV_DUAL,
    AUTO_DUAL,
    AUTO_NONAUTO_TRIPLE,
)

ALGORITHM_LABELS = {
    SINGLE_COV_DUAL: "Sub + Single-Auto-COV Dual GS",
    AUTO_DUAL: "Sub + Auto Dual GS",
    AUTO_NONAUTO_TRIPLE: "Sub + Auto + Non-Auto COV Triple GS",
}


@lru_cache(maxsize=1)
def _single_cov_namespace() -> Dict[str, Any]:
    return load_single_cov_namespace()


@lru_cache(maxsize=1)
def _complete_cov_namespace() -> Dict[str, Any]:
    return load_complete_cov_namespace()


def verify_native_backends() -> None:
    """Fail early when a compact backend needed for the 640-case run is absent."""

    missing = []
    if not compact_native_available():
        missing.append("compact Dual GS")
    if not compact_triple_native_available():
        missing.append("compact Triple GS")
    if missing:
        raise RuntimeError(
            "Missing native backend(s): "
            + ", ".join(missing)
            + ". Build the native extensions before running the full benchmark."
        )


def run_single_auto_cov_dual(
    presentation: WordState,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    options = dict(parameters)
    options.setdefault("namespace", _single_cov_namespace())
    return run_compact_dual_single_cov_gs(presentation, **options)


def run_auto_dual(
    presentation: WordState,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    options = dict(parameters)
    options.setdefault("namespace", _single_cov_namespace())
    return run_compact_dual_auto_cov_gs(presentation, **options)


def run_auto_nonauto_cov_triple(
    presentation: WordState,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    options = dict(parameters)
    options.setdefault("namespace", _complete_cov_namespace())
    return run_compact_triple_gs(presentation, **options)


RUNNERS: Dict[str, Callable[[WordState, Mapping[str, Any]], Dict[str, Any]]] = {
    SINGLE_COV_DUAL: run_single_auto_cov_dual,
    AUTO_DUAL: run_auto_dual,
    AUTO_NONAUTO_TRIPLE: run_auto_nonauto_cov_triple,
}


def run_algorithm(
    algorithm: str,
    presentation: WordState,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        runner = RUNNERS[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; choose from {tuple(RUNNERS)}."
        ) from exc
    return runner(presentation, parameters)


def warm_up(max_len: int = 50) -> None:
    """Pay one-time native initialization costs before the recorded benchmark."""

    verify_native_backends()
    trivial = ("x", "y")
    run_single_auto_cov_dual(
        trivial,
        {
            "total_node_budget": 1,
            "max_len": max_len,
            "seed_initial_cov_neighbors": False,
        },
    )
    run_auto_dual(
        trivial,
        {
            "total_node_budget": 1,
            "max_len": max_len,
            "seed_initial_auto_cov_neighbors": False,
        },
    )
    run_auto_nonauto_cov_triple(
        trivial,
        {
            "total_node_budget": 1,
            "max_len": max_len,
            "seed_initial_auto_cov_neighbors": False,
            "seed_initial_complete_cov_neighbors": False,
        },
    )


__all__ = [
    "ALGORITHM_LABELS",
    "ALGORITHM_ORDER",
    "AUTO_DUAL",
    "AUTO_NONAUTO_TRIPLE",
    "SINGLE_COV_DUAL",
    "run_algorithm",
    "verify_native_backends",
    "warm_up",
]
