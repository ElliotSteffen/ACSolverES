"""Compact C++ backend for the current length-ordered Dual GS algorithm."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .dual_single_cov_gs import WordState
from .standard_single_cov_gs import (
    _canonical_start,
    _generate_single_cov_records,
    _single_cov_move_from_candidate,
    load_single_cov_namespace,
)


try:
    from . import native_compact_dual_gs as _native
except ImportError:
    _native = None


def compact_native_available() -> bool:
    return _native is not None and hasattr(_native, "run_compact_dual_gs")


def run_compact_dual_single_cov_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    depth_tie_strategy: str = "deepest",
    cov_preference_threshold: int = 2,
    seed_initial_cov_neighbors: bool = True,
    seed_initial_additional_cov_moves: bool = False,
    min_standard_moves_between_cov: int = 10,
    max_live_frontier_states: Optional[int] = 5_000_000,
    namespace: Optional[Dict[str, Any]] = None,
    print_progress: bool = False,
    progress_interval: int = 10_000,
    **unsupported_options: Any,
) -> Dict[str, Any]:
    """Run the exact current Dual GS policy with compact packed C++ storage.

    This backend intentionally supports only the production length score and
    standard plus Single-COV moves. Unsupported heuristic or automorphism
    options fail explicitly instead of silently changing search behavior.
    """

    if not compact_native_available():
        raise RuntimeError(
            "The compact Dual GS extension is not built. Run "
            "`python -m pip install -e .` from the repository root first."
        )
    if seed_initial_additional_cov_moves:
        raise ValueError("The compact backend does not support additional COV moves.")
    if not deduplicate_next_states:
        raise ValueError("The compact backend requires deduplicate_next_states=True.")
    unsupported = {
        key: value
        for key, value in unsupported_options.items()
        if key
        not in {
            "random_seed",
            "use_native_standard_successors",
            "compact_standard_move_metadata",
            "share_identical_sub_activation_queue",
        }
        and value is not None
    }
    if unsupported:
        raise ValueError(f"Unsupported compact-backend options: {sorted(unsupported)}")

    namespace = namespace or load_single_cov_namespace()
    start_state = _canonical_start(namespace, presentation)

    def cov_expander(state: WordState, source_queue: str):
        records, raw_candidate_count, raw_valid_count = _generate_single_cov_records(
            namespace,
            state,
            max_len=max_len,
            cov_min_subword_len=cov_min_subword_len,
            deduplicate_next_states=True,
        )
        neighbors = []
        for record in records:
            next_state = record["next_state"]
            move = _single_cov_move_from_candidate(state, next_state, record["candidate"])
            move["source_queue"] = source_queue
            neighbors.append((next_state, move))
        return neighbors, raw_candidate_count, raw_valid_count

    result = _native.run_compact_dual_gs(
        start_state[0],
        start_state[1],
        cov_expander,
        total_node_budget=total_node_budget,
        max_total_length=max_len,
        cov_preference_threshold=cov_preference_threshold,
        seed_initial_cov_neighbors=seed_initial_cov_neighbors,
        min_standard_moves_between_cov=min_standard_moves_between_cov,
        depth_tie_strategy=depth_tie_strategy,
        max_live_frontier_states=(
            -1 if max_live_frontier_states is None else max_live_frontier_states
        ),
        progress_interval=progress_interval if print_progress else 0,
    )
    result["presentation"] = presentation
    return result


__all__ = ["compact_native_available", "run_compact_dual_single_cov_gs"]
