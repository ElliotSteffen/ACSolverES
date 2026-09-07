"""One-queue greedy search using standard substitutions and Single-COV moves.

This is the Single-COV analogue of ``standard_ccov_gs.py``: every popped state
expands ordinary AC substitution moves plus Single-COV neighbors, then all
resulting states go back into one global priority queue.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import single_cov
from .ccov_gs_experiment import WordState, _solver_state_key


def total_length(state: WordState) -> int:
    return len(state[0]) + len(state[1])


@dataclass(frozen=True)
class LayeredScoreWeights:
    cov_count_weight: float = 1.0
    last_inner_drop_weight: float = 0.0


DEFAULT_SCORE_WEIGHTS = LayeredScoreWeights(cov_count_weight=0.0, last_inner_drop_weight=0.0)
StateScore = Callable[[WordState], float]


def _length_score(state: WordState) -> float:
    """Baseline queue score: total relator length."""
    return float(total_length(state))


def load_single_cov_namespace() -> Dict[str, Any]:
    """Return the packaged Single-COV implementation namespace."""

    namespace = vars(single_cov)

    required = [
        "str_to_arr",
        "reduce_relator_nj",
        "canonical_pair_nj",
        "get_neighbors_nj",
        "state_to_key",
        "canonicalize_state_from_strings",
        "COVRelatorSolver",
        "build_cov_neighbor_state",
    ]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"Single COV namespace is missing required symbols: {missing}")
    return namespace


def _canonical_start(namespace: Dict[str, Any], presentation: WordState) -> WordState:
    state = namespace["canonicalize_state_from_strings"](*presentation)
    if state is None:
        raise ValueError(f"Could not canonicalize presentation: {presentation}")
    return namespace["state_to_key"](state)


def _is_trivial_solution(state: WordState) -> bool:
    return len(state[0]) == 1 and len(state[1]) == 1


def _priority(
    state: WordState,
    depth: int,
    insertion_index: int,
    *,
    state_score: StateScore,
    tie_strategy: str,
    random_tie_value: float,
) -> Tuple[Any, ...]:
    """Global queue priority: configured score, then configured depth tie."""

    if tie_strategy == "shallowest":
        depth_tie: float | int = depth
    elif tie_strategy == "deepest":
        depth_tie = -depth
    elif tie_strategy == "random":
        depth_tie = random_tie_value
    else:
        raise ValueError("tie_strategy must be one of: shallowest, deepest, random.")

    return (
        state_score(state),
        depth_tie,
        insertion_index,
        _solver_state_key(state),
    )


def _best_result_key(state: WordState, depth: int) -> Tuple[int, int, Tuple[int, Tuple[int, ...], int, Tuple[int, ...]]]:
    return (total_length(state), depth, _solver_state_key(state))


def _state_path_from_parents(
    parents: Dict[WordState, Optional[WordState]],
    state: WordState,
) -> List[WordState]:
    path: List[WordState] = []
    current: Optional[WordState] = state
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def _move_path_from_parents(
    path: Sequence[WordState],
    move_to_state: Dict[WordState, Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    moves: List[Dict[str, Any]] = []
    for state in path[1:]:
        move = move_to_state[state]
        if move is not None:
            moves.append(move)
    return moves


def _enqueue(
    *,
    queue: List[Tuple[Any, WordState, int]],
    insertion_counter: Iterable[int],
    seen_states: set[WordState],
    parents: Dict[WordState, Optional[WordState]],
    move_to_state: Dict[WordState, Optional[Dict[str, Any]]],
    state: WordState,
    parent: WordState,
    move: Dict[str, Any],
    depth: int,
    state_score: StateScore,
    tie_strategy: str,
    random_source: random.Random,
) -> bool:
    if state in seen_states:
        return False

    seen_states.add(state)
    parents[state] = parent
    move_to_state[state] = move
    insertion_index = next(insertion_counter)
    heapq.heappush(
        queue,
        (
            _priority(
                state,
                depth,
                insertion_index,
                state_score=state_score,
                tie_strategy=tie_strategy,
                random_tie_value=random_source.random(),
            ),
            state,
            depth,
        ),
    )
    return True


def _single_cov_move_from_candidate(
    source_state: WordState,
    next_state: WordState,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "move_type": "complete_cov",
        "cov_variant": "single_cov",
        "from_state": source_state,
        "to_state": next_state,
        "source_index": candidate.get("source_index"),
        "rotation_index": candidate.get("rotation_index"),
        "rotated_source_relator": candidate.get("rotated_source_relator"),
        "subword": candidate.get("subword"),
        "target": candidate.get("target"),
        "required_replacements": 1,
        "match_positions": candidate.get("match_positions"),
        "replacement": candidate.get("replacement"),
        "temporary_relator": candidate.get("temporary_relator"),
        "defining_relator": candidate.get("defining_relator"),
        "next_total_length": total_length(next_state),
    }


def _generate_single_cov_records(
    namespace: Dict[str, Any],
    state: WordState,
    *,
    max_len: int,
    cov_min_subword_len: int,
    deduplicate_next_states: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Return legal Single-COV neighbors and candidate-count diagnostics."""

    solver = namespace["COVRelatorSolver"](
        state[0],
        state[1],
        max_nodes=1,
        max_len=max_len,
        verbose=False,
        stop_early=False,
        cov_min_subword_len=cov_min_subword_len,
    )
    build_cov_neighbor_state = namespace["build_cov_neighbor_state"]
    state_to_key = namespace["state_to_key"]

    records: List[Dict[str, Any]] = []
    seen_next_states: set[WordState] = set()
    candidate_count = 0
    valid_candidate_count = 0

    for candidate in solver._generate_cov_candidates(*state):
        candidate_count += 1
        canonical_state = build_cov_neighbor_state(state[0], state[1], candidate)
        if canonical_state is None:
            continue
        next_state = state_to_key(canonical_state)
        if total_length(next_state) >= max_len:
            continue
        valid_candidate_count += 1
        if deduplicate_next_states and next_state in seen_next_states:
            continue
        seen_next_states.add(next_state)
        records.append(
            {
                "candidate": candidate,
                "next_state": next_state,
            }
        )

    return records, candidate_count, valid_candidate_count


def run_standard_single_cov_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    min_standard_moves_between_cov: int = 0,
    cov_gap_total_length_multiplier: Optional[float] = None,
    tie_strategy: str = "shallowest",
    state_score: StateScore = _length_score,
    random_seed: Optional[int] = 0,
    print_progress: bool = False,
    namespace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one-queue greedy search expanding standard and Single-COV moves."""

    if total_node_budget <= 0:
        raise ValueError("total_node_budget must be positive.")
    if min_standard_moves_between_cov < 0:
        raise ValueError("min_standard_moves_between_cov must be nonnegative.")
    if cov_gap_total_length_multiplier is not None and cov_gap_total_length_multiplier < 0:
        raise ValueError("cov_gap_total_length_multiplier must be nonnegative.")

    namespace = namespace or load_single_cov_namespace()
    random_source = random.Random(random_seed)
    start_state = _canonical_start(namespace, presentation)

    def required_cov_gap(state: WordState) -> int:
        fixed_gap = min_standard_moves_between_cov
        if cov_gap_total_length_multiplier is None:
            return fixed_gap
        dynamic_gap = math.ceil(cov_gap_total_length_multiplier * total_length(state))
        return max(fixed_gap, dynamic_gap)

    queue: List[Tuple[Any, WordState, int]] = []
    insertion_counter = count()
    parents: Dict[WordState, Optional[WordState]] = {start_state: None}
    move_to_state: Dict[WordState, Optional[Dict[str, Any]]] = {start_state: None}
    seen_states: set[WordState] = {start_state}
    expanded_states: set[WordState] = set()
    standard_steps_since_last_cov: Dict[WordState, int] = {
        start_state: required_cov_gap(start_state)
    }

    heapq.heappush(
        queue,
        (
            _priority(
                start_state,
                0,
                next(insertion_counter),
                state_score=state_score,
                tie_strategy=tie_strategy,
                random_tie_value=random_source.random(),
            ),
            start_state,
            0,
        ),
    )

    best_state = start_state
    best_depth = 0
    nodes_used = 0
    standard_generated_count = 0
    standard_enqueued_count = 0
    cov_expansion_count = 0
    cov_candidate_count = 0
    cov_valid_candidate_count = 0
    cov_enqueued_count = 0
    cov_gate_skipped_count = 0

    while queue and nodes_used < total_node_budget:
        _, current_state, depth = heapq.heappop(queue)
        if current_state in expanded_states:
            continue

        expanded_states.add(current_state)
        nodes_used += 1

        if _best_result_key(current_state, depth) < _best_result_key(best_state, best_depth):
            best_state = current_state
            best_depth = depth

        if _is_trivial_solution(current_state):
            path = _state_path_from_parents(parents, current_state)
            moves = _move_path_from_parents(path, move_to_state)
            return {
                "presentation": presentation,
                "canonical_start": start_state,
                "solved": True,
                "solved_trivial": True,
                "termination_reason": "solved",
                "total_nodes_used": nodes_used,
                "path_length": len(path) - 1,
                "path": path,
                "moves": moves,
                "final_state": current_state,
                "final_total_length": total_length(current_state),
                "outer_iterations": nodes_used,
                "queue_size": len(queue),
                "enqueued_state_count": len(seen_states),
                "expanded_state_count": len(expanded_states),
                "sub_pop_count": nodes_used,
                "cov_pop_count": cov_expansion_count,
                "cov_moves_in_path": sum(move.get("move_type") == "complete_cov" for move in moves),
                "cov_expansion_count": cov_expansion_count,
                "inner_patience_nodes": 0,
                "min_standard_moves_between_cov": min_standard_moves_between_cov,
                "cov_gap_total_length_multiplier": cov_gap_total_length_multiplier,
                "outer_tie_strategy": tie_strategy,
                "inner_tie_strategy": "none",
                "score_weights": DEFAULT_SCORE_WEIGHTS,
                "standard_generated_count": standard_generated_count,
                "standard_enqueued_count": standard_enqueued_count,
                "cov_candidate_count": cov_candidate_count,
                "cov_valid_candidate_count": cov_valid_candidate_count,
                "cov_enqueued_count": cov_enqueued_count,
                "cov_gate_skipped_count": cov_gate_skipped_count,
            }

        r1_arr = namespace["str_to_arr"](current_state[0])
        r2_arr = namespace["str_to_arr"](current_state[1])
        for nr1, nr2 in namespace["get_neighbors_nj"](r1_arr, r2_arr):
            nr1_reduced = namespace["reduce_relator_nj"](nr1)
            nr2_reduced = namespace["reduce_relator_nj"](nr2)
            if len(nr1_reduced) + len(nr2_reduced) >= max_len:
                continue

            canon_r1, canon_r2 = namespace["canonical_pair_nj"](nr1_reduced, nr2_reduced)
            next_state = namespace["state_to_key"]((canon_r1, canon_r2))
            standard_generated_count += 1
            move = {
                "move_type": "standard_substitution",
                "from_state": current_state,
                "to_state": next_state,
                "standard_moves_since_last_cov": standard_steps_since_last_cov.get(current_state, 0) + 1,
            }
            if _enqueue(
                queue=queue,
                insertion_counter=insertion_counter,
                seen_states=seen_states,
                parents=parents,
                move_to_state=move_to_state,
                state=next_state,
                parent=current_state,
                move=move,
                depth=depth + 1,
                state_score=state_score,
                tie_strategy=tie_strategy,
                random_source=random_source,
            ):
                standard_enqueued_count += 1
                standard_steps_since_last_cov[next_state] = standard_steps_since_last_cov.get(current_state, 0) + 1

        current_required_cov_gap = required_cov_gap(current_state)
        if standard_steps_since_last_cov.get(current_state, 0) < current_required_cov_gap:
            cov_gate_skipped_count += 1
            continue

        cov_expansion_count += 1
        cov_records, raw_candidate_count, raw_valid_count = _generate_single_cov_records(
            namespace,
            current_state,
            max_len=max_len,
            cov_min_subword_len=cov_min_subword_len,
            deduplicate_next_states=deduplicate_next_states,
        )
        cov_candidate_count += raw_candidate_count
        cov_valid_candidate_count += raw_valid_count
        for record in cov_records:
            next_state = record["next_state"]
            move = _single_cov_move_from_candidate(current_state, next_state, record["candidate"])
            move["standard_moves_since_last_cov"] = standard_steps_since_last_cov.get(current_state, 0)
            if _enqueue(
                queue=queue,
                insertion_counter=insertion_counter,
                seen_states=seen_states,
                parents=parents,
                move_to_state=move_to_state,
                state=next_state,
                parent=current_state,
                move=move,
                depth=depth + 1,
                state_score=state_score,
                tie_strategy=tie_strategy,
                random_source=random_source,
            ):
                cov_enqueued_count += 1
                standard_steps_since_last_cov[next_state] = 0

        if print_progress and nodes_used % 1000 == 0:
            print(
                f"nodes={nodes_used} queue={len(queue)} seen={len(seen_states)} "
                f"best_len={total_length(best_state)} cov_enqueued={cov_enqueued_count}",
                flush=True,
            )

    final_state = best_state
    path = _state_path_from_parents(parents, final_state)
    moves = _move_path_from_parents(path, move_to_state)
    return {
        "presentation": presentation,
        "canonical_start": start_state,
        "solved": False,
        "solved_trivial": False,
        "termination_reason": "node_budget_exhausted" if nodes_used >= total_node_budget else "queue_exhausted",
        "total_nodes_used": nodes_used,
        "path_length": len(path) - 1,
        "path": path,
        "moves": moves,
        "final_state": final_state,
        "final_total_length": total_length(final_state),
        "outer_iterations": nodes_used,
        "queue_size": len(queue),
        "enqueued_state_count": len(seen_states),
        "expanded_state_count": len(expanded_states),
        "sub_pop_count": nodes_used,
        "cov_pop_count": cov_expansion_count,
        "cov_moves_in_path": sum(move.get("move_type") == "complete_cov" for move in moves),
        "cov_expansion_count": cov_expansion_count,
                "inner_patience_nodes": 0,
                "min_standard_moves_between_cov": min_standard_moves_between_cov,
                "cov_gap_total_length_multiplier": cov_gap_total_length_multiplier,
                "outer_tie_strategy": tie_strategy,
        "inner_tie_strategy": "none",
        "score_weights": DEFAULT_SCORE_WEIGHTS,
        "standard_generated_count": standard_generated_count,
        "standard_enqueued_count": standard_enqueued_count,
        "cov_candidate_count": cov_candidate_count,
        "cov_valid_candidate_count": cov_valid_candidate_count,
        "cov_enqueued_count": cov_enqueued_count,
        "cov_gate_skipped_count": cov_gate_skipped_count,
    }


if __name__ == "__main__":
    result = run_standard_single_cov_gs(("XyxYY", "XXYxY"), total_node_budget=10_000)
    print(result["solved"], result["total_nodes_used"], result["path"])
