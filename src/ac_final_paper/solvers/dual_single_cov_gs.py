"""Dual-queue greedy search using standard moves and Single-COV moves.

The search keeps two priority queues:

1. ``Q_sub`` expands standard substitution neighbors using ``state_score``.
2. ``Q_cov`` stores states already popped from ``Q_sub`` and expands COV
   neighbors according to ``cov_activation_score``. By default this is the
   same score as ``Q_sub``, but it can remain total length when standard-move
   ranking uses a structural heuristic.
3. A second lazy index over the unexpanded ``Q_sub`` states is ordered by
   ``cov_activation_score``. This means COV activation can compare the true
   best activation-score state in each queue without changing standard pops.

Tie rule between queues: choose ``Q_sub``. Within each queue, priority uses
that queue's configured score, then depth, then insertion order.
"""

from __future__ import annotations

import heapq
import random
from itertools import count
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from . import native_standard_successors as _native_standard_successors
except ImportError:
    _native_standard_successors = None

from .standard_single_cov_gs import (
    WordState,
    _canonical_start,
    _generate_single_cov_records,
    _is_trivial_solution,
    _single_cov_move_from_candidate,
    _solver_state_key,
    _state_path_from_parents,
    load_single_cov_namespace,
    total_length,
)


QueueItem = Tuple[Tuple[Any, ...], WordState, int]
StateScore = Callable[[WordState], Any]
AdditionalStandardMoveGenerator = Callable[
    [Dict[str, Any], WordState, int],
    Iterable[Tuple[WordState, Dict[str, Any]]],
]
AdditionalCovMoveGenerator = Callable[
    [Dict[str, Any], WordState, int],
    Iterable[Tuple[WordState, Dict[str, Any]]],
]
_STANDARD_SUBSTITUTION_MOVE = object()


def total_length_score(state: WordState) -> float:
    """Return total relator length as a numeric queue score."""
    return float(total_length(state))


def _length_score(state: WordState) -> float:
    """Backward-compatible baseline queue score alias."""
    return total_length_score(state)


def _python_standard_successors(
    namespace: Dict[str, Any],
    state: WordState,
    max_len: int,
) -> Iterable[WordState]:
    """Yield the legacy Numba standard successors in their original order."""
    r1_arr = namespace["str_to_arr"](state[0])
    r2_arr = namespace["str_to_arr"](state[1])
    for nr1, nr2 in namespace["get_neighbors_nj"](r1_arr, r2_arr):
        nr1_reduced = namespace["reduce_relator_nj"](nr1)
        nr2_reduced = namespace["reduce_relator_nj"](nr2)
        if len(nr1_reduced) + len(nr2_reduced) >= max_len:
            continue
        canon_r1, canon_r2 = namespace["canonical_pair_nj"](nr1_reduced, nr2_reduced)
        yield namespace["state_to_key"]((canon_r1, canon_r2))


def _enqueue_cov_records_into_sub_queue(
    *,
    namespace: Dict[str, Any],
    current_state: WordState,
    depth: int,
    max_len: int,
    cov_min_subword_len: int,
    deduplicate_next_states: bool,
    sub_queue: List[QueueItem],
    sub_activation_queue: List[QueueItem],
    insertion_counter: Iterable[int],
    seen_states: set[WordState],
    parents: Dict[WordState, Optional[WordState]],
    move_to_state: Dict[WordState, Any],
    standard_moves_since_last_cov: Dict[WordState, int],
    state_score: StateScore,
    cov_activation_score: StateScore,
    depth_tie_strategy: str,
    random_source: random.Random,
    source_queue: str,
    additional_cov_claimed_states: Optional[set[WordState]] = None,
) -> Tuple[int, int, int, List[Tuple[WordState, int]], int]:
    """Generate Single-COV records from one state and enqueue new states in Q_sub."""

    cov_records, raw_candidate_count, raw_valid_count = _generate_single_cov_records(
        namespace,
        current_state,
        max_len=max_len,
        cov_min_subword_len=cov_min_subword_len,
        deduplicate_next_states=deduplicate_next_states,
    )

    enqueued_count = 0
    enqueued_states: List[Tuple[WordState, int]] = []
    blocked_by_additional_cov_count = 0
    for record in cov_records:
        next_state = record["next_state"]
        if next_state in seen_states:
            if additional_cov_claimed_states and next_state in additional_cov_claimed_states:
                blocked_by_additional_cov_count += 1
            continue

        move = _single_cov_move_from_candidate(current_state, next_state, record["candidate"])
        move["source_queue"] = source_queue
        seen_states.add(next_state)
        parents[next_state] = current_state
        move_to_state[next_state] = move
        standard_moves_since_last_cov[next_state] = 0
        _push_queue(
            sub_queue,
            insertion_counter,
            next_state,
            depth + 1,
            state_score=state_score,
            depth_tie_strategy=depth_tie_strategy,
            random_source=random_source,
        )
        if sub_activation_queue is not sub_queue:
            _push_queue(
                sub_activation_queue,
                insertion_counter,
                next_state,
                depth + 1,
                state_score=cov_activation_score,
                depth_tie_strategy=depth_tie_strategy,
                random_source=random_source,
            )
        enqueued_count += 1
        enqueued_states.append((next_state, depth + 1))

    return (
        raw_candidate_count,
        raw_valid_count,
        enqueued_count,
        enqueued_states,
        blocked_by_additional_cov_count,
    )


def _depth_tie(depth: int, *, depth_tie_strategy: str, random_tie_value: float) -> float | int:
    if depth_tie_strategy == "shallowest":
        return depth
    if depth_tie_strategy == "deepest":
        return -depth
    if depth_tie_strategy == "random":
        return random_tie_value
    raise ValueError("depth_tie_strategy must be one of: shallowest, deepest, random.")


def _queue_priority(
    state: WordState,
    depth: int,
    insertion_index: int,
    *,
    state_score: StateScore,
    depth_tie_strategy: str,
    random_tie_value: float,
) -> Tuple[Any, ...]:
    # insertion_index is globally unique, so no later field can participate
    # in heap ordering. Avoid retaining a large lexicographic copy of every
    # state in both priority indexes.
    return (
        state_score(state),
        _depth_tie(depth, depth_tie_strategy=depth_tie_strategy, random_tie_value=random_tie_value),
        insertion_index,
    )


def _push_queue(
    queue: List[QueueItem],
    insertion_counter: Iterable[int],
    state: WordState,
    depth: int,
    *,
    state_score: StateScore,
    depth_tie_strategy: str,
    random_source: random.Random,
) -> None:
    insertion_index = next(insertion_counter)
    if state_score is _length_score or state_score is total_length_score:
        score = float(len(state[0]) + len(state[1]))
    else:
        score = state_score(state)
    if depth_tie_strategy == "deepest":
        depth_priority: float | int = -depth
    elif depth_tie_strategy == "shallowest":
        depth_priority = depth
    elif depth_tie_strategy == "random":
        depth_priority = random_source.random()
    else:
        raise ValueError("depth_tie_strategy must be one of: shallowest, deepest, random.")
    heapq.heappush(
        queue,
        (
            (score, depth_priority, insertion_index),
            state,
            depth,
        ),
    )


def _prune_live_frontier(
    *,
    sub_queue: List[QueueItem],
    sub_activation_queue: List[QueueItem],
    cov_queue: List[QueueItem],
    sub_expanded_states: set[WordState],
    cov_expanded_states: set[WordState],
    seen_states: set[WordState],
    parents: Dict[WordState, Optional[WordState]],
    move_to_state: Dict[WordState, Any],
    standard_moves_since_last_cov: Dict[WordState, int],
    cov_queued_states: set[WordState],
    additional_cov_claimed_states: set[WordState],
    max_live_frontier_states: int,
) -> int:
    """Keep the best half of a frontier that has exceeded its state cap.

    The node budget counts expansions, but every unexpanded state also owns
    parent and queue metadata. Pruning removes discarded, unexpanded states
    from that metadata as well as all heaps so it releases real memory.
    """

    live_sub = {
        state: item
        for item in sub_queue
        for state in (item[1],)
        if state not in sub_expanded_states
    }
    live_cov = {
        state: item
        for item in cov_queue
        for state in (item[1],)
        if state not in cov_expanded_states
    }
    candidates = [
        (item[0], 0, state, "sub") for state, item in live_sub.items()
    ] + [
        (item[0], 1, state, "cov") for state, item in live_cov.items()
    ]
    if len(candidates) <= max_live_frontier_states:
        return 0

    retain_count = max(1, max_live_frontier_states // 2)
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1], candidate[2]))
    retained = {(state, queue_name) for _, _, state, queue_name in candidates[:retain_count]}
    retained_sub = {state for state, queue_name in retained if queue_name == "sub"}
    retained_cov = {state for state, queue_name in retained if queue_name == "cov"}
    retained_states = retained_sub | retained_cov
    discarded = {state for _, _, state, _ in candidates} - retained_states

    sub_queue[:] = [live_sub[state] for state in retained_sub]
    heapq.heapify(sub_queue)
    sub_activation_queue[:] = [
        item
        for item in sub_activation_queue
        if item[1] in retained_sub and item[1] not in sub_expanded_states
    ]
    heapq.heapify(sub_activation_queue)
    cov_queue[:] = [live_cov[state] for state in retained_cov]
    heapq.heapify(cov_queue)

    metadata_discarded = discarded - sub_expanded_states - cov_expanded_states
    for state in metadata_discarded:
        seen_states.discard(state)
        parents.pop(state, None)
        move_to_state.pop(state, None)
        standard_moves_since_last_cov.pop(state, None)
        cov_queued_states.discard(state)
        additional_cov_claimed_states.discard(state)
    return len(discarded)


def _state_to_queue_name(
    sub_queue: List[QueueItem],
    sub_activation_queue: List[QueueItem],
    cov_queue: List[QueueItem],
    *,
    cov_preference_threshold: int,
    cov_activation_score: StateScore,
) -> str:
    """Choose which queue to pop next.

    The standard queue wins unless the COV queue's head is better under
    ``cov_activation_score`` by at least ``cov_preference_threshold + 1``.
    This lets standard substitutions use a structural ordering while COV
    activation remains governed by presentation length.
    """

    if sub_queue and not cov_queue:
        return "sub"
    if cov_queue and not sub_queue:
        return "cov"
    if not sub_queue and not cov_queue:
        raise RuntimeError("Cannot choose from two empty queues.")
    if sub_queue and not sub_activation_queue:
        return "sub"

    sub_score = sub_activation_queue[0][0][0]
    cov_score = cov_activation_score(cov_queue[0][1])
    required_advantage = cov_preference_threshold + 1
    return "cov" if sub_score - cov_score >= required_advantage else "sub"


def _discard_expanded_queue_entries(queue: List[QueueItem], expanded_states: set[WordState]) -> None:
    """Lazily remove queue entries that were already expanded through another index."""
    while queue and queue[0][1] in expanded_states:
        heapq.heappop(queue)


def _move_path_from_compact_parents(
    path: Sequence[WordState],
    move_to_state: Dict[WordState, Any],
) -> List[Dict[str, Any]]:
    """Materialize ordinary move dictionaries only along the returned path."""

    moves: List[Dict[str, Any]] = []
    for from_state, to_state in zip(path, path[1:]):
        move = move_to_state[to_state]
        if move is _STANDARD_SUBSTITUTION_MOVE:
            moves.append(
                {
                    "move_type": "standard_substitution",
                    "from_state": from_state,
                    "to_state": to_state,
                    "source_queue": "sub",
                }
            )
        elif move is not None:
            moves.append(move)
    return moves


def _update_best(
    *,
    current_best: WordState,
    current_best_depth: int,
    candidate: WordState,
    candidate_depth: int,
) -> Tuple[WordState, int]:
    candidate_length = total_length(candidate)
    current_length = total_length(current_best)
    if candidate_length != current_length:
        if candidate_length < current_length:
            return candidate, candidate_depth
        return current_best, current_best_depth
    if candidate_depth != current_best_depth:
        if candidate_depth < current_best_depth:
            return candidate, candidate_depth
        return current_best, current_best_depth
    # Preserve the original final tie rule, but only construct expensive
    # solver-order keys when length and depth are genuinely tied.
    if _solver_state_key(candidate) < _solver_state_key(current_best):
        return candidate, candidate_depth
    return current_best, current_best_depth


def run_dual_single_cov_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    depth_tie_strategy: str = "shallowest",
    cov_preference_threshold: int = 0,
    seed_initial_cov_neighbors: bool = False,
    seed_initial_additional_cov_moves: bool = False,
    min_standard_moves_between_cov: int = 0,
    state_score: StateScore = _length_score,
    cov_activation_score: Optional[StateScore] = None,
    additional_standard_moves: Optional[AdditionalStandardMoveGenerator] = None,
    additional_cov_moves: Optional[AdditionalCovMoveGenerator] = None,
    enable_single_cov_moves: bool = True,
    additional_cov_integration_mode: str = "full",
    max_live_frontier_states: Optional[int] = 5_000_000,
    random_seed: Optional[int] = 0,
    namespace: Optional[Dict[str, Any]] = None,
    print_progress: bool = False,
    compact_standard_move_metadata: bool = True,
    share_identical_sub_activation_queue: bool = True,
    use_native_standard_successors: bool = True,
) -> Dict[str, Any]:
    """Run dual-queue GS with standard moves and Single-COV moves.

    ``total_node_budget`` counts both standard-queue pops and second-queue pops.
    A state popped from ``Q_sub`` is always queued once into ``Q_cov`` unless it
    has already been COV-queued. This safety check protects against duplicate
    canonical states entering through different paths. ``state_score`` orders
    standard substitutions; ``cov_activation_score`` orders and activates COV.
    When omitted, ``cov_activation_score`` equals ``state_score``. Optional
    ``additional_standard_moves`` are canonical symbolic neighbors added to
    ``Q_sub`` at every standard-queue expansion. ``additional_cov_moves`` are
    added only when a state is popped from ``Q_cov`` for a COV activation.
    Set ``enable_single_cov_moves=False`` to use that queue exclusively for an
    ``additional_cov_moves`` family such as free-group automorphisms.
    ``seed_initial_additional_cov_moves`` additionally applies those moves to
    the initial presentation before the first queue pop. The diagnostic
    ``additional_cov_integration_mode`` can be ``full`` (normal behavior),
    ``queue_no_activation`` (omit the COV-activation index), ``seen_only``
    (reserve generated states without enqueuing them), or ``generate_only``.
    When the optional C++ extension is available,
    ``use_native_standard_successors`` fuses standard successor generation,
    reduction, and canonicalization. Otherwise the exact Numba path is used.
    """

    if total_node_budget <= 0:
        raise ValueError("total_node_budget must be positive.")
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if cov_preference_threshold < 0:
        raise ValueError("cov_preference_threshold must be nonnegative.")
    if min_standard_moves_between_cov < 0:
        raise ValueError("min_standard_moves_between_cov must be nonnegative.")
    if seed_initial_cov_neighbors and not enable_single_cov_moves:
        raise ValueError(
            "seed_initial_cov_neighbors requires enable_single_cov_moves=True. "
            "Use seed_initial_additional_cov_moves for an alternate move family."
        )
    if max_live_frontier_states is not None and max_live_frontier_states < 2:
        raise ValueError("max_live_frontier_states must be at least 2 or None.")
    valid_additional_cov_modes = {
        "full",
        "queue_no_activation",
        "seen_only",
        "generate_only",
    }
    if additional_cov_integration_mode not in valid_additional_cov_modes:
        raise ValueError(
            "additional_cov_integration_mode must be one of "
            f"{sorted(valid_additional_cov_modes)}."
        )

    namespace = namespace or load_single_cov_namespace()
    cov_activation_score = cov_activation_score or state_score
    random_source = random.Random(random_seed)
    start_state = _canonical_start(namespace, presentation)

    sub_queue: List[QueueItem] = []
    shared_sub_activation_order = (
        share_identical_sub_activation_queue
        and cov_activation_score is state_score
        and depth_tie_strategy != "random"
    )
    sub_activation_queue: List[QueueItem] = sub_queue if shared_sub_activation_order else []
    cov_queue: List[QueueItem] = []
    insertion_counter = count()

    parents: Dict[WordState, Optional[WordState]] = {start_state: None}
    move_to_state: Dict[WordState, Any] = {start_state: None}
    standard_moves_since_last_cov: Dict[WordState, int] = {start_state: 0}
    seen_states: set[WordState] = {start_state}

    sub_expanded_states: set[WordState] = set()
    cov_queued_states: set[WordState] = set()
    cov_expanded_states: set[WordState] = set()

    _push_queue(
        sub_queue,
        insertion_counter,
        start_state,
        0,
        state_score=state_score,
        depth_tie_strategy=depth_tie_strategy,
        random_source=random_source,
    )
    if sub_activation_queue is not sub_queue:
        _push_queue(
            sub_activation_queue,
            insertion_counter,
            start_state,
            0,
            state_score=cov_activation_score,
            depth_tie_strategy=depth_tie_strategy,
            random_source=random_source,
        )
    # These primary queues have one active entry per enqueued state. Keeping
    # counters avoids scanning large heaps after every expansion just to check
    # whether the live-frontier cap has been crossed.
    live_sub_queue_entries = 1
    live_cov_queue_entries = 0

    best_state = start_state
    best_depth = 0
    minimum_total_length = total_length(start_state)
    minimum_states: set[WordState] = {start_state}
    nodes_used = 0
    sub_pop_count = 0
    cov_pop_count = 0
    standard_generated_count = 0
    standard_enqueued_count = 0
    cov_candidate_count = 0
    cov_valid_candidate_count = 0
    cov_enqueued_count = 0
    cov_queued_count = 0
    initial_cov_seed_enqueued_count = 0
    initial_additional_cov_seed_enqueued_count = 0
    cov_queue_eligible_count = 0
    cov_queue_distance_blocked_count = 0
    additional_cov_generated_count = 0
    additional_cov_valid_novel_count = 0
    additional_cov_duplicate_or_seen_count = 0
    additional_cov_filtered_length_count = 0
    additional_cov_seen_reserved_count = 0
    additional_cov_enqueued_count = 0
    additional_cov_activation_index_enqueued_count = 0
    additional_cov_sub_pop_count = 0
    standard_blocked_by_additional_cov_seen_count = 0
    cov_blocked_by_additional_cov_seen_count = 0
    additional_cov_claimed_states: set[WordState] = set()
    frontier_prune_count = 0
    frontier_states_pruned = 0
    frontier_peak_live_state_count = live_sub_queue_entries

    def _record_minimum_state(state: WordState) -> None:
        nonlocal minimum_total_length

        state_length = total_length(state)
        if state_length < minimum_total_length:
            minimum_total_length = state_length
            minimum_states.clear()
            minimum_states.add(state)
        elif state_length == minimum_total_length:
            minimum_states.add(state)

    def _integrate_additional_cov_successors(
        current_state: WordState,
        depth: int,
        source_queue: str,
    ) -> None:
        nonlocal additional_cov_generated_count
        nonlocal additional_cov_valid_novel_count
        nonlocal additional_cov_duplicate_or_seen_count
        nonlocal additional_cov_filtered_length_count
        nonlocal additional_cov_seen_reserved_count
        nonlocal additional_cov_enqueued_count
        nonlocal additional_cov_activation_index_enqueued_count
        nonlocal live_sub_queue_entries
        nonlocal best_state
        nonlocal best_depth

        if additional_cov_moves is None:
            return

        batch_states: set[WordState] = set()
        for next_state, move in additional_cov_moves(namespace, current_state, max_len):
            additional_cov_generated_count += 1
            if total_length(next_state) >= max_len:
                additional_cov_filtered_length_count += 1
                continue
            if next_state in seen_states or next_state in batch_states:
                additional_cov_duplicate_or_seen_count += 1
                continue

            batch_states.add(next_state)
            additional_cov_valid_novel_count += 1
            if additional_cov_integration_mode == "generate_only":
                continue

            seen_states.add(next_state)
            _record_minimum_state(next_state)
            additional_cov_claimed_states.add(next_state)
            additional_cov_seen_reserved_count += 1
            if additional_cov_integration_mode == "seen_only":
                continue

            move = dict(move)
            move.setdefault("move_type", "additional_cov_move")
            move["from_state"] = current_state
            move["to_state"] = next_state
            move["source_queue"] = source_queue
            parents[next_state] = current_state
            move_to_state[next_state] = move
            standard_moves_since_last_cov[next_state] = 0
            _push_queue(
                sub_queue,
                insertion_counter,
                next_state,
                depth + 1,
                state_score=state_score,
                depth_tie_strategy=depth_tie_strategy,
                random_source=random_source,
            )
            live_sub_queue_entries += 1
            additional_cov_enqueued_count += 1

            if additional_cov_integration_mode == "full":
                if sub_activation_queue is not sub_queue:
                    _push_queue(
                        sub_activation_queue,
                        insertion_counter,
                        next_state,
                        depth + 1,
                        state_score=cov_activation_score,
                        depth_tie_strategy=depth_tie_strategy,
                        random_source=random_source,
                    )
                additional_cov_activation_index_enqueued_count += 1

            best_state, best_depth = _update_best(
                current_best=best_state,
                current_best_depth=best_depth,
                candidate=next_state,
                candidate_depth=depth + 1,
            )

    if enable_single_cov_moves and seed_initial_cov_neighbors:
        cov_expanded_states.add(start_state)
        cov_queued_states.add(start_state)
        (
            raw_candidate_count,
            raw_valid_count,
            enqueued_count,
            enqueued_states,
            blocked_by_additional_cov_count,
        ) = _enqueue_cov_records_into_sub_queue(
            namespace=namespace,
            current_state=start_state,
            depth=0,
            max_len=max_len,
            cov_min_subword_len=cov_min_subword_len,
            deduplicate_next_states=deduplicate_next_states,
            sub_queue=sub_queue,
            sub_activation_queue=sub_activation_queue,
            insertion_counter=insertion_counter,
            seen_states=seen_states,
            parents=parents,
            move_to_state=move_to_state,
            standard_moves_since_last_cov=standard_moves_since_last_cov,
            state_score=state_score,
            cov_activation_score=cov_activation_score,
            depth_tie_strategy=depth_tie_strategy,
            random_source=random_source,
            source_queue="initial_cov_seed",
            additional_cov_claimed_states=additional_cov_claimed_states,
        )
        cov_blocked_by_additional_cov_seen_count += blocked_by_additional_cov_count
        cov_candidate_count += raw_candidate_count
        cov_valid_candidate_count += raw_valid_count
        cov_enqueued_count += enqueued_count
        initial_cov_seed_enqueued_count += enqueued_count
        live_sub_queue_entries += enqueued_count
        for state, state_depth in enqueued_states:
            _record_minimum_state(state)
            best_state, best_depth = _update_best(
                current_best=best_state,
                current_best_depth=best_depth,
                candidate=state,
                candidate_depth=state_depth,
            )

    if seed_initial_additional_cov_moves and additional_cov_moves is not None:
        additional_before_seed = additional_cov_enqueued_count
        _integrate_additional_cov_successors(start_state, 0, "initial_cov_seed")
        initial_additional_cov_seed_enqueued_count = (
            additional_cov_enqueued_count - additional_before_seed
        )

    while nodes_used < total_node_budget:
        _discard_expanded_queue_entries(sub_queue, sub_expanded_states)
        _discard_expanded_queue_entries(sub_activation_queue, sub_expanded_states)
        _discard_expanded_queue_entries(cov_queue, cov_expanded_states)
        if not sub_queue and not cov_queue:
            break

        queue_name = _state_to_queue_name(
            sub_queue,
            sub_activation_queue,
            cov_queue,
            cov_preference_threshold=cov_preference_threshold,
            cov_activation_score=cov_activation_score,
        )

        if queue_name == "sub":
            _, current_state, depth = heapq.heappop(sub_queue)
            live_sub_queue_entries = max(0, live_sub_queue_entries - 1)
            if current_state in sub_expanded_states:
                continue

            sub_expanded_states.add(current_state)
            nodes_used += 1
            sub_pop_count += 1
            if current_state in additional_cov_claimed_states:
                additional_cov_sub_pop_count += 1
            best_state, best_depth = _update_best(
                current_best=best_state,
                current_best_depth=best_depth,
                candidate=current_state,
                candidate_depth=depth,
            )

            if _is_trivial_solution(current_state):
                path = _state_path_from_parents(parents, current_state)
                moves = _move_path_from_compact_parents(path, move_to_state)
                return _result(
                    presentation=presentation,
                    start_state=start_state,
                    solved=True,
                    termination_reason="solved",
                    final_state=current_state,
                    final_depth=depth,
                    parents=parents,
                    move_to_state=move_to_state,
                    path=path,
                    moves=moves,
                    nodes_used=nodes_used,
                    sub_queue=sub_queue,
                    cov_queue=cov_queue,
                    seen_states=seen_states,
                    sub_expanded_states=sub_expanded_states,
                    cov_expanded_states=cov_expanded_states,
                    depth_tie_strategy=depth_tie_strategy,
                    cov_preference_threshold=cov_preference_threshold,
                    enable_single_cov_moves=enable_single_cov_moves,
                    seed_initial_cov_neighbors=seed_initial_cov_neighbors,
                    seed_initial_additional_cov_moves=seed_initial_additional_cov_moves,
                    min_standard_moves_between_cov=min_standard_moves_between_cov,
                    sub_pop_count=sub_pop_count,
                    cov_pop_count=cov_pop_count,
                    cov_queued_count=cov_queued_count,
                    initial_cov_seed_enqueued_count=initial_cov_seed_enqueued_count,
                    initial_additional_cov_seed_enqueued_count=(
                        initial_additional_cov_seed_enqueued_count
                    ),
                    cov_queue_eligible_count=cov_queue_eligible_count,
                    cov_queue_distance_blocked_count=cov_queue_distance_blocked_count,
                    standard_generated_count=standard_generated_count,
                    standard_enqueued_count=standard_enqueued_count,
                    cov_candidate_count=cov_candidate_count,
                    cov_valid_candidate_count=cov_valid_candidate_count,
                    cov_enqueued_count=cov_enqueued_count,
                    additional_cov_integration_mode=additional_cov_integration_mode,
                    additional_cov_generated_count=additional_cov_generated_count,
                    additional_cov_valid_novel_count=additional_cov_valid_novel_count,
                    additional_cov_duplicate_or_seen_count=additional_cov_duplicate_or_seen_count,
                    additional_cov_filtered_length_count=additional_cov_filtered_length_count,
                    additional_cov_seen_reserved_count=additional_cov_seen_reserved_count,
                    additional_cov_enqueued_count=additional_cov_enqueued_count,
                    additional_cov_activation_index_enqueued_count=(
                        additional_cov_activation_index_enqueued_count
                    ),
                    additional_cov_sub_pop_count=additional_cov_sub_pop_count,
                    standard_blocked_by_additional_cov_seen_count=(
                        standard_blocked_by_additional_cov_seen_count
                    ),
                    cov_blocked_by_additional_cov_seen_count=(
                        cov_blocked_by_additional_cov_seen_count
                    ),
                    max_live_frontier_states=max_live_frontier_states,
                    frontier_prune_count=frontier_prune_count,
                    frontier_states_pruned=frontier_states_pruned,
                    frontier_peak_live_state_count=frontier_peak_live_state_count,
                    minimum_states=minimum_states,
                )

            if use_native_standard_successors and _native_standard_successors is not None:
                standard_successors = _native_standard_successors.standard_successors(
                    current_state[0],
                    current_state[1],
                    max_len,
                )
            else:
                standard_successors = _python_standard_successors(
                    namespace,
                    current_state,
                    max_len,
                )
            next_depth = depth + 1
            next_standard_gap = standard_moves_since_last_cov[current_state] + 1
            if depth_tie_strategy == "deepest":
                fixed_next_depth_priority: Optional[int] = -next_depth
            elif depth_tie_strategy == "shallowest":
                fixed_next_depth_priority = next_depth
            else:
                fixed_next_depth_priority = None
            for next_state in standard_successors:
                standard_generated_count += 1
                if next_state in seen_states:
                    if next_state in additional_cov_claimed_states:
                        standard_blocked_by_additional_cov_seen_count += 1
                    continue

                move = (
                    _STANDARD_SUBSTITUTION_MOVE
                    if compact_standard_move_metadata
                    else {
                        "move_type": "standard_substitution",
                        "from_state": current_state,
                        "to_state": next_state,
                        "source_queue": "sub",
                    }
                )
                seen_states.add(next_state)
                _record_minimum_state(next_state)
                parents[next_state] = current_state
                move_to_state[next_state] = move
                standard_moves_since_last_cov[next_state] = next_standard_gap

                candidate_length = len(next_state[0]) + len(next_state[1])
                score = (
                    float(candidate_length)
                    if state_score is _length_score or state_score is total_length_score
                    else state_score(next_state)
                )
                depth_priority = (
                    fixed_next_depth_priority
                    if fixed_next_depth_priority is not None
                    else random_source.random()
                )
                heapq.heappush(
                    sub_queue,
                    (
                        (score, depth_priority, next(insertion_counter)),
                        next_state,
                        next_depth,
                    ),
                )
                live_sub_queue_entries += 1
                if sub_activation_queue is not sub_queue:
                    activation_score = (
                        float(candidate_length)
                        if cov_activation_score is _length_score
                        or cov_activation_score is total_length_score
                        else cov_activation_score(next_state)
                    )
                    activation_depth_priority = (
                        fixed_next_depth_priority
                        if fixed_next_depth_priority is not None
                        else random_source.random()
                    )
                    heapq.heappush(
                        sub_activation_queue,
                        (
                            (
                                activation_score,
                                activation_depth_priority,
                                next(insertion_counter),
                            ),
                            next_state,
                            next_depth,
                        ),
                    )
                standard_enqueued_count += 1
                best_length = len(best_state[0]) + len(best_state[1])
                if candidate_length < best_length or (
                    candidate_length == best_length
                    and (
                        next_depth < best_depth
                        or (
                            next_depth == best_depth
                            and _solver_state_key(next_state) < _solver_state_key(best_state)
                        )
                    )
                ):
                    best_state = next_state
                    best_depth = next_depth

            if additional_standard_moves is not None:
                for next_state, move in additional_standard_moves(
                    namespace, current_state, max_len
                ):
                    standard_generated_count += 1
                    if total_length(next_state) >= max_len or next_state in seen_states:
                        if next_state in additional_cov_claimed_states:
                            standard_blocked_by_additional_cov_seen_count += 1
                        continue

                    move = dict(move)
                    move.setdefault("move_type", "additional_standard_move")
                    move["from_state"] = current_state
                    move["to_state"] = next_state
                    move["source_queue"] = "sub"
                    seen_states.add(next_state)
                    _record_minimum_state(next_state)
                    parents[next_state] = current_state
                    move_to_state[next_state] = move
                    standard_moves_since_last_cov[next_state] = (
                        standard_moves_since_last_cov[current_state] + 1
                    )
                    _push_queue(
                        sub_queue,
                        insertion_counter,
                        next_state,
                        depth + 1,
                        state_score=state_score,
                        depth_tie_strategy=depth_tie_strategy,
                        random_source=random_source,
                    )
                    live_sub_queue_entries += 1
                    if sub_activation_queue is not sub_queue:
                        _push_queue(
                            sub_activation_queue,
                            insertion_counter,
                            next_state,
                            depth + 1,
                            state_score=cov_activation_score,
                            depth_tie_strategy=depth_tie_strategy,
                            random_source=random_source,
                        )
                    standard_enqueued_count += 1
                    best_state, best_depth = _update_best(
                        current_best=best_state,
                        current_best_depth=best_depth,
                        candidate=next_state,
                        candidate_depth=depth + 1,
                    )

            if current_state not in cov_queued_states:
                if standard_moves_since_last_cov[current_state] < min_standard_moves_between_cov:
                    cov_queue_distance_blocked_count += 1
                    continue

                cov_queue_eligible_count += 1
                cov_queued_states.add(current_state)
                _push_queue(
                    cov_queue,
                    insertion_counter,
                    current_state,
                    depth,
                    state_score=cov_activation_score,
                    depth_tie_strategy=depth_tie_strategy,
                    random_source=random_source,
                )
                live_cov_queue_entries += 1
                cov_queued_count += 1

        else:
            _, current_state, depth = heapq.heappop(cov_queue)
            live_cov_queue_entries = max(0, live_cov_queue_entries - 1)
            if current_state in cov_expanded_states:
                continue

            cov_expanded_states.add(current_state)
            nodes_used += 1
            cov_pop_count += 1
            best_state, best_depth = _update_best(
                current_best=best_state,
                current_best_depth=best_depth,
                candidate=current_state,
                candidate_depth=depth,
            )

            if enable_single_cov_moves:
                (
                    raw_candidate_count,
                    raw_valid_count,
                    enqueued_count,
                    enqueued_states,
                    blocked_by_additional_cov_count,
                ) = _enqueue_cov_records_into_sub_queue(
                    namespace=namespace,
                    current_state=current_state,
                    depth=depth,
                    max_len=max_len,
                    cov_min_subword_len=cov_min_subword_len,
                    deduplicate_next_states=deduplicate_next_states,
                    sub_queue=sub_queue,
                    sub_activation_queue=sub_activation_queue,
                    insertion_counter=insertion_counter,
                    seen_states=seen_states,
                    parents=parents,
                    move_to_state=move_to_state,
                    standard_moves_since_last_cov=standard_moves_since_last_cov,
                    state_score=state_score,
                    cov_activation_score=cov_activation_score,
                    depth_tie_strategy=depth_tie_strategy,
                    random_source=random_source,
                    source_queue="cov",
                    additional_cov_claimed_states=additional_cov_claimed_states,
                )
                cov_blocked_by_additional_cov_seen_count += blocked_by_additional_cov_count
                cov_candidate_count += raw_candidate_count
                cov_valid_candidate_count += raw_valid_count
                cov_enqueued_count += enqueued_count
                live_sub_queue_entries += enqueued_count
                for state, state_depth in enqueued_states:
                    _record_minimum_state(state)
                    best_state, best_depth = _update_best(
                        current_best=best_state,
                        current_best_depth=best_depth,
                        candidate=state,
                        candidate_depth=state_depth,
                    )

            if additional_cov_moves is not None:
                _integrate_additional_cov_successors(current_state, depth, "cov_activation")

        if print_progress and nodes_used % 1000 == 0:
            print(
                f"nodes={nodes_used} sub_q={len(sub_queue)} cov_q={len(cov_queue)} "
                f"seen={len(seen_states)} best_len={total_length(best_state)}",
                flush=True,
            )

        live_frontier_count = live_sub_queue_entries + live_cov_queue_entries
        frontier_peak_live_state_count = max(frontier_peak_live_state_count, live_frontier_count)
        if (
            max_live_frontier_states is not None
            and live_frontier_count > max_live_frontier_states
        ):
            pruned_count = _prune_live_frontier(
                sub_queue=sub_queue,
                sub_activation_queue=sub_activation_queue,
                cov_queue=cov_queue,
                sub_expanded_states=sub_expanded_states,
                cov_expanded_states=cov_expanded_states,
                seen_states=seen_states,
                parents=parents,
                move_to_state=move_to_state,
                standard_moves_since_last_cov=standard_moves_since_last_cov,
                cov_queued_states=cov_queued_states,
                additional_cov_claimed_states=additional_cov_claimed_states,
                max_live_frontier_states=max_live_frontier_states,
            )
            frontier_prune_count += 1
            frontier_states_pruned += pruned_count
            live_sub_queue_entries = len(sub_queue)
            live_cov_queue_entries = len(cov_queue)
            if print_progress:
                print(
                    f"frontier_prune={frontier_prune_count} removed={pruned_count} "
                    f"retained={len(sub_queue) + len(cov_queue)}",
                    flush=True,
                )

    final_state = best_state
    path = _state_path_from_parents(parents, final_state)
    moves = _move_path_from_compact_parents(path, move_to_state)
    return _result(
        presentation=presentation,
        start_state=start_state,
        solved=False,
        termination_reason="node_budget_exhausted" if nodes_used >= total_node_budget else "queues_exhausted",
        final_state=final_state,
        final_depth=best_depth,
        parents=parents,
        move_to_state=move_to_state,
        path=path,
        moves=moves,
        nodes_used=nodes_used,
        sub_queue=sub_queue,
        cov_queue=cov_queue,
        seen_states=seen_states,
        sub_expanded_states=sub_expanded_states,
        cov_expanded_states=cov_expanded_states,
        depth_tie_strategy=depth_tie_strategy,
        cov_preference_threshold=cov_preference_threshold,
        enable_single_cov_moves=enable_single_cov_moves,
        seed_initial_cov_neighbors=seed_initial_cov_neighbors,
        seed_initial_additional_cov_moves=seed_initial_additional_cov_moves,
        min_standard_moves_between_cov=min_standard_moves_between_cov,
        sub_pop_count=sub_pop_count,
        cov_pop_count=cov_pop_count,
        cov_queued_count=cov_queued_count,
        initial_cov_seed_enqueued_count=initial_cov_seed_enqueued_count,
        initial_additional_cov_seed_enqueued_count=(
            initial_additional_cov_seed_enqueued_count
        ),
        cov_queue_eligible_count=cov_queue_eligible_count,
        cov_queue_distance_blocked_count=cov_queue_distance_blocked_count,
        standard_generated_count=standard_generated_count,
        standard_enqueued_count=standard_enqueued_count,
        cov_candidate_count=cov_candidate_count,
        cov_valid_candidate_count=cov_valid_candidate_count,
        cov_enqueued_count=cov_enqueued_count,
        additional_cov_integration_mode=additional_cov_integration_mode,
        additional_cov_generated_count=additional_cov_generated_count,
        additional_cov_valid_novel_count=additional_cov_valid_novel_count,
        additional_cov_duplicate_or_seen_count=additional_cov_duplicate_or_seen_count,
        additional_cov_filtered_length_count=additional_cov_filtered_length_count,
        additional_cov_seen_reserved_count=additional_cov_seen_reserved_count,
        additional_cov_enqueued_count=additional_cov_enqueued_count,
        additional_cov_activation_index_enqueued_count=(
            additional_cov_activation_index_enqueued_count
        ),
        additional_cov_sub_pop_count=additional_cov_sub_pop_count,
        standard_blocked_by_additional_cov_seen_count=(
            standard_blocked_by_additional_cov_seen_count
        ),
        cov_blocked_by_additional_cov_seen_count=cov_blocked_by_additional_cov_seen_count,
        max_live_frontier_states=max_live_frontier_states,
        frontier_prune_count=frontier_prune_count,
        frontier_states_pruned=frontier_states_pruned,
        frontier_peak_live_state_count=frontier_peak_live_state_count,
        minimum_states=minimum_states,
    )


def _result(
    *,
    presentation: WordState,
    start_state: WordState,
    solved: bool,
    termination_reason: str,
    final_state: WordState,
    final_depth: int,
    parents: Dict[WordState, Optional[WordState]],
    move_to_state: Dict[WordState, Optional[Dict[str, Any]]],
    path: Sequence[WordState],
    moves: Sequence[Dict[str, Any]],
    nodes_used: int,
    sub_queue: List[QueueItem],
    cov_queue: List[QueueItem],
    seen_states: set[WordState],
    sub_expanded_states: set[WordState],
    cov_expanded_states: set[WordState],
    depth_tie_strategy: str,
    cov_preference_threshold: int,
    enable_single_cov_moves: bool,
    seed_initial_cov_neighbors: bool,
    seed_initial_additional_cov_moves: bool,
    min_standard_moves_between_cov: int,
    sub_pop_count: int,
    cov_pop_count: int,
    cov_queued_count: int,
    initial_cov_seed_enqueued_count: int,
    initial_additional_cov_seed_enqueued_count: int,
    cov_queue_eligible_count: int,
    cov_queue_distance_blocked_count: int,
    standard_generated_count: int,
    standard_enqueued_count: int,
    cov_candidate_count: int,
    cov_valid_candidate_count: int,
    cov_enqueued_count: int,
    additional_cov_integration_mode: str,
    additional_cov_generated_count: int,
    additional_cov_valid_novel_count: int,
    additional_cov_duplicate_or_seen_count: int,
    additional_cov_filtered_length_count: int,
    additional_cov_seen_reserved_count: int,
    additional_cov_enqueued_count: int,
    additional_cov_activation_index_enqueued_count: int,
    additional_cov_sub_pop_count: int,
    standard_blocked_by_additional_cov_seen_count: int,
    cov_blocked_by_additional_cov_seen_count: int,
    max_live_frontier_states: Optional[int],
    frontier_prune_count: int,
    frontier_states_pruned: int,
    frontier_peak_live_state_count: int,
    minimum_states: set[WordState],
) -> Dict[str, Any]:
    ordered_minimum_states = sorted(minimum_states, key=_solver_state_key)
    return {
        "presentation": presentation,
        "canonical_start": start_state,
        "solved": solved,
        "solved_trivial": solved and _is_trivial_solution(final_state),
        "termination_reason": termination_reason,
        "total_nodes_used": nodes_used,
        "path_length": len(path) - 1,
        "path": list(path),
        "moves": list(moves),
        "final_state": final_state,
        "final_total_length": total_length(final_state),
        "minimum_total_length": total_length(final_state),
        "minimum_state_count": len(ordered_minimum_states),
        "minimum_length_states": ordered_minimum_states,
        "outer_iterations": nodes_used,
        "queue_size": len(sub_queue) + len(cov_queue),
        "sub_queue_size": len(sub_queue),
        "cov_queue_size": len(cov_queue),
        "enqueued_state_count": len(seen_states),
        "expanded_state_count": len(sub_expanded_states) + len(cov_expanded_states),
        "sub_expanded_count": len(sub_expanded_states),
        "cov_expansion_count": len(cov_expanded_states),
        "sub_pop_count": sub_pop_count,
        "cov_pop_count": cov_pop_count,
        "cov_queued_count": cov_queued_count,
        "initial_cov_seed_enqueued_count": initial_cov_seed_enqueued_count,
        "initial_additional_cov_seed_enqueued_count": (
            initial_additional_cov_seed_enqueued_count
        ),
        "cov_queue_eligible_count": cov_queue_eligible_count,
        "cov_queue_distance_blocked_count": cov_queue_distance_blocked_count,
        "min_standard_moves_between_cov": min_standard_moves_between_cov,
        "outer_tie_strategy": depth_tie_strategy,
        "cov_preference_threshold": cov_preference_threshold,
        "cov_required_length_advantage": cov_preference_threshold + 1,
        "single_cov_moves_enabled": enable_single_cov_moves,
        "seed_initial_cov_neighbors": seed_initial_cov_neighbors,
        "seed_initial_additional_cov_moves": seed_initial_additional_cov_moves,
        "standard_generated_count": standard_generated_count,
        "standard_enqueued_count": standard_enqueued_count,
        "cov_candidate_count": cov_candidate_count,
        "cov_valid_candidate_count": cov_valid_candidate_count,
        "cov_enqueued_count": cov_enqueued_count,
        "additional_cov_integration_mode": additional_cov_integration_mode,
        "additional_cov_generated_count": additional_cov_generated_count,
        "additional_cov_valid_novel_count": additional_cov_valid_novel_count,
        "additional_cov_duplicate_or_seen_count": additional_cov_duplicate_or_seen_count,
        "additional_cov_filtered_length_count": additional_cov_filtered_length_count,
        "additional_cov_seen_reserved_count": additional_cov_seen_reserved_count,
        "additional_cov_enqueued_count": additional_cov_enqueued_count,
        "additional_cov_activation_index_enqueued_count": (
            additional_cov_activation_index_enqueued_count
        ),
        "additional_cov_sub_pop_count": additional_cov_sub_pop_count,
        "standard_blocked_by_additional_cov_seen_count": (
            standard_blocked_by_additional_cov_seen_count
        ),
        "cov_blocked_by_additional_cov_seen_count": cov_blocked_by_additional_cov_seen_count,
        "max_live_frontier_states": max_live_frontier_states,
        "frontier_prune_count": frontier_prune_count,
        "frontier_states_pruned": frontier_states_pruned,
        "frontier_peak_live_state_count": frontier_peak_live_state_count,
        "cov_variant": "single_cov" if enable_single_cov_moves else "disabled",
        "queue_policy": "dual_sub_then_cov_strict_length_cov",
        "final_depth": final_depth,
    }


if __name__ == "__main__":
    result = run_dual_single_cov_gs(("XyxYY", "XXYxY"), total_node_budget=1_000, print_progress=True)
    print(result["solved"], result["total_nodes_used"], result["path"])
