"""Three-queue greedy search with lazy non-automorphic Complete COV.

Q_sub expands ordinary substitutions, Q_auto expands the production 24-map
Auto COV family, and Q_complete expands only Complete COV candidates that use
at least two copies of a subword containing at least two target-family letters.
Every unique Complete COV root is inserted directly into Q_sub.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from itertools import count
from math import isqrt
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .ccov_gs_experiment import (
    _solver_state_key,
    automorphic_equivalence_key,
    load_complete_cov_namespace,
)
from .dual_auto_cov_gs import AUTO_COV_MAPS, auto_cov_successors
from .compact_dual_gs import _native as _compact_native
from .dual_single_cov_gs import (
    _native_standard_successors,
    _python_standard_successors,
)
from .standard_single_cov_gs import (
    WordState,
    _canonical_start,
    _is_trivial_solution,
    total_length,
)


QueueEntry = Tuple[Tuple[int, int, int], WordState, int]
TARGETS = ("x", "y")
INVERSE = {"x": "X", "X": "x", "y": "Y", "Y": "y", "z": "Z", "Z": "z"}

try:
    from . import native_compact_triple_gs as _compact_triple_native
except ImportError:
    _compact_triple_native = _compact_native

try:
    from . import native_compact_triple_gs_queue_activation as _compact_triple_queue_native
except ImportError:
    _compact_triple_queue_native = None


def _is_composite(value: int) -> bool:
    if value < 4:
        return False
    for divisor in range(2, isqrt(value) + 1):
        if value % divisor == 0:
            return True
    return False


def _target_count(word: str, target: str) -> int:
    inverse = INVERSE[target]
    return sum(letter in (target, inverse) for letter in word)


def _inverse_word(word: str) -> str:
    return "".join(INVERSE[letter] for letter in reversed(word))


def _cyclic_subword(word: str, start: int, length: int) -> str:
    return "".join(word[(start + offset) % len(word)] for offset in range(length))


def _maximum_other_relator_rewrite_patterns(
    word: str,
    subword: str,
) -> List[Tuple[Tuple[int, str], ...]]:
    """Return every maximum set of disjoint w->z or w^-1->Z rewrites."""

    inverse_subword = _inverse_word(subword)
    matches: List[Tuple[int, str]] = []
    for start in range(len(word)):
        observed = _cyclic_subword(word, start, len(subword))
        if observed == subword:
            matches.append((start, "z"))
        if observed == inverse_subword and inverse_subword != subword:
            matches.append((start, "Z"))

    if not matches:
        return [()]

    intervals = {
        match: {
            (match[0] + offset) % len(word) for offset in range(len(subword))
        }
        for match in matches
    }
    maximum_size = 0
    maximum_patterns: List[Tuple[Tuple[int, str], ...]] = []

    def search(
        next_index: int,
        selected: List[Tuple[int, str]],
        occupied: set[int],
    ) -> None:
        nonlocal maximum_size, maximum_patterns
        if len(selected) + len(matches) - next_index < maximum_size:
            return
        if next_index == len(matches):
            pattern = tuple(selected)
            if len(pattern) > maximum_size:
                maximum_size = len(pattern)
                maximum_patterns = [pattern]
            elif len(pattern) == maximum_size:
                maximum_patterns.append(pattern)
            return

        match = matches[next_index]
        interval = intervals[match]
        if not occupied.intersection(interval):
            selected.append(match)
            search(next_index + 1, selected, occupied | interval)
            selected.pop()
        search(next_index + 1, selected, occupied)

    search(0, [], set())
    return sorted(set(maximum_patterns))


def _apply_other_relator_rewrite_pattern(
    word: str,
    subword_length: int,
    pattern: Sequence[Tuple[int, str]],
) -> str:
    if not pattern:
        return word

    occupied = {
        (start + offset) % len(word)
        for start, _ in pattern
        for offset in range(subword_length)
    }
    cut = next(
        (position for position in range(len(word)) if position not in occupied),
        min(start for start, _ in pattern),
    )
    rotated_word = word[cut:] + word[:cut]
    rotated_pattern = sorted(
        ((start - cut) % len(word), replacement)
        for start, replacement in pattern
    )
    pieces = []
    cursor = 0
    for start, replacement in rotated_pattern:
        pieces.extend((rotated_word[cursor:start], replacement))
        cursor = start + subword_length
    pieces.append(rotated_word[cursor:])
    return "".join(pieces)


def _build_complete_cov_neighbor(
    namespace: Dict[str, Any],
    state: WordState,
    candidate: Dict[str, Any],
    rewritten_other_relator: str,
) -> Optional[WordState]:
    target = candidate["target"]
    replacement = candidate["replacement"]
    new_other = namespace["reduce_word"](
        namespace["substitute_target"](
            rewritten_other_relator,
            target,
            replacement,
        )
    )
    new_defining = namespace["reduce_word"](
        namespace["substitute_target"](
            candidate["defining_relator"],
            target,
            replacement,
        )
    )
    if not new_other or not new_defining:
        return None
    raw_state = namespace["canonicalize_cov_state_from_strings"](
        new_other,
        new_defining,
        target,
    )
    return None if raw_state is None else namespace["state_to_key"](raw_state)


def complete_cov_composite_pairs(state: WordState) -> Tuple[Tuple[int, str, int], ...]:
    """Return (relator index, target, n=N-1) pairs that can have proper divisors."""

    eligible = []
    for source_index, relator in enumerate(state):
        for target in TARGETS:
            n = _target_count(relator, target) - 1
            if _is_composite(n):
                eligible.append((source_index, target, n))
    return tuple(eligible)


def _depth_priority(depth: int, strategy: str) -> int:
    if strategy == "deepest":
        return -depth
    if strategy == "shallowest":
        return depth
    raise ValueError("depth_tie_strategy must be 'deepest' or 'shallowest'.")


def _push(
    queue: List[QueueEntry],
    state: WordState,
    depth: int,
    insertion_counter: Iterable[int],
    depth_tie_strategy: str,
) -> None:
    heapq.heappush(
        queue,
        (
            (
                total_length(state),
                _depth_priority(depth, depth_tie_strategy),
                next(insertion_counter),
            ),
            state,
            depth,
        ),
    )


def _standard_successors(
    namespace: Dict[str, Any],
    state: WordState,
    max_len: int,
    use_native: bool,
) -> Sequence[WordState] | Iterable[WordState]:
    if use_native and _native_standard_successors is not None:
        return _native_standard_successors.standard_successors(
            state[0], state[1], max_len
        )
    return _python_standard_successors(namespace, state, max_len)


def generate_non_automorphic_complete_cov_moves(
    namespace: Dict[str, Any],
    state: WordState,
    *,
    max_len: int = 50,
    min_subword_len: int = 2,
    automorphic_key: Optional[Callable[[WordState], WordState]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate proper-divisor Complete COV neighbors and their certificates."""

    equivalence_key = automorphic_key or (
        lambda candidate_state: automorphic_equivalence_key(
            namespace, candidate_state
        )
    )

    stats = {
        "composite_source_target_pair_count": 0,
        "cyclic_subword_count": 0,
        "target_count_filter_pass_count": 0,
        "occurrence_combination_count": 0,
        "valid_candidate_count": 0,
        "length_filtered_count": 0,
        "source_automorphic_filtered_count": 0,
        "duplicate_candidate_count": 0,
        "unique_neighbor_count": 0,
        "other_rewrite_pattern_count": 0,
        "other_rewrite_nonempty_pattern_count": 0,
        "other_rewrite_injected_z_count": 0,
        "length_rescued_candidate_count": 0,
    }
    composite_pairs = complete_cov_composite_pairs(state)
    stats["composite_source_target_pair_count"] = len(composite_pairs)
    if not composite_pairs:
        return [], stats

    pairs_by_source: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for source_index, target, n in composite_pairs:
        pairs_by_source[source_index].append((target, n))

    source_automorphic_key = equivalence_key(state)
    records = []
    seen_candidate_keys = set()

    for source_index, target_records in pairs_by_source.items():
        relator = state[source_index]
        max_subword_len = (len(relator) - 1) // 2
        if max_subword_len < min_subword_len:
            continue

        for rotation_index, rotation in enumerate(namespace["cyclic_rotations"](relator)):
            for subword_len in range(min_subword_len, max_subword_len + 1):
                subword = rotation[:subword_len]
                stats["cyclic_subword_count"] += 1
                match_list = namespace["find_cyclic_occurrences"](rotation, subword)

                for target, n in target_records:
                    target_in_subword = _target_count(subword, target)
                    if (
                        target_in_subword < 2
                        or target_in_subword > n // 2
                        or n % target_in_subword != 0
                    ):
                        continue
                    required_replacements = n // target_in_subword
                    if required_replacements < 2:
                        continue
                    if required_replacements * subword_len > len(rotation) - 1:
                        continue
                    stats["target_count_filter_pass_count"] += 1

                    candidate_key = (
                        source_index,
                        rotation_index,
                        subword,
                        target,
                        required_replacements,
                    )
                    if candidate_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(candidate_key)
                    if required_replacements > len(match_list):
                        continue

                    for match_positions in namespace[
                        "generate_nonoverlapping_occurrence_combinations"
                    ](
                        match_list,
                        required_replacements,
                        subword_len,
                        len(rotation),
                    ):
                        stats["occurrence_combination_count"] += 1
                        temporary_relator = namespace["apply_cyclic_replacements"](
                            rotation,
                            match_positions,
                            subword_len,
                            namespace["COV_INTRODUCED"],
                        )
                        if _target_count(temporary_relator, target) != 1:
                            continue
                        replacement = namespace["solve_for_single_target"](
                            temporary_relator, target
                        )
                        if replacement is None:
                            continue

                        candidate = {
                            "source_index": source_index,
                            "rotation_index": rotation_index,
                            "rotated_source_relator": rotation,
                            "subword": subword,
                            "target": target,
                            "target_total": n + 1,
                            "target_in_subword": target_in_subword,
                            "required_replacements": required_replacements,
                            "match_positions": tuple(match_positions),
                            "temporary_relator": temporary_relator,
                            "defining_relator": namespace["make_defining_relator"](
                                subword
                            ),
                            "replacement": replacement,
                        }
                        other_relator = state[1 - source_index]
                        unreduced_next_state = _build_complete_cov_neighbor(
                            namespace,
                            state,
                            candidate,
                            other_relator,
                        )
                        unreduced_length = (
                            None
                            if unreduced_next_state is None
                            else total_length(unreduced_next_state)
                        )

                        rewrite_patterns = _maximum_other_relator_rewrite_patterns(
                            other_relator,
                            subword,
                        )
                        stats["other_rewrite_pattern_count"] += len(
                            rewrite_patterns
                        )
                        for rewrite_pattern in rewrite_patterns:
                            rewritten_other = _apply_other_relator_rewrite_pattern(
                                other_relator,
                                subword_len,
                                rewrite_pattern,
                            )
                            stats["other_rewrite_nonempty_pattern_count"] += int(
                                bool(rewrite_pattern)
                            )
                            stats["other_rewrite_injected_z_count"] += len(
                                rewrite_pattern
                            )
                            candidate_variant = dict(candidate)
                            candidate_variant.update(
                                {
                                    "other_relator_before_rewrite": other_relator,
                                    "other_relator_after_rewrite": rewritten_other,
                                    "other_rewrite_occurrences": tuple(
                                        rewrite_pattern
                                    ),
                                    "other_rewrite_count": len(rewrite_pattern),
                                    "other_rewrite_maximized": True,
                                }
                            )
                            next_state = _build_complete_cov_neighbor(
                                namespace,
                                state,
                                candidate_variant,
                                rewritten_other,
                            )
                            if next_state is None:
                                continue
                            if total_length(next_state) >= max_len:
                                stats["length_filtered_count"] += 1
                                continue
                            if (
                                unreduced_length is not None
                                and unreduced_length >= max_len
                            ):
                                stats["length_rescued_candidate_count"] += 1
                            dedupe_key = equivalence_key(next_state)
                            if dedupe_key == source_automorphic_key:
                                stats["source_automorphic_filtered_count"] += 1
                                continue

                            stats["valid_candidate_count"] += 1
                            records.append(
                                {
                                    "next_state": next_state,
                                    "dedupe_next_state": dedupe_key,
                                    "candidate": candidate_variant,
                                }
                            )

    grouped: Dict[WordState, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dedupe_next_state"]].append(record)

    representatives = []
    for group in grouped.values():
        representative = min(
            group,
            key=lambda record: (
                total_length(record["next_state"]),
                _solver_state_key(record["next_state"]),
                record["candidate"]["source_index"],
                record["candidate"]["rotation_index"],
                record["candidate"]["subword"],
                record["candidate"]["target"],
                record["candidate"]["match_positions"],
            ),
        ).copy()
        representative["dedupe_group_size"] = len(group)
        representatives.append(representative)
        stats["duplicate_candidate_count"] += len(group) - 1

    representatives.sort(
        key=lambda record: (
            total_length(record["next_state"]),
            _solver_state_key(record["next_state"]),
        )
    )
    stats["unique_neighbor_count"] = len(representatives)
    return representatives, stats


def _path_to(
    parents: Dict[WordState, Optional[WordState]], final_state: WordState
) -> List[WordState]:
    path = []
    current: Optional[WordState] = final_state
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def run_triple_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    depth_tie_strategy: str = "deepest",
    auto_cov_preference_threshold: int = 2,
    complete_cov_preference_threshold: int = 2,
    seed_initial_auto_cov_neighbors: bool = True,
    seed_initial_complete_cov_neighbors: bool = False,
    namespace: Optional[Dict[str, Any]] = None,
    use_native_standard_successors: bool = True,
    print_progress: bool = False,
    progress_interval: int = 10_000,
) -> Dict[str, Any]:
    """Run the three-queue Auto COV plus filtered Complete COV search."""

    if total_node_budget <= 0:
        raise ValueError("total_node_budget must be positive.")
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if auto_cov_preference_threshold < 0:
        raise ValueError("auto_cov_preference_threshold must be nonnegative.")
    if complete_cov_preference_threshold < 0:
        raise ValueError("complete_cov_preference_threshold must be nonnegative.")
    _depth_priority(0, depth_tie_strategy)

    namespace = namespace or load_complete_cov_namespace()
    start_state = _canonical_start(namespace, presentation)
    insertion_counter = count()
    sub_queue: List[QueueEntry] = []
    auto_queue: List[QueueEntry] = []
    complete_queue: List[QueueEntry] = []
    _push(sub_queue, start_state, 0, insertion_counter, depth_tie_strategy)

    seen_states = {start_state}
    parents: Dict[WordState, Optional[WordState]] = {start_state: None}
    move_to_state: Dict[WordState, Dict[str, Any]] = {}
    depths = {start_state: 0}
    sub_expanded: set[WordState] = set()
    auto_queued: set[WordState] = set()
    auto_expanded: set[WordState] = set()
    complete_queued: set[WordState] = set()
    complete_expanded: set[WordState] = set()

    minimum_length = total_length(start_state)
    minimum_states = {start_state}
    best_state = start_state

    def record_state(state: WordState) -> None:
        nonlocal minimum_length, minimum_states, best_state
        length = total_length(state)
        if length < minimum_length:
            minimum_length = length
            minimum_states = {state}
        elif length == minimum_length:
            minimum_states.add(state)
        if (
            length,
            depths[state],
            _solver_state_key(state),
        ) < (
            total_length(best_state),
            depths[best_state],
            _solver_state_key(best_state),
        ):
            best_state = state

    def add_new_state(
        state: WordState,
        parent: WordState,
        move: Dict[str, Any],
        depth: int,
        *,
        enqueue: bool,
    ) -> bool:
        if state in seen_states:
            return False
        seen_states.add(state)
        parents[state] = parent
        move_to_state[state] = move
        depths[state] = depth
        record_state(state)
        if enqueue:
            _push(sub_queue, state, depth, insertion_counter, depth_tie_strategy)
        return True

    nodes_used = 0
    sub_pop_count = 0
    auto_pop_count = 0
    complete_pop_count = 0
    complete_ineligible_pop_count = 0
    complete_prefilter_accepted_count = 0
    complete_prefilter_rejected_count = 0
    auto_generated_count = 0
    auto_enqueued_count = 0
    standard_generated_count = 0
    standard_enqueued_count = 0
    complete_generation_call_count = 0
    complete_neighbor_count = 0
    complete_seen_neighbor_count = 0
    complete_direct_enqueued_count = 0
    candidate_stats: Dict[str, int] = defaultdict(int)
    peak_frontier = 1
    initial_auto_seed_enqueued_count = 0
    initial_complete_seed_enqueued_count = 0

    def unique_auto_successors(
        source: WordState, source_queue: str
    ) -> List[Tuple[WordState, Dict[str, Any]]]:
        nonlocal auto_generated_count
        batch = set()
        unique = []
        for next_state, raw_move in auto_cov_successors(namespace, source, max_len):
            auto_generated_count += 1
            if next_state in batch:
                continue
            batch.add(next_state)
            move = dict(raw_move)
            move.update(
                {
                    "from_state": source,
                    "to_state": next_state,
                    "source_queue": source_queue,
                }
            )
            unique.append((next_state, move))
        return unique

    if seed_initial_auto_cov_neighbors:
        auto_queued.add(start_state)
        auto_expanded.add(start_state)
        for next_state, move in unique_auto_successors(
            start_state, "initial_auto_cov_seed"
        ):
            if add_new_state(next_state, start_state, move, 1, enqueue=True):
                initial_auto_seed_enqueued_count += 1
                auto_enqueued_count += 1

    if seed_initial_complete_cov_neighbors:
        complete_queued.add(start_state)
        complete_expanded.add(start_state)
        complete_generation_call_count += 1
        records, call_stats = generate_non_automorphic_complete_cov_moves(
            namespace,
            start_state,
            max_len=max_len,
        )
        for key, value in call_stats.items():
            candidate_stats[key] += value
        complete_neighbor_count += len(records)
        novel_records = [
            record for record in records if record["next_state"] not in seen_states
        ]
        complete_seen_neighbor_count += len(records) - len(novel_records)
        for record in novel_records:
            root = record["next_state"]
            complete_move = {
                "move_type": "complete_cov",
                "from_state": start_state,
                "to_state": root,
                "source_queue": "initial_complete_cov_seed",
                "candidate": record["candidate"],
                "dedupe_group_size": record["dedupe_group_size"],
            }
            if add_new_state(
                root,
                start_state,
                complete_move,
                1,
                enqueue=True,
            ):
                initial_complete_seed_enqueued_count += 1
                complete_direct_enqueued_count += 1

    def discard_stale() -> None:
        while sub_queue and sub_queue[0][1] in sub_expanded:
            heapq.heappop(sub_queue)
        while auto_queue and auto_queue[0][1] in auto_expanded:
            heapq.heappop(auto_queue)
        while complete_queue and complete_queue[0][1] in complete_expanded:
            heapq.heappop(complete_queue)

    def choose_queue() -> Optional[str]:
        choices = []
        if sub_queue:
            choices.append((sub_queue[0][0][0], 0, "sub"))
        if auto_queue:
            choices.append(
                (
                    auto_queue[0][0][0] + auto_cov_preference_threshold,
                    1,
                    "auto",
                )
            )
        if complete_queue:
            choices.append(
                (
                    complete_queue[0][0][0]
                    + complete_cov_preference_threshold,
                    2,
                    "complete",
                )
            )
        return min(choices)[2] if choices else None

    solved = False
    termination_reason = "queues_exhausted"

    while nodes_used < total_node_budget:
        discard_stale()
        selected_queue = choose_queue()
        if selected_queue is None:
            break

        if selected_queue == "sub":
            _, current, depth = heapq.heappop(sub_queue)
            if current in sub_expanded:
                continue
            sub_expanded.add(current)
            nodes_used += 1
            sub_pop_count += 1
            record_state(current)
            if _is_trivial_solution(current):
                solved = True
                best_state = current
                termination_reason = "solved"
                break

            for successor in _standard_successors(
                namespace,
                current,
                max_len,
                use_native_standard_successors,
            ):
                standard_generated_count += 1
                move = {
                    "move_type": "standard_substitution",
                    "from_state": current,
                    "to_state": successor,
                    "source_queue": "sub",
                }
                if add_new_state(
                    successor, current, move, depth + 1, enqueue=True
                ):
                    standard_enqueued_count += 1

            if current not in auto_queued:
                auto_queued.add(current)
                _push(
                    auto_queue,
                    current,
                    depth,
                    insertion_counter,
                    depth_tie_strategy,
                )

            if current not in complete_queued:
                if complete_cov_composite_pairs(current):
                    complete_queued.add(current)
                    _push(
                        complete_queue,
                        current,
                        depth,
                        insertion_counter,
                        depth_tie_strategy,
                    )
                    complete_prefilter_accepted_count += 1
                else:
                    complete_prefilter_rejected_count += 1

        elif selected_queue == "auto":
            _, current, depth = heapq.heappop(auto_queue)
            if current in auto_expanded:
                continue
            auto_expanded.add(current)
            nodes_used += 1
            auto_pop_count += 1
            record_state(current)

            for next_state, move in unique_auto_successors(
                current, "auto_cov_activation"
            ):
                if add_new_state(
                    next_state, current, move, depth + 1, enqueue=True
                ):
                    auto_enqueued_count += 1

        else:
            _, current, depth = heapq.heappop(complete_queue)
            if current in complete_expanded:
                continue
            complete_expanded.add(current)
            complete_generation_call_count += 1
            records, call_stats = generate_non_automorphic_complete_cov_moves(
                namespace,
                current,
                max_len=max_len,
            )
            for key, value in call_stats.items():
                candidate_stats[key] += value
            if not records:
                complete_ineligible_pop_count += 1
                continue

            nodes_used += 1
            complete_pop_count += 1
            record_state(current)
            complete_neighbor_count += len(records)
            novel_records = [
                record for record in records if record["next_state"] not in seen_states
            ]
            complete_seen_neighbor_count += len(records) - len(novel_records)

            for record in novel_records:
                if nodes_used >= total_node_budget:
                    termination_reason = "node_budget_exhausted"
                    break
                root = record["next_state"]
                candidate = record["candidate"]
                complete_move = {
                    "move_type": "complete_cov",
                    "from_state": current,
                    "to_state": root,
                    "source_queue": "complete_cov_activation",
                    "candidate": candidate,
                    "dedupe_group_size": record["dedupe_group_size"],
                }
                if add_new_state(
                    root,
                    current,
                    complete_move,
                    depth + 1,
                    enqueue=True,
                ):
                    complete_direct_enqueued_count += 1

            if nodes_used >= total_node_budget:
                break

        peak_frontier = max(
            peak_frontier,
            len(sub_queue) + len(auto_queue) + len(complete_queue),
        )
        if (
            print_progress
            and progress_interval > 0
            and nodes_used > 0
            and nodes_used % progress_interval == 0
        ):
            print(
                f"nodes={nodes_used} sub={len(sub_queue)} auto={len(auto_queue)} "
                f"complete={len(complete_queue)} seen={len(seen_states)} "
                f"best={minimum_length} complete_pops={complete_pop_count}",
                flush=True,
            )

    if not solved and nodes_used >= total_node_budget:
        termination_reason = "node_budget_exhausted"

    path = _path_to(parents, best_state)
    moves = [move_to_state[state] for state in path[1:]]
    return {
        "algorithm_name": "triple_gs_auto_cov_complete_cov",
        "queue_policy": "sub_auto_complete_adjusted_length",
        "queue_tie_order": "sub_auto_complete",
        "presentation": presentation,
        "canonical_start": start_state,
        "solved": solved,
        "solved_trivial": solved,
        "termination_reason": termination_reason,
        "total_nodes_used": nodes_used,
        "sub_pop_count": sub_pop_count,
        "auto_cov_queue_pop_count": auto_pop_count,
        "complete_cov_queue_pop_count": complete_pop_count,
        "complete_cov_ineligible_pop_count": complete_ineligible_pop_count,
        "path": path,
        "moves": moves,
        "path_length": len(moves),
        "final_state": best_state,
        "final_total_length": total_length(best_state),
        "final_depth": depths[best_state],
        "minimum_total_length": minimum_length,
        "minimum_length_states": sorted(minimum_states, key=_solver_state_key),
        "minimum_state_count": len(minimum_states),
        "seen_state_count": len(seen_states),
        "sub_queue_size": len(sub_queue),
        "auto_cov_queue_size": len(auto_queue),
        "complete_cov_queue_size": len(complete_queue),
        "frontier_peak": peak_frontier,
        "depth_tie_strategy": depth_tie_strategy,
        "auto_cov_preference_threshold": auto_cov_preference_threshold,
        "auto_cov_required_length_advantage": (
            auto_cov_preference_threshold + 1
        ),
        "complete_cov_preference_threshold": complete_cov_preference_threshold,
        "complete_cov_required_length_advantage": (
            complete_cov_preference_threshold + 1
        ),
        "seed_initial_auto_cov_neighbors": seed_initial_auto_cov_neighbors,
        "initial_auto_cov_seed_enqueued_count": initial_auto_seed_enqueued_count,
        "seed_initial_complete_cov_neighbors": (
            seed_initial_complete_cov_neighbors
        ),
        "initial_complete_cov_seed_enqueued_count": (
            initial_complete_seed_enqueued_count
        ),
        "auto_cov_move_count": len(AUTO_COV_MAPS),
        "auto_cov_generated_count": auto_generated_count,
        "auto_cov_enqueued_count": auto_enqueued_count,
        "auto_cov_moves_in_path": sum(
            move.get("move_type") == "auto_cov" for move in moves
        ),
        "complete_cov_prefilter_accepted_count": (
            complete_prefilter_accepted_count
        ),
        "complete_cov_prefilter_rejected_count": (
            complete_prefilter_rejected_count
        ),
        "complete_cov_generation_call_count": complete_generation_call_count,
        "complete_cov_unique_neighbor_count": complete_neighbor_count,
        "complete_cov_seen_neighbor_count": complete_seen_neighbor_count,
        "complete_cov_moves_in_path": sum(
            move.get("move_type") == "complete_cov" for move in moves
        ),
        "complete_cov_direct_enqueued_count": complete_direct_enqueued_count,
        "standard_generated_count": standard_generated_count,
        "standard_enqueued_count": standard_enqueued_count,
        "complete_cov_candidate_stats": dict(candidate_stats),
        "max_len": max_len,
        "storage_backend": "python_queues_with_native_standard_successors",
    }


def compact_triple_native_available() -> bool:
    """Return whether the native extension includes the Triple GS engine."""

    return _compact_triple_native is not None and hasattr(
        _compact_triple_native, "run_compact_triple_gs"
    )


def run_compact_triple_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    depth_tie_strategy: str = "deepest",
    auto_cov_preference_threshold: int = 2,
    complete_cov_preference_threshold: int = 2,
    seed_initial_auto_cov_neighbors: bool = True,
    seed_initial_complete_cov_neighbors: bool = False,
    min_standard_moves_between_auto_cov: int = 0,
    max_live_frontier_states: Optional[int] = 5_000_000,
    namespace: Optional[Dict[str, Any]] = None,
    print_progress: bool = False,
    progress_interval: int = 10_000,
    active_checkpoint_path: Optional[str | Path] = None,
    checkpoint_interval_seconds: float = 0.0,
    resume_from_active_checkpoint: bool = True,
    checkpoint_interval_nodes: int = 0,
    progress_heartbeat_paths: Optional[Sequence[str | Path]] = None,
    progress_heartbeat_interval_seconds: float = 0.0,
    progress_heartbeat_interval_nodes: int = 0,
    max_resident_memory_bytes: int = 0,
    resource_check_interval_nodes: int = 10_000,
    sub_queue_score_mode: str = "length",
    cov_activation_score_mode: str = "total_length",
    _stop_after_active_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Run exact Triple GS scheduling with compact C++ queues and storage."""

    if not compact_triple_native_available():
        raise RuntimeError(
            "The compact Triple GS extension is not built. Run "
            "`python -m pip install -e .` from the repository root first."
        )
    if total_node_budget <= 0:
        raise ValueError("total_node_budget must be positive.")
    if max_len <= 0 or max_len > 65:
        raise ValueError("The compact backend requires 1 <= max_len <= 65.")
    if auto_cov_preference_threshold < 0:
        raise ValueError("auto_cov_preference_threshold must be nonnegative.")
    if complete_cov_preference_threshold < 0:
        raise ValueError("complete_cov_preference_threshold must be nonnegative.")
    if min_standard_moves_between_auto_cov < 0:
        raise ValueError("min_standard_moves_between_auto_cov must be nonnegative.")
    if checkpoint_interval_seconds < 0:
        raise ValueError("checkpoint_interval_seconds must be nonnegative.")
    if checkpoint_interval_nodes < 0:
        raise ValueError("checkpoint_interval_nodes must be nonnegative.")
    if progress_heartbeat_interval_seconds < 0:
        raise ValueError(
            "progress_heartbeat_interval_seconds must be nonnegative."
        )
    if progress_heartbeat_interval_nodes < 0:
        raise ValueError("progress_heartbeat_interval_nodes must be nonnegative.")
    if max_resident_memory_bytes < 0:
        raise ValueError("max_resident_memory_bytes must be nonnegative.")
    if resource_check_interval_nodes <= 0:
        raise ValueError("resource_check_interval_nodes must be positive.")
    if sub_queue_score_mode not in {
        "length",
        "length_then_matrix",
        "length_plus_matrix",
    }:
        raise ValueError(
            "sub_queue_score_mode must be length, length_then_matrix, or "
            "length_plus_matrix."
        )
    if cov_activation_score_mode not in {"total_length", "queue_score"}:
        raise ValueError(
            "cov_activation_score_mode must be total_length or queue_score."
        )
    if (
        cov_activation_score_mode == "queue_score"
        and _compact_triple_queue_native is None
    ):
        raise RuntimeError(
            "The queue-score activation extension is not built. Run "
            "Dual GS/speed_optimization/native/build_triple_queue_activation.cmd."
        )
    if checkpoint_interval_seconds > 0 and active_checkpoint_path is None:
        raise ValueError(
            "active_checkpoint_path is required when heartbeat checkpointing is enabled."
        )
    _depth_priority(0, depth_tie_strategy)

    namespace = namespace or load_complete_cov_namespace()
    start_state = _canonical_start(namespace, presentation)
    candidate_stats: Dict[str, int] = defaultdict(int)
    checkpoint_path = (
        Path(active_checkpoint_path).expanduser().resolve()
        if active_checkpoint_path is not None
        else None
    )
    heartbeat_path = (
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".heartbeat.json")
        if checkpoint_path is not None
        else None
    )
    progress_paths = tuple(
        dict.fromkeys(
            Path(path).expanduser().resolve()
            for path in (progress_heartbeat_paths or ())
        )
    )
    if (
        checkpoint_path is not None
        and resume_from_active_checkpoint
        and checkpoint_path.exists()
    ):
        metadata = _compact_triple_native.read_triple_active_checkpoint_metadata(
            str(checkpoint_path)
        )
        if isinstance(metadata, dict):
            checkpoint_score_mode = metadata.get(
                "sub_queue_score_mode", "length"
            )
            if checkpoint_score_mode != sub_queue_score_mode:
                raise ValueError(
                    "Active checkpoint sub_queue_score_mode does not match "
                    "the requested Triple GS configuration."
                )
            checkpoint_activation_mode = metadata.get(
                "cov_activation_score_mode", "total_length"
            )
            if checkpoint_activation_mode != cov_activation_score_mode:
                raise ValueError(
                    "Active checkpoint cov_activation_score_mode does not "
                    "match the requested Triple GS configuration."
                )
            for key, value in metadata.get(
                "complete_cov_candidate_stats", {}
            ).items():
                candidate_stats[key] += int(value)

    def checkpoint_metadata() -> Dict[str, Any]:
        return {
            "complete_cov_candidate_stats": dict(candidate_stats),
            "sub_queue_score_mode": sub_queue_score_mode,
            "cov_activation_score_mode": cov_activation_score_mode,
        }

    def checkpoint_heartbeat(payload: Dict[str, Any]) -> None:
        if heartbeat_path is None:
            return
        heartbeat = dict(payload)
        heartbeat.update(
            {
                "status": "active",
                "presentation": list(start_state),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete_cov_candidate_stats": dict(candidate_stats),
            }
        )
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = heartbeat_path.with_suffix(heartbeat_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(heartbeat, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, heartbeat_path)
        if print_progress:
            print(
                "heartbeat checkpoint: "
                f"nodes={heartbeat['nodes_used']:,} "
                f"seen={heartbeat['seen_state_count']:,} "
                f"bytes={heartbeat['checkpoint_bytes']:,}",
                flush=True,
            )

    def progress_heartbeat(payload: Dict[str, Any]) -> None:
        heartbeat = dict(payload)
        heartbeat.update(
            {
                "status": payload.get("event_type", "active"),
                "presentation": list(start_state),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete_cov_candidate_stats": dict(candidate_stats),
            }
        )
        for target in progress_paths:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(heartbeat, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                print(
                    f"warning: progress heartbeat write failed for {target}: {exc}",
                    flush=True,
                )
        if print_progress:
            resident_gib = heartbeat.get("resident_memory_bytes", 0) / 1024**3
            print(
                "progress heartbeat: "
                f"status={heartbeat['status']} "
                f"nodes={heartbeat['nodes_used']:,} "
                f"seen={heartbeat['seen_state_count']:,} "
                f"rss={resident_gib:.2f}GiB",
                flush=True,
            )

    native_progress_callback = (
        progress_heartbeat
        if (
            progress_paths
            or progress_heartbeat_interval_seconds > 0
            or progress_heartbeat_interval_nodes > 0
            or max_resident_memory_bytes > 0
        )
        else None
    )

    def automorphism_expander(
        state: WordState, source_queue: str
    ) -> Tuple[List[Tuple[WordState, Dict[str, Any]]], int, int]:
        generated_count = 0
        batch_states: set[WordState] = set()
        neighbors = []
        for next_state, raw_move in auto_cov_successors(namespace, state, max_len):
            generated_count += 1
            if next_state in batch_states:
                continue
            batch_states.add(next_state)
            move = dict(raw_move)
            move.update(
                {
                    "from_state": state,
                    "to_state": next_state,
                    "source_queue": source_queue,
                }
            )
            neighbors.append((next_state, move))
        return neighbors, generated_count, len(neighbors)

    def complete_cov_expander(
        state: WordState, source_queue: str
    ) -> Tuple[List[Tuple[WordState, Dict[str, Any]]], int, int]:
        records, call_stats = generate_non_automorphic_complete_cov_moves(
            namespace,
            state,
            max_len=max_len,
            automorphic_key=lambda candidate_state: tuple(
                _compact_triple_native.automorphic_equivalence_key(
                    candidate_state[0], candidate_state[1]
                )
            ),
        )
        for key, value in call_stats.items():
            candidate_stats[key] += value
        neighbors = []
        for record in records:
            next_state = record["next_state"]
            move = {
                "move_type": "complete_cov",
                "from_state": state,
                "to_state": next_state,
                "source_queue": source_queue,
                "candidate": record["candidate"],
                "dedupe_group_size": record["dedupe_group_size"],
            }
            neighbors.append((next_state, move))
        return neighbors, len(records), len(records)

    native_engine = (
        _compact_triple_queue_native
        if cov_activation_score_mode == "queue_score"
        else _compact_triple_native
    )
    native_kwargs = {
        "total_node_budget": total_node_budget,
        "max_total_length": max_len,
        "auto_preference_threshold": auto_cov_preference_threshold,
        "complete_preference_threshold": complete_cov_preference_threshold,
        "seed_initial_auto_neighbors": seed_initial_auto_cov_neighbors,
        "seed_initial_complete_neighbors": seed_initial_complete_cov_neighbors,
        "min_standard_moves_between_auto": min_standard_moves_between_auto_cov,
        "depth_tie_strategy": depth_tie_strategy,
        "max_live_frontier_states": (
            -1 if max_live_frontier_states is None else max_live_frontier_states
        ),
        "progress_interval": progress_interval if print_progress else 0,
        "active_checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_interval_seconds": checkpoint_interval_seconds,
        "resume_active_checkpoint": resume_from_active_checkpoint,
        "checkpoint_interval_nodes": checkpoint_interval_nodes,
        "stop_after_checkpoint": _stop_after_active_checkpoint,
        "checkpoint_metadata_provider": checkpoint_metadata,
        "checkpoint_callback": checkpoint_heartbeat,
        "progress_heartbeat_interval_seconds": progress_heartbeat_interval_seconds,
        "progress_heartbeat_interval_nodes": progress_heartbeat_interval_nodes,
        "progress_heartbeat_callback": native_progress_callback,
        "max_resident_memory_bytes": max_resident_memory_bytes,
        "resource_check_interval_nodes": resource_check_interval_nodes,
        "sub_queue_score_mode": sub_queue_score_mode,
    }
    if cov_activation_score_mode == "queue_score":
        native_kwargs["cov_activation_score_mode"] = cov_activation_score_mode

    result = native_engine.run_compact_triple_gs(
        start_state[0],
        start_state[1],
        automorphism_expander,
        complete_cov_expander,
        **native_kwargs,
    )
    result["algorithm_name"] = (
        f"triple_gs_auto_cov_complete_cov_{sub_queue_score_mode}_"
        f"{cov_activation_score_mode}_activation"
    )
    result["presentation"] = presentation
    result["complete_cov_candidate_stats"] = dict(candidate_stats)
    result["max_len"] = max_len
    result["auto_cov_move_count"] = len(AUTO_COV_MAPS)
    result["_active_checkpoint_path"] = (
        "" if checkpoint_path is None else str(checkpoint_path)
    )
    result["_active_heartbeat_path"] = (
        "" if heartbeat_path is None else str(heartbeat_path)
    )
    result["_progress_heartbeat_paths"] = [
        str(path) for path in progress_paths
    ]
    return result


__all__ = [
    "compact_triple_native_available",
    "complete_cov_composite_pairs",
    "generate_non_automorphic_complete_cov_moves",
    "run_compact_triple_gs",
    "run_triple_gs",
]
