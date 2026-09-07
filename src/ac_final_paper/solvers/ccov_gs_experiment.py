"""Complete-COV first move followed by substitution-only greedy search.

This experiment tests whether a single complete-COV macro move creates a state
that regular substitution-only greedy search can solve quickly.  It also runs a
control branch with no first COV move.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import complete_cov


WordState = Tuple[str, str]


EASY_PRESENTATION: WordState = ("XyxYY", "XXYxY")
INVERSE_CHAR = {"x": "X", "X": "x", "y": "Y", "Y": "y"}
SOLVER_CHAR_ORDER = {"Y": 0, "y": 1, "X": 2, "x": 3}


def _inverse_letter(letter: str) -> str:
    return INVERSE_CHAR[letter]


def _letter_automorphisms() -> List[Dict[str, str]]:
    """All signed letter-to-letter automorphisms of the two-generator alphabet."""

    automorphisms = []
    for x_image in ("x", "X", "y", "Y"):
        y_images = ("y", "Y") if x_image in ("x", "X") else ("x", "X")
        for y_image in y_images:
            mapping = {
                "x": x_image,
                "X": _inverse_letter(x_image),
                "y": y_image,
                "Y": _inverse_letter(y_image),
            }
            automorphisms.append(mapping)
    return automorphisms


LETTER_AUTOMORPHISMS = _letter_automorphisms()


def _automorphism_name(mapping: Dict[str, str]) -> str:
    return f"x->{mapping['x']}, y->{mapping['y']}"


def _apply_automorphism_to_word(word: str, mapping: Dict[str, str]) -> str:
    return "".join(mapping[char] for char in word)


def _apply_automorphism_to_state(state: WordState, mapping: Dict[str, str]) -> WordState:
    return (
        _apply_automorphism_to_word(state[0], mapping),
        _apply_automorphism_to_word(state[1], mapping),
    )


def _inverse_word(word: str) -> str:
    return "".join(INVERSE_CHAR[char] for char in reversed(word))


def _solver_word_key(word: str) -> Tuple[int, ...]:
    """Lexicographic word key matching the AC solver order Y < y < X < x."""

    return tuple(SOLVER_CHAR_ORDER[char] for char in word)


def _solver_state_key(state: WordState) -> Tuple[int, Tuple[int, ...], int, Tuple[int, ...]]:
    """Canonical state ordering matching the solver's pair comparison."""

    return (len(state[0]), _solver_word_key(state[0]), len(state[1]), _solver_word_key(state[1]))


def _cyclic_rotations(word: str) -> List[str]:
    if not word:
        return [word]
    return [word[index:] + word[:index] for index in range(len(word))]


def _canonical_word_variants(word: str) -> List[str]:
    """All cyclic rotations of a relator and its inverse."""

    return sorted(
        set(_cyclic_rotations(word) + _cyclic_rotations(_inverse_word(word))),
        key=_solver_word_key,
    )


def _state_variants_after_automorphism(state: WordState, mapping: Dict[str, str]) -> Iterable[WordState]:
    """All pair/inverse/cyclic variants after a signed generator automorphism."""

    transformed = _apply_automorphism_to_state(state, mapping)
    first_variants = _canonical_word_variants(transformed[0])
    second_variants = _canonical_word_variants(transformed[1])

    for first in first_variants:
        for second in second_variants:
            if (len(first), _solver_word_key(first)) <= (len(second), _solver_word_key(second)):
                yield first, second
            else:
                yield second, first


def _letter_counts(state: WordState) -> Dict[str, int]:
    combined = "".join(state)
    return {letter: combined.count(letter) for letter in ("x", "X", "y", "Y")}


def automorphic_canonical_state(namespace: Dict[str, Any], state: WordState) -> Tuple[WordState, str, Dict[str, int]]:
    """Return the globally lowest representative of the automorphic state orbit.

    This is the universal canonicalization used for deduplication.  It explores
    signed generator automorphisms, cyclic rotations, relator inversion, and
    swapping the two relators, then takes the smallest already-pair-canonical
    representative using the solver's order Y < y < X < x.  No letter-count
    preference is used.
    """

    candidates = []
    for mapping in LETTER_AUTOMORPHISMS:
        for canonical in _state_variants_after_automorphism(state, mapping):
            record = (canonical, _automorphism_name(mapping), _letter_counts(canonical))
            candidates.append(record)

    canonical, name, counts = min(candidates, key=lambda item: _solver_state_key(item[0]))
    return canonical, name, counts


def automorphic_equivalence_key(namespace: Dict[str, Any], state: WordState) -> WordState:
    """Canonical equivalence key across automorphisms and relator symmetries."""

    canonical, _, _ = automorphic_canonical_state(namespace, state)
    return canonical


def load_complete_cov_namespace() -> Dict[str, Any]:
    """Return the packaged Complete-COV implementation namespace."""

    return vars(complete_cov)


def _state_to_key(namespace: Dict[str, Any], state: Any) -> WordState:
    return namespace["state_to_key"](state)


def _canonical_start(namespace: Dict[str, Any], presentation: WordState) -> WordState:
    return namespace["canonical_pair_str"](*presentation)


def _path_to_keys(namespace: Dict[str, Any], path: Optional[Sequence[Any]]) -> Optional[List[WordState]]:
    if path is None:
        return None
    return [_state_to_key(namespace, state) for state in path]


def _state_path_from_visited(visited: Dict[WordState, Optional[WordState]], state_key: WordState) -> List[WordState]:
    """Reconstruct a key path from the solver's parent map."""

    path: List[WordState] = []
    current: Optional[WordState] = state_key
    while current is not None:
        path.append(current)
        current = visited[current]
    path.reverse()
    return path


def _state_depth_from_visited(
    visited: Dict[WordState, Optional[WordState]],
    state_key: WordState,
    cache: Dict[WordState, int],
) -> int:
    """Return the path depth of a state in the solver's parent map."""

    if state_key in cache:
        return cache[state_key]

    parent = visited[state_key]
    if parent is None:
        cache[state_key] = 0
    else:
        cache[state_key] = _state_depth_from_visited(visited, parent, cache) + 1
    return cache[state_key]


def _candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_index": candidate["source_index"],
        "rotation_index": candidate["rotation_index"],
        "rotated_source_relator": candidate["rotated_source_relator"],
        "subword": candidate["subword"],
        "target": candidate["target"],
        "required_replacements": candidate["required_replacements"],
        "match_positions": tuple(candidate["match_positions"]),
        "temporary_relator": candidate["temporary_relator"],
        "defining_relator": candidate["defining_relator"],
        "replacement": candidate["replacement"],
    }


def generate_first_complete_cov_moves(
    namespace: Dict[str, Any],
    presentation: WordState,
    *,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    automorphic_canonicalize: bool = True,
) -> List[Dict[str, Any]]:
    """Generate legal first complete-COV moves that are not start-equivalent."""

    start_key = _canonical_start(namespace, presentation)
    solver = namespace["COVRelatorSolver"](
        start_key[0],
        start_key[1],
        max_nodes=1,
        max_len=max_len,
        verbose=False,
        stop_early=False,
        cov_min_subword_len=cov_min_subword_len,
    )

    start_automorphic_key = automorphic_equivalence_key(namespace, start_key)
    records: List[Dict[str, Any]] = []
    for candidate in solver._generate_cov_candidates(*start_key):
        canonical_state = namespace["build_cov_neighbor_state"](*start_key, candidate)
        if canonical_state is None:
            continue

        next_key = _state_to_key(namespace, canonical_state)
        next_automorphic_key, start_filter_automorphism_name, start_filter_counts = automorphic_canonical_state(namespace, next_key)
        if next_automorphic_key == start_automorphic_key:
            continue

        if automorphic_canonicalize:
            canonical_key = next_automorphic_key
            automorphism_name = start_filter_automorphism_name
            canonical_counts = start_filter_counts
        else:
            canonical_key = next_key
            automorphism_name = "identity"
            canonical_counts = _letter_counts(next_key)

        if len(next_key[0]) + len(next_key[1]) >= max_len:
            continue

        records.append(
            {
                "first_move_type": "complete_cov",
                "start_state": start_key,
                # This is the actual COV-produced mathematical object.  It is
                # what downstream greedy search should use as the next state.
                "first_next_state": next_key,
                "mathematical_next_state": next_key,
                # This key may be an automorphed presentation.  It is only for
                # deduplication across equivalent branches, not for search.
                "dedupe_next_state": canonical_key,
                "automorphic_key": canonical_key,
                "automorphic_canonical_state": canonical_key,
                "automorphic_canonicalization": automorphism_name,
                "automorphic_letter_counts": canonical_counts,
                "first_next_total_length": len(next_key[0]) + len(next_key[1]),
                "dedupe_next_total_length": len(canonical_key[0]) + len(canonical_key[1]),
                "candidate": _candidate_summary(candidate),
            }
        )

    if not deduplicate_next_states:
        return records

    grouped: Dict[WordState, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dedupe_next_state"]].append(record)

    representatives: List[Dict[str, Any]] = []
    for dedupe_key, group in grouped.items():
        # The representative must be one of the actual COV neighbors, not the
        # automorphic key.  Pick the lowest ordinary canonical presentation in
        # this automorphism class.
        representative = min(
            group,
            key=lambda record: (
                len(record["first_next_state"][0]) + len(record["first_next_state"][1]),
                _solver_state_key(record["first_next_state"]),
                record["candidate"]["source_index"],
                record["candidate"]["rotation_index"],
                record["candidate"]["subword"],
                record["candidate"]["target"],
                record["candidate"]["match_positions"],
            ),
        ).copy()
        representative["dedupe_group_size"] = len(group)
        representative["dedupe_group_states"] = sorted(
            {record["first_next_state"] for record in group},
            key=_solver_state_key,
        )
        representative["dedupe_group_move_count"] = len(group)
        representative["dedupe_next_state"] = dedupe_key
        representative["automorphic_key"] = dedupe_key
        representatives.append(representative)

    representatives.sort(
        key=lambda record: (
            len(record["first_next_state"][0]) + len(record["first_next_state"][1]),
            _solver_state_key(record["first_next_state"]),
            _solver_state_key(record["dedupe_next_state"]),
        )
    )
    return representatives


def print_complete_cov_neighbors(
    presentation: WordState = EASY_PRESENTATION,
    *,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = False,
) -> List[Dict[str, Any]]:
    """Print every legal one-step Complete-COV neighbor and substitution detail."""

    namespace = load_complete_cov_namespace()
    start_key = _canonical_start(namespace, presentation)
    neighbors = generate_first_complete_cov_moves(
        namespace,
        start_key,
        max_len=max_len,
        cov_min_subword_len=cov_min_subword_len,
        deduplicate_next_states=deduplicate_next_states,
    )

    print("Complete-COV neighbors")
    print(f"Input presentation: {presentation}")
    print(f"Canonical start: {start_key}")
    print(f"Max total length cutoff: {max_len}")
    print(f"Deduplicate next states: {deduplicate_next_states}")
    print("Start-equivalent automorphic moves: filtered out")
    print(f"Neighbor count: {len(neighbors)}")

    for index, neighbor in enumerate(neighbors, start=1):
        candidate = neighbor["candidate"]
        print(f"\nNeighbor {index}")
        print(f"  next_state              = {neighbor['first_next_state']}")
        print(f"  next_total_length       = {neighbor['first_next_total_length']}")
        print(f"  source_index            = {candidate['source_index']}")
        print(f"  rotation_index          = {candidate['rotation_index']}")
        print(f"  rotated_source_relator  = {candidate['rotated_source_relator']}")
        print(f"  subword                 = {candidate['subword']}")
        print(f"  target                  = {candidate['target']}")
        print(f"  required_replacements   = {candidate['required_replacements']}")
        print(f"  match_positions         = {candidate['match_positions']}")
        print(f"  temporary_relator       = {candidate['temporary_relator']}")
        print(f"  defining_relator        = {candidate['defining_relator']}")
        print(f"  replacement             = {candidate['replacement']}")

    return neighbors


def print_unique_complete_cov_neighbors(
    presentation: WordState = EASY_PRESENTATION,
    *,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    automorphic_canonicalize: bool = True,
) -> Dict[WordState, List[Dict[str, Any]]]:
    """Print unique Complete-COV neighbors grouped by producing moves.

    By default, uniqueness uses automorphic canonicalization in addition to the
    existing relator/cyclic/inverse canonicalization.
    """

    namespace = load_complete_cov_namespace()
    start_key = _canonical_start(namespace, presentation)
    neighbors = generate_first_complete_cov_moves(
        namespace,
        start_key,
        max_len=max_len,
        cov_min_subword_len=cov_min_subword_len,
        deduplicate_next_states=False,
        automorphic_canonicalize=automorphic_canonicalize,
    )

    grouped: Dict[WordState, List[Dict[str, Any]]] = {}
    for neighbor in neighbors:
        grouped.setdefault(neighbor["dedupe_next_state"], []).append(neighbor)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (len(item[0][0]) + len(item[0][1]), _solver_state_key(item[0]), len(item[1])),
    )

    print("Unique Complete-COV neighbors")
    print(f"Input presentation: {presentation}")
    print(f"Canonical start: {start_key}")
    print(f"Automorphic canonicalization: {automorphic_canonicalize}")
    print("Canonical rule: lowest representative across signed generator automorphisms, rotations, inversions, and relator swap")
    print("Canonical order: solver order Y < y < X < x")
    print("Start-equivalent automorphic moves: filtered out")
    print(f"Raw COV move count: {len(neighbors)}")
    print(f"Unique canonical next states: {len(sorted_groups)}")

    for group_index, (dedupe_key, producing_moves) in enumerate(sorted_groups, start=1):
        representative = min(
            producing_moves,
            key=lambda record: (
                len(record["first_next_state"][0]) + len(record["first_next_state"][1]),
                _solver_state_key(record["first_next_state"]),
                record["candidate"]["source_index"],
                record["candidate"]["rotation_index"],
                record["candidate"]["subword"],
                record["candidate"]["target"],
                record["candidate"]["match_positions"],
            ),
        )
        representative_state = representative["first_next_state"]

        print(f"\nUnique neighbor {group_index}")
        print(f"  automorphic_key = {dedupe_key}")
        print(f"  representative_cov_state = {representative_state}")
        print(f"  representative_total_length = {len(representative_state[0]) + len(representative_state[1])}")
        print(f"  producing_substitutions = {len(producing_moves)}")

        for move_index, neighbor in enumerate(producing_moves, start=1):
            candidate = neighbor["candidate"]
            print(
                "    "
                f"{move_index}. "
                f"src={candidate['source_index']}, "
                f"rot={candidate['rotation_index']}, "
                f"w={candidate['subword']}, "
                f"target={candidate['target']}, "
                f"required={candidate['required_replacements']}, "
                f"matches={candidate['match_positions']}, "
                f"replacement={candidate['replacement']}"
            )
            print(
                "       "
                f"raw_next={neighbor['first_next_state']}, "
                f"auto={neighbor['automorphic_canonicalization']}"
            )
            print(
                "       "
                f"rotated={candidate['rotated_source_relator']}, "
                f"temporary={candidate['temporary_relator']}, "
                f"defining={candidate['defining_relator']}"
            )

    return grouped


def analyze_benchmark_initial_cov_automorphic_pairs(
    csv_path: str | Path,
    *,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    max_examples: int = 10,
) -> List[Dict[str, Any]]:
    """Check a benchmark CSV for automorphic duplicates among first COV neighbors."""

    namespace = load_complete_cov_namespace()
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    summaries: List[Dict[str, Any]] = []
    for row in rows:
        presentation = (row["r1"], row["r2"])
        moves = generate_first_complete_cov_moves(
            namespace,
            presentation,
            max_len=max_len,
            cov_min_subword_len=cov_min_subword_len,
            deduplicate_next_states=False,
            automorphic_canonicalize=False,
        )

        ordinary_groups: Dict[WordState, List[Dict[str, Any]]] = defaultdict(list)
        automorphic_groups: Dict[WordState, List[Dict[str, Any]]] = defaultdict(list)
        for move in moves:
            ordinary_groups[move["first_next_state"]].append(move)
            auto_key = automorphic_equivalence_key(namespace, move["first_next_state"])
            automorphic_groups[auto_key].append(move)

        collapse_groups = []
        for auto_key, group_moves in automorphic_groups.items():
            distinct_states = sorted({move["first_next_state"] for move in group_moves})
            if len(distinct_states) <= 1:
                continue
            collapse_groups.append(
                {
                    "automorphic_key": auto_key,
                    "distinct_state_count": len(distinct_states),
                    "move_count": len(group_moves),
                    "distinct_states": distinct_states,
                    "moves": group_moves,
                }
            )

        collapse_groups.sort(
            key=lambda group: (
                -group["distinct_state_count"],
                -group["move_count"],
                group["automorphic_key"],
            )
        )
        summaries.append(
            {
                "name": row.get("name", ""),
                "source": row.get("source", ""),
                "presentation": presentation,
                "raw_move_count": len(moves),
                "ordinary_unique_count": len(ordinary_groups),
                "automorphic_unique_count": len(automorphic_groups),
                "collapse_group_count": len(collapse_groups),
                "has_automorphic_pairs": bool(collapse_groups),
                "collapse_groups": collapse_groups,
            }
        )

    rows_with_pairs = [summary for summary in summaries if summary["has_automorphic_pairs"]]
    print("Initial Complete-COV automorphic-pair benchmark")
    print(f"CSV: {csv_path}")
    print(f"Rows checked: {len(summaries)}")
    print(f"Rows with automorphic pairs: {len(rows_with_pairs)}")
    print(f"Total raw first-COV moves: {sum(row['raw_move_count'] for row in summaries)}")
    print(f"Total ordinary unique next states: {sum(row['ordinary_unique_count'] for row in summaries)}")
    print(f"Total automorphic unique next states: {sum(row['automorphic_unique_count'] for row in summaries)}")

    print(f"\nFirst {min(max_examples, len(rows_with_pairs))} rows with collapses:")
    for summary in rows_with_pairs[:max_examples]:
        print(
            "\n"
            f"{summary['name']} ({summary['source']}): "
            f"{summary['presentation']}"
        )
        print(
            "  "
            f"raw={summary['raw_move_count']}, "
            f"ordinary_unique={summary['ordinary_unique_count']}, "
            f"automorphic_unique={summary['automorphic_unique_count']}, "
            f"collapse_groups={summary['collapse_group_count']}"
        )
        for group_index, group in enumerate(summary["collapse_groups"][:3], start=1):
            print(
                "  "
                f"group {group_index}: auto_key={group['automorphic_key']}, "
                f"distinct_states={group['distinct_state_count']}, "
                f"moves={group['move_count']}"
            )
            for state in group["distinct_states"][:5]:
                print(f"    {state}")

    return summaries


def solve_with_substitution_only(
    namespace: Dict[str, Any],
    start_state: WordState,
    *,
    max_nodes: int = 1000,
    max_len: int = 50,
    best_seen_tie_strategy: str = "deepest",
    random_tie_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the original substitution-only greedy solver from a given state."""

    solver = namespace["ACRelatorSolver"](
        start_state[0],
        start_state[1],
        max_nodes=max_nodes,
        max_len=max_len,
        verbose=False,
        stop_early=False,
    )
    path, nodes_visited, seen_states, solved_trivial = solver.solve()
    path_keys = _path_to_keys(namespace, path)
    depth_cache: Dict[WordState, int] = {}
    if best_seen_tie_strategy == "deepest":
        best_seen_state = min(
            seen_states,
            key=lambda state: (
                len(state[0]) + len(state[1]),
                -_state_depth_from_visited(solver.visited, state, depth_cache),
                _solver_state_key(state),
            ),
        )
    elif best_seen_tie_strategy == "shallowest":
        best_seen_state = min(
            seen_states,
            key=lambda state: (
                len(state[0]) + len(state[1]),
                _state_depth_from_visited(solver.visited, state, depth_cache),
                _solver_state_key(state),
            ),
        )
    elif best_seen_tie_strategy == "random":
        shortest_length = min(len(state[0]) + len(state[1]) for state in seen_states)
        tied_states = sorted(
            [state for state in seen_states if len(state[0]) + len(state[1]) == shortest_length],
            key=_solver_state_key,
        )
        best_seen_state = random.Random(random_tie_seed).choice(tied_states)
    else:
        raise ValueError("best_seen_tie_strategy must be one of: deepest, shallowest, random.")
    best_seen_path = _state_path_from_visited(solver.visited, best_seen_state)
    return {
        "solved": path_keys is not None,
        "solved_trivial": bool(solved_trivial),
        "nodes_visited": nodes_visited,
        "seen_state_count": len(seen_states),
        "path": path_keys,
        "substitution_move_count": None if path_keys is None else max(0, len(path_keys) - 1),
        "best_seen_state": best_seen_state,
        "best_seen_total_length": len(best_seen_state[0]) + len(best_seen_state[1]),
        "best_seen_path": best_seen_path,
        "best_seen_path_length": max(0, len(best_seen_path) - 1),
    }


def _print_path(label: str, path: Sequence[WordState], first_cov_record: Optional[Dict[str, Any]] = None) -> None:
    print(f"\n{label}")
    print(f"Total moves: {len(path) - 1}")

    if first_cov_record is not None:
        candidate = first_cov_record["candidate"]
        print("First move: complete COV")
        print(
            "  "
            f"source_index={candidate['source_index']}, "
            f"subword={candidate['subword']}, "
            f"target={candidate['target']}, "
            f"required_replacements={candidate['required_replacements']}, "
            f"replacement={candidate['replacement']}"
        )
    else:
        print("First move: none; substitution-only control")

    for index, state in enumerate(path):
        print(f"  State {index}: {state}")


def _cov_move_summary(first_cov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if first_cov is None:
        return {
            "cov_source_index": None,
            "cov_rotation_index": None,
            "cov_subword": None,
            "cov_target": None,
            "cov_required_replacements": None,
            "cov_match_positions": None,
            "cov_replacement": None,
            "cov_temporary_relator": None,
            "cov_defining_relator": None,
            "cov_first_next_state": None,
            "cov_automorphic_key": None,
        }

    candidate = first_cov["candidate"]
    return {
        "cov_source_index": candidate["source_index"],
        "cov_rotation_index": candidate["rotation_index"],
        "cov_subword": candidate["subword"],
        "cov_target": candidate["target"],
        "cov_required_replacements": candidate["required_replacements"],
        "cov_match_positions": candidate["match_positions"],
        "cov_replacement": candidate["replacement"],
        "cov_temporary_relator": candidate["temporary_relator"],
        "cov_defining_relator": candidate["defining_relator"],
        "cov_first_next_state": first_cov["first_next_state"],
        "cov_automorphic_key": first_cov["dedupe_next_state"],
    }


def _display_value(value: Any) -> str:
    return "-" if value is None else str(value)


def _truncate(value: Any, width: int) -> str:
    text = _display_value(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _cov_move_display(row: Dict[str, Any]) -> str:
    if row.get("cov_subword") is None:
        return "-"
    return (
        f"src={row['cov_source_index']} "
        f"rot={row['cov_rotation_index']} "
        f"w={row['cov_subword']} "
        f"target={row['cov_target']} "
        f"req={row['cov_required_replacements']} "
        f"matches={row['cov_match_positions']} "
        f"repl={row['cov_replacement']}"
    )


def print_control_vs_best_cov_table(rows: Sequence[Dict[str, Any]], *, limit: Optional[int] = None) -> None:
    """Print a compact comparison table for normal GS vs best first-COV branch."""

    rows_to_print = list(rows if limit is None else rows[:limit])
    if not rows_to_print:
        print("No comparison rows.")
        return

    headers = ("idx", "label", "presentation", "normal", "best_cov", "delta", "nodes", "cov_move")
    widths = (5, 12, 24, 8, 8, 7, 14, 72)
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))

    for row in rows_to_print:
        nodes = f"{_display_value(row.get('normal_nodes_visited'))}/{_display_value(row.get('best_cov_nodes_visited'))}"
        values = (
            row.get("index"),
            row.get("label"),
            row.get("presentation"),
            row.get("normal_path_length"),
            row.get("best_cov_path_length"),
            row.get("path_length_improvement"),
            nodes,
            _cov_move_display(row),
        )
        print(" | ".join(_truncate(value, width).ljust(width) for value, width in zip(values, widths)))

    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more row(s) not shown")


def print_solved_branch_table(results: Sequence[Dict[str, Any]], *, limit: int = 20) -> None:
    """Print a compact table of solved branches from run_ccov_gs_experiment."""

    solved = [row for row in results if row["solved"]]
    solved.sort(key=lambda row: (row["total_move_count"], row["nodes_visited"], row["first_next_total_length"]))
    if not solved:
        print("No solved branches.")
        return

    headers = ("branch", "moves", "nodes", "first_len", "cov_move")
    widths = (16, 7, 7, 9, 72)
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))

    for row in solved[:limit]:
        candidate = row.get("candidate") or {}
        cov_move = "-"
        if candidate:
            cov_move = (
                f"src={candidate['source_index']} "
                f"rot={candidate['rotation_index']} "
                f"w={candidate['subword']} "
                f"target={candidate['target']} "
                f"req={candidate['required_replacements']} "
                f"matches={candidate['match_positions']} "
                f"repl={candidate['replacement']}"
            )
        values = (
            row["branch"],
            row["total_move_count"],
            row["nodes_visited"],
            row["first_next_total_length"],
            cov_move,
        )
        print(" | ".join(_truncate(value, width).ljust(width) for value, width in zip(values, widths)))

    if len(solved) > limit:
        print(f"... {len(solved) - limit} more solved branch(es) not shown")


def compare_control_vs_best_cov(
    presentations: Sequence[WordState],
    *,
    labels: Optional[Sequence[str]] = None,
    max_nodes: int = 1000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    print_rows: bool = True,
) -> List[Dict[str, Any]]:
    """Compare normal GS against the best one-step Complete-COV + GS branch.

    Each COV branch is: one legal non-start-equivalent Complete-COV move, followed
    by substitution-only greedy search with the supplied node limit.
    """

    namespace = load_complete_cov_namespace()
    labels = labels or [str(index) for index in range(len(presentations))]
    rows: List[Dict[str, Any]] = []

    for index, presentation in enumerate(presentations):
        label = labels[index] if index < len(labels) else str(index)
        start_key = _canonical_start(namespace, presentation)

        control = solve_with_substitution_only(
            namespace,
            start_key,
            max_nodes=max_nodes,
            max_len=max_len,
        )

        first_cov_moves = generate_first_complete_cov_moves(
            namespace,
            start_key,
            max_len=max_len,
            cov_min_subword_len=cov_min_subword_len,
            deduplicate_next_states=deduplicate_next_states,
        )

        cov_results: List[Dict[str, Any]] = []
        for branch_index, first_cov in enumerate(first_cov_moves, start=1):
            solve_result = solve_with_substitution_only(
                namespace,
                first_cov["first_next_state"],
                max_nodes=max_nodes,
                max_len=max_len,
            )
            total_move_count = None
            if solve_result["path"] is not None:
                total_move_count = 1 + solve_result["substitution_move_count"]

            cov_results.append(
                {
                    "branch": f"first_cov_{branch_index}",
                    **first_cov,
                    **solve_result,
                    "total_move_count": total_move_count,
                }
            )

        solved_cov_results = [row for row in cov_results if row["solved"]]
        solved_cov_results.sort(
            key=lambda row: (
                row["total_move_count"],
                row["nodes_visited"],
                row["first_next_total_length"],
                _solver_state_key(row["first_next_state"]),
            )
        )
        best_cov = solved_cov_results[0] if solved_cov_results else None

        control_length = control["substitution_move_count"] if control["solved"] else None
        best_cov_length = best_cov["total_move_count"] if best_cov is not None else None
        improvement = None
        if control_length is not None and best_cov_length is not None:
            improvement = control_length - best_cov_length

        row = {
            "index": index,
            "label": label,
            "presentation": presentation,
            "canonical_start": start_key,
            "normal_solved": control["solved"],
            "normal_path_length": control_length,
            "normal_nodes_visited": control["nodes_visited"],
            "cov_branch_count": len(first_cov_moves),
            "best_cov_solved": best_cov is not None,
            "best_cov_path_length": best_cov_length,
            "best_cov_nodes_visited": None if best_cov is None else best_cov["nodes_visited"],
            "best_cov_branch": None if best_cov is None else best_cov["branch"],
            "path_length_improvement": improvement,
            **_cov_move_summary(best_cov),
        }
        rows.append(row)

    if print_rows:
        print_control_vs_best_cov_table(rows)

    return rows


def compare_control_vs_best_cov_csv(
    csv_path: str | Path,
    *,
    limit: Optional[int] = None,
    max_nodes: int = 1000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    print_rows: bool = True,
) -> List[Dict[str, Any]]:
    """Run the control-vs-best-COV comparison for a benchmark CSV."""

    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))

    if limit is not None:
        csv_rows = csv_rows[:limit]

    presentations = [(row["r1"], row["r2"]) for row in csv_rows]
    labels = [
        row.get("name")
        or row.get("index")
        or row.get("id")
        or str(index)
        for index, row in enumerate(csv_rows)
    ]
    return compare_control_vs_best_cov(
        presentations,
        labels=labels,
        max_nodes=max_nodes,
        max_len=max_len,
        cov_min_subword_len=cov_min_subword_len,
        deduplicate_next_states=deduplicate_next_states,
        print_rows=print_rows,
    )


def run_ccov_gs_experiment(
    presentation: WordState = EASY_PRESENTATION,
    *,
    max_nodes: int = 1000,
    max_len: int = 50,
    cov_min_subword_len: int = 2,
    deduplicate_next_states: bool = True,
    print_unsolved_summary: bool = True,
) -> List[Dict[str, Any]]:
    """Run control GS and every one-step Complete-COV + GS branch."""

    namespace = load_complete_cov_namespace()
    start_key = _canonical_start(namespace, presentation)

    print("CCOV+GS experiment")
    print(f"Input presentation: {presentation}")
    print(f"Canonical start: {start_key}")
    print(f"Substitution-only node limit after first move: {max_nodes}")
    print(f"Max total length cutoff: {max_len}")

    results: List[Dict[str, Any]] = []

    control_solve = solve_with_substitution_only(
        namespace,
        start_key,
        max_nodes=max_nodes,
        max_len=max_len,
    )
    control_result = {
        "branch": "control_no_first_cov",
        "first_move_type": "none",
        "start_state": start_key,
        "first_next_state": start_key,
        "first_next_total_length": len(start_key[0]) + len(start_key[1]),
        **control_solve,
        "total_move_count": control_solve["substitution_move_count"],
    }
    results.append(control_result)

    if control_result["solved"]:
        _print_path("Solved branch: control_no_first_cov", control_result["path"])

    first_cov_moves = generate_first_complete_cov_moves(
        namespace,
        start_key,
        max_len=max_len,
        cov_min_subword_len=cov_min_subword_len,
        deduplicate_next_states=deduplicate_next_states,
    )
    print(f"\nGenerated first complete-COV branches: {len(first_cov_moves)}")

    for branch_index, first_cov in enumerate(first_cov_moves, start=1):
        solve_result = solve_with_substitution_only(
            namespace,
            first_cov["first_next_state"],
            max_nodes=max_nodes,
            max_len=max_len,
        )
        full_path = None
        if solve_result["path"] is not None:
            full_path = [start_key] + solve_result["path"]

        result = {
            "branch": f"first_cov_{branch_index}",
            **first_cov,
            **solve_result,
            "path": full_path,
            "total_move_count": None if full_path is None else len(full_path) - 1,
        }
        results.append(result)

        if result["solved"]:
            _print_path(f"Solved branch: first_cov_{branch_index}", result["path"], first_cov)

    solved_results = [result for result in results if result["solved"]]
    solved_results.sort(key=lambda row: (row["total_move_count"], row["nodes_visited"], row["first_next_total_length"]))

    print("\nSummary")
    print(f"Total branches tested: {len(results)}")
    print(f"Solved branches: {len(solved_results)}")
    if solved_results:
        best = solved_results[0]
        print(
            "Best solved branch: "
            f"{best['branch']} with {best['total_move_count']} move(s), "
            f"{best['nodes_visited']} GS node(s), "
            f"first-next length {best['first_next_total_length']}"
        )
    elif print_unsolved_summary:
        best_unsolved = min(
            results,
            key=lambda row: (row["first_next_total_length"], row["nodes_visited"]),
        )
        print(
            "No solved branches. Shortest first-next branch: "
            f"{best_unsolved['branch']} length {best_unsolved['first_next_total_length']}"
        )

    return results


if __name__ == "__main__":
    run_ccov_gs_experiment()
