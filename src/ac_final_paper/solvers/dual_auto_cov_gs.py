"""Dual GS with standard substitutions and Auto COV activations.

The first queue expands ordinary substitution moves.  The second queue uses
the same length threshold and gap policy as Dual GS, but Auto COV expands only
the 24 canonical Nielsen automorphisms; it never generates a Single-COV
neighbor.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from .compact_dual_gs import _native, compact_native_available
from .dual_single_cov_gs import WordState, run_dual_single_cov_gs
from .standard_single_cov_gs import _canonical_start, load_single_cov_namespace


AutoCOVMap = Tuple[str, Dict[str, str], int]
MAX_RIGHT_POWER = 6


def _inverse_word(word: str) -> str:
    inverse_letters = {"x": "X", "X": "x", "y": "Y", "Y": "y"}
    return "".join(inverse_letters[letter] for letter in reversed(word))


def _signed_power(positive: str, negative: str, exponent: int) -> str:
    return positive * exponent if exponent >= 0 else negative * (-exponent)


def _build_auto_cov_maps() -> Tuple[AutoCOVMap, ...]:
    """Build the canonical right-power representatives of the bounded maps."""

    moves = []
    for generator, other_positive, other_negative in (("x", "y", "Y"), ("y", "x", "X")):
        inverse_generator = _inverse_word(generator)
        for exponent in range(-MAX_RIGHT_POWER, MAX_RIGHT_POWER + 1):
            if exponent == 0:
                continue
            image = generator + _signed_power(other_positive, other_negative, exponent)
            moves.append(
                (
                    f"{generator}_right_{exponent:+d}",
                    {generator: image, inverse_generator: _inverse_word(image)},
                    exponent,
                )
            )
    return tuple(moves)


AUTO_COV_MAPS = _build_auto_cov_maps()
assert len(AUTO_COV_MAPS) == 24


def auto_cov_successors(
    namespace: Dict[str, Any],
    state: WordState,
    max_len: int,
) -> Iterable[Tuple[WordState, Dict[str, Any]]]:
    """Generate the 24 canonical Auto COV successors."""

    for name, replacements, exponent in AUTO_COV_MAPS:
        transformed_r1 = "".join(replacements.get(letter, letter) for letter in state[0])
        transformed_r2 = "".join(replacements.get(letter, letter) for letter in state[1])
        reduced_r1 = namespace["reduce_relator_nj"](namespace["str_to_arr"](transformed_r1))
        reduced_r2 = namespace["reduce_relator_nj"](namespace["str_to_arr"](transformed_r2))
        if len(reduced_r1) + len(reduced_r2) >= max_len:
            continue
        canonical_pair = namespace["canonical_pair_nj"](reduced_r1, reduced_r2)
        next_state = namespace["state_to_key"](canonical_pair)
        yield next_state, {
            "move_type": "auto_cov",
            "automorphism": name,
            "letter_replacements": dict(replacements),
            "right_power": exponent,
            "activation_only": True,
            "cyclic_canonical_representative": True,
        }


def _annotate_auto_cov_result(
    result: Dict[str, Any],
    *,
    seed_initial_auto_cov_neighbors: bool,
    compact_backend: bool,
) -> Dict[str, Any]:
    """Add unambiguous move-family metrics to a shared-engine result."""

    if compact_backend:
        generated = result["cov_candidate_count"]
        valid = result["cov_valid_candidate_count"]
        enqueued = result["cov_enqueued_count"]
        initial_enqueued = result["initial_cov_seed_enqueued_count"]
    else:
        generated = result["additional_cov_generated_count"]
        valid = result["additional_cov_valid_novel_count"]
        enqueued = result["additional_cov_enqueued_count"]
        initial_enqueued = result["initial_additional_cov_seed_enqueued_count"]

    for move in result["moves"]:
        if move.get("move_type") != "auto_cov":
            continue
        if move.get("source_queue") == "initial_cov_seed":
            move["source_queue"] = "initial_auto_cov_seed"
        elif move.get("source_queue") == "cov_activation":
            move["source_queue"] = "auto_cov_activation"

    result.update(
        {
            "algorithm_name": "dual_queue_standard_plus_auto_cov",
            "queue_policy": "dual_sub_then_auto_cov_strict_length_activation",
            "second_queue_move_family": "auto_cov",
            "single_cov_moves_enabled": False,
            "cov_variant": "auto_cov",
            "elementary_automorphisms_enabled": True,
            "elementary_automorphisms_activation_only": True,
            "elementary_automorphism_move_count": len(
                AUTO_COV_MAPS
            ),
            "elementary_automorphism_max_power": MAX_RIGHT_POWER,
            "elementary_automorphism_names": [
                name for name, _, _ in AUTO_COV_MAPS
            ],
            "automorphism_generated_count": generated,
            "automorphism_valid_novel_count": valid,
            "automorphism_enqueued_count": enqueued,
            "automorphism_moves_in_path": sum(
                move.get("move_type") == "auto_cov"
                for move in result["moves"]
            ),
            "auto_cov_generated_count": generated,
            "auto_cov_valid_novel_count": valid,
            "auto_cov_enqueued_count": enqueued,
            "auto_cov_moves_in_path": sum(
                move.get("move_type") == "auto_cov" for move in result["moves"]
            ),
            "auto_cov_queue_pop_count": result["cov_pop_count"],
            "auto_cov_queue_queued_count": result["cov_queued_count"],
            "auto_cov_expansion_count": result["cov_expansion_count"],
            "automorphism_queue_pop_count": result["cov_pop_count"],
            "automorphism_queue_queued_count": result["cov_queued_count"],
            "automorphism_expansion_count": result["cov_expansion_count"],
            "second_queue_pop_count": result["cov_pop_count"],
            "second_queue_queued_count": result["cov_queued_count"],
            "second_queue_size": result["cov_queue_size"],
            "seed_initial_auto_cov_neighbors": seed_initial_auto_cov_neighbors,
            "seed_initial_automorphism_neighbors": seed_initial_auto_cov_neighbors,
            "initial_auto_cov_seed_enqueued_count": initial_enqueued,
            "initial_automorphism_seed_enqueued_count": initial_enqueued,
        }
    )
    # The shared engines retain legacy second-queue field names. Generation
    # fields must nevertheless report truthfully that no Single-COV ran.
    result["cov_candidate_count"] = 0
    result["cov_valid_candidate_count"] = 0
    result["cov_enqueued_count"] = 0
    result["initial_cov_seed_enqueued_count"] = 0
    result["seed_initial_cov_neighbors"] = False
    return result


def run_dual_auto_cov_gs(
    presentation: WordState,
    *,
    auto_cov_preference_threshold: int = 2,
    seed_initial_auto_cov_neighbors: bool = True,
    min_standard_moves_between_auto_cov: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the Python reference engine with no Single-COV generation."""

    forbidden = {
        "additional_cov_moves",
        "enable_single_cov_moves",
        "seed_initial_cov_neighbors",
        "seed_initial_additional_cov_moves",
        "cov_preference_threshold",
        "min_standard_moves_between_cov",
    } & kwargs.keys()
    if forbidden:
        raise ValueError(
            "The Auto COV variant fixes these options: "
            f"{sorted(forbidden)}"
        )

    result = run_dual_single_cov_gs(
        presentation,
        enable_single_cov_moves=False,
        additional_cov_moves=auto_cov_successors,
        seed_initial_cov_neighbors=False,
        seed_initial_additional_cov_moves=seed_initial_auto_cov_neighbors,
        cov_preference_threshold=auto_cov_preference_threshold,
        min_standard_moves_between_cov=min_standard_moves_between_auto_cov,
        **kwargs,
    )
    result["auto_cov_preference_threshold"] = auto_cov_preference_threshold
    result["automorphism_preference_threshold"] = auto_cov_preference_threshold
    result["min_standard_moves_between_auto_cov"] = min_standard_moves_between_auto_cov
    result["min_standard_moves_between_automorphisms"] = (
        min_standard_moves_between_auto_cov
    )
    return _annotate_auto_cov_result(
        result,
        seed_initial_auto_cov_neighbors=seed_initial_auto_cov_neighbors,
        compact_backend=False,
    )


def run_compact_dual_auto_cov_gs(
    presentation: WordState,
    *,
    total_node_budget: int = 1_000_000,
    max_len: int = 50,
    depth_tie_strategy: str = "deepest",
    auto_cov_preference_threshold: int = 2,
    seed_initial_auto_cov_neighbors: bool = True,
    min_standard_moves_between_auto_cov: int = 10,
    sub_queue_score_mode: str = "length",
    max_live_frontier_states: Optional[int] = 5_000_000,
    namespace: Optional[Dict[str, Any]] = None,
    print_progress: bool = False,
    progress_interval: int = 10_000,
) -> Dict[str, Any]:
    """Run Auto COV with compact C++ state storage."""

    if not compact_native_available():
        raise RuntimeError(
            "The compact Dual GS extension is not built. Run "
            "`python -m pip install -e .` from the repository root first."
        )
    if max_len <= 0 or max_len > 65:
        raise ValueError("The compact backend requires 1 <= max_len <= 65.")
    if auto_cov_preference_threshold < 0:
        raise ValueError("auto_cov_preference_threshold must be nonnegative.")
    if min_standard_moves_between_auto_cov < 0:
        raise ValueError(
            "min_standard_moves_between_auto_cov must be nonnegative."
        )
    if sub_queue_score_mode not in {
        "length",
        "length_then_matrix",
        "length_plus_matrix",
    }:
        raise ValueError(
            "sub_queue_score_mode must be length, length_then_matrix, or "
            "length_plus_matrix."
        )

    namespace = namespace or load_single_cov_namespace()
    start_state = _canonical_start(namespace, presentation)

    def automorphism_expander(state: WordState, source_queue: str):
        neighbors = []
        generated_count = 0
        batch_states: set[WordState] = set()
        move_source_queue = (
            "cov_activation" if source_queue == "cov" else source_queue
        )
        for next_state, raw_move in auto_cov_successors(
            namespace, state, max_len
        ):
            generated_count += 1
            if next_state in batch_states:
                continue
            batch_states.add(next_state)
            move = dict(raw_move)
            move["from_state"] = state
            move["to_state"] = next_state
            move["source_queue"] = move_source_queue
            neighbors.append((next_state, move))
        return neighbors, generated_count, len(neighbors)

    result = _native.run_compact_dual_gs(
        start_state[0],
        start_state[1],
        automorphism_expander,
        total_node_budget=total_node_budget,
        max_total_length=max_len,
        cov_preference_threshold=auto_cov_preference_threshold,
        seed_initial_cov_neighbors=seed_initial_auto_cov_neighbors,
        min_standard_moves_between_cov=min_standard_moves_between_auto_cov,
        depth_tie_strategy=depth_tie_strategy,
        max_live_frontier_states=(
            -1 if max_live_frontier_states is None else max_live_frontier_states
        ),
        progress_interval=progress_interval if print_progress else 0,
        sub_queue_score_mode=sub_queue_score_mode,
    )
    result["presentation"] = presentation
    result["auto_cov_preference_threshold"] = auto_cov_preference_threshold
    result["automorphism_preference_threshold"] = auto_cov_preference_threshold
    result["min_standard_moves_between_auto_cov"] = min_standard_moves_between_auto_cov
    result["min_standard_moves_between_automorphisms"] = (
        min_standard_moves_between_auto_cov
    )
    result["sub_queue_score_mode"] = sub_queue_score_mode
    return _annotate_auto_cov_result(
        result,
        seed_initial_auto_cov_neighbors=seed_initial_auto_cov_neighbors,
        compact_backend=True,
    )


__all__ = [
    "AUTO_COV_MAPS",
    "MAX_RIGHT_POWER",
    "auto_cov_successors",
    "run_dual_auto_cov_gs",
    "run_compact_dual_auto_cov_gs",
]
