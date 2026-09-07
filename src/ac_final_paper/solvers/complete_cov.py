"""Complete-COV move generation built on the packaged AC core."""

from __future__ import annotations

import heapq

from .ac_core import *  # noqa: F403


# -----------------------------------------------------
# Complete change-of-variables extension
# -----------------------------------------------------

_inverse_char = {'x': 'X', 'X': 'x', 'y': 'Y', 'Y': 'y', 'z': 'Z', 'Z': 'z'}
COV_TARGETS = ('x', 'y')
COV_INTRODUCED = 'z'


def inverse_word(word):
    return ''.join(_inverse_char[c] for c in reversed(word))


def free_reduce_word(word):
    pieces = []
    for char in word:
        if pieces and _inverse_char[pieces[-1]] == char:
            pieces.pop()
        else:
            pieces.append(char)
    return ''.join(pieces)


def reduce_word(word):
    word = free_reduce_word(word)
    changed = True
    while changed:
        changed = False

        if len(word) > 1 and _inverse_char[word[0]] == word[-1]:
            word = free_reduce_word(word[1:-1])
            changed = True
    return word


def cyclic_rotations(word):
    return [word[i:] + word[:i] for i in range(len(word))]


def cyclic_subword(word, start, length):
    word_len = len(word)
    return ''.join(word[(start + offset) % word_len] for offset in range(length))


def count_target_occurrences(word, target):
    inverse_target = _inverse_char[target]
    return sum(1 for char in word if char in (target, inverse_target))


def find_cyclic_occurrences(word, subword):
    return [
        start
        for start in range(len(word))
        if cyclic_subword(word, start, len(subword)) == subword
    ]


def cyclic_interval_positions(start, length, word_len):
    return tuple((start + offset) % word_len for offset in range(length))


def choose_cyclic_cut(word_len, starts, subword_len):
    if not starts:
        return 0

    occupied = set()
    for start in starts:
        occupied.update(cyclic_interval_positions(start, subword_len, word_len))

    for cut in range(word_len):
        if cut not in occupied:
            return cut

    # If the chosen occurrences tile the whole cyclic word, cutting at one of the
    # occurrence boundaries still gives a valid linear representative.
    return min(starts)


def rotate_word(word, rotation_index):
    return word[rotation_index:] + word[:rotation_index]


def apply_cyclic_replacements(word, starts, subword_len, replacement):
    if not starts:
        return word

    word_len = len(word)
    cut = choose_cyclic_cut(word_len, starts, subword_len)
    rotated_word = rotate_word(word, cut)
    rotated_starts = sorted((start - cut) % word_len for start in starts)

    pieces = []
    cursor = 0
    for start in rotated_starts:
        pieces.append(rotated_word[cursor:start])
        pieces.append(replacement)
        cursor = start + subword_len
    pieces.append(rotated_word[cursor:])
    return ''.join(pieces)


def generate_nonoverlapping_occurrence_combinations(match_list, required_replacements, subword_len, word_len):
    if required_replacements == 0:
        yield ()
        return

    starts = sorted(dict.fromkeys(match_list))
    interval_cache = {
        start: cyclic_interval_positions(start, subword_len, word_len)
        for start in starts
    }
    occupied = set()

    def backtrack(next_index, chosen):
        needed = required_replacements - len(chosen)
        if needed == 0:
            yield tuple(chosen)
            return

        if len(starts) - next_index < needed:
            return

        for idx in range(next_index, len(starts)):
            start = starts[idx]
            interval = interval_cache[start]
            if any(position in occupied for position in interval):
                continue

            chosen.append(start)
            occupied.update(interval)
            yield from backtrack(idx + 1, chosen)
            chosen.pop()
            occupied.difference_update(interval)

    yield from backtrack(0, [])


def solve_for_single_target(word, target):
    """Solve a relator word for target if target or its inverse occurs exactly once."""
    inverse_target = _inverse_char[target]
    positions = [i for i, char in enumerate(word) if char in (target, inverse_target)]
    if len(positions) != 1:
        return None

    pos = positions[0]
    prefix = word[:pos]
    suffix = word[pos + 1:]
    if word[pos] == target:
        return free_reduce_word(inverse_word(prefix) + inverse_word(suffix))
    return free_reduce_word(suffix + prefix)


def substitute_target(word, target, replacement):
    inverse_target = _inverse_char[target]
    inverse_replacement = inverse_word(replacement)
    pieces = []
    for char in word:
        if char == target:
            pieces.append(replacement)
        elif char == inverse_target:
            pieces.append(inverse_replacement)
        else:
            pieces.append(char)
    return ''.join(pieces)


def make_defining_relator(subword):
    # Even in the complete move, every selected occurrence of w is replaced by the
    # same new generator z, so one defining relation z = w is sufficient.
    return reduce_word(COV_INTRODUCED + inverse_word(subword))


def relabel_cov_word_to_xy(word, eliminated_target):
    survivor = 'y' if eliminated_target == 'x' else 'x'
    mapping = {
        survivor: 'x',
        _inverse_char[survivor]: 'X',
        COV_INTRODUCED: 'y',
        _inverse_char[COV_INTRODUCED]: 'Y',
    }
    if any(char not in mapping for char in word):
        return None
    return ''.join(mapping[char] for char in word)


def canonicalize_state_from_strings(r1, r2):
    # COV moves are built with string manipulation first, then reduced/canonicalized
    # so they re-enter the same state space as the original substitution search.
    r1_reduced = reduce(r1)
    r2_reduced = reduce(r2)
    if not r1_reduced or not r2_reduced:
        return None

    r1_arr = str_to_arr(r1_reduced)
    r2_arr = str_to_arr(r2_reduced)
    canon_r1, canon_r2 = canonical_pair_nj(reduce_relator_nj(r1_arr), reduce_relator_nj(r2_arr))
    return canon_r1, canon_r2


def canonicalize_cov_state_from_strings(r1, r2, eliminated_target):
    # A faithful COV move temporarily uses a third generator z, then removes one old
    # generator and relabels the surviving pair back to x,y before re-entering the search.
    r1_reduced = reduce_word(r1)
    r2_reduced = reduce_word(r2)
    if not r1_reduced or not r2_reduced:
        return None

    relabeled_r1 = relabel_cov_word_to_xy(r1_reduced, eliminated_target)
    relabeled_r2 = relabel_cov_word_to_xy(r2_reduced, eliminated_target)
    if relabeled_r1 is None or relabeled_r2 is None:
        return None
    return canonicalize_state_from_strings(relabeled_r1, relabeled_r2)


def build_cov_neighbor_state(r1, r2, candidate):
    relators = (r1, r2)
    other_relator = relators[1 - candidate['source_index']]
    target = candidate['target']
    replacement = candidate['replacement']

    # The source relator is deleted after we solve it for the old generator.
    # The resulting 2-generator state keeps only the transformed other relator and
    # the transformed defining relation for the introduced z.
    new_other = reduce_word(substitute_target(other_relator, target, replacement))
    new_defining = reduce_word(substitute_target(candidate['defining_relator'], target, replacement))
    if not new_other or not new_defining:
        return None
    return canonicalize_cov_state_from_strings(new_other, new_defining, target)


class COVRelatorSolver(ACRelatorSolver):
    """AC search that scores substitution and complete-COV neighbors in the same priority queue."""

    def __init__(
        self,
        r1,
        r2,
        max_nodes=10000,
        max_len=100,
        visited=None,
        verbose=True,
        stop_early=True,
        cov_min_subword_len=2,
    ):
        super().__init__(r1, r2, max_nodes=max_nodes, max_len=max_len, visited=visited, verbose=verbose, stop_early=stop_early)
        self.cov_min_subword_len = cov_min_subword_len
        self.cov_stats = {
            'cov_candidate_count': 0,
            'cov_valid_candidate_count': 0,
            'cov_enqueued_count': 0,
        }

    def _priority(self, r1, r2):
        return len(r1) + len(r2)

    def _generate_cov_candidates(self, r1, r2):
        # Each cyclic rotation already fixes a subword start position, so we keep the
        # prefix-only traversal from the single-COV solver. The complete-COV upgrade
        # then finds every cyclic occurrence of that subword inside the chosen rotation
        # and enumerates disjoint replacement sets.
        relators = (r1, r2)
        seen_candidates = set()
        for source_index, relator in enumerate(relators):
            max_subword_len = len(relator) - 2
            if max_subword_len < self.cov_min_subword_len:
                continue

            for rotation_index, rotation in enumerate(cyclic_rotations(relator)):
                target_totals = {
                    target: count_target_occurrences(rotation, target)
                    for target in COV_TARGETS
                }
                for subword_len in range(self.cov_min_subword_len, max_subword_len + 1):
                    subword = rotation[:subword_len]
                    defining_relator = make_defining_relator(subword)
                    match_list = find_cyclic_occurrences(rotation, subword)

                    for target in COV_TARGETS:
                        target_total = target_totals[target]
                        target_in_subword = count_target_occurrences(subword, target)

                        if target_total == 1:
                            if target_in_subword != 0:
                                continue
                            required_replacements = 0
                        else:
                            if target_in_subword == 0:
                                continue
                            if (target_total - 1) % target_in_subword != 0:
                                continue
                            required_replacements = (target_total - 1) // target_in_subword

                        candidate_key = (
                            source_index,
                            rotation_index,
                            subword,
                            target,
                            required_replacements,
                        )
                        if candidate_key in seen_candidates:
                            continue
                        seen_candidates.add(candidate_key)
                        self.cov_stats['cov_candidate_count'] += 1

                        if required_replacements > len(match_list):
                            continue

                        for match_positions in generate_nonoverlapping_occurrence_combinations(
                            match_list,
                            required_replacements,
                            subword_len,
                            len(rotation),
                        ):
                            temporary_relator = apply_cyclic_replacements(
                                rotation,
                                match_positions,
                                subword_len,
                                COV_INTRODUCED,
                            )
                            if count_target_occurrences(temporary_relator, target) != 1:
                                continue

                            solved_expression = solve_for_single_target(temporary_relator, target)
                            if solved_expression is None:
                                continue

                            self.cov_stats['cov_valid_candidate_count'] += 1
                            yield {
                                'source_index': source_index,
                                'rotation_index': rotation_index,
                                'rotated_source_relator': rotation,
                                'subword': subword,
                                'target': target,
                                'required_replacements': required_replacements,
                                'match_positions': match_positions,
                                'temporary_relator': temporary_relator,
                                'defining_relator': defining_relator,
                                'replacement': solved_expression,
                            }

    def _generate_cov_neighbors(self, parent_key, parent_depth):
        # A completed COV macro-move removes the solved source relator, keeps the other
        # original relator plus the z-defining relator, and relabels the survivors to x,y.
        r1, r2 = parent_key
        for candidate in self._generate_cov_candidates(r1, r2):
            canonical_state = build_cov_neighbor_state(r1, r2, candidate)
            if canonical_state is None:
                continue

            canon_r1, canon_r2 = canonical_state
            total_len = len(canon_r1) + len(canon_r2)
            if total_len >= self.max_len:
                continue

            key_new = state_to_key((canon_r1, canon_r2))
            if key_new in self.visited:
                continue

            self.visited[key_new] = parent_key
            self.new_seen.add(key_new)
            self.cov_stats['cov_enqueued_count'] += 1
            priority = self._priority(canon_r1, canon_r2)
            heapq.heappush(self.pq, (priority, parent_depth + 1, key_new))

    def solve(self):
        initial_key = state_to_key(self.initial_state)
        initial_priority = self._priority(self.initial_state[0], self.initial_state[1])
        heapq.heappush(self.pq, (initial_priority, 0, initial_key))
        self.visited[initial_key] = None
        self.new_seen = set()
        self.new_seen.add(initial_key)
        nodes_visited = 0

        while self.pq and nodes_visited < self.max_nodes:
            _, depth, key = heapq.heappop(self.pq)
            nodes_visited += 1
            r1, r2 = self._key_to_state(key)

            if self.verbose:
                total_len = len(r1) + len(r2)
                if total_len > self.max_priority:
                    print(f"First state of priority {total_len}, depth: {depth}, values: {len(r1)}, {len(r2)} ({arr_to_str(r1)},{arr_to_str(r2)}), nodes: {nodes_visited}")
                    self.max_priority = total_len

                if total_len < self.min_priority:
                    print(f"First state of priority {total_len}, depth: {depth}, values: {len(r1)}, {len(r2)} ({arr_to_str(r1)},{arr_to_str(r2)}), nodes: {nodes_visited}")
                    self.min_priority = total_len

            if len(r1) == 1 and len(r2) == 1:
                path = []
                state_key = key
                while state_key is not None:
                    path.append(self._key_to_state(state_key))
                    state_key = self.visited[state_key]
                path.reverse()
                return path, nodes_visited, self.new_seen, True
            standard_neighbors = []
            for nr1, nr2 in get_neighbors_nj(r1, r2):
                nr1r = reduce_relator_nj(nr1)
                nr2r = reduce_relator_nj(nr2)
                total_len = len(nr1r) + len(nr2r)

                if total_len < self.max_len:
                    canon_r1, canon_r2 = canonical_pair_nj(nr1r, nr2r)
                    key_new = state_to_key((canon_r1, canon_r2))
                    if key_new not in self.visited:
                        standard_neighbors.append((canon_r1, canon_r2, key_new, total_len))

            for canon_r1, canon_r2, key_new, _ in standard_neighbors:
                self.visited[key_new] = key
                self.new_seen.add(key_new)
                next_depth = depth + 1
                next_priority = self._priority(canon_r1, canon_r2)
                heapq.heappush(self.pq, (next_priority, next_depth, key_new))

            self._generate_cov_neighbors(key, depth)

        if self.verbose:
            print("No trivial relators found.")
            min_pres = min(self.new_seen, key=lambda k: len(k[0]) + len(k[1]))
            min_pres_r1 = min(self.new_seen, key=lambda k: (len(k[0]), len(k[1])))
            print(f"Minimal element found: r1 = {min_pres[0]}, r2 = {min_pres[1]}, Size: ({len(min_pres[0])}, {len(min_pres[1])})")
            print(f"Minimal element found: r1 = {min_pres_r1[0]}, r2 = {min_pres_r1[1]}, Size: ({len(min_pres_r1[0])}, {len(min_pres_r1[1])})")

        return None, nodes_visited, self.new_seen, False
