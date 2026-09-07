"""Single-COV move generation built on the packaged AC core."""

from __future__ import annotations

import heapq

from .ac_core import *  # noqa: F403


# -----------------------------------------------------
# Change-of-variables extension
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

def replace_one_subword(word, start, length, replacement):
    return word[:start] + replacement + word[start + length:]

def count_target_occurrences(word, target):
    inverse_target = _inverse_char[target]
    return sum(1 for char in word if char in (target, inverse_target))

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
    """AC search that scores substitution and COV neighbors in the same priority queue."""

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
        self.move_metadata = {}

    def _priority(self, r1, r2):
        return generator_count_product_priority(r1, r2)

    def _generate_cov_candidates(self, r1, r2):
        # Each cyclic rotation already represents a different start position in the relator,
        # so we only take prefixes of each rotation rather than sliding an extra window.
        # Deduplication keeps one candidate per (relator, cyclic start, subword length, target)
        # so repeated literal subwords in different positions are still treated as distinct moves.
        relators = (r1, r2)
        seen_candidates = set()
        for source_index, relator in enumerate(relators):
            max_subword_len = len(relator) - 2
            if max_subword_len < self.cov_min_subword_len:
                continue

            for rotation in cyclic_rotations(relator):
                target_totals = {
                    target: count_target_occurrences(rotation, target)
                    for target in COV_TARGETS
                }
                for subword_len in range(self.cov_min_subword_len, max_subword_len + 1):
                    subword = rotation[:subword_len]
                    defining_relator = make_defining_relator(subword)
                    for target in COV_TARGETS:
                        candidate_key = (source_index, rotation, subword_len, target)
                        if candidate_key in seen_candidates:
                            continue
                        seen_candidates.add(candidate_key)
                        self.cov_stats['cov_candidate_count'] += 1

                        # A single-COV move can only isolate the target if the chosen subword
                        # removes exactly all but one occurrence of that generator family.
                        target_total = target_totals[target]
                        target_in_subword = count_target_occurrences(subword, target)
                        if target_in_subword != target_total - 1:
                            continue

                        # After the cheap count check passes, build the modified relator and
                        # keep the exact solve step as a final validity check.
                        temporary_relator = replace_one_subword(rotation, 0, subword_len, COV_INTRODUCED)
                        solved_expression = solve_for_single_target(temporary_relator, target)
                        if solved_expression is not None:
                            self.cov_stats['cov_valid_candidate_count'] += 1
                            yield {
                                'source_index': source_index,
                                'rotated_source_relator': rotation,
                                'subword': subword,
                                'target': target,
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
            self.move_metadata[key_new] = {
                'move_type': 'cov',
                'cov_subword': candidate['subword'],
            }
            self.new_seen.add(key_new)
            self.cov_stats['cov_enqueued_count'] += 1
            priority = self._priority(canon_r1, canon_r2)
            heapq.heappush(self.pq, (priority, parent_depth + 1, key_new))

    def solve(self):
        initial_key = state_to_key(self.initial_state)
        initial_priority = self._priority(self.initial_state[0], self.initial_state[1])
        heapq.heappush(self.pq, (initial_priority, 0, initial_key))
        self.visited[initial_key] = None
        self.move_metadata[initial_key] = {
            'move_type': 'start',
            'cov_subword': None,
        }
        self.new_seen = set()
        self.new_seen.add(initial_key)
        nodes_visited = 0

        while self.pq and nodes_visited < self.max_nodes:
            _, depth, key = heapq.heappop(self.pq)
            nodes_visited += 1
            r1, r2 = self._key_to_state(key)

            if self.verbose:
                current_priority = self._priority(r1, r2)
                if current_priority > self.max_priority:
                    print(f"First state of priority {current_priority}, depth: {depth}, values: {len(r1)}, {len(r2)} ({arr_to_str(r1)},{arr_to_str(r2)}), nodes: {nodes_visited}")
                    self.max_priority = current_priority

                if current_priority < self.min_priority:
                    print(f"First state of priority {current_priority}, depth: {depth}, values: {len(r1)}, {len(r2)} ({arr_to_str(r1)},{arr_to_str(r2)}), nodes: {nodes_visited}")
                    self.min_priority = current_priority

            if len(r1) == 1 and len(r2) == 1:
                path = []
                state_key = key
                while state_key is not None:
                    path.append(self._key_to_state(state_key))
                    state_key = self.visited[state_key]
                path.reverse()
                return path, nodes_visited, self.new_seen, True
            elif self.stop_early and key in self.counterexamples:
                if self.verbose:
                    print(f"Found counterexample {self.counterexamples[key]} at depth {depth}, r1 = {arr_to_str(r1)}, r2 = {arr_to_str(r2)}, nodes = {nodes_visited}")
                path = []
                state_key = key
                while state_key is not None:
                    path.append(self._key_to_state(state_key))
                    state_key = self.visited[state_key]
                path.reverse()
                return path, nodes_visited, self.new_seen, False

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

            # Standard substitution neighbors are enqueued exactly as before.
            for canon_r1, canon_r2, key_new, _ in standard_neighbors:
                self.visited[key_new] = key
                self.move_metadata[key_new] = {
                    'move_type': 'standard',
                    'cov_subword': None,
                }
                self.new_seen.add(key_new)
                next_depth = depth + 1
                next_priority = self._priority(canon_r1, canon_r2)
                heapq.heappush(self.pq, (next_priority, next_depth, key_new))

            # COV expands on every popped state. Both move families feed the same priority
            # queue, with score = 10 * min(len(r1), len(r2)) + max(len(r1), len(r2)) and depth as the
            # only tie breaker.
            self._generate_cov_neighbors(key, depth)

        if self.verbose:
            print("No trivial relators found.")
            min_pres = min(
                self.new_seen,
                key=lambda k: generator_count_product_priority(k[0], k[1]),  # noqa: F405
            )
            min_pres_r1 = min(self.new_seen, key=lambda k: (len(k[0]), len(k[1])))
            print(f"Minimal element found: r1 = {min_pres[0]}, r2 = {min_pres[1]}, Size: ({len(min_pres[0])}, {len(min_pres[1])})")
            print(f"Minimal element found: r1 = {min_pres_r1[0]}, r2 = {min_pres_r1[1]}, Size: ({len(min_pres_r1[0])}, {len(min_pres_r1[1])})")

        return None, nodes_visited, self.new_seen, False
