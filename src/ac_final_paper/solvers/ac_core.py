"""Core rank-two Andrews-Curtis word operations.

This module packages and adapts the GS-Sub core distributed in ACSolverX's
``greedy_search.ipynb``. Keeping it as normal Python makes imports, testing,
and packaging deterministic. See ``THIRD_PARTY_NOTICES.md`` for the upstream
MIT notice and citation.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit


counterexamples = {}


# Map characters to integers and back

# We store the group elements as pairs of booleans. This is helps make comparisons and inversions faster.
# The first boolean represents the generator (x or y) and the second boolean represents inversion (x or X).
char_to_array = {'x': [True, True], 'X': [True, False], 'y': [False, True], 'Y': [False, False]}

# We use a function instead of a dictionary for better performance with Numba.
# This function is called a lot.
@njit(inline='always')
def array_to_char(bool1, bool2):
    if bool1:
        if bool2:
            return 'x'
        return 'X'
    elif bool2:
        return 'y'
    return 'Y'

# -----------------------------------------------------
# Numba helpers with boolean arrays
# -----------------------------------------------------

@njit(inline='always')
def is_inverse_nj(a, b):
    '''
        Check if two group elements are inverses of each other.

        The elements are assumed to be in the form [x, y] where x, y are bools.
    '''
    return (a[0] == b[0]) and (a[1] != b[1])

@njit(inline='always')
def is_equal_nj(a, b):
    '''
        Check if two group elements are equal.

        The elements are assumed to be in the form [x, y] where x, y are bools.
    '''
    return (a[0] == b[0]) and (a[1] == b[1])

@njit(inline='always')
def is_less_than(a, b):
    '''
        Returns a < b with order defined as the lexicographic order with True > False.

        The elements are assumed to be in the form [x, y] where x, y are bools.
    '''
    if a[0] != b[0]:
        # If the first elements are not equal, then either a = True, b = False or a = False, b = True
        # In either case, a < b == b.
        return b[0]
    else:
        # If the first elements are equal, compare the second elements.
        return a[1] < b[1]

@njit
def inverse_relator_nj(rel=np.array([[]])):
    '''
        Invert a relator.

        We assume that rel is a relator in array form. (Numpy array of shape (n, 2))
        Thus inversion corresponds to reversing the order and flipping the boolean in the second coordinate.
    '''
    res = rel.copy()
    res = np.flipud(res)
    res[:, 1] = np.logical_not(res[:, 1])  # Flip the second element of each pair

    return res

@njit
def reduce_relator_nj(rel):
    '''
        Reduce a relator by removing joined pairs of inverses. Does not modify the input array.

        Assumes that rel is a relator in array form. (Numpy array of shape (n, 2))
        Also removes pairs of inverses cyclically.

    '''

    n = len(rel)
    if n == 0:
        return rel.copy()

    # Stack reduction is safe even when the entire word cancels.  The original
    # notebook indexed past the end for a two-letter inverse pair.
    rel_list = np.zeros_like(rel)
    current_index = -1
    for add_index in range(n):
        if current_index >= 0 and is_inverse_nj(rel_list[current_index], rel[add_index]):
            current_index -= 1
        else:
            current_index += 1
            rel_list[current_index] = rel[add_index]

    rel_list = rel_list[:current_index + 1]
    if len(rel_list) == 0:
        return rel_list

    # If the first and last elements are inverses, we can remove them cyclically
    # and check to see if we can remove any more pairs.
    if is_inverse_nj(rel_list[0], rel_list[-1]):
        i = 1
        half_len = len(rel_list) / 2
        while i < half_len and is_inverse_nj(rel_list[i], rel_list[-1 - i]):
            i += 1
        rel_list = rel_list[i:-i]  # Remove the cyclic inverses.

    return rel_list

@njit
def find_minimal_rotation(rel):
    '''
        Find the minimal rotation of a string using Booth's algorithm.

        Uses the ordering `Y < y < X < x` for the characters.

        Booth's algorithm keeps canonicalization linear in the word length.
    '''

    n = len(rel)
    rel = np.concatenate((rel, rel))
    f = np.full(2 * n, -1, dtype=np.int32)
    k = 0
    for j in range(1, 2 * n):
        i = f[j - k - 1]
        while i != -1 and (not is_equal_nj(rel[j], rel[k + i + 1])):
            if is_less_than(rel[j], rel[k + i + 1]):
                k = j - i - 1
            i = f[i]
        if i == -1 and (not is_equal_nj(rel[j], rel[k])):
            if is_less_than(rel[j], rel[k]):
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return rel[k:k + n]

@njit
def lex_cmp_array(a, b):
    """
        Compare two numpy arrays of shape (n, 2) with bool types in lexiographic order.
        Returns a >= b.

        Note that when comparing, True > False so the implied order on symbols is: Y < y < X < x.
    """
    for x, y in zip(a, b):
        if x[0] == ~y[0]:
            # If the elements are different then either x = True, y = False or x = False, y = True
            #  In either case, x > y == x.
            return x[0]
        elif x[1] == ~y[1]:
            return x[1]

    # If we reach here the arrays are equal.
    return True

@njit
def lex_cmp_pair(a, b):
    """
        Compare two 1D numpy arrays of length 2 with bool types in lexiographic order.
        Returns a > b.

        Note that when comparing, True > False so the implied order on symbols is: Y < y < X < x.
    """
    if a[0] == ~b[0]:
        # If the elements are different then either a = True, b = False or a = False, b = True
        #  In either case, a > b = a.
        return a[0]
    elif a[1] == ~b[1]:
        return a[1]

    # If we reach here the arrays are equal.
    return False

@njit
def canonical_relator_nj(r):
    '''
        Find the canonical relator for a given relator.
    '''
    r_min = find_minimal_rotation(r)
    inv_min = find_minimal_rotation(inverse_relator_nj(r))

    # If r_min > inv_min return inv_min, else return r_min.
    if lex_cmp_array(r_min, inv_min):
        return inv_min
    return r_min


@njit
def canonical_pair_nj(r1, r2):
    '''
        Find the canonical pair of relators for a given pair of relators.

        Signed generator relabelings are handled separately where required.
    '''
    cr1 = canonical_relator_nj(r1)
    cr2 = canonical_relator_nj(r2)

    # sort pairs by length then lex order
    if len(cr1) > len(cr2) or (len(cr1) == len(cr2) and lex_cmp_array(cr1, cr2)):
        (cr1, cr2) = (cr2, cr1)

    return cr1, cr2

@njit
def get_neighbors_nj(r1, r2):
    '''
        Get all substitution neighbors of a pair of relators r1 and r2.

        Given two relators r1 and r2, let `i` be an index such that `r1[i] = (r2[i])^{-1}`.
        Then the neighour is defined by `neighbour = np.concatenate(r1[:i+1], r2, r2[i+1:])`
        and we have two neighbors of `(r1, r2)` namely `(neighbour, r2)` and `(r1, neighbour)`.

        We find all neighbors of both `(r1, r2)` and `(r1, r2^{-1})`.
    '''
    results = []
    candidates = [r2, inverse_relator_nj(r2)]
    for idx_c in range(2):
        c = candidates[idx_c]
        len_r1 = len(r1)
        len_c = len(c)
        for i in range(len_r1):
            rot1 = np.roll(r1, 2*i)
            for j in range(len_c):
                rot2 = np.roll(c, 2*j)
                if len(rot1) > 0 and len(rot2) > 0 and is_inverse_nj(rot1[-1], rot2[0]):
                    neighbour = np.concatenate((rot1, rot2))

                    # Replace r1
                    results.append((neighbour, r2))
                    # Replace r2
                    results.append((r1, neighbour))
    return results

# Helper to convert string to bool array and back
def str_to_arr(s):
    '''
        Convert a string in xXyY to a numpy array of shape (n, 2) with bool types.
    '''
    return np.array([char_to_array[c] for c in s], dtype=bool)

@njit(inline='always')
def arr_to_str(arr):
    '''
        Convert a numpy array of shape (n, 2) with bool types to a string in xXyY.
    '''
    # We use a function here as it lets us use Numba's njit decorator for performance.
    chars = [array_to_char(c[0], c[1]) for c in arr]
    return ''.join(chars)

@njit(inline='always')
def state_to_key(state):
    '''
        Convert tuple of arrays to tuple of strings for dict keys
    '''
    return (arr_to_str(state[0]), arr_to_str(state[1]))

def generator_count_product_priority(r1, r2):
    return 3 * min(len(r1), len(r2)) + max(len(r1), len(r2))

def canonical_pair_str(r1, r2):
    r1_arr = str_to_arr(r1)
    r2_arr = str_to_arr(r2)

    (canon_r1_arr, canon_r2_arr) = canonical_pair_nj(reduce_relator_nj(r1_arr), reduce_relator_nj(r2_arr))

    return arr_to_str(canon_r1_arr), arr_to_str(canon_r2_arr)

# ----------------------------------------------
# Main solver class
# ----------------------------------------------

class ACRelatorSolver:
    def __init__(self, r1, r2, max_nodes=10000, max_len = 100, visited=None, verbose=True, stop_early=True):
        self.max_nodes = max_nodes

        # Verbose mode controls what information is printed.
        self.verbose = verbose

        # stop_early controls whether we stop the search when we find a known counterexample.
        self.stop_early = stop_early

        # The input visited allows us to continue a search but passing in a collection of already visited states.
        if visited is None:
            self.visited = dict()
        else:
            self.visited = visited

        # The new_seen set is used to track newly seen states in the current search.
        self.new_seen = set()

        # The maximum length of relators we are willing to consider.
        # In general the smaller this is, the faster our search will be.
        self.max_len = max_len

        # Depth keeps track of the path length.
        # Mostly here in case we want to easily investigate check out BFS or A^* search.
        self.current_depth = 0
        self.max_depth = 0

        # The maximum and minimum greedy scores of the states we have seen so far.
        initial_r1_arr = str_to_arr(r1)
        initial_r2_arr = str_to_arr(r2)

        # Priority queue for the search.
        self.pq = []
        r1_arr = initial_r1_arr
        r2_arr = initial_r2_arr
        self.initial_state = canonical_pair_nj(reduce_relator_nj(r1_arr), reduce_relator_nj(r2_arr))
        self.max_priority = generator_count_product_priority(self.initial_state[0], self.initial_state[1])
        self.min_priority = generator_count_product_priority(self.initial_state[0], self.initial_state[1])
        self.counterexamples = {canonical_pair_str(k[0], k[1]): v for k, v in counterexamples.items()}

    def _priority(self, r1, r2):
        return generator_count_product_priority(r1, r2)

    def solve(self):
        initial_priority = self._priority(self.initial_state[0], self.initial_state[1])
        heapq.heappush(self.pq, (initial_priority, 0, state_to_key(self.initial_state)))
        self.visited[state_to_key(self.initial_state)] = None
        self.new_seen = set()
        self.new_seen.add(state_to_key(self.initial_state))
        nodes_visited = 0

        while self.pq and nodes_visited < self.max_nodes:
            _, depth, key = heapq.heappop(self.pq) # GS with scalar length score
            # depth, _, key = heapq.heappop(self.pq) # BFS
            nodes_visited += 1
            r1, r2 = self._key_to_state(key)

            if self.verbose:
                # if depth > self.max_depth:
                #     print(f"First state of depth {depth}, priority {len(r1)+len(r2)} ({arr_to_str(r1)},{arr_to_str(r2)})")
                #     self.max_depth = depth

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
                # print("Found trivial relators!")
                # print(f"r1 = {arr_to_str(r1)}, r2 = {arr_to_str(r2)}, depth = {depth}")
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

            for nr1, nr2 in get_neighbors_nj(r1, r2):
                nr1r = reduce_relator_nj(nr1)
                nr2r = reduce_relator_nj(nr2)

                if len(nr1r) + len(nr2r) < self.max_len:
                    canon_r1, canon_r2 = canonical_pair_nj(nr1r, nr2r)

                    key_new = state_to_key((canon_r1, canon_r2))
                    if key_new not in self.visited:
                        self.visited[key_new] = key
                        self.new_seen.add(key_new)
                        how_balanced = min(len(canon_r1), len(canon_r2)) / max(len(canon_r1), len(canon_r2))
                        priority = self._priority(canon_r1, canon_r2) # total length plus shorter relator
                        # priority = len(canon_r1) + len(canon_r2) + abs(len(canon_r1) - len(canon_r2))  # a new test priority
                        # if nodes_visited < 100000:
                        #     priority = 100-len(canon_r1) - len(canon_r2)  # another new test priority
                        # else:
                        #     priority = len(canon_r1) + len(canon_r2)
                        # heapq.heappush(self.pq, (depth+1, priority, key_new)) # BFS
                        heapq.heappush(self.pq, (priority, depth+1, key_new)) # GS

        # If we reach here, we have not found any trivial relators.
        # It can be interesting to see what the minimal relators found are.
        # This is particularly useful when the blocks checking len(r1) + len(r2) are commented out.
        if self.verbose:
            print("No trivial relators found.")
            min_pres = min(self.new_seen, key=lambda k: len(k[0]) + len(k[1]) + min(len(k[0]), len(k[1])))
            min_pres_r1 = min(self.new_seen, key=lambda k: (len(k[0]), len(k[1])))
            print(f"Minimal element found: r1 = {min_pres[0]}, r2 = {min_pres[1]}, Size: ({len(min_pres[0])}, {len(min_pres[1])})")
            print(f"Minimal element found: r1 = {min_pres_r1[0]}, r2 = {min_pres_r1[1]}, Size: ({len(min_pres_r1[0])}, {len(min_pres_r1[1])})")

        return None, nodes_visited, self.new_seen, False

    def _key_to_state(self, key):
        return (str_to_arr(key[0]), str_to_arr(key[1]))

def apply_automorphism(relator, automorphism):
    """Applies an automorphism to a relator."""
    return ''.join(automorphism[gen] for gen in relator)

def get_length_increase(path):
    '''Calculates the difference between the longest presentation and the starting presentation'''
    if not path:
        return 0
    max_length = max(len(r1) + len(r2) for r1, r2 in path)
    return max_length - (len(path[0][0]) + len(path[0][1]))

def reduce(rel):
    changed = True
    while changed:
        changed = False
        i = 0
        new_rel = ''
        while i < len(rel):
            if i + 1 < len(rel) and ((rel[i] == 'x' and rel[i+1] == 'X') or
                                        (rel[i] == 'X' and rel[i+1] == 'x') or
                                        (rel[i] == 'y' and rel[i+1] == 'Y') or
                                        (rel[i] == 'Y' and rel[i+1] == 'y')):
                i += 2
                changed = True
            else:
                new_rel += rel[i]
                i += 1
        rel = new_rel

        # Cyclic cancel
        if len(rel) > 1 and ((rel[0] == 'x' and rel[-1] == 'X') or
                                (rel[0] == 'X' and rel[-1] == 'x') or
                                (rel[0] == 'y' and rel[-1] == 'Y') or
                                (rel[0] == 'Y' and rel[-1] == 'y')):
            rel = rel[1:-1]
            changed = True
    return rel
