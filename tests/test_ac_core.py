from __future__ import annotations

import itertools
import unittest

from ac_final_paper.solvers import ac_core
from ac_final_paper.solvers.standard_single_cov_gs import (
    _canonical_start,
    load_single_cov_namespace,
)


class ACCoreTests(unittest.TestCase):
    def test_complete_inverse_pair_reduces_to_empty(self) -> None:
        for word in ("xX", "Xx", "yY", "Yy"):
            reduced = ac_core.reduce_relator_nj(ac_core.str_to_arr(word))
            self.assertEqual(ac_core.arr_to_str(reduced), "")

    def test_reduction_is_idempotent_for_short_words(self) -> None:
        alphabet = "xXyY"
        for length in range(1, 7):
            for letters in itertools.product(alphabet, repeat=length):
                word = "".join(letters)
                once = ac_core.reduce_relator_nj(ac_core.str_to_arr(word))
                twice = ac_core.reduce_relator_nj(once)
                self.assertEqual(ac_core.arr_to_str(once), ac_core.arr_to_str(twice))

    def test_packaged_single_cov_namespace_is_complete(self) -> None:
        namespace = load_single_cov_namespace()
        required = {
            "str_to_arr",
            "reduce_relator_nj",
            "canonical_pair_nj",
            "get_neighbors_nj",
            "state_to_key",
            "canonicalize_state_from_strings",
            "COVRelatorSolver",
            "build_cov_neighbor_state",
        }
        self.assertTrue(required.issubset(namespace))
        self.assertEqual(_canonical_start(namespace, ("YYXyx", "Yx")), ("Yx", "YYXyx"))


if __name__ == "__main__":
    unittest.main()
