import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unsolved124_pipeline import (  # noqa: E402
    build_algorithm_summary,
    build_coverage_summary,
    build_head_to_head_summary,
    build_triple_queue_summary,
    load_paired_results,
)
import unsolved124_pipeline as pipeline  # noqa: E402


class Unsolved124PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paired = load_paired_results(
            ROOT / "data" / "wandb_unsolved124_dual_triple.csv"
        )

    def test_pairing_and_key_findings(self):
        self.assertEqual(len(self.paired), 124)
        summaries = build_algorithm_summary(self.paired)
        self.assertEqual([row["algorithm_label"] for row in summaries], ["Auto Dual GS", "Triple GS"])
        self.assertEqual([row["reduced_count"] for row in summaries], [22, 28])

        head_to_head = {
            row["outcome"]: row["presentations"]
            for row in build_head_to_head_summary(self.paired)
        }
        self.assertEqual(head_to_head, {"Auto Dual better": 0, "Tie": 115, "Triple better": 9})

        coverage = {
            row["coverage_group"]: row["presentations"]
            for row in build_coverage_summary(self.paired)
        }
        self.assertEqual(
            coverage,
            {"Auto Dual only": 0, "Both reduced": 22, "Triple only": 6, "Neither reduced": 96},
        )

    def test_triple_queue_fractions_sum_to_one(self):
        queue_summary = build_triple_queue_summary(self.paired)
        self.assertAlmostEqual(
            sum(row["queue_pop_fraction"] for row in queue_summary),
            1.0,
            places=12,
        )

    def test_reduction_plots_derive_ranges_and_captions_from_data(self):
        paired = [
            {
                "presentation_index": 0,
                "presentation_label": "auto-only",
                "auto_dual_length_reduction": 4,
                "triple_length_reduction": 0,
                "auto_dual_reduced": 1,
                "triple_reduced": 0,
                "reduction_coverage_group": "Auto Dual only",
            },
            {
                "presentation_index": 1,
                "presentation_label": "triple-only",
                "auto_dual_length_reduction": 0,
                "triple_length_reduction": 5,
                "auto_dual_reduced": 0,
                "triple_reduced": 1,
                "reduction_coverage_group": "Triple only",
            },
        ]
        captured = []

        def capture_figure(figure, *_args, **_kwargs):
            captured.append(figure)
            return []

        with patch.object(pipeline, "_save_figure", side_effect=capture_figure):
            pipeline._plot_reduction_coverage(paired, Path("unused"), "slides")
            pipeline._plot_reduction_magnitude(paired, Path("unused"), "paper")
            pipeline._plot_reduced_cases_matrix(paired, Path("unused"), "paper")

        coverage_axis = captured[0].axes[0]
        self.assertEqual(
            coverage_axis.get_title(),
            "Triple GS reduced 1 additional hard presentation",
        )
        self.assertTrue(
            any(
                "1 Auto Dual reduction was not matched by Triple GS." == text.get_text()
                for text in coverage_axis.texts
            )
        )

        magnitude_axis = captured[1].axes[0]
        np.testing.assert_array_equal(magnitude_axis.get_xticks(), np.arange(1, 6))

        matrix_axis = captured[2].axes[0]
        self.assertEqual(matrix_axis.images[0].get_clim(), (0, 5))
        np.testing.assert_array_equal(captured[2].axes[1].get_yticks(), np.arange(0, 6))


if __name__ == "__main__":
    unittest.main()
