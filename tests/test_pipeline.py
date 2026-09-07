from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ac_final_paper.algorithm_adapters import (
    ALGORITHM_ORDER,
    AUTO_DUAL,
    AUTO_NONAUTO_TRIPLE,
    SINGLE_COV_DUAL,
    run_algorithm,
    verify_native_backends,
)
from ac_final_paper.experiment_runner import (
    load_presentations,
    run_experiment_suite,
)
from ac_final_paper.paper_plots import (
    build_paper_artifacts,
    load_matched_results,
    summary_rows,
)


class FinalPaperPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        verify_native_backends()

    def test_all_algorithm_adapters_solve_ms0(self) -> None:
        presentation = ("YYXyx", "Yx")
        parameters = {
            SINGLE_COV_DUAL: {
                "total_node_budget": 100,
                "seed_initial_cov_neighbors": True,
            },
            AUTO_DUAL: {
                "total_node_budget": 100,
                "seed_initial_auto_cov_neighbors": True,
            },
            AUTO_NONAUTO_TRIPLE: {
                "total_node_budget": 100,
                "seed_initial_auto_cov_neighbors": True,
                "seed_initial_complete_cov_neighbors": False,
            },
        }
        for algorithm in ALGORITHM_ORDER:
            with self.subTest(algorithm=algorithm):
                result = run_algorithm(algorithm, presentation, parameters[algorithm])
                self.assertTrue(result["solved"])
                self.assertGreater(result["total_nodes_used"], 0)
                self.assertGreaterEqual(result["path_length"], 0)

    def test_resumable_runner_and_matched_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.csv"
            with baseline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("pres_id", "r1", "r2", "nodes_explored", "path_length"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pres_id": "0",
                        "r1": "YYXyx",
                        "r2": "Yx",
                        "nodes_explored": "3",
                        "path_length": "2",
                    }
                )

            configurations = {
                SINGLE_COV_DUAL: {
                    "total_node_budget": 100,
                    "seed_initial_cov_neighbors": True,
                },
                AUTO_DUAL: {
                    "total_node_budget": 100,
                    "seed_initial_auto_cov_neighbors": True,
                },
                AUTO_NONAUTO_TRIPLE: {
                    "total_node_budget": 100,
                    "seed_initial_auto_cov_neighbors": True,
                    "seed_initial_complete_cov_neighbors": False,
                },
            }
            run_dir = run_experiment_suite(
                input_csv=baseline,
                output_root=root / "results",
                run_name="smoke",
                algorithm_parameters=configurations,
                print_progress=False,
            )
            run_experiment_suite(
                input_csv=baseline,
                output_root=root / "results",
                run_name="smoke",
                algorithm_parameters=configurations,
                resume=True,
                print_progress=False,
            )
            matched = load_matched_results(baseline, run_dir)
            summaries = summary_rows(matched)
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(row["presentations"] == 1 for row in summaries))
            artifacts = build_paper_artifacts(
                baseline_csv=baseline,
                run_dir=run_dir,
            )
            self.assertTrue(artifacts["summary_csv"].exists())
            self.assertTrue(artifacts["combined_csv"].exists())
            self.assertEqual(len(artifacts["figures"]), 12)
            self.assertTrue(all(path.exists() for path in artifacts["figures"]))
            for algorithm in ALGORITHM_ORDER:
                self.assertEqual(len(load_presentations(baseline)), 1)
                with (run_dir / f"{algorithm}.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
