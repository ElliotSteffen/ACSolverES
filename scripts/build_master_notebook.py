"""Generate the master experiment notebook with stable, reviewable cells."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
OUTPUT = REPOSITORY_ROOT / "notebooks" / "Final_Paper_Experiments.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (AC Solver)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}
notebook["cells"] = [
    markdown(
        """
        # Final Paper: 640-Presentation Algorithm Comparison

        This notebook controls and runs the three non-baseline algorithms:

        1. Sub + Single-Auto-COV Dual GS
        2. Sub + Auto Dual GS
        3. Sub + Auto + Non-Auto COV Triple GS

        Results are checkpointed locally after every presentation. The final
        analysis cell combines them with the supplied Sub-only GS baseline and
        exports the paper figures as both PNG and PDF files.
        """
    ),
    markdown("## 1. Locate and install the standalone repository"),
    code(
        r"""
        from pathlib import Path
        import importlib.util
        import subprocess
        import sys

        def find_repository_root() -> Path:
            cwd = Path.cwd().resolve()
            for candidate in (cwd, *cwd.parents):
                if (candidate / "pyproject.toml").exists() and (candidate / "src" / "ac_final_paper").exists():
                    return candidate
            raise FileNotFoundError("Could not locate the repository root.")

        REPOSITORY_ROOT = find_repository_root()
        if importlib.util.find_spec("ac_final_paper") is None:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", str(REPOSITORY_ROOT)]
            )

        print(f"Repository root: {REPOSITORY_ROOT}")
        """
    ),
    markdown("## 2. Imports and input validation"),
    code(
        r"""
        from ac_final_paper.algorithm_adapters import (
            AUTO_DUAL,
            AUTO_NONAUTO_TRIPLE,
            SINGLE_COV_DUAL,
            verify_native_backends,
        )
        from ac_final_paper.experiment_runner import load_presentations, result_coverage, run_experiment_suite
        from ac_final_paper.paper_plots import build_paper_artifacts

        BASELINE_CSV = REPOSITORY_ROOT / "data" / "gssub_baseline_ms640.csv"
        OUTPUT_ROOT = REPOSITORY_ROOT / "results"
        verify_native_backends()

        presentations = load_presentations(BASELINE_CSV)
        assert len(presentations) == 640, f"Expected 640 presentations, found {len(presentations)}"
        presentations[:3]
        """
    ),
    markdown(
        """
        ## 3. Master hyperparameter configuration

        Change `RUN_NAME` whenever any parameter changes. Use `LIMIT` or
        `PRESENTATION_IDS` for smoke tests; leave both as `None` for all 640.
        """
    ),
    code(
        r"""
        RUN_NAME = "ms640_final_paper_v1"
        RUN_DIR = OUTPUT_ROOT / RUN_NAME
        LIMIT = None
        PRESENTATION_IDS = None
        RESUME = True
        REQUIRE_ALL_SOLVED = True

        COMMON = {
            "total_node_budget": 1_000_000,
            "max_len": 50,
            "depth_tie_strategy": "deepest",
            "max_live_frontier_states": 5_000_000,
            "print_progress": False,
            "progress_interval": 10_000,
        }

        ALGORITHM_PARAMETERS = {
            SINGLE_COV_DUAL: {
                **COMMON,
                "cov_min_subword_len": 2,
                "cov_preference_threshold": 2,
                "seed_initial_cov_neighbors": True,
                "min_standard_moves_between_cov": 0,
            },
            AUTO_DUAL: {
                **COMMON,
                "auto_cov_preference_threshold": 2,
                "seed_initial_auto_cov_neighbors": True,
                "min_standard_moves_between_auto_cov": 0,
            },
            AUTO_NONAUTO_TRIPLE: {
                **COMMON,
                "auto_cov_preference_threshold": 2,
                "complete_cov_preference_threshold": 2,
                "seed_initial_auto_cov_neighbors": True,
                "seed_initial_complete_cov_neighbors": True,
                "min_standard_moves_between_auto_cov": 0,
            },
        }

        ALGORITHM_PARAMETERS
        """
    ),
    markdown(
        """
        ## 4. Run and checkpoint all three algorithms

        This cell writes one local CSV per algorithm under
        `results/<RUN_NAME>/`. It does not record runtime.
        """
    ),
    code(
        r"""
        RUN_DIR = run_experiment_suite(
            input_csv=BASELINE_CSV,
            output_root=OUTPUT_ROOT,
            run_name=RUN_NAME,
            algorithm_parameters=ALGORITHM_PARAMETERS,
            limit=LIMIT,
            pres_ids=PRESENTATION_IDS,
            resume=RESUME,
            fail_fast=True,
            require_all_solved=REQUIRE_ALL_SOLVED,
            print_progress=True,
        )

        print(f"Results saved to: {RUN_DIR}")
        result_coverage(RUN_DIR)
        """
    ),
    markdown(
        """
        ## 5. Process Sub-only plus new results and generate paper plots

        Run this cell after all three result files contain the full matched set.
        It creates summary CSVs and six figures in PNG and PDF formats.
        """
    ),
    code(
        r"""
        from IPython.display import Image, display

        ARTIFACTS = build_paper_artifacts(
            baseline_csv=BASELINE_CSV,
            run_dir=RUN_DIR,
            output_dir=RUN_DIR / "paper_artifacts",
            require_all_solved=True,
        )

        print(f"Summary table: {ARTIFACTS['summary_csv']}")
        print(f"Matched long-form data: {ARTIFACTS['combined_csv']}")
        for row in ARTIFACTS["summary"]:
            print(
                f"{row['algorithm_label']}: "
                f"median nodes={row['median_nodes']:.1f}, "
                f"mean nodes={row['mean_nodes']:.1f}, "
                f"geometric mean={row['geometric_mean_nodes']:.1f}, "
                f"median path={row['median_path_length']:.1f}, "
                f"fewer/equal/more={row['fewer_nodes_vs_sub_only']}/"
                f"{row['equal_nodes_vs_sub_only']}/{row['more_nodes_vs_sub_only']}"
            )

        for figure_path in ARTIFACTS["figures"]:
            if figure_path.suffix == ".png":
                display(Image(filename=str(figure_path), width=900))
        """
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
