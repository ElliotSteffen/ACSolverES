"""Generate the reusable notebook wrapper for the unsolved-124 plot pipeline."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Unsolved_124_Plots.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


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
        # Unsolved 124: Auto Dual GS vs. Triple GS

        This notebook validates the paired W&B export, writes processed tables,
        and creates paper and 16:9 slide figures. Auto Dual GS is always shown
        first.
        """
    ),
    markdown("## 1. Locate the pipeline and import it"),
    code(
        """
        from pathlib import Path
        import sys

        def find_pipeline_root() -> Path:
            for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
                if (candidate / "unsolved124_pipeline.py").exists():
                    return candidate
                nested = candidate / "analysis" / "unsolved_124"
                if (nested / "unsolved124_pipeline.py").exists():
                    return nested
            raise FileNotFoundError("Could not locate analysis/unsolved_124.")

        PIPELINE_ROOT = find_pipeline_root()
        if str(PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(PIPELINE_ROOT))

        from unsolved124_pipeline import build_artifacts
        print(f"Pipeline root: {PIPELINE_ROOT}")
        """
    ),
    markdown("## 2. Configure input and output paths"),
    code(
        """
        INPUT_CSV = PIPELINE_ROOT / "data" / "wandb_unsolved124_dual_triple.csv"
        OUTPUT_DIR = PIPELINE_ROOT / "outputs"

        assert INPUT_CSV.exists(), INPUT_CSV
        INPUT_CSV
        """
    ),
    markdown("## 3. Process the paired runs and generate all figures"),
    code(
        """
        ARTIFACTS = build_artifacts(INPUT_CSV, OUTPUT_DIR)

        print("Key findings")
        for key, value in ARTIFACTS["key_findings"].items():
            print(f"  {key}: {value}")

        print()
        print("Algorithm summaries")
        for row in ARTIFACTS["algorithm_summary"]:
            print(
                f"  {row['algorithm_label']}: "
                f"reduced {row['reduced_count']}/{row['presentations']}, "
                f"mean reduction {row['mean_length_reduction']:.3f}"
            )
        """
    ),
    markdown("## 4. Preview the slide figures"),
    code(
        """
        from IPython.display import Image, display

        slide_dir = OUTPUT_DIR / "figures" / "slides"
        for name in (
            "reduction_coverage.png",
            "paired_best_total_length.png",
            "reduction_magnitude.png",
            "reduced_cases_matrix.png",
            "triple_queue_mix.png",
        ):
            print(name)
            display(Image(filename=str(slide_dir / name), width=1000))
        """
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
