# ACSolverES

Elliot Steffen's Andrews-Curtis solver experiments.

Standalone, reproducible code and data for comparing four greedy-search
methods on 640 Andrews-Curtis presentations:

1. Sub-only Greedy Search (stored baseline)
2. Sub + Single-Auto-COV Dual GS
3. Sub + Auto Dual GS
4. Sub + Auto + Non-Auto COV Triple GS

The repository includes the exact compact C++ search engine, packaged Python
move generators, completed result CSVs, paper figures, and the separate
analysis of 124 unsolved presentations. No neighboring `AC_Solver_SFP` files
or dynamically executed notebook cells are required.

## Install

Python 3.10 or newer and a C++17 compiler are required. On Windows, install
the Visual Studio Build Tools workload **Desktop development with C++**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Installing the package compiles two modules from the same audited native
source: `native_compact_dual_gs` and `native_compact_triple_gs`.

## Verify

```bash
python -m pytest
python scripts/verify_repository.py
```

The tests check reduction/canonicalization edge cases, all three native solver
adapters, resumable CSV output, plotting, and the unsolved-124 analysis.
In a Git checkout, the repository hygiene check inspects tracked and unignored
files; local build products and caches excluded by `.gitignore` are not treated
as publication errors.

## Reproduce

Open `notebooks/Final_Paper_Experiments.ipynb` after installation. The notebook
uses `data/gssub_baseline_ms640.csv`, writes resumable runs below `results/`,
and regenerates matched tables and figures. Use a new run name whenever search
parameters change; the runner validates both the configuration and input hash.

The committed final run is in `results/ms640_final_paper_v1/`. All three result
files contain 640 unique presentation IDs, no recorded errors, and solved all
640 inputs. Runtime was intentionally not collected, so speed claims should
not be inferred from these files.

## Layout

- `src/ac_final_paper/`: experiment APIs and packaged solver code.
- `native/`: compact C++17 Dual/Triple GS engine.
- `notebooks/`: master experiment notebook.
- `data/`: 640-presentation Sub-only baseline.
- `results/ms640_final_paper_v1/`: final CSVs, manifest, tables, and figures.
- `analysis/unsolved_124/`: independent 124-presentation analysis and outputs.
- `tests/`: correctness and reproducibility checks.

## Data integrity

`python scripts/verify_repository.py` validates row counts, unique IDs,
recorded errors, input alignment, JSON fields, result checksums, stale files,
and files too large for ordinary GitHub storage. No file currently requires
Git LFS.

## Attribution and citation

ACSolverES is authored and maintained by Elliot Steffen. To cite this software,
use [`CITATION.cff`](CITATION.cff). Detailed code and dataset provenance is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The 640-case Sub-only GS baseline is externally supplied benchmark material
associated with the GS-Sub experiments of Fagan et al. The 124-case analysis
uses selected Miller-Schupp benchmark class representatives and reports new
Auto Dual GS and Triple GS measurements generated for this project. See the
README files in [`data/`](data/) and
[`analysis/unsolved_124/data/`](analysis/unsolved_124/data/) for the exact
scope, transformations, and limitations of those claims.

## License

Original ACSolverES software is released under the included MIT license.
Portions adapted from ACSolverX retain the upstream MIT notice in
`THIRD_PARTY_NOTICES.md`. Data and data-derived artifacts are released under
[`LICENSE-DATA`](LICENSE-DATA), the Creative Commons Attribution 4.0
International license. These licenses do not imply endorsement by the upstream
authors or the Math & AI Lab at Caltech.
