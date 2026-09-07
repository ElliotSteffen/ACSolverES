# Third-Party Notices and Research Attribution

This file distinguishes ACSolverES authorship from upstream code, benchmark
data, and mathematical sources. Elliot Steffen authored the ACSolverES-specific
algorithms, native engines, experiment orchestration, analysis pipeline, and
project modifications unless a file or section states otherwise.

## ACSolverX code

`src/ac_final_paper/solvers/ac_core.py` packages and adapts substantial portions
of the GS-Sub implementation in `greedy_search.ipynb` from
[Math-AI-Caltech/ACSolverX](https://github.com/Math-AI-Caltech/ACSolverX).
Solver implementations that build on this core retain this notice. ACSolverES
adds packaging, tests, edge-case fixes, COV and automorphism variants, compact
C++ engines, experiment runners, and analysis code. Those changes do not imply
endorsement by the ACSolverX authors.

Upstream code notice:

> MIT License
>
> Copyright (c) 2026 Math & AI Lab at Caltech
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The upstream repository requests citation of:

Lucas Fagan, Michele Tarquini, Ali Shehper, Maksymilian Manko, Angus Gruen,
Coco Huang, Giorgi Butbaia, Davide Passaro, and Sergei Gukov. "The Two-Hump
Problem: Bridging the Difficulty Gap in Mathematical Reinforcement Learning."
ICML 2026. arXiv:2606.21611. https://arxiv.org/abs/2606.21611

## Benchmark and baseline data

The Miller-Schupp presentation family originates in:

Charles F. Miller III and Paul E. Schupp. "Some presentations of the trivial
group." In *Groups, Languages and Geometry*, Contemporary Mathematics 250,
pages 113-115, American Mathematical Society, 1999.
https://doi.org/10.1090/conm/250/03848

The 1,190-presentation validation benchmark used to select the 640 solved cases
and the unsolved class representatives was established in prior work:

Ali Shehper, Anibal M. Medina-Mardones, Lucas Fagan, Bartlomiej Lewandowski,
Angus Gruen, Yang Qiu, Piotr Kucharski, Zhenghan Wang, and Sergei Gukov. "What
makes math problems hard for reinforcement learning: a case study."
arXiv:2408.15332, version 2, 2025. https://arxiv.org/abs/2408.15332

Its later GS-Sub results and equivalence-class classification are described by
Fagan et al. in arXiv:2606.21611 and distributed through ACSolverX. Upstream
datasets, including the Miller-Schupp benchmark, are licensed CC BY 4.0 by the
Math & AI Lab at Caltech. This repository's data license and modification
statement are in `LICENSE-DATA`.

`data/gssub_baseline_ms640.csv` is an externally supplied 640-case GS-Sub
baseline associated with that work. The repository history does not preserve
the identity of the individual who exported the CSV, so ACSolverES does not
claim original authorship of those baseline measurements.

The presentation-identifying fields in
`analysis/unsolved_124/data/wandb_unsolved124_dual_triple.csv` come from a
124-class subset of the Miller-Schupp benchmark representatives. Elliot
Steffen generated the paired Auto Dual GS and Triple GS measurements and the
derived summaries and figures in this repository.

## Python and C++ dependencies

ACSolverES depends on NumPy, Numba, Matplotlib, nbformat, tqdm, pybind11,
setuptools, and Python. These packages are not vendored here and remain under
their respective licenses. The dependency declarations in `pyproject.toml`
identify the versions supported by this project.
