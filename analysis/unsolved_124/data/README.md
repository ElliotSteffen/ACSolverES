# Unsolved-124 data provenance

`wandb_unsolved124_dual_triple.csv` contains two experiment records for each of
124 Miller-Schupp benchmark class representatives: one Auto Dual GS record and
one Triple GS record. Elliot Steffen generated these algorithm measurements.

The 124 labels and relator pairs exactly match the project source table
`miller_schupp_unsolved_124.csv`, whose SHA-256 digest is:

`D319DDF4C5E1449377C2BE867DA48110C21EC2E75FC9E8985DA53833599CF914`

That table selected one representative for each class still treated as
unsolved when these experiments were run. The underlying family is due to
Miller and Schupp (1999), and the benchmark/classification lineage is described
by Shehper et al. (arXiv:2408.15332) and Fagan et al. (arXiv:2606.21611).

The CSV is an adapted dataset: benchmark identifiers and relators are combined
with new ACSolverES measurements. Derived summaries and figures under
`../outputs/` were generated from this CSV. See the repository-level
`LICENSE-DATA` and `THIRD_PARTY_NOTICES.md` files for licensing and full
citations.
