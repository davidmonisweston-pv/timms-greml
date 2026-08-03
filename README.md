# timss-greml

Applying GREML-style variance-component estimation to TIMSS 2023 grade-8
mathematics, to ask whether differences between classrooms in achievement are
**structured by** what teachers report about their teaching practice.

> **This is a methods experiment, and it is designed so that it can fail.**
> The central risk — that the method silently reduces to an intraclass
> correlation wearing a costume — is tested up front by a diagnostic that runs
> before any achievement data is touched. A null result is a finding here, not
> a failure, and is reported as prominently as a positive one.

## The question

> Is between-classroom variance in TIMSS 2023 grade-8 mathematics achievement
> structured by measured teaching practice, over and above unstructured
> classroom-to-classroom variation? And how is that structure distributed
> across practice domains?

This is **descriptive, not causal**. Students are not randomly assigned to
classes, teachers are not randomly assigned to classes, and practice is
self-reported at the same time as the outcome is measured. Tracking, intake
composition, school quality and teacher assignment all load onto any estimate
we produce. Every number here is an **upper bound on association**, and the
causal share could be zero.

## Why GREML, and where it earns its keep

GREML (Yang et al. 2010; GCTA) estimates the variance jointly explained by a
large set of predictors without needing to identify which ones matter. Ported
out of genetics, it does three things the standard toolkit does poorly:

1. **Unbiased joint signal at `M ≈ C`.** With ~33 practice items and ~290
   classes per system, adding the items as fixed effects to a multilevel model
   inflates the apparent reduction in between-class variance by roughly
   `M/C ≈ 11%` through overfitting alone. REML shrinkage avoids that.
2. **It estimates signal *present*, not signal *extractable*.** Cross-fitted
   prediction is unbiased but answers a different question, and with ~290
   classes it will extract almost nothing even when real signal exists. This is
   the same gap as between SNP-heritability and polygenic-score R².
3. **Domain partitioning.** Multiple kernels split structured variance across
   practice domains simultaneously, avoiding the bias and order-dependence of
   nested-R² comparisons.

Where it adds nothing: causal identification. None whatsoever.

## The central methodological problem, and what we do about it

Practice is measured on teachers, so it is constant within a class. Writing `H`
for class membership and `W` for the class-level practice matrix, the kernel
factorises:

```
A = ZZ'/M = H (WW'/M) H' = H K H'
```

So the "student-level" random effect is really a **class-level** effect, and
thousands of students merely replicate rows of the kernel. If `K ∝ I`, the
model is *algebraically identical* to a plain exchangeable class random effect,
and the estimate is exactly the intraclass correlation — which would sweep up
tracking, intake, school quality and teacher selection and mislabel the lot
"classroom practice". Standardising many near-independent items pushes `K`
toward precisely that degenerate case.

Two responses, both load-bearing:

- **The headline model fits an ordinary class effect *and* the practice kernel
  together**, `V = τ²HH' + σ²_g HKH' + σ²_e I`, testing `σ²_g = 0` with the
  class effect retained. The estimand is the share of *between-class* variance
  structured by practice. The one-kernel model is reported only as a
  diagnostic, labelled as an ICC upper bound.
- **A go/no-go diagnostic runs first.** `scripts/kernel_feasibility.py`
  measures how far `K` departs from identity after projecting out fixed
  effects, and reports `SE(θ)/v ≈ sqrt(2/tr(D²))` per system. If that departure
  is negligible, the analysis is unidentified by construction and no sample
  size fixes it.

Full reasoning, including the falsification suite and the banned-vocabulary
list, is in [`analysis-plan/pre-analysis-plan.md`](analysis-plan/pre-analysis-plan.md),
which is committed before results.

## Repository layout

```
src/timss_greml/       download and ETL
scripts/               reconnaissance and diagnostics
  recon_design.py        sampling structure: classes, schools, linkage
  kernel_feasibility.py  the go/no-go kernel diagnostic
R/                     REML fitting and survey-aware benchmarks
analysis-plan/         pre-analysis plan and any deviations
docs/                  data use policy
data/                  gitignored; checksums only
outputs/               gitignored; regenerated from code
```

## Setup

Python via [uv](https://docs.astral.sh/uv/), R via `renv`.

```bash
# Python
uv sync

# R (needs a system R >= 4.5; on Debian/Ubuntu: sudo apt install r-base r-base-dev)
Rscript -e 'renv::restore()'
```

Direct R dependencies and the reason for each are listed in `DEPENDENCIES.R`;
`renv.lock` pins all 117 packages including transitive ones.

### A note on architecture

This was developed on `aarch64`. The precompiled `GCTA` and `LDAK` binaries are
x86-64 only and will not run there, so REML is fitted with R packages that
accept arbitrary user-supplied covariance matrices (`sommer`, `rrBLUP`,
`regress`). Three independent implementations must agree to three significant
figures before any estimate is reported — which is stronger verification than
trusting a single binary.

## Getting the data

The data is **not** in this repository, for licensing and size reasons both.
See [`docs/data-use-policy.md`](docs/data-use-policy.md).

```bash
uv run python -m timss_greml.download --grade 8   # ~3.7 GB
```

Downloads from `timss2023.org`, verifies size, records SHA-256 checksums in
`data/checksums/manifest.json`, and extracts. Resumes if interrupted.

## Observed design

From `scripts/recon_design.py` across all 47 education systems:

| | |
|---|---|
| Students | 323,917 |
| Classes | 13,553 |
| Schools | 9,406 |
| Schools with ≥2 sampled classes | 3,746 (39.8%) |
| Classes usable for within-school contrasts | 7,893 |

The effective sample size for a class-level exposure is the number of
**classes**, not students. Per-system estimates are consequently too noisy to
interpret individually; the meta-analysis across systems is the analysis, not a
robustness check.

Coverage for within-school contrasts is very uneven — USA 91.5%, Sweden 53.9%,
Türkiye 34.0%, but South Africa 0.0%, Uzbekistan 0.6%, Singapore 0.7%. Where a
school contributed one sampled class, school fixed effects annihilate the
practice kernel exactly (`P_S H = 0`), so that analysis is restricted by a
pre-set eligibility rule rather than run everywhere.

## Method provenance

The design was adversarially reviewed before implementation: by an independent
subagent survey of the GREML-outside-genetics literature, and by OpenAI Codex
(`gpt-5.6-sol`) instructed to attack the proposal as a hostile reviewer. Both
independently derived the `A = HKH'` factorisation and identified ICC
degeneracy as fatal to the naive specification. That critique produced the
two-component model, the class-level computation, the boundary-aware inference
and the negative-control suite. Codex's verdict on the *original* proposal was
blunt — "this is a bad use of GREML" — and the design here is the response to
it, not a dismissal of it.

## Citation

Any use of TIMSS data must cite:

> SOURCE: IEA's Trends in International Mathematics and Science Study —
> TIMSS 2023. Copyright © 2025 International Association for the Evaluation of
> Educational Achievement (IEA).

## Licence

Code is MIT ([`LICENSE`](LICENSE)). The licence covers the code only, not the
TIMSS data, which remains under the IEA's terms.
