# timss-greml

Can a variance-component method borrowed from statistical genetics (GREML) tell
us anything useful about the link between teaching practice and mathematics
achievement in TIMSS 2023?

> ### Status: design phase. No results.
>
> No achievement data has been analysed. No model has been fitted. What exists
> is a downloaded dataset, a documented sampling structure, a pre-analysis plan
> written before any outcome was examined, and two rounds of adversarial review
> that materially changed the design and found a real bug in the code.
>
> The current verdict from adversarial review is: *"GREML is no longer
> intrinsically indefensible here, but this implementation should not reach
> outcomes yet."* Seven blocking issues are open, documented in
> [`analysis-plan/open-issues.md`](analysis-plan/open-issues.md).

---

# Part 1 — Plain-language summary

## The idea

TIMSS 2023 tested about 324,000 fourteen-year-olds across 47 education systems,
and asked their teachers a long list of questions about how they teach. The
obvious question is whether how you teach shows up in how well students do.

The approach here borrows a trick from genetics. Geneticists face a problem:
there are millions of genes and no single one has an effect big enough to
detect reliably on its own. So they ask a sideways question instead — *"do
people with more similar genes overall have more similar heights?"* If yes,
genes matter in total, even though you still can't name a single culprit.

The plan was to do the same for teaching: rather than *"does setting homework
help?"*, ask **"do classrooms where teachers describe similar teaching get
similar results?"**

## The problem we found

Teachers answer the questionnaire once, about their class as a whole. So the
teaching description is really a label on the *classroom*, not on the student —
every child in a class gets an identical profile.

That breaks the naive version of the method, because it can no longer separate
two very different statements:

- *"These classes get similar results because the teaching is similar."*
- *"These students get similar results because they are in the same class."*

Students in the same class are alike for many reasons that have nothing to do
with teaching: they may have been sorted by prior attainment, they share a
school, a catchment area, a funding level, a curriculum.

It is like trying to work out whether recipes explain how good restaurant food
is, when the only thing you can observe is *which restaurant you are sitting
in*. You would "discover" that recipes explain nearly everything, when you had
really only discovered that restaurants differ.

The dangerous part is that the method gives no warning. It returns a
confident-looking number that you would naturally describe as *"X% of results
come down to teaching"* — and it would be wrong.

## What we did about it

The question was sharpened into one the method can actually answer. Not *"do
classes differ?"* (they plainly do), but: **"among classes that all differ from
each other anyway, do the ones reporting similar teaching get similar
results?"** Classroom-to-classroom difference is taken as given and subtracted
out first, so the answer cannot be manufactured by classes simply being
different.

The design was then attacked deliberately, twice — by an independent survey of
the research literature, and by a separate AI model instructed to act as a
hostile reviewer and destroy the proposal.

- **Round one:** "This is a bad use of GREML." It independently derived the
  restaurant problem above.
- **Round two,** after redesign: "Not intrinsically indefensible any more — but
  the implementation isn't trustworthy yet."

Two things came out of that worth highlighting:

**A real bug.** The code was silently mixing *science* teachers into a *maths*
analysis. It raised no error; it just gave wrong answers. The headline counts
survived, but one figure would have been reported as 98% when the truth is
1.7%. No amount of reasoning about statistics would have caught this — someone
had to check.

**A faulty safety check.** A quick test built to answer "is this project even
worth doing?" turned out to be built wrong, in a direction that would have
waved the project through. It was stopped mid-run rather than produce a number
already known to be meaningless.

## What to expect if this continues

Most likely a small, heavily-qualified number — or nothing at all. Three
realistic outcomes:

1. **Nothing detectable.** Interesting rather than disappointing: it would mean
   the questions TIMSS asks teachers do not capture whatever makes classrooms
   differ. That matters directly to the people who design these surveys.
2. **A small signal**, with no way to rule out that it reflects which students
   were placed in which class.
3. **The decoys light up.** Deliberate control tests are built from things that
   definitely are *not* teaching. If those score as highly as the real thing,
   the method is detecting class composition, and that gets reported.

**What this cannot produce, in any scenario, is "do more of practice X, it
works."** No one randomly assigned students to classrooms, so cause and effect
cannot be separated here by any method. At best this bounds *how much* might be
there — never *which parts*, and never *whether it caused anything*.

The real contribution is methodological: **can this genetics technique be
usefully transplanted into education data?** A careful, well-evidenced *"no,
and here is exactly why"* is a legitimate result, and much of that argument is
already documented here.

---

# Part 2 — Technical account

## Research question

> Is between-classroom variance in TIMSS 2023 grade-8 mathematics achievement
> **structured by** measured teaching practice, over and above unstructured
> classroom-to-classroom variation?

Descriptive, not causal. Students are not randomly assigned to classes,
teachers are not randomly assigned to classes, and practice is self-reported
contemporaneously with the outcome. Tracking, intake composition, school
quality and teacher assignment all load onto any estimate. Every quantity is an
**upper bound on association**; the causal share may be zero.

The pre-analysis plan bans `heritability`, `h²`, `attributable to`, `explained
by`, `effect of` and `impact of` from the write-up, because each smuggles in a
claim the design cannot support.

## Why GREML rather than a standard multilevel model

GREML earns its place for three specific reasons, not because it is novel:

1. **Bias at `M ≈ C`.** With `M ≈ 33` practice items and `C ≈ 290` classes per
   system, entering the items as fixed effects in a multilevel model inflates
   the apparent reduction in between-class variance by roughly `M/C ≈ 11%`
   through overfitting alone. REML shrinkage avoids this.
2. **Signal present vs signal extractable.** Cross-fitted prediction is
   unbiased but answers a different question, and with ~290 classes it will
   extract very little even where real signal exists. This is the same gap as
   between SNP-heritability and polygenic-score R².
3. **Domain partitioning.** Multiple kernels can split structured variance
   across practice domains simultaneously, avoiding the order-dependence and
   bias of nested-R² comparisons. (Now demoted — see open issues.)

Where GREML contributes nothing: causal identification.

## The identification problem

Practice is measured on teachers and is therefore constant within class. With
`H` the `N × C` class-membership matrix and `W` the `C × M` standardised
class-level practice matrix, `Z = HW`, so the kernel factorises:

```
A = ZZ'/M = H (WW'/M) H' = H K H'
```

The random effect is `g = Hu` with `u ~ N(0, σ²_g K)` — a **class-level**
effect whose class-to-class covariance is constrained to `K`. Three
consequences:

- `rank(A) ≤ min(C, M)` irrespective of sample size. Additional students per
  class refine `σ²_e` and the class means; they add no practice information.
  The effective sample size is the number of **classes**.
- If `K = κI`, the model is **algebraically identical** to an exchangeable
  class random effect, and `σ²_g/(σ²_g + σ²_e)` is exactly the intraclass
  correlation. Standardising many near-independent items drives `K` toward
  precisely this degenerate case.
- Fitting `A` alone would therefore report the entire between-class variance
  share — tracking, intake, school quality, teacher selection included — under
  the label "practice".

School fixed effects do not rescue this and can destroy it: where a school
contributed one sampled class, `H = S`, so `P_S H = 0` and `P_S A P_S = 0`
exactly. The component is annihilated, not merely attenuated.

## Revised specification

The one-kernel model is retained only as a diagnostic, explicitly labelled an
ICC upper bound. The primary model fits the ordinary class effect and the
practice kernel jointly:

```
V = τ² HH'  +  σ²_g HKH'  +  σ²_e I
H₀ : σ²_g = 0   (class effect retained)
ψ  = σ²_g / (σ²_g + τ²)
```

`τ²` absorbs everything making classes differ for unmeasured reasons. `σ²_g` is
identified only from the *pattern* of covariance among classes with similar
practice profiles. `ψ` is reported as a share of **between-class** variance,
never of total variance.

Inference must be boundary-aware: `σ²_g = 0` lies on the parameter-space
boundary, so the LRT reference is a `½χ²₀ + ½χ²₁` mixture calibrated by
design-preserving permutation (permuting whole teacher profiles across classes,
never student rows). Wald intervals on `ψ`, normal-theory meta-analysis of `ψ̂`,
and discarding boundary estimates are all excluded.

## Observed design

From `scripts/recon_design.py`, all 47 systems, grade 8, mathematics teachers
only:

| | |
|---|---|
| Students | 323,854 |
| Classes | 13,553 |
| Schools | 9,406 |
| Schools with ≥2 sampled classes | 3,746 (39.8%) |
| Classes usable for within-school contrasts | 7,893 |
| Mathematics teachers | 14,141 |
| Students linking to >1 maths teacher | 5,521 (1.7%) |

Per-system estimates will be too imprecise to interpret individually; the
cross-system meta-analysis is the analysis, not a robustness check.

Within-school coverage is highly uneven — USA 91.5%, Sweden 53.9%, Türkiye
34.0%, versus South Africa 0.0%, Uzbekistan 0.6%, Singapore 0.7%. Systems
therefore enter the within-school analysis by a pre-set eligibility rule rather
than universally.

## Plausible values and the conditioning model

Established from TIMSS 2023 Technical Report chapters 11–12 rather than
assumed. The conditioning model comprises principal components of the
**student** (and, at grade 4, parent) context variables — retained to 90% of
variance, capped at 5% of unweighted national sample size — plus student
gender, test language, an optional country-specific variable, and **a
criterion-scaled classroom-within-school indicator**.

Two consequences in opposite directions:

- **Teacher and school questionnaire content is absent.** Practice–achievement
  associations from the published plausible values are therefore **attenuated
  toward zero** (Mislevy 1991; Meng 1994), not circularly inflated. The project
  thus has identified bias in both directions — confounding inflates,
  attenuation deflates — and neither is claimed to cancel the other.
- **Class membership is conditioned on**, criterion-scaled to each class's mean
  interim achievement. Since conditioning removes attenuation of group
  differences *for grouping variables included in the model*, between-class
  variance — which `ψ` is a share of — should be properly recovered. Because
  practice is a class-level variable, its association operates through the
  class mean, which is exactly what is conditioned on.

The second point favours the project, so it is verified rather than asserted:
criterion scaling uses class means estimated from ~25 students. The BSA files
carry raw cognitive item responses, so a PV-based ICC can be compared against
an unconditioned number-correct ICC per system. TIMSS publishes no EAP/WLE or
number-correct score, so raw items are also the only available fallback if
plausible values prove unsuitable.

## Falsification suite (specified, not yet run)

Negative-control kernels built by identical code from variables that are *not*
practice — most importantly `BTBG13A–I` ("teaching limited by student
factors"), which is essentially class intake and should predict achievement
strongly. Also job satisfaction, professional development, school climate,
response-style indices, and matched-noise kernels.

**Decision rule fixed in advance:** if the intake or matched-noise kernels yield
a structured component comparable to the practice kernel, the practice kernel
is functioning as a class identifier and the headline result is reported as
null regardless of its nominal size.

Plus negative-control *outcomes* (class sex and immigrant composition, home
resources — things practice cannot cause but sorting reveals), specification-
curve analysis over standardisation and item-weighting choices, and simulation
calibration under the actual class sizes, missingness and observed `W`.

## Method provenance

The design was adversarially reviewed **before** implementation and again after
revision, by an independent literature survey and by OpenAI Codex
(`gpt-5.6-sol`) prompted as a hostile reviewer. Both independently derived the
`A = HKH'` factorisation and identified ICC degeneracy as fatal to the naive
specification.

- Round 1 verdict: *"This is a bad use of GREML... In its ICC-like limit it
  relabels all class clustering as practice."*
- Round 2 verdict, on the revised design: *"Reject the revised plan in its
  present form... These are repairable. GREML is no longer intrinsically
  indefensible here, but this implementation should not reach outcomes yet."*

The two-component model, class-level computation, boundary-aware inference and
negative-control suite are all responses to that critique.

---

# Part 3 — Status

## Complete

- Reproducible environment: `uv` for Python; R 4.5.2 with 117 packages pinned
  via `renv`
- Download pipeline: resumable, size-verified, SHA-256 checksummed; 47 systems
  fetched and extracted (~3.7 GB)
- Sampling-design reconnaissance across all systems
- Pre-analysis plan, committed before any outcome data was examined
- Conditioning-model facts established from primary sources
- Two rounds of adversarial review, incorporated

## Open — blocking

Detailed in [`analysis-plan/open-issues.md`](analysis-plan/open-issues.md):

1. **The feasibility gate computes the wrong information.** It uses the
   efficient information for a homoskedastic two-component model, while the
   planned model has three components with class-size-dependent
   heteroskedasticity. `K` must be residualised against both `I` and
   `R = diag(1/n_c)` under the `P₀`-weighted inner product. As written it would
   mechanically pass low-rank kernels — false reassurance exactly where the
   model is weakest.
2. **No school random effect** in the primary model, so the practice kernel can
   absorb same-school covariance.
3. **The class-mean collapse is not exact** as the plan claims; student-level
   covariates vary within class.
4. **Mean imputation may manufacture identification** — classes with heavy
   missingness are pulled toward the origin and so look more similar.
5. **The Gaussian null benchmark is a strawman**; needs ordinal,
   correlation-matched, missingness-preserving nulls.
6. **Domain partitioning is probably unidentified** at `M = 2–8` per domain;
   demoted from primary.
7. **Inclusion thresholds are arbitrary** round numbers rather than derived
   power criteria.

## Resolved

**Subject-filter bug.** Both scripts filtered the linkage file on `SUBJECT`,
which does not exist — the variable is `IDSUBJ` (1 = Mathematics). The fallback
then silently used *all* rows, mixing science teachers into a mathematics
analysis. Measured impact: design counts unchanged (class and school IDs are
identical across a student's subject rows), but teacher counts wrong
(34,265 → 14,141) and the multi-teacher rate badly so (98% → 1.7%). Both
scripts now raise on a missing `IDSUBJ`.

---

# Part 4 — Using this repository

## Layout

```
src/timss_greml/       download and ETL
scripts/               reconnaissance and diagnostics
  recon_design.py        sampling structure: classes, schools, linkage
  kernel_feasibility.py  go/no-go kernel diagnostic (see BLOCKING-1)
tests/                 numerical validation of the information formula
R/                     REML fitting and survey-aware benchmarks
analysis-plan/         pre-analysis plan, open issues
docs/                  data use policy
data/                  gitignored; checksums only
outputs/               gitignored; regenerated from code
```

## Setup

```bash
uv sync                                    # Python
Rscript -e 'renv::restore()'               # R (needs system R >= 4.5)
```

On Debian/Ubuntu, R itself needs
`sudo apt install r-base r-base-dev build-essential gfortran` plus the usual
`libcurl4-openssl-dev libssl-dev libxml2-dev` headers.

Direct R dependencies and the rationale for each are in `DEPENDENCIES.R`;
`renv.lock` pins all 117 packages including transitive ones.

**Architecture note.** Developed on `aarch64`. The precompiled `GCTA` and
`LDAK` binaries are x86-64 only, so REML is fitted with R packages accepting
arbitrary user-supplied covariance matrices (`sommer`, `rrBLUP`, `regress`).
Three independent implementations must agree to three significant figures
before any estimate is reported — stronger verification than trusting one
binary.

## Data

Not in this repository, for licensing and size reasons both. See
[`docs/data-use-policy.md`](docs/data-use-policy.md).

```bash
uv run python -m timss_greml.download --grade 8     # ~3.7 GB
uv run python -m timss_greml.download --verify-only # re-check checksums
```

Downloads from `timss2023.org`, verifies size, records SHA-256 checksums in
`data/checksums/manifest.json`, extracts, and resumes if interrupted. A
checksum mismatch against the committed manifest fails loudly rather than
continuing, since results computed on different versions of the source data are
not comparable.

## Citation

> SOURCE: IEA's Trends in International Mathematics and Science Study —
> TIMSS 2023. Copyright © 2025 International Association for the Evaluation of
> Educational Achievement (IEA).

## Licence

Code is MIT ([`LICENSE`](LICENSE)). This covers the code only, not the TIMSS
data, which remains under the IEA's terms.
