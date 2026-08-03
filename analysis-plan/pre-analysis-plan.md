# Pre-analysis plan

**Status:** draft, written before any outcome data was examined.
**Last substantive revision:** see git history. This file is committed before
results so that the analytic choices below can be checked against what was
actually run.

Nothing in this plan had sight of achievement data. The design reconnaissance
(`scripts/recon_design.py`) and kernel feasibility diagnostic
(`scripts/kernel_feasibility.py`) use the sampling structure and the teacher
questionnaire only. No plausible value was loaded before this plan was fixed.

---

## 1. The question

> Is between-classroom variance in TIMSS 2023 grade-8 mathematics achievement
> **structured by** measured teaching practice, over and above unstructured
> classroom-to-classroom variation?

And, conditional on there being such structure:

> How is it distributed across practice domains (cognitive activation, task
> types, homework, assessment, digital use)?

### 1.1 What this is not

This is **not** a causal question and the design cannot answer one. Students
are not randomly assigned to classes, teachers are not randomly assigned to
classes, and practice is self-reported contemporaneously with the outcome.
Ability tracking, intake composition, school quality and teacher assignment
all load onto any "practice" component we estimate.

The following words are **banned** from the write-up, because each smuggles in
a causal or genetic claim the design does not support:

- "heritability", `h²`, "SNP-heritability"
- "attributable to", "explained by", "due to", "impact of", "effect of"
- "determines", "drives", "causes"

The permitted framing is "structured by", "associated with", "co-varies with",
and any estimate is described as an **upper bound on association**.

### 1.2 Why GREML rather than a standard model

GREML is used for the narrow thing it does better than the alternatives, not
because it is fashionable:

- With `M ≈ 33` practice items and `C ≈ 290` classes per system, adding items
  as fixed effects to a multilevel model inflates the apparent drop in
  between-class variance by roughly `M/C ≈ 11%` through overfitting alone.
  REML shrinkage estimates the joint signal without that bias.
- Cross-fitted prediction (elastic net) is unbiased but answers a different
  question: how much signal can be *extracted* from ~290 classes, not how much
  signal is *present*. It is systematically conservative for the latter.
- A variance share is a single scale-free scalar, comparable across 47
  education systems with different languages and response distributions, in a
  way that 33 individual item coefficients are not.
- Multiple kernels partition structured variance across practice domains
  simultaneously, avoiding the order-dependence and bias of nested-R²
  comparisons.

Where GREML adds nothing: causal identification. It adds none.

---

## 2. Data

TIMSS 2023, grade 8, mathematics. All 47 participating education systems.
Downloaded by `src/timss_greml/download.py` from `timss2023.org`; checksums in
`data/checksums/manifest.json`. Not redistributed (see
`docs/data-use-policy.md`).

Files used, per system (`xxx` = ISO3 code, `M8` = the 2023 cycle):

| File | Contents |
|---|---|
| `BSGxxxM8` | student background questionnaire, weights, `JKZONE`/`JKREP` |
| `BSAxxxM8` | achievement, 5 plausible values (`BSMMAT01`–`05`) |
| `BTMxxxM8` | mathematics teacher questionnaire (the practice items) |
| `BSTxxxM8` | student–teacher linkage; class IDs and teacher weights |
| `BCGxxxM8` | school questionnaire |

Observed design (from `scripts/recon_design.py`, all 47 systems):
323,917 students; 13,553 classes; 9,406 schools; 3,746 schools (39.8%)
contributed ≥2 sampled classes, covering 7,893 classes.

---

## 3. The model

### 3.1 Why the naive specification is rejected

Practice is measured on teachers and is therefore constant within class. With
`H` the `N × C` class-membership matrix and `W` the `C × M` standardised
class-level practice matrix, the student-level design matrix is `Z = HW`, so
the kernel factorises:

```
A = ZZ'/M = H (WW'/M) H' = H K H'
```

The random effect is `g = Hu` with `u ~ N(0, σ²_g K)`. It is a **class-level**
effect, not a student-level one. Three consequences:

1. `rank(A) ≤ min(C, M)` regardless of how many students are sampled. Extra
   students per class refine `σ²_e` and class means; they add no practice
   information.
2. If `K = κI`, the model is **algebraically identical** to an exchangeable
   class random effect, and `σ²_g/(σ²_g+σ²_e)` is exactly the intraclass
   correlation. Standardising many near-independent items drives `K` towards
   precisely this degenerate case.
3. Fitting `A` alone would therefore report the whole between-class variance
   share — tracking, intake, school quality, teacher selection and all — under
   the label "practice".

The one-kernel model `V = σ²_g HKH' + σ²_e I` is consequently **not** the
headline specification. It is reported only as a diagnostic, explicitly
labelled as an ICC upper bound.

### 3.2 Primary specification

Fit at **class level** (see §4), the primary model is:

```
V = τ² HH'  +  σ²_g HKH'  +  σ²_e I
```

with the null of interest

```
H₀ : σ²_g = 0     (the ordinary class effect retained)
```

`τ²` absorbs everything that makes classes differ for unmeasured reasons.
`σ²_g` is identified only from the *pattern* of covariance between classes with
similar practice profiles. The reported estimand is:

```
ψ = σ²_g / (σ²_g + τ²)
```

the share of **between-class** variance that is structured by measured
practice. `ψ` is reported with a boundary-aware interval (§6), never as a share
of total variance.

### 3.3 Identification and its failure modes

`σ²_g` is identified only if, after projecting out fixed effects `X`, the
matrices `P_X HH' P_X`, `P_X HKH' P_X` and `P_X` are linearly independent.
Identification fails when:

- `K ∝ I` — exact confounding with the class effect. **Pre-tested** by
  `scripts/kernel_feasibility.py`; systems where the non-identity fraction is
  negligible are reported as unidentified rather than assigned an estimate.
- School fixed effects are used where a school contributed one sampled class.
  Then `H = S`, so `P_S H = 0` and `P_S A P_S = 0` exactly — the component is
  annihilated, not merely attenuated. The within-school analysis is therefore
  **restricted to the 3,746 schools with ≥2 sampled classes**, and systems
  with too few such schools (South Africa 0.0%, Uzbekistan 0.6%, Singapore
  0.7%) are excluded from it by a pre-set rule (§3.5), not post hoc.

### 3.4 Fixed effects

`X` contains, in all specifications: an intercept; student sex; a home
educational-resources index; language spoken at home. Specifications adding
school fixed effects are reported separately, never as the primary, because
they change the estimand (within-school contrasts only).

Student SES covariates do **not** repair confounding. They are included because
omitting them is worse, and their inclusion is not claimed to identify
anything. There is no baseline achievement measure in TIMSS; this is stated as
a limitation, not worked around.

### 3.5 Pre-set inclusion rules

Fixed before estimation:

- A system enters the main analysis if it has ≥50 classes with linked teacher
  data and a non-identity fraction above the degeneracy floor.
- A system enters the within-school analysis if ≥20% of its sampled schools
  contributed ≥2 classes **and** it has ≥30 such schools.
- Classes linking to more than one mathematics teacher have practice items
  averaged; the share of such classes is reported per system, and a
  sensitivity analysis drops them.

---

## 4. Computation

`A = HKH'` is low rank, so the analysis is done on **precision-weighted class
means** rather than the `N × N` student matrix. This is not an approximation of
convenience: it is exact for the class-level component structure, and it turns
5 plausible values × up to ~250 replicate weights ≈ 1,250 REML fits per system
from infeasible into routine. It also removes the memory problem — an `N × N`
matrix pooled across systems would be ~720 GB.

Collapsing to class means with class-size-dependent precision gives

```
ȳ ~ N(Xβ, τ² I_C + σ²_g K + σ²_e diag(1/n_c))
```

Three independent REML implementations must agree to 3 significant figures on
every reported estimate, or the result is not reported:

1. `sommer::mmer` with a user-supplied covariance matrix
2. `rrBLUP::mixed.solve` / `regress`
3. a from-scratch average-information REML in the repo, validated against
   closed-form solutions in the balanced case

---

## 5. Plausible values and survey weights

TIMSS supplies 5 plausible values and JK2 jackknife replicate weights
(`JKZONE`/`JKREP`; up to 125 zones and 250 replicates, read from the data per
system rather than assumed).

- Each estimate is computed for **every plausible value**, combined by the
  standard TIMSS procedure: sampling variance by replication, plus
  between-plausible-value measurement variance inflated by `(1 + 1/L)`.
- **Caveat that must be stated in the write-up:** Rubin-style pooling assumes a
  regular estimator. A variance component is bounded at zero, asymmetric, and
  may have a point mass at the boundary, so pooling is an approximation that
  degrades near zero. Coverage of the exact procedure is therefore established
  by simulation (§7), and pooling is not assumed valid on authority.
- **Congeniality check:** plausible values are only reliable for associations
  the TIMSS conditioning model preserved. Whether teacher practice variables
  entered that model is checked against the Technical Report and reported.
  If they did not, associations may be attenuated, and this is disclosed.
- Weights: `TOTWGT` for student-level quantities. Standardisation of `W` and
  kernel construction are recomputed inside each replicate, since they are
  weight-dependent operations. Naively multiplying likelihood contributions by
  `TOTWGT` is **not** a valid survey-weighted likelihood for a multilevel model
  and is not done.
- Teachers are not a representative sample of teachers; they are teachers
  linked to a representative sample of students. Target populations are stated
  accordingly.

---

## 6. Inference

`σ²_g = 0` is on the parameter-space boundary, so standard asymptotics do not
apply.

- Point estimates from constrained REML.
- Testing by restricted likelihood-ratio against a `½χ²₀ + ½χ²₁` reference,
  **calibrated** by design-preserving permutation rather than assumed.
- Intervals by profile likelihood, permitted to include zero.
- **Banned:** Wald intervals on `ψ`; normal-theory meta-analysis of `ψ̂`;
  discarding boundary estimates; reporting the largest component across many
  kernels.
- Meta-analysis across systems on a variance-stabilised scale, by random-effects
  models (`metafor`), with heterogeneity reported. Systems with boundary
  estimates are retained, not dropped.

### 6.1 Permutation null

Permute **entire teacher practice profiles across classes**, never individual
student rows. Class size, teacher multiplicity, missingness pattern, school
structure and weights are preserved, and standardisation and `K` are rebuilt
after each permutation. The permutation null runs the **complete pipeline**
including item selection, not a single pre-selected fit.

Schemes: across classes within system; within sampling strata; within schools
where multiple classes exist.

---

## 7. Falsification suite

Every item below is run and reported regardless of outcome. A result that
survives none of these is not reported as a finding.

### 7.1 Nested model ladder

Fit in order and report all four:

```
V = σ²_e I
V = τ² HH' + σ²_e I                      (ICC baseline)
V = σ²_g HKH' + σ²_e I                   (naive - diagnostic only)
V = τ² HH' + σ²_g HKH' + σ²_e I          (primary)
```

Reported alongside: kernel eigenvalues and effective rank; the non-identity
fraction; condition numbers; profile likelihoods; and the estimated correlation
between `τ̂²` and `σ̂²_g`. **If the fourth model cannot separate the components,
the estimate is an ICC surrogate and is reported as such.**

### 7.2 Negative-control kernels

Built by identical code from variables that are *not* classroom practice:

| Kernel | Variables | Why |
|---|---|---|
| `nc_class_composition` | `BTBG13A–I` (teaching limited by student factors) | pure intake/composition; should predict achievement strongly |
| `nc_job_satisfaction` | `BTBG08*`, `BTBG09*` | about the teacher, not the class |
| `nc_prof_development` | `BTBM22*` | teacher background, not practice |
| `nc_school_climate` | `BTBG06*`, `BTBG07*` | school-level construct |
| response-style | acquiescence and extreme-responding indices over all Likert items | detects response style masquerading as practice |
| matched-noise | random standardised variables, same `M` and correlation spectrum | detects pure kernel geometry |

**Decision rule fixed in advance:** if `nc_class_composition` or matched-noise
yields a structured component comparable to `practice_all`, the practice kernel
is functioning as a class identifier and the headline result is reported as
null, whatever its nominal size.

### 7.3 Negative-control outcomes

Outcomes that teaching practice cannot plausibly cause, but that sorting would
reveal: class sex composition, immigrant/language composition, home-resources
index, parental education. A kernel that "explains" these is detecting
selection.

Science achievement is reported as a non-specificity check (not a clean
negative control, since general school quality affects both). Similar estimates
for maths and science would undermine any practice-specific reading.

### 7.4 Scaling and specification sensitivity

The estimate must be shown to be a property of the data, not of the kernel
design. Re-run under: teacher- vs student-weighted standardisation; weighted vs
unweighted standardisation; numeric vs ordinal-normal scoring; equal-item vs
equal-domain weighting; reliability weighting; whitening; dropping highly
correlated items; deliberately duplicating one domain; adding matched noise
items; alternative missingness rules.

Results are reported as a **specification curve**. Large movement in `ψ̂`
demonstrates it is a kernel-design parameter rather than a population quantity,
and is reported as such.

### 7.5 Simulation calibration

Run before the real analysis is believed, using the **actual** schools, classes,
class sizes, weights, missingness and observed `W`:

1. **Pure ICC null** — `y = s + u_c + e`, practice causally irrelevant. How
   often does the naive model report positive `ψ`?
2. **Incremental null** — class and school effects independent of `W`; fit the
   primary model. Type-I error and interval coverage.
3. **Confounding without causation** — `u_c` correlated with `W`, zero causal
   coefficient. Quantifies false attribution from sorting alone.
4. **Known signal** — sparse, dense, domain-structured and unequal `β`.
   Compare `ψ̂` against the true finite-population variance of `Wβ`.
5. **Response-style contamination** — acquiescence/extreme-response factors
   correlated with school quality.
6. **Informative sampling** — inclusion probability depending on class
   achievement; compare unweighted, weighted and replicate-based estimators.
7. **Boundary calibration** — under `σ²_g = 0`, estimate the point mass at
   zero, the empirical LRT distribution, and coverage after PV pooling.

**Gate:** unless the simulations reproduce nominal type-I error and coverage
for the exact pipeline, inferential results are reported as uncalibrated.

---

## 8. Benchmarks

Run alongside, because GREML answers "how much" and these answer "which":

- Survey-aware multilevel models (students in classes in schools).
- Within-school fixed-effects specifications, on the eligible subset (§3.5).
- Cross-fitted elastic net, **with folds at school level**. Splitting students
  or classes within a school leaks, because train and test records share
  identical `Z`.
- Specification-curve analysis over the defensible analytic space.

The comparison between the GREML variance share and the cross-fitted
out-of-sample R² is itself a reported result: the gap between "signal present"
and "signal extractable" is informative.

---

## 9. What a null result means

A null is a publishable finding, not a failure, and will be reported as
prominently as a positive one. If the structured component is indistinguishable
from zero it means: **the classroom differences that TIMSS detects in
achievement are largely orthogonal to what TIMSS asks teachers about their
practice.** That is directly relevant to anyone designing international
assessment questionnaires.

## 10. Deviations

Any departure from this plan is recorded in `analysis-plan/deviations.md` with
its reason and date, and results are reported both ways where feasible.
