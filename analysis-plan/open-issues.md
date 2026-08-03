# Open issues

Unresolved problems that must be fixed before any estimate is believed. Each
came from adversarial review of the plan, and each is blocking unless marked
otherwise.

Round-2 review by OpenAI Codex (`gpt-5.6-sol`), prompted to attack the
*revised* design, returned: **"Reject the revised plan in its present form...
These are repairable. GREML is no longer intrinsically indefensible here, but
this implementation should not reach outcomes yet."**

That is the current status. The method is no longer considered unsound in
principle; the implementation is not yet trustworthy.

---

## BLOCKING-1. The feasibility gate does not compute information for the model it gates

**Status:** open. This invalidates the go/no-go number until fixed.

`scripts/kernel_feasibility.py` uses

```
D = QKQ - [tr(QKQ)/rank(Q)] Q ,    SE(theta)/v ~ sqrt(2/tr(D^2))
```

This is provably the correct efficient information — but **only** for the
homoskedastic two-component problem `L'y ~ N(0, vI + theta B)`. The planned
class-level model is

```
V0 = tau^2 I + sigma_e^2 R ,   R = diag(1/n_c)
```

with *two* nuisance components. The efficient information is the Schur
complement

```
I_{theta.nu} = I_{theta,theta} - I_{theta,nu} I_{nu,nu}^{-1} I_{nu,theta}
I_{ab}       = (1/2) tr(P0 G_a P0 G_b),   G_theta = K, G_tau = I, G_e = R
```

so `K` must be residualised against **both** `I` and `R` under the
`P0`-weighted inner product, not against `I` alone under the unweighted
Frobenius norm. TIMSS class sizes vary roughly fourfold, so this is not a
negligible correction.

Consequence: the current gate **will mechanically pass low-rank kernels**, i.e.
it gives false reassurance precisely where the model is weakest.

Fix: replace `identity_departure()` with the efficient-information calculation
over a grid of plausible nuisance variance ratios, and gate on simulated power
or maximum profile-likelihood interval width rather than `frac_non_identity`.

`tests/test_information_formula.py` already checks the formula numerically in
both balanced and unbalanced cases; it should fail in the unbalanced case until
this is fixed. **Run it and record the result.**

## BLOCKING-2. The primary model omits a school component

**Status:** open.

`V = tau^2 HH' + sigma_g^2 HKH' + sigma_e^2 I` has no `SS'` term. Practice
profiles are similar within a school (shared curriculum, training, intake), so
without a school random effect the practice kernel can absorb same-school
covariance and report it as practice structure.

Fix: add a school random effect to the primary model, and rerun every
information diagnostic after residualising `K` against `I`, `R` **and** `SS'`.

## BLOCKING-3. The class-mean collapse is not exact as claimed

**Status:** open. The plan currently overstates this and must be corrected.

The plan asserts collapsing to precision-weighted class means is "exact, not an
approximation of convenience". That is **wrong** as stated. Student-level
covariates in `X` (sex, home resources, language) vary *within* class.
Collapsing replaces them with class means, which is a different — ecological —
fixed-effects model, and discards the within-class information that identifies
`sigma_e^2`.

Fix: either (a) retain student-level fixed effects by residualising at student
level before collapsing, and state precisely what is preserved; or (b) drop the
exactness claim and fit at student level using Woodbury identities on the
low-rank structure. Do not leave the claim as written.

## BLOCKING-4. Mean imputation may manufacture identification

**Status:** open.

Missing items are imputed to zero after standardising. Classes with heavy
missingness are pulled toward the origin, making them look **more similar to
each other** — which manufactures off-diagonal structure in `K` and could
register as identification that is really a missingness pattern.

Fix: replace with a pre-specified ordinal multilevel imputation or latent
measurement model, propagate kernel uncertainty, and report
**missingness-kernel alignment** as a diagnostic in its own right.

## BLOCKING-5. The random benchmark is a strawman

**Status:** open.

`random_benchmark()` uses i.i.d. Gaussian columns. Real items are ordinal,
bounded, correlated and missing-imputed, so the real kernel will beat this
benchmark essentially by construction, making the comparison uninformative.

Fix: generate ordinal, correlation-matched, missingness-preserving null
kernels. Discard the Gaussian benchmark.

## BLOCKING-6. Domain partitioning is probably unidentified

**Status:** open. Demote from primary.

Domain kernels have M as small as 2 (`time_tools`) and typically 5-8, and the
domains are mutually correlated (teachers who do one thing do others). With
C ~ 200-450 the multi-kernel information matrix is likely ill-conditioned.

Fix: make the omnibus practice kernel primary. Attempt domain partitioning
**only** if the residualised kernel information matrix is demonstrably
well-conditioned, and report the condition number either way.

## BLOCKING-7. Inclusion thresholds are arbitrary

**Status:** open.

The pre-set counts (>=50 classes, >=20% schools with 2+ classes, >=30 such
schools) are round numbers, not derived quantities. Selecting systems on a
data-derived quantity (the non-identity fraction) before estimation also risks
selection bias in the meta-analysis.

Fix: replace count thresholds with system-specific power/precision criteria,
state the numerical gate explicitly in the plan, and define the meta-analytic
target over selected systems explicitly.

---

## RESOLVED

### R-1. Subject filter bug in the linkage files

**Fixed** in `scripts/recon_design.py` and `scripts/kernel_feasibility.py`.

Both filtered the student-teacher linkage file on a variable named `SUBJECT`.
No such variable exists; it is `IDSUBJ` (1 = Mathematics, 2 = Science,
9 = omitted). Worse than the wrong name was the fallback: `... if subject else
link` silently used **all** rows when the column was absent, mixing science
teachers into a mathematics analysis rather than failing.

Impact, measured by re-running both versions:

| quantity | buggy | fixed |
|---|---|---|
| students | 323,917 | 323,854 |
| classes | 13,553 | 13,553 (unchanged) |
| schools | 9,406 | 9,406 (unchanged) |
| schools with >=2 classes | 3,746 (39.8%) | 3,746 (39.8%) (unchanged) |
| teachers | 34,265 | 14,141 |
| students with >1 maths teacher | 318,180 | 5,521 |

The design counts were unaffected, because class and school IDs are identical
across a student's subject rows. Teacher counts and multi-teacher rates were
badly wrong: the true multi-teacher rate is **1.7%** of students, not 98%.

Both scripts now raise on a missing `IDSUBJ` rather than falling back.

The `kernel_feasibility.py` run in progress when this was found was killed
rather than allowed to finish, since its class-to-teacher merge could attach
maths practice data through a science link. It must be re-run after
BLOCKING-1 is fixed, not before — its current output would be meaningless
regardless.

---

## Also to correct in the write-up

Not blocking, but the plan currently overclaims:

- **The null interpretation is too strong.** `sigma_g^2 = 0` means "no
  detectable covariance matching this kernel under this model" — *not* that
  classroom achievement differences are "largely orthogonal to what TIMSS asks
  teachers". Low power, measurement error, missingness and kernel
  misspecification are all live alternative explanations and must be named
  alongside any null.

- **Measurement remains the single biggest weakness**, and no amount of REML
  fixes it. A contemporaneous, heavily-missing, self-reported questionnaire
  kernel has no validated separation between practice, response style, intake
  and school context. Even a flawless fit estimates covariance alignment with
  *that object*. The response-style negative control does not resolve this,
  because the response-style index is computed from the same items.
