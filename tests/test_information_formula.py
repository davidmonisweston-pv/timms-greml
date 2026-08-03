"""Numerically validate the Fisher-information formula the go/no-go gate uses.

`scripts/kernel_feasibility.py` decides whether the whole project is viable
using

    SE(theta) / v  ~  sqrt(2 / tr(D^2)),
    D = QKQ - [tr(QKQ)/rank(Q)] Q

That formula is an asymptotic, at-the-null approximation derived for a
two-component model with homoskedastic class means. If it is wrong, the gate is
wrong, and every downstream power claim with it.

So it is checked against brute force: simulate class means under the null,
fit the variance components by REML, and compare the empirical standard
deviation of theta-hat with what the formula predicts.

Two cases are checked separately, because the second is the one that actually
applies to TIMSS:

  balanced   - equal class sizes, homoskedastic class means. The regime the
               formula was derived for. Agreement here tests the algebra.
  unbalanced - unequal class sizes, so class means have variance
               tau^2 + sigma_e^2/n_c. Agreement here tests whether the
               approximation survives the real design; disagreement tells us
               the gate needs a heteroskedastic correction.

Run:  uv run pytest tests/test_information_formula.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

RNG = np.random.default_rng(11071963)


def build_kernel(W: np.ndarray) -> np.ndarray:
    K = W @ W.T / W.shape[1]
    return K / float(np.mean(np.diag(K)))


def identity_departure(K: np.ndarray, X: np.ndarray) -> float:
    """tr(D^2): information available to separate the kernel from an ICC."""
    C = K.shape[0]
    Q = np.eye(C) - X @ np.linalg.pinv(X)
    rank_q = int(round(np.trace(Q)))
    QKQ = Q @ K @ Q
    D = QKQ - (np.trace(QKQ) / rank_q) * Q
    return float(np.sum(D * D))


def reml_two_component(
    y: np.ndarray, X: np.ndarray, K: np.ndarray, resid_var: np.ndarray
) -> tuple[float, float]:
    """REML for  Var(y) = v*I + theta*K + diag(resid_var), returning (v, theta).

    theta is left unconstrained so that the sampling distribution at the null
    is not truncated - truncating it would hide exactly the boundary behaviour
    we are trying to measure.
    """
    C = len(y)

    def neg_restricted_ll(params: np.ndarray) -> float:
        log_v, theta = params
        v = np.exp(log_v)
        V = v * np.eye(C) + theta * K + np.diag(resid_var)
        # Keep the search inside the positive-definite region.
        eigenvalues = np.linalg.eigvalsh(V)
        if eigenvalues.min() <= 1e-8:
            return 1e10
        Vinv = np.linalg.inv(V)
        XtVinvX = X.T @ Vinv @ X
        P = Vinv - Vinv @ X @ np.linalg.solve(XtVinvX, X.T @ Vinv)
        _, logdet_v = np.linalg.slogdet(V)
        _, logdet_x = np.linalg.slogdet(XtVinvX)
        return 0.5 * (logdet_v + logdet_x + y @ P @ y)

    best = optimize.minimize(
        neg_restricted_ll,
        x0=np.array([np.log(np.var(y) + 1e-6), 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000},
    )
    return float(np.exp(best.x[0])), float(best.x[1])


def empirical_se(
    K: np.ndarray, X: np.ndarray, v_true: float, resid_var: np.ndarray, reps: int
) -> tuple[float, float]:
    """Standard deviation of theta-hat under the null theta = 0."""
    C = K.shape[0]
    V = v_true * np.eye(C) + np.diag(resid_var)
    chol = np.linalg.cholesky(V)
    estimates = []
    for _ in range(reps):
        y = chol @ RNG.standard_normal(C)
        _, theta = reml_two_component(y, X, K, resid_var)
        estimates.append(theta)
    arr = np.array(estimates)
    return float(arr.std(ddof=1)), float(arr.mean())


@pytest.mark.parametrize(
    ("label", "C", "M", "balanced"),
    [
        ("balanced", 200, 33, True),
        ("unbalanced", 200, 33, False),
    ],
)
def test_information_formula_matches_simulation(
    label: str, C: int, M: int, balanced: bool
) -> None:
    W = RNG.standard_normal((C, M))
    W = (W - W.mean(0)) / W.std(0, ddof=1)
    K = build_kernel(W)
    X = np.ones((C, 1))

    v_true = 1.0
    if balanced:
        # Homoskedastic class means: the regime the formula assumes.
        resid_var = np.zeros(C)
    else:
        # TIMSS-like: class sizes 10-40, so class-mean noise varies ~4x.
        n_c = RNG.integers(10, 41, size=C)
        resid_var = 9.0 / n_c  # sigma_e^2 / n_c with sigma_e^2 = 9

    tr_d2 = identity_departure(K, X)
    # v in the formula is the total variance of the class means.
    v_eff = v_true + float(np.mean(resid_var))
    predicted = np.sqrt(2.0 / tr_d2) * v_eff

    observed, bias = empirical_se(K, X, v_true, resid_var, reps=300)
    ratio = observed / predicted

    print(
        f"\n[{label}] C={C} M={M}  tr(D^2)={tr_d2:.1f}\n"
        f"  predicted SE : {predicted:.5f}\n"
        f"  observed SE  : {observed:.5f}\n"
        f"  ratio obs/pred: {ratio:.3f}\n"
        f"  mean theta-hat under null: {bias:+.5f} (should be ~0)"
    )

    # The formula is asymptotic, so exactness is not expected; a gate built on
    # it only needs to be right to within tens of percent. Outside this band it
    # would mislead about feasibility.
    assert 0.75 < ratio < 1.35, (
        f"[{label}] information formula off by {abs(1 - ratio):.0%}; "
        "the feasibility gate needs correcting before it is trusted"
    )

    # theta-hat should be near-unbiased at the null when left unconstrained.
    assert abs(bias) < 3 * observed / np.sqrt(300), (
        f"[{label}] theta-hat is biased at the null: {bias:+.5f}"
    )
