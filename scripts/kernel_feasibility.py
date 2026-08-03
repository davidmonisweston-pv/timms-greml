"""Go/no-go diagnostic: is the practice kernel distinguishable from an ICC?

The whole project turns on one question that can be answered from the teacher
questionnaire alone, without touching achievement data.

Because practice is measured at class level, the student-level kernel factors
as A = H K H', where H is class membership and K = WW'/M is the class-by-class
practice similarity matrix. The GREML variance component is therefore a class
random effect whose class-to-class covariance is constrained to equal K.

If K is proportional to the identity, that constraint is vacuous: the model
becomes algebraically identical to an ordinary exchangeable class random
effect, and "variance explained by classroom practice" is just the intraclass
correlation relabelled. Standardising many near-independent items pushes K
towards exactly that degenerate case (K_cc -> 1, K_cd -> 0).

So the quantity that decides feasibility is how far K departs from identity,
measured after projecting out the fixed effects:

    D       = QKQ - [tr(QKQ)/rank(Q)] Q          (kernel minus its identity part)
    I_theta = tr(D^2) / (2 v^2)                  (Fisher information at theta=0)
    SE(theta)/v ~ sqrt(2 / tr(D^2))

SE(theta)/v is the standard error of the structured component expressed as a
fraction of class-mean variance. Small is good. If D is ~zero, the practice
component is unidentified against a plain class effect and no amount of data
helps, because the information is zero by construction rather than by sample
size.

This script reports that number per education system, alongside a matched
random-kernel benchmark (same C, same M, i.i.d. standardised columns), which
is the value tr(D^2) takes when the items carry no shared structure at all.

Writes outputs/tables/kernel_feasibility.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "T23_Data_SPSS_G8" / "SPSS Data"
OUT_DIR = REPO_ROOT / "outputs" / "tables"

RNG = np.random.default_rng(20260803)

# ---------------------------------------------------------------------------
# Item sets. Kept as named domains so the multi-kernel partition and the
# negative controls draw on exactly the same construction code.
#
# "practice" is the target: things a teacher DOES. Deliberately excludes
# teacher background, job satisfaction, school climate and professional
# development, none of which are classroom practice.
# ---------------------------------------------------------------------------
DOMAINS: dict[str, list[str]] = {
    # Cognitive activation / instructional quality
    "cognitive_activation": [f"BTBG12{c}" for c in "ABCDEFG"],
    # What the teacher asks students to do in maths lessons
    "task_types": [f"BTBM15{c}" for c in "ABCDEFGH"],
    # Digital device use during maths
    "digital_use": [f"BTBM17D{c}" for c in "ABCDEF"],
    # Homework practice
    "homework": ["BTBM20A", "BTBM20BB", "BTBM20BC", "BTBM20BD", "BTBM20BE"],
    # Assessment strategy
    "assessment": [f"BTBM21{c}" for c in "ABCDE"],
    # Instructional time and tool policy
    "time_tools": ["BTBM14", "BTBM16"],
}

PRACTICE_ITEMS: list[str] = [v for items in DOMAINS.values() for v in items]

# Opportunity to learn - curriculum topics actually taught. Arguably practice,
# but a distinct construct, so it gets its own kernel rather than being mixed in.
OTL_ITEMS = [
    f"BTBM19{g}{c}"
    for g, n in (("A", 7), ("B", 8), ("C", 6), ("D", 4))
    for c in "ABCDEFGH"[:n]
]

# ---------------------------------------------------------------------------
# Negative controls. These are NOT classroom practice. If a kernel built from
# them yields a comparable structured component, the practice kernel is acting
# as a class identifier rather than measuring teaching.
# ---------------------------------------------------------------------------
NEGATIVE_CONTROLS: dict[str, list[str]] = {
    # Teacher's report of how much student factors limit teaching. This is
    # class composition and intake, and should track achievement strongly
    # precisely because it is a consequence of who is in the class.
    "nc_class_composition": [f"BTBG13{c}" for c in "ABCDEFGHI"],
    # Teacher job satisfaction and workload - about the teacher, not the class.
    "nc_job_satisfaction": [f"BTBG08{c}" for c in "ABCDEFG"]
    + [f"BTBG09{c}" for c in "ABCDEFGH"],
    # Professional development history - teacher background, not practice.
    "nc_prof_development": [f"BTBM22A{c}" for c in "ABCDEFG"]
    + [f"BTBM22B{c}" for c in "ABCDEFG"],
    # School climate as perceived by the teacher - a school-level construct.
    "nc_school_climate": [f"BTBG06{c}" for c in "ABCDEFGHIJK"]
    + [f"BTBG07{c}" for c in "ABCDEFG"],
}


def systems() -> list[str]:
    return sorted(p.name[3:6] for p in DATA_DIR.glob("btm*m8.sav"))


def load_teacher(iso: str) -> pd.DataFrame:
    frame, _ = pyreadstat.read_sav(
        str(DATA_DIR / f"btm{iso}m8.sav"), disable_datetime_conversion=True
    )
    frame.columns = [c.upper() for c in frame.columns]
    return frame


def load_link(iso: str) -> pd.DataFrame:
    frame, _ = pyreadstat.read_sav(
        str(DATA_DIR / f"bst{iso}m8.sav"), disable_datetime_conversion=True
    )
    frame.columns = [c.upper() for c in frame.columns]
    return frame[frame["SUBJECT"] == 1] if "SUBJECT" in frame.columns else frame


def class_level_matrix(
    teacher: pd.DataFrame, link: pd.DataFrame, items: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the class x item matrix W, standardised, with means imputed.

    Returns (W, class_ids, missing_fraction). Rows are classes; a class takes
    the responses of the teacher linked to it. Where a class links to several
    maths teachers the responses are averaged, which is the only neutral
    choice available and is recorded as a caveat.
    """
    present = [v for v in items if v in teacher.columns]
    if not present:
        raise ValueError("none of the requested items are in the teacher file")

    key = ["IDSCHOOL", "IDTEACH", "IDLINK"]
    tvals = teacher[key + present].copy()

    # Map each class to its linked teacher(s).
    lnk = link[key + ["IDCLASS"]].drop_duplicates()
    merged = lnk.merge(tvals, on=key, how="inner")
    if merged.empty:
        raise ValueError("teacher/link merge produced no rows")

    grouped = merged.groupby("IDCLASS")[present].mean()
    raw = grouped.to_numpy(dtype=float)
    class_ids = grouped.index.to_numpy()

    missing = float(np.isnan(raw).mean())

    # Standardise each item across classes, then impute missing to the mean
    # (zero after centring). Constant items carry no information and are dropped.
    mean = np.nanmean(raw, axis=0)
    sd = np.nanstd(raw, axis=0, ddof=1)
    keep = np.isfinite(sd) & (sd > 1e-10)
    raw, mean, sd = raw[:, keep], mean[keep], sd[keep]

    W = (raw - mean) / sd
    W[~np.isfinite(W)] = 0.0
    return W, class_ids, missing


def build_kernel(W: np.ndarray) -> np.ndarray:
    """K = WW'/M, then scaled so the mean diagonal is 1 (trace normalisation)."""
    K = W @ W.T / W.shape[1]
    diag_mean = float(np.mean(np.diag(K)))
    return K / diag_mean if diag_mean > 0 else K


def identity_departure(K: np.ndarray, X: np.ndarray | None = None) -> dict[str, float]:
    """How much of K is NOT explained by an identity (exchangeable) component.

    Q projects out the fixed effects X (an intercept at minimum). D removes the
    remaining identity component, so tr(D^2) is exactly the information
    available to distinguish the practice kernel from a plain class effect.
    """
    C = K.shape[0]
    if X is None:
        X = np.ones((C, 1))
    Q = np.eye(C) - X @ np.linalg.pinv(X)
    rank_q = int(round(np.trace(Q)))

    QKQ = Q @ K @ Q
    D = QKQ - (np.trace(QKQ) / rank_q) * Q
    tr_d2 = float(np.sum(D * D))
    tr_qkq2 = float(np.sum(QKQ * QKQ))

    return {
        "tr_D2": tr_d2,
        # Fraction of the kernel's projected variation that is orthogonal to
        # the identity direction. Near 0 => degenerate, it IS an ICC.
        "frac_non_identity": tr_d2 / tr_qkq2 if tr_qkq2 > 0 else 0.0,
        # Standard error of the structured component as a share of class-mean
        # variance. This is the headline feasibility number.
        "se_theta_over_v": float(np.sqrt(2.0 / tr_d2)) if tr_d2 > 0 else np.inf,
        "rank_q": rank_q,
    }


def random_benchmark(C: int, M: int, reps: int = 20) -> float:
    """tr(D^2) for a kernel built from pure noise with the same shape."""
    vals = []
    for _ in range(reps):
        W = RNG.standard_normal((C, M))
        W = (W - W.mean(0)) / W.std(0, ddof=1)
        vals.append(identity_departure(build_kernel(W))["tr_D2"])
    return float(np.mean(vals))


def eigen_summary(K: np.ndarray) -> dict[str, float]:
    ev = np.linalg.eigvalsh(K)[::-1]
    total = float(ev.sum())
    return {
        "eig_top1_share": float(ev[0] / total) if total > 0 else np.nan,
        "eig_top5_share": float(ev[:5].sum() / total) if total > 0 else np.nan,
        # Participation ratio: effective number of dimensions the kernel spans.
        "eig_effective_rank": float(total**2 / np.sum(ev**2)) if total > 0 else np.nan,
    }


def analyse(iso: str) -> list[dict[str, object]]:
    teacher, link = load_teacher(iso), load_link(iso)
    rows: list[dict[str, object]] = []

    kernels: dict[str, list[str]] = {
        "practice_all": PRACTICE_ITEMS,
        "otl_topics": OTL_ITEMS,
        **{f"domain_{k}": v for k, v in DOMAINS.items()},
        **NEGATIVE_CONTROLS,
    }

    for name, items in kernels.items():
        try:
            W, class_ids, missing = class_level_matrix(teacher, link, items)
        except ValueError:
            continue
        if W.shape[0] < 30 or W.shape[1] < 2:
            continue

        K = build_kernel(W)
        stats = identity_departure(K)
        rows.append(
            {
                "system": iso,
                "kernel": name,
                "C_classes": W.shape[0],
                "M_items": W.shape[1],
                "missing_frac": round(missing, 3),
                **{k: round(v, 5) for k, v in stats.items() if k != "rank_q"},
                **{k: round(v, 4) for k, v in eigen_summary(K).items()},
                "tr_D2_random_benchmark": round(
                    random_benchmark(W.shape[0], W.shape[1]), 1
                ),
            }
        )
    return rows


def main() -> int:
    only = sys.argv[1:] or systems()
    all_rows: list[dict[str, object]] = []

    for iso in only:
        try:
            rows = analyse(iso)
        except Exception as exc:  # noqa: BLE001
            print(f"  {iso}: FAILED - {type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)
        main_row = next((r for r in rows if r["kernel"] == "practice_all"), None)
        if main_row:
            print(
                f"  {iso}: C={main_row['C_classes']:>4} M={main_row['M_items']:>3}  "
                f"non-identity={main_row['frac_non_identity']:.3f}  "
                f"SE/v={main_row['se_theta_over_v']:.4f}  "
                f"trD2={main_row['tr_D2']:>9.1f} "
                f"(noise {main_row['tr_D2_random_benchmark']:>7.1f})"
            )

    table = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "kernel_feasibility.csv", index=False)

    print(f"\n{'=' * 72}\nVERDICT")
    prac = table[table["kernel"] == "practice_all"]
    print(f"  systems analysed              : {len(prac)}")
    print(f"  median non-identity fraction  : {prac['frac_non_identity'].median():.3f}")
    print(f"  median SE(theta)/v per system : {prac['se_theta_over_v'].median():.4f}")
    pooled = prac["se_theta_over_v"].median() / np.sqrt(len(prac))
    print(f"  implied pooled SE across all  : {pooled:.5f}")
    print(
        f"  median kernel/noise trD2 ratio: "
        f"{(prac['tr_D2'] / prac['tr_D2_random_benchmark']).median():.2f}"
    )
    print(f"\nWrote {OUT_DIR / 'kernel_feasibility.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
