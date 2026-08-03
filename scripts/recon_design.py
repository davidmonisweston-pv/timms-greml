"""Reconnaissance on the TIMSS 2023 grade-8 design, before any modelling.

Three facts decide whether the planned analysis is feasible at all, and none of
them can be assumed from documentation:

1. How many CLASSES each system sampled. The practice kernel is a class-level
   object, so the number of classes - not students - is the effective sample
   size, and it drives every standard error.

2. How many SCHOOLS contributed two or more sampled classes. If a school
   contributes exactly one class then the class-membership matrix equals the
   school-membership matrix, school fixed effects annihilate the practice
   kernel entirely (P_S H = 0 => P_S A P_S = 0), and the within-school
   analysis is not merely underpowered but unidentified.

3. How many students link to more than one mathematics teacher. Such students
   break the clean student -> class mapping the kernel construction assumes.

Writes outputs/tables/design_recon.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyreadstat

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "T23_Data_SPSS_G8" / "SPSS Data"
OUT_DIR = REPO_ROOT / "outputs" / "tables"


def systems() -> list[str]:
    return sorted(p.name[3:6] for p in DATA_DIR.glob("bsg*m8.sav"))


def read(prefix: str, iso: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / f"{prefix}{iso}m8.sav"
    frame, _ = pyreadstat.read_sav(
        str(path), usecols=columns, disable_datetime_conversion=True
    )
    return frame


def recon_one(iso: str) -> dict[str, object]:
    """Design summary for a single education system."""
    # The student-teacher linkage file is the only place the student -> class
    # -> teacher mapping exists. IDLINK distinguishes multiple classes taught
    # by the same teacher; IDCLASS is the sampled class.
    link = read("bst", iso)
    cols = {c.upper(): c for c in link.columns}

    def col(name: str) -> str | None:
        return cols.get(name)

    subject = col("SUBJECT")
    # SUBJECT==1 is mathematics in the grade-8 linkage file.
    maths = link[link[subject] == 1] if subject else link

    idstud, idteach = col("IDSTUD"), col("IDTEACH")
    idclass, idschool = col("IDCLASS"), col("IDSCHOOL")

    n_students = maths[idstud].nunique()
    n_teachers = maths[idteach].nunique()
    n_classes = maths[idclass].nunique() if idclass else pd.NA
    n_schools = maths[idschool].nunique() if idschool else pd.NA

    # How many maths teachers does each student link to?
    per_student = maths.groupby(idstud)[idteach].nunique()
    multi_teacher = int((per_student > 1).sum())

    # THE decisive quantity: classes per school.
    if idclass and idschool:
        per_school = maths.groupby(idschool)[idclass].nunique()
        schools_multi = int((per_school >= 2).sum())
        pct_schools_multi = 100.0 * schools_multi / len(per_school)
        classes_in_multi = int(per_school[per_school >= 2].sum())
        mean_classes_per_school = float(per_school.mean())
    else:
        schools_multi = pct_schools_multi = classes_in_multi = pd.NA
        mean_classes_per_school = pd.NA

    class_size = maths.groupby(idclass)[idstud].nunique() if idclass else None

    # Replicate-weight structure lives in the student background file.
    bsg = read("bsg", iso)
    bcols = {c.upper(): c for c in bsg.columns}
    n_zones = (
        int(bsg[bcols["JKZONE"]].nunique()) if "JKZONE" in bcols else pd.NA
    )

    return {
        "system": iso,
        "students": n_students,
        "classes": n_classes,
        "schools": n_schools,
        "teachers": n_teachers,
        "mean_class_size": round(float(class_size.mean()), 1) if class_size is not None else pd.NA,
        "median_class_size": float(class_size.median()) if class_size is not None else pd.NA,
        "mean_classes_per_school": round(mean_classes_per_school, 2),
        "schools_with_2plus_classes": schools_multi,
        "pct_schools_2plus": round(pct_schools_multi, 1),
        "classes_in_multiclass_schools": classes_in_multi,
        "students_multi_teacher": multi_teacher,
        "jk_zones": n_zones,
    }


def main() -> int:
    only = sys.argv[1:] or systems()
    rows = []
    for iso in only:
        try:
            row = recon_one(iso)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {iso}: FAILED - {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(
            f"  {iso}: {row['students']:>6} students  "
            f"{row['classes']:>4} classes  {row['schools']:>4} schools  "
            f"{row['pct_schools_2plus']:>5}% schools with 2+ classes"
        )

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "design_recon.csv", index=False)

    print(f"\n{'=' * 70}\nTotals across {len(table)} systems")
    print(f"  students : {table['students'].sum():,}")
    print(f"  classes  : {table['classes'].sum():,}")
    print(f"  schools  : {table['schools'].sum():,}")
    print(
        f"  schools with 2+ sampled classes: "
        f"{table['schools_with_2plus_classes'].sum():,} "
        f"({100 * table['schools_with_2plus_classes'].sum() / table['schools'].sum():.1f}%)"
    )
    print(
        f"  classes usable for within-school contrasts: "
        f"{table['classes_in_multiclass_schools'].sum():,}"
    )
    print(f"\nWrote {OUT_DIR / 'design_recon.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
