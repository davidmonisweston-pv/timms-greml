# Data use policy

## The data is not in this repository

No TIMSS data file is committed here, and none should be. The repository
contains the code to fetch the data and the checksums to verify you got the
same bytes we did.

Two reasons:

**Licensing.** The IEA's Disclaimer and License Agreement states (clause 2.1)
that "commercial exploitation, distribution, redistribution, reproduction
and/or transmitting in any form or by any means ... of these publications,
restricted use items, translations thereof and/or part thereof are prohibited
unless written permission has been provided by IEA."

That clause is written around *publications and restricted-use items*, and is
not an unambiguous statement about the public-use data files. We could not find
text that clearly permits or clearly forbids re-hosting the public-use
`.sav` files. Rather than resolve that ambiguity in our own favour, the
repository does not redistribute them. Anyone who needs the data downloads it
from the IEA's own host, on the IEA's own terms.

**Size.** The grade-8 SPSS archive is 927 MB compressed and 2.8 GB extracted —
well past what belongs in git, and it would bloat the history permanently even
if later removed.

## Getting the data

```bash
uv run python -m timss_greml.download --grade 8
```

This downloads from `https://timss2023.org/`, verifies the size, records a
SHA-256 checksum in `data/checksums/manifest.json`, and extracts the archive
into `data/raw/`. The download resumes if interrupted.

If the checksum of a file you download differs from the one committed here, the
script fails loudly rather than continuing. That means either the upstream file
was revised or your download is corrupt — investigate before analysing, because
results computed on different versions of the source data are not comparable.

To re-verify files you already have:

```bash
uv run python -m timss_greml.download --verify-only
```

## Disk requirements

| Item | Size |
|---|---|
| `T23_Data_SPSS_G8.zip` | 927 MB |
| extracted `.sav` files (47 systems × 8 file types) | 2.8 GB |
| User Guide PDF | 12 MB |
| **total for grade 8** | **~3.7 GB** |

Grade 4 is comparable if you add it. The download script refuses to start if
free space is under roughly three times the archive size, since the archive has
to be unzipped alongside itself.

## Restricted-use variables

Some variables are not in the public files: exact birth and testing dates,
precise test-mode indicators (`ITMODE_x`), device type (`ITDEV`), and
event-log/clickstream data. These require a separate application to the IEA
Study Data Repository under a non-disclosure agreement. This project does not
use them, and its analyses are reproducible from the public files alone.

## Required citation

Any use of this data must cite:

> SOURCE: IEA's Trends in International Mathematics and Science Study —
> TIMSS 2023. Copyright © 2025 International Association for the Evaluation of
> Educational Achievement (IEA).

## Privacy

The public-use files are already de-identified by the IEA: no student, teacher
or school names, and no exact dates. No attempt should be made to re-identify
individuals, schools or classes, including by linking to external data. Nothing
in this repository's outputs reports cell sizes small enough to be disclosive;
aggregate results are reported at system level or above.
