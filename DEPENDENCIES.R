# Explicit R dependency manifest for renv.
#
# renv's default "implicit" snapshot mode scans the project's R sources for
# library() calls. That makes the lockfile churn as code moves around, so this
# project uses "explicit" mode instead: renv snapshots exactly what is listed
# here. Add a package here before using it.
#
# Rebuild the environment from the lockfile with:  Rscript -e 'renv::restore()'

library(data.table)   # fast IO and grouped operations on the TIMSS files
library(haven)        # read SPSS .sav, preserving value labels and user NAs
library(jsonlite)     # exchange results with the Python side of the pipeline

library(survey)       # design-based estimation; used to validate weight handling
library(intsvy)       # reference implementation for TIMSS plausible values and
                      # JK2 replicate weights - we reproduce the published TIMSS
                      # country means with it as a correctness check

library(lme4)         # multilevel benchmark models
library(glmnet)       # cross-fitted elastic net benchmark
library(metafor)      # random-effects meta-analysis across education systems

library(sommer)       # REML with arbitrary user-supplied covariance matrices
library(rrBLUP)       # independent REML implementation, used to cross-check sommer
library(regress)      # third REML implementation; arbitrary V matrices
library(Matrix)       # sparse/structured matrix algebra
library(RSpectra)     # partial eigendecomposition of the practice kernel
