# second_seed_r.R - refit R nsprcomp with a second seed on every real-data input of a v2 run.
#
# Usage: Rterm --no-echo --no-restore --no-save --file=second_seed_r.R --args <run dir> [offset]
#
# For every nsprcomp fit spec of the run (one thread; k = 1, 2, 3), fits nsprcomp exactly as the
# benchmark's warm-up fit does (worker_r.R), but seeded with spec$seed + offset (default 7919),
# and saves the loadings to <run dir>/loadings_seed2/ under the spec's file name. second_seed.py
# compares them with the benchmark fit, to show how much R's own random starts move the fit.
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(nsprcomp)
  library(RcppCNPy)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
run_dir <- args[1]
offset <- if (length(args) > 1) as.integer(args[2]) else 7919L
specs <- list.files(file.path(run_dir, "specs"), pattern = "__nsprcomp__k[123]__t1\\.json$", full.names = TRUE)
if (length(specs) == 0) stop("no nsprcomp fit specs in ", run_dir)
out_dir <- file.path(run_dir, "loadings_seed2")
dir.create(out_dir, showWarnings = FALSE)
for (f in specs) {
  spec <- fromJSON(f)
  X <- npyLoad(spec$input)
  if (!is.null(spec$probe)) {                    # same layout guard as worker_r.R
    p <- as.matrix(spec$probe)
    if (max(abs(X[seq_len(nrow(p)), seq_len(ncol(p))] - p)) > 1e-12)
      stop("input layout mismatch: R does not see the matrix NumPy saved (", spec$input, ")")
  }
  set.seed(spec$seed + offset)
  W <- nsprcomp(X, ncomp = as.integer(spec$ncomp), nneg = TRUE, center = TRUE, scale. = FALSE)$rotation
  npySave(file.path(out_dir, basename(spec$out_npy)), matrix(as.double(W), nrow(W)))
}
cat(sprintf("second_seed_r.R: %d fits (seed offset %d) -> %s\n", length(specs), offset, out_dir))
