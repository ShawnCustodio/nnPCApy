# worker_r.R - one R benchmark cell, in a fresh process launched by run.py.
#
# Usage: Rterm --no-echo --no-restore --no-save --file=worker_r.R --args <spec.json>
#
# spec keys: input (.npy, or .npz for sparse: R then reads the .mtx beside it), solver,
# ncomp, threads, mode ("fit" | "baseline" | "env"), seed, out_npy, min_reps,
# max_reps (0: time the warm-up fit only), min_time.
#
#   nsprcomp            CRAN nsprcomp, unmodified
#   nsprcomp_crossprod  same, with one change: inside the
#                       internal empca(), t(Xp) %*% (omega * z) -> crossprod(Xp, omega * z).
#                       Patched in this process only; the installed package is untouched.
#   rcppml              RcppML::pca(t(X), k, nonneg = c(TRUE, FALSE)) at the package's
#                       defaults (one random start, tol 1e-5, maxit 200), threads pinned.
#                       t(X) is formed once, outside the timed region (input layout).
#   rcppml_5start       the same fit from 5 random starts, keeping the one with the largest
#                       sum(d^2): the effort of nsprcomp's default nrestart = 5. RcppML has no
#                       restart option, so whole fits are restarted, not single components.
#
# mode "baseline" loads the same packages and input but does not fit; "env" prints the
# R environment for env_info.py. Prints one `RESULT {json}` line on stdout.

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(jsonlite)
  library(RcppCNPy)
})
spec <- fromJSON(commandArgs(trailingOnly = TRUE)[1])

if (spec$mode == "env") {
  pkgs <- c("nsprcomp", "RcppML", "bench", "RcppCNPy", "jsonlite", "Rcpp", "RcppEigen", "Matrix")
  ver <- vapply(pkgs, function(p)
    if (nzchar(system.file(package = p))) as.character(packageVersion(p)) else NA_character_, "")
  desc <- if (nzchar(system.file(package = "RcppML"))) packageDescription("RcppML") else list()
  cat("RESULT ", toJSON(list(
    R_version = R.version.string, platform = R.version$platform,
    BLAS = extSoftVersion()[["BLAS"]], LAPACK = La_library(), LAPACK_version = La_version(),
    blas_note = "empty BLAS/LAPACK path = R's bundled reference BLAS/LAPACK (Rblas.dll/Rlapack.dll)",
    packages = as.list(ver), RcppML_RemoteSha = desc$RemoteSha, RcppML_RemoteRepo = desc$RemoteRepo,
    RcppML_RemoteUsername = desc$RemoteUsername, libPaths = .libPaths()
  ), auto_unbox = TRUE, null = "null"), "\n", sep = "")
  quit(save = "no")
}

sparse_in <- grepl("\\.npz$", spec$input)
if (sparse_in) {
  suppressPackageStartupMessages(library(Matrix))
  X <- as(Matrix::readMM(sub("\\.npz$", ".mtx", spec$input)), "CsparseMatrix")
} else {
  X <- npyLoad(spec$input)
}
# Layout guard: RcppCNPy ignores the .npy fortran_order flag, so a column-major file would
# arrive scrambled. run.py sends the top-left corner as NumPy read it.
if (!is.null(spec$probe)) {
  p <- as.matrix(spec$probe)
  if (max(abs(as.matrix(X[seq_len(nrow(p)), seq_len(ncol(p))]) - p)) > 1e-12)
    stop("input layout mismatch: R does not see the matrix NumPy saved (", spec$input, ")")
}
k <- as.integer(spec$ncomp)
iters <- integer()

if (spec$solver %in% c("nsprcomp", "nsprcomp_crossprod")) {
  suppressPackageStartupMessages(library(nsprcomp))
  # nsprcomp needs a dense matrix; the copy counts towards the fit's memory, not the baseline
  if (sparse_in && spec$mode == "fit") X <- as.matrix(X)
  ns <- asNamespace("nsprcomp")
  if (spec$solver == "nsprcomp_crossprod") {
    f <- get("empca", ns)
    b <- deparse(body(f))
    hit <- grep("t(Xp) %*% (omega * z)", b, fixed = TRUE)
    if (length(hit) != 1) stop("crossprod patch: expected exactly one t(Xp) %*% (omega * z) in empca()")
    b[hit] <- sub("t(Xp) %*% (omega * z)", "crossprod(Xp, omega * z)", b[hit], fixed = TRUE)
    body(f) <- parse(text = b)[[1]]
    unlockBinding("empca", ns); assign("empca", f, envir = ns); lockBinding("empca", ns)
  }
  fit <- function() nsprcomp(X, ncomp = k, nneg = TRUE, center = TRUE, scale. = FALSE)
  fit_warm <- fit
  loadings <- function(r) r$rotation
  sdev <- function(r) r$sdev
  # EM iterations: read empca()'s loop counter `ii` on exit (warm-up run only).
  .it <- new.env()
  .it$v <- integer()
  trace_on <- function() invisible(capture.output(suppressMessages(trace(
    "empca", where = ns, print = FALSE,
    exit = quote(assign("v", c(get("v", envir = .GlobalEnv$.it), ii), envir = .GlobalEnv$.it))))))
  trace_off <- function() invisible(capture.output(suppressMessages(untrace("empca", where = ns))))
  get_iters <- function(r) .it$v
} else if (spec$solver %in% c("rcppml", "rcppml_5start")) {
  suppressPackageStartupMessages(library(RcppML))
  Xt <- t(X)
  fit_seeded <- function(seed) RcppML::pca(Xt, k = k, nonneg = c(TRUE, FALSE), seed = seed,
                                           threads = as.integer(spec$threads))
  if (spec$solver == "rcppml") {
    fit <- function() fit_seeded(NULL)
    fit_warm <- function() fit_seeded(as.integer(spec$seed))
  } else {
    # every start's iterations are kept, as nsprcomp's count includes all its restarts
    best_of <- function(seeds) {
      fits <- lapply(seeds, fit_seeded)
      best <- fits[[which.max(vapply(fits, function(r) sum(r@d^2), 0))]]
      best@misc$iters_per_factor <- unlist(lapply(fits, function(r) r@misc$iters_per_factor))
      best
    }
    fit <- function() best_of(sample.int(.Machine$integer.max, 5L))
    fit_warm <- function() best_of(as.integer(spec$seed) + 0:4)
  }
  loadings <- function(r) r@u
  sdev <- function(r) r@d / sqrt(nrow(X) - 1)
  trace_on <- function() NULL
  trace_off <- function() NULL
  get_iters <- function(r) r@misc$iters_per_factor
} else {
  stop("unknown R solver: ", spec$solver)
}

result <- list(solver = spec$solver, mode = spec$mode, n_rows = nrow(X), n_cols = ncol(X),
               blas = list(list(internal_api = "R reference BLAS", num_threads = 1L)))
if (spec$solver %in% c("rcppml", "rcppml_5start"))
  result$blas[[2]] <- list(internal_api = "RcppML OpenMP", num_threads = as.integer(spec$threads))

if (spec$mode == "fit") {
  suppressPackageStartupMessages(library(bench))
  # Untimed warm-up: seeded, instrumented, and the one whose loadings are saved.
  set.seed(spec$seed)
  trace_on()
  t_warm <- bench::hires_time()
  res <- fit_warm()
  t_warm <- as.numeric(bench::hires_time() - t_warm)
  trace_off()
  W <- loadings(res)
  npySave(spec$out_npy, matrix(as.double(W), nrow(W)))

  if (spec$max_reps == 0) {         # single-fit mode: the (instrumented) warm-up is the measurement
    result$times <- t_warm
  } else {
    set.seed(spec$seed + 1)
    bm <- bench::mark(fit(), min_iterations = spec$min_reps, max_iterations = spec$max_reps,
                      min_time = spec$min_time, check = FALSE, memory = FALSE, filter_gc = FALSE)
    result$times <- as.numeric(bm$time[[1]])
  }
  result$iters <- as.integer(get_iters(res))
  result$sdev <- as.numeric(sdev(res))
}

cat("RESULT ", toJSON(result, auto_unbox = TRUE, digits = NA), "\n", sep = "")
