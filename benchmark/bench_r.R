# bench_r.R - R-side mirror of bench_python.py
#
# Reads the same _inputs/*.npy matrices the Python benchmark cached,
# calls nsprcomp() on each for ncomp = 1, 2, 3, times each call with
# proc.time() (5 trials + 1 discarded warm-up per cell), and writes
# r_timings.csv into the benchmark's results/ folder.
#
# Requirements:
#   install.packages(c("nsprcomp", "RcppCNPy"))
#
# Usage in RStudio: open this file, click Source (Ctrl+Shift+S)
# Usage from terminal: Rscript bench_r.R

suppressMessages({
  .libPaths(c("D:/R/library", .libPaths()))
  library(nsprcomp)
  library(RcppCNPy)
})

# Point this at the local nnPCApy/benchmark folder.
BENCH    <- "D:/work/nnPCApy/benchmark"
INPUTS   <- file.path(BENCH, "_inputs")
RESULTS  <- file.path(BENCH, "results")
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(file.path(INPUTS, "manifest.csv"))) {
  stop("manifest.csv not found in ", INPUTS,
       "\nRun `python datasets.py` first.")
}

manifest <- read.csv(file.path(INPUTS, "manifest.csv"), stringsAsFactors = FALSE)
N_TRIALS <- 5
NCOMPS   <- c(1, 2, 3)

iqr_simple <- function(xs) {
  if (length(xs) < 2) return(0)
  s <- sort(xs)
  s[floor(3 * length(s) / 4) + 1] - s[floor(length(s) / 4) + 1]
}

rows     <- list()
grand_t0 <- proc.time()[["elapsed"]]

cat(sprintf("Running R nnPCA benchmark on %d matrices x %d ncomps x %d trials\n",
            nrow(manifest), length(NCOMPS), N_TRIALS))

for (i in seq_len(nrow(manifest))) {
  m <- manifest[i, ]
  X <- npyLoad(file.path(INPUTS, paste0(m$tag, ".npy")))

  for (ncomp in NCOMPS) {
    # warm-up (discarded)
    try(nsprcomp(X, ncomp = ncomp, nneg = TRUE, center = TRUE, scale. = FALSE),
        silent = TRUE)

    times        <- numeric(N_TRIALS)
    mem_peaks_mb <- numeric(N_TRIALS)
    ok <- TRUE
    for (k in seq_len(N_TRIALS)) {
      gc(reset = TRUE, verbose = FALSE)
      t0 <- proc.time()[["elapsed"]]
      res <- tryCatch(
        nsprcomp(X, ncomp = ncomp, nneg = TRUE, center = TRUE, scale. = FALSE),
        error = function(e) e
      )
      times[k] <- proc.time()[["elapsed"]] - t0
      if (inherits(res, "error")) {
        ok <- FALSE
        message("  [", m$tag, " ncomp=", ncomp, "] trial failed: ",
                conditionMessage(res))
        break
      }
      g <- gc(verbose = FALSE)
      mem_peaks_mb[k] <- sum(g[, 6])
    }
    if (!ok) next

    rec <- data.frame(
      dataset      = m$dataset,
      gene_set     = m$gene_set,
      mode         = m$mode,
      n_rows       = as.integer(m$n_rows),
      n_cols       = as.integer(m$n_cols),
      ncomp        = ncomp,
      n_trials     = N_TRIALS,
      time_med_s   = median(times),
      time_min_s   = min(times),
      time_max_s   = max(times),
      time_iqr_s   = iqr_simple(times),
      heap_peak_mb = max(mem_peaks_mb),
      rss_delta_mb = NA_real_,
      stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1]] <- rec
    cat(sprintf("  %-55s ncomp=%d med=%7.2f ms  peak_heap=%6.2f MB\n",
                m$tag, ncomp, rec$time_med_s * 1000, rec$heap_peak_mb))
  }
}

elapsed_total <- proc.time()[["elapsed"]] - grand_t0
out <- do.call(rbind, rows)
write.csv(out, file.path(RESULTS, "r_timings.csv"), row.names = FALSE)

cat(sprintf("\nTotal R benchmark wall time: %.2fs\n", elapsed_total))
cat(sprintf("Wrote: %s  (%d rows)\n",
            file.path(RESULTS, "r_timings.csv"), nrow(out)))

writeLines(c(
  sprintf("total_elapsed_seconds: %.3f", elapsed_total),
  sprintf("n_matrices: %d", nrow(manifest)),
  sprintf("n_trials_per_cell: %d", N_TRIALS),
  sprintf("ncomps: %s", paste(NCOMPS, collapse = ",")),
  sprintf("R version: %s.%s", R.version$major, R.version$minor),
  sprintf("nsprcomp version: %s", as.character(packageVersion("nsprcomp")))
), con = file.path(RESULTS, "r_meta.txt"))
