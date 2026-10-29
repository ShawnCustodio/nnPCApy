# Regenerates the R reference fixture for tests/test_nsprcomp.py.
#
#   Rscript tests/data/nsprcomp_r/make_fixture.R      (from the repo root)
#
# Needs R with nsprcomp 0.5.1-2. Writes, next to this script:
#   X.csv               120 x 10 data: 3 planted non-negative sparse components, noise,
#                       and unequal column scales (so scale. = TRUE matters)
#   stream.csv          the first 2000 values of rnorm() after set.seed(7): every random
#                       start nsprcomp draws in the fits below, in order
#   rotation_<cfg>.csv  nsprcomp(X, ncomp = 3, center = TRUE, ...) after set.seed(7), for
#   sdev_<cfg>.csv      cfg = nneg (nneg = TRUE), nneg_scale (nneg = TRUE, scale. = TRUE)
#                       and dense (nneg = FALSE)
# Values are written with 17 significant digits, so they parse back exactly.

library(nsprcomp)
stopifnot(packageVersion("nsprcomp") == "0.5.1.2")

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))

put <- function(x, name) {
  x <- as.matrix(x)
  lines <- apply(x, 1, function(r) paste(sprintf("%.17g", r), collapse = ","))
  con <- file(file.path(here, name), "wb")
  writeLines(lines, con, sep = "\n")
  close(con)
}

set.seed(1)
n <- 120
d <- 10
X <- matrix(0, n, d)
for (j in 1:3) {
  w <- numeric(d)
  s <- sample(d, 5)
  w[s] <- runif(5) + 0.2
  X <- X + outer(rnorm(n) * 1.5 / j, w)
}
X <- X + 0.5 * matrix(rnorm(n * d), n)
X <- X * rep(seq(0.5, 3, length.out = d), each = n)
put(X, "X.csv")

set.seed(7)
put(rnorm(2000), "stream.csv")

cfgs <- list(nneg = list(nneg = TRUE, scale. = FALSE),
             nneg_scale = list(nneg = TRUE, scale. = TRUE),
             dense = list(nneg = FALSE, scale. = FALSE))
for (cfg in names(cfgs)) {
  set.seed(7)
  r <- do.call(nsprcomp, c(list(X, ncomp = 3, center = TRUE), cfgs[[cfg]]))
  put(r$rotation, sprintf("rotation_%s.csv", cfg))
  put(r$sdev, sprintf("sdev_%s.csv", cfg))
}
