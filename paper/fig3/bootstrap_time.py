"""Absolute run times of the Fig 3C and Fig 3D bootstraps in nnPCApy and in R nsprcomp.

Both run on one core (every BLAS/OpenMP pool limited to one thread) with 20 random restarts per
fit, as in the bootstraps. Fig 3C: one draw (resampling, z-scoring of the E and M genes, the two
fits) is timed in nnPCApy; R nsprcomp fits E and M on the full data. Fig 3D: nnPCApy and R
nsprcomp fit M and --sets randomly chosen C2 sets of the screen; one draw is M plus the mean set
time times the number of sets, plus the z-scoring of all screened genes. Z-scoring is the same
preprocessing in both, so its Python time is added to R's too. Totals are these per-draw times
times the number of draws: single-core figures (c2_screen.py itself runs in worker processes).

Inputs for R are written to results/bootstrap_time/; R is run as Rterm (NNPCA_RTERM overrides
the path). Writes results/bootstrap_time.csv. Times follow the CPU clock: run on AC power with
other programs closed (on battery with background load, R took 2-4 times as long).

Usage:
    python bootstrap_time.py              # a few minutes
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"               # one thread, here and in the R child process

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import c2_screen  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import resample as rs  # noqa: E402

RTERM = os.environ.get("NNPCA_RTERM", r"D:\R\bin\x64\Rterm.exe" if os.name == "nt" else "R")
ORDER = ["0d", "8h", "1d", "3d", "7d"]
NRESTART, DRAWS_3C, DRAWS_3D = 20, 2000, 300
R_CODE = r"""
library(nsprcomp)
library(RcppCNPy)
args <- commandArgs(trailingOnly = TRUE)
files <- sort(list.files(args[1], pattern = "\\.npy$"))
set.seed(1)
t <- sapply(files, function(f) {
  Z <- npyLoad(file.path(args[1], f))
  median(sapply(seq_len(as.integer(args[2])), function(i) system.time(
    nsprcomp(Z, ncomp = 1, nneg = TRUE, center = TRUE, scale. = FALSE, nrestart = 20))[["elapsed"]]))
})
write.csv(data.frame(file = files, r_s = t), file.path(args[1], "r_times.csv"), row.names = FALSE)
"""


def r_times(folder, reps: int) -> pd.Series:
    """R nsprcomp time (median of `reps`) for every .npy input in `folder`, by file name."""
    script = folder / "time_fits.R"
    script.write_text(R_CODE)
    subprocess.run([RTERM, "--no-echo", "--no-restore", "--no-save", f"--file={script}", "--args", str(folder),
                    str(reps)], check=True)
    return pd.read_csv(folder / "r_times.csv").set_index("file").r_s


def clock(f, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fig3c(out) -> dict:
    X, g2i, _genes, lib, meta = rs.load("A549_TGFB1")
    keep = meta["Time"].isin(ORDER).to_numpy()
    X, lib, meta = X[keep], lib[keep], meta[keep]
    groups = rs.strata(meta["Time"].to_numpy(), rs.cultures(meta))
    em = rs.parse_gmt(rs.DATA / "EM_signature.gmt")
    idx = {k: np.array([g2i[g] for g in em[f"Panchy_et_al_{k}_signature"] if g in g2i]) for k in ("E", "M")}
    union = np.unique(np.concatenate(list(idx.values())))
    col = {k: np.searchsorted(union, v) for k, v in idx.items()}
    Z = rs.zscores(X[:, union], lib)
    for k in col:
        np.save(out / f"{k}.npy", np.ascontiguousarray(Z[:, col[k]]))

    rng = np.random.default_rng(1)

    def zscore():
        rows = rs.resample_rows(X.shape[0], rng, groups)
        return rs.zscores(X[np.ix_(rows, union)], lib[rows])

    def draw():
        Zb = zscore()
        for j, k in enumerate(col):
            rs.fit(Zb[:, col[k]], 1, np.random.SeedSequence([1, j]), NRESTART)

    py_draw, zs = clock(draw, 20), clock(zscore, 20)
    r = r_times(out, 5)
    return {"analysis": "Fig 3C", "draws": DRAWS_3C, "fits_per_draw": 2, "cells": X.shape[0],
            "nnpcapy_s_per_draw": py_draw, "r_s_per_draw": zs + r.sum()}


def fig3d(out, n_sets: int) -> dict:
    W = c2_screen._sets()

    def zscore():
        Z = np.empty(W["X"].shape, np.float32)
        for j in range(0, Z.shape[1], 1024):
            Z[:, j:j + 1024] = rs.zscores(W["X"][:, j:j + 1024], W["lib"])
        return Z

    zs = clock(zscore, 3)
    Z = zscore()
    pick = np.random.default_rng(11).choice(len(W["col"]), n_sets, replace=False)
    blocks = {"M": W["colM"]} | {f"s{i:03d}": W["col"][k] for i, k in enumerate(pick)}
    py = {}
    for name, c in blocks.items():
        Zk = np.ascontiguousarray(Z[:, c], dtype=np.float64)
        np.save(out / f"{name}.npy", Zk)
        py[name] = clock(lambda Zk=Zk: rs.fit(Zk, 1, np.random.SeedSequence([5]), NRESTART), 1)
    r = {k.removesuffix(".npy"): v for k, v in r_times(out, 1).items()}
    n_all = len(W["col"])

    def set_mean(t):
        return float(np.mean([v for k, v in t.items() if k != "M"]))

    def per_draw(t):
        return zs + t["M"] + set_mean(t) * n_all

    return {"analysis": "Fig 3D", "draws": DRAWS_3D, "fits_per_draw": n_all + 1, "cells": W["X"].shape[0],
            "nnpcapy_s_per_draw": per_draw(py), "r_s_per_draw": per_draw(r),
            "nnpcapy_s_per_set": set_mean(py), "r_s_per_set": set_mean(r)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", type=int, default=100, help="C2 sets timed for Fig 3D")
    a = ap.parse_args()

    rows = []
    for name, f in (("3c", fig3c), ("3d", lambda o: fig3d(o, a.sets))):
        out = rs.RESULTS / "bootstrap_time" / name
        out.mkdir(parents=True, exist_ok=True)
        rows.append(f(out))
    d = pd.DataFrame(rows)
    d["nnpcapy_total_h"] = d.nnpcapy_s_per_draw * d.draws / 3600
    d["r_total_h"] = d.r_s_per_draw * d.draws / 3600
    d["ratio"] = d.r_total_h / d.nnpcapy_total_h
    d.to_csv(rs.RESULTS / "bootstrap_time.csv", index=False)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
