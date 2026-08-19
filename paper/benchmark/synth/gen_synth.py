"""
Synthetic-data generator for the Panel 1 speed benchmark.

Produces matrices with planted non-negative, sparse principal structure so the
EM solver converges in a realistic number of iterations (as it does on real
gene-set data), then column-standardises them the same way the real benchmark
hands data to nsprcomp (center=True, scale_=False on already gene-scaled data).

Each matrix is written twice, byte-for-byte the same numbers:
  * <tag>.npy  -- read by the Python variants (bench_py_synth.py)
  * <tag>.csv  -- read by the R reference (bench_r_synth.R), no RcppCNPy needed

A manifest.csv lists every matrix with its shape.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "_inputs"
OUT.mkdir(exist_ok=True)

# (n samples/rows, d features/cols) grid. n varies down columns, d across.
N_GRID = [100, 500, 2000, 8000]
D_GRID = [20, 50, 100, 200]

N_LATENT = 3          # planted non-negative components
SUPP_FRAC = 0.4       # fraction of features each latent component loads on
NOISE_SD = 0.5        # additive Gaussian noise level (post-structure)
SEED = 20260803


def _standardise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def make_matrix(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n x d matrix with N_LATENT planted non-negative sparse loadings + noise."""
    X = np.zeros((n, d), dtype=np.float64)
    k = max(3, int(round(SUPP_FRAC * d)))
    for j in range(N_LATENT):
        supp = rng.choice(d, size=k, replace=False)
        w = np.zeros(d)
        w[supp] = np.abs(rng.standard_normal(k)) + 0.2   # strictly positive loads
        w /= np.linalg.norm(w)
        z = rng.standard_normal(n) * (1.0 / (j + 1))     # decreasing variance
        X += np.outer(z, w)
    X += NOISE_SD * rng.standard_normal((n, d))
    return _standardise(X)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in N_GRID:
        for d in D_GRID:
            X = make_matrix(n, d, rng)
            tag = f"synth_n{n}_d{d}"
            np.save(OUT / f"{tag}.npy", X)
            # plain CSV (no header/index) so base-R read.csv reproduces it exactly
            np.savetxt(OUT / f"{tag}.csv", X, delimiter=",")
            rows.append({"tag": tag, "n_rows": n, "n_cols": d})
            print(f"  wrote {tag}  ({n} x {d})", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"\nManifest: {len(rows)} matrices -> {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
