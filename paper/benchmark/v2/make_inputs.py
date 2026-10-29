"""Build the input matrices for the v2 benchmark suites.

  realdata  the 47 matrices of the committed v1 benchmark (Panchy E/M and C2 pathways on
            the bulk panel and the three Cook A549 datasets), built by ../datasets.py into
            ../_inputs/. Reused unchanged so v1 and v2 numbers refer to the same inputs.
  dsweep    synthetic matrices with planted non-negative sparse structure
            (../synth/gen_synth.make_matrix), fixed N and increasing feature count D,
            to show where the covariance formulation stops paying off.

Usage:
    python make_inputs.py                # both suites
    python make_inputs.py realdata       # one suite
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import BENCH, INPUTS

DSWEEP_N = 5000
DSWEEP_D = [100, 200, 500, 1000, 2000, 5000]
DSWEEP_SEED = 20261029


def manifest_path(suite: str):
    return {"realdata": BENCH / "_inputs" / "manifest.csv",
            "dsweep": INPUTS / "dsweep" / "manifest.csv"}[suite]


def input_path(suite: str, tag: str):
    return {"realdata": BENCH / "_inputs", "dsweep": INPUTS / "dsweep"}[suite] / f"{tag}.npy"


def make_realdata() -> None:
    if manifest_path("realdata").exists():
        print(f"[realdata] reusing {manifest_path('realdata')}")
    else:
        sys.path.insert(0, str(BENCH))
        from datasets import prepare_all
        prepare_all()
    # Caches built before datasets.py saved in C order hold column-major files, which
    # RcppCNPy::npyLoad reads as a scrambled matrix. Rewrite them; values are unchanged.
    fixed = 0
    for tag in pd.read_csv(manifest_path("realdata"))["tag"]:
        p = input_path("realdata", tag)
        X = np.load(p)
        if not X.flags["C_CONTIGUOUS"]:
            np.save(p, np.ascontiguousarray(X))
            fixed += 1
    if fixed:
        print(f"[realdata] rewrote {fixed} column-major inputs in C order")


def make_dsweep() -> None:
    sys.path.insert(0, str(BENCH / "synth"))
    from gen_synth import make_matrix
    out = INPUTS / "dsweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DSWEEP_D:
        tag = f"synth_n{DSWEEP_N}_d{d}"
        path = out / f"{tag}.npy"
        if not path.exists():
            X = make_matrix(DSWEEP_N, d, np.random.default_rng([DSWEEP_SEED, d]))
            np.save(path, X)
        rows.append({"tag": tag, "dataset": "synthetic", "gene_set": f"d{d}", "mode": "dsweep",
                     "n_rows": DSWEEP_N, "n_cols": d})
        print(f"[dsweep] {tag}")
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)


if __name__ == "__main__":
    suites = sys.argv[1:] or ["realdata", "dsweep"]
    for s in suites:
        {"realdata": make_realdata, "dsweep": make_dsweep}[s]()
