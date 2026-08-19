"""
Single-cell-like ablation point (Python-only): times the four historical solver
variants V0 -> V3 on a seeded synthetic matrix that mirrors a Cook A549 gene-set
problem, and reports the speedup ladder behind Figure 2A.

This needs NO R. It reproduces the *Python* side of the ablation — V0 (naive
port) -> V3 (the shipping solver) — so a reviewer can verify the mechanism-by-
mechanism speedup, in particular the covariance-precompute step (V2 -> V3), which
is the one genuine algorithmic gain. The R reference bar in Fig 2A is already
measured and shipped in results/sc_ablation_D89_D200.csv for comparison.

Usage:
    python sc_ablation.py          # d=200 (matches the Fig 2A right panel)
    python sc_ablation.py 89       # d=89  (single-cell size, Fig 2A left panel)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from gen_synth import make_matrix   # same seeded generator as the main grid
from variants import VARIANTS       # V0_naive, V1_params, V2_twopass, V3_shipping

HERE = Path(__file__).resolve().parent
INP = HERE / "_inputs"
RES = HERE / "results"
INP.mkdir(parents=True, exist_ok=True)   # both are gitignored -> create on fresh clone
RES.mkdir(parents=True, exist_ok=True)

N, NCOMP, TRIALS, SEED = 12000, 1, 5, 20260804


def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tag = f"synth_n{N}_d{D}"

    rng = np.random.default_rng(SEED)
    X = make_matrix(N, D, rng)
    np.save(INP / f"{tag}.npy", X)
    print(f"generated {tag}: {X.shape}  ({TRIALS} trials, median ms)\n", flush=True)

    med: dict[str, float] = {}
    for name, fn in VARIANTS.items():
        fn(X, NCOMP)  # warm-up
        ts = []
        for _ in range(TRIALS):
            np.random.seed(12345)
            t0 = time.perf_counter(); fn(X, NCOMP); ts.append(time.perf_counter() - t0)
        med[name] = median(ts) * 1000
        print(f"  {name:12s} {med[name]:8.2f} ms", flush=True)

    pd.DataFrame([{"impl": k, "n_rows": N, "n_cols": D, "ncomp": NCOMP,
                   "time_med_ms": v} for k, v in med.items()]).to_csv(
        RES / "sc_ablation_py.csv", index=False)

    v0, v2, v3 = med["V0_naive"], med["V2_twopass"], med["V3_shipping"]
    print("\nSpeedup ladder (this run — ratios are machine-independent):")
    print(f"  V0 -> V3   Python naive -> shipping   : {v0 / v3:5.1f}x")
    print(f"  V2 -> V3   covariance precompute      : {v2 / v3:5.1f}x   <- the algorithmic gain")

    # Reference numbers from our run — includes the R bar the reviewer need not reproduce.
    ref_csv = RES / "sc_ablation_D89_D200.csv"
    if ref_csv.exists():
        ref = pd.read_csv(ref_csv).set_index("D")
        if D in ref.index:
            r = ref.loc[D]
            print(f"\nReference (our run, D={D}, from {ref_csv.name}):")
            print(f"  V0 -> V3   total          : {r.V0_naive / r.V3_shipping:5.1f}x")
            print(f"  V2 -> V3   covariance     : {r.V2_twopass / r.V3_shipping:5.1f}x")
            print(f"  R  -> V3   vs R package   : {r.R_nsprcomp / r.V3_shipping:5.1f}x   (R not run here)")
    print(f"\nCompare these V0-V3 bars/ratios against Figure 2A (D={D} panel).")
    print("Run fig2A_D200_explore.py to redraw our Fig 2A ablation from the reference CSV.")


if __name__ == "__main__":
    main()
