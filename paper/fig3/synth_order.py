"""PC1/PC2 order in a simulation with a known ratio of component SDs (Fig 3A).

Two non-negative components with disjoint supports (20 genes each, N = 1000 cells,
D = 40) are planted with score sd ratio s1/s2 and Gaussian noise. As s1/s2 approaches 1
the two become interchangeable. For each ratio, nnPCA with two components is refitted in
the three arms of resample.py (cells resampled without strata) and per draw records:

  top_is_w1   the first extracted component is closer (|cos|) to the planted
              higher-variance component w1 than to w2
  ratio       variance of the component matched to w2 / that matched to w1, from each
              component's projection of the data before deflation

The result describes this simulation and matching rule, not nearly tied nnPCA solutions
in general.

Writes results/synth_order.csv, and results/synth_order_loadings.npz with the first extracted
component's loadings in every draw (key r<ratio>_<arm>, draws x genes) and the planted w1, w2.

Usage:
    python synth_order.py                 # B = 1000 resamples, S = 200 seeds
    python synth_order.py --B 40 --S 10   # quick check
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import resample as rs

N, D, NOISE, DATA_SEED = 1000, 40, 0.4, 3
RATIOS = (2.0, 1.6, 1.3, 1.15, 1.05, 1.0)


def planted(ratio: float, rng: np.random.Generator):
    w1 = np.zeros(D)
    w1[:20] = np.abs(rng.standard_normal(20)) + 0.2
    w2 = np.zeros(D)
    w2[20:] = np.abs(rng.standard_normal(20)) + 0.2
    w1 /= np.linalg.norm(w1)
    w2 /= np.linalg.norm(w2)
    X = (np.outer(rng.standard_normal(N) * ratio, w1) + np.outer(rng.standard_normal(N), w2)
         + NOISE * rng.standard_normal((N, D)))
    return X, w1, w2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1000, help="draws in the sampling and both arms")
    ap.add_argument("--S", type=int, default=200, help="draws in the seed arm")
    ap.add_argument("--seed", type=int, default=rs.SEED)
    a = ap.parse_args()

    rs.RESULTS.mkdir(exist_ok=True)
    out, loads = [], {}
    for ratio in RATIOS:
        X, w1, w2 = planted(ratio, np.random.default_rng(DATA_SEED))
        for arm in rs.ARMS:
            nb = a.S if arm == "seed" else a.B
            top, rat = np.zeros(nb), np.zeros(nb)
            first = np.zeros((nb, D), np.float32)
            for b, rows, start in rs.draws(arm, N, nb, a.seed):
                Xb = X[rows] - X[rows].mean(axis=0)
                W = rs.fit(Xb, 2, start(0))["rotation"]
                if W.shape[1] < 2:
                    W = np.pad(W, ((0, 0), (0, 2 - W.shape[1])))
                first[b] = W[:, 0]
                top[b] = float(rs.abs_cos(W[:, 0], w1) >= rs.abs_cos(W[:, 0], w2))
                i1, i2 = rs.match2(W, np.column_stack([w1, w2]))
                raw = (Xb @ W).std(axis=0, ddof=1)
                rat[b] = raw[i2] ** 2 / raw[i1] ** 2 if raw[i1] > 0 else np.nan
            loads[f"r{ratio:g}_{arm}"] = first
            p = float(top.mean())
            lo, hi = rs.ci(rat[np.isfinite(rat)])
            out.append({"ratio": ratio, "arm": arm, "draws": nb, "p_top_is_w1": p,
                        "p_mcse": rs.support_se(p, nb), "var_ratio_med": float(np.nanmedian(rat)),
                        "var_ratio_lo": lo, "var_ratio_hi": hi})
            print(f"  s1/s2={ratio:.2f} {arm:>8}: P(top is w1)={p:.3f}  var ratio {np.nanmedian(rat):.3f} "
                  f"[{lo:.3f}, {hi:.3f}]", flush=True)
    pd.DataFrame(out).to_csv(rs.RESULTS / "synth_order.csv", index=False)
    np.savez(rs.RESULTS / "synth_order_loadings.npz", w1=w1, w2=w2, **loads)   # w1, w2 are the same for every ratio


if __name__ == "__main__":
    main()
