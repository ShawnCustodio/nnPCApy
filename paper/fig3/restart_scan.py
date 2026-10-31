"""How many restarts the Fig 3D screen needs (Methods), and the restart curve of Fig S2.

For every C2 set of the screen (c2_screen.py), on the full data: S single-restart nnPCA fits
(one component) from different random starts, each giving an sdev and a Spearman r with
the M score (M itself fitted with 50 restarts, as in the screen's point estimates). A fit
with r restarts keeps the restart with the largest sdev, so resampling the S single fits
estimates, for each r, the chance that a fit's r with M differs by more than 0.1 from that
of the best solution found. Also times one pass over all sets for r = 5, 20 and 50 restarts
(preprocessing excluded).

For Fig S2 the same is done with more seeds for REACTOME_TRANS_GOLGI_NETWORK_VESICLE_BUDDING:
the share of fits with r restarts that reach its higher-variance solution.

Writes results/restart_scan.csv (per set), results/restart_scan_summary.csv (per r) and
results/restart_curve_trans_golgi.csv.

Usage:
    python restart_scan.py              # S = 50; about 70 min
    python restart_scan.py --limit 300  # quick check
"""
from __future__ import annotations

import argparse
import time

import c2_screen  # first: sets one BLAS thread, as in the screen
import numpy as np
import pandas as pd
import resample as rs
from pc_order import spearman

_ranks, _sets = c2_screen._ranks, c2_screen._sets

RESTARTS = (1, 5, 10, 20, 30, 50)
GAP = 0.1                       # |r with M - r of the best solution| that counts as a miss
SIM = 2000
TRANS_GOLGI = "REACTOME_TRANS_GOLGI_NETWORK_VESICLE_BUDDING"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--S", type=int, default=50, help="single-restart fits per set")
    ap.add_argument("--seed", type=int, default=rs.SEED)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    W = _sets(a.limit)
    Z = np.empty(W["X"].shape, np.float32)
    for j in range(0, Z.shape[1], 1024):
        Z[:, j:j + 1024] = rs.zscores(W["X"][:, j:j + 1024], W["lib"])
    rM = _ranks(rs.fit(Z[:, W["colM"]], 1, np.random.SeedSequence([a.seed, 0, 0]), rs.REF_NRESTART)["x"][:, 0])
    K = len(W["names"])
    sd, r = np.zeros((K, a.S)), np.zeros((K, a.S))
    t0 = time.perf_counter()
    for k, c in enumerate(W["col"]):
        for s in range(a.S):
            f = rs.fit(Z[:, c], 1, np.random.SeedSequence([a.seed, 5, k, s]), nrestart=1)
            sd[k, s] = f["sdev"][0]
            x = _ranks(f["x"][:, 0])
            d = np.sqrt((x @ x) * (rM @ rM))
            r[k, s] = 0.0 if d == 0 else x @ rM / d
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{K} sets ({time.perf_counter() - t0:.0f} s)", flush=True)

    best = sd.argmax(1)
    r_best = r[np.arange(K), best]
    rng = np.random.default_rng(a.seed)
    per_set = {"gene_set": W["names"], "n_genes": W["sizes"], "r_best": r_best,
               "single_fit_gap_share": (np.abs(r - r_best[:, None]) > GAP).mean(1)}
    for nr in RESTARTS:
        idx = rng.integers(0, a.S, (SIM, nr))
        miss = np.empty(K)
        for k in range(K):
            pick = idx[np.arange(SIM), sd[k][idx].argmax(1)]
            miss[k] = (np.abs(r[k][pick] - r_best[k]) > GAP).mean()
        per_set[f"p_miss_{nr}"] = miss
    per_set = pd.DataFrame(per_set)
    per_set.to_csv(rs.RESULTS / "restart_scan.csv", index=False)

    rows = []
    for nr in RESTARTS:
        m = per_set[f"p_miss_{nr}"]
        rows.append({"nrestart": nr, "sets": K, "sets_p_miss_over_5pct": int((m > 0.05).sum()),
                     "sets_p_miss_over_1pct": int((m > 0.01).sum()), "expected_misses_per_pass": float(m.sum())})
    sample = rng.choice(K, min(300, K), replace=False)
    for row in rows:
        if row["nrestart"] in (5, 20, 50):
            t0 = time.perf_counter()
            for k in sample:
                rs.fit(Z[:, W["col"][k]], 1, 1, nrestart=row["nrestart"])
            row["s_per_pass"] = (time.perf_counter() - t0) / len(sample) * K
    summary = pd.DataFrame(rows)
    summary.to_csv(rs.RESULTS / "restart_scan_summary.csv", index=False)
    print(summary.round(1).to_string(index=False))

    if TRANS_GOLGI in W["names"]:
        k = W["names"].index(TRANS_GOLGI)
        Zt = Z[:, W["col"][k]]
        ref = rs.fit(Zt, 1, np.random.SeedSequence([a.seed, 0, 1]), nrestart=rs.REF_NRESTART)["sdev"][0]
        curve = []
        for nr in RESTARTS:
            fits = [rs.fit(Zt, 1, np.random.SeedSequence([a.seed, 6, nr, s]), nrestart=nr) for s in range(100)]
            sds = np.array([f["sdev"][0] for f in fits])
            rr = np.array([spearman(f["x"][:, 0], rM) for f in fits])
            curve.append({"nrestart": nr, "fits": 100, "share_best": float((sds >= ref * (1 - 1e-3)).mean()),
                          "share_r_below_0": float((rr < 0).mean())})
        pd.DataFrame(curve).to_csv(rs.RESULTS / "restart_curve_trans_golgi.csv", index=False)
        print(pd.DataFrame(curve).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
