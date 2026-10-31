"""E and M scores by time point and the steps between time points (Fig 3C).

Cells from the five induction time points (0d, 8h, 1d, 3d, 7d) are scored with nnPCA
(one component each for the E and M signatures), the scores are standardized across
cells, and each time point's mean is taken. Three analyses:

1. Bootstrap: the three arms of resample.py, cells resampled within time point x culture.
   For each adjacent step, the bootstrap support frequency is the share of draws in which
   the mean goes up. Interval endpoints and support frequencies are also reported from
   the first 500 draws, to show they are stable in B.
2. Replication across cultures: from the full-data fit, each step's change in mean within
   each culture separately (Mix1-Mix4; Mix3 has no 1d cells), how many cultures agree in
   direction with the pooled change, and the mean and sd across cultures.
3. Leave one culture out: refit without each culture and recompute every step.

Writes results/time_means.csv, results/time_steps.csv, results/time_steps_by_culture.csv,
results/time_steps_loco.csv.

Usage:
    python time_steps.py                 # B = 2000 resamples, S = 200 seeds
    python time_steps.py --B 40 --S 10   # quick check
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import resample as rs

ORDER = ["0d", "8h", "1d", "3d", "7d"]
B_CHECK = 500


def standardized(s: np.ndarray) -> np.ndarray:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s - s.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="A549_TGFB1")
    ap.add_argument("--B", type=int, default=2000, help="draws in the sampling and both arms")
    ap.add_argument("--S", type=int, default=200, help="draws in the seed arm")
    ap.add_argument("--seed", type=int, default=rs.SEED)
    a = ap.parse_args()

    X, g2i, _genes, lib, meta = rs.load(a.dataset)
    keep = meta["Time"].isin(ORDER).to_numpy()
    X, lib, meta = X[keep], lib[keep], meta[keep]
    n = X.shape[0]
    times = meta["Time"].to_numpy()
    cult = rs.cultures(meta)
    groups = rs.strata(times, cult)

    em = rs.parse_gmt(rs.DATA / "EM_signature.gmt")
    names = {"E": "Panchy_et_al_E_signature", "M": "Panchy_et_al_M_signature"}
    idx = {k: np.array([g2i[g] for g in em[v] if g in g2i]) for k, v in names.items()}
    union = np.unique(np.concatenate(list(idx.values())))
    col = {k: np.searchsorted(union, v) for k, v in idx.items()}

    def scores(rows, start, nrestart=rs.NRESTART):
        Z = rs.zscores(X[np.ix_(rows, union)], lib[rows])
        return {k: standardized(rs.fit(Z[:, col[k]], 1, start(j), nrestart=nrestart)["x"][:, 0])
                for j, k in enumerate(("E", "M"))}

    def means(s, t):
        return np.array([s[t == tp].mean() for tp in ORDER])

    full = np.arange(n)
    point = scores(full, lambda k: np.random.SeedSequence([a.seed, 0, k]), rs.REF_NRESTART)
    pmean = {k: means(point[k], times) for k in point}
    print(f"{a.dataset}: {n} cells at {ORDER}, {len(groups)} time x culture strata; "
          f"E {len(idx['E'])} genes, M {len(idx['M'])} genes; B={a.B}, S={a.S}", flush=True)

    rs.RESULTS.mkdir(exist_ok=True)
    mean_rows, step_rows = [], []
    for arm in rs.ARMS:
        nb = a.S if arm == "seed" else a.B
        D = {k: np.zeros((nb, len(ORDER))) for k in point}
        t0 = time.perf_counter()
        for b, rows, start in rs.draws(arm, n, nb, a.seed, groups):
            s = scores(rows, start)
            for k in s:
                D[k][b] = means(s[k], times[rows])
            if (b + 1) % 200 == 0 or b + 1 == nb:
                el = time.perf_counter() - t0
                print(f"  {arm}: {b + 1}/{nb} ({el:.0f} s, {el / (b + 1):.2f} s/draw)", flush=True)
        np.savez(rs.RESULTS / f"time_steps_{arm}.npz", **D)

        for k, d in D.items():
            for j, tp in enumerate(ORDER):
                lo, hi = rs.ci(d[:, j])
                row = {"arm": arm, "score": k, "time": tp, "draws": nb, "point": pmean[k][j],
                       "ci_lo": lo, "ci_hi": hi}
                if nb > B_CHECK:
                    row["ci_lo_500"], row["ci_hi_500"] = rs.ci(d[:B_CHECK, j])
                mean_rows.append(row)
            for j in range(len(ORDER) - 1):
                delta = d[:, j + 1] - d[:, j]
                p = float((delta > 0).mean())
                lo, hi = rs.ci(delta)
                row = {"arm": arm, "score": k, "from": ORDER[j], "to": ORDER[j + 1], "draws": nb,
                       "point_delta": pmean[k][j + 1] - pmean[k][j], "support_up": p,
                       "support_mcse": rs.support_se(p, nb), "delta_lo": lo, "delta_hi": hi}
                if nb > B_CHECK:
                    row["support_up_500"] = float((delta[:B_CHECK] > 0).mean())
                    row["delta_lo_500"], row["delta_hi_500"] = rs.ci(delta[:B_CHECK])
                step_rows.append(row)
    pd.DataFrame(mean_rows).to_csv(rs.RESULTS / "time_means.csv", index=False)
    steps = pd.DataFrame(step_rows)
    steps.to_csv(rs.RESULTS / "time_steps.csv", index=False)

    # replication across cultures, from the full-data fit
    cul_rows = []
    for k, s in point.items():
        for j in range(len(ORDER) - 1):
            pooled = pmean[k][j + 1] - pmean[k][j]
            for c in np.unique(cult):
                a0 = (cult == c) & (times == ORDER[j])
                a1 = (cult == c) & (times == ORDER[j + 1])
                if a0.any() and a1.any():
                    d = s[a1].mean() - s[a0].mean()
                    cul_rows.append({"score": k, "from": ORDER[j], "to": ORDER[j + 1], "culture": c,
                                     "cells_from": int(a0.sum()), "cells_to": int(a1.sum()), "delta": d,
                                     "agrees": bool(np.sign(d) == np.sign(pooled))})
    pd.DataFrame(cul_rows).to_csv(rs.RESULTS / "time_steps_by_culture.csv", index=False)

    # leave one culture out
    loco_rows = []
    for c in np.unique(cult):
        rows = np.flatnonzero(cult != c)
        s = scores(rows, lambda k: np.random.SeedSequence([a.seed, 0, k]), rs.REF_NRESTART)
        for k in s:
            m = means(s[k], times[rows])
            for j in range(len(ORDER) - 1):
                loco_rows.append({"score": k, "from": ORDER[j], "to": ORDER[j + 1], "left_out": c,
                                  "delta": m[j + 1] - m[j], "pooled_delta": pmean[k][j + 1] - pmean[k][j]})
    pd.DataFrame(loco_rows).to_csv(rs.RESULTS / "time_steps_loco.csv", index=False)

    cul = pd.DataFrame(cul_rows).groupby(["score", "from", "to"], sort=False).agg(
        cultures=("delta", "size"), agree=("agrees", "sum"), mean=("delta", "mean"), sd=("delta", "std"))
    loco = pd.DataFrame(loco_rows).groupby(["score", "from", "to"], sort=False).agg(
        loco_min=("delta", "min"), loco_max=("delta", "max"))
    both = steps[steps.arm == "both"].set_index(["score", "from", "to"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(both[["point_delta", "support_up", "delta_lo", "delta_hi"]].join(cul).join(loco)
              .round(3).to_string())


if __name__ == "__main__":
    main()
