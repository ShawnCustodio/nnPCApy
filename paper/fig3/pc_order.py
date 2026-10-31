"""PC1/PC2 order and the two-component fit under resampling (Fig 3B, Fig S2).

For each gene set (the E and M signatures and the 12 C2 EMT sets of Fig 3B, and the 4 sets
of Fig S2) in the Cook A549 TGF-beta data, nnPCA with two components is refitted in the
three arms of resample.py. Per draw it records:

  sd1, sd2       nsprcomp sdev by extraction order; the second is measured after deflation
  raw1, raw2     sd of each component's projection of the data before deflation
  order_kept     the draw's components match the reference's in order (summed |cos|)
  ratio          sdev² of the component matched to reference PC2 / that matched to PC1
  ratio_raw      the same with raw1, raw2
  angle          largest principal angle (degrees) between the draw's and the reference's
                 two-component loading spans
  cos1           |cos| between the component matched to reference PC1 and reference PC1
  flag1          nsprcomp marks PC1 as a local optimum (PC2 explains more of the data)

Refits use nsprcomp's default of 5 restarts (--nrestart), the setting Fig S2 is about.
The reference is the full-data fit with 50 restarts. For the Fig S2 sets it also records the
Spearman r between the set's one-component score and the M score in each draw (both
refitted on the same cells).

Writes results/pc_order_<arm>.npz (draws) and results/pc_order_summary.csv,
results/pc_order_corr_summary.csv.

Usage:
    python pc_order.py                      # B = 1000 resamples, S = 200 seeds
    python pc_order.py --B 20 --S 10 --sets Panchy_et_al_M_signature,HOUSTIS_ROS   # smoke test
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import resample as rs
from scipy.stats import rankdata

GMTS = ("EM_signature.gmt", "filtered.c2.gmt", "degeneracy_demo.gmt")
CORR_SETS = ("HOUSTIS_ROS", "REACTOME_TRANS_GOLGI_NETWORK_VESICLE_BUDDING")
M_SET = "Panchy_et_al_M_signature"
METRICS = ("sd1", "sd2", "raw1", "raw2", "order_kept", "ratio", "ratio_raw", "angle", "cos1", "flag1")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = rankdata(a) - (len(a) + 1) / 2
    b = rankdata(b) - (len(b) + 1) / 2
    d = np.sqrt((a @ a) * (b @ b))
    return 0.0 if d == 0 else float(a @ b / d)


def two(r: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Loadings and sdev padded to two components (nsprcomp stops early on rank-1 data),
    and whether PC1 is marked as a local optimum."""
    W, sd = r["rotation"], np.asarray(r["sdev"], float)
    k = W.shape[1]
    if k < 2:
        W = np.pad(W, ((0, 0), (0, 2 - k)))
        sd = np.pad(sd, (0, 2 - len(sd)))
    return W[:, :2], sd[:2], bool(r["suboptimal"][0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="A549_TGFB1")
    ap.add_argument("--B", type=int, default=1000, help="draws in the sampling and both arms")
    ap.add_argument("--S", type=int, default=200, help="draws in the seed arm")
    ap.add_argument("--seed", type=int, default=rs.SEED)
    ap.add_argument("--sets", default="", help="comma-separated subset of gene sets")
    ap.add_argument("--min-genes", type=int, default=5)
    ap.add_argument("--nrestart", type=int, default=5, help="restarts per refit (5 = nsprcomp's default)")
    a = ap.parse_args()

    X, g2i, _genes, lib, meta = rs.load(a.dataset)
    n = X.shape[0]
    groups = rs.strata(meta["Time"].to_numpy(), rs.cultures(meta))

    sets = {}
    for f in GMTS:
        for name, gg in rs.parse_gmt(rs.DATA / f).items():
            idx = np.array([g2i[g] for g in gg if g in g2i])
            if len(idx) >= a.min_genes:
                sets[name] = idx
    if a.sets:
        keep = set(a.sets.split(","))
        sets = {k: v for k, v in sets.items() if k in keep or k == M_SET}
    names = list(sets)
    corr_names = [s for s in CORR_SETS if s in sets]
    union = np.unique(np.concatenate(list(sets.values())))
    col = {name: np.searchsorted(union, idx) for name, idx in sets.items()}

    def zscores(rows):
        return rs.zscores(X[np.ix_(rows, union)], lib[rows])

    Z = zscores(np.arange(n))
    ref = {}
    for k, name in enumerate(names):
        r = rs.fit(Z[:, col[name]], 2, np.random.SeedSequence([a.seed, 0, k]), nrestart=rs.REF_NRESTART)
        ref[name] = two(r)[:2]
    print(f"{a.dataset}: {n} cells, {len(groups)} time x culture strata, {len(names)} gene sets, "
          f"{len(union)} genes; B={a.B}, S={a.S}, nrestart={a.nrestart}", flush=True)

    rs.RESULTS.mkdir(exist_ok=True)
    rows_out, corr_out = [], []
    for arm in rs.ARMS:
        nb = a.S if arm == "seed" else a.B
        D = {(name, m): np.zeros(nb) for name in names for m in METRICS}
        C = {name: np.zeros(nb) for name in corr_names}
        t0 = time.perf_counter()
        for b, rows, start in rs.draws(arm, n, nb, a.seed, groups):
            Zb = Z if arm == "seed" else zscores(rows)
            for k, name in enumerate(names):
                Zs = Zb[:, col[name]]
                W, sd, flag1 = two(rs.fit(Zs, 2, start(k), nrestart=a.nrestart))
                raw = (Zs @ W).std(axis=0, ddof=1)
                Wr = ref[name][0]
                i1, i2 = rs.match2(W, Wr)
                v = {"sd1": sd[0], "sd2": sd[1], "raw1": raw[0], "raw2": raw[1],
                     "order_kept": float((i1, i2) == (0, 1)),
                     "ratio": sd[i2] ** 2 / sd[i1] ** 2 if sd[i1] > 0 else np.nan,
                     "ratio_raw": raw[i2] ** 2 / raw[i1] ** 2 if raw[i1] > 0 else np.nan,
                     "angle": rs.max_angle_deg(W, Wr), "cos1": rs.abs_cos(W[:, i1], Wr[:, 0]),
                     "flag1": float(flag1)}
                for m in METRICS:
                    D[name, m][b] = v[m]
            if corr_names:
                sM = rs.fit(Zb[:, col[M_SET]], 1, start(1000), nrestart=a.nrestart)["x"][:, 0]
                for j, name in enumerate(corr_names):
                    s = rs.fit(Zb[:, col[name]], 1, start(1001 + j), nrestart=a.nrestart)["x"][:, 0]
                    C[name][b] = spearman(s, sM)
            if (b + 1) % 50 == 0 or b + 1 == nb:
                el = time.perf_counter() - t0
                print(f"  {arm}: {b + 1}/{nb} ({el:.0f} s, {el / (b + 1):.2f} s/draw)", flush=True)
        np.savez(rs.RESULTS / f"pc_order_{arm}.npz",
                 **{f"{name}|{m}": D[name, m] for name in names for m in METRICS},
                 **{f"{name}|r_M": C[name] for name in corr_names})

        for name in names:
            d = {m: D[name, m] for m in METRICS}
            p = float(d["order_kept"].mean())
            row = {"gene_set": name, "n_genes": len(sets[name]), "arm": arm, "draws": nb,
                   "ref_sd1": ref[name][1][0], "ref_sd2": ref[name][1][1],
                   "p_order_kept": p, "p_order_kept_mcse": rs.support_se(p, nb),
                   "p_flag1": float(d["flag1"].mean())}
            for m in ("ratio", "ratio_raw", "angle", "cos1", "sd1", "sd2", "raw1", "raw2"):
                x = d[m][np.isfinite(d[m])]
                lo, hi = rs.ci(x)
                row |= {f"{m}_med": float(np.median(x)), f"{m}_lo": lo, f"{m}_hi": hi}
            row["ratio_lo_mcse"], row["ratio_hi_mcse"] = rs.ci_mc_se(d["ratio"][np.isfinite(d["ratio"])])
            rows_out.append(row)
        for name in corr_names:
            x = C[name]
            lo, hi = rs.ci(x)
            corr_out.append({"gene_set": name, "arm": arm, "draws": nb, "r_med": float(np.median(x)),
                             "r_lo": lo, "r_hi": hi, "frac_below_0": float((x < 0).mean())})

    summ = pd.DataFrame(rows_out)
    summ.to_csv(rs.RESULTS / "pc_order_summary.csv", index=False)
    pd.DataFrame(corr_out).to_csv(rs.RESULTS / "pc_order_corr_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(summ[["gene_set", "arm", "p_order_kept", "ratio_med", "ratio_lo", "ratio_hi",
                    "angle_med", "angle_hi"]].round(3).to_string(index=False))
        if corr_out:
            print(pd.DataFrame(corr_out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
