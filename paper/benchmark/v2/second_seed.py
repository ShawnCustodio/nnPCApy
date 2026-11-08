"""Agreement with R nsprcomp under independent random starts (Supplementary Table S1B).

On the real-data inputs of a v2 run, compares with R nsprcomp's benchmark fit (the warm-up fit of
worker_r.R, seeded with SEED) the loadings of nnPCApy from the same run and of R nsprcomp refitted
with a second seed (second_seed_r.R, seed + 7919). For every input and k = 1, 2, 3 it records the
number of negative loadings, the cumulative adjusted variance tr(Q'CQ) relative to R's fit and the
largest principal angle between the k-component loading spans (analyze.py's definitions).
R's second seed shows how far R's own random starts move the fit. --old-run adds the negative
loadings of the solver in that run (the submitted nnPCApy 0.1.0).

Runs second_seed_r.R first unless --skip-r (R as Rterm; NNPCA_RTERM overrides the path).
Writes results/<run>/second_seed_cells.csv and results/<run>/second_seed_summary.csv.

Usage:
    python second_seed.py realdata_89c9f66_r1 --old-run realdata_head3b31cf9_t1_corder
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd
from analyze import adjusted_variance, principal_angles_deg
from common import HERE, RESULTS, RTERM
from make_inputs import input_path, load_input


def cell(tag, solver, k):
    return f"{tag}__{solver}__k{k}__t1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="a realdata run in results/")
    ap.add_argument("--old-run", help="run with the submitted solver, for its negative loadings")
    ap.add_argument("--skip-r", action="store_true", help="reuse loadings_seed2/ from an earlier call")
    a = ap.parse_args()
    run = RESULTS / a.run_id
    if not a.skip_r:
        subprocess.run([RTERM, "--no-echo", "--no-restore", "--no-save", f"--file={HERE / 'second_seed_r.R'}",
                        "--args", str(run)], check=True)

    s = pd.read_csv(run / "summary_cells.csv")
    s = s[(s["mode"] == "fit") & (s.threads == 1)]
    inputs = s[["suite", "tag", "dataset"]].drop_duplicates()
    old = None
    if a.old_run:
        o = pd.read_csv(RESULTS / a.old_run / "summary_cells.csv")
        old = o[(o["mode"] == "fit") & (o.solver == "nnpcapy") & (o.threads == 1)].set_index(["tag", "ncomp"])
    rows = []
    for _, i in inputs.iterrows():
        X = load_input(input_path(i.suite, i.tag))
        Xc = X - X.mean(axis=0)
        for k in (1, 2, 3):
            Wr = np.load(run / "loadings" / f"{cell(i.tag, 'nsprcomp', k)}.npy")
            ref = adjusted_variance(Xc, Wr)
            r = {"tag": i.tag, "dataset": i.dataset, "single_cell": i.dataset != "bulk", "k": k}
            for name, W in (("nnpcapy", np.load(run / "loadings" / f"{cell(i.tag, 'nnpcapy', k)}.npy")),
                            ("r_seed2", np.load(run / "loadings_seed2" / f"{cell(i.tag, 'nsprcomp', k)}.npy"))):
                r[f"neg_{name}"] = int((W < 0).sum())
                r[f"var_{name}"] = adjusted_variance(Xc, W) / ref
                r[f"angle_{name}"] = float(principal_angles_deg(W, Wr).max())
            if old is not None:
                r["neg_old"] = int(old.loc[(i.tag, k), "neg_loadings"])
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(run / "second_seed_cells.csv", index=False)

    out = []
    for k, g in d.groupby("k"):
        sc, bulk = g[g.single_cell], g[~g.single_cell]
        row = {"k": k, "inputs": len(g)}
        if old is not None:
            row["inputs_neg_old"] = int((g.neg_old > 0).sum())
        for name in ("nnpcapy", "r_seed2"):
            v = g[f"var_{name}"]
            row |= {f"inputs_neg_{name}": int((g[f"neg_{name}"] > 0).sum()), f"var_med_{name}": v.median(),
                    f"var_min_{name}": v.min(), f"var_max_{name}": v.max(),
                    f"angle_med_sc_{name}": sc[f"angle_{name}"].median(),
                    f"sc_over5_{name}": int((sc[f"angle_{name}"] > 5).sum()), "sc_inputs": len(sc),
                    f"angle_med_bulk_{name}": bulk[f"angle_{name}"].median()}
        out.append(row)
    summ = pd.DataFrame(out)
    summ.to_csv(run / "second_seed_summary.csv", index=False)
    for r in summ.itertuples(index=False):
        neg = f"{r.inputs_neg_old} -> {r.inputs_neg_nnpcapy}" if old is not None else f"{r.inputs_neg_nnpcapy}"
        print(f"k={r.k}: negative-loading inputs {neg} (R seed 2: {r.inputs_neg_r_seed2}); "
              f"variance/R nnPCApy {r.var_med_nnpcapy:.3f} [{r.var_min_nnpcapy:.3f}-{r.var_max_nnpcapy:.3f}], "
              f"R seed 2 {r.var_med_r_seed2:.3f} [{r.var_min_r_seed2:.3f}-{r.var_max_r_seed2:.3f}]; "
              f"angle (single-cell) nnPCApy {r.angle_med_sc_nnpcapy:.1f} ({r.sc_over5_nnpcapy}/{r.sc_inputs}), "
              f"R seed 2 {r.angle_med_sc_r_seed2:.1f} ({r.sc_over5_r_seed2}/{r.sc_inputs}); bulk median "
              f"{r.angle_med_bulk_nnpcapy:.1f} / {r.angle_med_bulk_r_seed2:.1f}")


if __name__ == "__main__":
    main()
