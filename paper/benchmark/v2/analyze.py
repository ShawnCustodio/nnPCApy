"""Summarize a v2 run: speed and memory against R nsprcomp, and agreement of every solver's
loadings with nsprcomp's on every component.

Usage: python analyze.py <run-id>

Writes into results/<run-id>/:
  summary_cells.csv    one row per fitted cell, with
      speedup             nsprcomp median time / this solver's median time (same input, k, threads)
      mem_delta_mb        peak working set minus this solver's load-only baseline for the input
      neg_loadings        number of negative loadings (should be 0 for every non-negative PC)
      cos_pc{j}           |cosine| between this solver's PC j loading and nsprcomp's
      max_angle_deg       largest principal angle between the k-dimensional loading subspaces
      adj_var_ratio       cumulative adjusted variance tr(Q'CQ), Q an orthonormal basis of the
                          loadings and C the input covariance, relative to nsprcomp's; one
                          convention applied to every solver
  summary_overall.csv  per (suite, solver, ncomp, threads): total and median time, median
                       speedup, median memory delta, worst-case accuracy
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import RESULTS
from make_inputs import input_path

REF = "nsprcomp"


def adjusted_variance(Xc: np.ndarray, W: np.ndarray) -> float:
    W = W[:, np.linalg.norm(W, axis=0) > 0]
    if W.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(W)
    P = Xc @ Q
    return float((P * P).sum() / (Xc.shape[0] - 1))


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


def main(run_id: str) -> None:
    run_dir = RESULTS / run_id
    cells = pd.read_csv(run_dir / "cells.csv").drop_duplicates("cell", keep="last")
    base = cells[cells["mode"] == "baseline"].set_index(["tag", "solver"])["peak_ws_mb"]
    fits = cells[(cells["mode"] == "fit") & cells["ok"]].copy()
    ref = fits[fits["solver"] == REF].set_index(["tag", "ncomp", "threads"])

    rows = []
    X_cache: dict[str, np.ndarray] = {}
    for _, c in fits.iterrows():
        key = (c["tag"], c["ncomp"], c["threads"])
        r = c.to_dict()
        r["mem_delta_mb"] = c["peak_ws_mb"] - base.get((c["tag"], c["solver"]), np.nan)
        W = np.load(run_dir / "loadings" / f"{c['cell']}.npy")
        r["neg_loadings"] = int((W < 0).sum())
        r["min_loading"] = float(W.min())
        if key in ref.index:
            rc = ref.loc[key]
            r["speedup"] = rc["t_med_s"] / c["t_med_s"]
            Wr = np.load(run_dir / "loadings" / f"{rc['cell']}.npy")
            nr = np.linalg.norm(Wr, axis=0)
            nw = np.linalg.norm(W, axis=0)
            cos = np.abs((Wr * W).sum(0)) / np.where(nr * nw > 0, nr * nw, 1)
            for j, v in enumerate(cos, 1):
                r[f"cos_pc{j}"] = float(v)
            r["max_angle_deg"] = float(principal_angles_deg(W, Wr).max())
            if c["tag"] not in X_cache:              # keep one centred input in memory at a time
                X = np.load(input_path(c["suite"], c["tag"]))
                X_cache = {c["tag"]: X - X.mean(axis=0)}
            Xc = X_cache[c["tag"]]
            r["adj_var_ratio"] = adjusted_variance(Xc, W) / adjusted_variance(Xc, Wr)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "summary_cells.csv", index=False)

    g = out.groupby(["suite", "solver", "ncomp", "threads"])
    overall = g.agg(n_cells=("cell", "size"), total_t_med_s=("t_med_s", "sum"),
                    median_t_med_s=("t_med_s", "median"), median_speedup=("speedup", "median"),
                    median_mem_delta_mb=("mem_delta_mb", "median"), max_mem_delta_mb=("mem_delta_mb", "max"),
                    cells_with_neg=("neg_loadings", lambda s: int((s > 0).sum())),
                    max_angle_deg=("max_angle_deg", "max"), min_adj_var_ratio=("adj_var_ratio", "min"),
                    median_iters_total=("iters_total", "median")).reset_index()
    tot_ref = overall[overall["solver"] == REF].set_index(["suite", "ncomp", "threads"])["total_t_med_s"]
    overall["total_speedup"] = [tot_ref.get((s, k, t), np.nan) / v for s, k, t, v in
                                overall[["suite", "ncomp", "threads", "total_t_med_s"]].itertuples(index=False)]
    overall.to_csv(run_dir / "summary_overall.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(overall.round(3).to_string(index=False))
    env = json.loads((run_dir / "env.json").read_text())
    print(f"\nnnPCApy solver: HEAD {env['nnpcapy_git_head'][:10]}, src dirty={env['nnpcapy_solver_dirty']}; "
          f"R {env['r'].get('R_version')}, RcppML {str(env['r'].get('RcppML_RemoteSha'))[:10]}; "
          f"{env['cpu']}, AC power={env['on_ac_power']}")


if __name__ == "__main__":
    main(sys.argv[1])
