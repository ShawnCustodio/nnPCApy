"""Run a v2 benchmark suite: every (input, ncomp, threads, solver) cell in a fresh process.

Usage:
    python run.py --suite realdata                     # all solvers, ncomp 1-3, 1 thread
    python run.py --suite dsweep --ncomps 1 --threads 1,2
    python run.py --suite realdata --run-id NAME       # resume: finished cells are skipped
    python run.py --suite realdata --limit 2           # first 2 inputs only (smoke test)

Writes results/<run-id>/: env.json, cells.csv (one row per cell, appended as cells
finish), loadings/<cell>.npy (warm-up loadings), logs/ (worker stdout/stderr).
Solvers for one input and ncomp run in a shuffled order (seeded by the input tag), so
slow drift such as laptop thermal throttling does not favour one solver.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from common import (MAX_REPS, MIN_REPS, MIN_TIME_S, PY_SOLVERS, R_SOLVERS, RESULTS, SEED,
                    parse_result, run_child, thread_env, worker_cmd)
from make_inputs import input_path, manifest_path

ALL_SOLVERS = R_SOLVERS + PY_SOLVERS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=["realdata", "dsweep"])
    ap.add_argument("--solvers", default=",".join(ALL_SOLVERS))
    ap.add_argument("--ncomps", default="1,2,3")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N inputs")
    ap.add_argument("--min-reps", type=int, default=MIN_REPS)
    ap.add_argument("--max-reps", type=int, default=MAX_REPS)
    ap.add_argument("--min-time", type=float, default=MIN_TIME_S)
    ap.add_argument("--timeout", type=float, default=7200, help="seconds per cell")
    a = ap.parse_args()

    solvers = [s for s in a.solvers.split(",") if s]
    bad = set(solvers) - set(ALL_SOLVERS)
    if bad:
        sys.exit(f"unknown solvers: {sorted(bad)}; choose from {ALL_SOLVERS}")
    ncomps = [int(x) for x in a.ncomps.split(",")]
    threads_list = [int(x) for x in a.threads.split(",")]

    if not manifest_path(a.suite).exists():
        sys.exit(f"no inputs for suite {a.suite!r}: run `python make_inputs.py {a.suite}` first")
    manifest = pd.read_csv(manifest_path(a.suite))
    if a.limit:
        manifest = manifest.head(a.limit)

    run_id = a.run_id or f"{a.suite}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = RESULTS / run_id
    for sub in ("loadings", "logs", "specs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    if not (run_dir / "env.json").exists():
        from env_info import collect
        (run_dir / "env.json").write_text(json.dumps(collect(threads_list[0]), indent=2))
    cells_csv = run_dir / "cells.csv"
    done = set(pd.read_csv(cells_csv)["cell"]) if cells_csv.exists() else set()

    plan = []
    for _, row in manifest.iterrows():
        rng = random.Random(zlib.crc32(row["tag"].encode()))
        for s in solvers:                         # memory baselines: once per input and solver
            plan.append((row, s, 0, 1, "baseline"))
        for k in ncomps:
            for t in threads_list:
                order = solvers[:]
                rng.shuffle(order)
                plan += [(row, s, k, t, "fit") for s in order]
    todo = [p for p in plan if cell_id(*p) not in done]
    print(f"[{run_id}] {len(plan)} cells, {len(plan) - len(todo)} already done, {len(todo)} to run", flush=True)

    for i, (row, solver, k, t, mode) in enumerate(todo, 1):
        cid = cell_id(row, solver, k, t, mode)
        inp = input_path(a.suite, row["tag"])
        # top-left corner as NumPy reads it; worker_r.R stops if R sees anything else
        probe = np.asarray(np.load(inp, mmap_mode="r")[:3, :3]).tolist()
        spec = {"input": str(inp), "probe": probe, "solver": solver, "ncomp": max(k, 1),
                "threads": t, "mode": mode, "seed": SEED, "out_npy": str(run_dir / "loadings" / f"{cid}.npy"),
                "min_reps": a.min_reps, "max_reps": a.max_reps, "min_time": a.min_time}
        spec_path = run_dir / "specs" / f"{cid}.json"
        spec_path.write_text(json.dumps(spec))
        out = run_child(worker_cmd(solver, spec_path), thread_env(t), a.timeout, run_dir / "logs" / cid)
        res = parse_result(out["stdout"]) or {}
        times = _as_list(res.get("times"))
        rec = {
            "cell": cid, "suite": a.suite, "tag": row["tag"], "dataset": row["dataset"],
            "gene_set": row["gene_set"], "n_rows": int(row["n_rows"]), "n_cols": int(row["n_cols"]),
            "solver": solver, "lang": "R" if solver in R_SOLVERS else "Python", "ncomp": k, "threads": t,
            "mode": mode, "ok": out["returncode"] == 0 and bool(res),
            "n_reps": len(times),
            "t_med_s": pd.Series(times).median() if times else None,
            "t_min_s": min(times) if times else None, "t_max_s": max(times) if times else None,
            "times_s": json.dumps(times), "iters": json.dumps(_as_list(res.get("iters"))),
            "iters_total": sum(_as_list(res.get("iters"))) if res.get("iters") is not None else None,
            "sdev": json.dumps(_as_list(res.get("sdev"))), "blas": json.dumps(res.get("blas")),
            "peak_ws_mb": out["peak_ws_mb"], "peak_private_mb": out["peak_private_mb"],
            "process_wall_s": out["wall_s"],
            "error": "" if out["returncode"] == 0 and res else out["stderr"][-500:].replace("\n", " | "),
        }
        pd.DataFrame([rec]).to_csv(cells_csv, mode="a", header=not cells_csv.exists(), index=False)
        tmed = f"{rec['t_med_s'] * 1e3:9.2f} ms" if rec["t_med_s"] is not None else "        --"
        flag = "" if rec["ok"] else "  FAILED: " + rec["error"][:200]
        print(f"  [{i}/{len(todo)}] {cid:<80} {tmed}  peak {rec['peak_ws_mb']:7.1f} MB{flag}", flush=True)


def cell_id(row, solver: str, k: int, t: int, mode: str) -> str:
    return f"{row['tag']}__{solver}__baseline" if mode == "baseline" else f"{row['tag']}__{solver}__k{k}__t{t}"


def _as_list(x):
    # jsonlite's auto_unbox turns length-1 vectors into scalars
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


if __name__ == "__main__":
    main()
