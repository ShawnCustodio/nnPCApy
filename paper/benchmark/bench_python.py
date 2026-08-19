"""Python-side nnPCA benchmark with per-dataset chunking + incremental CSV.

Usage:
  python bench_python.py                        # run all datasets
  python bench_python.py --datasets=bulk,...    # run only listed datasets, append to CSV
"""
from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[0])
# Make `nnpcapy` importable when running from a checkout without `pip install`.
sys.path.insert(0, str(REPO_ROOT / "src"))

from nnpcapy import nsprcomp

if not (HERE / "_inputs" / "manifest.csv").exists():
    sys.path.insert(0, str(HERE))
    from datasets import prepare_all
    prepare_all()

MANIFEST = pd.read_csv(HERE / "_inputs" / "manifest.csv")
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

N_TRIALS = 5
NCOMPS = [1, 2, 3]

try:
    import psutil
    _proc = psutil.Process(os.getpid())
    def rss_mb():
        return _proc.memory_info().rss / (1024 * 1024)
except ImportError:
    def rss_mb():
        return float("nan")


def time_one_call(X, ncomp):
    gc.collect()
    rss_before = rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    nsprcomp(X, ncomp=ncomp, nneg=True, center=True, scale_=False)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_mb()
    return elapsed, peak_bytes / (1024 * 1024), rss_after - rss_before


def iqr(xs):
    if len(xs) < 2:
        return 0.0
    s = sorted(xs)
    return s[(3 * len(s)) // 4] - s[len(s) // 4]


def main():
    filt = None
    mode_filt = None
    ncomp_filt = None
    for a in sys.argv[1:]:
        if a.startswith("--datasets="):
            filt = a.split("=", 1)[1].split(",")
        elif a.startswith("--mode="):
            mode_filt = a.split("=", 1)[1]
        elif a.startswith("--ncomps="):
            ncomp_filt = [int(x) for x in a.split("=", 1)[1].split(",")]
    manifest = MANIFEST if filt is None else MANIFEST[MANIFEST["dataset"].isin(filt)]
    if mode_filt is not None:
        manifest = manifest[manifest["mode"] == mode_filt]
    ncomps_use = NCOMPS if ncomp_filt is None else ncomp_filt
    out_csv = RESULTS / "python_timings.csv"
    if filt is None and out_csv.exists():
        out_csv.unlink()

    rows = []
    total_calls = len(manifest) * len(ncomps_use) * N_TRIALS
    print("Running %d nsprcomp calls (%d matrices x %d ncomps x %d trials) filter=%s"
          % (total_calls, len(manifest), len(ncomps_use), N_TRIALS, filt), flush=True)

    grand_t0 = time.perf_counter()
    for _, row in manifest.iterrows():
        X = np.load(HERE / "_inputs" / (row["tag"] + ".npy"))
        for ncomp in ncomps_use:
            times, heaps, rss_deltas = [], [], []
            try:
                nsprcomp(X, ncomp=ncomp, nneg=True, center=True, scale_=False)
            except Exception as e:
                print("  [%s ncomp=%d] warm-up failed: %s" % (row["tag"], ncomp, e), flush=True)
                continue
            ok = True
            for _ in range(N_TRIALS):
                try:
                    t, h, d = time_one_call(X, ncomp)
                except Exception as e:
                    print("  [%s ncomp=%d] trial failed: %s" % (row["tag"], ncomp, e), flush=True)
                    ok = False
                    break
                times.append(t); heaps.append(h); rss_deltas.append(d)
            if not ok or not times:
                continue
            rec = {
                "dataset": row["dataset"], "gene_set": row["gene_set"],
                "mode": row["mode"], "n_rows": int(row["n_rows"]),
                "n_cols": int(row["n_cols"]), "ncomp": ncomp,
                "n_trials": N_TRIALS,
                "time_med_s": median(times), "time_min_s": min(times),
                "time_max_s": max(times), "time_iqr_s": iqr(times),
                "heap_peak_mb": max(heaps), "rss_delta_mb": median(rss_deltas),
            }
            rows.append(rec)
            print("  %-55s ncomp=%d med=%7.2f ms peak_heap=%6.2f MB"
                  % (row["tag"], ncomp, rec["time_med_s"]*1000, rec["heap_peak_mb"]),
                  flush=True)

    elapsed_total = time.perf_counter() - grand_t0
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, mode="a", header=not out_csv.exists())
    print("\nChunk elapsed: %.2fs (%d new rows)" % (elapsed_total, len(df)), flush=True)
    print("Wrote: %s" % out_csv, flush=True)


if __name__ == "__main__":
    main()
