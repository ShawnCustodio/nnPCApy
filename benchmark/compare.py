"""
Merge python_timings.csv and r_timings.csv into summary.csv + plots.

Run after both bench_python.py and bench_r.R have produced timings.
Tolerant of one side being missing (writes a one-sided summary in that case).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS   = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

KEY_COLS = ["dataset", "gene_set", "mode", "ncomp"]
VAL_COLS = ["time_med_s", "time_min_s", "time_max_s",
            "time_iqr_s", "heap_peak_mb", "n_rows", "n_cols"]


def _render_python_only_plots(py: pd.DataFrame) -> None:
    """Refresh the two Python-only plots from the latest python_timings.csv."""
    palette = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    # single-gene-set bar chart
    single = py[py["mode"] == "single"].copy()
    single["label"] = single["dataset"] + " / " + single["gene_set"]
    labels = sorted(single["label"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.25
    for i, ncomp in enumerate([1, 2, 3]):
        sub = single[single["ncomp"] == ncomp].set_index("label").reindex(labels)
        xs = np.arange(len(labels)) + (i - 1) * w
        ax.bar(xs, sub["time_med_s"] * 1000, width=w,
               color=palette[ncomp], label=f"ncomp={ncomp}",
               yerr=sub["time_iqr_s"] * 1000 / 2, capsize=3)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median wall time (ms, ±IQR/2)")
    ax.set_title("Python nnPCA timing — single gene set "
                 "(8 conditions × 3 ncomps × 5 trials)")
    ax.legend(title="ncomp", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "python_single_timings.png", dpi=130)
    plt.close(fig)
    print(f"Wrote: {PLOTS/'python_single_timings.png'}")

    # multi-gene-set total time across C2 pathways
    multi = py[py["mode"] == "multi"].copy()
    if not multi.empty:
        agg = (multi.groupby(["dataset", "ncomp"], as_index=False)
                    .agg(total_ms=("time_med_s", lambda s: s.sum() * 1000)))
        ds_order = ["bulk", "A549_TGFB1", "A549_EGF", "A549_TNF"]
        ds_order = [d for d in ds_order if d in set(agg["dataset"])]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, ncomp in enumerate([1, 2, 3]):
            sub = agg[agg["ncomp"] == ncomp].set_index("dataset").reindex(ds_order)
            xs = np.arange(len(ds_order)) + (i - 1) * w
            ax.bar(xs, sub["total_ms"], width=w,
                   color=palette[ncomp], label=f"ncomp={ncomp}")
        ax.set_xticks(np.arange(len(ds_order)))
        ax.set_xticklabels(ds_order)
        ax.set_ylabel("Total wall time across all C2 pathways (ms)")
        ax.set_title("Python nnPCA multi-gene-set sweep (filtered.c2.gmt)")
        ax.legend(title="ncomp", frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS / "python_multi_totals.png", dpi=130)
        plt.close(fig)
        print(f"Wrote: {PLOTS/'python_multi_totals.png'}")


def _load(path: Path, suffix: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [missing] {path.name}")
        return None
    df = pd.read_csv(path)
    df = df[KEY_COLS + VAL_COLS].rename(
        columns={c: f"{c}_{suffix}" for c in VAL_COLS})
    return df


def main() -> None:
    py = _load(RESULTS / "python_timings.csv", "py")
    r  = _load(RESULTS / "r_timings.csv",      "r")
    if py is None and r is None:
        print("No timing files found - run bench_python.py and bench_r.R first.")
        return

    if py is not None and r is not None:
        merged = py.merge(r, on=KEY_COLS, how="outer")
        merged["speedup"]    = merged["time_med_s_r"] / merged["time_med_s_py"]
        merged["mem_ratio"]  = merged["heap_peak_mb_r"] / merged["heap_peak_mb_py"]
        # Use Python-side n_rows/n_cols if available, else R-side.
        merged["n_rows"] = merged["n_rows_py"].fillna(merged["n_rows_r"]).astype(int)
        merged["n_cols"] = merged["n_cols_py"].fillna(merged["n_cols_r"]).astype(int)
    elif py is not None:
        merged = py.copy()
        merged["n_rows"] = merged["n_rows_py"].astype(int)
        merged["n_cols"] = merged["n_cols_py"].astype(int)
    else:
        merged = r.copy()
        merged["n_rows"] = merged["n_rows_r"].astype(int)
        merged["n_cols"] = merged["n_cols_r"].astype(int)

    merged.to_csv(RESULTS / "summary.csv", index=False)
    print(f"\nWrote: {RESULTS/'summary.csv'}  ({len(merged)} rows)")

    # ── Python-only plots — always regenerated from the latest CSV ──────
    if py is not None:
        _render_python_only_plots(pd.read_csv(RESULTS / "python_timings.csv"))

    if py is None or r is None:
        return  # cross-language plots only make sense with both sides

    # ── plot 1: speedup by dataset/ncomp (single gene set only) ──────────
    single = merged[merged["mode"] == "single"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for ncomp in sorted(single["ncomp"].unique()):
        sub = single[single["ncomp"] == ncomp].copy()
        sub["label"] = sub["dataset"] + " / " + sub["gene_set"]
        sub = sub.sort_values("label").reset_index(drop=True)
        xs = np.arange(len(sub)) + (ncomp - 2) * 0.25
        ax.bar(xs, sub["speedup"], width=0.22,
               color=palette[ncomp], label=f"ncomp={ncomp}")
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["label"], rotation=30, ha="right")
    ax.set_ylabel("Speedup (R median time / Python median time)")
    ax.set_title("nnPCA speedup: Python (nsprcomp port) vs R (nsprcomp)")