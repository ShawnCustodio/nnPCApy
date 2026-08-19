"""
Exploration (with MEASURED R from d:/R): does a more conservative gene-set size
(D=200) change the ablation conclusions vs D=89?  Reads sc_ablation_D89_D200.csv
(R measured on d:/R; Python V0-V3 timed on the same seeded matrices) and draws the
two ablations side by side, annotating the vectorization × covariance decomposition.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
(HERE / "plots").mkdir(parents=True, exist_ok=True)   # gitignored -> create on fresh clone
df = pd.read_csv(HERE / "results" / "sc_ablation_D89_D200.csv").set_index("D")
COL = {"R_nsprcomp": "#d62728", "V0_naive": "#9aa0a6", "V1_params": "#1f6feb",
       "V2_twopass": "#f0a01e", "V3_shipping": "#0f8a7e"}
ORDER = ["R_nsprcomp", "V0_naive", "V1_params", "V2_twopass", "V3_shipping"]
LABS = ["R", "V0", "V1", "V2", "V3"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, D in zip(axes, [89, 200]):
        r = df.loc[D]
        xs = np.arange(len(ORDER))
        ax.bar(xs, [r[k] for k in ORDER], color=[COL[k] for k in ORDER], width=0.72)
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels(LABS, fontsize=11)
        ax.set_title(f"N=12,000, D={D}", fontsize=13)
        for x, k in zip(xs, ORDER):
            ax.text(x, r[k] * 1.08, f"{r[k]:.0f}", ha="center", va="bottom", fontsize=9.5)
        ax.set_ylim(top=r["R_nsprcomp"] * 3.4)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.text(0.5, 0.015,
                f"total R→V3 = {r.R_nsprcomp/r.V3_shipping:.0f}×\n"
                f"= vectorization {r.R_nsprcomp/r.V2_twopass:.1f}× (R→V2)  ×  "
                f"covariance {r.V2_twopass/r.V3_shipping:.1f}× (V2→V3)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#333")
    axes[0].set_ylabel("Median time (ms)", fontsize=12)
    fig.suptitle("Speedup at single-cell scale is robust to gene-set size "
                 "(total ≈ 21× either way; covariance matters more at larger D)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        fig.savefig(HERE / "plots" / f"fig2A_D89_vs_D200.{ext}", dpi=150, bbox_inches="tight")
    print("wrote fig2A_D89_vs_D200")


if __name__ == "__main__":
    main()
