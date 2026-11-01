"""Fig 2 from the benchmark v2 runs: where the speed comes from, accuracy against R, the
gene-count sweep and the whole-transcriptome fit.

Two versions: plots/fig2 (the manuscript figure, the current package only) and plots/figR2 (the
response-letter figure, which adds the solver before the non-negativity fix to the accuracy and
whole-transcriptome panels). Reads results/<run>/summary_cells.csv (written by analyze.py).
All fits use 1 thread. Speedups are relative to R nsprcomp on the same input and k; for the
real-data inputs each cell's median time is the median over the three repeat runs. RcppML is
reported in a supplementary table, not here.

Usage:
    python plot_fig2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
RES, PLOTS = HERE / "results", HERE / "plots"
REAL = ["realdata_89c9f66_r1", "realdata_89c9f66_r2", "realdata_89c9f66_r3"]
REAL_OLD = "realdata_head3b31cf9_t1_corder"       # the solver before the non-negativity fix
DSWEEP, PBMC, PBMC_OLD = "dsweep_89c9f66", "pbmc3k_k16", "pbmc3k_k16_old3b31cf9"

BLUE, ORANGE, AQUA, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#8a8985"
INK, MUTED = "#0b0b0b", "#52514e"
SOLVERS = {                                        # name: (label, colour, marker)
    "nsprcomp": ("R nsprcomp", GREY, "D"),
    "nsprcomp_crossprod": ("R nsprcomp + crossprod", AQUA, "^"),
    "V2": ("V2 (R Algorithm in NumPy)", ORANGE, "s"),
    "nnpcapy_old": ("nnPCApy before Fix (Dense)", MAGENTA, "v"),
    "nnpcapy": ("nnPCApy", BLUE, "o"),
}
COMPONENTS = {1: "1 Component", 2: "2 Components", 3: "3 Components"}

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "svg.fonttype": "none"})


def cells(run: str) -> pd.DataFrame:
    d = pd.read_csv(RES / run / "summary_cells.csv")
    return d[(d["mode"] == "fit") & d.ok.astype(bool) & (d.threads == 1)]


def letter(ax, s: str, dx: float = -0.2, dy: float = 1.05) -> None:
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/{name}.png and .svg")


def real_speedups() -> pd.DataFrame:
    """Per input, k and solver: speedup over nsprcomp from the median time over the repeat runs."""
    d = pd.concat([cells(r) for r in REAL])
    key = ["dataset", "gene_set", "ncomp"]
    t = d.groupby([*key, "solver"]).t_med_s.median().unstack("solver")
    out = t.rdiv(t["nsprcomp"], axis=0).stack().rename("speedup").reset_index()
    out["single_cell"] = out.dataset != "bulk"
    return out


def panel_a(ax) -> None:
    s = real_speedups()
    show = ["nsprcomp_crossprod", "V2", "nnpcapy"]
    rng = np.random.default_rng(0)
    for k in (1, 2, 3):
        for j, name in enumerate(show):
            _, col, _ = SOLVERS[name]
            x0 = (k - 1) * 4 + j
            for sc, mk, fill in ((True, "o", col), (False, "^", "none")):
                v = s[(s.ncomp == k) & (s.solver == name) & (s.single_cell == sc)].speedup
                ax.scatter(x0 + rng.uniform(-0.28, 0.28, len(v)), v, s=7, marker=mk, facecolor=fill,
                           edgecolor=col, lw=0.6, alpha=0.8, zorder=2)
            grp = s[(s.ncomp == k) & (s.solver == name)].speedup
            med = s[(s.ncomp == k) & (s.solver == name) & s.single_cell].speedup.median()
            ax.hlines(med, x0 - 0.38, x0 + 0.38, color=INK, lw=1.0, zorder=3)
            ax.text(x0, grp.max() * 1.35, f"{med:.1f}×" if med < 10 else f"{med:.0f}×", ha="center",
                    va="bottom", fontsize=7, zorder=4)
    ax.axhline(1, color=MUTED, ls=":", lw=0.7)
    ax.set_yscale("log")
    ax.set_xticks([1, 5, 9], COMPONENTS.values())
    ax.set_xlim(-0.7, 10.7)
    ax.set_ylabel("Speedup over R nsprcomp")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=SOLVERS[n][1], label=SOLVERS[n][0])
               for n in show]
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=3.5, mfc=MUTED, mec=MUTED, label="Single-Cell Input (33)"),
                plt.Line2D([], [], ls="none", marker="^", ms=3.5, mfc="none", mec=MUTED, label="Bulk Input (14)"),
                plt.Line2D([], [], color=INK, lw=1.0, label="Single-Cell Median")]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.3)
    ax.set_ylim(0.5, 1e4)


def panel_b(ax, before_fix: bool) -> None:
    series = [(cells(REAL[0]).query("solver == 'nnpcapy'"), "nnpcapy")]
    if before_fix:
        series.insert(0, (cells(REAL_OLD).query("solver == 'nnpcapy'"), "nnpcapy_old"))
    n = len(series)
    rng = np.random.default_rng(1)
    for k in (1, 2, 3):
        for j, (d, name) in enumerate(series):
            _, col, _ = SOLVERS[name]
            v = d[d.ncomp == k]
            x0 = (k - 1) * (n + 1) + j
            x = x0 + rng.uniform(-0.25, 0.25, len(v))
            neg = v.neg_loadings.to_numpy() > 0
            ax.scatter(x[~neg], v.adj_var_ratio[~neg], s=7, marker="o", facecolor=col, edgecolor="none",
                       alpha=0.8, zorder=2)
            ax.scatter(x[neg], v.adj_var_ratio[neg], s=9, marker="x", color=col, lw=0.7, alpha=0.9, zorder=2)
            ax.text(x0, 1.027, f"{neg.sum()}/{len(v)}", ha="center", fontsize=7, color=INK)
    ax.text(-0.6, 1.036, "Inputs with Negative Loadings", fontsize=7, color=MUTED)
    ax.axhline(1, color=MUTED, ls=":", lw=0.7)
    ax.set_xticks([(k - 1) * (n + 1) + (n - 1) / 2 for k in (1, 2, 3)], COMPONENTS.values())
    ax.set(xlim=(-0.8, 3 * (n + 1) - 1.2), ylim=(0.94, 1.045), ylabel="Adjusted Variance / R nsprcomp")
    if before_fix:
        handles = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=SOLVERS[m][1], label=SOLVERS[m][0])
                   for m in ("nnpcapy_old", "nnpcapy")]
        handles.append(plt.Line2D([], [], ls="none", marker="x", ms=4, color=MUTED, label="Fit with Negative Loadings"))
        ax.legend(handles=handles, frameon=False, loc="lower left", handletextpad=0.3)
    else:
        ax.set_title("nnPCApy, 47 Inputs", loc="left", fontsize=7, color=MUTED)


def panel_c(axes) -> None:
    d = cells(DSWEEP)
    for ax, k in zip(axes, (1, 3)):
        for name in ("nsprcomp", "nsprcomp_crossprod", "V2", "nnpcapy"):
            lab, col, mk = SOLVERS[name]
            s = d[(d.solver == name) & (d.ncomp == k)].sort_values("n_cols")
            ax.plot(s.n_cols, s.t_med_s, marker=mk, ms=3, lw=1.1, color=col, label=lab)
        ax.set(xscale="log", yscale="log", xlabel="Genes (5,000 Cells)")
        ax.set_title(COMPONENTS[k], loc="left", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Time per Fit (s)")
    axes[1].tick_params(labelleft=False)


def panel_d(axes, before_fix: bool) -> None:
    new, old = cells(PBMC), cells(PBMC_OLD).query("solver == 'nnpcapy'").assign(solver="nnpcapy_old")
    d = pd.concat([new, old]).set_index("solver")
    order = ["nsprcomp", "nsprcomp_crossprod", "V2", *(["nnpcapy_old"] if before_fix else []), "nnpcapy"]
    for ax, col_name, ylab, fmt in ((axes[0], "t_med_s", "Time per Fit (s)", "{:,.0f}"),
                                    (axes[1], "mem_delta_mb", "Extra Memory (MB)", "{:,.0f}")):
        v = d.loc[order, col_name]
        ax.bar(range(len(order)), v, color=[SOLVERS[n][1] for n in order], width=0.75)
        for i, x in enumerate(v):
            ax.text(i, x * 1.15, fmt.format(x) if x >= 100 else f"{x:.0f}" if x >= 10 else f"{x:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(order)), [])
        ax.set_ylim(top=v.max() * 20)
    axes[0].set_title("Whole Transcriptome, 16 Components", loc="left", fontsize=7, color=MUTED)


def figure(name: str, before_fix: bool) -> None:
    fig = plt.figure(figsize=(7, 5.2))
    g = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.32,
                         left=0.08, right=0.97, top=0.95, bottom=0.12)
    a = fig.add_subplot(g[0, 0]); panel_a(a); letter(a, "A", dx=-0.14)
    b = fig.add_subplot(g[0, 1]); panel_b(b, before_fix); letter(b, "B")
    gc = g[1, 0].subgridspec(1, 2, wspace=0.12)
    cs = [fig.add_subplot(gc[0])]
    cs.append(fig.add_subplot(gc[1], sharey=cs[0]))
    panel_c(cs); letter(cs[0], "C", dx=-0.3)
    gd = g[1, 1].subgridspec(1, 2, wspace=0.55)
    ds = [fig.add_subplot(gd[0]), fig.add_subplot(gd[1])]
    panel_d(ds, before_fix); letter(ds[0], "D", dx=-0.5)
    shown = [k for k in SOLVERS if before_fix or k != "nnpcapy_old"]
    handles = [plt.Line2D([], [], color=SOLVERS[k][1], marker=SOLVERS[k][2], ms=3.5, lw=1.1, label=SOLVERS[k][0])
               for k in shown]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=len(shown), bbox_to_anchor=(0.52, 0.035),
               handletextpad=0.3, columnspacing=1.2)      # C and D share these
    save(fig, name)


def main() -> None:
    figure("fig2", before_fix=False)     # manuscript
    figure("figR2", before_fix=True)     # response letter


if __name__ == "__main__":
    main()
