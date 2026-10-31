"""Figures from the Fig 3 analyses: Fig 3A-D, Fig S2 and nnPCA vs dense PC1 (supplementary).

Reads results/ (written by synth_order.py, pc_order.py, time_steps.py, c2_screen.py,
restart_scan.py and dense_pc1.py) and writes png and svg to plots/. Panel E of Fig 3 is drawn
separately. Unless a panel compares arms, intervals are 95% percentile intervals from the
"both" arm (resampled cells and new random starts).

Usage:
    python plot_fig3.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import resample as rs
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as blended

plt.switch_backend("Agg")

PLOTS = rs.HERE / "plots"
ORDER = ["0d", "8h", "1d", "3d", "7d"]
TG = "REACTOME_TRANS_GOLGI_NETWORK_VESICLE_BUDDING"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
ARMS = {"seed": (BLUE, "o", "New Starts, All Cells"),
        "sampling": (ORANGE, "s", "Resampled Cells, Fixed Starts"),
        "both": (AQUA, "^", "Resampled Cells, New Starts")}

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "svg.fonttype": "none"})


def wrap(name: str, width: int = 28) -> str:
    """Break a gene-set name after underscores into lines of about `width` characters, keeping every character."""
    lines, cur = [], ""
    for part in name.split("_"):
        add = f"{cur}_{part}" if cur else part
        if len(add) > width and cur:
            lines.append(cur + "_")
            cur = part
        else:
            cur = add
    return "\n".join([*lines, cur])


def letter(ax, s: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote plots/{name}.png and .svg ({w:.1f} x {h:.1f} in before trimming)")


def panel_a(ax) -> None:
    d = pd.read_csv(rs.RESULTS / "synth_order.csv")
    for arm, (col, mk, lab) in ARMS.items():
        s = d[d.arm == arm].sort_values("ratio")
        ax.errorbar(s.ratio, s.p_top_is_w1, yerr=1.96 * s.p_mcse, color=col, marker=mk, ms=3.5, lw=1.1,
                    capsize=0, label=lab)
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.invert_xaxis()
    ax.set(xlabel="Planted Variance Ratio of the Two Components", ylim=(0.4, 1.04),
           ylabel="Share of Fits Whose First Component\nIs the Planted Larger One")
    ax.legend(frameon=False, loc="center left")


def panel_b(ax) -> None:
    s2 = set(rs.parse_gmt(rs.DATA / "degeneracy_demo.gmt"))
    s = pd.read_csv(rs.RESULTS / "pc_order_summary.csv")
    s = s[(s.arm == "both") & ~s.gene_set.isin(s2)].sort_values("raw1_med").reset_index(drop=True)
    y = np.arange(len(s))
    for k, col, mk, off in ((1, BLUE, "o", 0.17), (2, ORANGE, "s", -0.17)):
        ax.hlines(y + off, s[f"raw{k}_lo"], s[f"raw{k}_hi"], color=col, lw=1.3)
        ax.plot(s[f"raw{k}_med"], y + off, mk, color=col, ms=3, ls="none",
                label=("First" if k == 1 else "Second") + " Component")
    ax.set_yticks(y, [wrap(g) for g in s.gene_set])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.set_xlabel("SD of the Component Score before Deflation")
    ax.legend(frameon=False, loc="lower right")


def panel_c(ax) -> None:
    m = pd.read_csv(rs.RESULTS / "time_means.csv").query("arm == 'both'")
    st = pd.read_csv(rs.RESULTS / "time_steps.csv").query("arm == 'both'")
    x = np.arange(len(ORDER))
    top = blended(ax.transData, ax.transAxes)
    ax.text(-0.45, 1.17, "Share of Draws in Which the Mean Rises", transform=top, color=MUTED, fontsize=6)
    for score, col, mk, yrow in (("E", BLUE, "o", 1.10), ("M", ORANGE, "s", 1.03)):
        sub = m[m.score == score].set_index("time").reindex(ORDER)
        ax.errorbar(x, sub.point, yerr=[sub.point - sub.ci_lo, sub.ci_hi - sub.point], color=col, marker=mk,
                    ms=3.5, lw=1.1, capsize=2, label=f"{score} Score")
        ax.text(-0.45, yrow, score, transform=top, color=col, fontsize=6, fontweight="bold", va="center")
        for i, p in enumerate(st[st.score == score].support_up):
            txt = f"{p:.1%}" if 0 < p < 0.005 or 0.995 <= p < 1 else f"{p:.0%}"
            ax.text(i + 0.5, yrow, txt, transform=top, fontsize=6, ha="center", va="center")
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(x, ORDER)
    ax.set(xlim=(-0.5, len(ORDER) - 0.5), xlabel="Time after TGF-β", ylabel="Mean Score (SD Units)")
    ax.legend(frameon=False, loc="center left")


def panel_d(fig, spec):
    d = pd.read_csv(rs.RESULTS / "c2_screen.csv")
    g = spec.subgridspec(2, 1, hspace=0.45)
    axes = []
    for j, (sub, which, freq) in enumerate(((d.nlargest(5, "r_point"), "Positively", "top15_pos_freq"),
                                            (d.nsmallest(5, "r_point"), "Negatively", "top15_neg_freq"))):
        ax = fig.add_subplot(g[j])
        sub = sub.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub))
        ax.hlines(y, sub.r_lo, sub.r_hi, color=INK, lw=1.3)
        ax.plot(sub.r_point, y, "o", color=INK, ms=3)
        ax.set_yticks(y, [wrap(gs) for gs in sub.gene_set])
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(f"Most {which} Correlated of {len(d):,} Sets", loc="left", fontsize=6.5, color=MUTED)
        right = blended(ax.transAxes, ax.transData)
        ax.text(1.03, len(sub) - 0.3, "In Top 15", transform=right, fontsize=6, color=MUTED, va="bottom")
        for yy, f in zip(y, sub[freq]):
            ax.text(1.03, yy, f"{f:.0%}", transform=right, fontsize=6, va="center")
        axes.append(ax)
    axes[-1].set_xlabel("Spearman r with the M Score")
    return axes


def fig3() -> None:
    fig = plt.figure(figsize=(7, 7.2))
    g = fig.add_gridspec(2, 2, left=0.09, right=0.92, top=0.94, bottom=0.07, wspace=1.0, hspace=0.42,
                         width_ratios=[0.8, 1], height_ratios=[1.15, 1])
    ax = fig.add_subplot(g[0, 0]); panel_a(ax); letter(ax, "A", dx=-0.3)
    ax = fig.add_subplot(g[0, 1]); panel_b(ax); letter(ax, "B", dx=-0.95)
    axc = fig.add_subplot(g[1, 0]); panel_c(axc); letter(axc, "C", dx=-0.3, dy=1.2)
    axes = panel_d(fig, g[1, 1])
    to_fig = fig.transFigure.inverted()
    y_c = to_fig.transform(axc.transAxes.transform((0, 1.2)))[1]       # D level with C
    x_d = to_fig.transform(axes[0].transAxes.transform((-0.95, 0)))[0]
    fig.text(x_d, y_c, "D", fontsize=10, fontweight="bold", va="bottom")
    save(fig, "fig3_AD")


def _rm_hist(ax, name: str) -> None:
    bins = np.linspace(-1, 1, 81)
    for arm, (col, _mk, lab) in ARMS.items():
        r = np.load(rs.RESULTS / f"pc_order_{arm}.npz")[f"{name}|r_M"]
        ax.hist(r, bins=bins, weights=np.full(len(r), 1 / len(r)), histtype="step", color=col, lw=1.1,
                label=lab)
    ax.set(xlabel="Spearman r with the M Score", ylabel="Share of Draws", xlim=(-1, 1))
    ax.set_title(wrap(name, 30) + " (5 Restarts)", loc="left", fontsize=6, color=MUTED)


def fig_s2() -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7, 2.6), gridspec_kw={"wspace": 0.45})
    cur = pd.read_csv(rs.RESULTS / "restart_curve_trans_golgi.csv")
    a.plot(cur.nrestart, cur.share_best, "o-", color=BLUE, ms=3.5, lw=1.1)
    a.set_xscale("log")
    a.set_xticks(cur.nrestart, [str(v) for v in cur.nrestart])
    a.minorticks_off()
    a.set(xlabel="Random Restarts", ylabel="Share of Fits Reaching the\nHigher-Variance Solution", ylim=(0, 1.04))
    a.set_title(wrap(TG, 30), loc="left", fontsize=6, color=MUTED)
    _rm_hist(b, TG)
    _rm_hist(c, "HOUSTIS_ROS")
    fig.legend(*c.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    for ax, s in zip((a, b, c), "ABC"):
        letter(ax, s, dx=-0.3, dy=1.12)
    save(fig, "figS2")


def fig_dense() -> None:
    d = pd.read_csv(rs.RESULTS / "dense_pc1.csv")
    c2 = pd.read_csv(rs.RESULTS / "c2_screen.csv")
    shown = set(c2.nlargest(5, "r_point").gene_set) | set(c2.nsmallest(5, "r_point").gene_set) | {TG}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7, 2.4), gridspec_kw={"wspace": 0.45})
    dot = {"s": 2, "color": BLUE, "alpha": 0.35, "lw": 0, "rasterized": True}
    a.scatter(d.neg_share, d.r_scores.abs(), **dot)
    a.set(xlabel="Share of the Dense PC1 Loading on\nGenes with Negative Weight", ylim=(0, 1.02),
          ylabel="|Spearman r| between\nnnPCA and Dense PC1 Scores")
    b.scatter(d.neg_share, d.var_kept, **dot)
    b.set(xlabel="Share of the Dense PC1 Loading on\nGenes with Negative Weight",
          ylabel="Variance of the nnPCA Score /\nVariance of Dense PC1")
    c.axline((0, 0), slope=1, color=MUTED, lw=0.6, ls=":")
    c.axline((0, 0), slope=-1, color=MUTED, lw=0.6, ls=":")
    c.scatter(d.rM_nn, d.rM_dense, **dot)
    s = d[d.gene_set.isin(shown)]
    c.scatter(s.rM_nn, s.rM_dense, s=14, color=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    spots = {TG: ((-0.98, 0.3), 16), "BANDRES_RESPONSE_TO_CARMUSTIN_MGMT_48HR_DN": ((-0.62, 0.93), 24)}
    for _, r in s[np.sign(s.rM_nn) != np.sign(s.rM_dense)].iterrows():
        xy_text, width = spots.get(r.gene_set, ((0, 0), 24))
        c.annotate(wrap(r.gene_set, width), (r.rM_nn, r.rM_dense), xytext=xy_text,
                   fontsize=5, va="center", ha="left", zorder=4,
                   bbox={"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.85},
                   arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5, "shrinkB": 2})
    c.set(xlabel="Spearman r with M, nnPCA", ylabel="Spearman r with M, Dense PC1", xlim=(-1, 1), ylim=(-1, 1))
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=BLUE, alpha=0.6, label=f"All {len(d):,} C2 Sets"),
               Line2D([], [], ls="none", marker="o", ms=4, color=ORANGE, label="Shown in Fig 3D or S2"),
               Line2D([], [], ls=":", lw=0.8, color=MUTED, label="Same r or Opposite r")]
    c.legend(handles=handles, loc="lower right", fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none",
             borderpad=0.3, handletextpad=0.4)
    for ax, s_ in zip((a, b, c), "ABC"):
        letter(ax, s_, dx=-0.3, dy=1.04)
    save(fig, "fig_dense_pc1")


def main() -> None:
    fig3()
    fig_s2()
    fig_dense()


if __name__ == "__main__":
    main()
