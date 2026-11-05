"""Figures from the Fig 3 analyses: Fig 3A-D, Fig S2 and nnPCA vs dense PC1 (supplementary).

Reads results/ (written by synth_order.py, pc_order.py, time_steps.py, c2_screen.py,
restart_scan.py and dense_pc1.py) and writes png and svg to plots/. The former panel E is a
supplementary figure (paper/panelE/fig3E_showcase.py). Fig 3B-D show each statistic's bootstrap
distribution from the "both" arm (resampled cells and new random starts) with its median and 95%
percentile interval. All text is at least 7 pt at the printed size.

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
B_SETS = ["Panchy_et_al_E_signature", "Panchy_et_al_M_signature", "FOROUTAN_TGFB_EMT_UP",
          "ONDER_CDH1_TARGETS_2_DN", "HOLLERN_EMT_BREAST_TUMOR_DN", "BOWIE_RESPONSE_TO_TAMOXIFEN"]
EXAMPLE_DRAWS = 60
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
ARMS = {"seed": (BLUE, "o", "New Starts, All Cells"),
        "sampling": (ORANGE, "s", "Resampled Cells, Fixed Starts"),
        "both": (AQUA, "^", "Resampled Cells, New Starts")}
FS = 7

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": FS, "axes.labelsize": FS, "legend.fontsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.titlesize": FS, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "svg.fonttype": "none"})


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


def violins(ax, data, pos, color, width=0.32, vert=False) -> None:
    """Bootstrap distributions as violins, each with its median (dot) and 95% interval (line)."""
    parts = ax.violinplot(data, positions=pos, widths=width, vert=vert, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
        body.set_linewidth(0.5)
    for d, p in zip(data, pos):
        lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
        if vert:
            ax.vlines(p, lo, hi, color=INK, lw=0.9, zorder=3)
            ax.plot(p, med, "o", color=INK, ms=2.2, zorder=4)
        else:
            ax.hlines(p, lo, hi, color=INK, lw=0.9, zorder=3)
            ax.plot(med, p, "o", color=INK, ms=2.2, zorder=4)


def panel_a(ax) -> None:
    d = pd.read_csv(rs.RESULTS / "synth_order.csv")
    for arm, (col, mk, lab) in ARMS.items():
        s = d[d.arm == arm].sort_values("ratio")
        ax.errorbar(s.ratio, s.p_top_is_w1, yerr=1.96 * s.p_mcse, color=col, marker=mk, ms=3.5, lw=1.1,
                    capsize=0, label=lab)
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.invert_xaxis()
    ax.set(xlabel="Planted SD Ratio", ylim=(0.4, 1.04),
           ylabel="Share of Fits Whose First\nComponent Is the Larger One")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.6), handlelength=1.6)


def panel_a_examples(axes) -> None:
    z = np.load(rs.RESULTS / "synth_order_loadings.npz")
    n1 = int((z["w1"] > 0).sum())
    vmax = max(float(z[f"r{r:g}_both"][:EXAMPLE_DRAWS].max()) for r in (2.0, 1.0))
    for ax, ratio in zip(axes, (2.0, 1.0)):
        w = z[f"r{ratio:g}_both"][:EXAMPLE_DRAWS]
        w = w[np.argsort(-w[:, :n1].sum(1))]              # fits on the first block on top
        im = ax.imshow(w, aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="none")
        ax.axvline(n1 - 0.5, color=INK, lw=0.7)
        ax.set_xticks([n1 / 2 - 0.5, n1 + (w.shape[1] - n1) / 2 - 0.5], ["Larger", "Smaller"])
        ax.tick_params(axis="x", length=0)
        ax.set_yticks([])
        ax.set_title(f"SD Ratio {ratio:.1f}", loc="left")
        ax.set_xlabel("Planted Component's Genes")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel(f"{EXAMPLE_DRAWS} Resampled Fits")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.08, pad=0.04)
    cb.set_label("Loading of the\nFirst Component")
    cb.outline.set_visible(False)


def panel_b(ax) -> None:
    z = np.load(rs.RESULTS / "pc_order_both.npz")
    y = np.arange(len(B_SETS))[::-1]
    violins(ax, [z[f"{g}|raw1"] for g in B_SETS], y + 0.2, BLUE, width=0.36)
    violins(ax, [z[f"{g}|raw2"] for g in B_SETS], y - 0.2, ORANGE, width=0.36)
    ax.set_yticks(y, [wrap(g, 24) for g in B_SETS])
    ax.set_ylim(-0.7, len(B_SETS) - 0.3)
    ax.set_xlabel("SD of the Component Score\nbefore Deflation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6, label=lab)
               for c, lab in ((BLUE, "First Component"), (ORANGE, "Second Component"))]
    handles.append(Line2D([], [], color=INK, marker="o", ms=2.2, lw=0.9, label="Median, 95% Interval"))
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=1.2)


def panel_c(ax) -> None:
    z = np.load(rs.RESULTS / "time_steps_both.npz")
    m = pd.read_csv(rs.RESULTS / "time_means.csv").query("arm == 'both'")
    st = pd.read_csv(rs.RESULTS / "time_steps.csv").query("arm == 'both'")
    x = np.arange(len(ORDER))
    top = blended(ax.transData, ax.transAxes)
    ax.text(-0.45, 1.19, "Share of Draws in Which the Mean Rises", transform=top, color=MUTED)
    for score, col, off, yrow in (("E", BLUE, -0.14, 1.11), ("M", ORANGE, 0.14, 1.03)):
        violins(ax, list(z[score].T), x + off, col, width=0.3, vert=True)
        pt = m[m.score == score].set_index("time").reindex(ORDER).point
        ax.plot(x + off, pt, color=col, lw=0.9, zorder=2)
        ax.text(-0.45, yrow, score, transform=top, color=col, fontweight="bold", va="center")
        for i, p in enumerate(st[st.score == score].support_up):
            txt = f"{p:.1%}" if 0 < p < 0.005 or 0.995 <= p < 1 else f"{p:.0%}"
            ax.text(i + 0.5, yrow, txt, transform=top, ha="center", va="center")
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(x, ORDER)
    ax.set(xlim=(-0.5, len(ORDER) - 0.5), xlabel="Time after TGF-β", ylabel="Mean Score (SD Units)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6, label=f"{s} Score")
                       for s, c in (("E", BLUE), ("M", ORANGE))], frameon=False, loc="center left")


def panel_d(axes) -> None:
    d = pd.read_csv(rs.RESULTS / "c2_screen.csv")
    draw_dir = next((rs.RESULTS / "c2_screen").glob(f"{len(d)}sets_seed*"))
    names = pd.read_csv(draw_dir / "point.csv").gene_set.tolist()
    R = np.stack([np.load(f) for f in sorted(draw_dir.glob("b*.npy"))])          # draws x sets
    col = {g: i for i, g in enumerate(names)}
    for ax, (sub, which, freq) in zip(axes, ((d.nlargest(5, "r_point"), "Positive", "top15_pos_freq"),
                                             (d.nsmallest(5, "r_point"), "Negative", "top15_neg_freq"))):
        sub = sub.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub))
        violins(ax, [R[:, col[g]] for g in sub.gene_set], y, MUTED, width=0.7)
        ax.set_yticks(y, [wrap(g, 24) for g in sub.gene_set])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(f"Most {which} r", loc="left", color=MUTED)
        right = blended(ax.transAxes, ax.transData)
        ax.text(1.03, len(sub) - 0.35, "In Top 15", transform=right, color=MUTED, va="bottom")
        for yy, f in zip(y, sub[freq]):
            ax.text(1.03, yy, f"{f:.0%}", transform=right, va="center")
        ax.set_xlabel(f"Spearman r with the M Score\n({len(d):,} C2 Sets Screened)")


def fig3() -> None:
    fig = plt.figure(figsize=(6.9, 8.6))
    # one grid per row, so each row leaves the room its own labels need
    r1 = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 0.95], wspace=0.35, left=0.1, right=0.9, top=0.97,
                          bottom=0.75)
    panel_a(fig.add_subplot(r1[0]))
    panel_a_examples([fig.add_subplot(r1[1]), fig.add_subplot(r1[2])])
    r2 = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.32, left=0.25, right=0.98, top=0.6,
                          bottom=0.37)
    panel_b(fig.add_subplot(r2[0]))
    panel_c(fig.add_subplot(r2[1]))
    r3 = fig.add_gridspec(1, 2, wspace=1.62, left=0.25, right=0.9, top=0.27, bottom=0.06)
    ds = [fig.add_subplot(r3[0]), fig.add_subplot(r3[1])]
    panel_d(ds)
    for s, x, y in (("A", 0.0, 0.985), ("B", 0.0, 0.66), ("C", 0.58, 0.66), ("D", 0.0, 0.31)):
        fig.text(x, y, s, fontsize=10, fontweight="bold", va="bottom")
    save(fig, "fig3_AD")


def _rm_hist(ax, name: str) -> None:
    bins = np.linspace(-1, 1, 81)
    for arm, (col, _mk, lab) in ARMS.items():
        r = np.load(rs.RESULTS / f"pc_order_{arm}.npz")[f"{name}|r_M"]
        ax.hist(r, bins=bins, weights=np.full(len(r), 1 / len(r)), histtype="step", color=col, lw=1.1,
                label=lab)
    ax.set(xlabel="Spearman r with the M Score", ylabel="Share of Draws", xlim=(-1, 1))
    ax.set_title(wrap(name, 26) + "\n5 Restarts", loc="left", color=MUTED)


def fig_s2() -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7, 2.7), gridspec_kw={"wspace": 0.5})
    cur = pd.read_csv(rs.RESULTS / "restart_curve_trans_golgi.csv")
    a.plot(cur.nrestart, cur.share_best, "o-", color=BLUE, ms=3.5, lw=1.1)
    a.set_xscale("log")
    a.set_xticks(cur.nrestart, [str(v) for v in cur.nrestart])
    a.minorticks_off()
    a.set(xlabel="Random Restarts", ylabel="Share of Fits Reaching the\nHigher-Variance Solution", ylim=(0, 1.04))
    a.set_title(wrap(TG, 26), loc="left", color=MUTED)
    _rm_hist(b, TG)
    _rm_hist(c, "HOUSTIS_ROS")
    fig.legend(*c.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    for ax, s in zip((a, b, c), "ABC"):
        letter(ax, s, dx=-0.32, dy=1.14)
    save(fig, "figS2")


def fig_dense() -> None:
    d = pd.read_csv(rs.RESULTS / "dense_pc1.csv")
    c2 = pd.read_csv(rs.RESULTS / "c2_screen.csv")
    shown = set(c2.nlargest(5, "r_point").gene_set) | set(c2.nsmallest(5, "r_point").gene_set) | {TG}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7, 2.6), gridspec_kw={"wspace": 0.5})
    dot = {"s": 2, "color": BLUE, "alpha": 0.35, "lw": 0, "rasterized": True}
    a.scatter(d.neg_share, d.r_scores.abs(), **dot)
    a.set(xlabel="Share of the Dense PC1 Loading\non Genes with Negative Weight", ylim=(0, 1.02),
          ylabel="|Spearman r| between nnPCA\nand Dense PC1 Scores")
    b.scatter(d.neg_share, d.var_kept, **dot)
    b.set(xlabel="Share of the Dense PC1 Loading\non Genes with Negative Weight",
          ylabel="Variance of the nnPCA Score /\nVariance of Dense PC1")
    c.axline((0, 0), slope=1, color=MUTED, lw=0.6, ls=":")
    c.axline((0, 0), slope=-1, color=MUTED, lw=0.6, ls=":")
    c.scatter(d.rM_nn, d.rM_dense, **dot)
    s = d[d.gene_set.isin(shown)]
    c.scatter(s.rM_nn, s.rM_dense, s=14, color=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    flips = s[np.sign(s.rM_nn) != np.sign(s.rM_dense)].sort_values("rM_dense", ascending=False)
    key = []
    for i, (_, r) in enumerate(flips.iterrows(), 1):    # numbered in the panel, named in the key below
        c.annotate(str(i), (r.rM_nn, r.rM_dense), xytext=(5, 0), textcoords="offset points", va="center")
        key.append(f"{i} = {r.gene_set}")
    c.set(xlabel="Spearman r with M, nnPCA", ylabel="Spearman r with M, Dense PC1", xlim=(-1, 1), ylim=(-1, 1))
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=BLUE, alpha=0.6, label=f"All {len(d):,} C2 Sets"),
               Line2D([], [], ls="none", marker="o", ms=4, color=ORANGE, label="Shown in Fig 3D or S2"),
               Line2D([], [], ls=":", lw=0.8, color=MUTED, label="Same r or Opposite r")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.22, "Sign Changes in C: " + "; ".join(key), ha="center", va="top")
    for ax, s_ in zip((a, b, c), "ABC"):
        letter(ax, s_, dx=-0.32, dy=1.04)
    save(fig, "fig_dense_pc1")


def main() -> None:
    fig3()
    fig_s2()
    fig_dense()


if __name__ == "__main__":
    main()
