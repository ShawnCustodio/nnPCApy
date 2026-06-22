"""
Section 2.7-2.10: E vs M and M1 vs M2 scatter plots
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .plotdata import RebuildResult

COLORS = [
    "#F87189", "#CE9031", "#A48CF5", "#97A430", "#39A7D0",
    "#E57D5F", "#84C7B9", "#E1AF64", "#C26CCF", "#B0BF43",
    "#57C3E8", "#F29D9E", "#92AAE6",
]
REF_PALETTE = COLORS
E_COL = "Panchy_et_al_E_signature"
M_COL = "Panchy_et_al_M_signature"

def plot_em_panel(data: pd.DataFrame, xcol: str, ycol: str, title: str,
                  ax=None, palette=None) -> plt.Figure | None:
    """v2 Section 2.7 - E vs M scatter with KDE contours and mean +/- SD crosses."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    if xcol not in data.columns or ycol not in data.columns:
        print(f"[SKIP] {title}")
        print("Missing columns:", xcol, ycol)
        print("Available:", list(data.columns))
        return None
    cell_types = sorted(data["celltype_annotation"].dropna().unique())
    colors = (palette or REF_PALETTE)[: len(cell_types)]
    color_map = dict(zip(cell_types, colors))
    for ct in cell_types:
        subset = data[data["celltype_annotation"] == ct]
        if len(subset) > 5:
            sns.kdeplot(
                data=subset, x=xcol, y=ycol,
                fill=True, alpha=0.18, levels=5, thresh=0.05,
                color=color_map[ct], ax=ax,
            )
    sns.scatterplot(
        data=data, x=xcol, y=ycol,
        hue="celltype_annotation", hue_order=cell_types,
        palette=color_map, s=36, alpha=0.55, edgecolor=None, ax=ax,
    )
    for ct in cell_types:
        subset = data[data["celltype_annotation"] == ct]
        mX, mY = subset[xcol].mean(), subset[ycol].mean()
        sX, sY = subset[xcol].std(),  subset[ycol].std()
        ax.errorbar(mX, mY, xerr=sX, yerr=sY,
                    fmt="none", ecolor="black",
                    elinewidth=3, capsize=0, zorder=5)
        ax.scatter(mX, mY, s=220, color=color_map[ct],
                   edgecolor="white", linewidth=2.2, zorder=6)
    ax.set_xlabel(xcol, fontsize=13, style="italic")
    ax.set_ylabel(ycol, fontsize=13, style="italic")
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black"); spine.set_linewidth(1.2)
    ax.set_facecolor("white")
    leg = ax.legend(
        title="celltype_annotation",
        bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=True, fontsize=10, title_fontsize=11,
    )
    if leg is not None:
        leg.get_title().set_fontstyle("italic")
    if standalone:
        plt.tight_layout()
    if standalone:
        plt.close(fig)
    return fig

def plot_em_section(rb: RebuildResult) -> list[plt.Figure]:
    """Section 2.7 - render the three E-vs-M panels."""
    return [
        plot_em_panel(rb.nnPCA_em,  E_COL, M_COL, "E vs M Scores (nnPCA)"),
        plot_em_panel(rb.aucell_em, E_COL, M_COL, "E vs M Scores (AUCell)"),
        plot_em_panel(rb.ssgsea_em, E_COL, M_COL, "E vs M Scores (ssGSEA)"),
    ]

def _plot_m1_m2_into(ax, nnPCA_mm: pd.DataFrame) -> None:
    cell_types = nnPCA_mm["celltype_annotation"].unique()
    for i, ct in enumerate(cell_types):
        subset = nnPCA_mm[nnPCA_mm["celltype_annotation"] == ct]
        if len(subset) > 5:
            sns.kdeplot(data=subset, x="M1", y="M2",
                        fill=True, alpha=0.25, levels=6, thresh=0.05,
                        color=COLORS[i % len(COLORS)], zorder=1, ax=ax)
    sns.scatterplot(data=nnPCA_mm, x="M1", y="M2", hue="celltype_annotation",
                    palette={ct: COLORS[i % len(COLORS)] for i, ct in enumerate(cell_types)},
                    s=35, alpha=0.6, edgecolor=None, zorder=2, ax=ax)
    for i, ct in enumerate(cell_types):
        subset = nnPCA_mm[nnPCA_mm["celltype_annotation"] == ct]
        m1, m2 = subset["M1"].mean(), subset["M2"].mean()
        s1, s2 = subset["M1"].std(),  subset["M2"].std()
        ax.errorbar(m1, m2, xerr=s1, yerr=s2,
                    fmt="none", ecolor="black", elinewidth=3, capsize=0, zorder=3)
        ax.scatter(m1, m2, s=260, color=COLORS[i % len(COLORS)],
                   edgecolor="white", linewidth=2, zorder=4)
    ax.set_xlabel("M1 Score", fontsize=12)
    ax.set_ylabel("M2 Score", fontsize=12)
    ax.set_title("M1 vs M2 Scores (nnPCA on M signature)", fontsize=14)
    ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")

def plot_m1_m2(nnPCA_mm: pd.DataFrame) -> plt.Figure:
    """Section 2.8 - M1 vs M2 scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_m1_m2_into(ax, nnPCA_mm)
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_combined_em_m1_m2(nnPCA_em: pd.DataFrame,
                            nnPCA_mm: pd.DataFrame) -> plt.Figure:
    """Section 2.9 - side-by-side E vs M and M1 vs M2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_em_panel(nnPCA_em, E_COL, M_COL, "E vs M Scores (nnPCA)", ax=axes[0])
    _plot_m1_m2_into(axes[1], nnPCA_mm)
    fig.suptitle("Panchy_et_al", fontsize=15, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_m1_histogram(nnPCA_mm: pd.DataFrame) -> plt.Figure:
    """Section 2.10 - distribution of M1 scores across cell types."""
    nnPCA_mm = nnPCA_mm.copy()
    nnPCA_mm["celltype_annotation"] = nnPCA_mm["celltype_annotation"].astype(str)
    cell_types = nnPCA_mm["celltype_annotation"].unique()
    palette = {ct: COLORS[i % len(COLORS)] for i, ct in enumerate(cell_types)}
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    sns.histplot(data=nnPCA_mm, x="M1", hue="celltype_annotation",
                 multiple="layer", palette=palette,
                 alpha=0.6, bins=30, edgecolor=None, ax=ax)
    ax.set_xlabel("M1 Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of M1 Scores Across Cell Types")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Cell Type")
        leg.set_bbox_to_anchor((1.05, 1))
        leg._loc = 2
    fig.tight_layout()
    plt.close(fig)
    return fig
