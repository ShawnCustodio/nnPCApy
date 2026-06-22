"""
Section 2.11 - Heatmaps (full M-signature and top PC1/PC2 driver genes)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from numpy.linalg import svd as npsvd

COLORS = [
    "#F87189", "#CE9031", "#A48CF5", "#97A430", "#39A7D0",
    "#E57D5F", "#84C7B9", "#E1AF64", "#C26CCF", "#B0BF43",
    "#57C3E8", "#F29D9E", "#92AAE6",
]
def plot_full_m_heatmap(geneExp: pd.DataFrame,
                        M_sig: pd.DataFrame,
                        nnPCA_em: pd.DataFrame,
                        verbose: bool = True) -> plt.Figure:
    """v2 2.11.0 - full M-signature heatmap (all Panchy M genes detected)."""
    M_full = [g for g in M_sig["GeneName"].tolist() if g in geneExp.columns]
    if verbose:
        print(f"Full-M heatmap: {len(M_full)} of {len(M_sig)} signature genes present")

    expr_full = geneExp[M_full].T  # genes × samples
    valid_samples = [s for s in expr_full.columns if s in nnPCA_em.index.values]
    expr_full = expr_full[valid_samples]

    col_order = leaves_list(linkage(pdist(expr_full.T.values), method="average"))
    row_order = leaves_list(linkage(pdist(expr_full.values),   method="average"))
    expr_ord  = expr_full.iloc[row_order, col_order]
    sample_order = expr_ord.columns.tolist()
    gene_order   = expr_ord.index.tolist()

    n_genes   = len(gene_order)
    n_samples = len(sample_order)
    fig_w = max(14, n_samples * 0.09)
    fig_h = max(10, n_genes   * 0.10 + 3)
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, hm_w = 0.08, 0.82
    top = 0.97
    dend_h = 0.10; dend_b = top - dend_h
    anno_h = 0.018; anno_b = dend_b - anno_h - 0.004
    hm_h   = 0.80; hm_b   = anno_b - hm_h - 0.004

    ax_dend = fig.add_axes([left, dend_b, hm_w, dend_h])
    ax_anno = fig.add_axes([left, anno_b, hm_w, anno_h])
    ax_hm   = fig.add_axes([left, hm_b,   hm_w, hm_h])

    col_link = linkage(pdist(expr_full.T.values[col_order]), method="average")
    dendrogram(col_link, ax=ax_dend, color_threshold=0,
               above_threshold_color="black", no_labels=True)
    ax_dend.set_facecolor("white"); ax_dend.axis("off")

    cell_types = nnPCA_em.loc[sample_order, "celltype_annotation"].values
    unique_ct  = sorted(pd.Series(cell_types).dropna().unique())
    ct_palette = dict(zip(unique_ct, COLORS[: len(unique_ct)]))
    anno_rgb = np.array([[mcolors.to_rgb(ct_palette.get(c, "#CCCCCC"))]
                          for c in cell_types]).transpose(1, 0, 2)
    ax_anno.imshow(anno_rgb, aspect="auto", interpolation="none")
    ax_anno.set_xticks([]); ax_anno.set_yticks([])
    for sp in ax_anno.spines.values(): sp.set_visible(False)

    Z = expr_ord.subtract(expr_ord.mean(axis=1), axis=0).divide(
            expr_ord.std(axis=1).replace(0, 1), axis=0)
    vmax = float(np.nanpercentile(np.abs(Z.values), 99)) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    hm_img = ax_hm.imshow(Z.values, aspect="auto", cmap="RdBu_r",
                          norm=norm, interpolation="none")
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    for sp in ax_hm.spines.values(): sp.set_visible(False)

    ax_hm_r = ax_hm.twinx()
    ax_hm_r.set_ylim(ax_hm.get_ylim())
    label_stride = max(1, n_genes // 60)
    tick_pos = list(range(0, n_genes, label_stride))
    ax_hm_r.set_yticks(tick_pos)
    ax_hm_r.set_yticklabels([gene_order[i] for i in tick_pos], fontsize=6)
    ax_hm_r.tick_params(right=False, length=0)
    for sp in ax_hm_r.spines.values(): sp.set_visible(False)

    cb_ax = fig.add_axes([left + hm_w + 0.015, hm_b + 0.3, 0.015, 0.2])
    cb = plt.colorbar(hm_img, cax=cb_ax)
    cb.set_label("Expr (z-score)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    leg_ax = fig.add_axes([left + hm_w + 0.015, hm_b + 0.55, 0.12, 0.20])
    leg_ax.axis("off")
    handles = [plt.Line2D([], [], marker="s", linestyle="None",
                          color=ct_palette[c], markersize=8, label=c)
               for c in unique_ct]
    leg_ax.legend(handles=handles, title="Cell type",
                  fontsize=7, title_fontsize=8, loc="upper left",
                  frameon=False)

    fig.suptitle("Panchy et al. M-signature - full heatmap",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.11   Top PC1/PC2 driver-gene heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_pc_driver_heatmap(geneExp: pd.DataFrame,
                            M_sig: pd.DataFrame,
                            nnPCA_em: pd.DataFrame,
                            nnPCA_mm: pd.DataFrame) -> plt.Figure:
    """v2 2.11 - driver-gene heatmap (top PC1 + top PC2 genes, M1/M2 strips)."""
    top_pc1 = [g for g in ["VIM","LGALS1","FSTL1","MSN","CAV1","TPM2","CALD1","PDGFC","WWTR1","EMP3"]
               if g in geneExp.columns]
    top_pc2 = [g for g in ["MFAP4","CXCR4","FXYD6","SPARC","TCF4","IGFBP5","TUBA1A","FHL1","FYN","DPYSL3"]
               if g in geneExp.columns]
    all_top_genes = top_pc1 + top_pc2
    gene_pc_label = {g: "PC1" for g in top_pc1}
    gene_pc_label.update({g: "PC2" for g in top_pc2})

    M_genes_present = [g for g in M_sig["GeneName"].tolist() if g in geneExp.columns]
    X_c = geneExp[M_genes_present].values - geneExp[M_genes_present].values.mean(axis=0)
    U, s_sv, Vt = npsvd(X_c, full_matrices=False)
    pc1_loadings = pd.Series(np.abs(Vt[0]), index=M_genes_present)
    pc2_loadings = pd.Series(np.abs(Vt[1]), index=M_genes_present)
    pc1_bar = pc1_loadings[all_top_genes].values
    pc2_bar = pc2_loadings[all_top_genes].values

    expr_sub = geneExp[all_top_genes].T
    valid_in_expr = [s for s in geneExp.index if s in nnPCA_mm.index.values]
    expr_filtered = expr_sub[valid_in_expr]
    col_order = leaves_list(linkage(pdist(expr_filtered.T.values), method="average"))
    expr_ordered = expr_filtered.iloc[:, col_order]
    sample_order = expr_ordered.columns.tolist()

    valid_samples  = [s for s in sample_order if s in nnPCA_mm.index.values]
    scores_aligned = nnPCA_mm.loc[valid_samples]
    m1_vals = scores_aligned["M1"].values
    m2_vals = scores_aligned["M2"].values
    expr_ordered = expr_ordered[valid_samples]

    expr_vals = expr_ordered.values
    data_min, data_max = expr_vals.min(), expr_vals.max()
    data_mid = (data_min + data_max) / 2
    norm_hm  = mcolors.TwoSlopeNorm(vmin=data_min, vcenter=data_mid, vmax=data_max)

    n_genes, n_samples = len(all_top_genes), len(valid_samples)
    fig_w = max(12, n_samples * 0.13)
    fig_h = max(8,  n_genes   * 0.38 + 3)
    fig   = plt.figure(figsize=(fig_w, fig_h))

    left_hm = 0.09; hm_w = 0.67
    pc1_l = left_hm + hm_w + 0.008; pc_w = 0.022
    pc2_l = pc1_l + pc_w + 0.005
    cb_l  = pc2_l + pc_w + 0.045
    top   = 0.96
    dend_h = 0.12; dend_b = top - dend_h
    m1_h = 0.034;  m1_b   = dend_b - m1_h - 0.004
    m2_h = 0.034;  m2_b   = m1_b   - m2_h - 0.003
    hm_h = 0.72;   hm_b   = m2_b   - hm_h - 0.004

    col_link = linkage(pdist(expr_filtered.T.values[col_order]), method="average")

    ax_dend  = fig.add_axes([left_hm,   dend_b, hm_w,   dend_h])
    ax_m1    = fig.add_axes([left_hm,   m1_b,   hm_w,   m1_h])
    ax_m2    = fig.add_axes([left_hm,   m2_b,   hm_w,   m2_h])
    ax_label = fig.add_axes([0.04,      hm_b,   0.035,  hm_h])
    ax_hm    = fig.add_axes([left_hm,   hm_b,   hm_w,   hm_h])
    ax_pc1   = fig.add_axes([pc1_l,     hm_b,   pc_w,   hm_h])
    ax_pc2   = fig.add_axes([pc2_l,     hm_b,   pc_w,   hm_h])

    dendrogram(col_link, ax=ax_dend, color_threshold=0,
               above_threshold_color="black", no_labels=True)
    ax_dend.set_facecolor("white"); ax_dend.axis("off")

    m1_norm = (m1_vals - m1_vals.min()) / (m1_vals.max() - m1_vals.min() + 1e-9)
    ax_m1.imshow(np.array([[plt.cm.Greys(v) for v in m1_norm]]), aspect="auto", interpolation="none")
    ax_m1.set_yticks([0]); ax_m1.set_yticklabels(["M_PC1_score"], fontsize=7)
    ax_m1.set_xticks([]); ax_m1.tick_params(left=False, length=0)
    for sp in ax_m1.spines.values(): sp.set_visible(False)

    m2_norm = (m2_vals - m2_vals.min()) / (m2_vals.max() - m2_vals.min() + 1e-9)
    ax_m2.imshow(np.array([[plt.cm.RdPu(0.1 + 0.9 * v) for v in m2_norm]]), aspect="auto", interpolation="none")
    ax_m2.set_yticks([0]); ax_m2.set_yticklabels(["M_PC2_score"], fontsize=7)
    ax_m2.set_xticks([]); ax_m2.tick_params(left=False, length=0)
    for sp in ax_m2.spines.values(): sp.set_visible(False)

    label_colors = ["#2166AC" if gene_pc_label.get(g) == "PC1" else "#D6604D" for g in all_top_genes]
    ax_label.imshow(np.array([[mcolors.to_rgb(c)] for c in label_colors]), aspect="auto", interpolation="none")
    ax_label.set_xticks([]); ax_label.set_yticks([])
    ax_label.set_xlabel("Label", fontsize=7, labelpad=3)
    for sp in ax_label.spines.values(): sp.set_visible(False)

    hm_img = ax_hm.imshow(expr_ordered.values, aspect="auto", cmap="RdBu_r", norm=norm_hm, interpolation="none")
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    for sp in ax_hm.spines.values(): sp.set_visible(False)
    ax_hm_r = ax_hm.twinx()
    ax_hm_r.set_ylim(ax_hm.get_ylim())
    ax_hm_r.set_yticks(range(n_genes)); ax_hm_r.set_yticklabels(all_top_genes, fontsize=8)
    ax_hm_r.tick_params(right=False, length=0)
    for sp in ax_hm_r.spines.values(): sp.set_visible(False)

    pc1_norm = pc1_bar / (pc1_bar.max() + 1e-9)
    ax_pc1.imshow(np.array([[plt.cm.Blues(v)] for v in pc1_norm]), aspect="auto", interpolation="none")
    ax_pc1.set_xticks([]); ax_pc1.set_yticks([]); ax_pc1.set_xlabel("PC1", fontsize=7, labelpad=3)
    for sp in ax_pc1.spines.values(): sp.set_visible(False)

    pc2_norm = pc2_bar / (pc2_bar.max() + 1e-9)
    ax_pc2.imshow(np.array([[plt.cm.Reds(v)] for v in pc2_norm]), aspect="auto", interpolation="none")
    ax_pc2.set_xticks([]); ax_pc2.set_yticks([]); ax_pc2.set_xlabel("PC2", fontsize=7, labelpad=3)
    for sp in ax_pc2.spines.values(): sp.set_visible(False)

    ax_leg = fig.add_axes([cb_l, top - 0.09, 0.12, 0.08])
    ax_leg.axis("off")
    ax_leg.legend(handles=[mpatches.Patch(color="#2166AC", label="M_PC1"),
                            mpatches.Patch(color="#D6604D", label="M_PC2")],
                  title="Label", fontsize=7, title_fontsize=8, loc="upper left", frameon=False)
    ax_cb1 = fig.add_axes([cb_l, top-0.30, 0.025, 0.17])
    ColorbarBase(ax_cb1, cmap=plt.cm.Greys,
                 norm=mcolors.Normalize(vmin=m1_vals.min(), vmax=m1_vals.max()),
                 orientation="vertical").set_label("M_PC1_score", fontsize=7)
    ax_cb1.tick_params(labelsize=6)
    ax_cb2 = fig.add_axes([cb_l, top-0.52, 0.025, 0.17])
    cb2 = ColorbarBase(ax_cb2, cmap=plt.cm.RdBu_r,
                       norm=mcolors.TwoSlopeNorm(vmin=data_min, vcenter=data_mid, vmax=data_max),
                       orientation="vertical")
    cb2.set_label("Gene Expr", fontsize=7)
    cb2.set_ticks([data_min, data_mid, data_max])
    cb2.set_ticklabels([f"{data_min:.0f}", f"{data_mid:.0f}", f"{data_max:.0f}"])
    ax_cb2.tick_params(labelsize=6)
    ax_cb3 = fig.add_axes([cb_l, top-0.69, 0.025, 0.13])
    ColorbarBase(ax_cb3, cmap=plt.cm.Blues,
                 norm=mcolors.Normalize(vmin=0, vmax=pc1_bar.max()),
                 orientation="vertical").set_label("PC1", fontsize=7)
    ax_cb3.tick_params(labelsize=6)
    ax_cb4 = fig.add_axes([cb_l, top-0.84, 0.025, 0.11])
    ColorbarBase(ax_cb4, cmap=plt.cm.RdPu,
                 norm=mcolors.Normalize(vmin=m2_vals.min(), vmax=m2_vals.max()),
                 orientation="vertical").set_label("M_PC2_score", fontsize=7)
    ax_cb4.tick_params(labelsize=6)
    plt.close(fig)
    return fig
