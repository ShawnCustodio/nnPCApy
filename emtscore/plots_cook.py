"""
emtscore.plots_cook  -  3.4 / 3.5 / 3.6 plots
"""

from __future__ import annotations

from colorsys import hls_to_rgb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.stats import gaussian_kde, pearsonr

from utility.data_paths import resolve_data_file

from .nsprcomp import nsprcomp

# helpers ---------------------------------------------------------------

def _gg_hue_palette(n: int) -> list[tuple[float, float, float]]:
    """ggplot-style hue palette, evenly spaced in HSL."""
    hues = np.linspace(15, 375, n + 1)[:-1]
    return [hls_to_rgb(h / 360.0, 0.55, 1.0) for h in hues]


def _log_norm_3_4(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(counts / lib * 1e4)


def _nnpca_scores_3_4(adata, genes: list[str], ncomp: int = 1) -> np.ndarray:
    available = [g for g in genes if g in adata.var_names]
    if len(available) < 3:
        raise ValueError(
            f"Only {len(available)} signature genes present in adata."
        )
    X = adata[:, available].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    Xn = _log_norm_3_4(X)
    Z = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-12)
    return nsprcomp(Z, ncomp=ncomp, nneg=True, center=True, scale_=False)["x"]


def _draw_kde_fill(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                   color: tuple, levels: int = 8) -> None:
    """Filled per-group KDE density contours (R geom_density_2d_filled-style)."""
    if len(x) < 5:
        return
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 5:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return
    pad_x = 0.10 * (x.max() - x.min() + 1e-9)
    pad_y = 0.10 * (y.max() - y.min() + 1e-9)
    xx, yy = np.mgrid[x.min() - pad_x:x.max() + pad_x:80j,
                      y.min() - pad_y:y.max() + pad_y:80j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    cmap = LinearSegmentedColormap.from_list(
        "_kdef",
        [to_rgba(color, 0.00), to_rgba(color, 0.55)],
    )
    ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, antialiased=True,
                zorder=1)


def _scatter_panel(ax: plt.Axes, df: pd.DataFrame, x: str, y: str,
                    hue: str, present: list[str], palette: list,
                    title: str) -> None:

    # Layer 1 - per-group KDE density.
    for cat, color in zip(present, palette):
        sub = df[df[hue] == cat]
        if len(sub) >= 5:
            _draw_kde_fill(ax, sub[x].to_numpy(), sub[y].to_numpy(), color)

    # Layer 2 - scatter (low alpha + small dots so the layers above show).
    for cat, color in zip(present, palette):
        sub = df[df[hue] == cat]
        ax.scatter(sub[x], sub[y], s=8, alpha=0.22,
                   color=color, edgecolors="none", zorder=2)

    # Layers 3 + 4 - cross + centroid per group.
    centroid_handles = []
    for cat, color in zip(present, palette):
        sub = df[df[hue] == cat]
        if sub.empty:
            continue
        mx, my = float(sub[x].mean()), float(sub[y].mean())
        sx, sy = float(sub[x].std()),  float(sub[y].std())
        # white halo bars
        ax.plot([mx - sx, mx + sx], [my, my],
                color="white", linewidth=7.0,
                solid_capstyle="butt", zorder=3.5)
        ax.plot([mx, mx], [my - sy, my + sy],
                color="white", linewidth=7.0,
                solid_capstyle="butt", zorder=3.5)
        # black bars on top.
        ax.plot([mx - sx, mx + sx], [my, my],
                color="black", linewidth=4.0,
                solid_capstyle="butt", zorder=4)
        ax.plot([mx, mx], [my - sy, my + sy],
                color="black", linewidth=4.0,
                solid_capstyle="butt", zorder=4)
        # centroid marker: filled circle with white edge.
        h = ax.scatter([mx], [my], s=260, color=color,
                        edgecolors="white", linewidths=2.8,
                        zorder=5, label=cat)
        centroid_handles.append(h)

    ax.set_xlabel(x, style="italic")
    ax.set_ylabel(y, style="italic")
    ax.set_title(title)
    ax.legend(handles=centroid_handles, title=hue, loc="best",
              fontsize=8, frameon=True, facecolor="white",
              edgecolor="0.7")
    ax.grid(alpha=0.15)


# 3.4  E-vs-M / PC1-vs-PC2 panels ---------------------------------------

def plot_em_pc_panels_cook(adata_sc=None) -> tuple:
    """v2 3.4 - three vignette-style panels for Cook A549_TGFB1."""
    if adata_sc is None:
        # Prefer the lightweight em_expr loader (works out-of-the-box with the
        # bundled data); fall back to the raw 4-file loader if needed.
        from .sc import load_cook_adatas
        adata_sc = load_cook_adatas(verbose=False,
                                    datasets=["A549_TGFB1"])["A549_TGFB1"]

    E_genes = pd.read_csv(resolve_data_file("Panchy_et_al_E_signature.csv"))["GeneName"].tolist()
    M_genes = pd.read_csv(resolve_data_file("Panchy_et_al_M_signature.csv"))["GeneName"].tolist()

    pt = adata_sc.obs["Pseudotime"].values

    Escore = _nnpca_scores_3_4(adata_sc, E_genes, ncomp=1)[:, 0]
    if np.corrcoef(pt, Escore)[0, 1] > 0:
        Escore = -Escore

    M_pcs = _nnpca_scores_3_4(adata_sc, M_genes, ncomp=2)
    Mscore, Mscore_PC2 = M_pcs[:, 0], M_pcs[:, 1]
    if np.corrcoef(pt, Mscore)[0, 1] < 0:
        Mscore = -Mscore

    adata_sc.obs["Escore"]     = Escore
    adata_sc.obs["Mscore"]     = Mscore
    adata_sc.obs["Mscore_PC1"] = Mscore
    adata_sc.obs["Mscore_PC2"] = Mscore_PC2

    plot_df = (
        adata_sc.obs[["Time", "Escore", "Mscore", "Mscore_PC1", "Mscore_PC2"]]
        .dropna().copy()
    )
    plot_df["Time"] = plot_df["Time"].astype(str)

    TIME_ORDER = ["0d", "8h", "8h_rm", "1d", "1d_rm",
                  "3d", "3d_rm", "7d"]
    present = [t for t in TIME_ORDER if t in set(plot_df["Time"])]
    palette = _gg_hue_palette(len(present))

    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    _scatter_panel(ax_a, plot_df, "Escore", "Mscore", "Time",
                    present, palette,
                    "section 3.4  E score vs M score  -  Cook A549_TGFB1")
    fig_a.tight_layout()
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    _scatter_panel(ax_b, plot_df, "Mscore_PC1", "Mscore_PC2", "Time",
                    present, palette,
                    "section 3.4  M-score PC1 vs PC2  -  Cook A549_TGFB1")
    fig_b.tight_layout()
    plt.close(fig_b)

    fig_c, axes = plt.subplots(1, 2, figsize=(13, 6))
    _scatter_panel(axes[0], plot_df, "Escore", "Mscore", "Time",
                    present, palette, "E score vs M score")
    _scatter_panel(axes[1], plot_df, "Mscore_PC1", "Mscore_PC2", "Time",
                    present, palette, "M-score PC1 vs PC2")
    fig_c.suptitle("Cook_et_al  -  A549_TGFB1", fontsize=13, y=1.02)
    fig_c.tight_layout()
    plt.close(fig_c)

    return fig_a, fig_b, fig_c, adata_sc


# 3.5  Stemness + Senescence --------------------------------------------

def _load_tsv_genes(name: str, column: str = "gene_symbol") -> list[str]:
    """Load a one-column TSV signature file and return the gene list."""
    return pd.read_csv(resolve_data_file(name), sep="\t")[column].dropna().tolist()


def _signature_score(adata, genes: list[str]) -> np.ndarray:
    """Per-cell score: log1p(CPM) -> per-gene z-score -> mean over signature."""
    available = [g for g in genes if g in adata.var_names]
    if not available:
        raise ValueError("No signature genes present in adata.")
    X = adata[:, available].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    Xn = np.log1p(X / lib * 1e4)
    Z = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-12)
    return Z.mean(axis=1)


def compute_stem_senescence(adata):
    """v2 3.5 - add Stemness_Score and Senescence_Score columns to adata.obs."""
    stem_genes = _load_tsv_genes("stemsig.tsv")
    sen_genes  = _load_tsv_genes("cellular_senescence_sig.tsv")
    adata.obs["Stemness_Score"]   = _signature_score(adata, stem_genes)
    adata.obs["Senescence_Score"] = _signature_score(adata, sen_genes)
    return adata


# 3.6  Stemness/Senescence vs E/M relationship plots --------------------

def _annotated_scatter(ax: plt.Axes, x: pd.Series, y: pd.Series,
                        xlabel: str, ylabel: str, color: str = "#1f77b4",
                        title: str | None = None) -> None:
    valid = (~x.isna()) & (~y.isna())
    xs, ys = x[valid].values, y[valid].values
    r, p = pearsonr(xs, ys)
    ax.scatter(xs, ys, s=10, alpha=0.45, color=color, edgecolors="none")
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        xfit = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xfit, m * xfit + b, color="black", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(f"{title}\n(r={r:.2f}, p={p:.1e})", fontsize=10)
    else:
        ax.set_title(f"r={r:.2f}, p={p:.1e}", fontsize=10)
    ax.grid(alpha=0.2)


def plot_stemness_vs_senescence(adata) -> plt.Figure:
    df = adata.obs[["Stemness_Score", "Senescence_Score"]].dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _annotated_scatter(
        ax, df["Stemness_Score"], df["Senescence_Score"],
        xlabel="Stemness_Score", ylabel="Senescence_Score",
        color="#7b3294",
        title="section 3.6  Stemness vs Senescence (per cell)",
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_em_vs_stem_sen(adata) -> plt.Figure:
    df = adata.obs[["Escore", "Mscore",
                     "Stemness_Score", "Senescence_Score"]].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _annotated_scatter(axes[0, 0], df["Escore"], df["Stemness_Score"],
                        "Escore", "Stemness_Score",
                        color="#1f77b4", title="Stemness vs E")
    _annotated_scatter(axes[0, 1], df["Mscore"], df["Stemness_Score"],
                        "Mscore", "Stemness_Score",
                        color="#1f77b4", title="Stemness vs M")
    _annotated_scatter(axes[1, 0], df["Escore"], df["Senescence_Score"],
                        "Escore", "Senescence_Score",
                        color="#d62728", title="Senescence vs E")
    _annotated_scatter(axes[1, 1], df["Mscore"], df["Senescence_Score"],
                        "Mscore", "Senescence_Score",
                        color="#d62728", title="Senescence vs M")
    fig.suptitle("section 3.6  Stemness / Senescence vs EMT axes", fontsize=13)
    fig.tight_layout()
    plt.close(fig)
    return fig


__all__ = [
    "compute_stem_senescence",
    "plot_em_pc_panels_cook",
    "plot_em_vs_stem_sen",
    "plot_stemness_vs_senescence",
]
