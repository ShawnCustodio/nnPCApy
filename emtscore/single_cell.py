"""
Single-cell gene-set scoring and publication-quality plots (the Figure 3E style),
in a few function calls.

Public API
----------
log_cpm(counts)
    Total-library log1p(CPM) normalization.
score_signature(counts, genes, ...)
    Per-cell nnPCA score(s) + loadings for one gene set (total-library log-CPM +
    per-gene z-score + non-negative sparse PCA).
plot_em_scatter(scores, x, y, group=..., ci=True, ...)
    Per-cell x-vs-y scatter with per-group density contours and group centroids
    marked by 95% bootstrap confidence intervals (or mean +/- SD).
plot_signature_heatmap(counts, gene_groups, cell_group=..., ...)
    Z-scored expression heatmap of chosen genes across cells grouped by a
    categorical (e.g. time point), with a group colour strip and gene-group
    row annotation.

Plotting needs the optional `viz` extra (`pip install nnpcapy[viz]`); matplotlib
and SciPy are imported lazily inside the functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nsprcomp import nsprcomp


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _as_matrix(counts, var_names):
    """Return (float ndarray cells x genes, list of gene names)."""
    if isinstance(counts, pd.DataFrame):
        return counts.to_numpy(np.float64), list(counts.columns)
    counts = np.asarray(counts, dtype=np.float64)
    if var_names is None:
        raise ValueError("var_names is required when counts is not a DataFrame")
    if len(var_names) != counts.shape[1]:
        raise ValueError("len(var_names) must equal counts.shape[1]")
    return counts, list(var_names)


def log_cpm(counts):
    """Total-library log1p(counts / library * 1e4). counts: cells x genes."""
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * 1e4)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_signature(counts, genes, *, var_names=None, ncomp=1, standardize=True,
                    sign_ref=None, normalize="logcpm"):
    """Score a gene signature per cell with non-negative sparse PCA.

    Parameters
    ----------
    counts : DataFrame (cells x genes) or ndarray
        Raw counts of the *full* matrix (library size is taken over all columns).
    genes : list of str
        Signature genes; those present in the matrix are used.
    var_names : list of str, optional
        Gene names when `counts` is an ndarray.
    ncomp : int
        Number of components to extract.
    standardize : bool
        Z-score the returned scores across cells (recommended for plotting).
    sign_ref : array-like of length n_cells, optional
        Flip each component so its score correlates *positively* with this vector
        (e.g. pseudotime, or the per-cell mean signature expression).
    normalize : {"logcpm", "none"}
        Total-library log-CPM (default) or use the raw values.

    Returns
    -------
    scores : ndarray (n_cells, ncomp)
    loadings : ndarray (n_present_genes, ncomp), non-negative
    used_genes : list of str
    """
    mat, names = _as_matrix(counts, var_names)
    gi = {g: i for i, g in enumerate(names)}
    idx = [gi[g] for g in genes if g in gi]
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} of {len(genes)} signature genes present (need >= 3)")
    if normalize == "logcpm":
        lib = mat.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
        expr = np.log1p(mat[:, idx] / lib * 1e4)
    elif normalize == "none":
        expr = mat[:, idx]
    else:
        raise ValueError("normalize must be 'logcpm' or 'none'")
    Z = (expr - expr.mean(0)) / (expr.std(0) + 1e-12)
    res = nsprcomp(Z, ncomp=ncomp, nneg=True, center=True, scale_=False)
    scores = np.asarray(res["x"], float)[:, :ncomp].copy()
    load = np.asarray(res["rotation"], float)[:, :ncomp]
    if sign_ref is not None:
        ref = np.asarray(sign_ref, float)
        for k in range(scores.shape[1]):
            if np.std(scores[:, k]) > 0 and np.corrcoef(scores[:, k], ref)[0, 1] < 0:
                scores[:, k] *= -1; load = load.copy(); load[:, k] *= -1
    if standardize:
        scores = (scores - scores.mean(0)) / (scores.std(0) + 1e-12)
    return scores, load, [names[i] for i in idx]


def top_loading_genes(loadings, genes, n=5):
    """Top-`n` genes by |loading| for each component. Returns list-of-lists."""
    load = np.abs(np.asarray(loadings, float))
    if load.ndim == 1:
        load = load[:, None]
    out = []
    for k in range(load.shape[1]):
        order = np.argsort(-load[:, k])
        out.append([genes[i] for i in order[:n]])
    return out


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def _palette_map(groups, palette):
    import matplotlib.pyplot as plt
    if isinstance(palette, dict):
        return {g: palette[g] for g in groups}
    if isinstance(palette, (list, tuple)):
        return {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    cmap = plt.get_cmap(palette)
    if len(groups) == 1:
        return {groups[0]: cmap(0.5)}
    return {g: cmap(v) for g, v in zip(groups, np.linspace(0.05, 0.95, len(groups)))}


def _kde_fill(ax, x, y, color, *, levels=7, thresh=0.12):
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 5:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return
    px, py = 0.12 * (np.ptp(x) + 1e-9), 0.12 * (np.ptp(y) + 1e-9)
    xx, yy = np.mgrid[x.min() - px:x.max() + px:90j, y.min() - py:y.max() + py:90j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lv = np.linspace(thresh * zz.max(), zz.max(), levels)
    cmap = LinearSegmentedColormap.from_list("_k", [to_rgba(color, 0.15), to_rgba(color, 0.55)])
    ax.contourf(xx, yy, zz, levels=lv, cmap=cmap, antialiased=True, zorder=1, extend="neither")


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_em_scatter(scores, x, y, *, group=None, ci=True, n_boot=300, ci_level=95,
                    palette="viridis", group_order=None, kde=True, point_size=8,
                    point_alpha=0.2, ax=None, seed=0):
    """Per-cell x-vs-y scatter with per-group density and centroid uncertainty.

    Parameters
    ----------
    scores : DataFrame
        One row per cell; must contain columns `x`, `y`, and (optionally) `group`.
    x, y : str
        Column names to plot (e.g. an epithelial and a mesenchymal score).
    group : str, optional
        Column to colour / summarize by (e.g. "time"). If None, all cells are one group.
    ci : bool
        Mark each group centroid with a 95% bootstrap CI cross (True) or mean +/- SD (False).
    n_boot, ci_level, seed :
        Bootstrap settings (resamples cells within each group).
    palette : str | list | dict
        A matplotlib colormap name (sampled in group order), an explicit list, or a mapping.
    kde : bool
        Draw thresholded per-group kernel-density contours.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    df = scores if group is not None else scores.assign(_grp="all")
    g = group or "_grp"
    groups = list(group_order) if group_order else list(pd.unique(df[g]))
    cmap_t = _palette_map(groups, palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    if kde:
        for grp in groups:
            s = df[df[g] == grp]
            _kde_fill(ax, s[x].to_numpy(float), s[y].to_numpy(float), cmap_t[grp])
    for grp in groups:
        s = df[df[g] == grp]
        ax.scatter(s[x], s[y], s=point_size, alpha=point_alpha, color=cmap_t[grp],
                   edgecolors="none", zorder=2)
    rng = np.random.default_rng(seed)
    lo_q = (100 - ci_level) / 2.0; hi_q = 100 - lo_q
    handles = []
    for grp in groups:
        s = df[df[g] == grp]; xs = s[x].to_numpy(float); ys = s[y].to_numpy(float)
        mx, my = xs.mean(), ys.mean()
        if ci and len(xs) > 2:
            bx = np.empty(n_boot); by = np.empty(n_boot)
            for b in range(n_boot):
                j = rng.integers(0, len(xs), len(xs)); bx[b] = xs[j].mean(); by[b] = ys[j].mean()
            xlo, xhi = np.percentile(bx, [lo_q, hi_q]); ylo, yhi = np.percentile(by, [lo_q, hi_q])
        else:
            sx, sy = xs.std(), ys.std(); xlo, xhi, ylo, yhi = mx - sx, mx + sx, my - sy, my + sy
        xerr = [[mx - xlo], [xhi - mx]]; yerr = [[my - ylo], [yhi - my]]
        ax.errorbar(mx, my, xerr=xerr, yerr=yerr, fmt="none", ecolor="white",
                    elinewidth=5.5, capsize=8, capthick=5.5, zorder=3.5)
        ax.errorbar(mx, my, xerr=xerr, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=2.0, capsize=6, capthick=2.0, zorder=4)
        handles.append(ax.scatter([mx], [my], s=95, color=cmap_t[grp], edgecolors="white",
                                  linewidths=1.8, zorder=5, label=str(grp)))
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=0.15)
    if group is not None:
        lab = f"{group}  (● mean ± {ci_level}% CI)" if ci else f"{group}  (● mean ± SD)"
        ax.legend(handles=handles, title=lab, frameon=True, fontsize=9, title_fontsize=9,
                  loc="best", framealpha=0.9)
    return fig, ax


def plot_signature_heatmap(counts, gene_groups, *, var_names=None, cell_group=None,
                           group_order=None, group_label=None, cap=150, cmap="RdBu_r",
                           palette="viridis", group_colors=None, gene_labels=True,
                           vmax=None, fig=None, seed=0):
    """Z-scored (log-CPM) expression heatmap for chosen genes.

    Parameters
    ----------
    counts : DataFrame (cells x genes) or ndarray
        Raw counts of the full matrix.
    gene_groups : dict {label: [genes]} or list of genes
        Genes to show; a dict adds a labelled row annotation (e.g. {"E PC1": [...], "M PC1": [...]}).
    var_names : list of str, optional
        Gene names when `counts` is an ndarray.
    cell_group : array-like length n_cells, optional
        Categorical per-cell label used to order columns and draw a top colour strip
        (e.g. time point).
    group_order : list, optional
        Order for `cell_group` (and its colour strip).
    cap : int
        Max cells shown per group (evenly subsampled) to keep the heatmap legible.
    palette : colormap name | list | dict
        Colours for the `cell_group` strip.
    cmap : str
        Diverging colormap for the z-scored expression.

    Returns
    -------
    fig
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mat, names = _as_matrix(counts, var_names)
    present = set(names); gi = {g: i for i, g in enumerate(names)}
    if isinstance(gene_groups, dict):
        groups_items = [(lbl, [g for g in gl if g in present]) for lbl, gl in gene_groups.items()]
    else:
        groups_items = [("", [g for g in gene_groups if g in present])]
    genes = [g for _, gl in groups_items for g in gl]
    if not genes:
        raise ValueError("none of the requested genes are present in the matrix")
    gidx = np.array([gi[g] for g in genes])

    # column order (+ optional subsample within each cell group)
    n = mat.shape[0]; rng = np.random.default_rng(seed)
    if cell_group is not None:
        cg = np.asarray(cell_group)
        order = list(group_order) if group_order else list(pd.unique(cg))
        cols = []
        for grp in order:
            sel = np.where(cg == grp)[0]
            if len(sel) > cap:
                sel = sel[np.linspace(0, len(sel) - 1, cap).astype(int)]
            cols.extend(sel.tolist())
        cols = np.array(cols); cg_ord = cg[cols]
    else:
        cols = np.arange(n)
        if len(cols) > cap * 6:
            cols = cols[np.linspace(0, len(cols) - 1, cap * 6).astype(int)]
        order = None; cg_ord = None

    lib = mat[cols].sum(1, keepdims=True); lib[lib == 0] = 1.0
    expr = np.log1p(mat[np.ix_(cols, gidx)] / lib * 1e4)
    Z = ((expr - expr.mean(0)) / (expr.std(0) + 1e-12)).T          # genes x cells
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(Z), 98)) or 1.0

    has_strip = cell_group is not None
    has_rowann = isinstance(gene_groups, dict) and len(groups_items) > 1
    if fig is None:
        fig = plt.figure(figsize=(9, max(2.4, 0.32 * len(genes) + 1.2)))
    hr = [0.05, 1.0] if has_strip else [1e-6, 1.0]
    # columns: [row-annotation, heatmap, gene-label gap, colourbar]
    gene_gap = 0.14 if gene_labels else 1e-6
    wr = [0.05 if has_rowann else 1e-6, 1.0, gene_gap, 0.035]
    gs = fig.add_gridspec(2, 4, height_ratios=hr, width_ratios=wr, hspace=0.02, wspace=0.04)
    ax_hm = fig.add_subplot(gs[1, 1]); ax_cb = fig.add_subplot(gs[1, 3])

    # top colour strip for cell groups
    if has_strip:
        ax_ts = fig.add_subplot(gs[0, 1])
        cmap_t = _palette_map(order, palette)
        ax_ts.imshow(np.array([[mcolors.to_rgb(cmap_t[c]) for c in cg_ord]]),
                     aspect="auto", interpolation="none")
        ax_ts.set_xticks([]); ax_ts.set_yticks([])
        for sp in ax_ts.spines.values():
            sp.set_visible(False)
        b = 0
        for grp in order:
            k = int((cg_ord == grp).sum())
            ax_ts.text(b + k / 2, -0.8, str(grp), ha="center", va="bottom", fontsize=9)
            b += k
            if b < len(cg_ord):
                ax_hm.axvline(b - 0.5, color="white", lw=1.0)

    # left row-group annotation
    if has_rowann:
        ax_rg = fig.add_subplot(gs[1, 0])
        default = _palette_map([lbl for lbl, _ in groups_items], "tab10")
        gc = group_colors or default
        strip = []; off = 0
        for lbl, gl in groups_items:
            strip += [[mcolors.to_rgb(gc[lbl])]] * len(gl)
        ax_rg.imshow(np.array(strip), aspect="auto", interpolation="none")
        ax_rg.set_xticks([]); ax_rg.set_yticks([])
        for lbl, gl in groups_items:
            ax_rg.text(-0.8, off + (len(gl) - 1) / 2, str(lbl), rotation=90, va="center",
                       ha="center", fontsize=9, color=mcolors.to_rgb(gc[lbl]), fontweight="bold")
            off += len(gl)
        for sp in ax_rg.spines.values():
            sp.set_visible(False)

    im = ax_hm.imshow(Z, aspect="auto", cmap=cmap,
                      norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), interpolation="none")
    off = 0
    for lbl, gl in groups_items[:-1]:
        off += len(gl); ax_hm.axhline(off - 0.5, color="black", lw=1.2)
    ax_hm.set_xticks([])
    if gene_labels:
        ax_hm.set_yticks(range(len(genes))); ax_hm.set_yticklabels(genes, fontsize=8)
        ax_hm.yaxis.tick_right(); ax_hm.tick_params(right=False)
    else:
        ax_hm.set_yticks([])
    if cell_group is None:
        xlab = f"cells (n={len(cols)})"
    else:
        nm = group_label or (cell_group.name if hasattr(cell_group, "name") and cell_group.name else None)
        xlab = f"cells (n={len(cols)}), grouped by {nm}" if nm else f"cells (n={len(cols)})"
    ax_hm.set_xlabel(xlab, fontsize=10)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=ax_cb); cb.set_label("expression (z-score)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return fig


__all__ = ["log_cpm", "score_signature", "top_loading_genes",
           "plot_em_scatter", "plot_signature_heatmap"]
