"""
Section 3.1, 3.1-plot, 3.2, 3.2-plot: Single-cell analysis
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.mixture import GaussianMixture

from utility.data_paths import resolve_data_file
from utility.load_cook2020 import load_all_cook2020, load_cook2020

from .nsprcomp import nsprcomp


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Cook 2020 AnnData loader (lightweight, em_expr-only)
# ─────────────────────────────────────────────────────────────────────────────

_COOK_META_COLS = ["Pseudotime", "Treatment", "Time", "Cluster"]


def _adata_from_em_expr(path) -> "anndata.AnnData":
    """Build a minimal AnnData from a Cook ``*_em_expr.csv`` file.

    The CSV is shaped (cells × [E+M genes + 4 metadata cols]); cell barcodes
    are the row index. The resulting AnnData has:
        - ``.X``         expression matrix as float32
        - ``.var_names`` gene symbols (E+M signature genes)
        - ``.obs``       Pseudotime / Treatment / Time / Cluster
        - ``.obs_names`` cell barcodes
    """
    import anndata
    df = pd.read_csv(path, index_col=0)
    meta_cols = [c for c in _COOK_META_COLS if c in df.columns]
    expr_cols = [c for c in df.columns if c not in meta_cols]
    X = df[expr_cols].to_numpy(dtype=np.float32, copy=True)
    obs = df[meta_cols].copy()
    var = pd.DataFrame(index=pd.Index(expr_cols, name="gene"))
    return anndata.AnnData(X=X, obs=obs, var=var)


def load_cook_adatas(verbose: bool = True,
                     datasets: list[str] | None = None) -> dict:
    """Load Cook 2020 datasets as AnnData objects.

    Tries the bundled lightweight ``cook2020/{name}_em_expr.csv`` files first
    (these ship with the repo and contain the E+M signature genes plus
    Pseudotime / Treatment / Time / Cluster metadata — sufficient for every
    §3.x step in the notebook). Falls back to the full 4-file Cook format
    (``{name}_counts.csv`` + genes/cells/metadata) if a user has placed the
    raw data on disk.
    """
    DATASETS = datasets or ["A549_EGF", "A549_TGFB1", "A549_TNF"]
    adatas: dict = {}
    for name in DATASETS:
        try:
            path = resolve_data_file(f"cook2020/{name}_em_expr.csv")
        except FileNotFoundError:
            path = None

        if path is not None:
            adatas[name] = _adata_from_em_expr(path)
        else:
            # Raw 4-file format fallback (rarely available in this repo).
            try:
                adatas[name] = load_cook2020(name)
            except FileNotFoundError as e:
                if verbose:
                    print(f"  [skip] {name}: {e}")
                continue

    if verbose:
        for name, adata in adatas.items():
            pt = adata.obs["Pseudotime"]
            print(f"{name}: {adata.shape},  Pseudotime: "
                  f"{pt.min():.3f} - {pt.max():.3f}")
    return adatas


# ─────────────────────────────────────────────────────────────────────────────
# 3.2   GMM in E-M space across Cook datasets
# ─────────────────────────────────────────────────────────────────────────────

def _lognorm_3_2(expr: pd.DataFrame, scale: float = 10_000) -> np.ndarray:
    lib_size = expr.sum(axis=1).values[:, None]
    lib_size = np.where(lib_size == 0, 1, lib_size)
    return np.log1p(expr.values / lib_size * scale)


def _scale_genes_3_2(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std  = X.std(axis=0); std[std == 0] = 1
    return (X - mean) / std


def _build_gmm_em(df: pd.DataFrame, E_genes: list[str], M_genes: list[str]):
    meta_cols = ["Pseudotime", "Treatment", "Time", "Cluster"]
    expr_cols = df.columns.difference(meta_cols)
    pt = df["Pseudotime"].values

    full_expr = _lognorm_3_2(df[expr_cols])
    full_expr = pd.DataFrame(full_expr, index=df.index, columns=expr_cols)

    E_mat = _scale_genes_3_2(full_expr[[g for g in E_genes if g in full_expr.columns]].values)
    M_mat = _scale_genes_3_2(full_expr[[g for g in M_genes if g in full_expr.columns]].values)

    E_raw = nsprcomp(E_mat, ncomp=1)["x"][:, 0]
    M_raw = nsprcomp(M_mat, ncomp=1)["x"][:, 0]

    if np.corrcoef(pt, E_raw)[0, 1] > 0:
        E_raw = -E_raw
    if np.corrcoef(pt, M_raw)[0, 1] < 0:
        M_raw = -M_raw

    E_score = (E_raw - E_raw.mean()) / E_raw.std()
    M_score = (M_raw - M_raw.mean()) / M_raw.std()
    em = np.column_stack([E_score, M_score])

    gmm = GaussianMixture(n_components=3, random_state=42, n_init=5).fit(em)
    labels = gmm.predict(em)
    order  = np.argsort(gmm.means_[:, 0])
    state_map = {order[0]: "M", order[1]: "EM1", order[2]: "E"}
    state_labels = np.array([state_map[l] for l in labels])
    return em, state_labels


def build_gmm_in_em_space(datasets: list[str] | None = None) -> dict[str, dict]:
    """v2 3.2 - run GMM in E-M space across Cook datasets and plot."""
    DATASETS = datasets or ["A549_TGFB1", "A549_TNF", "A549_EGF"]
    gmm_colors = {"E": "#F8766D", "EM1": "#619CFF", "M": "#00BA38"}

    E_genes = pd.read_csv(resolve_data_file("Panchy_et_al_E_signature.csv"))["GeneName"].tolist()
    M_genes = pd.read_csv(resolve_data_file("Panchy_et_al_M_signature.csv"))["GeneName"].tolist()

    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "axes.spines.top": False,  "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 12,
    })

    gmm_results: dict[str, dict] = {}
    fig, axes = plt.subplots(1, len(DATASETS),
                              figsize=(7 * len(DATASETS), 6), sharey=False)
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, name in zip(axes, DATASETS):
        try:
            df = pd.read_csv(resolve_data_file(f"cook2020/{name}_em_expr.csv"),
                             index_col=0)
        except FileNotFoundError:
            ax.set_title(f"{name} (file missing)", fontsize=12)
            ax.axis("off")
            continue

        em, state_labels = _build_gmm_em(df, E_genes, M_genes)
        gmm_results[name] = {
            "em": em, "states": state_labels,
            "pseudotime": df["Pseudotime"].values,
            "time": df["Time"].values if "Time" in df.columns else None,
            "index": df.index,
        }

        for state in ["E", "EM1", "M"]:
            mask = state_labels == state
            ax.scatter(em[mask, 0], em[mask, 1],
                       s=14, alpha=0.65, color=gmm_colors[state],
                       label=state, linewidths=0)

        ax.set_xlabel("Escore", fontsize=13)
        ax.set_ylabel("Mscore", fontsize=13)
        ax.set_title(f"{name} - GMM clustering", fontsize=14, fontweight="normal")
        ax.legend(title="Cluster", title_fontsize=11, fontsize=10,
                  frameon=False, markerscale=1.5, loc="upper right")

    fig.tight_layout()
    return gmm_results


# ─────────────────────────────────────────────────────────────────────────────
# 3.1-plot   EMT score vs Pseudotime 
# ─────────────────────────────────────────────────────────────────────────────

def _loess_1d(x: np.ndarray, y: np.ndarray, frac: float = 0.4,
              n_eval: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Tricube-weighted local linear regression — LOESS without statsmodels."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); x = x[order]; y = y[order]
    n = len(x)
    k = max(int(np.ceil(frac * n)), 5)
    x_eval = np.linspace(x.min(), x.max(), n_eval)
    y_eval = np.empty(n_eval)
    for i, xi in enumerate(x_eval):
        d = np.abs(x - xi)
        h = np.partition(d, k - 1)[k - 1]
        if h == 0:
            h = 1e-9
        u = np.clip(d / h, 0, 1)
        w = (1 - u**3) ** 3
        W = w.sum()
        mx = (w * x).sum() / W
        my = (w * y).sum() / W
        sxy = (w * (x - mx) * (y - my)).sum()
        sxx = (w * (x - mx) ** 2).sum()
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        y_eval[i] = a + b * xi
    return x_eval, y_eval


def plot_emt_vs_pseudotime(datasets: list[str] | None = None,
                            verbose: bool = False) -> plt.Figure:
    """v2 3.1 - EMT score (= M_score - E_score) vs Pseudotime per condition.
    """
    DATASETS = datasets or ["A549_EGF", "A549_TGFB1"]
    palette = {"A549_EGF": "#F8766D", "A549_TGFB1": "#00BFC4",
               "A549_TNF": "#7CAE00"}

    E_genes = pd.read_csv(resolve_data_file("Panchy_et_al_E_signature.csv"))["GeneName"].tolist()
    M_genes = pd.read_csv(resolve_data_file("Panchy_et_al_M_signature.csv"))["GeneName"].tolist()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)

    for name in DATASETS:
        try:
            df = pd.read_csv(resolve_data_file(f"cook2020/{name}_em_expr.csv"),
                              index_col=0)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: _em_expr.csv missing")
            continue

        meta_cols = ["Pseudotime", "Treatment", "Time", "Cluster"]
        expr_cols = df.columns.difference(meta_cols)
        pt = df["Pseudotime"].values

        full = _lognorm_3_2(df[expr_cols])
        full = pd.DataFrame(full, index=df.index, columns=expr_cols)

        E_mat = _scale_genes_3_2(full[[g for g in E_genes if g in full.columns]].values)
        M_mat = _scale_genes_3_2(full[[g for g in M_genes if g in full.columns]].values)
        E_raw = nsprcomp(E_mat, ncomp=1)["x"][:, 0]
        M_raw = nsprcomp(M_mat, ncomp=1)["x"][:, 0]
        if np.corrcoef(pt, E_raw)[0, 1] > 0:
            E_raw = -E_raw
        if np.corrcoef(pt, M_raw)[0, 1] < 0:
            M_raw = -M_raw
        E_score = (E_raw - E_raw.mean()) / E_raw.std()
        M_score = (M_raw - M_raw.mean()) / M_raw.std()
        emt = M_score - E_score

        xs, ys = _loess_1d(pt, emt, frac=0.5, n_eval=300)
        ax.plot(xs, ys, lw=2.6, color=palette.get(name, None), label=name)

    ax.set_xlabel("Pseudotime", fontsize=12)
    ax.set_ylabel("EMT_score", fontsize=12)
    ax.legend(title="Condition", frameon=False, fontsize=10, title_fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.2-plot   GMM cluster ↔ true Time Sankey/alluvial per dataset
# ─────────────────────────────────────────────────────────────────────────────

def _alluvial_curve(x0: float, y0_top: float, y0_bot: float,
                     x1: float, y1_top: float, y1_bot: float,
                     ax, color, alpha: float = 0.55) -> None:
    """Draw a single filled flow band (cubic Bezier on each edge)."""
    from matplotlib.path import Path as _MplPath
    from matplotlib.patches import PathPatch
    cx0, cx1 = x0 + 0.45 * (x1 - x0), x1 - 0.45 * (x1 - x0)
    verts = [
        (x0, y0_top),
        (cx0, y0_top), (cx1, y1_top), (x1, y1_top),  
        (x1, y1_bot),
        (cx1, y1_bot), (cx0, y0_bot), (x0, y0_bot), 
        (x0, y0_top),
    ]
    codes = [_MplPath.MOVETO,
             _MplPath.CURVE4, _MplPath.CURVE4, _MplPath.CURVE4,
             _MplPath.LINETO,
             _MplPath.CURVE4, _MplPath.CURVE4, _MplPath.CURVE4,
             _MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(_MplPath(verts, codes),
                            facecolor=color, edgecolor="none", alpha=alpha))


def plot_gmm_sankey(gmm_results: dict[str, dict],
                     datasets: list[str] | None = None) -> list[plt.Figure]:
    """v2 3.2 supplementary - Sankey/alluvial of GMM cluster vs true Time.
    """
    if datasets is None:
        datasets = list(gmm_results.keys())

    cluster_colors = {"E": "#F8766D", "EM1": "#619CFF", "M": "#00BA38"}
    cluster_order  = ["E", "EM1", "M"]
    time_order     = ["0d", "8h", "8h_rm", "1d", "1d_rm",
                      "3d", "3d_rm", "7d"]

    figs: list[plt.Figure] = []
    for name in datasets:
        if name not in gmm_results:
            continue
        info = gmm_results[name]
        states = np.asarray(info["states"])
        if info.get("time") is None:
            df = pd.read_csv(resolve_data_file(f"cook2020/{name}_em_expr.csv"),
                              index_col=0, usecols=["Unnamed: 0", "Time"])
            times = df["Time"].astype(str).values
        else:
            times = np.asarray(info["time"]).astype(str)

        present_times = [t for t in time_order if t in set(times)]
        ct = pd.DataFrame(
            0, index=cluster_order, columns=present_times, dtype=int
        )
        for c, t in zip(states, times):
            if c in ct.index and t in ct.columns:
                ct.loc[c, t] += 1

        cluster_totals = ct.sum(axis=1)
        time_totals    = ct.sum(axis=0)
        N = cluster_totals.sum()
        gap = N * 0.02  

        def _stack_top_to_bot(totals, order):
            tops = {}; running = N + gap * (len(order) - 1)
            for k in order:
                size = totals[k]
                tops[k] = (running, running - size)  
                running -= size + gap
            return tops

        left  = _stack_top_to_bot(cluster_totals, cluster_order)
        right = _stack_top_to_bot(time_totals, present_times)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(-0.05, 1.10)
        y_max = N + gap * (max(len(cluster_order), len(present_times)) - 1)
        ax.set_ylim(-gap, y_max + gap)

        x_left, x_right = 0.05, 0.95
        bar_w = 0.04

        # Draw bars
        for c in cluster_order:
            top, bot = left[c]
            ax.add_patch(plt.Rectangle((x_left - bar_w/2, bot), bar_w, top - bot,
                                         facecolor="#D9D9D9", edgecolor="black",
                                         linewidth=0.6))
            ax.text(x_left, (top + bot) / 2, c, ha="center", va="center",
                     fontsize=12, fontweight="bold")

        for t in present_times:
            top, bot = right[t]
            ax.add_patch(plt.Rectangle((x_right - bar_w/2, bot), bar_w, top - bot,
                                         facecolor="#D9D9D9", edgecolor="black",
                                         linewidth=0.6))
            ax.text(x_right, (top + bot) / 2, t, ha="center", va="center",
                     fontsize=10, fontweight="bold")

        # Flow ribbons.  Within each cluster, sub-segments stack in time_order.
        # Within each time, sub-segments stack in cluster_order.
        l_cursor = {c: left[c][0] for c in cluster_order}
        r_cursor = {t: right[t][0] for t in present_times}

        for c in cluster_order:
            for t in present_times:
                n = ct.loc[c, t]
                if n <= 0:
                    continue
                l_top = l_cursor[c]; l_bot = l_top - n
                r_top = r_cursor[t]; r_bot = r_top - n
                _alluvial_curve(x_left + bar_w/2, l_top, l_bot,
                                  x_right - bar_w/2, r_top, r_bot,
                                  ax, color=cluster_colors[c], alpha=0.55)
                l_cursor[c] = l_bot
                r_cursor[t] = r_bot

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=cluster_colors[c], label=c, alpha=0.6)
                            for c in cluster_order],
                  title="Cluster", loc="center right",
                  bbox_to_anchor=(1.18, 0.5), frameon=False)

        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["GMM", "TrueLabel"], fontsize=11)
        ax.set_ylabel("Number of Cells", fontsize=12)
        ax.set_title(f"Sankey Diagram: GMM Cluster vs True Cell Labels - {name}",
                      fontsize=13)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_yticks(np.linspace(0, N, 5).round().astype(int))
        fig.tight_layout()
        plt.close(fig)
        figs.append(fig)
    return figs

