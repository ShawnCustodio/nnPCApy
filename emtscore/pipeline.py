"""
emtscore.pipeline
=================
High-level helpers that the automated notebook uses. Each function is a
one-liner or two-liner worth of work from a notebook cell, so the notebook
stays a thin front-end and the real logic lives here.

All plotting helpers draw into a Matplotlib Axes (passed in or freshly
created) and return the ``Axes`` / ``Figure`` so notebooks can show or save.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .nnpca    import run_nnPCA
from .aucell   import execute_aucell
from .ssGSEA   import execute_ssgsva
from .nsprcomp import nsprcomp, compute_M1_M2_scores

from utility.data_paths   import resolve_data_file
from utility.load_cook2020 import load_cook2020, COOK_DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# 2.1 – 2.4   Bulk data loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BulkData:
    """Container for the bulk-RNA inputs used in vignette 2."""
    cell_annot: pd.DataFrame | None
    geneExp:    pd.DataFrame | None   # samples × genes
    E_sig:      pd.DataFrame
    M_sig:      pd.DataFrame
    gmt_path:   str

    @property
    def has_expression(self) -> bool:
        return self.geneExp is not None


def load_bulk_data(build_gmt: bool = True) -> BulkData:
    """
    Load cell annotation, bulk expression, E & M signatures (2.1 - 2.4).
    """
    try:
        cell_annot = pd.read_csv(resolve_data_file("cell_annotation_file.csv"))
    except FileNotFoundError:
        cell_annot = None

    try:
        geneExp = pd.read_csv(resolve_data_file("geneExp.csv"), index_col=0).T
    except FileNotFoundError:
        geneExp = None

    E_sig = pd.read_csv(resolve_data_file("Panchy_et_al_E_signature.csv"))
    M_sig = pd.read_csv(resolve_data_file("Panchy_et_al_M_signature.csv"))

    gmt_path = str(resolve_data_file("EM_signature.gmt"))
    if build_gmt:
        gmt_path = build_em_gmt(E_sig, M_sig, gmt_path)

    return BulkData(cell_annot, geneExp, E_sig, M_sig, gmt_path)


def build_em_gmt(E_sig: pd.DataFrame, M_sig: pd.DataFrame,
                 out_path: str | Path) -> str:
    """Write an E/M .gmt file (2.4) and return its path."""
    E_genes = E_sig["GeneName"].tolist()
    M_genes = M_sig["GeneName"].tolist()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("Panchy_et_al_E_signature\tNA\t" + "\t".join(E_genes) + "\n")
        f.write("Panchy_et_al_M_signature\tNA\t" + "\t".join(M_genes) + "\n")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# 2.5   Multi-method scoring
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(float)
    keep = df.std(axis=0, skipna=True).fillna(0) > 0
    z = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    if keep.any():
        sub = df.loc[:, keep].fillna(df.loc[:, keep].mean())
        z.loc[:, keep] = StandardScaler().fit_transform(sub)
    return z


def run_multi_method_scoring(geneExp: pd.DataFrame, gmt_path: str) -> dict[str, pd.DataFrame]:
    """Score ``geneExp`` with nnPCA, AUCell, and ssGSEA (2.5)."""
    return {
        "nnPCA":   run_nnPCA(geneExp,  gmt_file=gmt_path, dimension=1),
        "AUCell":  execute_aucell(geneExp, gmt_file=gmt_path),
        "ssGSEA":  execute_ssgsva(geneExp, gmt_file=gmt_path),
    }


def build_comparison_table(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Z-score each method and collapse E/M columns to a single value (2.5)."""
    return pd.DataFrame({
        method: _zscore_block(df).mean(axis=1) for method, df in scores.items()
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2.7 – 2.10   Bulk plots (E vs M, M1 vs M2, histograms)
# ─────────────────────────────────────────────────────────────────────────────

def _pc1_em(geneExp: pd.DataFrame, E_sig: pd.DataFrame, M_sig: pd.DataFrame,
            ncomp_M: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """PC1 of E signature and PC1/PC2 of M signature on bulk geneExp."""
    E_genes = [g for g in E_sig["GeneName"] if g in geneExp.columns]
    M_genes = [g for g in M_sig["GeneName"] if g in geneExp.columns]
    E_mat = StandardScaler().fit_transform(geneExp[E_genes].values)
    M_mat = StandardScaler().fit_transform(geneExp[M_genes].values)
    Es = nsprcomp(E_mat, ncomp=1)["x"][:, 0]
    Ms = nsprcomp(M_mat, ncomp=ncomp_M)["x"]
    Mp1 = Ms[:, 0]
    Mp2 = Ms[:, 1] if ncomp_M > 1 else None
    return Es, Mp1, Mp2


def plot_em_scatter(data: BulkData, ax: plt.Axes | None = None) -> plt.Figure:
    """2.7 — Escore vs Mscore scatter coloured by Type."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    Es, Mp1, _ = _pc1_em(data.geneExp, data.E_sig, data.M_sig, ncomp_M=1)
    df = pd.DataFrame({"Escore": Es, "Mscore": Mp1}, index=data.geneExp.index)
    if data.cell_annot is not None and "Type" in data.cell_annot.columns:
        types = data.cell_annot.set_index(data.cell_annot.columns[0])["Type"]
        df["Type"] = types.reindex(df.index).fillna("NA")
        for t, g in df.groupby("Type"):
            ax.scatter(g["Escore"], g["Mscore"], s=22, alpha=0.8, label=str(t))
        ax.legend(title="Type", frameon=False)
    else:
        ax.scatter(df["Escore"], df["Mscore"], s=22, alpha=0.8)
    ax.set_xlabel("Escore"); ax.set_ylabel("Mscore")
    ax.set_title("2.7 — Escore vs Mscore (bulk)")
    return fig


def plot_m1_m2_scatter(data: BulkData, ax: plt.Axes | None = None) -> plt.Figure:
    """2.8 — M1 vs M2 component scatter (second M PC)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    _, M1, M2 = _pc1_em(data.geneExp, data.E_sig, data.M_sig, ncomp_M=2)
    if M2 is None:
        ax.text(0.5, 0.5, "M signature only has 1 PC", ha="center",
                transform=ax.transAxes); ax.axis("off"); return fig
    ax.scatter(M1, M2, s=22, alpha=0.8, color="#d62728")
    ax.set_xlabel("Mscore_PC1"); ax.set_ylabel("Mscore_PC2")
    ax.set_title("2.8 — M1 vs M2 component")
    return fig


def plot_combined_scatter(data: BulkData) -> plt.Figure:
    """2.9 — side-by-side E/M and M1/M2 scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_em_scatter(data, axes[0])
    plot_m1_m2_scatter(data, axes[1])
    fig.tight_layout()
    return fig


def plot_m1_histogram(data: BulkData, ax: plt.Axes | None = None) -> plt.Figure:
    """2.10 — M1 histogram coloured by Type."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    _, M1, _ = _pc1_em(data.geneExp, data.E_sig, data.M_sig, ncomp_M=1)
    df = pd.DataFrame({"Mscore": M1}, index=data.geneExp.index)
    if data.cell_annot is not None and "Type" in data.cell_annot.columns:
        types = data.cell_annot.set_index(data.cell_annot.columns[0])["Type"]
        df["Type"] = types.reindex(df.index).fillna("NA")
        for t, g in df.groupby("Type"):
            ax.hist(g["Mscore"], bins=20, alpha=0.6, label=str(t))
        ax.legend(title="Type", frameon=False)
    else:
        ax.hist(df["Mscore"], bins=20, alpha=0.8, color="#555")
    ax.set_xlabel("Mscore"); ax.set_ylabel("count")
    ax.set_title("2.10 — Mscore distribution")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.11   Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_m_heatmap_full(data: BulkData, top_samples: int | None = None) -> plt.Figure:
    """2.11.0 — full M-signature heatmap (all M genes in geneExp)."""
    import seaborn as sns
    M_genes = [g for g in data.M_sig["GeneName"] if g in data.geneExp.columns]
    mat = data.geneExp[M_genes].T
    if top_samples:
        mat = mat.iloc[:, :top_samples]
    mat = (mat.sub(mat.mean(axis=1), axis=0)
              .div(mat.std(axis=1).replace(0, 1), axis=0))
    g = sns.clustermap(mat, cmap="vlag", center=0, figsize=(10, 12),
                        xticklabels=False, yticklabels=(len(M_genes) < 60))
    g.fig.suptitle("2.11.0 — Full M-signature heatmap", y=1.02)
    return g.fig


def plot_m_heatmap_clustered(data: BulkData, n_genes: int = 30) -> plt.Figure:
    """2.11 — clustered heatmap on the top-``n_genes`` M genes by variance."""
    import seaborn as sns
    M_genes = [g for g in data.M_sig["GeneName"] if g in data.geneExp.columns]
    mat = data.geneExp[M_genes].T
    var = mat.var(axis=1).sort_values(ascending=False)
    mat = mat.loc[var.head(n_genes).index]
    mat = (mat.sub(mat.mean(axis=1), axis=0)
              .div(mat.std(axis=1).replace(0, 1), axis=0))
    g = sns.clustermap(mat, cmap="vlag", center=0,
                        figsize=(10, max(4, 0.25 * len(mat))))
    g.fig.suptitle("2.11 — Clustered M-signature heatmap", y=1.02)
    return g.fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.1   Cook dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lognorm(expr: pd.DataFrame, scale: float = 10_000) -> np.ndarray:
    lib = expr.sum(axis=1).values[:, None]
    lib = np.where(lib == 0, 1, lib)
    return np.log1p(expr.values / lib * scale)


def _scale_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1
    return (X - mu) / sd


def load_cook_dataset(name: str):
    """3.1 — load a Cook dataset (auto-resolves data path)."""
    return load_cook2020(name)


def compute_em_scores(adata, E_genes: list[str], M_genes: list[str],
                     ncomp_M: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """3.1 — Escore (PC1), Mscore (PC1), Mscore_PC2 aligned to pseudotime."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    pt = adata.obs["Pseudotime"].values
    full = pd.DataFrame(_lognorm(expr), index=expr.index, columns=expr.columns)

    E_in = [g for g in E_genes if g in full.columns]
    M_in = [g for g in M_genes if g in full.columns]
    E_mat = _scale_cols(full[E_in].values)
    M_mat = _scale_cols(full[M_in].values)

    Es = nsprcomp(E_mat, ncomp=1)["x"][:, 0]
    Ms = nsprcomp(M_mat, ncomp=ncomp_M)["x"]
    Mp1 = Ms[:, 0]
    Mp2 = Ms[:, 1] if ncomp_M > 1 else None

    # Align sign to pseudotime
    if np.corrcoef(pt, Es)[0, 1] > 0:
        Es = -Es
    if np.corrcoef(pt, Mp1)[0, 1] < 0:
        Mp1 = -Mp1

    # Attach back onto adata.obs for downstream use
    adata.obs["Escore"] = Es
    adata.obs["Mscore"] = Mp1
    if Mp2 is not None:
        adata.obs["Mscore_PC2"] = Mp2
    return Es, Mp1, Mp2


# ─────────────────────────────────────────────────────────────────────────────
# 3.2   GMM in E-M space
# ─────────────────────────────────────────────────────────────────────────────

_GMM_COLORS = {"E": "#F8766D", "EM1": "#619CFF", "M": "#00BA38"}


def run_cook_gmm(adata, E_genes: list[str], M_genes: list[str],
                 random_state: int = 42) -> dict[str, Any]:
    """3.2 — 3-state GMM on (Escore, Mscore) for one Cook dataset."""
    Es, Mp1, _ = compute_em_scores(adata, E_genes, M_genes, ncomp_M=1)
    Es = (Es - Es.mean()) / Es.std()
    Ms = (Mp1 - Mp1.mean()) / Mp1.std()
    em = np.column_stack([Es, Ms])
    gmm = GaussianMixture(n_components=3, random_state=random_state,
                          n_init=5).fit(em)
    labels = gmm.predict(em)
    order  = np.argsort(gmm.means_[:, 0])                 # by Escore mean
    state_map = {order[0]: "M", order[1]: "EM1", order[2]: "E"}
    states = np.array([state_map[l] for l in labels])
    adata.obs["GMM_state"] = states
    return {"em": em, "states": states, "gmm": gmm,
            "pseudotime": adata.obs["Pseudotime"].values}


def plot_cook_gmm(ax: plt.Axes, name: str, result: dict[str, Any]) -> None:
    em, states = result["em"], result["states"]
    for s in ("E", "EM1", "M"):
        m = states == s
        ax.scatter(em[m, 0], em[m, 1], s=14, alpha=0.65,
                   color=_GMM_COLORS[s], label=s, linewidths=0)
    ax.set_xlabel("Escore"); ax.set_ylabel("Mscore")
    ax.set_title(f"{name} — GMM (M / EM1 / E)")
    ax.legend(title="Cluster", frameon=False)


def run_cook_gmm_all(E_sig: pd.DataFrame, M_sig: pd.DataFrame,
                     datasets: list[str] | None = None) -> dict[str, dict]:
    """3.2 — run GMM on all Cook datasets and draw the 3-panel figure.
    """
    datasets = datasets or COOK_DATASETS
    E_genes = E_sig["GeneName"].tolist()
    M_genes = M_sig["GeneName"].tolist()
    results: dict[str, dict] = {}

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6.5 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, name in zip(axes, datasets):
        try:
            adata = load_cook_dataset(name)
        except FileNotFoundError as e:
            ax.set_title(f"{name}\n(data missing)"); ax.axis("off")
            print(f"[warn] {e}")
            continue
        res = run_cook_gmm(adata, E_genes, M_genes)
        res["adata"] = adata
        plot_cook_gmm(ax, name, res)
        results[name] = res

    fig.suptitle("3.2 — GMM (E / EM1 / M) across Cook et al. 2020 datasets",
                 y=1.02)
    fig.tight_layout()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.3   Pathway correlation
# ─────────────────────────────────────────────────────────────────────────────

def run_pathway_correlation(adata, gmt_path: str | None = None,
                             top_n: int = 10) -> pd.DataFrame:
    """3.3 — Pearson-correlate Escore vs ssGSEA pathways from filtered.c2.gmt."""
    gmt_path = gmt_path or str(resolve_data_file("filtered.c2.gmt"))
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    scores = execute_ssgsva(expr, gmt_file=gmt_path)
    scores = scores.reindex(adata.obs_names)

    Es = adata.obs["Escore"].values
    corr = scores.apply(lambda c: np.corrcoef(c.values, Es)[0, 1], axis=0)
    df = (pd.DataFrame({"pathway": corr.index, "r": corr.values})
            .sort_values("r", ascending=False))
    top_pos = df.head(top_n).reset_index(drop=True)
    top_neg = df.tail(top_n).sort_values("r").reset_index(drop=True)
    return pd.concat(
        [top_pos.assign(direction="positive"),
         top_neg.assign(direction="negative")],
        ignore_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.4   E vs M + PC1 vs PC2 panels across Cook datasets
# ─────────────────────────────────────────────────────────────────────────────

def plot_cook_em_panels(E_sig: pd.DataFrame, M_sig: pd.DataFrame,
                         datasets: list[str] | None = None) -> plt.Figure:
    """3.4 — 2×N grid: (E vs M) and (PC1 vs PC2) per Cook dataset."""
    datasets = datasets or COOK_DATASETS
    E_genes = E_sig["GeneName"].tolist()
    M_genes = M_sig["GeneName"].tolist()

    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(5.5 * len(datasets), 10),
                             squeeze=False)

    for col, name in enumerate(datasets):
        try:
            adata = load_cook_dataset(name)
        except FileNotFoundError as e:
            for row in (0, 1):
                axes[row, col].set_title(f"{name}\n(data missing)")
                axes[row, col].axis("off")
            print(f"[warn] {e}")
            continue

        Es, M1, M2 = compute_em_scores(adata, E_genes, M_genes, ncomp_M=2)
        pt = adata.obs["Pseudotime"].values

        sc = axes[0, col].scatter(Es, M1, c=pt, s=10, cmap="viridis", alpha=0.7)
        axes[0, col].set_title(f"{name} — E vs M")
        axes[0, col].set_xlabel("Escore"); axes[0, col].set_ylabel("Mscore")
        plt.colorbar(sc, ax=axes[0, col], label="Pseudotime")

        if M2 is not None:
            sc2 = axes[1, col].scatter(M1, M2, c=pt, s=10, cmap="viridis", alpha=0.7)
            axes[1, col].set_title(f"{name} — M_PC1 vs M_PC2")
            axes[1, col].set_xlabel("Mscore_PC1"); axes[1, col].set_ylabel("Mscore_PC2")
            plt.colorbar(sc2, ax=axes[1, col], label="Pseudotime")
        else:
            axes[1, col].axis("off")

    fig.suptitle("3.4 — E vs M and PC1 vs PC2 across Cook datasets", y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 – 3.6   Stemness and senescence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _load_tsv_genes(name: str, column: str = "gene_symbol") -> list[str]:
    return pd.read_csv(resolve_data_file(name), sep="\t")[column].dropna().tolist()


def compute_stem_senescence(adata) -> pd.DataFrame:
    """3.5 — add Stemness_Score + Senescence_Score to ``adata.obs``."""
    stem = _load_tsv_genes("stemsig.tsv")
    sen  = _load_tsv_genes("cellular_senescence_sig.tsv")

    def _sig(genes: list[str]) -> np.ndarray:
        avail = [g for g in genes if g in adata.var_names]
        X = adata[:, avail].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1
        Xn = np.log1p(X / lib * 1e4)
        Z = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-12)
        return Z.mean(axis=1)

    adata.obs["Stemness_Score"]   = _sig(stem)
    adata.obs["Senescence_Score"] = _sig(sen)
    return adata.obs[["Escore", "Mscore", "Stemness_Score", "Senescence_Score"]]


def plot_stem_senescence(adata) -> plt.Figure:
    """3.6 — Stemness vs Escore and Senescence vs Mscore scatter panels."""
    from scipy.stats import pearsonr

    df = adata.obs[["Escore", "Mscore", "Stemness_Score",
                     "Senescence_Score"]].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    r1, p1 = pearsonr(df["Stemness_Score"], df["Escore"])
    ax[0].scatter(df["Escore"], df["Stemness_Score"],
                  s=10, alpha=0.5, color="#1f77b4")
    ax[0].set_xlabel("Escore"); ax[0].set_ylabel("Stemness_Score")
    ax[0].set_title(f"3.6 — Stemness vs Escore  (r={r1:.2f}, p={p1:.1e})")

    r2, p2 = pearsonr(df["Senescence_Score"], df["Mscore"])
    ax[1].scatter(df["Mscore"], df["Senescence_Score"],
                  s=10, alpha=0.5, color="#d62728")
    ax[1].set_xlabel("Mscore"); ax[1].set_ylabel("Senescence_Score")
    ax[1].set_title(f"3.6 — Senescence vs Mscore  (r={r2:.2f}, p={p2:.1e})")

    fig.tight_layout()
    return fig


def plot_bulk_panels(data: BulkData) -> list[plt.Figure]:
    """2.7 – 2.10 — convenience wrapper, returns the four bulk-data figures."""
    return [
        plot_em_scatter(data),
        plot_m1_m2_scatter(data),
        plot_combined_scatter(data),
        plot_m1_histogram(data),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public surface
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "BulkData",
    # 2
    "load_bulk_data", "build_em_gmt",
    "run_multi_method_scoring", "build_comparison_table",
    "plot_em_scatter", "plot_m1_m2_scatter", "plot_combined_scatter",
    "plot_m1_histogram", "plot_bulk_panels", "plot_m_heatmap_clustered", "plot_m_heatmap_full",
    # 3
    "load_cook_dataset", "compute_em_scores",
    "run_cook_gmm", "plot_cook_gmm", "run_cook_gmm_all",
    "run_pathway_correlation", "plot_cook_em_panels",
    "compute_stem_senescence", "plot_stem_senescence",
]
