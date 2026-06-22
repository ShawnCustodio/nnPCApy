"""
2.6 + 2.7-   Prepare plot data and rebuild nnPCA per gene set

Combine scores with metadata and expression data for plotting. Handles
re-alignment of indices and recomputation of nnPCA scores per gene set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .nnpca   import parse_gmt, get_nnPCA_result
from .ssGSEA  import execute_ssgsva
from .aucell  import execute_aucell
from .nsprcomp import compute_M1_M2_scores
from .scoring import Scores


@dataclass
class PlotData:
    nnPCA_em:   pd.DataFrame
    aucell_em:  pd.DataFrame
    ssgsea_em:  pd.DataFrame
    nnPCA_mm:   pd.DataFrame


@dataclass
class RebuildResult:
    geneExp:    pd.DataFrame  # aligned
    cell_annot: pd.DataFrame  # aligned (indexed by name)
    nnPCA_em:   pd.DataFrame
    aucell_em:  pd.DataFrame
    ssgsea_em:  pd.DataFrame


def _prepare_plot_data(scores_df: pd.DataFrame,
                       annotations_df: pd.DataFrame) -> pd.DataFrame:
    """v2 2.6 - safely align scores with annotation by sample name."""
    annotations_df = annotations_df.copy()
    if "name" in annotations_df.columns:
        annotations_df = annotations_df.set_index("name")
    common = scores_df.index.intersection(annotations_df.index)
    return annotations_df.loc[common].join(scores_df.loc[common])


def prepare_plot_dataframes(scores: Scores,
                            geneExp: pd.DataFrame,
                            M_sig: pd.DataFrame,
                            cell_annot: pd.DataFrame,
                            verbose: bool = True) -> PlotData:
    """2.6 - rename method columns, compute M1/M2, join with cell_annot."""
    nnPCA   = scores.nnPCA.rename(columns={
        "Escore": "Panchy_et_al_E_signature",
        "Mscore": "Panchy_et_al_M_signature",
    })
    aucell  = scores.AUCell.copy();  aucell.columns  = ["Panchy_et_al_E_signature", "Panchy_et_al_M_signature"]
    ssgsea  = scores.ssGSEA.copy();  ssgsea.columns  = ["Panchy_et_al_E_signature", "Panchy_et_al_M_signature"]

    M_genes_filtered = [g for g in M_sig["GeneName"].tolist() if g in geneExp.columns]
    M_scores_df = compute_M1_M2_scores(geneExp, M_genes_filtered, perturbation=1e-4)

    nnPCA_em   = _prepare_plot_data(nnPCA,   cell_annot)
    aucell_em  = _prepare_plot_data(aucell,  cell_annot)
    ssgsea_em  = _prepare_plot_data(ssgsea,  cell_annot)
    nnPCA_mm   = _prepare_plot_data(M_scores_df, cell_annot)

    if verbose:
        print("All data prepared. Shapes:")
        print(f"  nnPCA_em:   {nnPCA_em.shape}")
        print(f"  aucell_em:  {aucell_em.shape}")
        print(f"  ssgsea_em:  {ssgsea_em.shape}")
        print(f"  nnPCA_mm:   {nnPCA_mm.shape}")

    return PlotData(nnPCA_em, aucell_em, ssgsea_em, nnPCA_mm)


def rebuild_em_for_plot(geneExp: pd.DataFrame,
                        cell_annot: pd.DataFrame,
                        gmt_path: str,
                        verbose: bool = True) -> RebuildResult:
    """v2 cell 16 - enforce same index, recompute nnPCA per gene set."""
    if "name" in cell_annot.columns:
        cell_annot = cell_annot.set_index("name")

    common = geneExp.index.intersection(cell_annot.index)
    geneExp = geneExp.loc[common]
    cell_annot = cell_annot.loc[common]

    _genesets = parse_gmt(gmt_path)
    nnPCA_scores = pd.DataFrame(
        {name: get_nnPCA_result(geneExp, genes, align_direction="positive")
         for name, genes in _genesets.items()},
        index=geneExp.index,
    )
    aucell_scores = execute_aucell(geneExp, gmt_file=gmt_path)
    ssgsea_scores = execute_ssgsva(geneExp, gmt_file=gmt_path)

    nnPCA_em  = nnPCA_scores.join(cell_annot["celltype_annotation"])
    aucell_em = aucell_scores.join(cell_annot["celltype_annotation"])
    ssgsea_em = ssgsea_scores.join(cell_annot["celltype_annotation"])

    if verbose:
        print("nnPCA_em columns:", list(nnPCA_em.columns))
        print("aucell_em columns:", list(aucell_em.columns))
        print("ssgsea_em columns:", list(ssgsea_em.columns))

    return RebuildResult(geneExp, cell_annot, nnPCA_em, aucell_em, ssgsea_em)
