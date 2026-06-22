"""
2.5   Multi-method scoring (nnPCA, AUCell, ssGSEA)

Run the three scoring methods on bulk expression and gene signatures,
then build a z-scored comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from .nnpca   import run_nnPCA
from .ssGSEA  import execute_ssgsva
from .aucell  import execute_aucell


@dataclass
class Scores:
    nnPCA:   pd.DataFrame
    AUCell:  pd.DataFrame
    ssGSEA:  pd.DataFrame
    comparison: pd.DataFrame


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(float)
    keep = df.std(axis=0, skipna=True).fillna(0) > 0
    z = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    if keep.any():
        sub = df.loc[:, keep].fillna(df.loc[:, keep].mean())
        z.loc[:, keep] = StandardScaler().fit_transform(sub)
    return z


def score_all_methods(geneExp: pd.DataFrame, gmt_path: str,
                      verbose: bool = True) -> Scores:
    """2.5 - run nnPCA + AUCell + ssGSEA and build the z-scored comparison."""
    nnPCA_scores  = run_nnPCA(geneExp,   gmt_file=gmt_path, dimension=1)
    aucell_scores = execute_aucell(geneExp, gmt_file=gmt_path)
    ssgsea_scores = execute_ssgsva(geneExp, gmt_file=gmt_path)

    if verbose:
        print("nnPCA  :", nnPCA_scores.shape,  list(nnPCA_scores.columns))
        print("AUCell :", aucell_scores.shape, list(aucell_scores.columns))
        print("ssGSEA :", ssgsea_scores.shape, list(ssgsea_scores.columns))

    nnPCA_z   = _zscore(nnPCA_scores)
    aucell_z  = _zscore(aucell_scores)
    ssgsea_z  = _zscore(ssgsea_scores)

    comparison = pd.DataFrame({
        "nnPCA":   nnPCA_z.mean(axis=1),
        "AUCell":  aucell_z.mean(axis=1),
        "ssGSEA":  ssgsea_z.mean(axis=1),
    })
    if verbose:
        print(comparison.head())

    return Scores(nnPCA_scores, aucell_scores, ssgsea_scores, comparison)
