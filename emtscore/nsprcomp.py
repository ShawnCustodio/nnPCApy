"""
Non-negative sparse PCA for the emtscore layer.

The solver is nnpcapy's: ``nsprcomp`` below is ``nnpcapy.nsprcomp.nsprcomp``, re-exported so
that ``emtscore.nsprcomp.nsprcomp`` and ``emtscore.nnpca_solver`` keep working. There is one
implementation, validated against R in tests/test_nsprcomp.py. This module adds
compute_M1_M2_scores.
"""
import numpy as np
import pandas as pd

from nnpcapy.nsprcomp import nsprcomp  # re-exported as emtscore.nsprcomp.nsprcomp


def compute_M1_M2_scores(
    geneExp: pd.DataFrame,
    genes: list,
    perturbation: float = 1e-4,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Compute M1/M2 EMT scores using nsprcomp.
    """
    available = [g for g in genes if g in geneExp.columns]

    if len(available) == 0:
        return pd.DataFrame(
            np.zeros((geneExp.shape[0], n_components)),
            index=geneExp.index,
            columns=[f"M{i+1}" for i in range(n_components)]
        )

    X = geneExp[available].values.astype(np.float64)

    np.random.seed(42)
    X = X + np.random.randn(*X.shape) * perturbation

    result = nsprcomp(X, ncomp=n_components, nneg=True, center=True, scale_=False)
    scores = result["x"][:, :n_components]

    return pd.DataFrame(
        scores,
        index=geneExp.index,
        columns=[f"M{i+1}" for i in range(n_components)]
    )
