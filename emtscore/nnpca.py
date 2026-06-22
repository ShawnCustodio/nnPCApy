import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# GMT parsing
# ============================================================
def parse_gmt(gmt_file):
    genesets = {}
    with open(gmt_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                genesets[parts[0]] = parts[2:]
    return genesets


# ============================================================
# nnPCA core scoring
# ============================================================
def get_nnPCA_result(
    geneExp: pd.DataFrame,
    genes: list,
    align_direction: str = "positive",
):
    """
    Single gene set scoring using nsprcomp result.
    """

    from emtscore.nsprcomp import nsprcomp

    available = [g for g in genes if g in geneExp.columns]

    if len(available) == 0:
        return np.zeros(geneExp.shape[0])

    X = geneExp[available].values.astype(np.float64)

    X = X - X.min(axis=0)

    res = nsprcomp(X, ncomp=1, nneg=True, center=True, scale_=False)
    scores = res["x"][:, 0]

    if align_direction == "positive":
        if np.mean(scores) < 0:
            scores = -scores

    elif align_direction == "negative":
        if np.mean(scores) > 0:
            scores = -scores

    return scores


# ============================================================
# MAIN EMTscore-style nnPCA
# ============================================================
def run_nnPCA(
    geneExp: pd.DataFrame,
    gmt_file: str,
    dimension: int = 1
):
    """
    EMTscore-style nnPCA implementation.
    """

    genesets = parse_gmt(gmt_file)

    if len(genesets) < 2:
        raise ValueError("GMT must contain at least 2 gene sets")

    # --------------------------------------------------------
    #  Score each gene set independently
    # --------------------------------------------------------
    score_dict = {}

    for name, genes in genesets.items():
        score_dict[name] = get_nnPCA_result(
            geneExp,
            genes,
            align_direction="positive"
        )

    score_mat = pd.DataFrame(score_dict, index=geneExp.index)

    # --------------------------------------------------------
    # Build latent EMT axis 
    # --------------------------------------------------------
    if score_mat.shape[1] > 1:
        pca = PCA(n_components=dimension)
        emt_axis = pca.fit_transform(score_mat)

        if dimension == 1:
            return pd.DataFrame(
                emt_axis.flatten(),
                index=geneExp.index,
                columns=["EMT_score"]
            )

        return pd.DataFrame(
            emt_axis,
            index=geneExp.index,
            columns=[f"EMT_PC{i+1}" for i in range(dimension)]
        )

    # fallback
    return pd.DataFrame(
        score_mat.iloc[:, 0],
        index=geneExp.index,
        columns=["EMT_score"]
    )


# ============================================================
# Single gene set helper
# ============================================================
def execute_nnPCA_single(
    geneExp: pd.DataFrame,
    gmt_file: str,
    score_name: str = "nnPCA"
):
    genesets = parse_gmt(gmt_file)
    first_genes = next(iter(genesets.values()))

    scores = get_nnPCA_result(geneExp, first_genes)

    return pd.DataFrame(scores, index=geneExp.index, columns=[score_name])


# ============================================================
# Multi-dim helper (M1/M2 style)
# ============================================================
def execute_nnPCA_multidim(
    geneExp: pd.DataFrame,
    genes: list,
    n_components: int = 2,
    score_prefix: str = "nnPCA"
):
    from emtscore.nsprcomp import nsprcomp

    available = [g for g in genes if g in geneExp.columns]

    if len(available) == 0:
        return pd.DataFrame(
            np.zeros((geneExp.shape[0], n_components)),
            index=geneExp.index,
            columns=[f"{score_prefix}_{i+1}" for i in range(n_components)]
        )

    X = geneExp[available].values

    res = nsprcomp(X, ncomp=n_components, nneg=True, center=True, scale_=False)

    return pd.DataFrame(
        res["x"][:, :n_components],
        index=geneExp.index,
        columns=[f"{score_prefix}_{i+1}" for i in range(n_components)]
    )