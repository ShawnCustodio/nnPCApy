import numpy as np
import pandas as pd


# ============================================================
# AUCell scoring
# ============================================================
def aucell_score(expr_matrix: pd.DataFrame, genes: list, top_percent: float = 0.05):
    """
    AUCell-like enrichment score:
    fraction of signature genes in top expressed genes per sample.
    Returns: Series indexed by samples.
    """

    genes = [g for g in genes if g in expr_matrix.columns]

    if len(genes) == 0:
        return pd.Series(np.nan, index=expr_matrix.index)

    n_genes = expr_matrix.shape[1]
    max_rank = max(1, int(n_genes * top_percent))

    scores = []

    for sample in expr_matrix.index:
        values = expr_matrix.loc[sample]

        ranked_genes = values.sort_values(ascending=False)
        top_genes = ranked_genes.iloc[:max_rank].index

        enrichment = len(set(genes).intersection(top_genes)) / len(genes)
        scores.append(enrichment)

    return pd.Series(scores, index=expr_matrix.index)


# ============================================================
# GMT parsing
# ============================================================
def parse_gmt(gmt_file: str) -> dict:
    genesets = {}
    with open(gmt_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                genesets[parts[0]] = parts[2:]
    return genesets


# ============================================================
# MULTI-GENESET AUCell
# ============================================================
def execute_aucell(expr_matrix: pd.DataFrame, gmt_file: str) -> pd.DataFrame:
    """
    Runs AUCell across all gene sets.
    """

    genesets = parse_gmt(gmt_file)
    result_dict = {}

    for name, genes in genesets.items():
        result_dict[name] = aucell_score(expr_matrix, genes)

    return pd.DataFrame(result_dict, index=expr_matrix.index)


# ============================================================
# SINGLE GENESET AUCell
# ============================================================
def execute_aucell_single(
    expr_matrix: pd.DataFrame,
    gmt_file: str,
    score_name: str,
    gene_set_index: int = 0
):
    """
    Compute AUCell for a single gene set.
    """

    with open(gmt_file) as f:
        lines = f.readlines()

    parts = lines[gene_set_index].strip().split("\t")
    genes = parts[2:]

    scores = aucell_score(expr_matrix, genes)

    return pd.DataFrame({score_name: scores})