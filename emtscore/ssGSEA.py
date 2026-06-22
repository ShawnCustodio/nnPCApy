import pandas as pd
import numpy as np


# ============================================================
# ssGSEA scoring 
# ============================================================
def ssgsea_score(expr_matrix: pd.DataFrame, genes: list, alpha: float = 0.25) -> pd.Series:

    genes = [g for g in genes if g in expr_matrix.columns]

    if len(genes) == 0:
        return pd.Series(np.nan, index=expr_matrix.index)

    scores = []

    for sample in expr_matrix.index:
        values = expr_matrix.loc[sample]

        ranked = values.sort_values(ascending=False)
        ranked_genes = ranked.index.values
        ranked_values = ranked.values

        hits = np.isin(ranked_genes, genes)

        Nh = hits.sum()
        Nm = len(hits) - Nh

        if Nh == 0:
            scores.append(np.nan)
            continue

        weights = np.abs(ranked_values) ** alpha

        hit_score = np.cumsum(hits * weights / (weights[hits].sum() + 1e-12))
        miss_score = np.cumsum(~hits / (Nm + 1e-12))

        running_ES = hit_score - miss_score

        # FIX: use max deviation (standard GSEA-style)
        ES = np.max(running_ES)

        scores.append(ES)

    return pd.Series(scores, index=expr_matrix.index)


# ============================================================
# GMT parser
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
# MULTI-GENESET ssGSEA
# ============================================================
def execute_ssgsva(expr_matrix: pd.DataFrame, gmt_file: str) -> pd.DataFrame:
    genesets = parse_gmt(gmt_file)

    result_dict = {}

    for name, genes in genesets.items():
        result_dict[name] = ssgsea_score(expr_matrix, genes)

    return pd.DataFrame(result_dict, index=expr_matrix.index)


# ============================================================
# SINGLE GENESET ssGSEA
# ============================================================
def execute_ssgsea_single(
    expr_matrix: pd.DataFrame,
    gmt_file: str,
    score_name: str = "ssGSVA",
    gene_set_index: int = 0
) -> pd.DataFrame:

    with open(gmt_file) as f:
        lines = f.readlines()

    if gene_set_index >= len(lines):
        raise IndexError("gene_set_index out of range")

    parts = lines[gene_set_index].strip().split("\t")
    genes = parts[2:]

    scores = ssgsea_score(expr_matrix, genes)

    return pd.DataFrame({score_name: scores})