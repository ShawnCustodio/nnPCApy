"""
2.1 - 2.4   Load inputs (cell annotation, gene expression, signatures, GMT)

Load bulk expression data, E and M signature definitions, and build a GMT
file for use with multi-method scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utility.data_paths import resolve_data_file


@dataclass
class Inputs:
    cell_annot: pd.DataFrame
    geneExp:    pd.DataFrame   
    E_sig:      pd.DataFrame
    M_sig:      pd.DataFrame
    gmt_path:   str


def load_inputs(verbose: bool = True) -> Inputs:
    """2.1-2.4 - load cell annotation, bulk expression, signatures, build GMT."""
    cell_annot = pd.read_csv(resolve_data_file("cell_annotation_file.csv"))

    geneExp = pd.read_csv(resolve_data_file("geneExp.csv"), index_col=0).T
    if verbose:
        print(f"geneExp shape: {geneExp.shape}  (samples × genes)")

    E_sig = pd.read_csv(resolve_data_file("Panchy_et_al_E_signature.csv"))
    M_sig = pd.read_csv(resolve_data_file("Panchy_et_al_M_signature.csv"))
    if verbose:
        print("E signature:", E_sig.shape)
        print("M signature:", M_sig.shape)

    # Write GMT next to the data dir we resolved geneExp from
    gmt_path = str(resolve_data_file("Panchy_et_al_E_signature.csv").parent
                   / "EM_signature.gmt")
    E_genes = E_sig["GeneName"].tolist()
    M_genes = M_sig["GeneName"].tolist()
    with open(gmt_path, "w") as f:
        f.write("Panchy_et_al_E_signature\tNA\t" + "\t".join(E_genes) + "\n")
        f.write("Panchy_et_al_M_signature\tNA\t" + "\t".join(M_genes) + "\n")
    if verbose:
        print(f"GMT saved: {gmt_path}")

    return Inputs(cell_annot, geneExp, E_sig, M_sig, gmt_path)
