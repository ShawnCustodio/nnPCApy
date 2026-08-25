"""Tests for AUCell and ssGSEA (src/nnpcapy and emtscore), previously untested.

Covers: correctness on constructed ground truth, edge cases (missing genes,
empty gene sets), the single/multi gene-set GMT entry points, and that the
"dual-module" src/nnpcapy vs emtscore copies stay in sync -- these are
meant to be identical implementations.
"""

import numpy as np
import pandas as pd
import pytest

from emtscore import aucell as emt_aucell
from emtscore import ssGSEA as emt_ssgsea
from nnpcapy import aucell as nn_aucell
from nnpcapy import ssGSEA as nn_ssgsea


def _expr_matrix(rng, n_samples=6, n_genes=40):
    cols = [f"G{i}" for i in range(n_genes)]
    idx = [f"S{i}" for i in range(n_samples)]
    data = rng.random((n_samples, n_genes))
    return pd.DataFrame(data, index=idx, columns=cols)


# --------------------------------------------------------------------------- #
# AUCell
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mod", [nn_aucell, emt_aucell])
def test_aucell_full_overlap_scores_one(mod):
    rng = np.random.default_rng(0)
    expr = _expr_matrix(rng)
    # top_percent=0.05 * 40 genes -> max_rank = 2; make the signature exactly
    # the top-2 genes for every sample so enrichment must be 1.0.
    top_genes = ["G0", "G1"]
    for s in expr.index:
        expr.loc[s, top_genes] = expr.loc[s].max() + 10

    scores = mod.aucell_score(expr, top_genes, top_percent=0.05)
    assert (scores == 1.0).all()


@pytest.mark.parametrize("mod", [nn_aucell, emt_aucell])
def test_aucell_no_overlap_scores_zero(mod):
    rng = np.random.default_rng(1)
    expr = _expr_matrix(rng)
    # Force the signature genes to the bottom of every sample's ranking.
    sig = ["G0", "G1"]
    for s in expr.index:
        expr.loc[s, sig] = expr.loc[s].min() - 10

    scores = mod.aucell_score(expr, sig, top_percent=0.05)
    assert (scores == 0.0).all()


@pytest.mark.parametrize("mod", [nn_aucell, emt_aucell])
def test_aucell_missing_genes_filtered_silently(mod):
    rng = np.random.default_rng(2)
    expr = _expr_matrix(rng)
    scores = mod.aucell_score(expr, ["G0", "NOT_A_GENE"], top_percent=0.5)
    assert not scores.isna().any()


@pytest.mark.parametrize("mod", [nn_aucell, emt_aucell])
def test_aucell_empty_gene_set_returns_nan(mod):
    rng = np.random.default_rng(3)
    expr = _expr_matrix(rng)
    scores = mod.aucell_score(expr, ["NOT_A_GENE", "ALSO_MISSING"])
    assert scores.isna().all()


@pytest.mark.parametrize("mod", [nn_aucell, emt_aucell])
def test_execute_aucell_single_and_multi(mod, tmp_path):
    rng = np.random.default_rng(4)
    expr = _expr_matrix(rng, n_genes=20)
    gmt = tmp_path / "sets.gmt"
    gmt.write_text(
        "SET_A\tdesc\tG0\tG1\tG2\n"
        "SET_B\tdesc\tG10\tG11\n"
    )

    multi = mod.execute_aucell(expr, str(gmt))
    assert list(multi.columns) == ["SET_A", "SET_B"]
    assert len(multi) == len(expr)

    single = mod.execute_aucell_single(expr, str(gmt), score_name="SET_A_score", gene_set_index=0)
    assert list(single.columns) == ["SET_A_score"]
    pd.testing.assert_series_equal(
        single["SET_A_score"], multi["SET_A"], check_names=False
    )


# --------------------------------------------------------------------------- #
# ssGSEA
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mod", [nn_ssgsea, emt_ssgsea])
def test_ssgsea_top_ranked_signature_scores_positive(mod):
    rng = np.random.default_rng(5)
    expr = _expr_matrix(rng, n_genes=50)
    sig = ["G0", "G1", "G2"]
    for s in expr.index:
        expr.loc[s, sig] = expr.loc[s].max() + 10

    scores = mod.ssgsea_score(expr, sig)
    assert (scores > 0).all()


@pytest.mark.parametrize("mod", [nn_ssgsea, emt_ssgsea])
def test_ssgsea_bottom_ranked_signature_scores_negative(mod):
    rng = np.random.default_rng(6)
    expr = _expr_matrix(rng, n_genes=50)
    sig = ["G0", "G1", "G2"]
    for s in expr.index:
        expr.loc[s, sig] = expr.loc[s].min() - 10

    scores = mod.ssgsea_score(expr, sig)
    assert (scores < 0).all()


@pytest.mark.parametrize("mod", [nn_ssgsea, emt_ssgsea])
def test_ssgsea_no_hits_returns_nan(mod):
    rng = np.random.default_rng(7)
    expr = _expr_matrix(rng)
    scores = mod.ssgsea_score(expr, ["NOT_A_GENE"])
    assert scores.isna().all()


@pytest.mark.parametrize("mod", [nn_ssgsea, emt_ssgsea])
def test_execute_ssgsea_single_out_of_range_raises(mod, tmp_path):
    rng = np.random.default_rng(8)
    expr = _expr_matrix(rng, n_genes=10)
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("SET_A\tdesc\tG0\tG1\n")
    with pytest.raises(IndexError):
        mod.execute_ssgsea_single(expr, str(gmt), gene_set_index=5)


# --------------------------------------------------------------------------- #
# src/nnpcapy vs emtscore parity (the "dual-module pattern")
# --------------------------------------------------------------------------- #

def test_aucell_src_and_emtscore_copies_agree():
    rng = np.random.default_rng(9)
    expr = _expr_matrix(rng, n_genes=30)
    sig = ["G3", "G7", "G12", "G20"]
    a = nn_aucell.aucell_score(expr, sig)
    b = emt_aucell.aucell_score(expr, sig)
    pd.testing.assert_series_equal(a, b)


def test_ssgsea_src_and_emtscore_copies_agree():
    rng = np.random.default_rng(10)
    expr = _expr_matrix(rng, n_genes=30)
    sig = ["G3", "G7", "G12", "G20"]
    a = nn_ssgsea.ssgsea_score(expr, sig)
    b = emt_ssgsea.ssgsea_score(expr, sig)
    pd.testing.assert_series_equal(a, b)
