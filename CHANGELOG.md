# Changelog

## 0.2.0

### Fixed

- **Components after the first could have negative loadings.** With `nneg=True`, version
  0.1.0 clipped negative entries inside the EM and then projected the loading against the
  earlier components, which put negative entries back. The loading is now only rescaled
  against the earlier components, as in R's `nsprcomp`, and the orthonormal vector used for
  deflation is kept separate from the loading. Every component is non-negative. On the 47
  benchmark inputs 0.1.0 gave negative loadings in 29 (two components) and 39 (three
  components); 0.2.0 gives none.
- Results for `ncomp > 1` therefore differ from 0.1.0. **Refits with 0.1.0 should be
  repeated.**
- `get_nnPCA_result` no longer flips the score sign by the mean of the centred scores, which
  is zero up to round-off. The loadings are non-negative, so the fitted sign already means
  higher expression.
- The E and M scores no longer have their sign set from pseudotime, in `compute_em_scores`,
  `plot_em_pc_panels_cook` and the single-cell plotting helpers.

### Added

- `random_state` for reproducible fits that leave NumPy's global generator alone.
- `LocalOptimumWarning` and a `suboptimal` flag in the result, for components whose EM has
  stopped at a local optimum that a later component shows to be improvable.
- `method="auto" | "cov" | "data"`. The new `"data"` path works on the data matrix as R does;
  sparse input (any `scipy.sparse` matrix, including AnnData's `.X`) is never made dense.
  The covariance is deflated in place instead of being rebuilt for each component.
- Sparse and AnnData input to `score_signature` and `plot_signature_heatmap`.
- Tests against output of R `nsprcomp` 0.5.1-2 (`tests/data/nsprcomp_r/`, made by
  `make_fixture.R`): loadings and standard deviations for three components under matched
  random starts.
- Code for the paper's figures: `paper/benchmark/v2/` (benchmark) and `paper/fig3/`
  (bootstrap analyses).

### Changed (may break code)

- `score_signature` no longer takes `sign_ref`.
- `emtscore.nsprcomp.nsprcomp` is now `nnpcapy.nsprcomp.nsprcomp`; the second copy of the
  solver is gone.

## 0.1.0

First release.
