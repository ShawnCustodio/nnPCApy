# Fig 3 and Fig S2: resampling analyses

Uncertainty analyses on the Cook et al. (2020) A549 TGF-β time course
(`../panelE/cook_full/`, loaded by `../panelE/sc_full.py`). Every score is library-size
log-CPM, z-scored per gene, then nnPCA (`nnpcapy.nsprcomp`, non-negative, centred).

## Three arms

`resample.py` refits each statistic in three ways, so the variation from nnPCA's random
starts can be told apart from the variation from resampling cells:

| arm | cells | random starts |
|---|---|---|
| `seed` | all cells | new each draw |
| `sampling` | resampled | the same in every draw (`nsprcomp(random_state=...)`) |
| `both` | resampled (same cells as `sampling`) | new each draw |

Cells are resampled with replacement within time point × culture. Cook et al. ran the time
course twice with two plates each time, so Mix1–Mix4 are four independent cultures; Mix3a/b
and Mix4a/b are two 10x libraries of one plate and are merged. The intervals describe
uncertainty given these cultures. Intervals are 95% percentile intervals; support
frequencies and interval endpoints come with Monte Carlo standard errors.

## Scripts

| script | panels | output (`results/`, not tracked) |
|---|---|---|
| `pc_order.py` | Fig 3B, Fig S2 | `pc_order_summary.csv`, `pc_order_corr_summary.csv`, `pc_order_<arm>.npz` |

```
python pc_order.py                 # B = 1000 resamples, S = 200 seeds; ~30 min on a laptop
python pc_order.py --B 20 --S 10   # quick check
```
