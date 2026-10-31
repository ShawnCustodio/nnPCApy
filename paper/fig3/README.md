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

## Restarts

nsprcomp's default of 5 random restarts often stops at a weaker solution for some gene
sets (`pc_order.py` shows this for Fig S2). Refits here use `resample.NRESTART = 20` and
full-data reference fits `REF_NRESTART = 50`; `pc_order.py` keeps the default 5, which is
what Fig S2 is about. nsprcomp marks a component it can prove is a weaker solution
(`suboptimal`, `LocalOptimumWarning`); the scripts record the flag instead of warning.

## Scripts

| script | panels | output (`results/`, not tracked) |
|---|---|---|
| `synth_order.py` | Fig 3A | `synth_order.csv` |
| `pc_order.py` | Fig 3B, Fig S2 | `pc_order_summary.csv`, `pc_order_corr_summary.csv`, `pc_order_<arm>.npz` |
| `time_steps.py` | Fig 3C | `time_means.csv`, `time_steps.csv`, `time_steps_by_culture.csv`, `time_steps_loco.csv` |
| `c2_screen.py` | Fig 3D | `c2_screen.csv`, one file per draw in `c2_screen/` |
| `restart_scan.py` | Methods (restarts), Fig S2 | `restart_scan.csv`, `restart_scan_summary.csv`, `restart_curve_trans_golgi.csv` |

```
python synth_order.py              # B = 1000, S = 200; about 2 min
python pc_order.py                 # B = 1000, S = 200; about 20 min
python time_steps.py               # B = 2000, S = 200; a few minutes
python c2_screen.py --B 300 --workers 2    # hours: ~2-4 min per draw per worker
python restart_scan.py             # S = 50; about 70 min
```

Each script takes `--B` and `--S` (or `--limit`) for a quick check.
