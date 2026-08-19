# Benchmark

R-vs-Python timing comparison for `nsprcomp`. Methodology and headline
numbers are in the [top-level README](../README.md#benchmark).

Files:

- `datasets.py` builds 47 cached `.npy` input matrices from the EMT
  signature + expression data.
- `bench_python.py` times `nnpcapy.nsprcomp` on each input.
- `bench_r.R` does the same on the R side using the original `nsprcomp`
  package.
- `compare.py` merges both timing CSVs into `results/summary.csv` and
  writes the comparison plots.

Run order:

```
python datasets.py
python bench_python.py
# In RStudio: open bench_r.R and Source it
python compare.py
```

`_inputs/` is gitignored - it's a ~130 MB cache that `datasets.py`
regenerates from scratch. `results/` and `plots/` are committed so the
numbers from the most recent run are visible without re-running.
