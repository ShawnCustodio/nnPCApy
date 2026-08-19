# Reviewer checklist — `nnpcapy`

A short guide for a collaborator checking this package before the PR is merged.
The highest-signal check is a **clean-room install in a fresh environment** (ideally
on a machine/OS other than the author's), not just reading the diff.

## 1. Clean-room install

```bash
git clone -b reorg https://github.com/ShawnCustodio/nnPCApy.git
cd nnPCApy
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install ".[full,dev]"
```

Expected: installs with no errors. `[full]` pulls the scoring/plotting/single-cell
extras; `[dev]` adds pytest + ruff.

## 2. Imports and light-import contract

```bash
python -c "import nnpcapy, emtscore; print('import ok')"
```

- [ ] Both import without error.
- [ ] `import emtscore` is **light** — it must not eagerly load matplotlib/scipy/anndata.
      Verify:
      ```bash
      python -c "import sys, emtscore; \
        print('heavy loaded:', any(m in sys.modules for m in ('matplotlib.pyplot','scipy','anndata')))"
      ```
      Expected: `heavy loaded: False`.

## 3. Tests and lint

```bash
pytest -q
ruff check src/ emtscore/ tests/
```

- [ ] All tests pass (currently 3).
- [ ] `ruff check` reports **All checks passed** — it uses the profile pinned in
      `pyproject.toml` (`[tool.ruff]`), so results are the same on any machine
      regardless of a global ruff config. If you see findings from rules like `S`,
      `BLE`, `C4`, `N`, or `EXE`, you're picking up a global config instead of the
      project's — run from the repo root so `pyproject.toml` takes precedence.

## 4. Build

```bash
python -m build
```

- [ ] Wheel and sdist build cleanly.
- [ ] The wheel bundles the application layer and small data, and **excludes** the
      large files and all of `paper/`:
      ```bash
      python - <<'PY'
      import glob, zipfile
      z = zipfile.ZipFile(glob.glob("dist/*.whl")[0])
      names = z.namelist()
      assert any(n.startswith("emtscore/") for n in names), "emtscore missing from wheel"
      assert any("_bundled_data/" in n for n in names), "bundled data missing"
      assert not any(n.startswith("paper/") for n in names), "paper/ leaked into wheel"
      assert not any("geneExp.csv" in n for n in names), "large data leaked into wheel"
      print("wheel layout ok")
      PY
      ```

## 5. Scoring + plotting API (the "easy-to-use" goal)

Works on user-supplied data, no checkout required:

```bash
python - <<'PY'
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, emtscore as emt
rng = np.random.default_rng(0)
genes = [f"G{i}" for i in range(40)]
counts = pd.DataFrame(rng.poisson(5, size=(200, 40)), columns=genes)
s, load, used = emt.score_signature(counts, genes[:15], ncomp=2)
scores = pd.DataFrame({"E score": s[:,0], "M score": s[:,1],
                       "grp": rng.choice(list("ABC"), 200)})
fig, ax = emt.plot_em_scatter(scores, "E score", "M score", group="grp")
fig.savefig("scatter.png", bbox_inches="tight")
pc = emt.top_loading_genes(load, used, n=4)
emt.plot_signature_heatmap(counts, {"PC1": pc[0], "PC2": pc[1]},
                           cell_group=scores["grp"].to_numpy(),
                           group_order=list("ABC")).savefig("heatmap.png", bbox_inches="tight")
print("plots written")
PY
```

- [ ] `scatter.png` and `heatmap.png` are produced without error.

## 6. Data resolution behaves

- [ ] Bundled small files resolve after a plain install:
      ```bash
      python -c "from utility.data_paths import resolve_data_file as r; \
        print(r('Panchy_et_al_E_signature.csv'))"
      ```
- [ ] Large files (not shipped) fail with a **clear, actionable** error rather than a
      traceback about a missing path:
      ```bash
      python -c "from utility.data_paths import resolve_data_file as r; r('geneExp.csv')"
      ```
      Expected: `FileNotFoundError` explaining the file isn't bundled and to use a
      checkout or `$NNPCAPY_DATA`.

## 7. Figure reproduction (from the checkout)

```bash
python paper/panelE/fig3E_showcase.py     # reproduces the Fig 3E scatter + heatmap
# Optional, slower: the speedup benchmark
python paper/benchmark/datasets.py && python paper/benchmark/bench_python.py
```

- [ ] `fig3E_showcase.py` runs and writes plots under `paper/panelE/plots/`.

## What to look for / red flags

- Any import of `emtscore` pulling in heavy deps (breaks the light-import contract).
- `paper/` drafts, exploratory scripts, or large/regenerable data reappearing in the
  wheel or the tree.
- Data-loading functions raising bare `FileNotFoundError`/`StopIteration` instead of the
  guided message.
- Hard-coded absolute paths (scripts should locate the repo root themselves).
