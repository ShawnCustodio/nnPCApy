"""
utility/data_paths.py
---------------------
Single source of truth for where the EMTscore demo data lives.

The Python_Conv package ships with small signature files bundled in
``Python_Conv/data/``, but large CSVs (bulk expression, Cook et al. 2020
counts) are expected to live next to the user's checkout of the PythonLab
repo.  ``resolve_data_file`` walks a predictable list of candidate
directories, returning the first existing match, so notebooks Just Work
from any CWD.

Search order for every file:

    1. bundled: ``<Python_Conv>/data/<relpath>``
    2. each ``<ancestor>/data/<relpath>`` walking up from CWD
    3. each ``<ancestor>/data/<relpath>`` walking up from the package root

Public helpers
--------------
    resolve_data_file(relpath) -> Path
    data_root() -> Path
    find(name) -> Path
"""

from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # Python_Conv/


def _candidates(relpath: str | Path) -> list[Path]:
    rel = Path(relpath)
    seen: list[Path] = []

    # 1. Bundled with Python_Conv
    seen.append(_PACKAGE_ROOT / "data" / rel)

    # 2. CWD + every ancestor
    cwd = Path.cwd().resolve()
    for anc in [cwd, *cwd.parents]:
        seen.append(anc / "data" / rel)

    # 3. Package ancestors (e.g. PythonLab/data/...)
    for anc in _PACKAGE_ROOT.parents:
        seen.append(anc / "data" / rel)

    # deduplicate while preserving order
    out, known = [], set()
    for p in seen:
        rp = p.resolve()
        if rp not in known:
            out.append(p)
            known.add(rp)
    return out


def resolve_data_file(relpath: str | Path) -> Path:
    """Return the first existing candidate path for ``data/<relpath>``.

    Raises FileNotFoundError with every path tried if the file is missing.
    """
    for cand in _candidates(relpath):
        if cand.exists():
            return cand
    tried = "\n".join(f"  {p}" for p in _candidates(relpath))
    raise FileNotFoundError(
        f"Could not locate data file '{relpath}'. Tried:\n{tried}"
    )


def data_root() -> Path:
    """Return the bundled data directory (may or may not contain everything)."""
    return _PACKAGE_ROOT / "data"


def find(name: str) -> Path:
    """Short alias for :func:`resolve_data_file`."""
    return resolve_data_file(name)
