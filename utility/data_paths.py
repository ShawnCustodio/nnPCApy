"""
utility/data_paths.py
---------------------
Single source of truth for where the EMTscore demo data lives.

The installed ``nnpcapy`` wheel bundles the *small* files (E/M signatures,
annotation, and the ``.gmt`` gene-set files) under ``nnpcapy/_bundled_data/``,
so signature-based scoring works straight after ``pip install``.  The *large*
files (bulk expression ``geneExp.csv``, the full MSigDB C2 ``.gmt``, and the
Cook et al. 2020 single-cell CSVs) are **not** shipped — they live next to a
checkout of the repository, or in a directory you point ``$NNPCAPY_DATA`` at.

``resolve_data_file`` walks a predictable list of candidate locations and
returns the first existing match, so notebooks and scripts Just Work from any
CWD, whether running from a source checkout or from an installed wheel.

Search order for every file:

    1. ``$NNPCAPY_DATA/<relpath>``            (explicit user override, if set)
    2. ``<repo>/data/<relpath>``             (source checkout, next to the package)
    3. ``<cwd or any ancestor>/data/<relpath>``
    4. ``<package ancestor>/data/<relpath>``
    5. ``nnpcapy/_bundled_data/<relpath>``   (small files shipped in the wheel)

Public helpers
--------------
    resolve_data_file(relpath) -> Path
    data_root() -> Path
    find(name) -> Path
"""

from __future__ import annotations

import os
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # repo root in a checkout


def _bundled_dir() -> Path | None:
    """Directory of small data files shipped inside the installed wheel, if any."""
    try:
        from importlib.resources import files
        d = files("nnpcapy").joinpath("_bundled_data")
        # Only usable when it resolves to a real filesystem path (normal installs).
        p = Path(str(d))
        return p if p.is_dir() else None
    except (ImportError, ModuleNotFoundError, TypeError, NotADirectoryError):
        return None


def _candidates(relpath: str | Path) -> list[Path]:
    rel = Path(relpath)
    seen: list[Path] = []

    # 1. Explicit user override
    env = os.environ.get("NNPCAPY_DATA")
    if env:
        seen.append(Path(env) / rel)

    # 2. Next to the package (source checkout: <repo>/data/...)
    seen.append(_PACKAGE_ROOT / "data" / rel)

    # 3. CWD + every ancestor
    cwd = Path.cwd().resolve()
    for anc in [cwd, *cwd.parents]:
        seen.append(anc / "data" / rel)

    # 4. Package ancestors
    for anc in _PACKAGE_ROOT.parents:
        seen.append(anc / "data" / rel)

    # 5. Small files bundled inside the installed wheel
    bundled = _bundled_dir()
    if bundled is not None:
        seen.append(bundled / rel)

    # deduplicate while preserving order
    out, known = [], set()
    for p in seen:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp not in known:
            out.append(p)
            known.add(rp)
    return out


# Large files that are intentionally not shipped in the wheel.
_UNBUNDLED_HINT = {
    "geneExp.csv", "c2.all.v2025.1.Hs.symbols.gmt",
}


def resolve_data_file(relpath: str | Path) -> Path:
    """Return the first existing candidate path for ``data/<relpath>``.

    Raises FileNotFoundError (listing every location tried) if the file is
    missing, with a hint for the large files that ship only with a checkout.
    """
    for cand in _candidates(relpath):
        if cand.exists():
            return cand
    tried = "\n".join(f"  {p}" for p in _candidates(relpath))
    name = Path(relpath).name
    hint = ""
    if name in _UNBUNDLED_HINT or name.endswith("_em_expr.csv"):
        hint = (
            f"\n\n'{name}' is a large file that is not bundled with the installed "
            "package. Run from a checkout of the repository (where it lives under "
            "data/), or set the NNPCAPY_DATA environment variable to a directory "
            "containing it."
        )
    raise FileNotFoundError(
        f"Could not locate data file '{relpath}'. Tried:\n{tried}{hint}"
    )


def data_root() -> Path:
    """Return a data directory that exists (checkout ``data/`` or the bundled one)."""
    checkout = _PACKAGE_ROOT / "data"
    if checkout.is_dir():
        return checkout
    bundled = _bundled_dir()
    if bundled is not None:
        return bundled
    return checkout


def find(name: str) -> Path:
    """Short alias for :func:`resolve_data_file`."""
    return resolve_data_file(name)
