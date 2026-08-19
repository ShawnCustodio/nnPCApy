"""
Compress the full Cook 2020 count matrix for sharing on GitHub.

fetch_cook_A549_TGFB1.R writes an ASCII MatrixMarket file (~157 MB). This packs it
into a compact binary sparse .npz (~24 MB: uint16 counts + int32 CSC indices,
DEFLATE-compressed) that sc_full.load() reads directly. The .npz + genes/cells/meta
(~25 MB total) are small enough to commit to git without LFS; the raw .mtx does not
need to be committed.

Usage:
  python pack_cook_data.py            # packs every cook_full/<ds>_counts.mtx it finds
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio

DIR = Path(__file__).resolve().parent / "cook_full"


def pack(mtx_path: Path):
    ds = mtx_path.name.replace("_counts.mtx", "")
    M = sio.mmread(mtx_path).tocsc()                 # genes x cells, integer counts
    data = M.data.astype(np.uint16)
    if not np.array_equal(data, M.data):
        raise ValueError(f"{ds}: counts exceed uint16 range")
    out = DIR / f"{ds}_counts.npz"
    np.savez_compressed(out, data=data, indices=M.indices.astype(np.int32),
                        indptr=M.indptr.astype(np.int32), shape=np.array(M.shape, np.int32))
    print(f"{ds}: {mtx_path.stat().st_size/1e6:.0f} MB .mtx -> {out.stat().st_size/1e6:.0f} MB .npz "
          f"({M.shape[0]}x{M.shape[1]}, nnz={M.nnz})")


def main():
    mtx = sorted(DIR.glob("*_counts.mtx"))
    if not mtx:
        print(f"No *_counts.mtx in {DIR}. Run fetch_cook_A549_TGFB1.R first.")
        return
    for p in mtx:
        pack(p)


if __name__ == "__main__":
    main()
