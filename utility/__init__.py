"""utility helpers for the EMTscore Python port.

``data_paths`` (file resolution) is stdlib-only and imported eagerly. The Cook
2020 loaders in ``load_cook2020`` need the optional ``sc`` extra (anndata, scipy)
and are exposed lazily, so ``from utility.data_paths import ...`` stays cheap and
does not require those heavy dependencies.
"""

from .data_paths import (  # noqa: F401
    resolve_data_file,
    data_root,
    find,
)

_LAZY_FUNCS = {
    "load_cook2020": "load_cook2020",
    "load_all_cook2020": "load_cook2020",
    "list_cook_datasets": "load_cook2020",
    "resolve_cook_dir": "load_cook2020",
    "COOK_DATASETS": "load_cook2020",
}


def __getattr__(name):  # PEP 562: pull Cook loaders only when actually used
    if name in _LAZY_FUNCS:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY_FUNCS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "resolve_data_file", "data_root", "find",
    "load_cook2020", "load_all_cook2020", "list_cook_datasets",
    "resolve_cook_dir", "COOK_DATASETS",
]
