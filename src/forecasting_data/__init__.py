"""Canonical, leakage-safe forecasting ground-truth construction."""

__all__ = ["build_canonical_dataset"]


def build_canonical_dataset(*args, **kwargs):
    from .build import build_canonical_dataset as _build
    return _build(*args, **kwargs)
