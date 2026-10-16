"""Deterministic, no-imputation audit for the thesis PV candidate corpus."""


def run_audit(*args, **kwargs):
    """Load the report runner lazily so ``python -m data_audit.report`` stays clean."""
    from .report import run_audit as _run_audit

    return _run_audit(*args, **kwargs)


__all__ = ["run_audit"]
