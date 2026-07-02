"""Vendored official Spider evaluator (see README.md).

Empty by default: drop the upstream files under _vendor/ to enable it. Until
then, execution.py and difficulty.py use their built-in fallbacks.
"""

from __future__ import annotations

__all__ = ["is_available"]


def is_available() -> bool:
    """True if the vendored execution matcher is importable."""
    try:
        from .exec_eval import eval_exec_match  # noqa: F401

        return True
    except Exception:
        return False