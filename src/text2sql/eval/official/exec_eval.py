"""Adapter: expose the vendored test-suite matcher in the shape execution.py wants.

execution.py imports `eval_exec_match(db_path, predicted, gold) -> bool` from here.
The upstream `eval_exec_match` (test-suite-sql-eval) does single-instance result
comparison but with a wider, version-dependent signature; this bridges it.

If `_vendor/` isn't populated yet, the import below raises and execution.py
falls back to its local comparison — that's intended.
"""

from __future__ import annotations

# Upstream single-instance matcher. Populate official/_vendor/ per README.
from ._vendor.exec_eval import eval_exec_match as _impl  # type: ignore

__all__ = ["eval_exec_match"]


def eval_exec_match(db_path: str, predicted: str, gold: str) -> bool:
    """Whether `predicted` and `gold` agree on ONE database instance.

    execution.py calls this once per seeded instance and requires agreement on
    all of them (that's the test-suite part). We keep value-plugging off and do
    not force DISTINCT, matching plain execution semantics.

    NOTE: upstream keyword names have varied. If your vendored copy differs,
    adjust the single call below.
    """
    try:
        return bool(
            _impl(
                db=db_path,
                p_str=predicted,
                g_str=gold,
                plug_value=False,
                keep_distinct=False,
                progress_bar_for_each_datapoint=False,
            )
        )
    except TypeError:
        # Fall back to a positional call for older/newer signatures.
        return bool(_impl(db_path, predicted, gold, False, False, False))