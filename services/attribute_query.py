"""
Attribute-query evaluation, shared by the query builder dialog, the
``query_select`` plugin, and its tests.

A query is a list of ``{field, op, value, value2}`` conditions combined with a
single AND/OR ``combiner``. Each condition tests one per-point field (resolved
via ``services.point_fields``) against a numeric value. Keeping this here -- in a
plainly importable module, not the numeric-prefixed plugin file -- lets the GUI
and the plugin share one evaluator with no path hacks, so they never drift.
"""

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from services.point_fields import resolve_field


# Operator key -> human label. Keys are stable strings persisted in params.
OPERATORS: "OrderedDict[str, str]" = OrderedDict([
    ("==", "=  (equals)"),
    ("!=", "≠  (not equal)"),
    ("<", "<  (less than)"),
    ("<=", "≤  (at most)"),
    (">", ">  (greater than)"),
    (">=", "≥  (at least)"),
    ("between", "between  (inclusive)"),
])


def _coerce_float(value):
    """Parse a clause value to float, or None if it isn't numeric."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_clause(pc, clause: Dict[str, Any]):
    """Boolean mask for a single ``{field, op, value, value2}`` condition over
    ``pc``, or None if the field is missing or the value isn't numeric.

    NaNs in a field never satisfy a comparison (NumPy returns False), so they
    are simply excluded from the selection rather than raising.
    """
    field = clause.get("field")
    op = clause.get("op")
    arr = resolve_field(pc, field)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64).ravel()

    v = _coerce_float(clause.get("value"))
    if v is None:
        return None

    if op == "==":
        return arr == v
    if op == "!=":
        return arr != v
    if op == "<":
        return arr < v
    if op == "<=":
        return arr <= v
    if op == ">":
        return arr > v
    if op == ">=":
        return arr >= v
    if op == "between":
        v2 = _coerce_float(clause.get("value2"))
        if v2 is None:
            return None
        lo, hi = (v, v2) if v <= v2 else (v2, v)
        return (arr >= lo) & (arr <= hi)
    return None


def evaluate_query(pc, clauses, combiner: str):
    """Combine the per-clause masks into one selection mask over ``pc``.

    ``combiner`` is "AND" (every valid clause must hold) or "OR" (any may).
    Clauses that don't resolve (missing field / non-numeric value) are skipped.
    Returns an all-False mask when no clause is usable, so a malformed query
    selects nothing rather than everything.
    """
    n = len(pc.points)
    masks = []
    for clause in clauses or []:
        m = evaluate_clause(pc, clause)
        if m is not None and m.shape[0] == n:
            masks.append(m)

    if not masks:
        return np.zeros(n, dtype=bool)

    combined = masks[0].copy()
    for m in masks[1:]:
        if str(combiner).upper() == "OR":
            combined |= m
        else:
            combined &= m
    return combined
