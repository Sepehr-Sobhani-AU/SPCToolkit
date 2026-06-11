"""
Lineage-aware helpers for branch set operations (subtract / intersect / union).

When branches descend from a common ancestor through a *pure subset chain* -- a
path made only of boolean ``masks`` selections, attribute-only "identity"
transforms, and ``class_reference`` filters -- their set relationships can be
expressed exactly as boolean masks over that ancestor's points, with no
reconstruction and no coordinate matching:

    subtract  A - B = inA & ~inB
    intersect A & B = inA &  inB
    union     A | B = inA |  inB

where ``inX`` is the per-point membership mask of branch X over the common
ancestor. This module provides:

- :func:`lowest_common_ancestor` / :func:`membership_mask_over_ancestor` --
  the primitives intersect and union use to compose those boolean combinations.
- :func:`row_membership` -- the GPU/CPU coordinate-matching fallback used when
  no compatible lineage exists.

The fast paths mirror the conventions established by ``SubtractPlugin``.
"""

from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover - CuPy is optional
    _cp = None
    _HAS_CUPY = False


# Node types whose transformer preserves point identity, order and count (they
# only attach per-point attributes / colors or pass the cloud through). Mirrors
# ``SubtractPlugin._IDENTITY_TYPES``: cluster_labels merely adds a per-point
# label attribute (no subsetting); transform_matrix MOVES points but keeps the
# same point at the same index, so index-based masks stay valid.
IDENTITY_TYPES = frozenset({
    "values", "eigenvalues", "colors", "normals", "dist_to_ground",
    "cluster_labels", "container", "vector_feature", "cad_object",
    "transform_matrix",
})


def ancestor_chain(node, data_nodes) -> list:
    """Return the chain from ``node`` up to root, inclusive (node first)."""
    chain = []
    current = node
    while current is not None:
        chain.append(current)
        current = data_nodes.get_node(current.parent_uid) if current.parent_uid else None
    return chain


def _find_cluster_labels(node, data_nodes):
    """
    Per-point integer labels of the nearest ``cluster_labels`` ancestor of
    ``node`` (the array a ``class_reference`` filters on), or None.
    """
    current = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
    while current is not None:
        if current.data_type == "cluster_labels":
            return getattr(current.data, "labels", None)
        current = data_nodes.get_node(current.parent_uid) if current.parent_uid else None
    return None


def lowest_common_ancestor(nodes: list, data_nodes):
    """
    The deepest node present in every node's ancestor chain, or None if the
    nodes share no common ancestor.
    """
    if not nodes:
        return None
    chains = [ancestor_chain(n, data_nodes) for n in nodes]
    other_sets = [set(n.uid for n in chain) for chain in chains[1:]]
    for candidate in chains[0]:  # ordered nearest-first
        if all(candidate.uid in s for s in other_sets):
            return candidate
    return None


def membership_mask_over_ancestor(branch, ancestor, data_nodes, n_ancestor) -> Optional[np.ndarray]:
    """
    Boolean mask over the ancestor's points marking which survive into
    ``branch`` (True = that ancestor point is present in the branch).

    Walks ancestor -> branch, composing the subset chain into indices over the
    ancestor's ``n_ancestor`` points. Selections preserve ascending index order,
    so ``np.flatnonzero(mask)`` lines up positionally with the branch's
    reconstructed points.

    Returns None if the ancestor -> branch path contains a transform that isn't
    a pure subset (i.e. reorders or rebuilds points), in which case the caller
    must fall back to coordinate matching.
    """
    # Collect the path branch -> ... -> child-of-ancestor (walking up).
    sub_path = []
    node = branch
    reached = False
    while node is not None:
        if node.uid == ancestor.uid:
            reached = True
            break
        sub_path.append(node)
        node = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
    if not reached:
        return None

    # Compose top-down (ancestor -> branch): start with every ancestor index and
    # narrow it through each subset step.
    idx = np.arange(n_ancestor)
    for node in reversed(sub_path):
        data_type = node.data_type
        if data_type in IDENTITY_TYPES:
            continue  # Count and order preserved; nothing to select.
        if data_type == "masks":
            mask = node.data.mask
            if mask.shape[0] != idx.shape[0]:
                return None  # Lineage assumption violated; bail out safely.
            idx = idx[mask]
        elif data_type == "class_reference":
            labels = _find_cluster_labels(node, data_nodes)
            if labels is None or labels.shape[0] != idx.shape[0]:
                return None
            sel = np.isin(labels, node.data.cluster_ids)
            idx = idx[sel]
        else:
            return None  # Unknown / reordering transform -> can't reason.

    out = np.zeros(n_ancestor, dtype=bool)
    out[idx] = True
    return out


# === Coordinate-matching fallback ===========================================

# Rows processed per cancel-checked chunk. Bounds how long a Cancel click takes
# to take effect on the exact-match path (the heavy fallback).
_MATCH_CHUNK = 2_000_000


def row_membership(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Boolean mask of rows in A that ARE present in B (exact match).

    On GPU (CuPy) runs the whole pipeline on the device; on CPU reduces rows to
    void-views and uses ``np.isin``. Cancellable in bounded chunks.
    """
    n = len(A)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if len(B) == 0:
        return np.zeros(n, dtype=bool)

    common_dtype = np.promote_types(A.dtype, B.dtype)
    A = np.ascontiguousarray(A, dtype=common_dtype)
    B = np.ascontiguousarray(B, dtype=common_dtype)

    if A.shape[1] != B.shape[1]:
        raise ValueError("Point dimensions do not match between the two point clouds")

    if _HAS_CUPY:
        try:
            return _row_membership_gpu(A, B)
        except Exception as exc:
            logger.warning("Row-membership: CuPy path failed (%s); falling back to NumPy.", exc)
    return _row_membership_cpu(A, B)


def _cancel_event():
    from config.config import global_variables
    return global_variables.global_cancel_event


def _row_membership_gpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    cancel_event = _cancel_event()
    n = len(A)

    B_unique = _cp.unique(_cp.asarray(B), axis=0)
    if cancel_event.is_set():
        raise RuntimeError("Operation cancelled.")

    mask = np.empty(n, dtype=bool)
    for start in range(0, n, _MATCH_CHUNK):
        if cancel_event.is_set():
            raise RuntimeError("Operation cancelled.")
        end = min(start + _MATCH_CHUNK, n)
        chunk = _cp.asarray(A[start:end])
        stacked = _cp.concatenate([chunk, B_unique], axis=0)
        # cp.unique(axis=0) deduplicates whole rows; return_inverse gives a
        # shared id space for the chunk's rows and B's rows.
        _, inv = _cp.unique(stacked, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        c_ids = inv[: end - start]
        b_ids = inv[end - start:]
        mask[start:end] = _cp.asnumpy(_cp.isin(c_ids, b_ids))
    return mask


def _row_membership_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    cancel_event = _cancel_event()
    n = len(A)

    row_bytes = A.dtype.itemsize * A.shape[1]
    void_dtype = np.dtype((np.void, row_bytes))

    B_keys = np.unique(np.ascontiguousarray(B).reshape(-1).view(void_dtype))
    if cancel_event.is_set():
        raise RuntimeError("Operation cancelled.")

    A_void = np.ascontiguousarray(A).reshape(-1).view(void_dtype)
    mask = np.empty(n, dtype=bool)
    for start in range(0, n, _MATCH_CHUNK):
        if cancel_event.is_set():
            raise RuntimeError("Operation cancelled.")
        end = min(start + _MATCH_CHUNK, n)
        mask[start:end] = np.isin(A_void[start:end], B_keys)
    return mask
