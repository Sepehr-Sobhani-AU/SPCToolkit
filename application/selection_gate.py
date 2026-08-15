"""
Selection gate — shared pieces for "this plugin needs a selection".

A plugin declares what it needs at execute time via ``requires_selection()``
(see ``plugins/interfaces.py``). Two callers reuse the helpers here so the two
behave identically:

* The manual menu run path (``MainWindow._gate_selection_then``) prompts the
  user — non-modally, so the viewer and tree stay live — *only* when the needed
  selection is absent, then proceeds.
* Pipeline replay (``application/pipeline_runner.py``) *always* pauses at such a
  step, because a selection carried over from a previous step no longer applies
  to the freshly-produced intermediate.

Both share ``SelectionPrompt`` (the non-modal dialog) and ``selection_present``
(the "is something selected?" check).

No custom signals/slots — only the built-in ``QPushButton.clicked`` is used,
which the project permits (only *custom* pyqtSignals are disallowed).
"""

import logging
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from config.config import global_variables

logger = logging.getLogger(__name__)

# Normalised selection kinds returned by ``selection_kind``.
POINTS = "points"
BRANCHES = "branches"
EITHER = "either"


class SelectionPrompt(QDialog):
    """Non-modal prompt shown while a run pauses for a viewer/tree selection.

    Non-modal so the 3D viewer and tree stay interactive — the user makes their
    selection, then clicks Continue (or Cancel). The caller keeps a reference
    alive (Qt would otherwise garbage-collect a non-modal dialog) and closes it
    from the button callbacks.
    """

    def __init__(self, parent, message, on_continue, on_cancel,
                 title="Selection Needed", cancel_text="Cancel",
                 continue_text="Continue"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)

        row = QHBoxLayout()
        cancel_btn = QPushButton(cancel_text)
        continue_btn = QPushButton(continue_text)
        continue_btn.setDefault(True)
        cancel_btn.clicked.connect(on_cancel)
        continue_btn.clicked.connect(on_continue)
        row.addWidget(cancel_btn)
        row.addWidget(continue_btn)
        layout.addLayout(row)


def selection_kind(plugin_class) -> Optional[str]:
    """Normalise a plugin's ``requires_selection()`` into a selection kind.

    Returns ``"points"``, ``"branches"``, ``"either"``, or ``None`` when the
    plugin needs no selection. Tolerates the legacy boolean contract
    (``True`` ⇒ ``"points"``) and never raises — a missing/misbehaving hook is
    treated as needing nothing.
    """
    if plugin_class is None:
        return None
    try:
        req = getattr(plugin_class(), "requires_selection", None)
        if not callable(req):
            return None
        value = req()
    except Exception:
        return None

    if value is True:
        return POINTS
    if not value:  # False, None, "", 0
        return None
    text = str(value).strip().lower()
    if text in (POINTS, BRANCHES, EITHER):
        return text
    # Unknown truthy value: be permissive and treat it as its legacy meaning.
    return POINTS


def selection_present(kind: Optional[str]) -> bool:
    """Whether a selection of ``kind`` is currently available.

    Points come from the viewer (picked points or stored selection polygons);
    branches come from the controller's selected-branch list. ``None`` means no
    selection is required, so it is trivially satisfied.
    """
    if not kind:
        return True

    viewer = global_variables.global_pcd_viewer_widget
    controller = global_variables.global_application_controller

    has_points = bool(getattr(viewer, "picked_points_indices", None)) or \
        bool(getattr(viewer, "_selection_polygons", None))
    has_branches = bool(getattr(controller, "selected_branches", None))

    if kind == POINTS:
        return has_points
    if kind == BRANCHES:
        return has_branches
    if kind == EITHER:
        return has_points or has_branches
    return True


def selectable_cloud_indices(node, n_points=None):
    """Which rows of *node*'s cloud the viewer would let the user select.

    ``picked_cloud_indices`` widens a polygon selection to full resolution, and
    that widening bypasses the viewer's own filters because they work in
    rendered-index space (see that function's note). This answers the same
    question in *cloud*-index space, so callers have something to pass as
    ``allowed``: a point is admissible unless its cluster is locked against
    selection, or it is noise.

    Args:
        node: The DataNode the plugin is about to operate on.
        n_points: Length of the cloud, when the caller knows it. Used only to
            notice that the labels describe a different cloud, in which case
            they are ignored rather than trusted.

    Returns:
        Sorted ``np.intp`` array of admissible rows, or ``None`` when the node
        carries no usable cluster labels — meaning nothing is excluded, which is
        exactly what ``allowed=None`` means to ``picked_cloud_indices``.
    """
    if node is None or getattr(node, "data_type", None) != "cluster_labels":
        return None

    clusters = getattr(node, "data", None)
    labels = getattr(clusters, "labels", None)
    if labels is None:
        return None

    labels = np.asarray(labels)
    if n_points is not None and len(labels) != n_points:
        logger.warning(
            f"Cluster labels ({len(labels):,}) do not match the cloud "
            f"({n_points:,}); not filtering the selection by them."
        )
        return None

    admissible = labels != -1                       # noise is never selectable
    locked = getattr(clusters, "locked_clusters", None) or {}
    locked_ids = [cid for cid, locks in locked.items() if "select" in locks]
    if locked_ids:
        np.logical_and(admissible, ~np.isin(labels, locked_ids), out=admissible)

    return np.flatnonzero(admissible).astype(np.intp)


def picked_cloud_indices(viewer, pc_points, kdtree=None, allowed=None):
    """Map the viewer's current point picks onto indices into *pc_points*.

    The viewer renders a possibly sub-sampled copy of the branch, so a picked
    index is not an index into the reconstructed cloud. Picks are matched back by
    coordinate, then the stored selection polygons are re-tested against the full
    cloud so a polygon selection covers every point it encloses rather than only
    the sub-sampled ones the user could see.

    **That re-test does not go through the viewer's selection filters.** The
    viewer applies them when the polygon is closed (``_filter_selection``:
    branch membership, cluster locks, noise) — but they work in rendered-index
    space, and re-testing works in cloud-index space, so the widened result comes
    back ungated. A polygon drawn over a few permitted points therefore returns
    every point it encloses, permitted or not. Measured in the line-extension
    window: the viewer honestly reported 32 points picked while this handed back
    thousands, and a whole bush joined the traced line.

    A caller that knows which cloud indices are admissible passes them as
    *allowed*; the result is intersected with them. Anything that cares what it
    is picking should pass it — only the caller knows the answer in cloud-index
    space.

    Pass *kdtree* (a ``cKDTree`` over *pc_points*) when the caller already has
    one. Returns a sorted index array, or ``None`` when the picks could not be
    resolved to any coordinate at all — which callers report as "no points".

    Everything here is done with numpy rather than Python loops and sets. A
    lasso leaves millions of picked points, and at that size building a list of
    per-point slices and then a ``set`` of boxed integers costs seconds and
    hundreds of MB on its own — more than the polygon re-test it feeds.
    """
    viewer_points = viewer.points
    if viewer_points is None or not viewer.picked_points_indices:
        return None

    picked_rows = np.fromiter(viewer.picked_points_indices, dtype=np.intp,
                              count=len(viewer.picked_points_indices))
    picked_rows = picked_rows[(picked_rows >= 0) & (picked_rows < len(viewer_points))]
    if picked_rows.size == 0:
        return None

    if kdtree is None:
        kdtree = cKDTree(pc_points)
    _dist, local = kdtree.query(
        np.ascontiguousarray(viewer_points[picked_rows, :3], dtype=np.float32))
    indices = np.atleast_1d(np.asarray(local, dtype=np.intp))

    polygon_mask = viewer.retest_polygon_selection(pc_points)
    if polygon_mask is not None:
        # union with the re-tested polygon, still sorted and unique afterwards
        indices = np.union1d(indices, np.flatnonzero(polygon_mask))
    else:
        indices = np.unique(indices)

    indices = indices.astype(np.intp, copy=False)
    if allowed is None:
        return indices
    return np.intersect1d(indices, np.asarray(allowed, dtype=np.intp))
