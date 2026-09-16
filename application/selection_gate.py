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

    Points come from the viewer's per-branch selection masks; branches come from
    the controller's selected-branch list. ``None`` means no selection is
    required, so it is trivially satisfied.
    """
    if not kind:
        return True

    viewer = global_variables.global_pcd_viewer_widget
    controller = global_variables.global_application_controller

    has_selection = getattr(viewer, "has_selection", None)
    has_points = bool(has_selection()) if callable(has_selection) else False
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

    A point is admissible unless its cluster is locked against selection, or it
    is noise. The viewer applies this once, in cloud space, as each selection
    gesture completes — see ``PCDViewerWidget.selectable_cloud_mask``, which
    wraps this so there is one definition of "selectable" rather than two.

    It used to be something each plugin passed back in as ``allowed=``, because
    the viewer's own filters ran in rendered-index space and the full-resolution
    widening could not reach them. Nothing has to remember to pass it now.

    Args:
        node: The DataNode the plugin is about to operate on.
        n_points: Length of the cloud, when the caller knows it. Used only to
            notice that the labels describe a different cloud, in which case
            they are ignored rather than trusted.

    Returns:
        Sorted ``np.intp`` array of admissible rows, or ``None`` when the node
        carries no usable cluster labels — meaning nothing is excluded.
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


def selected_cloud_mask(viewer, uid, pc_points=None):
    """Branch *uid*'s selection as a boolean mask over its full-resolution cloud.

    THE plugin-facing read of the selection — prefer it over
    ``selected_cloud_indices``. The viewer already holds the selection in this
    exact form, in the branch's own cloud order, so this is a lookup and
    nothing else.

    A mask is what nearly every consumer wants, because what they do with the
    answer is gather: ``labels[mask]``, ``points[mask]``, ``annotations[mask]``.
    Indices gather identically, cost eight bytes per selected point to
    materialise, and can point past the end of an array — which is why the
    plugins that took them all carried a ``rows[rows < len(labels)]`` clamp. A
    mask either matches the cloud's length or is refused below, so there is
    nothing to clamp.

    Returns None when nothing is selected in that branch. That is deliberately
    distinct from an all-False mask: a plugin handed None should tell the user
    it has no selection to work with, rather than run on nothing and appear to
    succeed. (The viewer never stores an all-False mask — see
    ``set_branch_selection`` — so None is the only way "nothing" arrives.)

    Pass *pc_points* when the caller has the cloud to hand and wants the mask
    checked against its length before use; a mask that describes a different
    cloud is refused rather than returned misaligned.
    """
    if viewer is None:
        return None
    reader = getattr(viewer, "selection_mask_for_cloud", None)
    if not callable(reader):
        return None
    return reader(uid, pc_points)


def selected_cloud_indices(viewer, uid, pc_points=None):
    """Branch *uid*'s selected rows, as indices into its full-resolution cloud.

    ``selected_cloud_mask`` is the better read for almost every caller; this one
    is for the two that genuinely need positions rather than a per-point
    yes/no — intersecting the selection with another index list
    (``line_extension_window``), and carrying a subset of it forward as rows
    (``split_clusters``). Everything else gathers, and a mask gathers just as
    well without paying for the index array.

    Returns None when nothing is selected, as ``selected_cloud_mask`` does.

    This replaced a function that took the viewer's rendered picks, matched them
    back to the cloud through a ``cKDTree``, and unioned the result with a
    re-test of the stored selection polygons. All of that was the cost of
    deriving the answer lazily, per plugin, per call — the selection is now
    built once in cloud space when the gesture completes.
    """
    mask = selected_cloud_mask(viewer, uid, pc_points)
    if mask is None:
        return None
    # int32 is enough: a branch would need 2.1 billion points to overflow it,
    # and these are rows within ONE branch.
    return np.flatnonzero(mask).astype(np.int32)
