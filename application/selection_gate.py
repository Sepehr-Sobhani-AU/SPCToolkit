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


def picked_cloud_indices(viewer, pc_points, kdtree=None):
    """Map the viewer's current point picks onto indices into *pc_points*.

    The viewer renders a possibly sub-sampled copy of the branch, so a picked
    index is not an index into the reconstructed cloud. Picks are matched back by
    coordinate, then the stored selection polygons are re-tested against the full
    cloud so a polygon selection covers every point it encloses rather than only
    the sub-sampled ones the user could see.

    Pass *kdtree* (a ``cKDTree`` over *pc_points*) when the caller already has
    one. Returns a sorted index array, or ``None`` when the picks could not be
    resolved to any coordinate at all — which callers report as "no points".
    """
    coords = [viewer.points[i, :3] for i in viewer.picked_points_indices
              if i < len(viewer.points)]
    if not coords:
        return None

    if kdtree is None:
        kdtree = cKDTree(pc_points)
    _dist, local = kdtree.query(np.asarray(coords, dtype=np.float32))
    picked = set(int(i) for i in np.atleast_1d(local))

    polygon_mask = viewer.retest_polygon_selection(pc_points)
    if polygon_mask is not None:
        picked |= set(int(i) for i in np.where(polygon_mask)[0])

    return np.array(sorted(picked), dtype=np.intp)
