"""
Isolated run — processing only the points inside the viewer's isolation box.

When isolation is on and the user starts a process from a menu that works on
points, ``MainWindow`` asks whether to run it on the isolated points or on all
points. For the isolated points, each selected point branch first gets a mask
branch holding every point inside the box — from the FULL data, not just what
level of detail drew — and the process runs on that branch like on any other.
So no plugin needs a change, the result is a new branch as always, and it
stays inside the box: a traced line stops at the box edge.

A mask branch is made only when a process runs on the isolated points;
isolating just to view never makes one. When the box has not changed since a
branch was masked, that mask branch is reused rather than made again. The
source data is never touched.

The selection travels with the points: whatever was selected in the source
branch inside the box is selected in its mask branch, clicks in click order,
so plugins that need a selection still get it.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt

from config.config import global_variables

logger = logging.getLogger(__name__)

# Top-level menus whose plugins process the points of the selected branches.
# Viewing, measuring, branch bookkeeping, selection and cluster editing are not
# offered an isolated run: they either act on the view or edit a branch in
# place, and the selection is already limited to what the box shows.
PROCESSING_MENUS = ("Points", "Features", "Classify", "Segmentation", "Vectorise",
                    "Machine Learning")

# Node types that are not point data and so cannot be masked.
_NOT_POINTS = ("vector_feature", "cad_object", "container")

BRANCH_NAME = "Isolated Points"


def processes_points(menu_path: Optional[str]) -> bool:
    """Whether a plugin in *menu_path* (e.g. "060_Segmentation/020_Lines")
    processes points, and so should be offered an isolated run."""
    if not menu_path:
        return False
    top = re.sub(r"^\d+_", "", menu_path.split("/")[0])
    return top in PROCESSING_MENUS


def child_selection(parent_mask: Optional[np.ndarray],
                    inside: np.ndarray) -> Optional[np.ndarray]:
    """The source branch's selection, restricted to the box, in the mask
    branch's rows. None when nothing selected lies inside."""
    if parent_mask is None or len(parent_mask) != len(inside):
        return None
    mask = np.asarray(parent_mask, dtype=bool)[inside]
    return mask if mask.any() else None


def child_picks(picks: List[Tuple[str, int]], uid: str,
                inside: np.ndarray) -> List[int]:
    """Click picks of branch *uid* that lie inside the box, as rows of the mask
    branch, in click order."""
    rank = np.cumsum(inside) - 1          # source row -> mask-branch row
    return [int(rank[row]) for pick_uid, row in picks
            if pick_uid == uid and 0 <= row < len(inside) and inside[row]]


class IsolatedRun:
    """Makes (or reuses) the mask branches an isolated run works on.

    Remembers which mask branch was made for which source branch and box, so
    running a second process on the same box reuses it.
    """

    def __init__(self):
        # (source uid, box key) -> mask branch uid
        self._made: Dict[Tuple[str, tuple], str] = {}

    def branches_for(self, uids: List[str], box) -> List[str]:
        """The branches to run on instead of *uids*.

        Point branches are swapped for their isolated mask branch; anything
        that is not point data (a vector feature) is kept as it is, as is a
        branch that already IS the mask branch for this box.

        Raises:
            ValueError: if the box holds no points of any selected branch.
        """
        key = box.key()
        made_for_box = {v for (_src, k), v in self._made.items() if k == key}
        controller = global_variables.global_application_controller

        out, any_points = [], False
        for uid in uids:
            node = controller.get_node(uid)
            if node is None or node.data_type in _NOT_POINTS or uid in made_for_box:
                out.append(uid)
                any_points = any_points or uid in made_for_box
                continue
            mask_uid = self._reuse(uid, key) or self._make(uid, node, box)
            if mask_uid is not None:
                out.append(mask_uid)
                any_points = True

        if not any_points:
            raise ValueError("The isolation box holds no points of the selected "
                             "branches. Move or grow the box, or run on all points.")
        return out

    def _reuse(self, uid: str, key: tuple) -> Optional[str]:
        mask_uid = self._made.get((uid, key))
        if mask_uid is None:
            return None
        tree = global_variables.global_tree_structure_widget
        controller = global_variables.global_application_controller
        if controller.get_node(mask_uid) is None or mask_uid not in tree.branches_dict:
            self._made.pop((uid, key), None)      # deleted since
            return None
        return mask_uid

    def _make(self, uid: str, node, box) -> Optional[str]:
        """Add a mask branch under *uid* holding its points inside *box*."""
        from core.entities.masks import Masks

        controller = global_variables.global_application_controller
        tree = global_variables.global_tree_structure_widget
        viewer = global_variables.global_pcd_viewer_widget

        points = controller.reconstruct(uid).points
        inside = box.contains(points)
        if not inside.any():
            return None

        params = {"centre": box.centre.round(3).tolist(),
                  "half_extents": box.half_extents.round(3).tolist()}
        mask_uid = controller.add_analysis_result(
            Masks(inside), "masks", [node.uid], node, "isolated_points", params)
        tree.add_branch(mask_uid, uid, BRANCH_NAME, branch_type="masks",
                        tooltip=f"isolated_points,{params}")

        # It only exists to be processed: drawing it would show the same
        # points twice, on top of the source branch.
        tree.visibility_status[mask_uid] = False
        item = tree.branches_dict.get(mask_uid)
        if item is not None:
            tree.blockSignals(True)
            item.setCheckState(0, Qt.Unchecked)
            tree.blockSignals(False)

        # Carry the selection across, so plugins that need one still have it.
        selection = child_selection(viewer.selection_mask_for(uid), inside)
        if selection is not None:
            viewer.set_branch_selection(mask_uid, selection)
        store = viewer._mask_store()
        for row in child_picks(viewer.picked_points, uid, inside):
            store.add_pick(mask_uid, row)

        self._made[(uid, box.key())] = mask_uid
        logger.info(f"Isolated {int(inside.sum()):,} of {len(points):,} points "
                    f"of {uid[:8]} into {mask_uid[:8]}")
        return mask_uid
