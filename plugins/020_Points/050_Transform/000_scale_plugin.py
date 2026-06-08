"""
Scale Plugin (Points > Transform > Scale)

Scales the selected branch's point cloud along one or more axes about the cloud
centroid. The scaling is stored as a derived ``transform_matrix`` DataNode added
as a CHILD of the selected branch -- it is applied lazily during reconstruction
(see core/transformers/transform_matrix_transformer.py), so NO duplicate point
cloud is created. Toggling the child off restores the original geometry.

Scaling about the centroid keeps the cloud in place; negative factors mirror it
across that centroid. A single 4x4 homogeneous matrix encodes the scaling via
``TransformMatrix.from_scale`` (linear algebra: T(c) @ S @ T(-c)).
"""

import logging
from typing import Dict, Any

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.transform_matrix import TransformMatrix

logger = logging.getLogger(__name__)

# Which coordinate axes each direction scales (0=x, 1=y, 2=z).
_DIRECTION_AXES = {
    "z": (2,),
    "x": (0,),
    "y": (1,),
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
    "xyz": (0, 1, 2),
}


class ScalePlugin(ActionPlugin):
    """Scale a branch along selected axes via a derived transform_matrix node."""

    def get_name(self) -> str:
        return "scale"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "direction": {
                "type": "choice",
                "options": ["z", "x", "y", "xy", "xz", "yz", "xyz"],
                "default": "z",
                "label": "Scale Direction",
                "description": "Axis or axes to scale along (about the cloud centroid)."
            },
            "scale": {
                "type": "float",
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "decimals": 3,
                "label": "Scale Factor",
                "description": "Multiplier along the selected axes. "
                               "1 keeps the size unchanged; negatives mirror the cloud."
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        logger.info("ScalePlugin.execute() called")

        controller = global_variables.global_application_controller
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validation: exactly one branch ---
        selected_branches = controller.selected_branches
        if len(selected_branches) != 1:
            QMessageBox.warning(
                main_window,
                "Select One Branch",
                "Please select exactly one branch to scale."
            )
            return

        # --- Validate direction and build per-axis factors ---
        direction = params.get("direction", "z")
        axes = _DIRECTION_AXES.get(direction)
        if axes is None:
            QMessageBox.warning(
                main_window,
                "Invalid Direction",
                f"Unknown scale direction '{direction}'. "
                f"Choose one of: {', '.join(_DIRECTION_AXES)}."
            )
            return

        scale = float(params.get("scale", 1.0))
        factors = np.ones(3, dtype=np.float64)
        for axis in axes:
            factors[axis] = scale

        source_uid = selected_branches[0]

        # --- Processing UI ---
        main_window.tree_overlay.position_over(tree_widget)
        main_window.tree_overlay.show_processing("Scaling branch...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            parent_node = controller.get_node(source_uid)
            if parent_node is None:
                raise ValueError(f"Selected branch {source_uid} not found")

            # Pivot = centroid. Reconstruct once only to read the points for the
            # centroid; the reconstructed cloud is discarded (no node stored).
            pc = controller.reconstruct(source_uid)
            center = np.asarray(pc.points, dtype=np.float64).mean(axis=0)

            # Build the 4x4 scale transform about the centroid (linear algebra).
            transform = TransformMatrix.from_scale(
                float(factors[0]), float(factors[1]), float(factors[2]),
                center=center
            )

            # Add as a derived CHILD node -- applied lazily during reconstruction.
            new_uid = controller.add_analysis_result(
                transform, "transform_matrix",
                [parent_node.uid], parent_node, "scale", params
            )

            new_name = f"scale {direction} x{scale:g}"

            # Hide the parent BEFORE add_branch so the single render triggered by
            # add_branch shows only the scaled result.
            tree_widget.blockSignals(True)
            parent_item = tree_widget.branches_dict.get(source_uid)
            if parent_item:
                parent_item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[source_uid] = False
            tree_widget.blockSignals(False)

            tree_widget.add_branch(
                new_uid, source_uid, new_name,
                tooltip=f"scale,{params}"
            )
            new_node = controller.get_node(new_uid)
            if new_node is not None:
                tree_widget.update_cache_tooltip(new_uid, new_node.memory_size)

            logger.info(f"Scale transform added as child {new_uid} of {source_uid}")

        except Exception as e:
            logger.error(f"Scale failed: {e}", exc_info=True)
            QMessageBox.critical(
                main_window,
                "Scale Error",
                f"Failed to scale branch:\n{str(e)}"
            )
        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
