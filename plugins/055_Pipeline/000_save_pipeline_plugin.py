"""
Save Pipeline -- capture the analysis chain that produced the selected branch
as a reusable .json macro that can be replayed on another branch / cloud.

The chain is read straight from branch lineage (each analysis node records its
plugin + params), so there is nothing to "record" ahead of time: process a
cloud however you like, then save any outcome branch as a pipeline.

Steps produced by action plugins (e.g. classify/cut/merge clusters) are not
recorded in lineage and are reported as omitted -- see
``docs/pipeline_macro_design.md``.
"""

import os
import uuid
from typing import Any, Dict

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.services.pipeline import capture_pipeline, save_pipeline


class SavePipelinePlugin(ActionPlugin):

    def get_name(self) -> str:
        return "save_pipeline"

    def get_parameters(self) -> Dict[str, Any]:
        # File chosen via dialog in execute().
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        selected = controller.selected_branches
        if not selected:
            QMessageBox.warning(
                main_window, "No Branch Selected",
                "Select the outcome branch whose steps you want to save.",
            )
            return
        branch_uid = selected[0]

        # Any plugin that creates a lineage node is replayable: analysis plugins
        # plus "producer" action plugins (e.g. scale -> transform_matrix). In-place
        # action plugins (classify/cut/merge) create no node, so never appear in a
        # chain anyway.
        pm = controller.plugin_manager
        supported = set(pm.get_analysis_plugins()) | set(pm.action_plugins)
        pipeline, unsupported = capture_pipeline(branch_uid, data_nodes, supported)

        if not pipeline.steps:
            QMessageBox.warning(
                main_window, "Nothing to Save",
                "This branch has no replayable analysis steps in its history.",
            )
            return

        if unsupported:
            names = ", ".join(dict.fromkeys(unsupported))  # de-dup, keep order
            reply = QMessageBox.question(
                main_window, "Some Steps Can't Be Saved",
                f"These steps aren't replayable and will be omitted:\n\n{names}\n\n"
                f"The saved pipeline ({len(pipeline.steps)} step(s)) may not fully "
                f"reproduce this branch. Save anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        node = data_nodes.get_node(uuid.UUID(str(branch_uid)))
        suggested = (getattr(node, "alias", "") or getattr(node, "params", "")
                     or "pipeline")
        path, _ = QFileDialog.getSaveFileName(
            main_window, "Save Pipeline", f"{suggested}.json",
            "Pipeline (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        pipeline.name = os.path.splitext(os.path.basename(path))[0]
        try:
            save_pipeline(pipeline, path)
        except (OSError, TypeError, ValueError) as e:
            QMessageBox.warning(
                main_window, "Save Failed",
                f"Could not save the pipeline.\n\nDetails: {e}",
            )
            return

        QMessageBox.information(
            main_window, "Pipeline Saved",
            f"Saved {len(pipeline.steps)} step(s) to:\n{path}",
        )
