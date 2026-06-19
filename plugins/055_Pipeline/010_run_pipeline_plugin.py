"""
Run Pipeline -- replay a saved .json macro on the selected branch.

Steps run one after another over the existing analysis path: the pipeline is
launched here, then advanced by MainWindow's completion poll (see
``application/pipeline_runner.py``). v1 replays linear analysis chains; steps
needing live selection are out of scope until Phase 2.
"""

from typing import Any, Dict

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.services.pipeline import load_pipeline
from application.pipeline_runner import PipelineRunner


class RunPipelinePlugin(ActionPlugin):

    def get_name(self) -> str:
        return "run_pipeline"

    def get_parameters(self) -> Dict[str, Any]:
        # File chosen via dialog in execute().
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller

        if controller.is_analysis_running():
            QMessageBox.warning(
                main_window, "Busy",
                "An analysis is already running. Wait for it to finish.",
            )
            return

        selected = controller.selected_branches
        if not selected:
            QMessageBox.warning(
                main_window, "No Branch Selected",
                "Select the branch to run the pipeline on.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            main_window, "Run Pipeline", "", "Pipeline (*.json)",
        )
        if not path:
            return

        try:
            pipeline = load_pipeline(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(
                main_window, "Load Failed",
                f"Could not read the pipeline file.\n\nDetails: {e}",
            )
            return

        if not pipeline.steps:
            QMessageBox.warning(
                main_window, "Empty Pipeline",
                "This pipeline has no steps.",
            )
            return

        # Every step must map to a currently-loaded plugin (analysis or a
        # producer action plugin such as scale).
        pm = controller.plugin_manager
        available = set(pm.get_analysis_plugins()) | set(pm.action_plugins)
        missing = [s.plugin for s in pipeline.steps if s.plugin not in available]
        if missing:
            names = ", ".join(dict.fromkeys(missing))
            QMessageBox.warning(
                main_window, "Unknown Steps",
                f"This pipeline uses plugins that aren't available:\n\n{names}",
            )
            return

        # Pre-flight: bind every cross-branch reference to a live branch now, so
        # the run is fully resolved before it starts (the file stores no UUIDs).
        if not self._bind_references(main_window, pipeline):
            return

        runner = PipelineRunner(
            main_window, pipeline.steps, start_uid=selected[0], name=pipeline.name,
        )
        main_window._pipeline_runner = runner
        runner.start()

    def _bind_references(self, main_window, pipeline) -> bool:
        """Resolve each branch reference (stored only as a hint) to a live branch.

        Returns True to proceed (including the no-references case), False to
        abort (user cancelled, or nothing to bind to).
        """
        needed = [(i, step, pname, hint)
                  for i, step in enumerate(pipeline.steps)
                  for pname, hint in step.bindings.items()]
        if not needed:
            return True

        data_nodes = global_variables.global_data_nodes
        branch_options = {
            str(uid): (node.alias or node.params or str(uid))
            for uid, node in data_nodes.data_nodes.items()
        }
        if not branch_options:
            QMessageBox.warning(
                main_window, "Can't Run Pipeline",
                "This pipeline references other branches, but the project has "
                "none to bind them to.",
            )
            return False

        default_uid = next(iter(branch_options))
        schema, keys = {}, []
        for j, (i, step, pname, hint) in enumerate(needed):
            key = f"bind_{j}"
            keys.append(key)
            schema[key] = {
                "type": "dropdown",
                "options": branch_options,
                "default": default_uid,
                "label": f"{step.plugin} — {hint}",
                "description": f"Step {i + 1}: choose the branch to use as '{hint}'.",
            }

        from gui.dialog_boxes.dynamic_dialog import DynamicDialog
        dialog = DynamicDialog("Bind Pipeline References", schema, main_window)
        if not dialog.exec_():
            return False

        picks = dialog.get_parameters()
        for j, (i, step, pname, hint) in enumerate(needed):
            step.params[pname] = picks[keys[j]]
        return True
