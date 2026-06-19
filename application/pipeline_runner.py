"""
Pipeline (macro) replay orchestration.

``PipelineRunner`` replays a saved pipeline by driving the *existing*
single-analysis execution path one step at a time. It owns no threads and no
timer: it calls ``main_window._start_analysis(plugin, params)`` to launch a
step, and is advanced by ``MainWindow._check_analysis_completion`` (the 100 ms
QTimer poll) once that step finishes -- exactly the path a manual run takes, so
every intermediate appears in the tree and reconstruction between steps is
handled automatically by ``AnalysisExecutor``.

Only one analysis runs at a time (the executor enforces this); the runner
respects that by kicking the next step only after the previous one completes.
No custom signals/slots -- direct calls + the shared poll, per project
convention. See ``docs/pipeline_macro_design.md``.
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from config.config import global_variables

logger = logging.getLogger(__name__)


class _SelectionPrompt(QDialog):
    """Non-modal prompt shown while replay pauses for a viewer selection.

    Non-modal so the 3D viewer stays interactive — the user makes their
    selection, then clicks Continue (or Cancel). Built-in QPushButton ``clicked``
    signals are used, which the project permits (only *custom* signals are
    disallowed).
    """

    def __init__(self, parent, message, on_continue, on_cancel):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Paused — Selection Needed")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)

        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel Pipeline")
        continue_btn = QPushButton("Continue")
        continue_btn.setDefault(True)
        cancel_btn.clicked.connect(on_cancel)
        continue_btn.clicked.connect(on_continue)
        row.addWidget(cancel_btn)
        row.addWidget(continue_btn)
        layout.addLayout(row)


class PipelineRunner:
    """Sequences a pipeline's steps over the shared analysis-execution path."""

    def __init__(self, main_window, steps, start_uid, name: str = ""):
        self._main_window = main_window
        self._steps = list(steps)
        self._current_uid = str(start_uid)
        self._name = name or "pipeline"
        self._index = -1
        self._active = False
        self._prompt = None  # live _SelectionPrompt while paused for selection

    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Kick the first step. Subsequent steps are driven by completion polling."""
        if not self._steps:
            return
        self._active = True
        self._index = 0
        self._run_current()

    def on_step_finished(self, ok: bool, new_uid, error: str = None) -> None:
        """Called by MainWindow when the current step's analysis completes.

        Advances to the next step on success, or stops the run on failure /
        cancellation. Re-entrancy-safe: launching the next step just restarts
        the completion poll and returns; it does not recurse synchronously.
        """
        if not self._active:
            return
        if not ok:
            self._stop_with_error(error)
            return
        if new_uid is None:
            self._stop_with_error("step produced no result")
            return

        self._current_uid = str(new_uid)
        self._index += 1
        if self._index < len(self._steps):
            self._run_current()
        else:
            self._finish()

    # --- internals -------------------------------------------------------

    def _run_current(self) -> None:
        step = self._steps[self._index]
        controller = global_variables.global_application_controller
        # Each step runs on the previous step's output (the start branch for
        # step 0). Both run_analysis and action plugins read selected_branches,
        # so pin it to the current branch.
        controller.set_selected_branches([self._current_uid])
        logger.info(
            "Pipeline '%s' step %d/%d: %s",
            self._name, self._index + 1, len(self._steps), step.plugin,
        )
        # Selection is checked FIRST: a plugin can be an action plugin AND need a
        # picked seed (e.g. surface_region_growing), so it must pause before any
        # action/analysis dispatch.
        if self._needs_selection(step.plugin, controller):
            self._pause_for_selection(step)
        elif controller.plugin_manager.get_plugin_type(step.plugin) == "action":
            self._run_action_step(step, controller)
        else:
            # Analysis plugin: async path. dict(...) so the plugin can't mutate
            # the stored params between steps.
            self._main_window._start_analysis(step.plugin, dict(step.params))

    def _needs_selection(self, plugin_name, controller) -> bool:
        plugin_class = controller.plugin_manager.get_plugin(plugin_name)
        if plugin_class is None:
            return False
        try:
            req = getattr(plugin_class(), "requires_selection", None)
            return bool(req()) if callable(req) else False
        except Exception:
            return False

    def _pause_for_selection(self, step) -> None:
        """Make the intermediate visible and show a non-modal prompt; the run
        resumes from _on_selection_continue once the user has selected."""
        self._show_branch(self._current_uid)
        msg = (f"Step {self._index + 1}/{len(self._steps)}: “{step.plugin}” "
               f"needs a selection.\n\n"
               f"Select the points/clusters in the viewer, then click Continue.")
        self._prompt = _SelectionPrompt(
            self._main_window, msg,
            on_continue=self._on_selection_continue,
            on_cancel=self._on_selection_cancel,
        )
        self._prompt.show()
        self._prompt.raise_()

    def _on_selection_continue(self) -> None:
        self._close_prompt()
        step = self._steps[self._index]
        viewer = global_variables.global_pcd_viewer_widget
        has_selection = bool(getattr(viewer, "picked_points_indices", None)) or \
            bool(getattr(viewer, "_selection_polygons", None))
        if not has_selection:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self._main_window, "No Selection",
                "Nothing is selected. Make a selection in the viewer, then Continue.")
            self._pause_for_selection(step)
            return
        controller = global_variables.global_application_controller
        controller.set_selected_branches([self._current_uid])
        if controller.plugin_manager.get_plugin_type(step.plugin) == "action":
            # Producer action plugin that needed a seed (e.g.
            # surface_region_growing): run synchronously now that it's picked.
            self._run_action_step(step, controller)
        else:
            # Analysis plugin: async path. The worker reads the live selection
            # before completion clears it, then the run advances as usual.
            self._main_window._start_analysis(step.plugin, dict(step.params))

    def _on_selection_cancel(self) -> None:
        self._close_prompt()
        self._active = False
        self._main_window._pipeline_runner = None
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self._main_window, "Pipeline Cancelled",
            f"'{self._name}' was cancelled at step "
            f"{self._index + 1}/{len(self._steps)}.")

    def _close_prompt(self) -> None:
        if self._prompt is not None:
            self._prompt.close()
            self._prompt = None

    def _show_branch(self, uid) -> None:
        tree = global_variables.global_tree_structure_widget
        if tree is None:
            return
        key = str(uid)
        tree.blockSignals(True)
        item = tree.branches_dict.get(key)
        if item is not None:
            item.setCheckState(0, Qt.Checked)
        tree.visibility_status[key] = True
        tree.blockSignals(False)
        self._main_window.render_visible_data(zoom_extent=False)

    def _run_action_step(self, step, controller) -> None:
        """Run a producer action plugin (e.g. scale) synchronously.

        Action plugins do GUI work and can't move to the analysis worker thread,
        so we run them inline on the main thread. They return None and create
        their child node internally, so the new branch is recovered by diffing
        the node set, then we advance exactly as an async step would.
        """
        data_nodes = global_variables.global_data_nodes
        before = set(data_nodes.data_nodes.keys())
        plugin_class = controller.plugin_manager.get_plugin(step.plugin)
        if plugin_class is None:
            self._stop_with_error(f"plugin '{step.plugin}' not found")
            return
        try:
            plugin_class().execute(self._main_window, dict(step.params))
        except Exception as e:
            self._stop_with_error(f"{step.plugin}: {e}")
            return

        # An action plugin may emit several nodes (e.g. estimate_normals creates
        # both a normals and an eigenvalues branch). Continue from the new node
        # whose type matches what this step originally produced; fall back to the
        # sole new node when the recorded type doesn't disambiguate.
        new = set(data_nodes.data_nodes.keys()) - before
        chosen = None
        if step.produces:
            matches = [u for u in new
                       if data_nodes.get_node(u).data_type == step.produces]
            if len(matches) == 1:
                chosen = matches[0]
        if chosen is None and len(new) == 1:
            chosen = next(iter(new))

        if chosen is not None:
            self.on_step_finished(True, str(chosen))
        else:
            self._stop_with_error(
                f"action step '{step.plugin}' produced {len(new)} new branch(es); "
                f"couldn't tell which to continue from "
                f"(expected one of type '{step.produces or '?'}')."
            )

    def _finish(self) -> None:
        self._active = False
        self._main_window._pipeline_runner = None
        try:
            self._main_window.render_visible_data(zoom_extent=False)
        except Exception as e:  # rendering is best-effort; don't mask success
            logger.debug("Pipeline post-run render failed: %s", e)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self._main_window, "Pipeline Complete",
            f"Ran {len(self._steps)} step(s) of '{self._name}'.",
        )

    def _stop_with_error(self, error) -> None:
        failed_step = self._index + 1
        total = len(self._steps)
        self._active = False
        self._main_window._pipeline_runner = None
        logger.warning(
            "Pipeline '%s' stopped at step %d/%d: %s",
            self._name, failed_step, total, error,
        )
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(
            self._main_window, "Pipeline Stopped",
            f"'{self._name}' stopped at step {failed_step}/{total}.\n\n"
            f"Details: {error or 'unknown error'}",
        )
