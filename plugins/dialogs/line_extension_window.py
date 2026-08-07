"""
Guided extension of traced linear features.

Linear region growing rarely reaches the end of every feature: a march gives up
at an occlusion, a density drop, or a spurious bend, and the trace stops short.
This window turns fixing that from "re-pick the seeds and run the plugin again"
into a walk through the stops.

For each stop it shows where and why the trace ended, brings it into view, and
waits. If the feature really does continue, the user picks a few of the points
lying beyond it; growth then re-runs seeded with the line's own points around the
stop PLUS those picks, and the result is spliced into the existing line. If the
stop is a genuine feature end, one click dismisses it for good.

Growth is never loosened to reach further on its own — the picks are the evidence
that the feature continues, and for the same reason the picks are always adopted
whether or not growth could follow them. See
``LinearRegionGrower.extend_from_stop`` and
``plugins/020_Points/020_Clustering/LINEAR_REGION_GROWING.md``.

An extension leaves the user ON the line it just extended, and leaves the camera
where they put it: the end it reached becomes the new marker, in the same place
in the queue, so they can see what their picks did and keep pushing the same
feature out. Only navigation (Previous / Next / Real end, which lands on a stop
they have not seen) moves the view. Every change is snapshotted for Undo —
growth is a judgement made from a picture on screen, and the user has to be able
to look at a result and say "no, not that".

Stops are ranked by how many unclaimed points lie ahead of them, so the ones
worth looking at come first and genuine ends sink to the bottom of the queue.

While a stop is under review the points worth picking at it are promoted to a
cluster of their own — bright, and the only points in the cloud that will accept
a click, since everything else is left as noise and the viewer refuses to select
noise. That is also what makes the workflow possible at all: unclaimed points
are noise, so before this nothing could be picked on a result branch. See
``_mark_candidates``.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QMessageBox, QProgressBar, QSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer

from config.config import global_variables
from application.selection_gate import picked_cloud_indices
from core.services.linear_region_grower import (
    STOP_REASONS,
    cylinders_to_vector_feature,
    centerlines_to_vector_feature,
    lines_to_traces,
    stop_key,
)


# Colour of the marker drawn at the stop currently under review. Deliberately
# not one of the STOP_REASONS colours — this says "you are here", not "this is
# why it stopped", and must read as distinct from the per-reason stop branches.
_FOCUS_COLOR = np.array([0.1, 1.0, 0.4], dtype=np.float32)   # green

# How much of the scene to frame around the stop under review, as a multiple of
# the search reach. Wide enough that the continuation (if any) is on screen and
# pickable without panning; tight enough to see individual points.
_FOCUS_EXTENT_FACTOR = 4.0

# Poll interval for the viewer's pick count (ms). Matches AnnotationWindow.
_POLL_MS = 200

# Colour given to the candidate cluster — the points offered for picking at the
# current stop. Bright and unlike anything set_random_color produces for a real
# cluster, so "these are the ones you may click" reads at a glance.
_CANDIDATE_COLOR = np.array([1.0, 1.0, 0.0], dtype=np.float32)   # yellow

# What a point goes back to when it stops being a candidate: the noise colour
# Clusters.set_random_color paints label -1 with.
_NOISE_COLOR = np.array([0.2, 0.2, 0.2], dtype=np.float32)

# Default reach of the candidate cone, as a multiple of the search reach. Set
# well beyond what growth itself would attempt: the whole point of picking is to
# reach across a hole the march could not.
_CANDIDATE_REACH_FACTOR = 8.0


class LineExtensionWindow(QDialog):
    """Step through the stops of traced lines and extend the ones that continue.

    Modeless on purpose: the viewer must stay live so the user can orbit and
    pick while the window is open.
    """

    def __init__(self, result_uid, pc_points, lines, grower, params,
                 resolved=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extend Traced Lines")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.viewer = global_variables.global_pcd_viewer_widget
        self.controller = global_variables.global_application_controller
        self.tree_widget = global_variables.global_tree_structure_widget

        self.result_uid = result_uid
        self.pc_points = pc_points
        self.lines = list(lines)
        self.grower = grower
        self.params = dict(params)

        # Stops the user has dismissed as genuine feature ends. Keyed by value
        # (see stop_key) rather than identity, so the set survives both the line
        # objects being rebuilt by an extension and a save/load round trip —
        # a reopened session must not walk the user back through ends they have
        # already settled.
        self.resolved = set() if resolved is None else set(resolved)

        # [(label, MarchStop), ...] ranked once on open, most-promising first.
        self.queue = []
        self.total_stops = 0
        self.pos = 0
        self.extensions_made = 0

        # Snapshots taken before each change, newest last — what Undo walks back
        # through. Growth is a judgement call made from a picture on screen, so
        # the user has to be able to look at a result and say "no, not that".
        # Cheap to keep: GrownLine is a namedtuple of arrays already built, so a
        # snapshot copies references, not point data.
        self.history = []

        # Branch showing which stop is under review. One branch, re-pointed at
        # each step — everything drawn is a tree-controllable branch, never an
        # ad-hoc viewer overlay.
        self.focus_uid = None

        # {stop reason: branch uid} for the per-reason stop markers the growth
        # run drew, discovered once at open. Empty when the user did not ask for
        # them — in which case the window never creates any, since which debug
        # geometry is on screen is their choice, not ours.
        self.stop_branches = self._discover_stop_branches()
        self.track_stop_branches = bool(self.stop_branches)

        # Cloud indices currently promoted to the candidate cluster, so they can
        # be put back to noise when the review moves on. None = nothing marked.
        self.marked_indices = None

        # Whether the lines arrived carrying their search cylinders. A result
        # saved before cylinders were persisted comes back without them, and
        # rebuilding the branch from an extension alone would throw away the
        # whole original run's geometry — better a stale branch than a gutted one.
        self.had_cylinders = any(len(line.cylinders) for line in self.lines)

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._refresh_pick_count)

        self._claim_tree_selection()
        self._setup_ui()
        self._build_queue()
        self._show_current(focus=True)
        self.poll_timer.start(_POLL_MS)

    def _claim_tree_selection(self):
        """Make the result branch the selected one.

        Picking is filtered to the branches selected in the tree
        (``_is_index_in_selected_branch``). After growth the selection is still
        the input cloud — which is also hidden by then — so nothing at all is
        pickable until this is put right, whatever the labels say.
        """
        if self.controller is not None:
            self.controller.set_selected_branches([self.result_uid])
        if self.tree_widget is None:
            return
        item = self.tree_widget.branches_dict.get(self.result_uid)
        if item is None:
            return
        self.tree_widget.blockSignals(True)
        self.tree_widget.clearSelection()
        item.setSelected(True)
        self.tree_widget.blockSignals(False)

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel(
            "Each stop below is where a traced line gave up.\n"
            "If the feature continues, pick a few points just past the marker\n"
            "(Shift+Click, or P for polygon select) and press Extend.\n"
            "Extending leaves the camera where you put it, so you can look at\n"
            "the result, pick further and extend again — or undo it."
        ))

        box = QGroupBox("Current stop")
        box_layout = QVBoxLayout()
        self.stop_label = QLabel("—")
        self.stop_label.setWordWrap(True)
        self.ahead_label = QLabel("")
        self.picks_label = QLabel("0 points picked")
        box_layout.addWidget(self.stop_label)
        box_layout.addWidget(self.ahead_label)
        box_layout.addWidget(self.picks_label)
        box.setLayout(box_layout)
        layout.addWidget(box)

        rollback = QHBoxLayout()
        rollback.addWidget(QLabel("Discard last"))
        self.rollback_spin = QSpinBox()
        self.rollback_spin.setRange(0, 20)
        self.rollback_spin.setValue(0)
        self.rollback_spin.setToolTip(
            "Throw away this many search cylinders from the end before growing "
            "again.\n\nA march often stops because the last step or two went "
            "wrong — the window caught a neighbouring object and the heading "
            "drifted off the feature. Growing from that tip inherits the bad "
            "direction. Rolling back starts the re-grow from before the damage."
        )
        rollback.addWidget(self.rollback_spin)
        rollback.addWidget(QLabel("cylinder(s) before growing"))
        rollback.addStretch()
        layout.addLayout(rollback)

        candidates = QHBoxLayout()
        self.candidate_box = QCheckBox("Offer only points ahead, within")
        self.candidate_box.setChecked(True)
        self.candidate_box.setToolTip(
            "Points you can pick are put in their own bright cluster; the rest "
            "of the cloud stays grey and cannot be clicked at all.\n\nUntick to "
            "offer every unclaimed point, wherever it is."
        )
        self.candidate_box.stateChanged.connect(self._on_candidate_setting)
        self.range_spin = QSpinBox()
        self.range_spin.setRange(1, 1000)
        self.range_spin.setValue(max(1, int(round(
            self.grower.cylinder_length * self.grower.reach_factor
            * _CANDIDATE_REACH_FACTOR))))
        self.range_spin.setToolTip(
            "How far ahead of the stop to offer points, in metres.\n\nRaise it "
            "to pick across a long occlusion — anything beyond this distance "
            "stays grey and unclickable."
        )
        self.range_spin.valueChanged.connect(self._on_candidate_setting)
        candidates.addWidget(self.candidate_box)
        candidates.addWidget(self.range_spin)
        candidates.addWidget(QLabel("m"))
        candidates.addStretch()
        layout.addLayout(candidates)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.extend_btn = QPushButton("Extend from picks")
        self.extend_btn.clicked.connect(self._extend)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip(
            "Put the line back the way it was before the last extension."
        )
        self.undo_btn.clicked.connect(self._undo)
        self.end_btn = QPushButton("Real end")
        self.end_btn.setToolTip(
            "The feature genuinely ends here — do not ask about this stop again."
        )
        self.end_btn.clicked.connect(self._mark_real_end)
        for btn in (self.extend_btn, self.undo_btn, self.end_btn):
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        # Navigation is kept apart from the buttons that CHANGE something:
        # Previous / Next only move the review along, leaving the stop
        # undecided, and unlike the others they do bring the new stop into view.
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.setToolTip("Go back to the stop before this one.")
        self.prev_btn.clicked.connect(self._previous)
        self.next_btn = QPushButton("Next >")
        self.next_btn.setToolTip("Leave this stop undecided and move on.")
        self.next_btn.clicked.connect(self._next)
        self.recentre_btn = QPushButton("Re-centre view")
        self.recentre_btn.clicked.connect(self._focus_current)
        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.close)
        for btn in (self.prev_btn, self.next_btn, self.recentre_btn,
                    self.finish_btn):
            nav.addWidget(btn)
        layout.addLayout(nav)

        self.setLayout(layout)

    # ------------------------------------------------------------------ #
    # Queue                                                              #
    # ------------------------------------------------------------------ #

    def _claimed_mask(self):
        """Points belonging to any traced line — what "unclaimed ahead" excludes."""
        mask = np.zeros(len(self.pc_points), dtype=bool)
        for line in self.lines:
            mask[line.indices] = True
        return mask

    def _build_queue(self):
        """Rank every unresolved stop once, most points-ahead first.

        Ranked ONCE, on open. Re-ranking after each extension would reshuffle the
        list under the user and throw away their place in the walk — they would
        be bounced back to the top every time they fixed something. The order is
        a starting suggestion, not a live scoreboard; the points-ahead figure
        shown for the current stop is recomputed live instead.
        """
        claimed = self._claimed_mask()
        scored = [
            (label, stop, int(self.grower.unclaimed_ahead(stop, claimed).size))
            for label, line in enumerate(self.lines)
            for stop in line.stops
            if stop_key(label, stop) not in self.resolved
        ]
        scored.sort(key=lambda row: row[2], reverse=True)
        self.queue = [(label, stop) for label, stop, _n in scored]
        self.total_stops = len(self.queue)
        self.pos = 0

    def _settle_current(self):
        """Drop the stop just decided from the queue.

        *pos* is left alone, so removing the current entry naturally lands on the
        next one — the user carries on where they were. Only a decision that
        FINISHES with a line settles it; an extension replaces the stop in place
        instead (see ``_extend``), keeping the user on the line they are working.
        """
        if 0 <= self.pos < len(self.queue):
            self.queue.pop(self.pos)

    def _current(self):
        if 0 <= self.pos < len(self.queue):
            return self.queue[self.pos]
        return None

    # ------------------------------------------------------------------ #
    # Display                                                            #
    # ------------------------------------------------------------------ #

    def _show_current(self, focus=False):
        """Refresh the panel for the stop at *pos*.

        *focus* moves the camera onto it, and is for NAVIGATION only — landing
        on a stop the user has not seen yet. Changing something (extend, undo)
        must leave the view alone: the user framed that view themselves to pick
        into, and yanking the camera the moment they press a button hides the
        very thing they pressed it to see.
        """
        current = self._current()
        # Measures stops SETTLED against the total ever queued, so it advances
        # as decisions are made instead of jumping about as the queue shrinks.
        self.progress.setMaximum(max(1, self.total_stops))
        self.progress.setValue(self.total_stops - len(self.queue))
        # Stays live even at the end of the queue: the last thing the user did
        # may be the thing they want back.
        self.undo_btn.setEnabled(bool(self.history))
        self.prev_btn.setEnabled(self.pos > 0)
        self.next_btn.setEnabled(self.pos + 1 < len(self.queue))

        widgets = (self.extend_btn, self.end_btn,
                   self.recentre_btn, self.rollback_spin)
        if current is None:
            done = "No stops left to review." if not self.queue else \
                ("End of the list — the stops still on it were stepped past, "
                 "not settled. Use Previous to go back to them.")
            self.stop_label.setText(
                f"{done}<br>{self.extensions_made} extension(s) made this session."
            )
            self.ahead_label.setText("")
            for widget in widgets:
                widget.setEnabled(False)
            self._clear_focus_branch()
            self._refresh_candidates()   # nothing under review, nothing offered
            return

        for widget in widgets:
            widget.setEnabled(True)

        label, stop = current
        # Recomputed live rather than read off the queue: earlier extensions have
        # claimed points since the ranking was built.
        n_ahead = int(self.grower.unclaimed_ahead(stop, self._claimed_mask()).size)
        reason_text = STOP_REASONS.get(stop.reason, (stop.reason, None))[0]
        self.stop_label.setText(
            f"Stop {self.pos + 1} of {len(self.queue)} — <b>{reason_text}</b><br>"
            f"Line {label + 1}, at "
            f"({stop.tip[0]:.2f}, {stop.tip[1]:.2f}, {stop.tip[2]:.2f})"
        )
        if n_ahead:
            self.ahead_label.setText(
                f"{n_ahead} unclaimed point(s) lie ahead — the feature may continue."
            )
        else:
            self.ahead_label.setText(
                "Nothing unclaimed ahead — most likely a genuine feature end."
            )
        self._draw_focus_branch(stop)
        self._refresh_candidates()
        if focus:
            self._focus_current()
        self._refresh_pick_count()

    def _focus_current(self):
        current = self._current()
        if current is None or self.viewer is None:
            return
        _label, stop = current
        extent = (self.grower.cylinder_length * self.grower.reach_factor
                  * _FOCUS_EXTENT_FACTOR)
        self.viewer.focus_on(stop.tip, extent)

    def _refresh_pick_count(self):
        count = 0 if self.viewer is None else len(self.viewer.picked_points_indices)
        self.picks_label.setText(f"{count} point(s) picked")

    # ------------------------------------------------------------------ #
    # The "you are here" branch                                          #
    # ------------------------------------------------------------------ #

    def _stop_cylinder(self, stop):
        """A stop drawn as a cylinder sitting at its tip along the heading the
        march was on, so the user can see which way the feature was going.

        Slightly wider than the search tube when that is very thin, or the
        marker is invisible at the zoom you review a whole line at.
        """
        length = self.grower.cylinder_length
        radius = max(self.grower.cylinder_radius, length * 0.1)
        return (np.asarray(stop.tip, dtype=float),
                np.asarray(stop.direction, dtype=float), radius, length)

    def _focus_feature(self, stop):
        """Wireframe marking the stop currently under review."""
        return cylinders_to_vector_feature(
            [self._stop_cylinder(stop)],
            color=_FOCUS_COLOR, symbol_type="Stop under review",
        )

    def _draw_focus_branch(self, stop):
        feature = self._focus_feature(stop)
        if feature is None:
            return
        feature.cluster_reference = self.result_uid

        if self.focus_uid is None:
            result_node = self.controller.get_node(self.result_uid)
            if result_node is None:
                return
            self.focus_uid = self.controller.add_analysis_result(
                feature, "vector_feature", [result_node.uid], result_node,
                "stop_under_review", self.params,
            )
            self.tree_widget.blockSignals(True)
            self.tree_widget.add_branch(
                self.focus_uid, self.result_uid, "stop_under_review",
                branch_type="vector_feature",
                tooltip="Marks the line end currently under review — "
                        "removed when the Extend window closes.",
            )
            item = self.tree_widget.branches_dict.get(self.focus_uid)
            if item:
                item.setCheckState(0, Qt.Checked)
            self.tree_widget.visibility_status[self.focus_uid] = True
            self.tree_widget.blockSignals(False)
        else:
            # Re-point the existing branch instead of churning the tree once per
            # step: same branch, new geometry.
            node = self.controller.get_node(self.focus_uid)
            if node is not None:
                node.data = feature
            self.controller.cache_service.invalidate(self.focus_uid)

        self._render()

    def _clear_focus_branch(self):
        if self.focus_uid is None:
            return
        uid, self.focus_uid = self.focus_uid, None
        self.tree_widget.blockSignals(True)
        self._remove_branch(uid)
        self.tree_widget.blockSignals(False)
        self._render()

    # ------------------------------------------------------------------ #
    # The per-reason stop markers                                        #
    # ------------------------------------------------------------------ #

    def _discover_stop_branches(self):
        """The ``stop_<reason>`` branches the growth run drew, as {reason: uid}.

        Matched against the known reasons rather than by prefix, so the window's
        own ``stop_under_review`` branch is never mistaken for one of them.
        """
        result_node = self.controller.get_node(self.result_uid)
        if result_node is None:
            return {}
        by_name = {f"stop_{reason}": reason for reason in STOP_REASONS}
        return {
            by_name[child.params]: str(child_uid)
            for child_uid, child in self.controller.data_nodes.data_nodes.items()
            if child.parent_uid == result_node.uid and child.params in by_name
        }

    def _update_stop_branches(self):
        """Repaint the per-reason stop markers from the lines' CURRENT stops.

        Left to itself this geometry is a record of the original run, so after
        an extension it marks ends that no longer exist and misses the ones that
        do — the line visibly runs past a red "too few points" marker. The stops
        are the machine-readable truth (``end_cylinders`` is display leftovers
        that only accumulate), so the markers are rebuilt from those: a branch
        per reason, created when a reason first appears and removed when its last
        stop is gone.

        Only touched when the growth run drew these branches in the first place.
        """
        if not self.track_stop_branches:
            return
        result_node = self.controller.get_node(self.result_uid)
        if result_node is None:
            return

        by_reason = {}
        for line in self.lines:
            for stop in line.stops:
                by_reason.setdefault(stop.reason, []).append(
                    self._stop_cylinder(stop))

        self.tree_widget.blockSignals(True)
        for reason, cylinders in by_reason.items():
            label, color = STOP_REASONS.get(
                reason, (reason, np.array([1.0, 1.0, 1.0], dtype=np.float32)))
            feature = cylinders_to_vector_feature(
                cylinders, color=color, symbol_type=f"Stop: {label}")
            if feature is None:
                continue
            feature.cluster_reference = self.result_uid
            uid = self.stop_branches.get(reason)
            node = None if uid is None else self.controller.get_node(uid)
            if node is None:
                self.stop_branches[reason] = self._add_stop_branch(
                    result_node, reason, feature)
            else:
                node.data = feature
                self.controller.cache_service.invalidate(uid)

        for reason in [r for r in self.stop_branches if r not in by_reason]:
            self._remove_branch(self.stop_branches.pop(reason))
        self.tree_widget.blockSignals(False)

    def _add_stop_branch(self, result_node, reason, feature):
        name = f"stop_{reason}"
        uid = self.controller.add_analysis_result(
            feature, "vector_feature", [result_node.uid], result_node,
            name, self.params,
        )
        self.tree_widget.add_branch(
            uid, self.result_uid, name, branch_type="vector_feature",
            tooltip=f"linear_region_growing,{self.params}",
        )
        item = self.tree_widget.branches_dict.get(uid)
        if item:
            item.setCheckState(0, Qt.Checked)
        self.tree_widget.visibility_status[uid] = True
        return uid

    def _remove_branch(self, uid):
        """Drop a branch this window owns. Callers block tree signals around it —
        blockSignals does not nest, so unblocking here would silently re-enable
        them partway through a batch."""
        self.tree_widget.remove_branch(uid)
        self.controller.remove_node(uid)

    def _render(self):
        main_window = global_variables.global_main_window
        if main_window is not None:
            main_window.render_visible_data(zoom_extent=False)

    # ------------------------------------------------------------------ #
    # Actions                                                            #
    # ------------------------------------------------------------------ #

    def _picked_indices(self):
        """Viewer picks mapped onto reconstructed-cloud indices — the same
        mapping the growth plugin uses for its seeds, so a pick here means
        exactly what a seed pick means there."""
        if self.viewer is None:
            return np.empty(0, dtype=np.intp)
        picked = picked_cloud_indices(self.viewer, self.pc_points,
                                      self.grower.kdtree)
        return np.empty(0, dtype=np.intp) if picked is None else picked

    def _extend(self):
        current = self._current()
        if current is None:
            return
        label, stop = current
        settled_stop = stop        # what the queue held, before any rollback

        picks = self._picked_indices()
        if picks.size == 0:
            QMessageBox.information(
                self, "No Points Picked",
                "Pick the points this line should grow into first — Shift+Click, "
                "or press P in the viewer for polygon select.\n\n"
                "If the feature genuinely ends here, press 'Real end' instead."
            )
            return

        line = self.lines[label]
        before = len(line.indices)

        # Optionally throw away the last few steps first. A march often stops
        # because those steps went wrong — the window caught something else and
        # the heading drifted — so re-seeding from that tip would inherit the bad
        # direction. Rolling back gives the re-seed a clean body to start from.
        n_back = self.rollback_spin.value()
        if n_back:
            rolled = self.grower.rollback_stop(line, stop, n_back)
            if rolled is None:
                QMessageBox.warning(
                    self, "Cannot Roll Back That Far",
                    f"Discarding {n_back} cylinder(s) would consume this whole "
                    f"line, leaving nothing to grow from.\n\nUse a smaller number."
                )
                return
            line, stop = rolled

        result = self.grower.extend_from_stop(stop, line, picks)
        if result is None:
            QMessageBox.warning(
                self, "Nothing Added",
                "None of the picked points lie ahead of this stop, so there is "
                "nothing for the line to grow into.\n\n"
                "Pick points on the side of the marker the line was heading "
                "towards. If the feature genuinely ends here, press 'Real end'."
            )
            return

        self._snapshot()
        self.lines[label] = result.line
        self.extensions_made += 1
        # Dismiss the stop as it was recorded, so a reopened session does not
        # walk the user back through an end they have already dealt with.
        self.resolved.add(stop_key(label, settled_stop))

        # Stay on this line. The frontier the extension left behind REPLACES the
        # stop in place rather than being queued behind everything else, so the
        # user sees what their picks did and can keep pushing the same line out —
        # pick, extend, look, pick again — instead of being thrown to an
        # unrelated stop the moment they press the button.
        self.queue[self.pos] = (label, result.stop)
        self._clear_picks()
        self._commit()
        self._show_current()

        # Net of any rollback, so the figure matches what is now on screen.
        gained = len(result.line.indices) - before
        rolled_note = f", after discarding {n_back} cylinder(s)" if n_back else ""
        if result.marched:
            note = (f"{gained:+d} points, {len(result.line.indices)} on the line"
                    f"{rolled_note}")
        else:
            note = (f"growth could not carry on, so your {picks.size} picked "
                    f"point(s) were added to the line{rolled_note} — "
                    f"pick further ahead and extend again")
        self.stop_label.setText(f"{self.stop_label.text()}<br><i>Last: {note}</i>")

    def _undo(self):
        """Put everything back as it was before the last change."""
        if not self.history:
            QMessageBox.information(
                self, "Nothing to Undo",
                "No changes have been made since this window opened."
            )
            return
        (self.lines, self.queue, self.pos, self.resolved,
         self.extensions_made) = self.history.pop()
        self._clear_picks()
        self._commit()
        self._show_current()
        self.stop_label.setText(f"{self.stop_label.text()}<br><i>Last change "
                                f"undone.</i>")

    def _snapshot(self):
        """Record the current state so ``_undo`` can restore it."""
        self.history.append((list(self.lines), list(self.queue), self.pos,
                             set(self.resolved), self.extensions_made))

    def _mark_real_end(self):
        current = self._current()
        if current is None:
            return
        label, stop = current
        self._snapshot()
        self.resolved.add(stop_key(label, stop))
        self._settle_current()
        self._commit()
        # Settling drops this entry, so *pos* now holds a different stop: this
        # is navigation as much as a decision, and the user has not seen it yet.
        self._show_current(focus=True)

    def _next(self):
        """Leave this stop undecided and move on. Not a change — nothing to
        snapshot, and Previous brings it back."""
        self.pos = min(self.pos + 1, len(self.queue))
        self._show_current(focus=True)

    def _previous(self):
        self.pos = max(self.pos - 1, 0)
        self._show_current(focus=True)

    def _clear_picks(self):
        if self.viewer is not None:
            self.viewer.picked_points_indices.clear()
            self.viewer.update()

    # ------------------------------------------------------------------ #
    # Writing back                                                       #
    # ------------------------------------------------------------------ #

    def _commit(self):
        """Write the current line set back onto the result branch.

        Called after every decision rather than only on Finish, so closing the
        window by any route (or the app dying) never loses accepted work.

        Labels are rebuilt from the lines every time, so this always leaves the
        branch holding ONLY its real clusters — the candidate cluster is erased
        by construction and never has to be unpicked. Whoever needs it back puts
        it back (``_refresh_candidates``); on close nobody does.
        """
        node = self.controller.get_node(self.result_uid)
        if node is None or node.data is None:
            return
        clusters = node.data

        labels = np.full(len(self.pc_points), -1, dtype=np.int32)
        for label, line in enumerate(self.lines):
            labels[line.indices] = label
        clusters.labels = labels
        clusters.line_traces = lines_to_traces(self.lines, self.params,
                                               resolved=self.resolved)
        clusters.set_random_color()
        self.marked_indices = None      # the relabel above wiped any candidates

        self.controller.cache_service.invalidate(self.result_uid)
        self.controller.cache_service.invalidate_descendants(self.result_uid)
        self._update_debug_branches()
        self._update_stop_branches()
        self._render()

    # ------------------------------------------------------------------ #
    # The candidate cluster                                              #
    # ------------------------------------------------------------------ #

    def _candidate_indices(self):
        """Cloud indices to offer for picking at the current stop.

        The cone ahead of it (``pick_candidates``), or every unclaimed point
        when the user has turned the restriction off.
        """
        current = self._current()
        if current is None:
            return np.empty(0, dtype=np.intp)
        claimed = self._claimed_mask()
        if not self.candidate_box.isChecked():
            return np.where(~claimed)[0].astype(np.intp)
        _label, stop = current
        return self.grower.pick_candidates(stop, claimed,
                                           float(self.range_spin.value()))

    def _refresh_candidates(self):
        """Re-offer for the stop now under review, without rewriting the branch.

        Stepping between stops must not pay for a full relabel of the cloud, so
        this only touches the points that were offered and the ones now being
        offered — everything else is left exactly as ``_commit`` wrote it. Only
        safe while the LINES are unchanged, which is why every path that changes
        them goes through ``_commit`` instead.
        """
        node = self.controller.get_node(self.result_uid)
        if node is None or node.data is None:
            return
        clusters = node.data
        self._unmark_candidates(clusters)
        self._mark_candidates(clusters)
        self.controller.cache_service.invalidate(self.result_uid)
        self.controller.cache_service.invalidate_descendants(self.result_uid)
        self._render()

    def _unmark_candidates(self, clusters):
        """Put the points offered last time back to being noise."""
        if self.marked_indices is None or self.marked_indices.size == 0:
            self.marked_indices = None
            return
        idx = self.marked_indices
        idx = idx[idx < len(clusters.labels)]
        clusters.labels[idx] = -1
        if clusters.colors is not None:
            clusters.colors[idx] = _NOISE_COLOR
        self.marked_indices = None

    def _mark_candidates(self, clusters):
        """Promote the candidate points to a cluster of their own.

        This is the whole of the "fade the rest and lock it" behaviour, and it
        needs no viewer support because a label already carries both halves of
        it: a ``-1`` point is drawn dark grey (``set_random_color``'s noise
        colour) AND refused by the picking filters (``_filter_noise_points``).
        Unclaimed points are ``-1``, which is why they cannot be picked today —
        so the fix is not to fade anything, it is to stop the points the user
        needs from being noise.

        The new label is one above the highest in use. That matters: colours are
        handed out in sorted label order, so a label that sorts LAST leaves every
        existing cluster's colour untouched, while one sorting first (-2, say)
        would shift every line's colour each time candidates came and went.

        The colour is written straight onto the colour array rather than through
        ``custom_colors``, so nothing about this survives the next commit.
        """
        indices = self._candidate_indices()
        if indices.size == 0:
            self.marked_indices = None
            return
        label = int(clusters.labels.max()) + 1
        clusters.labels[indices] = label
        if clusters.colors is not None:
            clusters.colors[indices] = _CANDIDATE_COLOR
        self.marked_indices = indices

    def _on_candidate_setting(self, _value=None):
        """Re-offer under the new setting. Not a change to the lines, so it is
        not snapshotted and does not touch the queue."""
        self._refresh_candidates()

    def _rebuilt_debug_feature(self, name):
        """Wireframe for the debug branch *name*, rebuilt from the current lines.

        The names match the ones the growth plugin gives these branches, so an
        extension refreshes the same geometry the run drew rather than adding a
        second, competing branch.
        """
        if name == "centerlines":
            return centerlines_to_vector_feature(
                [line.centerline for line in self.lines]
            )
        if name == "cylinders" and self.had_cylinders:
            return cylinders_to_vector_feature(
                [c for line in self.lines for c in line.cylinders]
            )
        return None

    def _update_debug_branches(self):
        """Keep the growth plugin's debug wireframes in step with the extended
        lines — the centerlines and the search cylinders.

        Only branches that are already there are updated: whether this geometry
        is drawn at all stays the user's choice from the growth dialog. Without
        this the cylinders stop at the original stop while the line runs on past
        it, which reads as the extension not having happened.

        The per-reason stop markers are handled separately, in
        ``_update_stop_branches`` — they come and go with the reasons that exist.
        """
        result_node = self.controller.get_node(self.result_uid)
        if result_node is None:
            return
        for child_uid, child in self.controller.data_nodes.data_nodes.items():
            if child.parent_uid != result_node.uid:
                continue
            feature = self._rebuilt_debug_feature(child.params)
            if feature is None:
                continue
            feature.cluster_reference = self.result_uid
            child.data = feature
            self.controller.cache_service.invalidate(str(child_uid))

    def closeEvent(self, event):
        self.poll_timer.stop()
        self._clear_focus_branch()
        # Hands the branch back with only its real clusters on it — the candidate
        # cluster exists to make picking possible, and has no business surviving
        # into classification or a saved project.
        self._commit()
        self._clear_picks()
        super().closeEvent(event)
