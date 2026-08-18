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


def _line_name(label):
    """The cluster name a traced line carries. Matches what the growth plugin
    writes, so editing a line set renames nothing that was not already named
    this way."""
    return f"Line {label + 1}"


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

# Name for the candidate cluster. A linear-region-growing result names its
# clusters ("Line 1", "Line 2"), and for a NAMED Clusters the renderer colours
# by name (ClustersTransformer -> get_named_colors) and ignores the per-point
# colour array completely — so on those branches the candidates must be named
# too or they come out the unnamed default grey, however yellow the array says
# they are. Removed again the moment they stop being candidates.
_CANDIDATE_NAME = "Pick candidates"

# What a point goes back to when it stops being a candidate: the noise colour
# Clusters.set_random_color paints label -1 with.
_NOISE_COLOR = np.array([0.2, 0.2, 0.2], dtype=np.float32)

# Width the centerlines are drawn at while this window is open. At the default
# 1.0 a centerline is a one-pixel hair, and picking reads the depth buffer at
# exactly the pixel clicked — so aiming at it is a game of pixel-hunting, and a
# miss reads back empty and picks nothing at all. Restored on close.
_EDIT_LINE_WIDTH = 4.0

# How near a click has to land to count as naming a line, when it did not land
# on the line's own points. Expressed in the growth's own terms — one search
# reach — so it scales with the feature being traced instead of being a guess in
# metres. See _line_under.
_CLICK_REACH_FACTOR = 1.0

# How many picks the edit buttons look through, newest gestures aside. Bounded
# because a polygon selection leaves millions of picks and this runs on the
# 5 Hz poll: resolving every one of them to a line would freeze the window. Two
# distinct lines is all Join needs, and it stops as soon as it has them.
_EDIT_PICK_SCAN = 64


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

        # Cloud indices currently promoted to the candidate cluster, and the
        # label they were given, so both can be undone when the review moves on.
        # None = nothing marked.
        self.marked_indices = None
        self.marked_label = None

        # Point branches switched off so they stop drawing a second copy of the
        # cloud over this one. Switched back on when the window closes.
        self._hidden_branches = []
        self._saved_line_width = None

        # Last answer to "which lines are picked", against the pick list it was
        # computed from. The poll asks 5 times a second and the answer only
        # changes when the user clicks.
        self._picked_lines_cache = (None, [])

        # Whether the lines arrived carrying their search cylinders. A result
        # saved before cylinders were persisted comes back without them, and
        # rebuilding the branch from an extension alone would throw away the
        # whole original run's geometry — better a stale branch than a gutted one.
        self.had_cylinders = any(len(line.cylinders) for line in self.lines)

        # What the last Trim/Delete/Join did, shown in the edit panel. Kept off
        # stop_label so it is not wiped by the next step of the review.
        self.edit_note = ""

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._refresh_pick_count)

        self._claim_the_viewport()
        self._setup_ui()
        self._build_queue()
        self._show_current(focus=True)
        self.poll_timer.start(_POLL_MS)

    # ------------------------------------------------------------------ #
    # Owning what is on screen while the window is open                  #
    # ------------------------------------------------------------------ #

    def _claim_the_viewport(self):
        """Leave the result branch as the only point cloud on screen.

        Any other visible point branch — above all the cloud this result was
        grown from — draws a SECOND copy of the same points, in the same place.
        Those copies carry no cluster labels, so they are picked in preference to
        the real thing and they accept every click: the candidate cluster is
        painted correctly underneath while the user clicks grey duplicates on top
        of it and wonders why nothing is highlighted. Measured on the real
        controller: with both branches visible and the input cloud selected,
        every one of its points is selectable and none of the result's are.

        Hiding them is not a display preference, then — it is what makes the
        picking rules mean anything. Restored on close.
        """
        self._hidden_branches = []
        if self.tree_widget is None or self.viewer is None:
            return

        # Fatten the wireframes. A centerline drawn one pixel wide is aimed at
        # by pixel-hunting, and picking resolves the depth buffer at exactly the
        # pixel clicked — miss the hair and it reads back empty, so the click
        # does nothing at all rather than nearly working.
        self._saved_line_width = self.viewer.line_width
        self.viewer.line_width = max(self._saved_line_width, _EDIT_LINE_WIDTH)

        # Branches currently drawing POINTS. Vector features (centerlines,
        # cylinders, the stop markers) draw as lines, never appear here, and are
        # left alone — they are the context worth keeping.
        drawing_points = [uid for uid in self.viewer._branch_offsets
                          if uid != self.result_uid]

        self.tree_widget.blockSignals(True)
        for uid in drawing_points:
            item = self.tree_widget.branches_dict.get(uid)
            if item is not None:
                item.setCheckState(0, Qt.Unchecked)
            self.tree_widget.visibility_status[uid] = False
            self._hidden_branches.append(uid)

        # And make sure the result itself IS on screen — reopening a saved
        # result gives no guarantee of that, and hiding everything else would
        # otherwise leave an empty viewport.
        result_item = self.tree_widget.branches_dict.get(self.result_uid)
        if result_item is not None:
            result_item.setCheckState(0, Qt.Checked)
            self.tree_widget.clearSelection()
            result_item.setSelected(True)
        self.tree_widget.visibility_status[self.result_uid] = True
        self.tree_widget.blockSignals(False)

        # Picking is also filtered to the tree SELECTION
        # (``_is_index_in_selected_branch``), and after growth that is still the
        # input cloud, so nothing would be pickable whatever the labels say.
        if self.controller is not None:
            self.controller.set_selected_branches([self.result_uid])

    def _restore_hidden_branches(self):
        """Show again whatever was hidden to clear the viewport, and put the
        wireframe width back the way the user had it."""
        if self._saved_line_width is not None and self.viewer is not None:
            self.viewer.line_width = self._saved_line_width
            self._saved_line_width = None
        if not self._hidden_branches or self.tree_widget is None:
            return
        self.tree_widget.blockSignals(True)
        for uid in self._hidden_branches:
            item = self.tree_widget.branches_dict.get(uid)
            if item is not None:
                item.setCheckState(0, Qt.Checked)
            self.tree_widget.visibility_status[uid] = True
        self.tree_widget.blockSignals(False)
        self._hidden_branches = []

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel(
            "Each stop below is where a traced line gave up.\n"
            "Points you can pick are bright yellow and drawn larger; unclaimed\n"
            "points are grey, see-through, and ignore clicks. Pick a few just\n"
            "past the marker\n"
            "(Shift+Click, or P for polygon select) and press Extend.\n"
            "Extending leaves the camera where you put it, so you can look at\n"
            "the result, pick further and extend again — or undo it.\n"
            "Other point clouds are hidden while this is open, and shown again\n"
            "on close — a second copy on screen would take your clicks."
        ))

        box = QGroupBox("Current stop")
        box_layout = QVBoxLayout()
        self.stop_label = QLabel("—")
        self.stop_label.setWordWrap(True)
        self.ahead_label = QLabel("")
        self.ahead_label.setWordWrap(True)
        self.picks_label = QLabel("0 points picked")
        self.picks_label.setWordWrap(True)
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
        self.candidate_box = QCheckBox("Offer only the points ahead")
        self.candidate_box.setChecked(True)
        self.candidate_box.setToolTip(
            "The points counted above are put in their own bright cluster; the "
            "rest of the cloud stays grey and cannot be clicked at all.\n\n"
            "Untick to offer every unclaimed point in the cloud instead — for "
            "the rare continuation that lies outside the corridor entirely."
        )
        self.candidate_box.stateChanged.connect(self._on_candidate_setting)
        candidates.addWidget(self.candidate_box)
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

        # Editing the lines themselves, as opposed to growing them. All three
        # act on what is picked in the viewer, exactly like Extend — click ON a
        # line (its own points take the click, and so does its centerline) and
        # press the button.
        edit = QGroupBox("Edit lines")
        edit_layout = QVBoxLayout()
        self.edit_label = QLabel(
            "Click a traced line — on the centreline or near its points — then:"
        )
        self.edit_label.setWordWrap(True)
        edit_layout.addWidget(self.edit_label)

        edit_buttons = QHBoxLayout()
        self.trim_btn = QPushButton("Trim")
        self.trim_btn.setToolTip(
            "Remove the clicked centreline segment and release its points.\n\n"
            "A cut in the middle leaves TWO lines — a traced line carries one "
            "polyline, so a gap cannot be part of it."
        )
        self.trim_btn.clicked.connect(self._trim)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setToolTip(
            "Remove the whole clicked line, its cylinders and its markers. "
            "Its points go back to unclaimed."
        )
        self.delete_btn.clicked.connect(self._delete_line)
        self.join_btn = QPushButton("Join")
        self.join_btn.setToolTip(
            "Join two clicked lines into one, keeping the FIRST one clicked. "
            "Click a point on each, in that order."
        )
        self.join_btn.clicked.connect(self._join)
        for btn in (self.trim_btn, self.delete_btn, self.join_btn):
            edit_buttons.addWidget(btn)
        edit_layout.addLayout(edit_buttons)
        self.edit_status = QLabel("")
        self.edit_status.setWordWrap(True)
        edit_layout.addWidget(self.edit_status)
        edit.setLayout(edit_layout)
        layout.addWidget(edit)

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
        # This count IS the yellow: _candidate_indices offers the same set, so
        # the number and the points on screen can never disagree.
        if n_ahead:
            self.ahead_label.setText(
                f"{n_ahead} unclaimed point(s) lie ahead — the feature may "
                f"continue. Those are the yellow ones."
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
        text = f"{count} point(s) picked"
        self._refresh_edit_target()

        # Say so out loud if another point cloud comes back on screen. It draws a
        # second copy of these same points, unlabelled, which takes the clicks —
        # the failure is completely silent otherwise: the candidates look right
        # and the clicks do nothing.
        if self.viewer is not None:
            others = [uid for uid in self.viewer._branch_offsets
                      if uid != self.result_uid]
            if others:
                text += ("<br><b>Another point cloud is visible.</b> Its points "
                         "sit on top of these and will take your clicks — hide "
                         "it in the tree.")
        self.picks_label.setText(text)

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
        """Viewer picks mapped onto reconstructed-cloud indices, kept to the
        points actually on offer.

        The mapping is the one the growth plugin uses for its seeds, so a pick
        here means what a seed pick means there — but it cannot be trusted on its
        own. ``picked_cloud_indices`` re-tests a polygon selection against the
        FULL cloud (so a polygon covers everything it encloses, not only the
        points LOD happened to draw), and that re-test does not go through the
        viewer's selection filters. A polygon drawn over the corridor therefore
        comes back holding every point inside it — measured on real data: the
        viewer honestly reported 32 points picked while this handed back
        thousands, and since picks are always adopted, a whole bush joined the
        line.

        Handing the offer to the mapping as *allowed* settles it. The candidate
        cluster is the statement of what may be picked at this stop; anything
        else in the polygon was never on offer, whichever code path found it.
        """
        if self.viewer is None or self.marked_indices is None:
            return np.empty(0, dtype=np.intp)
        picked = picked_cloud_indices(self.viewer, self.pc_points,
                                      self.grower.kdtree,
                                      allowed=self.marked_indices)
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
                "Only the yellow points count. A polygon may cover a good deal "
                "more than those; everything else inside it is ignored.\n\n"
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
            note = (f"{gained:+d} points from {picks.size} pick(s), "
                    f"{len(result.line.indices)} on the line{rolled_note}")
        else:
            note = (f"growth could not carry on, so your {picks.size} picked "
                    f"point(s) were added to the line{rolled_note} — "
                    f"pick further ahead and extend again")
        self.stop_label.setText(f"{self.stop_label.text()}<br><i>Last: {note}</i>")

    # ------------------------------------------------------------------ #
    # Editing the lines: trim, delete, join                              #
    # ------------------------------------------------------------------ #

    def _refresh_edit_target(self):
        """Say which line the edit buttons would act on, live.

        Without this the buttons are a leap of faith: the panel counts points
        picked but never says which LINE that named, so Trim and Delete are
        pressed blind and a Join silently keeps whichever line the click order
        happened to put first. The names shown here are the ones the buttons
        will use.
        """
        labels = []
        for label, _point in self._picked_lines_cached():
            if label not in labels:
                labels.append(label)

        if not labels:
            self.edit_label.setText(
                "Click a traced line — on the centreline or near its points — "
                "then:")
        elif len(labels) == 1:
            self.edit_label.setText(
                f"<b>{_line_name(labels[0])}</b> is picked. Trim cuts the "
                f"segment you clicked; Delete removes the line. Join needs a "
                f"second line.")
        else:
            self.edit_label.setText(
                f"<b>{_line_name(labels[0])}</b>, then "
                f"<b>{_line_name(labels[1])}</b> — Join keeps "
                f"{_line_name(labels[0])}. Trim and Delete act on "
                f"{_line_name(labels[0])}.")

    def _line_labels(self):
        """Per-point line label from the LINES, not from the branch.

        The branch's own labels cannot answer this while a stop is under review:
        the candidate points have been promoted out of noise into a cluster of
        their own, so the branch would report them as belonging to a line.
        """
        labels = np.full(len(self.pc_points), -1, dtype=np.int32)
        for label, line in enumerate(self.lines):
            labels[line.indices] = label
        return labels

    def _line_under(self, point):
        """The label of the traced line nearest *point*, or None if none is near.

        "Near" is one search reach of the growth that produced these lines, so
        it scales with the feature rather than being a guess in metres. This is
        what lets a click land BESIDE a line and still name it — on the grey
        points around a cable, on a yellow candidate next to it, anywhere within
        reach — instead of having to hit the centreline itself.
        """
        reach = (self.grower.cylinder_length * self.grower.reach_factor
                 * _CLICK_REACH_FACTOR)
        best, best_dist = None, reach
        for label, line in enumerate(self.lines):
            _seg, dist = self.grower.nearest_segment(line, point)
            if dist < best_dist:
                best, best_dist = label, dist
        return best

    def _picked_lines_in_order(self, stop_after=2):
        """``[(label, point), ...]`` naming the lines the user clicked, in the
        order they clicked them, stopping once *stop_after* distinct lines are
        named.

        Click order is the whole reason this does not go through
        ``picked_cloud_indices``: that returns a sorted set, and Join has to know
        which line was clicked FIRST.

        Each pick names a line two ways, in order of certainty. If the picked
        point IS on a traced line, that line wins outright — nothing is more
        direct than clicking the thing itself. Otherwise the nearest centreline
        within reach is taken (``_line_under``), which is what makes clicking
        *near* a line work: the depth buffer only resolves geometry that was
        actually drawn at that pixel, so a click three pixels off a centreline
        lands on whatever is behind it, and before this that was the end of it.

        Bounded at ``_EDIT_PICK_SCAN`` picks: a polygon selection leaves
        millions, and this is called from the 5 Hz poll to keep the readout live.
        """
        if self.viewer is None or self.viewer.points is None:
            return []

        rows = np.fromiter(self.viewer.picked_points_indices, dtype=np.int64,
                           count=len(self.viewer.picked_points_indices))
        rows = rows[(rows >= 0) & (rows < len(self.viewer.points))]
        if rows.size == 0:
            return []
        rows = rows[:_EDIT_PICK_SCAN]

        # One batch query rather than one per pick.
        coords = np.asarray(self.viewer.points[rows, :3], dtype=np.float32)
        _dist, cloud_rows = self.grower.kdtree.query(coords)
        labels = self._line_labels()

        picked, seen = [], set()
        for xyz, row in zip(coords.astype(float), np.atleast_1d(cloud_rows)):
            row = int(row)
            label = int(labels[row]) if 0 <= row < len(labels) else -1
            if label < 0:
                label = self._line_under(xyz)
            if label is None:
                continue
            picked.append((int(label), xyz))
            seen.add(int(label))
            if stop_after is not None and len(seen) >= stop_after:
                break
        return picked

    def _picked_lines_cached(self):
        """``_picked_lines_in_order`` for the readout, recomputed only when the
        picks change. The poll asks five times a second; the answer does not."""
        picks = () if self.viewer is None else tuple(
            self.viewer.picked_points_indices[:_EDIT_PICK_SCAN])
        fingerprint = (len(getattr(self.viewer, "picked_points_indices", ())),
                       picks, len(self.lines))
        if self._picked_lines_cache[0] != fingerprint:
            self._picked_lines_cache = (fingerprint, self._picked_lines_in_order())
        return self._picked_lines_cache[1]

    def _no_line_picked(self, what):
        reach = self.grower.cylinder_length * self.grower.reach_factor
        many = len(self.viewer.picked_points_indices) > _EDIT_PICK_SCAN \
            if self.viewer is not None else False
        extra = (f"\n\nThere are a lot of points picked at the moment, and only "
                 f"the first {_EDIT_PICK_SCAN} are looked through. Clear the "
                 f"selection (Shift+Right-click, or press C) and click just the "
                 f"line." if many else "")
        QMessageBox.information(
            self, "No Line Picked",
            f"Shift+Click the line you want to {what} first — on its centreline, "
            f"on its points, or anywhere within {reach:.1f} m of it.\n\n"
            f"Nothing picked landed that near a traced line. The panel names the "
            f"line as soon as one is picked, so you can check before pressing."
            + extra
        )

    def _restructure(self, new_lines, note):
        """Adopt an edited line list.

        Editing changes how many lines there are, and a line's label is just its
        position in this list — so every label above the edit shifts. Two things
        are keyed by label and have to be rebuilt rather than carried over: the
        review queue (rebuilt, which restarts the walk) and the resolved set
        (re-keyed by stop VALUE, so ends the user already settled stay settled).
        """
        self._snapshot()
        self.lines = new_lines
        self.resolved = self._remapped_resolved(new_lines)
        self._commit()
        self._build_queue()
        self._clear_picks()
        self._show_current()
        self.edit_status.setText(f"<i>{note}</i>")

    def _remapped_resolved(self, new_lines):
        """Carry the resolved ends across a change of labels.

        Matched on the stop's own values (reason and tip) rather than its label,
        the same way ``stop_key`` defines identity in the first place. A stop the
        user called a real end is still a real end after the line it sits on has
        been renumbered by an edit elsewhere.
        """
        settled = {key[1:] for key in self.resolved}
        return {stop_key(label, stop)
                for label, line in enumerate(new_lines)
                for stop in line.stops
                if stop_key(label, stop)[1:] in settled}

    def _trim(self):
        picked = self._picked_lines_in_order()
        if not picked:
            self._no_line_picked("trim")
            return
        label, point = picked[0]
        line = self.lines[label]

        segment = self.grower.segment_at(line, point)
        if segment is None:
            QMessageBox.warning(
                self, "Nothing to Trim",
                f"{_line_name(label)} has no centreline to cut, so there is no "
                f"segment to remove. Use Delete to drop the whole line."
            )
            return

        pieces, released = self.grower.trim_segment(line, segment)
        new_lines = self.lines[:label] + pieces + self.lines[label + 1:]
        split = " — the cut was mid-line, so it is now two lines" \
            if len(pieces) > 1 else ""
        self._restructure(
            new_lines,
            f"Trimmed segment {segment + 1} of {_line_name(label)}: "
            f"{released.size} point(s) released{split}."
        )

    def _delete_line(self):
        picked = self._picked_lines_in_order()
        if not picked:
            self._no_line_picked("delete")
            return
        label, _point = picked[0]
        released = len(self.lines[label].indices)
        new_lines = self.lines[:label] + self.lines[label + 1:]
        self._restructure(
            new_lines,
            f"Deleted {_line_name(label)}: {released} point(s) released."
        )

    def _join(self):
        picked = self._picked_lines_in_order()
        labels = []
        for label, _point in picked:
            if label not in labels:
                labels.append(label)
        if len(labels) < 2:
            QMessageBox.information(
                self, "Pick Two Lines",
                "Join needs a click on each of the two lines — the first one "
                "clicked is the one that is kept.\n\n"
                + ("Both clicks landed on the same line."
                   if len(labels) == 1 else
                   "Shift+Click a point on each line, then press Join.")
            )
            return

        keep, absorb = labels[0], labels[1]
        joined = self.grower.join_lines(self.lines[keep], self.lines[absorb])
        new_lines = [joined if i == keep else line
                     for i, line in enumerate(self.lines) if i != absorb]
        self._restructure(
            new_lines,
            f"Joined {_line_name(absorb)} into {_line_name(keep)}: "
            f"{len(joined.indices)} point(s) on the line."
        )

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
        self.edit_status.setText("")
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
            self.viewer.clear_selection()

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
        # Before set_random_color, which syncs cluster_colors from the names and
        # would otherwise carry a phantom candidate cluster forward.
        self._forget_candidate_naming(clusters)
        self.marked_indices = None      # the relabel above wiped any candidates

        # Names are rebuilt alongside the labels because editing changes how
        # MANY lines there are: a trim splits one into two, a delete and a join
        # each remove one. Left alone, the names written when the branch was
        # grown would keep pointing at labels that have shifted underneath them
        # — "Line 4" naming what is now line 2. Extension never hits this (the
        # count cannot change), so this rewrites nothing on that path.
        #
        # AFTER forgetting the candidate naming, not before: the candidate label
        # is one above the lines, so a trim that adds a line makes the two
        # collide, and forgetting the candidate would then delete the name of a
        # real line that had just been given the same number.
        if clusters.cluster_names:
            clusters.cluster_names = {k: _line_name(k)
                                      for k in range(len(self.lines))}
        clusters.set_random_color()

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

        Exactly the points the panel counts as lying ahead
        (``unclaimed_ahead``), or every unclaimed point when the user has turned
        the restriction off.

        The same set the count reports, deliberately: one number on screen, one
        set of yellow points, and no way for them to disagree. An earlier version
        offered a much wider cone, which lit up a whole tree canopy beside a
        count of 77 and read as a bug.
        """
        current = self._current()
        if current is None:
            return np.empty(0, dtype=np.intp)
        claimed = self._claimed_mask()
        if not self.candidate_box.isChecked():
            return np.where(~claimed)[0].astype(np.intp)
        _label, stop = current
        return self.grower.unclaimed_ahead(stop, claimed)

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
        self._apply_emphasis()
        self._render()

    def _unmark_candidates(self, clusters):
        """Put the points offered last time back to being noise."""
        self._forget_candidate_naming(clusters)
        if self.marked_indices is None or self.marked_indices.size == 0:
            self.marked_indices = None
            return
        idx = self.marked_indices
        idx = idx[idx < len(clusters.labels)]
        clusters.labels[idx] = -1
        if clusters.colors is not None:
            clusters.colors[idx] = _NOISE_COLOR
        self.marked_indices = None

    def _forget_candidate_naming(self, clusters):
        """Drop the candidate cluster's name and colour entries.

        Kept apart from unmarking the points because ``_commit`` wipes the labels
        wholesale without going through it, and a name left behind would show up
        as a phantom cluster — one that classification and any DXF export would
        take perfectly seriously.
        """
        if self.marked_label is not None:
            clusters.cluster_names.pop(self.marked_label, None)
        clusters.cluster_colors.pop(_CANDIDATE_NAME, None)
        self.marked_label = None

    def _mark_candidates(self, clusters):
        """Promote the candidate points to a cluster of their own.

        This is the whole of the "fade the rest and lock it" behaviour, and it
        needs no viewer support because a label already carries both halves of
        it: an unnamed/noise point is drawn grey AND refused by the picking
        filters (``_filter_locked_and_noise``). Unclaimed points are ``-1``, which is
        why they cannot be picked today — so the fix is not to fade anything, it
        is to stop the points the user needs from being noise.

        The new label is one above the highest in use. That matters: colours are
        handed out in sorted label order, so a label that sorts LAST leaves every
        existing cluster's colour untouched, while one sorting first (-2, say)
        would shift every line's colour each time candidates came and went.

        Colouring has to follow whichever scheme the branch uses, and a growth
        result uses the named one ("Line 1", "Line 2"): for a named Clusters the
        renderer calls ``get_named_colors`` and never looks at the per-point
        colour array, painting every unnamed label the 0.7 grey default. Writing
        yellow into ``colors`` alone is therefore invisible on exactly the
        branches this window is opened on — the candidates have to be named.
        """
        indices = self._candidate_indices()
        if indices.size == 0:
            self.marked_indices = None
            return
        label = int(clusters.labels.max()) + 1
        clusters.labels[indices] = label
        self.marked_indices = indices
        self.marked_label = label

        if clusters.has_names():
            clusters.cluster_names[label] = _CANDIDATE_NAME
            clusters.cluster_colors[_CANDIDATE_NAME] = _CANDIDATE_COLOR
        elif clusters.colors is not None:
            clusters.colors[indices] = _CANDIDATE_COLOR

    def _apply_emphasis(self):
        """Draw the offered points big and the unclaimed ones see-through.

        Colour alone was not enough on real data: at the zoom a whole line is
        reviewed at, one yellow point is the same handful of pixels as the grey
        one beside it, and a cable seen through a canopy is mostly canopy. Size
        separates them at any zoom, and the fade stops the clutter in FRONT of
        the feature from hiding it — faded points write no depth, so a candidate
        behind a screen of unclaimed returns still shows through.

        The line points are left alone deliberately: they are what the user
        clicks for Trim, Delete and Join, so fading them would make the editing
        controls harder to aim than the picking they sit beside.
        """
        if self.viewer is None:
            return
        marked = (np.empty(0, dtype=np.intp) if self.marked_indices is None
                  else self.marked_indices)
        unclaimed = np.where(~self._claimed_mask())[0]
        self.viewer.set_point_emphasis(self.result_uid, emphasised=marked,
                                       faded=np.setdiff1d(unclaimed, marked))

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
        if self.viewer is not None:
            # One size, fully opaque again — the emphasis belongs to reviewing,
            # not to the branch.
            self.viewer.set_point_emphasis(self.result_uid)
        self._clear_focus_branch()
        self._restore_hidden_branches()
        # Hands the branch back with only its real clusters on it — the candidate
        # cluster exists to make picking possible, and has no business surviving
        # into classification or a saved project.
        self._commit()
        self._clear_picks()
        super().closeEvent(event)
