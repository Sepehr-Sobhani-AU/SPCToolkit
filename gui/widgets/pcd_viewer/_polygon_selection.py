import logging
import threading
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QTimer

from config.config import global_variables
from core.services.screen_selection import point_in_polygon, select_in_polygon

logger = logging.getLogger(__name__)


class PolygonSelectionMixin:
    """Polygon selection and deselection mode for PCDViewerWidget.

    A closed lasso is applied to every visible branch's FULL-RESOLUTION cloud,
    on a background thread, and the result is combined into that branch's
    selection mask. The mask is the truth about what is selected; the highlight
    is derived from it at paint time.

    This replaced storing the polygon and re-testing it inside whichever plugin
    happened to ask. That was correct but lazy in both senses: every plugin
    repeated the projection, and every *deselect* path threw the stored polygons
    away — which silently collapsed the selection to the LOD subset, with
    nothing on screen to show it had happened.
    """

    def _init_polygon_state(self):
        """Initialize polygon selection state."""
        self._polygon_mode = False        # Whether polygon selection mode is active
        self._polygon_vertices = []       # List of (x, y) tuples in Qt widget coordinates

        # uids whose mask is being (re)built on a worker thread. A plugin
        # launched meanwhile has to wait for these — see selection_ready().
        self._selection_building = set()
        # Polls for those builds so the GUI thread, not the worker, repaints.
        self._selection_poll = None

    def enter_polygon_mode(self):
        """Activate polygon mode. Left clicks add vertices; a left double-click
        closes the polygon and selects, a right double-click closes it and
        deselects."""
        if self.points is None:
            return
        if self._zoom_window_mode:
            self.exit_zoom_window_mode()
        self._polygon_mode = True
        self._polygon_vertices = []
        self.setCursor(Qt.CrossCursor)
        self.update()

    def exit_polygon_mode(self):
        """Deactivate polygon selection mode and restore normal cursor."""
        self._polygon_mode = False
        self._polygon_vertices = []
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _close_polygon_and_select(self, at=None):
        """Close the polygon and add its points to the selection."""
        self._close_polygon("or", at)

    def _close_polygon_and_deselect(self, at=None):
        """Close the polygon and remove its points from the selection."""
        self._close_polygon("andnot", at)

    def _close_polygon(self, op, at=None):
        """Close the lasso and apply it to every visible branch as *op*.

        *op* is ``"or"`` to select or ``"andnot"`` to deselect. They are the same
        operation on the mask, which is why select and deselect are symmetric
        here — the old code had them as two methods that drifted, one of which
        discarded the stored polygons and quietly changed what every later
        plugin would see.

        *at* is where the closing double-click landed. Inside the shape it means
        the points the shape encloses; outside it means everything else. So the
        same two buttons give four gestures — select inside, select outside,
        deselect inside, deselect outside — and "keep only this region" stops
        needing a lasso drawn all the way around the rest of the cloud.
        """
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        # Nothing to select against. enter_polygon_mode() refuses to start
        # without points, but a lasso already in progress survives the branches
        # being hidden under it, and there is no cloud left to test by the time
        # it closes.
        if self.points is None:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        # OpenGL's glGetDoublev returns column-major matrices. In numpy (row-major)
        # these appear transposed: mv_np = ModelView^T, proj_np = Projection^T.
        # Use row-vector multiplication: clip_row = point_row @ MV^T @ P^T
        mv = np.array(self.model_view_matrix, dtype=np.float64)   # 4x4 (M^T)
        proj = np.array(self.projection_matrix, dtype=np.float64)  # 4x4 (P^T)
        viewport = tuple(self.viewport)

        # Where the closing click landed decides which side of the shape the
        # gesture means. No click position (a programmatic close) reads as
        # inside, which is the behaviour every caller had before.
        invert = at is not None and not point_in_polygon(
            at.x(), at.y(), polygon)

        uids = list(self._visible_branches)
        self.exit_polygon_mode()
        if not uids:
            return

        self._start_selection_build(uids, polygon, mv, proj, viewport, op, invert)

    # ------------------------------------------------------------------
    # Background mask building
    # ------------------------------------------------------------------

    def _start_selection_build(self, uids, polygon, mv, proj, viewport, op,
                               invert=False):
        """Apply a closed lasso to *uids* on a worker thread.

        Fire-and-forget, in the shape ``_build_coarse_index`` already uses: the
        GUI thread never blocks on it, and the result is published by a single
        dict assignment the next reader picks up. Unlike that one, this has a
        readiness flag, because a plugin launched before the build lands must
        wait rather than see a half-built selection (see ``selection_ready``).

        The full-resolution cloud this needs is already in memory: the render
        path caches every visible branch as it draws it, so there is no
        reconstruction cost here beyond the projection itself.
        """
        for uid in uids:
            self._selection_building.add(uid)

        threading.Thread(
            target=self._build_selection,
            args=(uids, polygon, mv, proj, viewport, op, invert),
            name="selection-mask", daemon=True,
        ).start()

        # Repainting is the GUI thread's job, so the worker cannot ask for it
        # directly — a QTimer polls for the build instead, which is the pattern
        # the project uses for async status everywhere else. Without this the
        # new highlight would not appear until something else happened to
        # trigger a repaint, such as the user nudging the camera.
        if self._selection_poll is None:
            self._selection_poll = QTimer(self)
            self._selection_poll.timeout.connect(self._poll_selection_build)
        self._selection_poll.start(100)

    def _poll_selection_build(self):
        """Repaint once the masks are ready, then stop polling."""
        if not self.selection_ready():
            return
        self._selection_poll.stop()
        self.refresh_selection_readout()
        self.update()

    def _build_selection(self, uids, polygon, mv, proj, viewport, op, invert):
        """Worker: test each branch's full cloud against the lasso."""
        controller = global_variables.global_application_controller
        try:
            for uid in uids:
                try:
                    self._apply_polygon_to_branch(
                        controller, uid, polygon, mv, proj, viewport, op,
                        invert)
                except Exception:
                    logger.error(
                        f"Failed to apply the selection polygon to {uid[:8]}:\n"
                        f"{traceback.format_exc()}")
                finally:
                    self._selection_building.discard(uid)
        finally:
            # A raise anywhere above must not leave the flag set, or every
            # later plugin run would wait forever for a build that is over.
            # The repaint is the poll timer's job — see _start_selection_build.
            for uid in uids:
                self._selection_building.discard(uid)

    def _apply_polygon_to_branch(self, controller, uid, polygon, mv, proj,
                                 viewport, op, invert=False):
        """Combine one branch's lasso result into its selection mask."""
        if controller is None:
            return
        point_cloud = controller.reconstruct(uid)
        pts = point_cloud.points
        n = len(pts)
        if n == 0:
            return

        # Blocked and float32 — see core.services.screen_selection for why the
        # whole-cloud float64 version this replaced needed 19 GB of scratch at
        # 170M points.
        inside = select_in_polygon(pts, polygon, mv, proj, viewport)
        if invert:
            # The click landed outside the shape, so the gesture names
            # everything the shape does NOT enclose.
            inside = ~inside

        if op == "or":
            # The gate belongs on what is being ADDED. Applying it to a deselect
            # would make a locked cluster impossible to remove from a selection
            # it is already in.
            gate = self.selectable_cloud_mask(uid, n)
            if gate is not None:
                inside &= gate

        existing = self._mask_store().selection_mask(uid)
        if existing is None or len(existing) != n:
            existing = np.zeros(n, dtype=bool)

        if op == "or":
            combined = existing | inside
        else:
            combined = existing & ~inside

        self.set_branch_selection(uid, combined)

    def selection_ready(self) -> bool:
        """Whether every pending selection build has finished.

        A plugin must not read the selection while this is False: it would get
        whichever branches happened to be done. ``MainWindow`` polls it before
        launching one.
        """
        return not self._selection_building

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def clear_selection(self):
        """Drop the whole selection — every branch's mask and every click pick.

        Called after a plugin consumes the selection so highlighted points are
        de-highlighted once the operation completes. This is the one place that
        knows what "clearing the selection" means — everywhere else calls it
        rather than clearing the containers by hand, which is how 17 of the 20
        previous call sites came to leave stored state behind.
        """
        store = self._mask_store()
        store.clear_selection_masks()
        store.clear_picks()
        self._selection_rows_cache.clear()
        self.refresh_selection_readout()
        self.update()

    def selection_mask_for_cloud(self, uid, points_3d=None):
        """Branch *uid*'s selection mask, checked against a cloud's length.

        The plugin-facing read. *points_3d* is optional and used only to notice
        that the mask describes a different cloud than the caller holds, in
        which case it is refused rather than returned misaligned.

        Returns None when nothing is selected in that branch — which a plugin
        should report as "no selection", not run on as an empty set.
        """
        mask = self._mask_store().selection_mask(uid)
        if mask is None:
            return None
        if points_3d is not None and len(mask) != len(points_3d):
            logger.warning(
                f"Selection mask for {str(uid)[:8]} covers {len(mask):,} points "
                f"but the caller holds {len(points_3d):,}; ignoring it."
            )
            return None
        return mask
