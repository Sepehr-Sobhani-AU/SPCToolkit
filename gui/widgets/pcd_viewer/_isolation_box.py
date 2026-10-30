import logging

import numpy as np
from OpenGL.GL import (
    glClipPlane, glEnable, glDisable, GL_CLIP_PLANE0,
    glPushAttrib, glPopAttrib, GL_ALL_ATTRIB_BITS,
    GL_DEPTH_TEST, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glColor4f, glBegin, glEnd, glVertex3f, GL_LINES, GL_POINTS,
    glLineWidth, glPointSize,
)

from config.config import global_variables
from core.services.isolation_box import IsolationBox, CENTRE_HANDLE, FACE_HANDLES

logger = logging.getLogger(__name__)


class IsolationBoxMixin:
    """Isolation box and slab for PCDViewerWidget.

    Hides everything outside a box, on every visible branch at once, through
    six OpenGL clipping planes. Nothing is copied or cut: the box is view
    state, like the camera, so it needs no branch and changes no data.

    I toggles isolation. The box first appears around the centre of rotation
    and is then independent of it: seven grip handles (one per face, one in the
    middle) resize and move it. S turns the box into a thin slab facing the
    camera and back; [ and ] step the slab by half its thickness; { and }
    change its thickness.

    Everything that reads points directly rather than through the screen —
    picking, the lasso, the zoom window — asks ``active_clip_box`` and skips
    what it hides. Under level of detail the box is filled at full resolution
    (see RenderingCoordinator.prepare_branches, ``focus_box``).
    """

    def _init_clip_state(self):
        """Initialise isolation state."""
        self._clip_box = None          # IsolationBox, kept between isolations
        self._isolated = False
        self._slab_mode = False
        self._box_before_slab = None   # restored when the slab key is pressed again
        # (handle index, one_sided, last QPoint) while a handle is dragged
        self._handle_drag = None
        # Key of the box the last refill was for, or None. Lets a click on a
        # handle that moved nothing skip the refill.
        self._lod_filled_box_key = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_isolated(self) -> bool:
        return self._isolated

    def active_clip_box(self):
        """The box currently hiding points, or None when nothing is hidden."""
        return self._clip_box if self._isolated else None

    def lod_focus_box(self):
        """The box level of detail should fill at full resolution, or None.

        Handle drags do not re-render on each move — the refill reads every
        point of every visible branch — so it happens on release (see
        ``end_handle_drag``).
        """
        return self.active_clip_box()

    def toggle_isolation(self):
        """Hide everything outside the box, or show everything again."""
        if not self._isolated:
            if self._clip_box is None:
                self._clip_box = IsolationBox.cube(self.center, self.isolation_box_size)
            self._isolated = True
        else:
            self._isolated = False
            self._handle_drag = None
        self.update()
        self._refill_level_of_detail()

    def toggle_slab(self):
        """Squash the box into a slab facing the camera, or give the box back."""
        if self._slab_mode:
            self._clip_box = self._box_before_slab
            self._box_before_slab = None
            self._slab_mode = False
        else:
            if self._clip_box is None:
                self._clip_box = IsolationBox.cube(self.center, self.isolation_box_size)
            self._box_before_slab = self._clip_box.copy()
            self._clip_box = self._clip_box.slab(self._camera_axes(),
                                                 self.slab_thickness)
            self._slab_mode = True
        self._isolated = True
        self.update()
        self._refill_level_of_detail()

    def step_slab(self, direction: int):
        """Move the slab half its thickness away from (+1) or towards (-1) you.

        Half, so neighbouring steps overlap and no point is skipped.
        """
        if not (self._isolated and self._slab_mode):
            return
        self._clip_box.step(2, direction * self._clip_box.half_extents[2])
        self.update()
        self._refill_level_of_detail()

    def scale_slab_thickness(self, factor: float):
        """Make the slab thicker (factor > 1) or thinner, keeping its centre."""
        if not (self._isolated and self._slab_mode):
            return
        self._clip_box.half_extents[2] = max(
            self._clip_box.half_extents[2] * factor, 0.005)
        self.slab_thickness = float(self._clip_box.half_extents[2] * 2.0)
        self._show_message(f"Slab thickness {self.slab_thickness:.3g} m")
        self.update()
        self._refill_level_of_detail()

    def forget_box_if_outside(self, point):
        """A new centre of rotation outside the box resets it to the default.

        Called after a double-click. Inside the box, the box stays as it is.
        """
        if self._isolated or self._clip_box is None:
            return
        if not self._clip_box.contains_point(point):
            self._clip_box = None
            self._box_before_slab = None
            self._slab_mode = False

    def _camera_axes(self) -> np.ndarray:
        """(3, 3) rows: the camera's right, up and viewing direction, in world
        space."""
        # model_view_matrix holds OpenGL's column-major matrix, so its
        # transpose is the world-to-eye rotation and its rows are the eye axes.
        rot = np.array(self.model_view_matrix, dtype=np.float64)[:3, :3].T
        right, up, back = rot[0], rot[1], rot[2]
        axes = np.stack([right, up, -back])
        return axes / np.linalg.norm(axes, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _enable_clip_planes(self):
        """Clip every following draw to the box. Call with the scene's
        model-view matrix loaded: planes are given in world space."""
        box = self.active_clip_box()
        if box is None:
            return
        for i, eq in enumerate(box.planes()):
            glClipPlane(GL_CLIP_PLANE0 + i, eq)
            glEnable(GL_CLIP_PLANE0 + i)

    def _disable_clip_planes(self):
        if self.active_clip_box() is None:
            return
        for i in range(6):
            glDisable(GL_CLIP_PLANE0 + i)

    def render_clip_box_overlay(self):
        """Draw the box outline and its grip handles on top of the scene.

        Viewer furniture like the axis symbol, not data, so not a branch.
        """
        box = self.active_clip_box()
        if box is None:
            return

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLineWidth(1.0)
        glColor4f(1.0, 1.0, 1.0, 0.35)
        glBegin(GL_LINES)
        for start, end in box.edges():
            glVertex3f(*start)
            glVertex3f(*end)
        glEnd()

        handles = box.handle_positions()
        glPointSize(self._HANDLE_SIZE_PX)
        glBegin(GL_POINTS)
        for k in self._live_handles():
            if k == CENTRE_HANDLE:
                glColor4f(0.2, 0.9, 1.0, 0.9)
            else:
                glColor4f(1.0, 0.85, 0.1, 0.9)
            glVertex3f(*handles[k])
        glEnd()

        glPopAttrib()

    def _live_handles(self):
        """Handles that can be grabbed. A slab's front and back faces always
        sit on top of its centre on screen, so they are left out; { and }
        change the thickness instead."""
        if self._slab_mode:
            return [0, 1, 2, 3, CENTRE_HANDLE]
        return list(range(len(FACE_HANDLES))) + [CENTRE_HANDLE]

    # ------------------------------------------------------------------
    # Handle dragging
    # ------------------------------------------------------------------

    def _screen_of(self, pts):
        """(N, 2) widget coordinates of world points, and a validity mask."""
        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)
        sx, sy, valid = self._project_points_to_screen(
            np.asarray(pts, dtype=np.float64).reshape(-1, 3), mv, proj,
            self.viewport)
        return np.stack([sx, sy], axis=1), valid

    def handle_at(self, mouse_pos):
        """Index of the grip handle under the cursor, or None."""
        box = self.active_clip_box()
        if box is None:
            return None
        # Centre first: a face seen edge-on sits right on top of the centre
        # on screen, and the tie must go to the centre, which can always move.
        live = [CENTRE_HANDLE] + [k for k in self._live_handles() if k != CENTRE_HANDLE]
        screen, valid = self._screen_of(box.handle_positions()[live])
        d = np.hypot(screen[:, 0] - mouse_pos.x(), screen[:, 1] - mouse_pos.y())
        d[~valid] = np.inf
        j = int(np.argmin(d))
        if d[j] > self._HANDLE_PICK_RADIUS_PX:
            return None
        return live[j]

    def begin_handle_drag(self, mouse_pos, one_sided: bool) -> bool:
        """Start dragging the handle under the cursor. False if there is none."""
        handle = self.handle_at(mouse_pos)
        if handle is None:
            return False
        self._handle_drag = (handle, one_sided, mouse_pos)
        return True

    def drag_handle(self, mouse_pos):
        """Follow the mouse with the handle being dragged."""
        handle, one_sided, last = self._handle_drag
        box = self._clip_box
        move = np.array([mouse_pos.x() - last.x(), mouse_pos.y() - last.y()],
                        dtype=np.float64)
        self._handle_drag = (handle, one_sided, mouse_pos)

        if handle == CENTRE_HANDLE:
            # Move in the screen plane through the centre: how far the
            # camera's right and up axes travel on screen per metre, solved
            # for the mouse movement.
            right, up, _ = self._camera_axes()
            base = box.centre
            screen, valid = self._screen_of(np.stack([base, base + right, base + up]))
            if not valid.all():
                return
            jac = np.stack([screen[1] - screen[0], screen[2] - screen[0]], axis=1)
            try:
                a, b = np.linalg.solve(jac, move)
            except np.linalg.LinAlgError:
                return
            box.move(a * right + b * up)
        else:
            axis, sign = FACE_HANDLES[handle]
            outward = sign * box.axes[axis]
            p = box.handle_positions()[handle]
            screen, valid = self._screen_of(np.stack([p, p + outward]))
            if not valid.all():
                return
            per_metre = screen[1] - screen[0]
            length_sq = float(per_metre @ per_metre)
            if length_sq < 4.0:     # face points at the camera; no clear direction
                return
            box.drag_face(handle, float(move @ per_metre) / length_sq, one_sided,
                          scale_axes=(0, 1) if self._slab_mode else (0, 1, 2))

        self.update()

    def end_handle_drag(self):
        self._handle_drag = None
        self._refill_level_of_detail()

    # ------------------------------------------------------------------
    # Level of detail
    # ------------------------------------------------------------------

    def _refill_level_of_detail(self):
        """Re-render so the box is drawn at full resolution, if LOD thins it.

        Only needed when the visible points are over the point budget; below
        it everything is drawn already and the clip planes are the whole job.
        """
        controller = global_variables.global_application_controller
        main_window = global_variables.global_main_window
        if controller is None or main_window is None:
            return
        rc = controller.rendering_coordinator
        if rc is None or rc.total_visible_points <= rc._point_budget:
            return

        box = self.active_clip_box()
        key = box.key() if box is not None else None
        if key == self._lod_filled_box_key:
            return
        self._lod_filled_box_key = key
        main_window.render_visible_data(zoom_extent=False)

        if box is not None and rc.focus_thinned:
            self._show_message(
                f"The box holds {rc.focus_points_inside:,} points, more than the "
                f"point budget of {rc._point_budget:,}; showing them thinned.")

    def _show_message(self, text: str):
        main_window = global_variables.global_main_window
        bar = getattr(main_window, "statusbar", None)
        if bar is not None:
            bar.showMessage(text, 6000)
        logger.info(text)
