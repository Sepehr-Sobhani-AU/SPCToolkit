import logging
import numpy as np
from OpenGL.GL import glReadPixels, GL_DEPTH_COMPONENT, GL_FLOAT
from OpenGL.GLU import gluProject, gluUnProject

logger = logging.getLogger(__name__)


class PointPickingMixin:
    """Point selection and deselection logic for PCDViewerWidget."""

    @staticmethod
    def _project_points_to_screen(pts_3d, mv, proj, viewport):
        """Project an array of 3D points to 2D screen coordinates (Qt: top-left origin, Y-down).

        Args:
            pts_3d: (N, 3) float64 array of world coordinates.
            mv: 4x4 model-view matrix (column-major transposed).
            proj: 4x4 projection matrix (same convention).
            viewport: (vp_x, vp_y, vp_w, vp_h).

        Returns:
            tuple: (screen_x, screen_y, valid_mask) arrays of shape (N,).
        """
        n = pts_3d.shape[0]
        pts_homo = np.empty((n, 4), dtype=np.float64)
        pts_homo[:, :3] = pts_3d
        pts_homo[:, 3] = 1.0

        # Collapse the two 4x4s first. Written as ``pts_homo @ mv @ proj`` this
        # evaluates left to right and runs two N-sized matrix products, with a
        # full (N, 4) intermediate between them; folding the matrices leaves one.
        clip = pts_homo @ (mv @ proj)
        w = clip[:, 3]
        valid_mask = w > 0

        # Normalise in place in the clip buffer rather than into fresh arrays.
        ndc = clip[:, :2]
        np.divide(ndc, w[:, None], out=ndc, where=valid_mask[:, None])
        ndc[~valid_mask] = 0.0

        vp_x, vp_y, vp_w, vp_h = viewport[0], viewport[1], viewport[2], viewport[3]
        screen_x = ndc[:, 0] * (0.5 * vp_w) + (0.5 * vp_w + vp_x)
        screen_y = (0.5 * vp_h + vp_y) - ndc[:, 1] * (0.5 * vp_h)

        return screen_x, screen_y, valid_mask

    def _unproject_mouse_to_world(self, mouse_pos):
        """Unproject a mouse position through the depth buffer to world coordinates.

        Resolves whatever geometry was drawn at that pixel — points, lines, or
        anything else — so it carries no dependency on point data.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.

        Returns:
            numpy.ndarray: (3,) world coordinates, or None if the pixel is empty.
        """
        self.makeCurrent()

        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()

        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        if win_z == 1.0:
            return None

        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        return np.array(world_coords[:3])

    def _unproject_mouse_to_nearest_point(self, mouse_pos):
        """Unproject a mouse position through the depth buffer and find the nearest point.

        Returns:
            tuple: (point_index, point_3d_coords) or (None, None) if no point found.
        """
        # Line geometry renders independently of point branches, so a filled
        # depth pixel does not imply there is a cloud to snap to.
        if self.points is None or self.max_extent is None:
            return None, None

        pick_point = self._unproject_mouse_to_world(mouse_pos)
        if pick_point is None:
            return None, None

        threshold = self.max_extent * self.picking_point_threshold_factor

        min_index = self._nearest_point_within(pick_point, threshold)
        if min_index is None:
            return None, None
        return min_index, self.points[min_index, :3].copy()

    def _nearest_point_within(self, target, radius):
        """Row of ``self.points`` closest to *target*, or None if none is within
        *radius*.

        The one place every snap-to-point goes through — Shift+Left select,
        Ctrl+Shift+Right cluster deselect, and double-click to re-centre the
        view — so all three get the same speed from one implementation.

        Uses the per-branch pick grid when it is ready, which measures the
        distance to the points in the cursor's cell instead of to every point in
        the cloud. Falls back to a straight scan while a grid is still building,
        which is what this did before the grid existed.

        The *radius* is worth knowing about: it is
        ``max_extent * picking_point_threshold_factor`` and the factor defaults
        to 1.0, so it is the whole size of the cloud and excludes nothing. It is
        honoured rather than assumed small.

        Args:
            target: (3,) world coordinates.
            radius: Maximum accepted distance.

        Returns:
            int row index, or None.
        """
        ready, index = self._nearest_via_pick_grids(target, radius)
        if ready:
            return index
        return self._nearest_by_scan(target, radius)

    def _nearest_via_pick_grids(self, target, radius):
        """``(ready, index)`` from the per-branch pick grids.

        *ready* is False when any visible branch has no grid yet, in which case
        the caller scans instead. All-or-nothing on purpose: mixing a gridded
        branch with a scanned one would give the same answer but two code paths
        to keep in step for no gain.

        Every branch is asked for its grid even once one is known to be missing,
        so that one click starts every build. Returning early instead meant only
        the first branch without a grid began building, so N visible branches
        needed N clicks — each of them a full scan — before the fast path ever
        engaged.
        """
        offsets = self._branch_offsets
        if not offsets:
            return False, None

        best_sq = float(radius) ** 2
        best_index = None
        ready = True

        for uid, (start, _end) in offsets.items():
            grid = self._pick_grid_for(uid)
            if grid is None:
                ready = False
                continue

            slc = self._branch_vertices.get(uid)
            row, sq = grid.nearest(slc, target, max_dist=radius)
            if row is not None and sq < best_sq:
                best_sq, best_index = sq, start + int(row)

        return (True, best_index) if ready else (False, None)

    def _nearest_by_scan(self, target, radius):
        """Row of ``self.points`` closest to *target* by straight scan.

        The fallback for while a pick grid is still being built. Chunked so the
        intermediate masks stay bounded on a very large buffer.
        """
        pts = self.points
        if pts is None or len(pts) == 0:
            return None

        tx, ty, tz = (float(target[0]), float(target[1]), float(target[2]))
        best_sq = float(radius) ** 2
        best_index = None

        for start in range(0, len(pts), self._PICK_SCAN_CHUNK):
            block = pts[start:start + self._PICK_SCAN_CHUNK, :3]

            near = np.abs(block[:, 0] - tx) < radius
            np.logical_and(near, np.abs(block[:, 1] - ty) < radius, out=near)
            np.logical_and(near, np.abs(block[:, 2] - tz) < radius, out=near)
            rows = np.flatnonzero(near)
            if rows.size == 0:
                continue

            near_pts = block[rows].astype(np.float64)
            near_pts[:, 0] -= tx
            near_pts[:, 1] -= ty
            near_pts[:, 2] -= tz
            sq = np.einsum('ij,ij->i', near_pts, near_pts)

            j = int(np.argmin(sq))
            if sq[j] < best_sq:
                best_sq = float(sq[j])
                best_index = start + int(rows[j])

        return best_index

    def pick_point(self, mouse_pos, select=True):
        """
        Handle point picking or deselecting points in the point cloud.

        This method is used to pick or deselect points in the point cloud based on a mouse click position. It uses
        OpenGL to project the clicked point onto the screen space and determines whether a point in the point cloud
        is close enough to be picked or deselected. If `select` is True, the method attempts to pick a point;
        otherwise, it attempts to deselect a point.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
            select (bool, optional): A flag indicating whether to select (True) or deselect (False) the point. Defaults
                to True.
        """

        if select:
            self.select_point_at(mouse_pos)
        else:
            self.deselect_point_at(mouse_pos)


    def select_point_at(self, mouse_pos):
        """
        Select a point in the point cloud at the given mouse position.

        This method is used to select a point in the point cloud based on the mouse click position in widget coordinates.
        It reads the depth buffer to get the depth value at the mouse position and then unprojects the screen coordinates
        to world coordinates. The closest point to the unprojected coordinates is selected if it lies within a specified
        threshold.

        The distance threshold used for selecting points is defined by the `picking_point_threshold_factor` attribute.
        You can adjust this attribute to control how close a point must be to be considered selectable.

        If a point is successfully selected, its index is added to the `picked_points_indices` attribute, which stores the
        indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        idx, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if idx is None:
            return

        if not self._is_point_selectable(idx):
            return
        if idx not in self.picked_points_indices:
            self.picked_points_indices.append(idx)

    def deselect_point_at(self, mouse_pos):
        """
        Deselect a point in the point cloud at the given mouse position.

        This method is used to deselect a previously selected point in the point cloud based on the mouse click position
        in widget coordinates. It projects the picked points to screen space and determines the closest point to the
        mouse click position. If the closest point lies within a specified pixel threshold, it is deselected.

        The pixel threshold used for deselecting points is defined by the `pixel_threshold` attribute. You can adjust
        this attribute to control how close a point must be to be considered for deselection.

        If a point is successfully deselected, its index is removed from the `picked_points_indices` attribute, which
        stores the indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Picked indices outlive the branches they came from, so the cloud may
        # be gone while the selection list is still populated.
        if self.points is None or not self.picked_points_indices:
            return

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Project all picked points in one pass rather than a gluProject call
        # each: a polygon selection leaves millions of picked points, and a GL
        # round trip per point makes a single right-click take seconds.
        selected = np.fromiter(
            self.picked_points_indices, dtype=np.int64,
            count=len(self.picked_points_indices)
        )
        selected = selected[(selected >= 0) & (selected < len(self.points))]
        if selected.size == 0:
            return

        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)
        screen_x, screen_y, valid_mask = self._project_points_to_screen(
            self.points[selected, :3].astype(np.float64), mv, proj, self.viewport
        )

        # _project_points_to_screen reports Qt widget coordinates (top-left
        # origin, Y down), which is what mouse_pos already is — no Y flip.
        dx = screen_x - mouse_pos.x()
        dy = screen_y - mouse_pos.y()
        dist_sq = dx * dx + dy * dy
        dist_sq[~valid_mask] = np.inf

        nearest = int(np.argmin(dist_sq))

        # Define a threshold in pixels (e.g., radius of the sphere in screen space)
        # TODO: Not sure if the pixel threshold is appropriate for all cases
        pixel_threshold = self.pixel_threshold

        if dist_sq[nearest] <= pixel_threshold ** 2:
            # Remove the point from picked points
            self.picked_points_indices.remove(int(selected[nearest]))
            # Invalidate stored polygons so plugins fall back to coordinate matching
            self._selection_polygons.clear()

    def deselect_cluster_at(self, mouse_pos):
        """
        Deselect all selected points that share the same color as the clicked point.

        This removes an entire cluster from the selection by matching the clicked point's
        RGB color against the colors of all currently selected points. Points with matching
        colors are removed from picked_points_indices.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """
        if not self.picked_points_indices:
            return

        clicked_index, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if clicked_index is None:
            return

        # Get the color of the clicked point
        target_color = self.points[clicked_index, 3:6]

        # Drop picks that no longer address a drawn point before indexing with
        # them. picked_points_indices outlives set_branches() on purpose, so
        # after LOD drops a level it still holds rows past the end of the
        # shorter buffer — and negative entries would wrap to the far end of it
        # rather than being ignored.
        selected = np.fromiter(self.picked_points_indices, dtype=np.int64,
                               count=len(self.picked_points_indices))
        selected = selected[(selected >= 0) & (selected < len(self.points))]
        if selected.size == 0:
            self.clear_selection()
            return

        # Vectorized removal of all selected points matching this color
        colors = self.points[selected, 3:6]
        matches = np.all(colors == target_color, axis=1)
        self.picked_points_indices[:] = selected[~matches].tolist()
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.update()
