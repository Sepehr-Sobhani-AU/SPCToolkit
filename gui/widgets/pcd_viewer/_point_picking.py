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
        Ctrl+Shift+Left/Right cluster select/deselect, and double-click to
        re-centre the view — so they all get the same speed from one
        implementation.

        Uses the per-branch coarse spatial index when it is ready, which measures the
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
        ready, index = self._nearest_via_coarse_indexes(target, radius)
        if ready:
            return index
        return self._nearest_by_scan(target, radius)

    def _nearest_via_coarse_indexes(self, target, radius):
        """``(ready, index)`` from the per-branch coarse spatial indexes.

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
            grid = self._coarse_index_for(uid)
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

        The fallback for while a coarse spatial index is still being built. Chunked so the
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
        """Select the point under the cursor.

        The pick is resolved to a ``(branch uid, cloud row)`` pair and recorded
        two ways on the branch itself: recorded as a click, which keeps the
        order, and set in that branch's selection mask, which is what plugins
        read. Both, so the highlight and the plugins can never disagree.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """
        idx, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if idx is None:
            return

        uid, row = self._locate_render_index(idx)
        if uid is None:
            return
        if not self._is_cloud_point_selectable(uid, row):
            return

        n = self._cloud_point_count(uid)
        if n is None or not (0 <= row < n):
            return

        mask = self._mask_store().selection_mask(uid)
        if mask is None or len(mask) != n:
            mask = np.zeros(n, dtype=bool)
        else:
            mask = mask.copy()
        mask[row] = True

        if not self._mask_store().add_pick(uid, row):
            return                       # already picked
        self.set_branch_selection(uid, mask)
        self.refresh_selection_readout()

    def deselect_point_at(self, mouse_pos):
        """Deselect the selected point nearest the cursor.

        Only DRAWN points are candidates — you can only aim at what you can
        see — so this projects the rendered rows the highlight is showing rather
        than the whole selection. After a lasso those differ by the LOD factor,
        and projecting the full cloud-space selection would make one right-click
        take seconds on a large cloud.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """
        if not self._mask_store().selection_masks() or self.points is None:
            return

        self.makeCurrent()

        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)

        best = None                      # (dist_sq, uid, cloud_row)
        for uid in self._visible_branches:
            rows = self._selection_draw_rows(uid)
            if rows is None or rows.size == 0:
                continue
            slc = self._branch_vertices.get(uid)
            if slc is None:
                continue

            screen_x, screen_y, valid = self._project_points_to_screen(
                slc[rows, :3].astype(np.float64), mv, proj, self.viewport)

            # _project_points_to_screen reports Qt widget coordinates (top-left
            # origin, Y down), which is what mouse_pos already is — no Y flip.
            dx = screen_x - mouse_pos.x()
            dy = screen_y - mouse_pos.y()
            dist_sq = dx * dx + dy * dy
            dist_sq[~valid] = np.inf

            nearest = int(np.argmin(dist_sq))
            if best is None or dist_sq[nearest] < best[0]:
                cloud_row = self.cloud_index(uid, int(rows[nearest]))
                if cloud_row >= 0:
                    best = (float(dist_sq[nearest]), uid, cloud_row)

        if best is None or best[0] > self.pixel_threshold ** 2:
            return

        _dist, uid, row = best
        mask = self._mask_store().selection_mask(uid)
        if mask is None or not (0 <= row < len(mask)):
            return

        mask = mask.copy()
        mask[row] = False
        self.set_branch_selection(uid, mask)

        # A point selected by a lasso or a cluster click has no recorded pick;
        # clearing its bit is then the whole job.
        self._mask_store().remove_pick(uid, row)

        self.refresh_selection_readout()
        self.update()

    def _cluster_at(self, mouse_pos):
        """The cluster under the cursor, as ``(uid, cloud_row, label)``.

        Clusters are identified by their label within their own branch, never by
        colour: two clusters can be drawn in the same RGB, and label 3 of one
        branch is unrelated to label 3 of another.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.

        Returns:
            tuple or None: None when nothing is under the cursor, the point's
            branch carries no cluster labels, or the point is noise (-1).
        """
        clicked_index, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if clicked_index is None:
            return None

        uid, row = self._locate_render_index(clicked_index)
        if uid is None:
            return None

        labels = self._get_cluster_labels(uid)
        if labels is None or not (0 <= row < len(labels)):
            return None

        label = int(labels[row])
        if label == -1:
            return None
        return uid, row, label

    def _apply_cluster(self, mouse_pos, select):
        """Add or remove the whole cluster under the cursor.

        The widening to every point carrying the label happens HERE, in cloud
        space, rather than in each plugin. It used to add only the rendered rows
        and leave every plugin to re-widen by label on its own — correct by
        convention, but nothing enforced it, so a plugin that read the geometric
        selection instead silently received just the LOD subset of the cluster
        and looked like it had worked.
        """
        cluster = self._cluster_at(mouse_pos)
        if cluster is None:
            return
        uid, _row, label = cluster

        labels = np.asarray(self._get_cluster_labels(uid))
        n = len(labels)
        in_cluster = (labels == label)

        if select:
            gate = self.selectable_cloud_mask(uid, n)
            if gate is not None:
                in_cluster &= gate

        existing = self._mask_store().selection_mask(uid)
        if existing is None or len(existing) != n:
            existing = np.zeros(n, dtype=bool)

        combined = (existing | in_cluster) if select else (existing & ~in_cluster)
        self.set_branch_selection(uid, combined)

        if not select:
            # Click picks inside the cluster go with it; the ordered list must
            # not keep naming points that are no longer selected.
            self._mask_store().remove_picks(uid, np.flatnonzero(in_cluster))

        self.refresh_selection_readout()
        self.update()

    def select_cluster_at(self, mouse_pos):
        """Add every point of the cluster under the cursor to the selection."""
        self._apply_cluster(mouse_pos, select=True)

    def deselect_cluster_at(self, mouse_pos):
        """Remove every point of the cluster under the cursor from the selection."""
        self._apply_cluster(mouse_pos, select=False)
