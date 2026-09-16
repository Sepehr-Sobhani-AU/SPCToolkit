import logging
import numpy as np

from config.config import global_variables

logger = logging.getLogger(__name__)


class BranchSelectionMixin:
    """Branch-scoped selection helpers for PCDViewerWidget."""

    def _init_branch_state(self):
        """Initialize LOD state. (Per-branch offsets live on the data mixin
        as a lazy property — see DataManagementMixin._branch_offsets.)"""
        self._current_sample_rate: float = 1.0
        self._lod_enabled: bool = True  # Dynamic LOD for large point clouds

    def _branch_accepts_selection(self, uid) -> bool:
        """Whether branch *uid* may receive a selection at all.

        The branch-membership filter, which used to be a scan over rendered
        index ranges. In cloud space it is one question per branch rather than
        one per point: either the tree has branches selected and this is one of
        them, or nothing is selected in the tree and every branch is fair game.
        """
        controller = global_variables.global_application_controller
        if controller is None:
            return True
        selected = controller.selected_branches
        if not selected:
            return True
        return str(uid) in {str(u) for u in selected}

    def _get_cluster_lock_info(self, uid):
        """Get (labels, locked_clusters) for a cluster_labels branch, or None."""
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        node = controller.get_node(uid)
        if node is None or node.data_type != "cluster_labels":
            return None
        clusters = node.data
        if not getattr(clusters, 'locked_clusters', None):
            return None
        # Check if any cluster is locked against selection
        has_select_lock = any("select" in locks for locks in clusters.locked_clusters.values())
        if not has_select_lock:
            return None
        return clusters.labels, clusters.locked_clusters

    def _label_of(self, uid, labels, index, start):
        """The label *labels* holds for the rendered point *index* of branch *uid*.

        Goes through ``cloud_index`` rather than ``index - start``: labels are
        full-resolution source data, while the rendered rows are whatever LOD
        kept. Subtracting the offset alone silently reads a different point's
        label on any cloud big enough to be subsampled. Returns None when the
        point has no label.
        """
        row = self.cloud_index(uid, index - start)
        if 0 <= row < len(labels):
            return int(labels[row])
        return None

    def selectable_cloud_mask(self, uid, n_points):
        """Which rows of branch *uid*'s full cloud the user is allowed to select.

        Returns a boolean mask over the branch's cloud, or None when nothing is
        excluded — which callers read as "all of it", so the common case costs
        no allocation.

        This is the cloud-space replacement for the old ``_filter_selection``.
        That one worked in rendered-index space, which is why the full-
        resolution re-test had to bypass it and hand plugins an ungated result
        (patched over with an ``allowed=`` argument every caller had to
        remember). Asking in cloud space means the gate is applied once, where
        the selection is actually decided, and there is nothing left to bypass.

        Delegates the label rules to ``selection_gate.selectable_cloud_indices``
        so "what may be selected" has one definition, not two that can drift.
        """
        if not self._branch_accepts_selection(uid):
            return np.zeros(n_points, dtype=bool)

        from application.selection_gate import selectable_cloud_indices

        controller = global_variables.global_application_controller
        node = None if controller is None else controller.get_node(uid)
        allowed = selectable_cloud_indices(node, n_points)
        if allowed is None:
            return None

        mask = np.zeros(n_points, dtype=bool)
        allowed = allowed[(allowed >= 0) & (allowed < n_points)]
        mask[allowed] = True
        return mask

    def _is_cloud_point_selectable(self, uid, row) -> bool:
        """Whether one cloud row of branch *uid* may be selected.

        The single-click counterpart of ``selectable_cloud_mask``: a click names
        one point, so building a whole-cloud mask to answer for it would be
        wasteful. Same two rules — noise is never selectable, nor is a cluster
        locked against selection.
        """
        if not self._branch_accepts_selection(uid):
            return False

        labels = self._get_cluster_labels(uid)
        if labels is None:
            return True
        if not (0 <= row < len(labels)):
            return False

        label = int(labels[row])
        if label == -1:                       # noise
            return False

        info = self._get_cluster_lock_info(uid)
        if info is None:
            return True
        _, locked = info
        return "select" not in locked.get(label, set())

    def _cloud_point_count(self, uid):
        """How many points branch *uid*'s full-resolution cloud has, or None.

        Cheap: the render path caches every visible branch's reconstruction as
        it draws it, so this is a dict lookup in all the cases that matter.
        """
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        try:
            return len(controller.reconstruct(uid).points)
        except Exception:
            logger.debug(f"Could not size the cloud for {str(uid)[:8]}")
            return None

    def _locate_render_index(self, index):
        """The ``(uid, cloud_row)`` a row of the combined render buffer names.

        Returns ``(None, -1)`` when the index falls in no visible branch, or
        names a rendered row LOD cannot resolve back to a source row.
        """
        for uid, (start, end) in self._branch_offsets.items():
            if start <= index < end:
                row = self.cloud_index(uid, index - start)
                return (uid, row) if row >= 0 else (None, -1)
        return None, -1

    def _get_cluster_labels(self, uid):
        """Get cluster labels array for a cluster_labels branch, or None."""
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        node = controller.get_node(uid)
        if node is None or node.data_type != "cluster_labels":
            return None
        return getattr(node.data, 'labels', None)
