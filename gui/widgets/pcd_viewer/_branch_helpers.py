import logging
import numpy as np

from config.config import global_variables

logger = logging.getLogger(__name__)


class BranchSelectionMixin:
    """Branch-scoped selection helpers for PCDViewerWidget."""

    def _init_branch_state(self):
        """Initialize branch offset and LOD state."""
        # Per-branch index ranges in combined vertex array: uid -> (start, end)
        self._branch_offsets = {}

        # LOD state (for triggering DataManager re-render)
        self._current_sample_rate: float = 1.0
        self._lod_enabled: bool = True  # Dynamic LOD for large point clouds

    def set_branch_offsets(self, offsets: dict):
        """Store per-branch index ranges for selection filtering."""
        self._branch_offsets = offsets

    def _is_index_in_selected_branch(self, index: int) -> bool:
        """Check if a point index belongs to one of the selected branches."""
        if not self._branch_offsets:
            return True
        controller = global_variables.global_application_controller
        if controller is None:
            return True
        selected = controller.selected_branches
        if not selected:
            return True
        for uid in selected:
            rng = self._branch_offsets.get(uid)
            if rng and rng[0] <= index < rng[1]:
                return True
        return False

    def _get_selected_branch_index_range(self):
        """Return (start, end) tuples for all selected branches, or None if no filtering."""
        if not self._branch_offsets:
            return None
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        selected = controller.selected_branches
        if not selected:
            return None
        ranges = []
        for uid in selected:
            rng = self._branch_offsets.get(uid)
            if rng:
                ranges.append(rng)
        return ranges if ranges else None

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

    def _is_point_selection_locked(self, index: int) -> bool:
        """Check if a single point index belongs to a cluster locked against selection."""
        if not self._branch_offsets:
            return False
        for uid, (start, end) in self._branch_offsets.items():
            if start <= index < end:
                info = self._get_cluster_lock_info(uid)
                if info is None:
                    return False
                labels, locked = info
                local_idx = index - start
                if local_idx < len(labels):
                    cid = int(labels[local_idx])
                    return "select" in locked.get(cid, set())
                return False
        return False

    def _filter_selection_locked(self, indices):
        """Filter out indices belonging to clusters locked against selection."""
        if not self._branch_offsets:
            return indices
        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            info = self._get_cluster_lock_info(uid)
            if info is None:
                continue
            labels, locked = info
            locked_ids = {cid for cid, locks in locked.items() if "select" in locks}
            for i, idx in enumerate(indices):
                if start <= idx < end:
                    local_idx = idx - start
                    if local_idx < len(labels) and int(labels[local_idx]) in locked_ids:
                        keep_mask[i] = False
        return indices[keep_mask]

    def _get_cluster_labels(self, uid):
        """Get cluster labels array for a cluster_labels branch, or None."""
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        node = controller.get_node(uid)
        if node is None or node.data_type != "cluster_labels":
            return None
        return getattr(node.data, 'labels', None)

    def _is_noise_point(self, index: int) -> bool:
        """Check if a point index is a noise point (cluster label == -1)."""
        if not self._branch_offsets:
            return False
        for uid, (start, end) in self._branch_offsets.items():
            if start <= index < end:
                labels = self._get_cluster_labels(uid)
                if labels is None:
                    return False
                local_idx = index - start
                if local_idx < len(labels):
                    return int(labels[local_idx]) == -1
                return False
        return False

    def _filter_selection(self, indices):
        """Apply the full selection filtering pipeline to an array of point indices.

        Filters in order: branch membership, selection lock, noise points.

        Args:
            indices (np.ndarray): 1D array of candidate point indices.

        Returns:
            np.ndarray: Filtered array of valid point indices.
        """
        if indices.size == 0:
            return indices

        # 1. Branch membership filter
        branch_ranges = self._get_selected_branch_index_range()
        if branch_ranges is not None:
            mask = np.zeros(indices.size, dtype=bool)
            for start, end in branch_ranges:
                mask |= (indices >= start) & (indices < end)
            indices = indices[mask]

        # 2. Selection lock filter
        if indices.size > 0:
            indices = self._filter_selection_locked(indices)

        # 3. Noise filter
        if indices.size > 0:
            indices = self._filter_noise_points(indices)

        return indices

    def _is_point_selectable(self, index):
        """Check whether a single point passes all selection filters.

        Args:
            index (int): Point index to check.

        Returns:
            bool: True if the point passes all filters.
        """
        if not self._is_index_in_selected_branch(index):
            return False
        if self._is_point_selection_locked(index):
            return False
        if self._is_noise_point(index):
            return False
        return True

    def _filter_noise_points(self, indices):
        """Filter out indices that are noise points (cluster label == -1)."""
        if not self._branch_offsets:
            return indices
        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            labels = self._get_cluster_labels(uid)
            if labels is None:
                continue
            for i, idx in enumerate(indices):
                if start <= idx < end:
                    local_idx = idx - start
                    if local_idx < len(labels) and int(labels[local_idx]) == -1:
                        keep_mask[i] = False
        return indices[keep_mask]
