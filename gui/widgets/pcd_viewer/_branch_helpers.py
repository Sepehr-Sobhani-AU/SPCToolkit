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
                cid = self._label_of(uid, labels, index, start)
                if cid is None:
                    return False
                return "select" in locked.get(cid, set())
        return False

    def _branch_labels_of(self, uid, labels, indices, start, end):
        """The labels *labels* holds for whichever of *indices* belong to branch
        *uid*.

        The array counterpart of ``_label_of``, and it resolves LOD the same way
        — through ``cloud_indices`` rather than ``indices - start``, since labels
        are full-resolution source data while the rendered rows are whatever LOD
        kept. Indices outside the branch, or whose source row has no label, are
        simply not returned.

        Args:
            uid (str): Branch uid.
            labels (np.ndarray): Full-resolution per-point labels.
            indices (np.ndarray): Indices into the combined render buffer.
            start (int): The branch's start offset in that buffer.
            end (int): The branch's end offset in that buffer.

        Returns:
            tuple: (positions, label_values) — positions index into *indices*.
        """
        empty = np.empty(0, dtype=np.int64)
        in_branch = np.flatnonzero((indices >= start) & (indices < end))
        if in_branch.size == 0:
            return empty, empty

        rows = self.cloud_indices(uid, indices[in_branch] - start)
        resolved = (rows >= 0) & (rows < len(labels))
        return in_branch[resolved], labels[rows[resolved]]

    def _filter_selection_locked(self, indices):
        """Filter out indices belonging to clusters locked against selection.

        Kept for callers that want the lock rule on its own; the selection
        pipeline uses ``_filter_locked_and_noise`` instead, which applies this
        rule and the noise rule from a single label lookup.
        """
        if not self._branch_offsets:
            return indices
        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            info = self._get_cluster_lock_info(uid)
            if info is None:
                continue
            labels, locked = info
            locked_ids = [cid for cid, locks in locked.items() if "select" in locks]
            if not locked_ids:
                continue
            positions, values = self._branch_labels_of(uid, labels, indices, start, end)
            if positions.size == 0:
                continue
            keep_mask[positions[np.isin(values, locked_ids)]] = False
        return indices[keep_mask]

    def _filter_locked_and_noise(self, indices):
        """Drop indices that are locked against selection *or* are noise.

        Both rules ask the same question — what label does this point carry? —
        so they share one lookup. Run separately they each resolved LOD rows and
        gathered labels for every candidate index, and on a lasso holding a
        million points that duplicated pass was the single largest cost of
        closing a lasso, larger than the point-in-polygon test itself.
        """
        if not self._branch_offsets:
            return indices

        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            labels = self._get_cluster_labels(uid)
            if labels is None:
                continue

            positions, values = self._branch_labels_of(uid, labels, indices, start, end)
            if positions.size == 0:
                continue

            drop = (values == -1)                       # noise
            info = self._get_cluster_lock_info(uid)
            if info is not None:
                _, locked = info
                locked_ids = [cid for cid, locks in locked.items() if "select" in locks]
                if locked_ids:
                    np.logical_or(drop, np.isin(values, locked_ids), out=drop)

            keep_mask[positions[drop]] = False

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
                return self._label_of(uid, labels, index, start) == -1
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

        # 2. Selection lock and noise, from one label lookup
        if indices.size > 0:
            indices = self._filter_locked_and_noise(indices)

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
            positions, values = self._branch_labels_of(uid, labels, indices, start, end)
            if positions.size == 0:
                continue
            keep_mask[positions[values == -1]] = False
        return indices[keep_mask]
