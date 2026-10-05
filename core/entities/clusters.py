"""
A class for a DataNode's 'data' attribute containing cluster labels and optional colors.

This unified class replaces both the old Clusters and FeatureClasses classes,
providing cluster labels with optional semantic naming and color support.
"""

import numpy as np
from typing import Dict, List, Optional


# The two reserved labels. Everything >= 0 is a real cluster.
#
# NOISE: a point a clustering algorithm looked at and rejected (DBSCAN/HDBSCAN
# output). Drawn dark grey, never selectable, never a cluster.
#
# UNASSIGNED: a point no step has assigned YET — the rest of the cloud on a
# branch that labels only some of it (e.g. linear region growing's lines). Drawn
# in the parent branch's own colours and selectable, so the user can keep
# picking from it; never a cluster.
NOISE_LABEL = -1
UNASSIGNED_LABEL = -2

# Colour for UNASSIGNED points when the parent has no colours of its own — the
# viewer's default for an uncoloured cloud, so the rest looks like the plain
# cloud rather than like noise.
UNASSIGNED_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def is_cluster(labels):
    """Whether each label is a real cluster — neither noise nor unassigned.

    The one test every consumer should use instead of comparing against -1, so
    a new reserved label never has to be hunted down across the plugins again.
    Accepts a scalar (returns bool) or an array (returns a bool array).
    """
    if np.isscalar(labels):
        return int(labels) >= 0
    return np.asarray(labels) >= 0


class Clusters:
    """
    A class for a DataNode's 'data' attribute containing cluster labels and optional colors.

    This class stores clustering results produced by clustering algorithms like DBSCAN.
    It maintains cluster labels for each point and optional RGB colors for visualization.

    When cluster_names is provided, clusters have semantic meaning (e.g., "Car", "Tree")
    and can be colored by their names using cluster_colors.

    Args:
        labels (np.ndarray): Integer array of cluster labels with shape (n_points,).
                            ``NOISE_LABEL`` (-1) marks noise and
                            ``UNASSIGNED_LABEL`` (-2) points not assigned yet;
                            see the module constants.
        colors (np.ndarray, optional): Array of RGB color values with shape (n_points, 3).
                                      Values should be in the range [0, 1].
                                      Defaults to None.
        cluster_names (Dict[int, str], optional): Mapping from cluster IDs to names.
                                                  E.g., {0: "Car", 1: "Tree", 2: "Car"}
        cluster_colors (Dict[str, np.ndarray], optional): Mapping from names to RGB colors.
                                                          E.g., {"Car": [1, 0, 0], "Tree": [0, 1, 0]}
        line_traces (Dict, optional): For clusters produced by linear region
            growing — everything needed to CONTINUE a short trace in a later
            session instead of re-growing it from scratch::

                {"params": {...},                     # rebuilds the grower
                 "lines": [{"label": 0,
                            "centerline": [[x, y, z], ...],
                            "stops": [{"tip": [...], "direction": [...],
                                       "reason": str, "resolved": bool}]}]}

            Plain data on purpose: it outlives the session inside the project
            file, so it must not depend on any service class staying importable
            at the same path.
    """

    def __init__(
        self,
        labels: np.ndarray,
        colors: np.ndarray = None,
        cluster_names: Optional[Dict[int, str]] = None,
        cluster_colors: Optional[Dict[str, np.ndarray]] = None,
        locked_clusters: Optional[Dict[int, set]] = None,
        custom_colors: Optional[Dict[int, np.ndarray]] = None,
        line_traces: Optional[Dict] = None,
        tint_locked: bool = True
    ):
        # Type conversion
        self.labels = labels.astype(np.int32)
        self.colors = colors.astype(np.float32) if colors is not None else None
        self.cluster_names = cluster_names if cluster_names is not None else {}
        self.cluster_colors = cluster_colors if cluster_colors is not None else {}
        self.locked_clusters = locked_clusters if locked_clusters is not None else {}
        self.custom_colors = custom_colors if custom_colors is not None else {}
        self.line_traces = line_traces if line_traces is not None else {}
        # Whether locked clusters are tinted grey-blue. On by default as feedback
        # for a lock the user chose; off where locks are automatic and every
        # cluster carries one (linear region growing locks each line against
        # selection), since tinting them all would only wash the result out.
        self.tint_locked = tint_locked

        # Validate labels
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("The labels must be a numpy array.")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("The labels must be an integer array.")
        if len(self.labels.shape) != 1:
            raise ValueError("The labels must be a 1D array.")
        if len(self.labels) == 0:
            raise ValueError("The labels must have at least one value.")

        # Validate colors if provided
        if self.colors is not None:
            if not isinstance(self.colors, np.ndarray):
                raise ValueError("The colors must be a numpy array.")
            if not np.issubdtype(self.colors.dtype, np.floating):
                raise ValueError("The colors must be a floating-point array.")
            if len(self.colors.shape) != 2 or self.colors.shape[1] != 3:
                raise ValueError("The colors must be a 2D array with shape (n_points, 3).")
            if len(self.colors) != len(self.labels):
                raise ValueError(
                    f"The number of colors ({len(self.colors)}) must match "
                    f"the number of labels ({len(self.labels)})."
                )

            # Ensure color values are in range [0, 1]
            if np.min(self.colors) < 0 or np.max(self.colors) > 1:
                print("Warning: Some color values are outside the range [0, 1]. Clipping values.")
                self.colors = np.clip(self.colors, 0, 1)

    def __getattr__(self, name):
        """Provide defaults for attributes added after pickle-serialized objects were saved."""
        if name == "locked_clusters":
            self.locked_clusters = {}
            return self.locked_clusters
        if name == "custom_colors":
            self.custom_colors = {}
            return self.custom_colors
        if name == "line_traces":
            self.line_traces = {}
            return self.line_traces
        if name == "tint_locked":
            self.tint_locked = True
            return self.tint_locked
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setstate__(self, state):
        """Unpickle, then bring an old linear-region-growing result up to date.

        Those results were saved with the rest of the cloud as -1 and their lines
        unlocked: on load the rest could not be picked, so no further line could
        be grown from it. Their -1 never meant noise — linear growing does not
        produce noise, it labels the lines and leaves the rest — so it becomes
        UNASSIGNED, and each line is locked against selection the way a fresh
        result is. Identified by its line traces; every other branch loads as
        saved.
        """
        self.__dict__.update(state)
        traces = self.__dict__.get("line_traces")
        if not (isinstance(traces, dict) and "lines" in traces):
            return
        self.labels[self.labels == NOISE_LABEL] = UNASSIGNED_LABEL
        locked = self.__dict__.setdefault("locked_clusters", {})
        for label in np.unique(self.labels[is_cluster(self.labels)]):
            locked.setdefault(int(label), set()).add("select")
        self.tint_locked = False

    def has_names(self) -> bool:
        """Check if this Clusters object has semantic names assigned."""
        return len(self.cluster_names) > 0

    def get_unique_names(self) -> List[str]:
        """
        Get the list of unique cluster names.

        Returns:
            List[str]: List of unique cluster names.
        """
        return list(set(self.cluster_names.values()))

    def get_points_by_name(self, name: str) -> np.ndarray:
        """
        Get a boolean mask for points belonging to clusters with the given name.

        Args:
            name (str): The cluster name to filter by.

        Returns:
            np.ndarray: Boolean mask identifying points of clusters with that name.
        """
        unique_names = self.get_unique_names()
        if name not in unique_names:
            raise ValueError(f"Cluster name '{name}' not found in the mapping.")

        # Find all cluster IDs that map to this name
        cluster_ids = [cid for cid, n in self.cluster_names.items() if n == name]

        if not cluster_ids:
            return np.zeros(len(self.labels), dtype=bool)

        # Use np.isin for vectorized membership test (much faster than loop)
        return np.isin(self.labels, cluster_ids)

    def get_named_colors(self) -> np.ndarray:
        """
        Get color values for all points based on their cluster names.

        Uses vectorized lookup table for O(n) performance instead of O(n*k).
        Requires cluster_names and cluster_colors to be set. UNASSIGNED points get
        ``UNASSIGNED_COLOR``; the clusters transformer replaces it with the
        parent's own colours where the parent has them.

        Returns:
            np.ndarray: Array of RGB color values for each point with shape (n_points, 3).
        """
        n_points = len(self.labels)
        default_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)

        if len(self.cluster_names) == 0:
            return np.tile(default_color, (n_points, 1))

        # Find the range of label values for lookup table sizing
        min_label = int(np.min(self.labels))
        max_label = int(np.max(self.labels))

        # Offset to handle negative labels (e.g., -1 for noise)
        offset = -min_label if min_label < 0 else 0
        table_size = max_label + offset + 1

        # Build color lookup table (O(k) where k = number of mappings)
        color_lut = np.tile(default_color, (table_size, 1))

        for label_id, cluster_name in self.cluster_names.items():
            if cluster_name in self.cluster_colors:
                idx = label_id + offset
                if 0 <= idx < table_size:
                    color_lut[idx] = self.cluster_colors[cluster_name]

        # Apply lookup using vectorized indexing (O(n))
        adjusted_labels = self.labels + offset
        colors = color_lut[adjusted_labels]
        colors[self.labels == UNASSIGNED_LABEL] = UNASSIGNED_COLOR

        return colors.astype(np.float32)

    def set_random_color(self, noise_color: np.ndarray = np.array([0.2, 0.2, 0.2])):
        """
        Generates random colors for each cluster and assigns them to points.

        This method creates a unique random color for each cluster label and
        assigns a specified color to noise points (``NOISE_LABEL``).

        UNASSIGNED points take no part in the palette: they get
        ``UNASSIGNED_COLOR`` (the clusters transformer replaces it with the
        parent's own colours), and every other label keeps exactly the colour it
        would have without them. UNASSIGNED sorts before every other label, so
        giving it a palette slot would shift the colour of every cluster.

        Args:
            noise_color (np.ndarray): The RGB color to assign to noise points.
                                     Defaults to [0.2, 0.2, 0.2] (dark gray).
        """
        # Get unique labels and their indexes
        unique_labels, label_indexes = np.unique(self.labels, return_inverse=True)

        # Random colours for every label except UNASSIGNED (noise included, then
        # overwritten — that keeps the palette order it has always had).
        np.random.seed(42)  # Optional: Set a seed for reproducibility
        in_palette = unique_labels != UNASSIGNED_LABEL
        cluster_colors = np.empty((len(unique_labels), 3))
        cluster_colors[in_palette] = np.random.rand(int(in_palette.sum()), 3)
        cluster_colors[~in_palette] = UNASSIGNED_COLOR

        # Assign a color for noise
        noise_labels = unique_labels == NOISE_LABEL
        cluster_colors[noise_labels] = noise_color

        # Assign colors to each point based on their label_indexes
        point_colors = cluster_colors[label_indexes]

        # Apply user-defined custom colors per cluster
        if self.custom_colors:
            for label_val, idx_in_unique in zip(unique_labels, range(len(unique_labels))):
                cid = int(label_val)
                if cid in self.custom_colors:
                    mask = label_indexes == idx_in_unique
                    point_colors[mask] = self.custom_colors[cid]

        # Tint locked clusters to provide visual feedback
        if self.locked_clusters and self.tint_locked:
            lock_tint = np.array([0.4, 0.5, 0.6], dtype=np.float32)
            for label_val, idx_in_unique in zip(unique_labels, range(len(unique_labels))):
                if int(label_val) in self.locked_clusters:
                    mask = label_indexes == idx_in_unique
                    point_colors[mask] = point_colors[mask] * 0.5 + lock_tint * 0.5

        self.colors = np.float32(point_colors)

        # Sync cluster_colors dict so get_named_colors() works when has_names() is true
        if self.cluster_names:
            for label_val, idx_in_unique in zip(unique_labels, range(len(unique_labels))):
                cid = int(label_val)
                if cid in self.cluster_names and is_cluster(cid):
                    name = self.cluster_names[cid]
                    # Use custom color if set, otherwise the random color
                    if cid in self.custom_colors:
                        self.cluster_colors[name] = np.array(self.custom_colors[cid], dtype=np.float32)
                    else:
                        self.cluster_colors[name] = np.array(cluster_colors[idx_in_unique], dtype=np.float32)

    def __repr__(self):
        n_clusters = len(np.unique(self.labels[is_cluster(self.labels)]))
        n_noise = np.sum(self.labels == NOISE_LABEL)
        has_colors = self.colors is not None
        has_names = self.has_names()

        if has_names:
            n_unique_names = len(self.get_unique_names())
            return (f"Clusters(n_points={len(self.labels)}, n_clusters={n_clusters}, "
                    f"n_noise={n_noise}, has_colors={has_colors}, "
                    f"has_names={has_names}, n_unique_names={n_unique_names})")
        else:
            return (f"Clusters(n_points={len(self.labels)}, n_clusters={n_clusters}, "
                    f"n_noise={n_noise}, has_colors={has_colors})")

    def __str__(self):
        return self.__repr__()
