"""
Plugin for classifying points into geometric shape classes based on eigenvalue features.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
After reconstruction by AnalysisExecutor, eigenvalues are in point_cloud.attributes['eigenvalues'].

Classifies each point into one of 6 classes aligned with eigenvalue features:
- Planar:       High planarity          (flat surfaces — walls, ground, roofs)
- Linear:       High linearity          (elongated features — cables, poles)
- Cylindrical:  Both moderate planarity  (cylinder surfaces — tree trunks, pipes)
                AND moderate linearity
- Scatter:      High sphericity         (isotropic — vegetation, noise)
- Sparse:       Extreme total_variance  (percentile outliers)
- Transition:   Everything else         (mixed geometry, catch-all)
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {
    0: "Planar",
    1: "Linear",
    2: "Scatter",
    3: "Sparse",
    4: "Transition",
    5: "Cylindrical",
}

CLASS_COLORS = {
    "Planar":       np.array([0.9, 0.2, 0.1], dtype=np.float32),
    "Linear":       np.array([0.1, 0.8, 0.2], dtype=np.float32),
    "Scatter":      np.array([0.2, 0.3, 0.9], dtype=np.float32),
    "Sparse":       np.array([0.5, 0.5, 0.5], dtype=np.float32),
    "Transition":   np.array([1.0, 0.75, 0.0], dtype=np.float32),
    "Cylindrical":  np.array([0.0, 0.8, 0.8], dtype=np.float32),
}


def classify_points(eigenvalues: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Classify points by eigenvalue features into geometric shape classes.

    Args:
        eigenvalues: (N, 3) array of eigenvalues sorted ascending (lambda1 <= lambda2 <= lambda3).
        params: Dict with threshold keys.

    Returns:
        Tuple of (labels, counts) where labels is (N,) int32 array and counts is {label_id: count}.
    """
    n_points = len(eigenvalues)

    features = EigenvalueUtils().compute_geometric_features(eigenvalues)
    planarity = features["planarity"]
    linearity = features["linearity"]
    sphericity = features["sphericity"]
    total_variance = features["total_variance"]

    linear_thresh = params["linear_threshold"]
    planar_thresh = params["planar_threshold"]
    spher_thresh = params["sphericity_threshold"]
    cyl_thresh = params["cylindrical_threshold"]
    sparse_pct = params["sparse_percentile"]

    # Default: Transition (catch-all)
    labels = np.full(n_points, 4, dtype=np.int32)
    classified = np.zeros(n_points, dtype=bool)

    # 1. Sparse: extreme total_variance (bottom or top percentile)
    if sparse_pct > 0.0:
        lo = np.percentile(total_variance, sparse_pct)
        hi = np.percentile(total_variance, 100.0 - sparse_pct)
        sparse_mask = (total_variance < lo) | (total_variance > hi)
        labels[sparse_mask] = 3
        classified |= sparse_mask

    # 2. Cylindrical: both moderate linearity AND moderate planarity
    #    Captures the planarity-linearity edge of the ternary diagram
    #    (cylinder surfaces where lambda3 > lambda2 >> lambda1)
    cyl_mask = (~classified) & (linearity > cyl_thresh) & (planarity > cyl_thresh)
    labels[cyl_mask] = 5
    classified |= cyl_mask

    # 3. Linear: high linearity (remaining points)
    linear_mask = (~classified) & (linearity > linear_thresh)
    labels[linear_mask] = 1
    classified |= linear_mask

    # 4. Planar: high planarity (remaining points)
    planar_mask = (~classified) & (planarity > planar_thresh)
    labels[planar_mask] = 0
    classified |= planar_mask

    # 5. Scatter: high sphericity
    scatter_mask = (~classified) & (sphericity > spher_thresh)
    labels[scatter_mask] = 2
    classified |= scatter_mask

    # 6. Transition: everything else (already default)

    counts = {}
    for label_id in CLASS_NAMES:
        counts[label_id] = int(np.sum(labels == label_id))

    return labels, counts


class GeometricClassificationPlugin(Plugin):
    """
    Classify each point into a geometric shape class based on eigenvalue features.

    Decision tree (cascading, vectorized):
        1. SPARSE:       total_variance in bottom/top sparse_percentile
        2. CYLINDRICAL:  min(linearity, planarity) > cylindrical_threshold
        3. LINEAR:       linearity > linear_threshold
        4. PLANAR:       planarity > planar_threshold
        5. SCATTER:      sphericity > sphericity_threshold
        6. TRANSITION:   everything else (catch-all)
    """

    def get_name(self) -> str:
        return "geometric_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "cylindrical_threshold": {
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "label": "Cylindrical Threshold",
                "description": "Points with BOTH linearity and planarity above this are Cylindrical (0 = disable)"
            },
            "linear_threshold": {
                "type": "float",
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "label": "Linear Threshold",
                "description": "Linearity cutoff — points above this are classified as Linear"
            },
            "planar_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Planar Threshold",
                "description": "Planarity cutoff — points above this are classified as Planar"
            },
            "sphericity_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Sphericity Threshold",
                "description": "Sphericity cutoff — points above this are classified as Scatter"
            },
            "sparse_percentile": {
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "max": 25.0,
                "label": "Sparse Percentile",
                "description": "Bottom/top N% of total variance are classified as Sparse (0 = disable)"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        # --- Extract eigenvalues ---
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Geometric Classification requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        # --- Classify ---
        labels, counts = classify_points(eigenvalues, params)

        # --- Print class distribution ---
        n_points = len(eigenvalues)
        for label_id, name in CLASS_NAMES.items():
            count = counts[label_id]
            pct = 100.0 * count / n_points if n_points > 0 else 0.0
            print(f"  {name:12s}: {count:>8,d} points ({pct:5.1f}%)")

        # --- Build Clusters result ---
        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
