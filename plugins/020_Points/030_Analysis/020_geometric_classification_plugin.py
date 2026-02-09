"""
Plugin for classifying points into geometric shape classes based on eigenvalue features.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
After reconstruction by AnalysisExecutor, eigenvalues are in point_cloud.attributes['eigenvalues'].

Classifies each point into one of 5 classes:
- Planar: High planarity, low change-of-curvature (walls, ground, roofs)
- Linear: High linearity (cables, edges, poles)
- Cylindrical: Moderate everything — catch-all (tree trunks, pipes)
- Edge: High change-of-curvature (sharp transitions)
- Sparse: Extreme total_variance (isolated or noisy points)
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
    2: "Cylindrical",
    3: "Edge",
    4: "Sparse",
}

CLASS_COLORS = {
    "Planar":      np.array([0.2, 0.5, 0.9], dtype=np.float32),
    "Linear":      np.array([0.9, 0.1, 0.1], dtype=np.float32),
    "Cylindrical": np.array([0.1, 0.8, 0.3], dtype=np.float32),
    "Edge":        np.array([1.0, 0.75, 0.0], dtype=np.float32),
    "Sparse":      np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class GeometricClassificationPlugin(Plugin):
    """
    Classify each point into a geometric shape class based on eigenvalue features.

    Decision tree (cascading, vectorized):
        1. SPARSE:      total_variance in bottom/top sparse_percentile
        2. LINEAR:      linearity > linear_threshold
        3. PLANAR:      planarity > planar_threshold AND coc < coc_edge_threshold
        4. EDGE:        coc > coc_edge_threshold
        5. CYLINDRICAL: everything else
    """

    def get_name(self) -> str:
        return "geometric_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
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
                "description": "Planarity cutoff — points above this (with low curvature) are Planar"
            },
            "coc_edge_threshold": {
                "type": "float",
                "default": 0.10,
                "min": 0.0,
                "max": 0.5,
                "label": "Edge Curvature Threshold",
                "description": "Change-of-curvature cutoff — points above this are classified as Edge"
            },
            "sparse_percentile": {
                "type": "float",
                "default": 5.0,
                "min": 0.5,
                "max": 25.0,
                "label": "Sparse Percentile",
                "description": "Bottom/top N% of total variance are classified as Sparse"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        # --- Extract eigenvalues ---
        # AnalysisExecutor reconstructs non-PointCloud nodes before calling execute(),
        # so eigenvalues arrive in point_cloud.attributes['eigenvalues'] (Nx3 array).
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

        n_points = len(eigenvalues)

        # --- Compute geometric features ---
        features = EigenvalueUtils().compute_geometric_features(eigenvalues)
        planarity = features["planarity"]
        linearity = features["linearity"]
        total_variance = features["total_variance"]

        # Change of curvature: λ₁ / (λ₁ + λ₂ + λ₃)
        epsilon = 1e-10
        sum_eigenvalues = eigenvalues[:, 0] + eigenvalues[:, 1] + eigenvalues[:, 2]
        coc = eigenvalues[:, 0] / np.maximum(sum_eigenvalues, epsilon)

        # --- Extract parameters ---
        linear_thresh = params["linear_threshold"]
        planar_thresh = params["planar_threshold"]
        coc_edge_thresh = params["coc_edge_threshold"]
        sparse_pct = params["sparse_percentile"]

        # --- Cascading classification ---
        labels = np.full(n_points, 2, dtype=np.int32)  # Default: Cylindrical
        classified = np.zeros(n_points, dtype=bool)

        # 1. Sparse: extreme total_variance (bottom or top percentile)
        lo = np.percentile(total_variance, sparse_pct)
        hi = np.percentile(total_variance, 100.0 - sparse_pct)
        sparse_mask = (total_variance < lo) | (total_variance > hi)
        labels[sparse_mask] = 4
        classified |= sparse_mask

        # 2. Linear: high linearity
        linear_mask = (~classified) & (linearity > linear_thresh)
        labels[linear_mask] = 1
        classified |= linear_mask

        # 3. Planar: high planarity AND low change-of-curvature
        planar_mask = (~classified) & (planarity > planar_thresh) & (coc < coc_edge_thresh)
        labels[planar_mask] = 0
        classified |= planar_mask

        # 4. Edge: high change-of-curvature
        edge_mask = (~classified) & (coc > coc_edge_thresh)
        labels[edge_mask] = 3
        classified |= edge_mask

        # 5. Cylindrical: everything else (already default)

        # --- Print class distribution ---
        for label_id, name in CLASS_NAMES.items():
            count = int(np.sum(labels == label_id))
            pct = 100.0 * count / n_points if n_points > 0 else 0.0
            print(f"  {name:12s}: {count:>8,d} points ({pct:5.1f}%)")

        # --- Build Clusters result ---
        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
