"""
Plugin for classifying points as Sparse or Non-Sparse based on eigenvalue total variance.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
Produces 2 clusters: Sparse (extreme total variance outliers) and Non-Sparse (everything else).
Points with total variance below the low percentile or above the high percentile are Sparse.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {0: "Sparse", 1: "Non-Sparse"}
CLASS_COLORS = {
    "Sparse":     np.array([0.5, 0.5, 0.5], dtype=np.float32),
    "Non-Sparse": np.array([0.3, 0.7, 0.3], dtype=np.float32),
}


class SparseClassificationPlugin(Plugin):
    """Classify each point as Sparse or Non-Sparse based on total variance percentile outliers."""

    def get_name(self) -> str:
        return "sparse_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
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
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Sparse Classification requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        features = EigenvalueUtils().compute_geometric_features(eigenvalues)
        total_variance = features["total_variance"]
        sparse_pct = params["sparse_percentile"]

        n_points = len(eigenvalues)

        if sparse_pct > 0.0:
            lo = np.percentile(total_variance, sparse_pct)
            hi = np.percentile(total_variance, 100.0 - sparse_pct)
            match_mask = (total_variance < lo) | (total_variance > hi)
        else:
            match_mask = np.zeros(n_points, dtype=bool)

        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Sparse (1)
        labels[match_mask] = 0  # Sparse

        n_match = int(np.sum(match_mask))
        pct = 100.0 * n_match / n_points if n_points > 0 else 0.0
        print(f"  Sparse:     {n_match:>8,d} points ({pct:5.1f}%)")
        print(f"  Non-Sparse: {n_points - n_match:>8,d} points ({100.0 - pct:5.1f}%)")

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
