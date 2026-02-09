"""
Plugin for classifying points as Cylindrical or Non-Cylindrical based on eigenvalue features.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
Produces 2 clusters: Cylindrical (both linearity AND planarity above thresholds) and Non-Cylindrical.
Two independent thresholds allow asymmetric tuning of the linearity-planarity boundary.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {0: "Cylindrical", 1: "Non-Cylindrical"}
CLASS_COLORS = {
    "Cylindrical":     np.array([0.0, 0.8, 0.8], dtype=np.float32),
    "Non-Cylindrical": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class CylindricalClassificationPlugin(Plugin):
    """Classify each point as Cylindrical or Non-Cylindrical based on eigenvalue linearity and planarity."""

    def get_name(self) -> str:
        return "cylindrical_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "linearity_threshold": {
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "label": "Linearity Threshold",
                "description": "Points must have linearity above this to be Cylindrical"
            },
            "planarity_threshold": {
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "label": "Planarity Threshold",
                "description": "Points must have planarity above this to be Cylindrical"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Cylindrical Classification requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        features = EigenvalueUtils().compute_geometric_features(eigenvalues)
        match_mask = (
            (features["linearity"] > params["linearity_threshold"]) &
            (features["planarity"] > params["planarity_threshold"])
        )

        n_points = len(eigenvalues)
        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Cylindrical (1)
        labels[match_mask] = 0  # Cylindrical

        n_match = int(np.sum(match_mask))
        pct = 100.0 * n_match / n_points if n_points > 0 else 0.0
        print(f"  Cylindrical:     {n_match:>8,d} points ({pct:5.1f}%)")
        print(f"  Non-Cylindrical: {n_points - n_match:>8,d} points ({100.0 - pct:5.1f}%)")

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
