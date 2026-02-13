"""
Multi-scale vegetation classification plugin.

Computes eigenvalues at multiple neighborhood sizes (k values) and classifies
points as Vegetation vs Non-Vegetation based on multi-scale sphericity consensus.
Real vegetation scatters uniformly at all scales, while noise and sparse outliers
only appear spherical at isolated scales.

Operates directly on a raw PointCloud — no pre-computed eigenvalues needed.
Returns standard Clusters.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {0: "Vegetation", 1: "Non-Vegetation"}
CLASS_COLORS = {
    "Vegetation":     np.array([0.2, 0.8, 0.2], dtype=np.float32),
    "Non-Vegetation": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class VegetationClassificationPlugin(Plugin):
    """Classify points as Vegetation or Non-Vegetation using multi-scale sphericity."""

    def get_name(self) -> str:
        return "vegetation_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k_small": {
                "type": "int",
                "default": 15,
                "min": 5,
                "max": 50,
                "label": "K Small",
                "description": "Small-scale neighborhood size"
            },
            "k_medium": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 80,
                "label": "K Medium",
                "description": "Medium-scale neighborhood size"
            },
            "k_large": {
                "type": "int",
                "default": 60,
                "min": 20,
                "max": 150,
                "label": "K Large",
                "description": "Large-scale neighborhood size"
            },
            "smooth": {
                "type": "bool",
                "default": True,
                "label": "Smooth Eigenvalues",
                "description": "Smooth eigenvalues per scale before computing features"
            },
            "sphericity_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Sphericity Threshold",
                "description": "Sphericity cutoff per scale — points above this count as passing"
            },
            "min_scales": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 3,
                "label": "Min Scales Passing",
                "description": "How many scales must exceed threshold (3=strict, 2=majority, 1=any)"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Vegetation Classification requires a PointCloud. "
                "Select a point cloud node."
            )

        k_values = [params["k_small"], params["k_medium"], params["k_large"]]
        threshold = params["sphericity_threshold"]
        min_scales = params["min_scales"]
        smooth = params["smooth"]

        n_points = len(point_cloud.points)
        utils = EigenvalueUtils()

        # Compute sphericity pass/fail at each scale
        vote_count = np.zeros(n_points, dtype=np.int32)
        for i, k in enumerate(k_values):
            print(f"  Computing eigenvalues at k={k} (scale {i + 1}/3)...")
            eigenvalues = point_cloud.get_eigenvalues(k, smooth=smooth)
            features = utils.compute_geometric_features(eigenvalues)
            vote_count += (features["sphericity"] > threshold).astype(np.int32)

        # Classify based on multi-scale consensus
        veg_mask = vote_count >= min_scales

        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Vegetation (1)
        labels[veg_mask] = 0  # Vegetation

        n_veg = int(np.sum(veg_mask))
        pct = 100.0 * n_veg / n_points if n_points > 0 else 0.0
        print(f"  Vegetation:     {n_veg:>8,d} points ({pct:5.1f}%)")
        print(f"  Non-Vegetation: {n_points - n_veg:>8,d} points ({100.0 - pct:5.1f}%)")

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
