"""
Multi-scale vegetation classification plugin.

Computes eigenvalues at multiple neighborhood sizes (k values) and classifies
points as Vegetation vs Non-Vegetation based on how close their three eigenvalues
are to each other across scales.

Closeness metric:  min(λ₁/λ₂, λ₂/λ₃, λ₃/λ₁)
  - 1.0 = all three eigenvalues identical
  - Near 0 = eigenvalues very different

Real vegetation scatters uniformly at all scales, producing nearly equal eigenvalues.
Noise and sparse outliers only appear isotropic at isolated scales.

Operates directly on a raw PointCloud — no pre-computed eigenvalues needed.
Returns standard Clusters.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters


CLASS_NAMES = {0: "Vegetation", 1: "Non-Vegetation"}
CLASS_COLORS = {
    "Vegetation":     np.array([0.2, 0.8, 0.2], dtype=np.float32),
    "Non-Vegetation": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class VegetationClassificationPlugin(Plugin):
    """Classify points as Vegetation or Non-Vegetation using multi-scale eigenvalue closeness."""

    def get_name(self) -> str:
        return "vegetation_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "formula": {
                "type": "info",
                "default": (
                    "closeness = min(\u03bb\u2081/\u03bb\u2082, \u03bb\u2082/\u03bb\u2083, \u03bb\u2083/\u03bb\u2081)\n"
                    "1.0 = all three eigenvalues identical  |  0.0 = very different\n"
                    "Points above threshold at enough scales \u2192 Vegetation"
                ),
                "label": "Formula",
            },
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
                "description": "Smooth eigenvalues per scale before computing closeness"
            },
            "closeness_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Closeness Threshold",
                "description": "Minimum eigenvalue closeness to count as passing at a given scale"
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
        threshold = params["closeness_threshold"]
        min_scales = params["min_scales"]
        smooth = params["smooth"]

        from config.config import global_variables

        n_points = len(point_cloud.points)
        epsilon = 1e-10

        # Compute eigenvalue closeness at each scale and vote
        vote_count = np.zeros(n_points, dtype=np.int32)
        for i, k in enumerate(k_values):
            global_variables.global_progress = (int(i * 30), f"Computing eigenvalues at k={k} (scale {i + 1}/3)...")
            eigenvalues = point_cloud.get_eigenvalues(k, smooth=smooth)

            # closeness = min(λ₁/λ₂, λ₂/λ₃, λ₃/λ₁)
            l1 = eigenvalues[:, 0]
            l2 = eigenvalues[:, 1]
            l3 = eigenvalues[:, 2]

            r12 = l1 / np.maximum(l2, epsilon)
            r23 = l2 / np.maximum(l3, epsilon)
            r31 = l3 / np.maximum(l1, epsilon)

            closeness = np.nan_to_num(np.minimum(np.minimum(r12, r23), r31))

            vote_count += (closeness > threshold).astype(np.int32)

        # Classify based on multi-scale consensus
        global_variables.global_progress = (90, "Classifying points...")
        veg_mask = vote_count >= min_scales

        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Vegetation (1)
        labels[veg_mask] = 0  # Vegetation

        n_veg = int(np.sum(veg_mask))
        pct = 100.0 * n_veg / n_points if n_points > 0 else 0.0

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
