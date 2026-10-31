"""
Filter a point cloud down to edge points using curvature variation.

For each point, we compute the change-of-curvature (Pauly et al. surface
variation) sigma = lambda_1 / (lambda_1 + lambda_2 + lambda_3) from the
already-computed eigenvalues, then measure the standard deviation of sigma
over each point's k nearest neighbors. Points where the neighborhood sigma-std
exceeds the threshold sit on geometric discontinuities (corners, creases,
silhouettes) and are kept.

Requires eigenvalues to be already computed on the selected node
(run Compute Eigenvalues first).
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from config.config import global_variables


class CurvatureEdgeFilterPlugin(Plugin):
    """Filter a point cloud to keep only points on geometric edges."""

    def get_name(self) -> str:
        return "curvature_edge_filter"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 200,
                "label": "Neighbors (k)",
                "description": "Neighborhood size for curvature-variance computation"
            },
            "use_percentile": {
                "type": "bool",
                "default": True,
                "label": "Percentile Threshold",
                "description": "If on, threshold is a percentile of neighborhood std; if off, threshold is an absolute value"
            },
            "percentile": {
                "type": "float",
                "default": 90.0,
                "min": 50.0,
                "max": 99.9,
                "label": "Percentile",
                "description": "Keep points whose curvature std is in the top (100 - percentile)% (used when Percentile Threshold is on)"
            },
            "std_threshold": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 1.0,
                "label": "Absolute Std Threshold",
                "description": "Absolute curvature-std cutoff (used when Percentile Threshold is off)"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Curvature Edge Filter requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        # lambda_1 <= lambda_2 <= lambda_3 (ascending)
        total = eigenvalues.sum(axis=1)
        sigma = np.divide(
            eigenvalues[:, 0], total,
            out=np.zeros(len(eigenvalues), dtype=np.float32),
            where=total > 0.0,
        )

        registry = global_variables.global_backend_registry
        if registry is None:
            raise RuntimeError(
                "Backend registry unavailable; cannot run GPU/CPU KNN query."
            )
        knn_backend = registry.get_knn()

        k = int(params["k_neighbors"])
        points_f32 = point_cloud.points.astype(np.float32, copy=False)
        # k + 1 because query includes the point itself as its own nearest neighbor
        _, indices = knn_backend.query(points_f32, k=k + 1)

        # Neighborhood std of sigma: gather sigma over each point's neighbors
        neighbor_sigma = sigma[indices]          # (N, k+1)
        curvature_std = neighbor_sigma.std(axis=1).astype(np.float32)

        if params["use_percentile"]:
            cutoff = float(np.percentile(curvature_std, params["percentile"]))
        else:
            cutoff = float(params["std_threshold"])

        mask = curvature_std > cutoff

        n_total = len(mask)
        n_edges = int(mask.sum())
        pct = 100.0 * n_edges / n_total if n_total > 0 else 0.0
        print(f"  Curvature-std cutoff: {cutoff:.6f}")
        print(f"  Edge points:     {n_edges:>8,d} / {n_total:,d} ({pct:5.1f}%)")

        subset = point_cloud.get_subset(mask, inplace=False)
        if isinstance(subset, PointCloud) and len(subset.points) == n_edges:
            subset.add_attribute("curvature_std", curvature_std[mask])

        return subset, "point_cloud", [data_node.uid]
