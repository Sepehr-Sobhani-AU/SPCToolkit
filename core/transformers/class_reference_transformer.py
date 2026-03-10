"""
Transformer for filtering a point cloud to points belonging to a specific class.

Uses the cluster_labels attribute (set by ClustersTransformer) to create a boolean
mask, then returns the subset colored with the class color.
"""

import numpy as np

from core.entities.point_cloud import PointCloud
from core.entities.class_reference import ClassReference


class ClassReferenceTransformer:

    def __init__(self, point_cloud: PointCloud, class_reference: ClassReference):
        self.point_cloud = point_cloud
        self.class_reference = class_reference

    def execute(self) -> PointCloud:
        cluster_labels = self.point_cloud.attributes.get("cluster_labels")
        if cluster_labels is None:
            raise ValueError(
                "No 'cluster_labels' attribute found in point cloud. "
                "ClassReferenceTransformer requires a parent ClustersTransformer step."
            )

        mask = np.isin(cluster_labels, self.class_reference.cluster_ids)
        subset = self.point_cloud.get_subset(mask=mask)

        # Apply uniform class color
        n_points = len(subset.points)
        subset.colors = np.tile(self.class_reference.color, (n_points, 1))

        return subset
