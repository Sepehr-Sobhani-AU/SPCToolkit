"""
Iterative cable-tracing algorithm using RANSAC line fitting and cylindrical region growing.

Given seed points on a cable, traces the cable in both directions until a pole
(large angle change) or empty space is reached.
"""

import numpy as np
from scipy.spatial import cKDTree

from core.services.ransac import RANSAC, LineModel3D


class PowerLineTracer:
    """
    Traces power-line cables through a point cloud from seed points.

    Parameters:
        all_points: (N, 3) array — the full point cloud.
        kdtree: Pre-built cKDTree for *all_points*.
        cylinder_radius: Search cylinder radius in metres.
        cylinder_length: Search cylinder length per step in metres.
        min_points: Stop growth if fewer points found in cylinder.
        max_angle_deg: Max direction change (degrees) before stopping (pole detection).
        ransac_threshold: RANSAC inlier distance threshold in metres.
    """

    def __init__(
        self,
        all_points: np.ndarray,
        kdtree: cKDTree,
        cylinder_radius: float = 0.5,
        cylinder_length: float = 5.0,
        min_points: int = 5,
        max_angle_deg: float = 15.0,
        ransac_threshold: float = 0.3,
    ):
        self.all_points = all_points
        self.kdtree = kdtree
        self.cylinder_radius = cylinder_radius
        self.cylinder_length = cylinder_length
        self.min_points = min_points
        self.max_angle_cos = np.cos(np.radians(max_angle_deg))
        self.ransac_threshold = ransac_threshold

    def trace_cable(self, seed_indices: np.ndarray) -> np.ndarray:
        """
        Trace a single cable starting from *seed_indices*.

        Returns an array of point indices belonging to the cable
        (indices into ``self.all_points``).
        """
        seed_pts = self.all_points[seed_indices]

        # Fit initial direction from seed points
        mask, model = RANSAC.run(
            seed_pts,
            LineModel3D(),
            distance_threshold=self.ransac_threshold,
            max_iterations=200,
            min_inlier_ratio=0.3,
        )
        if model is None:
            return seed_indices

        direction = model.direction

        # Project seeds onto the fitted line to find endpoints
        projections = np.dot(seed_pts - model.point, direction)
        min_proj_idx = int(np.argmin(projections))
        max_proj_idx = int(np.argmax(projections))

        start_a = seed_pts[max_proj_idx]
        start_b = seed_pts[min_proj_idx]

        # Grow in both directions
        collected = set(int(i) for i in seed_indices)
        collected |= self._grow_one_direction(start_a, direction, collected)
        collected |= self._grow_one_direction(start_b, -direction, collected)

        return np.array(sorted(collected), dtype=np.intp)

    def _grow_one_direction(
        self,
        tip: np.ndarray,
        direction: np.ndarray,
        existing: set,
    ) -> set:
        """
        Iteratively grow a cable from *tip* along *direction*.

        Returns a set of new point indices collected during growth.
        """
        collected = set()
        half_len = self.cylinder_length / 2.0
        search_radius = np.sqrt(self.cylinder_radius ** 2 + half_len ** 2)
        max_steps = 500

        current_tip = tip.copy()
        current_dir = direction.copy()

        for _ in range(max_steps):
            # Place search centre ahead of current tip
            centre = current_tip + half_len * current_dir

            # Ball query
            candidate_idx = self.kdtree.query_ball_point(centre, search_radius)
            if not candidate_idx:
                break

            candidate_idx = np.array(candidate_idx, dtype=np.intp)
            pts = self.all_points[candidate_idx]

            # Filter to actual cylinder
            vecs = pts - current_tip
            along = vecs @ current_dir  # projection onto axis

            # Forward only + within cylinder length
            length_mask = (along > 0) & (along < self.cylinder_length)

            # Perpendicular distance within radius
            proj_on_axis = np.outer(along, current_dir)
            perp = vecs - proj_on_axis
            perp_dist = np.linalg.norm(perp, axis=1)
            radius_mask = perp_dist < self.cylinder_radius

            valid_mask = length_mask & radius_mask
            if np.sum(valid_mask) < self.min_points:
                break

            valid_idx = candidate_idx[valid_mask]
            valid_pts = pts[valid_mask]

            # RANSAC line fit on cylinder contents
            inlier_mask, new_model = RANSAC.run(
                valid_pts,
                LineModel3D(),
                distance_threshold=self.ransac_threshold,
                max_iterations=100,
                min_inlier_ratio=0.2,
            )
            if new_model is None:
                break

            new_dir = new_model.direction
            # Ensure new direction points forward (same general direction)
            if np.dot(new_dir, current_dir) < 0:
                new_dir = -new_dir

            # Pole detection: stop if angle change is too large
            cos_angle = np.dot(new_dir, current_dir)
            if cos_angle < self.max_angle_cos:
                break

            # Collect inlier points
            inlier_idx = valid_idx[inlier_mask]
            for idx in inlier_idx:
                collected.add(int(idx))

            # Advance tip to farthest inlier along new direction
            inlier_pts = self.all_points[inlier_idx]
            proj = np.dot(inlier_pts - current_tip, new_dir)
            farthest = int(np.argmax(proj))
            current_tip = inlier_pts[farthest]
            current_dir = new_dir

        return collected
