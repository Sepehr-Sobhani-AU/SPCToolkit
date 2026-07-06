"""
Generic crease-edge tracing — the intersection line of two surfaces.

A *crease edge* is where two surfaces meet at an angle (kerb top/bottom,
building corners, roof ridges, wall/floor joints). This service traces that
edge as a polyline, given the whole cloud and a single **seed point** the user
picked near the edge. It knows nothing about kerbs or buildings: those features
are *outcomes* of running this one brick (and, later, a separate junction brick
that joins edges at corners). See ``DECISIONS.md`` 2026-06-26 / 2026-06-28.

Method — a **local march** (one cell per step, each centred on the edge and
rotating to follow it):

1. Bootstrap: snap the picked point to the nearest cloud point, gather a ball
   around it, split the normals into two planes (2-means, sign-invariant), fit
   and intersect them → the initial edge heading. The selection only *locates*
   the edge; the code discovers which two planes form it.
2. March: at each step gather points in a box of ``edge_length`` (along the
   tangent) × ``perpendicular_size`` × ``perpendicular_size`` (across), centred
   on the current edge point and oriented to the local tangent; split into two
   planes by normal; fit each (shared RANSAC ``core/services/ransac.fit``);
   intersect; emit the point on that line nearest the box centre as a vertex.
3. Re-centre and rotate: the next box steps one ``edge_length`` along the local
   tangent and re-orients to it, so a single row of cells follows the crease —
   straight or curved — with the edge through each box's centre.
4. The march runs both directions from the seed and stops when a box no longer
   straddles two distinctly-oriented surfaces (the edge ended).

The user sets only ``edge_length``; ``perpendicular_size`` is suggested by
``suggest_perpendicular`` (how far to reach to capture both surfaces) and
pre-filled in the dialog. Per-point normals are *consumed* from upstream (PLY
``nx/ny/nz`` or the normal_estimation plugin), never computed here.

Deliberately **not** here (separate, deferred bricks per ``DECISIONS.md``):
corner resolution (extend-to-intersection of two edge polylines) and
clipped-segment output. One brick traces one crease.
"""

import numpy as np
from scipy.spatial import cKDTree

from core.services.ransac import fit
from core.entities.vector_feature import VectorFeature


_EPS = 1e-9

# Cap on points fed to a single plane fit. A planar patch is over-determined by
# a few hundred points; capping bounds RANSAC's per-iteration distance pass.
_MAX_POINTS_PER_SIDE = 256

# Minimum points each side of a box needs for a stable plane fit; below this the
# box no longer straddles two surfaces, so the march stops.
_MIN_POINTS_PER_SIDE = 6

# Stop the march if the edge tangent turns more than this between steps — a
# sharp bend usually means the box has wandered off the feature.
_MAX_ANGLE_DEG = 60.0

# Safety cap on steps per march direction.
_MAX_STEPS = 100_000

_EDGE_COLOR = np.array([1.0, 0.55, 0.0], dtype=np.float32)  # orange


class CreaseTracer:
    """
    Trace the intersection line of two surfaces, seeded by one picked point.

    Parameters:
        points: ``(N, 3)`` cloud points (the whole branch).
        normals: ``(N, 3)`` per-point normals, consumed from upstream.
        seed_point: ``(3,)`` a point near the edge (the user's pick). Only locates
            the edge; the code finds the two intersecting planes itself.
        edge_length: Box length along the edge — also the step (m). The only size
            the user sets. Smaller follows tighter curves and spaces vertices
            more finely.
        perpendicular_size: Box width across the edge (m), i.e. how far each cell
            reaches to capture both surfaces. Auto-suggested (see
            ``suggest_perpendicular``) and pre-filled in the dialog; the march
            uses whatever value it is given.
        min_dihedral_deg: Stop the march when a box's two fitted planes meet at
            less than this angle — they are one surface, so the edge has ended.
        ransac_threshold: Plane RANSAC inlier distance threshold (m).
        ransac_iterations: Max RANSAC hypotheses per plane fit.
        backend: RANSAC backend — ``"auto"`` (GPU when available), ``"cpu"``,
            or ``"gpu"``.
        seed: Optional RANSAC seed for reproducibility.
        record_debug: Record per-step geometry for the show-* overlays.
    """

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        seed_point: np.ndarray,
        edge_length: float = 0.3,
        perpendicular_size: float = 0.3,
        min_dihedral_deg: float = 20.0,
        ransac_threshold: float = 0.03,
        ransac_iterations: int = 100,
        backend: str = "auto",
        seed: int = None,
        record_debug: bool = False,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.seed_point = np.asarray(seed_point, dtype=np.float64).reshape(3)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {self.points.shape}")
        if self.normals.shape != self.points.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} does not match points "
                f"shape {self.points.shape}"
            )
        if edge_length <= 0 or perpendicular_size <= 0:
            raise ValueError("edge_length and perpendicular_size must be positive.")

        self.edge_length = float(edge_length)
        self.perpendicular_size = float(perpendicular_size)
        self.min_dihedral_deg = float(min_dihedral_deg)
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_iterations = int(ransac_iterations)
        self.backend = backend
        self.seed = seed

        self._kdtree = None

        # Debug geometry recorded during trace() for the show-* overlays.
        self.record_debug = bool(record_debug)
        self.debug_point_cell = None   # (N,) step id of the box a point fell in, -1 = unused
        self.debug_point_side = None   # (N,) cluster 0/1 per split point, -1 = unused
        self.debug_cells = []          # (center, R) oriented box per step
        self.debug_planes = []         # (vertex, n_a, n_b) per step

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def trace(self) -> np.ndarray:
        """
        Trace the crease and return ordered polyline vertices.

        Returns an ``(M, 3)`` array of vertices in march order along the crease.
        Empty ``(0, 3)`` if no crease was found at the seed.
        """
        self.debug_point_cell = np.full(len(self.points), -1, dtype=np.int64)
        self.debug_point_side = np.full(len(self.points), -1, dtype=np.int8)
        self.debug_cells = []
        self.debug_planes = []

        boot = self._bootstrap()
        if boot is None:
            return np.empty((0, 3), dtype=np.float64)
        seed_point, seed_dir, ref_axes = boot  # _bootstrap built self._kdtree

        forward = self._march(seed_point, seed_dir, ref_axes)
        backward = self._march(seed_point, -seed_dir, ref_axes)
        # Both marches emit the seed box first; drop the backward duplicate and
        # lay the backward steps (reversed) before the forward ones so the
        # vertices read in a single sweep along the crease.
        records = list(reversed(backward[1:])) + forward

        if self.record_debug:
            self._record_debug(records)

        if len(records) < 2:
            return np.empty((0, 3), dtype=np.float64)
        vertices = np.array([r["vertex"] for r in records], dtype=np.float64)
        return _dedup(vertices, self.edge_length * 0.25)

    # ------------------------------------------------------------------ #
    # Bootstrap                                                          #
    # ------------------------------------------------------------------ #

    def _bootstrap(self):
        """Seed the march from a local two-plane fit at the picked point.

        Returns ``(seed_point, seed_dir, ref_axes)`` — the snapped seed, the
        initial edge heading, and the two plane normals (for consistent
        side-colouring) — or ``None`` when the neighbourhood of the pick does not
        resolve into two distinctly-oriented planes (the pick wasn't near a
        crease, or ``perpendicular_size`` is too small).
        """
        self._kdtree = cKDTree(self.points)
        _, seed_idx = self._kdtree.query(self.seed_point)
        seed = self.points[seed_idx]

        cand = self._kdtree.query_ball_point(seed, self.perpendicular_size)
        if len(cand) < 2 * _MIN_POINTS_PER_SIDE:
            return None
        cand = np.asarray(cand, dtype=np.intp)
        labels = _split_two_planes_by_normal(self.normals[cand])
        side_a = cand[labels == 0]
        side_b = cand[labels == 1]
        if len(side_a) < _MIN_POINTS_PER_SIDE or len(side_b) < _MIN_POINTS_PER_SIDE:
            return None

        model_a = self._fit_plane(self.points[side_a])
        model_b = self._fit_plane(self.points[side_b])
        if model_a is None or model_b is None:
            return None
        n_a = _unit(model_a.normal)
        n_b = _unit(model_b.normal)
        if abs(float(n_a @ n_b)) > np.cos(np.radians(self.min_dihedral_deg)):
            return None

        line = _intersect_line(n_a, model_a.point, n_b, model_b.point)
        if line is None:
            return None
        _, direction = line
        return seed, direction, (n_a, n_b)

    # ------------------------------------------------------------------ #
    # March                                                              #
    # ------------------------------------------------------------------ #

    def _march(self, start, direction, ref_axes) -> list:
        """March from *start* along *direction*, one box per step.

        ``ref_axes`` anchors the side-colour labelling so both march directions
        agree at the seed. Returns per-step records (dicts) in traversal order.
        """
        half_len = self.edge_length / 2.0
        half_perp = self.perpendicular_size / 2.0
        search_radius = np.sqrt(half_len ** 2 + 2.0 * half_perp ** 2)  # box circumradius
        cos_dihedral = np.cos(np.radians(self.min_dihedral_deg))
        cos_bend = np.cos(np.radians(_MAX_ANGLE_DEG))

        tip = np.asarray(start, dtype=np.float64).copy()
        heading = _unit(direction)
        ref_a, ref_b = ref_axes
        out = []

        for _ in range(_MAX_STEPS):
            cand = self._kdtree.query_ball_point(tip, search_radius)
            if len(cand) < 2 * _MIN_POINTS_PER_SIDE:
                break
            cand = np.asarray(cand, dtype=np.intp)
            u2, u3 = _perp_basis(heading)
            rot = np.stack([heading, u2, u3])  # rows = box axes
            local = np.abs((self.points[cand] - tip) @ rot.T)
            in_box = (local[:, 0] <= half_len) & (local[:, 1] <= half_perp) & (local[:, 2] <= half_perp)
            box_idx = cand[in_box]
            if len(box_idx) < 2 * _MIN_POINTS_PER_SIDE:
                break

            # Split the box into two planes by normal; both must be present.
            labels = _split_two_planes_by_normal(self.normals[box_idx])
            side_a = box_idx[labels == 0]
            side_b = box_idx[labels == 1]
            if len(side_a) < _MIN_POINTS_PER_SIDE or len(side_b) < _MIN_POINTS_PER_SIDE:
                break  # only one surface in the box: the edge has ended

            model_a = self._fit_plane(self.points[side_a])
            model_b = self._fit_plane(self.points[side_b])
            if model_a is None or model_b is None:
                break
            n_a = _unit(model_a.normal)
            n_b = _unit(model_b.normal)
            if abs(float(n_a @ n_b)) > cos_dihedral:
                break  # near-parallel: one surface, not a crease

            line = _intersect_line(n_a, model_a.point, n_b, model_b.point)
            if line is None:
                break
            point_on_line, edge_dir = line
            if edge_dir @ heading < 0:
                edge_dir = -edge_dir
            if edge_dir @ heading < cos_bend:
                break  # sharp turn: box has likely wandered off the feature

            # Keep side 0 = the plane closest to the running reference, so the
            # two surfaces stay one colour each across steps and directions.
            if abs(float(n_a @ ref_a)) >= abs(float(n_a @ ref_b)):
                side = labels.astype(np.int8)
                ref_a, ref_b = n_a, n_b
            else:
                side = (1 - labels).astype(np.int8)
                ref_a, ref_b = n_b, n_a

            # Re-centre: the vertex is the edge point nearest the box centre.
            vertex = point_on_line + float((tip - point_on_line) @ edge_dir) * edge_dir
            ev2, ev3 = _perp_basis(edge_dir)
            out.append({
                "vertex": vertex,
                "frame": np.stack([edge_dir, ev2, ev3]),
                "box_idx": box_idx,
                "side": side,
                "n_a": n_a,
                "n_b": n_b,
            })

            # Step one cell along the local edge and re-orient to it.
            tip = vertex + self.edge_length * edge_dir
            heading = edge_dir

        return out

    def _fit_plane(self, points):
        model, _ = fit(
            _subsample(points, self.seed),
            "plane",
            threshold=self.ransac_threshold,
            max_iterations=self.ransac_iterations,
            min_inlier_ratio=0.0,  # this orchestrator owns its own gates
            seed=self.seed,
            backend=self.backend,
        )
        return model

    # ------------------------------------------------------------------ #
    # Debug                                                              #
    # ------------------------------------------------------------------ #

    def _record_debug(self, records):
        self.debug_cells = [(r["vertex"], r["frame"]) for r in records]
        self.debug_planes = [(r["vertex"], r["n_a"], r["n_b"]) for r in records]
        for step_no, r in enumerate(records):
            self.debug_point_cell[r["box_idx"]] = step_no
            self.debug_point_side[r["box_idx"]] = r["side"]


# --------------------------------------------------------------------------- #
# Perpendicular-size suggestion (for pre-filling the dialog)                   #
# --------------------------------------------------------------------------- #


def suggest_perpendicular(points, seed_xyz, target_points=120, cap=None):
    """Suggest a ``perpendicular_size`` so a box at the seed holds enough points.

    Pure and normals-free: the full width that brings ~``target_points`` points
    within reach of the seed, via an O(N) k-th-nearest distance (no KD-tree
    build), so it is cheap enough to call while the dialog opens. Returns
    ``None`` if there are too few points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return None
    seed_xyz = np.asarray(seed_xyz, dtype=np.float64).reshape(3)
    d2 = np.einsum("ij,ij->i", points - seed_xyz, points - seed_xyz)
    k = min(int(target_points), len(d2) - 1)
    radius = float(np.sqrt(np.partition(d2, k)[k]))
    width = 2.0 * radius  # full box width ~ twice the reach
    if cap is not None:
        width = min(width, float(cap))
    return width


# --------------------------------------------------------------------------- #
# Geometry helpers (pure)                                                      #
# --------------------------------------------------------------------------- #


def _split_two_planes_by_normal(normals: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Sign-invariant 2-means on per-point normals.

    Treats ``n`` and ``-n`` as the same orientation (plane normals carry a sign
    ambiguity), so the distance is ``1 - (a·b)^2``: assignment maximises
    ``|n·c|`` and each centroid is the dominant eigenvector of its cluster's
    normal scatter matrix. Returns a ``(N,)`` array of 0/1 cluster labels.
    """
    n = _unit_rows(np.asarray(normals, dtype=np.float64))
    if len(n) < 2:
        return np.zeros(len(n), dtype=np.intp)

    # Seed 0: the dominant orientation of all normals. Seed 1: the normal most
    # perpendicular to it (the best candidate for a second plane).
    c0 = _principal_axis(n)
    c1 = n[int(np.argmin(np.abs(n @ c0)))]
    centroids = np.stack([c0, _unit(c1)])

    labels = np.zeros(len(n), dtype=np.intp)
    for _ in range(max_iter):
        sims = np.abs(n @ centroids.T)            # (N, 2), sign-invariant
        new_labels = np.argmax(sims, axis=1).astype(np.intp)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            grp = n[labels == k]
            if len(grp) > 0:
                centroids[k] = _principal_axis(grp)
    return labels


def _intersect_line(n_a, p_a, n_b, p_b):
    """Intersection line of two planes as ``(point_on_line, unit_direction)``,
    or ``None`` if the planes are parallel."""
    n_a = _unit(n_a)
    n_b = _unit(n_b)
    direction = np.cross(n_a, n_b)
    norm = float(np.linalg.norm(direction))
    if norm < _EPS:
        return None
    direction = direction / norm

    # Solve [n_a; n_b; dir] x = [n_a·p_a, n_b·p_b, 0] for a point on the line.
    matrix = np.stack([n_a, n_b, direction])
    rhs = np.array([float(n_a @ p_a), float(n_b @ p_b), 0.0])
    try:
        point_on_line = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    return point_on_line, direction


def _dedup(vertices: np.ndarray, tol: float) -> np.ndarray:
    """Drop consecutive vertices closer than *tol* (collapses any seed overlap)."""
    if len(vertices) < 2:
        return vertices
    kept = [vertices[0]]
    for v in vertices[1:]:
        if np.linalg.norm(v - kept[-1]) >= tol:
            kept.append(v)
    return np.asarray(kept, dtype=np.float64)


def _principal_axis(pts: np.ndarray) -> np.ndarray:
    """Unit eigenvector of the largest covariance eigenvalue of *pts*."""
    cov = pts.T @ pts
    _, eigvecs = np.linalg.eigh(cov)
    return _unit(eigvecs[:, -1])


def _perp_basis(direction):
    """Two orthonormal vectors spanning the plane perpendicular to *direction*."""
    d = _unit(direction)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(d, ref))
    v = np.cross(d, u)
    return u, v


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    return vec / (np.linalg.norm(vec) + 1e-12)


def _unit_rows(vecs: np.ndarray) -> np.ndarray:
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


def _subsample(points: np.ndarray, seed) -> np.ndarray:
    """Randomly cap a box-side's points at ``_MAX_POINTS_PER_SIDE``."""
    if len(points) <= _MAX_POINTS_PER_SIDE:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), _MAX_POINTS_PER_SIDE, replace=False)
    return points[idx]


# --------------------------------------------------------------------------- #
# Renderable output                                                            #
# --------------------------------------------------------------------------- #


def vertices_to_polyline_feature(vertices, color=None, closed: bool = False):
    """Build a render-only polyline ``VectorFeature`` from ordered vertices.

    Returns ``None`` if there are fewer than two vertices (nothing to draw).
    The result is added by the caller as an ordinary, controllable tree branch.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) < 2:
        return None
    dims = (verts.max(axis=0) - verts.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type="crease_edge",
        geometry_type="polyline",
        geometry={"vertices": verts, "closed": bool(closed)},
        transform_matrix=np.eye(4),
        dimensions=dims,
        color=_EDGE_COLOR if color is None else np.asarray(color, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# Debug overlays — one renderable wireframe branch per processing stage.       #
#                                                                              #
# trace() records, per march step, the oriented box, the two fitted planes,    #
# and per-point box/side ids. These helpers turn that into render-only         #
# VectorFeature wireframes so a plugin can add each stage ("cells", "planes",  #
# "normals") as an ordinary controllable branch. Voxel POINTS are coloured per #
# box via debug_point_cell, which the plugin turns into a Clusters branch.     #
# All helpers are pure: no GUI access.                                         #
# --------------------------------------------------------------------------- #

_CELL_COLOR = np.array([0.1, 0.9, 0.9], dtype=np.float32)       # cyan
_PLANE_COLOR = np.array([1.0, 0.2, 1.0], dtype=np.float32)      # magenta
_NORMAL_A_COLOR = np.array([1.0, 0.85, 0.1], dtype=np.float32)  # yellow (cluster A)
_NORMAL_B_COLOR = np.array([0.2, 0.6, 1.0], dtype=np.float32)   # blue   (cluster B)

# Box corners centred on the origin (box-frame), scaled by the box dims at draw.
_CUBE_CORNERS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.float64,
) - 0.5
_CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
               (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]


def cells_to_vector_feature(cells, dims, color=None):
    """Wireframe boxes for the per-step cells.

    ``cells`` is a list of ``(center, R)`` where ``R`` rows are the box's local
    axes (edge tangent + two perpendiculars). ``dims`` is ``(edge_length,
    perpendicular_size, perpendicular_size)`` — the box is drawn centred on the
    edge and rotated to follow it.
    """
    if not cells:
        return None
    verts, edges = [], []
    corner_offsets = _CUBE_CORNERS * np.asarray(dims, dtype=np.float64)
    for center, rot in cells:
        corners = corner_offsets @ rot + np.asarray(center, dtype=np.float64)
        base = len(verts)
        verts.extend(corners)
        for a, b in _CUBE_EDGES:
            edges.append([base + a, base + b])
    return _wireframe_vector_feature(
        "debug_cells", verts, edges, _CELL_COLOR if color is None else color
    )


def planes_to_vector_feature(planes, size, color=None):
    """Wireframe square patch (+ a short normal stub) for each fitted plane."""
    if not planes:
        return None
    verts, edges = [], []
    half = size * 0.5
    for centre, n_a, n_b in planes:
        for normal in (n_a, n_b):
            u, v = _perp_basis(normal)
            base = len(verts)
            verts.extend([
                centre - half * u - half * v,
                centre + half * u - half * v,
                centre + half * u + half * v,
                centre - half * u + half * v,
            ])
            edges += [[base, base + 1], [base + 1, base + 2],
                      [base + 2, base + 3], [base + 3, base]]
            stub = len(verts)
            verts.extend([np.asarray(centre, dtype=np.float64),
                          np.asarray(centre, dtype=np.float64) + _unit(normal) * half])
            edges.append([stub, stub + 1])
    return _wireframe_vector_feature(
        "debug_planes", verts, edges, _PLANE_COLOR if color is None else color
    )


def normals_to_vector_feature(points, normals, scale, color):
    """One short line segment per point, from the point along its normal."""
    if len(points) == 0:
        return None
    units = _unit_rows(np.asarray(normals, dtype=np.float64))
    verts, edges = [], []
    for p, nrm in zip(np.asarray(points, dtype=np.float64), units):
        base = len(verts)
        verts.extend([p, p + nrm * scale])
        edges.append([base, base + 1])
    return _wireframe_vector_feature("debug_normals", verts, edges, color)


def _wireframe_vector_feature(symbol_type, verts, edges, color):
    vertices = np.asarray(verts, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int32)
    dims = (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type=symbol_type,
        geometry_type="mesh",
        geometry={"vertices": vertices, "faces": [], "edges": edges},
        transform_matrix=np.eye(4),
        dimensions=dims,
        color=np.asarray(color, dtype=np.float32),
    )


def debug_vector_features(tracer, show_cells, show_planes, show_normals):
    """Return ``[(branch_name, VectorFeature), ...]`` for the requested wireframe
    overlays, built from a traced ``CreaseTracer``'s recorded debug geometry."""
    out = []
    edge_len = tracer.edge_length
    perp = tracer.perpendicular_size
    if show_cells:
        vf = cells_to_vector_feature(tracer.debug_cells, (edge_len, perp, perp))
        if vf is not None:
            out.append(("debug_cells", vf))
    if show_planes:
        vf = planes_to_vector_feature(tracer.debug_planes, perp)
        if vf is not None:
            out.append(("debug_planes", vf))
    if show_normals and tracer.debug_point_side is not None:
        # One branch per cluster so the two surfaces' normals read in two colours.
        side = tracer.debug_point_side
        scale = 0.4 * min(edge_len, perp)
        for value, name, col in (
            (0, "debug_normals_a", _NORMAL_A_COLOR),
            (1, "debug_normals_b", _NORMAL_B_COLOR),
        ):
            mask = side == value
            vf = normals_to_vector_feature(
                tracer.points[mask], tracer.normals[mask], scale, col
            )
            if vf is not None:
                out.append((name, vf))
    return out
