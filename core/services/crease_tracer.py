"""
Generic crease-edge tracing — the intersection line of two surfaces.

A *crease edge* is where two surfaces meet at an angle (kerb top/bottom,
building corners, roof ridges, wall/floor joints). This service traces that
edge as a polyline from a single user-selected swath that straddles **both**
surfaces and the edge between them. It knows nothing about kerbs or buildings:
those features are *outcomes* of running this one brick (and, later, a separate
junction brick that joins edges at corners). See ``DECISIONS.md`` 2026-06-26 /
2026-06-28.

Method — a **local march** (one cell per step, each centred on the edge and
rotating to follow it):

1. Bootstrap: a global 2-means split of all the swath's normals into two plane
   orientations gives the edge direction ``nA x nB`` and a seed point on the
   edge (intersection of the two global planes, nearest the swath centre).
2. March: at each step, gather points in one ``cell_size`` cube centred on the
   current edge point and oriented to the local edge tangent; split them into
   two planes by **2-means on their normals** (sign-invariant); fit a plane to
   each side (shared RANSAC engine ``core/services/ransac.fit``); intersect the
   two planes; emit the point on that line nearest the cube centre as a vertex.
3. Re-centre and rotate: the next cube is placed one cell along the *local* edge
   tangent and re-oriented to it, so a single row of cells follows the crease —
   straight or curved — with the edge through each cube's centre.
4. The march runs both directions from the seed and stops when a cube no longer
   straddles two distinctly-oriented surfaces (the edge ended).

Per-point normals are *consumed* from upstream (PLY ``nx/ny/nz`` or the
normal_estimation plugin), never computed here.

Deliberately **not** here (separate, deferred bricks per ``DECISIONS.md``):
corner resolution (extend-to-intersection of two edge polylines) and
clipped-segment output. One brick traces one crease.
"""

import numpy as np
from scipy.spatial import cKDTree

from core.services.ransac import fit
from core.entities.vector_feature import VectorFeature


_EPS = 1e-9

# Cap on points fed to a single plane fit. A planar patch in one cell is
# over-determined by a few hundred points; capping bounds RANSAC's per-iteration
# distance pass, matching the candidate cap surface_region_growing uses.
_MAX_POINTS_PER_SIDE = 256

# Stop the march if the edge tangent turns more than this between steps — a
# sharp bend usually means the cube has wandered off the feature.
_MAX_ANGLE_DEG = 60.0

# Safety cap on steps per march direction.
_MAX_STEPS = 100_000

_EDGE_COLOR = np.array([1.0, 0.55, 0.0], dtype=np.float32)  # orange


class CreaseTracer:
    """
    Trace the intersection line of two surfaces through a swath of points.

    Parameters:
        points: ``(N, 3)`` swath points covering both surfaces and the edge.
        normals: ``(N, 3)`` per-point normals, consumed from upstream. Used to
            split each cell's points into two planes and to seed the march.
        cell_size: Cube edge length (m) — also the step length along the edge.
            One cube per step is centred on the edge; smaller follows tighter
            curves and spaces vertices more finely, but each cube then holds
            fewer points to fit two planes from.
        min_points_per_cell: Stop the march if a cube holds fewer points.
        min_dihedral_deg: Stop the march when a cube's two fitted planes meet at
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
        cell_size: float = 0.3,
        min_points_per_cell: int = 10,
        min_dihedral_deg: float = 20.0,
        ransac_threshold: float = 0.03,
        ransac_iterations: int = 100,
        backend: str = "auto",
        seed: int = None,
        record_debug: bool = False,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {self.points.shape}")
        if self.normals.shape != self.points.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} does not match points "
                f"shape {self.points.shape}"
            )
        if cell_size <= 0:
            raise ValueError("cell_size must be positive.")

        self.cell_size = float(cell_size)
        self.min_points_per_cell = int(min_points_per_cell)
        self.min_dihedral_deg = float(min_dihedral_deg)
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_iterations = int(ransac_iterations)
        self.backend = backend
        self.seed = seed

        self._kdtree = None
        self._global_axes = None       # (n_a, n_b) global plane orientations

        # Debug geometry recorded during trace() for the show-* overlays, so a
        # plugin can add each stage as an ordinary controllable branch.
        self.record_debug = bool(record_debug)
        self.debug_point_cell = None   # (N,) step id of the cube a point fell in, -1 = unused
        self.debug_point_side = None   # (N,) global cluster 0/1 per split point, -1 = unused
        self.debug_cells = []          # (center, R) oriented cube per step
        self.debug_planes = []         # (vertex, n_a, n_b) per step

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def trace(self) -> np.ndarray:
        """
        Trace the crease and return ordered polyline vertices.

        Returns an ``(M, 3)`` array of vertices in march order along the crease.
        Empty ``(0, 3)`` if no crease was found.
        """
        self.debug_point_cell = np.full(len(self.points), -1, dtype=np.int64)
        self.debug_point_side = np.full(len(self.points), -1, dtype=np.int8)
        self.debug_cells = []
        self.debug_planes = []

        boot = self._bootstrap()
        if boot is None:
            return np.empty((0, 3), dtype=np.float64)
        seed_point, seed_dir = boot  # _bootstrap built self._kdtree

        forward = self._march(seed_point, seed_dir)
        backward = self._march(seed_point, -seed_dir)
        # Both marches emit the seed cube first; drop the backward duplicate and
        # lay the backward steps (reversed) before the forward ones so the
        # vertices read in a single sweep along the crease.
        records = list(reversed(backward[1:])) + forward

        if self.record_debug:
            self._record_debug(records)

        if len(records) < 2:
            return np.empty((0, 3), dtype=np.float64)
        vertices = np.array([r["vertex"] for r in records], dtype=np.float64)
        return _dedup(vertices, self.cell_size * 0.25)

    # ------------------------------------------------------------------ #
    # Bootstrap                                                          #
    # ------------------------------------------------------------------ #

    def _bootstrap(self):
        """Seed the march from a *global* two-plane fit of the whole swath.

        Returns ``(seed_point, seed_dir)`` — a point on the global edge line
        nearest the swath centre, and the edge direction — or ``None`` when the
        swath does not resolve into two distinctly-oriented planes.
        """
        self._global_axes = None
        if len(self.points) < 6:
            return None
        labels = _split_two_planes_by_normal(self.normals)
        mask_a = labels == 0
        mask_b = labels == 1
        if mask_a.sum() < 3 or mask_b.sum() < 3:
            return None

        n_a = _principal_axis(self.normals[mask_a])
        n_b = _principal_axis(self.normals[mask_b])
        self._global_axes = (n_a, n_b)
        edge = np.cross(n_a, n_b)
        if np.linalg.norm(edge) < 1e-6:  # parallel: no crease
            return None
        direction = _unit(edge)  # average edge tangent — initial march heading

        # Seed where the two surfaces meet *spatially*: the point whose local
        # neighbourhood is the most balanced mix of the two clusters. This is
        # robust to curvature — unlike intersecting the two global *flat* planes,
        # whose straight chord can run nowhere near a curved edge.
        n = len(self.points)
        self._kdtree = cKDTree(self.points)
        k = min(16, n)
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(n, 2000, replace=False) if n > 2000 else np.arange(n)
        _, nbr = self._kdtree.query(self.points[sample], k=k)
        frac_b = labels[nbr].mean(axis=1)  # fraction of cluster-B neighbours
        seed_point = self.points[sample[int(np.argmin(np.abs(frac_b - 0.5)))]]
        return seed_point, direction

    # ------------------------------------------------------------------ #
    # March                                                              #
    # ------------------------------------------------------------------ #

    def _march(self, start, direction) -> list:
        """March from *start* along *direction*, one cube per step.

        Returns a list of per-step records (dicts) in traversal order.
        """
        half = self.cell_size / 2.0
        search_radius = half * np.sqrt(3.0)  # cube circumradius
        cos_dihedral = np.cos(np.radians(self.min_dihedral_deg))
        cos_bend = np.cos(np.radians(_MAX_ANGLE_DEG))

        tip = np.asarray(start, dtype=np.float64).copy()
        heading = _unit(direction)
        out = []

        for _ in range(_MAX_STEPS):
            # Gather points in the cube centred at the tip, oriented to heading.
            cand = self._kdtree.query_ball_point(tip, search_radius)
            if len(cand) < self.min_points_per_cell:
                break
            cand = np.asarray(cand, dtype=np.intp)
            u2, u3 = _perp_basis(heading)
            rot = np.stack([heading, u2, u3])  # rows = cube axes
            local = (self.points[cand] - tip) @ rot.T
            in_cube = np.all(np.abs(local) <= half, axis=1)
            box_idx = cand[in_cube]
            if len(box_idx) < self.min_points_per_cell:
                break

            # Split the cube into two planes by normal; both must be present.
            labels = _split_two_planes_by_normal(self.normals[box_idx])
            side_a = box_idx[labels == 0]
            side_b = box_idx[labels == 1]
            if len(side_a) < 3 or len(side_b) < 3:
                break  # only one surface in the cube: the edge has ended

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
                break  # sharp turn: cube has likely wandered off the feature

            # Re-centre: the vertex is the edge point nearest the cube centre.
            vertex = point_on_line + float((tip - point_on_line) @ edge_dir) * edge_dir
            ev2, ev3 = _perp_basis(edge_dir)
            out.append({
                "vertex": vertex,
                "frame": np.stack([edge_dir, ev2, ev3]),
                "box_idx": box_idx,
                "side": _align_side(labels, self.normals[box_idx], self._global_axes),
                "n_a": n_a,
                "n_b": n_b,
            })

            # Step one cell along the local edge and re-orient to it.
            tip = vertex + self.cell_size * edge_dir
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


def _align_side(labels, cell_normals, global_axes):
    """Map a cube's local 0/1 split onto the *global* cluster identity.

    The per-cube 2-means assigns label 0/1 by an arbitrary index, so the same
    surface can be 0 in one cube and 1 in the next. Re-key against the global
    plane orientations so 0 = global cluster A everywhere — making the normal
    colours consistent and a per-cube mis-split visibly stand out. Returns int8
    0/1 labels.
    """
    out = labels.astype(np.int8)
    if global_axes is None or not (labels == 0).any():
        return out
    axis_a, axis_b = global_axes
    cluster0 = _principal_axis(cell_normals[labels == 0])
    if abs(float(cluster0 @ axis_b)) > abs(float(cluster0 @ axis_a)):
        out = (1 - out).astype(np.int8)  # local cluster 0 is really global B
    return out


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
    # The three rows are independent (dir ⟂ both normals), so the system is
    # well-conditioned away from the parallel case handled above.
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
    """Randomly cap a cube-side's points at ``_MAX_POINTS_PER_SIDE``."""
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
# trace() records, per march step, the oriented cube, the two fitted planes,   #
# and per-point cube/side ids. These helpers turn that into render-only        #
# VectorFeature wireframes so a plugin can add each stage ("cells", "planes",  #
# "normals") as an ordinary controllable branch. Voxel POINTS are coloured per #
# cube via debug_point_cell, which the plugin turns into a Clusters branch     #
# (points are not wireframe geometry). All helpers are pure: no GUI access.    #
# --------------------------------------------------------------------------- #

_CELL_COLOR = np.array([0.1, 0.9, 0.9], dtype=np.float32)       # cyan
_PLANE_COLOR = np.array([1.0, 0.2, 1.0], dtype=np.float32)      # magenta
_NORMAL_A_COLOR = np.array([1.0, 0.85, 0.1], dtype=np.float32)  # yellow (cluster A)
_NORMAL_B_COLOR = np.array([0.2, 0.6, 1.0], dtype=np.float32)   # blue   (cluster B)

# Cube corners centred on the origin (box-frame), scaled by cell_size at draw.
_CUBE_CORNERS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.float64,
) - 0.5
_CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
               (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]


def cells_to_vector_feature(cells, cell_size, color=None):
    """Wireframe cubes for the per-step cells.

    ``cells`` is a list of ``(center, R)`` where ``R`` rows are the cube's local
    axes (edge tangent + two perpendiculars), so each cube is drawn centred on
    the edge and rotated to follow it.
    """
    if not cells:
        return None
    verts, edges = [], []
    corner_offsets = _CUBE_CORNERS * cell_size
    for center, rot in cells:
        corners = corner_offsets @ rot + np.asarray(center, dtype=np.float64)
        base = len(verts)
        verts.extend(corners)
        for a, b in _CUBE_EDGES:
            edges.append([base + a, base + b])
    return _wireframe_vector_feature(
        "debug_cells", verts, edges, _CELL_COLOR if color is None else color
    )


def planes_to_vector_feature(planes, cell_size, color=None):
    """Wireframe square patch (+ a short normal stub) for each fitted plane."""
    if not planes:
        return None
    verts, edges = [], []
    half = cell_size * 0.5
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
    if show_cells:
        vf = cells_to_vector_feature(tracer.debug_cells, tracer.cell_size)
        if vf is not None:
            out.append(("debug_cells", vf))
    if show_planes:
        vf = planes_to_vector_feature(tracer.debug_planes, tracer.cell_size)
        if vf is not None:
            out.append(("debug_planes", vf))
    if show_normals and tracer.debug_point_side is not None:
        # One branch per global cluster so the two surfaces' normals read in
        # two consistent colours.
        side = tracer.debug_point_side
        for value, name, col in (
            (0, "debug_normals_a", _NORMAL_A_COLOR),
            (1, "debug_normals_b", _NORMAL_B_COLOR),
        ):
            mask = side == value
            vf = normals_to_vector_feature(
                tracer.points[mask], tracer.normals[mask],
                tracer.cell_size * 0.4, col,
            )
            if vf is not None:
                out.append((name, vf))
    return out
