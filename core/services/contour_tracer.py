"""
Generic contour tracing — the level set of any per-point field, grown from a seed.

A *contour* is the line where a per-point scalar field equals a chosen level.
This service traces those lines as polylines, given a cloud, a per-point field, a
level, and one seed point the user picked. It knows nothing about what the field
means: height contours, slope-break lines, distance-to-ground bands and intensity
edges are all *outcomes* of running this one brick with a different field. It is
the level-set sibling of ``crease_tracer`` (a line by *intersecting* two planes)
and ``linear_region_grower`` (a line by *fitting* an axis).

Method — a **flood along the level set**, one ball per step:

1. Seed: snap the picked point to the nearest cloud point; its ball goes first.
2. Per ball: gather the points within ``proximity``, PCA them onto their local
   best-fit plane (inside a small ball the surface is nearly flat, and Delaunay
   needs 2-D input), triangulate the (u, v) scatter, and drop triangles whose
   longest edge exceeds ``max_triangle_edge`` — Delaunay fills its convex hull,
   so without that cut it bridges holes and the outside with fake surface, and a
   contour crossing fake surface is a line across empty space.
3. Marching triangles: a triangle whose three field values do not all sit on the
   same side of the level is crossed on exactly two of its edges. Each crossing
   is interpolated **in 3-D**, not in the local uv plane, so the vertex lands
   back on the real surface.
4. Every crossing is keyed by the **pair of cloud point indices** whose edge it
   lies on. That key is the whole trick: the vertex depends only on those two
   points and the level — never on the ball or the local frame — so overlapping
   balls deduplicate exactly, two ends that meet join by themselves, and a closed
   loop closes with no distance tolerance anywhere.
5. Frontier: a crossing touching only one segment is an open end. Each open end
   snaps to its nearest cloud point and queues that point's ball. A cloud point
   is a ball centre at most once, so the flood is finite and stops on its own —
   no contour leaving a ball, no open ends left, queue empty.
6. The segment graph is chained into ordered polylines at the end.

Several separate contour lines can come out of one run: every open end is grown,
not only the one through the seed, so the flood follows the level set wherever it
reaches. The seed picks the level and where to start, not which line to keep.
"""

from collections import deque

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from core.entities.vector_feature import VectorFeature


_EPS = 1e-12

# A ball needs at least this many points before a triangulation means anything.
_MIN_BALL_POINTS = 4

# Balls between progress reports — the flood is fast per ball, so reporting on
# every one would cost more than the work.
_PROGRESS_EVERY = 25

_CONTOUR_COLOR = np.array([1.0, 0.85, 0.1], dtype=np.float32)  # amber


class ContourTracer:
    """Flood the level set of a per-point field outward from a seed point.

    Args:
        points: (N, 3) cloud coordinates.
        values: (N,) the per-point field being contoured (any scalar — the tracer
            does not care what it means).
        level: the field value the contour follows.
        proximity: ball radius per step (m). The local patch that gets
            triangulated; also how far the flood reaches across gaps.
        max_triangle_edge: drop triangles whose longest edge exceeds this (m).
        kdtree: optional prebuilt ``cKDTree`` over *points*, to avoid rebuilding.
    """

    def __init__(self, points, values, level, proximity, max_triangle_edge,
                 kdtree=None):
        self.points = np.asarray(points, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64).ravel()
        if len(values) != len(self.points):
            raise ValueError(
                f"values has {len(values)} entries but points has {len(self.points)}."
            )
        if not self._is_positive(proximity) or not self._is_positive(max_triangle_edge):
            raise ValueError("proximity and max_triangle_edge must be finite and > 0.")

        self.level = float(level)
        self.proximity = float(proximity)
        self.max_triangle_edge = float(max_triangle_edge)
        self.kdtree = cKDTree(self.points) if kdtree is None else kdtree

        # Field measured from the level, so the contour is where this is zero. A
        # value sitting exactly on the level makes the sign test ambiguous and
        # the edge interpolation divide by zero, so nudge those off zero once.
        self._signed = values - self.level
        self._signed[np.abs(self._signed) < _EPS] = _EPS

        self._crossings = {}        # (i, j) i<j       -> (3,) vertex on that edge
        self._segments = set()      # {(key_a, key_b)} -> the contour segments
        self._adjacency = {}        # key              -> {neighbour keys}
        self._visited_centers = set()   # cloud point indices already used as a centre

        self.n_balls = 0

    @staticmethod
    def _is_positive(value) -> bool:
        return bool(np.isfinite(value)) and float(value) > 0.0

    # ------------------------------------------------------------------ #
    # The flood                                                          #
    # ------------------------------------------------------------------ #

    def trace(self, seed_point, progress_cb=None, cancel_event=None):
        """Grow the level set from *seed_point*; return a list of ordered (M, 3)
        polylines. On cancel, returns whatever was traced so far."""
        seed_point = np.asarray(seed_point, dtype=np.float64).ravel()[:3]
        _, seed_idx = self.kdtree.query(seed_point)

        # Queue entries are (ball centre point index, the open end that queued it).
        # The seed has no end behind it, so it always runs.
        queue = deque([(int(seed_idx), None)])

        while queue:
            if cancel_event is not None and cancel_event.is_set():
                break
            center_idx, from_key = queue.popleft()
            if center_idx in self._visited_centers:
                continue
            # The end that queued this ball may have been closed since, by another
            # end growing into it. Then it is no longer an end and needs no ball:
            # this is where two lines meeting drop each other from the queue.
            if from_key is not None and len(self._adjacency.get(from_key, ())) != 1:
                continue

            self._visited_centers.add(center_idx)
            touched = self._process_ball(center_idx)
            self.n_balls += 1
            self._queue_open_ends(touched, queue)

            if progress_cb is not None and self.n_balls % _PROGRESS_EVERY == 0:
                progress_cb(
                    self.n_balls, self.n_balls + len(queue),
                    f"Tracing contours — {self.n_balls:,} steps, "
                    f"{len(self._segments):,} segments",
                )

        return self._chain_polylines()

    def _process_ball(self, center_idx):
        """Triangulate the ball at *center_idx* and march it. Returns the crossing
        keys this ball touched (the only ones whose ends can have changed)."""
        idx = np.asarray(
            self.kdtree.query_ball_point(self.points[center_idx], self.proximity),
            dtype=np.intp,
        )
        if len(idx) < _MIN_BALL_POINTS:
            return []
        uv = self._project_to_plane(self.points[idx])
        # The centre in the ball's own frame — it is one of the ball's points.
        center_uv = uv[int(np.flatnonzero(idx == center_idx)[0])]
        triangles = self._triangulate(uv, center_uv)
        if triangles is None:
            return []
        return self._march_triangles(idx, triangles)

    @staticmethod
    def _project_to_plane(pts):
        """PCA the ball onto its best-fit plane. Inside a small ball the surface is
        nearly flat, so the two dominant directions carry it with little error —
        and Delaunay needs 2-D input."""
        centered = pts - pts.mean(axis=0)
        # SVD rows come out ordered by singular value, so the first two span the
        # surface and the third is its normal.
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        return centered @ basis[:2].T

    def _triangulate(self, uv, center_uv):
        """Delaunay in the local plane, minus the triangles this ball may not
        vouch for. Two cuts, for two different lies:

        1. **Fake surface.** Delaunay fills its convex hull, so it bridges holes
           and wraps the outside with long thin triangles that no points support.
           A contour crossing one of those is a line through empty space, so
           anything longer than ``max_triangle_edge`` goes.
        2. **The ball's rim.** A local triangle is also the *whole cloud's*
           Delaunay triangle only when its circumcircle holds no other point —
           and this ball can only vouch for that while the circumcircle stays
           inside it. Rim triangles are built from points whose real neighbours
           lie outside the ball, so the next ball triangulates the same spot
           differently; keeping them forks the contour into junctions and
           fragments instead of one clean line. Dropping them costs nothing: the
           flood re-centres on the open ends, so the next ball's core covers
           exactly what this one gave up.
        """
        try:
            simplices = Delaunay(uv).simplices
        except Exception:
            return None  # degenerate scatter (collinear, duplicate points, …)
        if len(simplices) == 0:
            return None

        corners = uv[simplices]  # (T, 3, 2)
        longest = np.max(
            np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2), axis=1
        )
        keep = longest <= self.max_triangle_edge

        center, radius = self._circumcircles(corners)
        reach = np.linalg.norm(center - center_uv, axis=1) + radius
        keep &= reach <= self.proximity

        return simplices[keep] if np.any(keep) else None

    @staticmethod
    def _circumcircles(corners):
        """Circumcircle centre and radius of each (3, 2) triangle. A collinear
        triangle lands its centre at infinity, which the caller's reach test then
        rejects on its own."""
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        ax, ay = a[:, 0], a[:, 1]
        bx, by = b[:, 0], b[:, 1]
        cx, cy = c[:, 0], c[:, 1]

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        d = np.where(np.abs(d) < _EPS, _EPS, d)
        sq_a = ax ** 2 + ay ** 2
        sq_b = bx ** 2 + by ** 2
        sq_c = cx ** 2 + cy ** 2

        center = np.stack([
            (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / d,
            (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / d,
        ], axis=1)
        return center, np.linalg.norm(a - center, axis=1)

    def _march_triangles(self, idx, triangles):
        """Marching triangles over one ball; returns the crossing keys touched.

        A triangle whose three values do not all sit on the same side of the level
        is crossed on exactly two edges — never one, never three.
        """
        touched = []
        signs = self._signed[idx][triangles] > 0  # (T, 3)
        crossed = ~(np.all(signs, axis=1) | ~np.any(signs, axis=1))

        for triangle in triangles[crossed]:
            keys = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ia, ib = int(idx[triangle[a]]), int(idx[triangle[b]])
                if (self._signed[ia] > 0) == (self._signed[ib] > 0):
                    continue  # both corners on the same side: edge not crossed
                keys.append(self._crossing(ia, ib))
            if len(keys) != 2:
                continue  # numerically degenerate — skip rather than guess
            self._add_segment(keys[0], keys[1])
            touched.extend(keys)
        return touched

    def _crossing(self, ia, ib):
        """The contour vertex on cloud edge ``(ia, ib)``, keyed by that edge.

        Keying by the point pair is what makes the flood work: the vertex depends
        only on the two points and the level, so every ball that crosses this edge
        computes the identical vertex under the identical key. Overlapping balls
        therefore collapse onto each other instead of drawing the line twice.
        """
        key = (ia, ib) if ia < ib else (ib, ia)
        if key not in self._crossings:
            value_a, value_b = self._signed[key[0]], self._signed[key[1]]
            point_a, point_b = self.points[key[0]], self.points[key[1]]
            # Opposite signs, so the denominator can't vanish. Interpolate in 3-D
            # so the vertex lands on the real surface, not the flattened one.
            t = -value_a / (value_b - value_a)
            self._crossings[key] = point_a + t * (point_b - point_a)
        return key

    def _add_segment(self, key_a, key_b):
        """Link two crossings, unless either already has both its segments.

        A crossing lies on one cloud edge, and an edge has at most two triangles
        beside it — so two segments is all a crossing can carry. That cap is what
        keeps a curved surface clean: neighbouring balls fit slightly different
        local planes, project the same points slightly differently, and Delaunay
        answers a near-tie between two diagonals differently in each. The vertices
        agree exactly (they are edge-keyed), only the pairing differs, so a third
        segment is a second opinion on a pairing already settled — and taking it
        forks the line into junctions and stubs. First ball to reach a crossing
        settles it; the walk order from the seed makes that deterministic.
        """
        if key_a == key_b:
            return
        if (len(self._adjacency.get(key_a, ())) >= 2
                or len(self._adjacency.get(key_b, ())) >= 2):
            return
        segment = (key_a, key_b) if key_a < key_b else (key_b, key_a)
        if segment in self._segments:
            return
        self._segments.add(segment)
        self._adjacency.setdefault(key_a, set()).add(key_b)
        self._adjacency.setdefault(key_b, set()).add(key_a)

    def _queue_open_ends(self, touched, queue):
        """Queue a ball at every open end this step produced. A crossing touching
        one segment is an end the contour still wants to leave through; two means
        the line already continues past it."""
        for key in touched:
            if len(self._adjacency.get(key, ())) != 1:
                continue
            _, center_idx = self.kdtree.query(self._crossings[key])
            center_idx = int(center_idx)
            if center_idx not in self._visited_centers:
                queue.append((center_idx, key))

    # ------------------------------------------------------------------ #
    # Segment graph -> polylines                                         #
    # ------------------------------------------------------------------ #

    def _chain_polylines(self):
        """Chain the segment graph into ordered polylines.

        Open lines first: walk from every end (one segment) to its far end, so a
        loop hanging off an open line isn't cut mid-way. Whatever edges are left
        belong to closed loops, which come back to their own start and so carry no
        end to start from. Every crossing has one or two segments (``_add_segment``
        caps it), so a walk only ever ends at a real end or back where it began.
        """
        polylines = []
        walked = set()

        def edge_of(a, b):
            return (a, b) if a < b else (b, a)

        def walk(start, first):
            chain = [self._crossings[start]]
            previous, current = start, first
            while True:
                edge = edge_of(previous, current)
                if edge in walked:
                    break
                walked.add(edge)
                chain.append(self._crossings[current])
                onward = [k for k in self._adjacency[current] if k != previous]
                if len(onward) != 1:
                    break  # an end, or a junction
                previous, current = current, onward[0]
            return chain

        for start, neighbours in self._adjacency.items():
            if len(neighbours) != 1:
                continue
            first = next(iter(neighbours))
            if edge_of(start, first) not in walked:
                polylines.append(walk(start, first))

        for start, neighbours in self._adjacency.items():
            for first in neighbours:
                if edge_of(start, first) not in walked:
                    polylines.append(walk(start, first))

        return [np.asarray(c, dtype=np.float64) for c in polylines if len(c) >= 2]


# --------------------------------------------------------------------------- #
# Dialog pre-fill                                                             #
# --------------------------------------------------------------------------- #


def suggest_spacing(points, seed_point, sample=256):
    """Median nearest-neighbour spacing at the pick — the scale both ``proximity``
    and ``max_triangle_edge`` are derived from.

    One O(N) distance pass plus a tiny local tree, so it is cheap enough to
    pre-fill the dialog on a 10M-point cloud. Returns 0.0 when it can't tell.
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    if len(pts) < 2:
        return 0.0
    seed = np.asarray(seed_point, dtype=np.float64).ravel()[:3]

    offsets = pts - seed
    distances_sq = np.einsum("ij,ij->i", offsets, offsets)
    count = min(sample, len(pts))
    local = pts[np.argpartition(distances_sq, count - 1)[:count]]
    if len(local) < 2:
        return 0.0

    neighbour_distances, _ = cKDTree(local).query(local, k=2)
    return float(np.median(neighbour_distances[:, 1]))


# --------------------------------------------------------------------------- #
# Renderable output                                                           #
# --------------------------------------------------------------------------- #


def contours_to_vector_feature(polylines, color=None):
    """One VectorFeature holding every traced contour, for a single branch.

    Each polyline becomes its own connected edge chain in a single mesh, so the
    separate contours draw connected within themselves with no spurious edge
    bridging one to the next. A closed contour needs no flag — it is simply a
    chain that returns to its first vertex.

    (``linear_region_grower`` and ``crease_tracer`` each carry their own copy of
    this wireframe helper; kept local here to match, rather than importing across
    sibling services.)
    """
    vertices = []
    edges = []
    for polyline in polylines:
        if polyline is None or len(polyline) < 2:
            continue
        base = len(vertices)
        vertices.extend(np.asarray(polyline, dtype=np.float64))
        edges.extend([base + i, base + i + 1] for i in range(len(polyline) - 1))
    if not edges:
        return None

    vertices = np.asarray(vertices, dtype=np.float32)
    dimensions = (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type="Contours",
        geometry_type="mesh",
        geometry={"vertices": vertices, "faces": [], "edges": np.asarray(edges, dtype=np.int32)},
        transform_matrix=np.eye(4),
        dimensions=dimensions,
        color=_CONTOUR_COLOR if color is None else np.asarray(color, dtype=np.float32),
    )
