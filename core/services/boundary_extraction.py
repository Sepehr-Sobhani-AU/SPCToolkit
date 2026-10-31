"""
Boundary extraction — the outline methods behind the Vectorise > Outline plugins.

Each method traces a cluster's *raw* outline: a closed 2D ring in the
cluster's plane, at whatever detail the method naturally gives. The shared
clean-up in ``core.services.outline_cleanup`` then turns that ring into a
clean, closed 3D polyline (straight segments, heights from the cluster), so
every method ends in the same kind of result.

- **Grid outline** (the main method) — mark the grid cells that hold points,
  close small gaps, trace the outer edge of the filled area. Fast on millions
  of points and steady when density varies; the cell is sized from the
  points' own spacing.
- **Convex hull** — the tightest convex outline (scipy ``ConvexHull``). Right
  only for convex shapes; quick and setting-free.
- **Concave hull** — the convex hull "dug into" along Delaunay edges longer
  than a length (a chi-shape), so it follows bays and notches.
- **Alpha shape** — keep Delaunay triangles whose circumradius is at most
  alpha; the outer ring of what is left is the outline.
- **K-nearest concave hull** (Moreira & Santos 2007) — walk round the edge,
  each step to one of the k nearest unvisited points.

Every method's ``fn(prep, ...)`` takes an ``outline_cleanup.Prepared`` cluster
and returns ``(ring (M,2) or None, step)``, where ``step`` is the raw ring's
own step size. Default scales come from the measured point spacing, so no
method needs tuning per cloud. The Delaunay-based methods (concave, alpha)
and the k-nearest walk run on the CPU; for large clusters use the grid.
"""

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from core.entities.vector_feature import VectorFeature
from core.services import outline_cleanup

# Wireframe colour per method (render-only).
METHOD_COLORS = {
    "grid_outline": np.array([0.2, 0.8, 1.0], dtype=np.float32),   # sky blue
    "convex_hull": np.array([0.2, 1.0, 0.3], dtype=np.float32),    # green
    "concave_hull": np.array([1.0, 0.6, 0.1], dtype=np.float32),   # orange
    "alpha_shape": np.array([0.7, 0.3, 1.0], dtype=np.float32),    # purple
    "knn_concave_hull": np.array([1.0, 0.3, 0.6], dtype=np.float32),  # pink
}

class OutlineCancelled(RuntimeError):
    """Raised when ``should_stop()`` asks a long outline run to stop."""


# Grid outline: a cell is 1.45x a point's 6th-neighbour distance, which holds
# about 4 points of a random sampling, so a cell inside the cluster is rarely
# empty.
GRID_CELL_FACTOR = 1.45

# Ceiling on grid cells; past it the cell grows. 40M cells is 40 MB of grid.
GRID_MAX_CELLS = 40_000_000


# --------------------------------------------------------------------------- #
# Convex hull                                                                  #
# --------------------------------------------------------------------------- #

def convex_hull_indices(coords2d: np.ndarray) -> np.ndarray:
    """Ordered boundary point indices (counter-clockwise ring) of the convex hull."""
    from scipy.spatial import ConvexHull

    return ConvexHull(np.asarray(coords2d, dtype=np.float64)).vertices


# --------------------------------------------------------------------------- #
# Delaunay-based boundaries (concave hull + alpha shape)                       #
# --------------------------------------------------------------------------- #

def _triangles(coords2d: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    return Delaunay(np.asarray(coords2d, dtype=np.float64)).simplices


def _circumradii(coords2d: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Circumradius of every triangle; degenerate (collinear) triangles → inf."""
    p = np.asarray(coords2d, dtype=np.float64)
    a = np.linalg.norm(p[tris[:, 1]] - p[tris[:, 2]], axis=1)
    b = np.linalg.norm(p[tris[:, 0]] - p[tris[:, 2]], axis=1)
    c = np.linalg.norm(p[tris[:, 0]] - p[tris[:, 1]], axis=1)
    ab = p[tris[:, 1]] - p[tris[:, 0]]
    ac = p[tris[:, 2]] - p[tris[:, 0]]
    area = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = (a * b * c) / (4.0 * area)
    radius[area <= 1e-12] = np.inf
    return radius


def _boundary_edges(tris: np.ndarray) -> List[Tuple[int, int]]:
    """Edges belonging to exactly one triangle in ``tris`` (the outline)."""
    count = defaultdict(int)
    for t in tris:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            count[frozenset((int(t[i]), int(t[j])))] += 1
    return [tuple(e) for e, n in count.items() if n == 1]


def alpha_shape_edges(coords2d: np.ndarray, alpha: float) -> List[Tuple[int, int]]:
    """Boundary edges of the alpha shape (keep triangles with circumradius ≤ alpha)."""
    tris = _triangles(coords2d)
    keep = tris[_circumradii(coords2d, tris) <= float(alpha)]
    if len(keep) == 0:
        return []
    return _boundary_edges(keep)


def concave_hull_edges(coords2d: np.ndarray, max_edge_length: float) -> List[Tuple[int, int]]:
    """Concave-hull boundary edges via Delaunay "digging" (chi-shape).

    Start from the convex-hull outline, then repeatedly replace the longest
    boundary edge longer than ``max_edge_length`` with the other two edges of
    its adjacent triangle — carving the outline inward toward concavities. An
    edge is only carved if the introduced vertex is not already on the
    boundary, which keeps the resulting ring simple.
    """
    import heapq

    p = np.asarray(coords2d, dtype=np.float64)
    tris = _triangles(coords2d)

    # Map each undirected edge to the triangles that own it.
    edge_tris = defaultdict(list)
    for ti, t in enumerate(tris):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge_tris[frozenset((int(t[i]), int(t[j])))].append(ti)

    boundary = set(map(frozenset, _boundary_edges(tris)))
    on_boundary = defaultdict(int)
    for e in boundary:
        for v in e:
            on_boundary[v] += 1

    def length(edge):
        a, b = tuple(edge)
        return float(np.linalg.norm(p[a] - p[b]))

    heap = [(-length(e), tuple(e)) for e in boundary]
    heapq.heapify(heap)
    removed_tris = set()

    while heap:
        neg_len, (a, b) = heapq.heappop(heap)
        edge = frozenset((a, b))
        if edge not in boundary or -neg_len <= max_edge_length:
            continue
        # The single live interior triangle sharing this boundary edge.
        live = [t for t in edge_tris[edge] if t not in removed_tris]
        if not live:
            continue
        t = tris[live[0]]
        c = int(next(v for v in t if v != a and v != b))
        # Carving toward a vertex already on the boundary would pinch the ring.
        if on_boundary[c] > 0:
            continue
        removed_tris.add(live[0])
        boundary.discard(edge)
        for v in (a, b):
            on_boundary[v] -= 1
        for new in (frozenset((a, c)), frozenset((b, c))):
            boundary.add(new)
            for v in new:
                on_boundary[v] += 1
            heapq.heappush(heap, (-length(new), tuple(new)))

    return [tuple(e) for e in boundary]


# --------------------------------------------------------------------------- #
# Edge → ring ordering                                                         #
# --------------------------------------------------------------------------- #

def order_edges_into_loops(edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Order a set of undirected edges into closed loops of vertex indices."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    used = set()
    loops = []
    for start_a, start_b in edges:
        if frozenset((start_a, start_b)) in used:
            continue
        used.add(frozenset((start_a, start_b)))
        loop = [start_a, start_b]
        prev, cur = start_a, start_b
        while True:
            nbrs = [n for n in adj[cur] if frozenset((cur, n)) not in used]
            nbrs = [n for n in nbrs if n != prev] or nbrs
            if not nbrs:
                break
            nxt = nbrs[0]
            used.add(frozenset((cur, nxt)))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def largest_loop(edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """The longest closed loop from ``edges`` (the main outer ring), or None."""
    loops = order_edges_into_loops(edges)
    if not loops:
        return None
    return max(loops, key=len)


# --------------------------------------------------------------------------- #
# Outline methods: fn(prep, ...) -> (raw ring (M,2) or None, step)              #
# --------------------------------------------------------------------------- #

def grid_ring(prep, cell=None, close_cells=1):
    """Grid outline: mark the cells holding points, close small gaps, trace
    the outer edge of the filled area (vertices are cell centres).

    Density-adaptive: where points are sparser than the cell, each point
    also fills a disk the size of its own "4-point cell", so sparse parts stay
    joined while dense parts keep full detail. The disk size comes from the
    neighbours' spacing, so a lone stray stays small.
    """
    import cv2

    uv = prep.uv
    # Sparseness judged by the neighbours' spacing: an edge point (which
    # always looks ~1.4x sparser than it is) or a lone stray does not count.
    want = GRID_CELL_FACTOR * prep.local_ref
    cell = float(cell) if cell else float(np.median(want))
    span = uv.max(axis=0) - uv.min(axis=0)
    while (span[0] / cell) * (span[1] / cell) > GRID_MAX_CELLS:
        cell *= 1.5
    # From 1.7x sparser than the cell a point fills a disk of 1 cell, 2 from
    # 3.7x, ...: at 2x a cell would be empty a third of the time and the
    # sparse side's edge would come out notched. Starting lower (1.5x) splats
    # edge points of uniform clusters too; higher (1.9x) leaves 2x steps
    # notched (swept on uniform, ragged and 4x-density-step scenes).
    radius = np.clip(np.ceil(0.5 * (want / cell - 1.0) - 0.35), 0, 25).astype(np.int64)
    pad = close_cells + int(radius.max()) + 2
    lo = uv.min(axis=0) - pad * cell
    ij = np.floor((uv - lo) / cell).astype(np.int64)
    nx, ny = ij[:, 0].max() + pad + 1, ij[:, 1].max() + pad + 1
    grid = np.zeros((ny, nx), dtype=np.uint8)
    for r in np.unique(radius):
        layer = np.zeros_like(grid)
        sel = radius == r
        layer[ij[sel, 1], ij[sel, 0]] = 255
        if r > 0:
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            layer = cv2.dilate(layer, disk)
        grid |= layer
    if close_cells > 0:
        disk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_cells + 1, 2 * close_cells + 1))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, disk)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, cell
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    return lo + (c + 0.5) * cell, cell


def convex_ring(prep):
    """Convex hull of the cluster."""
    return prep.uv[convex_hull_indices(prep.uv)], prep.spacing


def concave_ring(prep, max_edge=None):
    """Concave hull (chi-shape). ``max_edge`` defaults to 2x the cluster's
    sparse-end spacing, so it bridges the sparse parts of an uneven cluster."""
    max_edge = float(max_edge) if max_edge else 2.0 * prep.sparse_scale
    loop = largest_loop(concave_hull_edges(prep.uv, max_edge))
    return (None if loop is None else prep.uv[loop]), prep.spacing


def alpha_ring(prep, alpha=None):
    """Alpha shape. It can give several rings (holes, pieces): the one that
    encloses the most area is the outer outline. ``alpha`` defaults to 2x the
    sparse-end spacing."""
    alpha = float(alpha) if alpha else 2.0 * prep.sparse_scale
    loops = order_edges_into_loops(alpha_shape_edges(prep.uv, alpha))
    loops = [l for l in loops if len(l) >= 3]
    if not loops:
        return None, prep.spacing
    best = max(loops, key=lambda l: abs(outline_cleanup.ring_area(prep.uv[l])))
    return prep.uv[best], prep.spacing


def _segments_cross(p1, p2, q1, q2):
    """Proper crossing of segment p1p2 with each segment q1[i]q2[i]."""
    r = p2 - p1
    s = q2 - q1
    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    qp = q1 - p1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / rxs
    eps = 1e-9
    return bool(np.any((rxs != 0) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)))


def _knn_walk(pts, tree, k, should_stop=None):
    """One k-nearest walk round the edge, counter-clockwise from the lowest
    point; each step takes the candidate reached by the smallest CCW turn
    from "back where we came from" that keeps the ring simple. Returns ring
    indices, or None when it gets stuck."""
    n = len(pts)
    used = np.zeros(n, dtype=bool)
    first = int(np.argmin(pts[:, 1] + 1e-9 * pts[:, 0]))
    hull = [first]
    used[first] = True
    back = np.array([-1.0, 0.0])
    cur = first
    for step in range(n + 1):
        if should_stop is not None and step % 256 == 0 and should_stop():
            raise OutlineCancelled("Cancelled by user")
        if step == 3:
            used[first] = False         # allow closing the ring from now on
        q = min(n, 3 * k + 3)
        while True:
            _, nn = tree.query(pts[cur], k=q)
            nn = np.atleast_1d(nn)
            cand = nn[~used[nn]][:k]
            if len(cand) >= min(k, n - used.sum()) or q >= n:
                break
            q = min(n, q * 2)
        if len(cand) == 0:
            return None
        vec = pts[cand] - pts[cur]
        ang = (np.arctan2(vec[:, 1], vec[:, 0]) - np.arctan2(back[1], back[0])) % (2 * np.pi)
        ang[ang <= 1e-12] = 2 * np.pi
        hp = pts[np.array(hull)]
        chosen = None
        for c in cand[np.argsort(ang)]:
            # Edges not adjacent to the new one: all but the last (and the
            # first too when the new edge closes the ring).
            lo = 1 if c == first else 0
            q1, q2 = hp[lo:-2], hp[lo + 1:-1]
            if len(q1) == 0 or not _segments_cross(pts[cur], pts[c], q1, q2):
                chosen = int(c)
                break
        if chosen is None:
            return None
        if chosen == first:
            return np.array(hull)
        back = pts[cur] - pts[chosen]
        cur = chosen
        hull.append(cur)
        used[cur] = True
    return None


def knn_ring(prep, k=10, k_max=None, inside_frac=0.995, should_stop=None):
    """K-nearest concave hull. If the walk gets stuck, or leaves more than
    0.5% of the points outside, it retries with a larger k (up to ``k_max``,
    default k + 30). ``should_stop()`` is polled during the walk (it is a
    Python loop and can take minutes on big clusters)."""
    from matplotlib.path import Path
    from scipy.spatial import cKDTree

    pts = np.unique(np.round(prep.uv, 9), axis=0)
    tree = cKDTree(pts)
    k_max = k + 30 if k_max is None else k_max
    for kk in range(max(3, k), k_max + 1, 2):
        idx = _knn_walk(pts, tree, kk, should_stop)
        if idx is None or len(idx) < 3:
            continue
        ring = pts[idx]
        path = Path(ring)
        inside = path.contains_points(pts, radius=1e-6) | path.contains_points(pts, radius=-1e-6)
        if inside.mean() >= inside_frac:
            # The walk jumps several point gaps per step, so its own step is
            # its typical edge length, not the point spacing.
            step = float(np.median(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)))
            return ring, max(step, prep.spacing)
    return None, prep.spacing


METHODS = {
    "grid_outline": grid_ring,
    "convex_hull": convex_ring,
    "concave_hull": concave_ring,
    "alpha_shape": alpha_ring,
    "knn_concave_hull": knn_ring,
}


# --------------------------------------------------------------------------- #
# One call: points -> outline VectorFeature                                    #
# --------------------------------------------------------------------------- #

def outline_feature(points: np.ndarray, method: str, tolerance: float,
                    height_tol: Optional[float] = None, projection: str = "best_fit",
                    drop_strays: bool = True, should_stop=None, **method_params
                    ) -> Tuple[Optional[VectorFeature], dict]:
    """A cluster's clean, closed 3D outline as a polyline ``VectorFeature``.

    Args:
        points: (N,3) cluster points.
        method: a key of ``METHODS``.
        tolerance: plan tolerance of the clean-up (m); ``height_tol``
            defaults to it.
        projection: ``best_fit`` (default), ``xy``, ``xz`` or ``yz``.
        drop_strays: ignore isolated stray points.
        should_stop: optional fn() -> bool polled by the k-nearest walk (the
            one long Python loop); True raises ``OutlineCancelled``.
        method_params: passed to the method (e.g. ``cell``, ``max_edge``,
            ``alpha``, ``k``); None means its spacing-based default.

    Returns:
        (feature or None when no outline was found, info dict).
    """
    fn = METHODS[method]
    params = {k: v for k, v in method_params.items() if v is not None}
    if method == "knn_concave_hull" and should_stop is not None:
        params["should_stop"] = should_stop
    verts, info = outline_cleanup.build_outline(
        points, lambda prep: fn(prep, **params), tolerance, height_tol=height_tol,
        projection=projection, drop_strays=drop_strays)
    if verts is None:
        return None, info
    info["vertices"] = len(verts)
    return ring_to_polyline_feature(verts, method, METHOD_COLORS[method]), info


def ring_to_polyline_feature(ring_xyz: np.ndarray, symbol_type: str,
                             color: np.ndarray) -> Optional[VectorFeature]:
    """A closed-polyline ``VectorFeature`` from an ordered ring of 3D points."""
    verts = np.asarray(ring_xyz, dtype=np.float32)
    if len(verts) < 3:
        return None
    dims = (verts.max(axis=0) - verts.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type=symbol_type,
        geometry_type="polyline",
        geometry={"vertices": verts, "closed": True},
        transform_matrix=np.eye(4),
        dimensions=dims,
        color=np.asarray(color, dtype=np.float32),
    )
