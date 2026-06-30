"""
Boundary extraction — turn a (near-planar) point cloud into a boundary outline.

Three related "Lego bricks", all sharing the same pipeline: project the cloud
onto a plane, compute a boundary in 2D, then return the boundary as real 3D
cloud points wired into a render-only ``VectorFeature`` branch.

- **Convex hull** — the tightest *convex* outline (scipy ``ConvexHull``).
- **Concave hull** — convex hull "dug into" along Delaunay edges longer than a
  user length, so the outline hugs concavities (a chi-shape).
- **Alpha shape** — the alpha-complex boundary: keep Delaunay triangles whose
  circumradius ≤ alpha, the boundary edges are the outline. Handles holes and
  disjoint pieces, so it is returned as a wireframe (not a single ring).

All functions here are pure geometry: no GUI, no global state. Boundary
vertices are *actual* cloud points (the 2D projection is only used to decide
which points are on the boundary), so coordinates stay exact.
"""

from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.entities.vector_feature import VectorFeature

_CONVEX_COLOR = np.array([0.2, 1.0, 0.3], dtype=np.float32)   # green
_CONCAVE_COLOR = np.array([1.0, 0.6, 0.1], dtype=np.float32)  # orange
_ALPHA_COLOR = np.array([0.7, 0.3, 1.0], dtype=np.float32)    # purple


# --------------------------------------------------------------------------- #
# Projection                                                                   #
# --------------------------------------------------------------------------- #

def project_to_plane(points: np.ndarray, mode: str = "best_fit") -> np.ndarray:
    """Project ``points`` (N,3) onto a plane and return 2D coords (N,2).

    The 2D coords are only used to decide the boundary; the caller maps the
    chosen boundary indices back to the original 3D points, so no information
    is lost. ``mode`` is one of ``best_fit`` (PCA plane of the cloud) or the
    axis-aligned planes ``xy`` / ``xz`` / ``yz``.
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    if mode == "xy":
        return pts[:, [0, 1]]
    if mode == "xz":
        return pts[:, [0, 2]]
    if mode == "yz":
        return pts[:, [1, 2]]
    if mode == "best_fit":
        centre = pts.mean(axis=0)
        centred = pts - centre
        # Two largest principal axes span the best-fit plane.
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        basis = vh[:2]  # (2, 3)
        return centred @ basis.T
    raise ValueError(f"Unknown projection mode '{mode}'.")


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
# VectorFeature builders                                                       #
# --------------------------------------------------------------------------- #

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


def edges_to_wireframe_feature(points_xyz: np.ndarray, edges: List[Tuple[int, int]],
                               symbol_type: str, color: np.ndarray) -> Optional[VectorFeature]:
    """A wireframe (mesh-edges) ``VectorFeature`` for an arbitrary edge set.

    Used for alpha shapes, whose boundary may contain holes or disjoint pieces
    that a single polyline can't represent. Only the referenced points are kept
    and the edge indices are remapped to that compact vertex set.
    """
    if not edges:
        return None
    used = sorted({v for e in edges for v in e})
    remap = {old: new for new, old in enumerate(used)}
    verts = np.asarray(points_xyz, dtype=np.float32)[used]
    edge_arr = np.array([[remap[a], remap[b]] for a, b in edges], dtype=np.int32)
    dims = (verts.max(axis=0) - verts.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type=symbol_type,
        geometry_type="mesh",
        geometry={"vertices": verts, "faces": [], "edges": edge_arr},
        transform_matrix=np.eye(4),
        dimensions=dims,
        color=np.asarray(color, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# High-level one-call entry points (used by the plugins)                       #
# --------------------------------------------------------------------------- #

def convex_hull_feature(points: np.ndarray, projection: str = "best_fit",
                        color: Optional[np.ndarray] = None) -> Optional[VectorFeature]:
    coords2d = project_to_plane(points, projection)
    ring = convex_hull_indices(coords2d)
    pts3d = np.asarray(points, dtype=np.float32)[:, :3]
    return ring_to_polyline_feature(
        pts3d[ring], "convex_hull", _CONVEX_COLOR if color is None else color
    )


def concave_hull_feature(points: np.ndarray, max_edge_length: float,
                         projection: str = "best_fit",
                         color: Optional[np.ndarray] = None) -> Optional[VectorFeature]:
    coords2d = project_to_plane(points, projection)
    edges = concave_hull_edges(coords2d, max_edge_length)
    ring = largest_loop(edges)
    if ring is None:
        return None
    pts3d = np.asarray(points, dtype=np.float32)[:, :3]
    return ring_to_polyline_feature(
        pts3d[ring], "concave_hull", _CONCAVE_COLOR if color is None else color
    )


def alpha_shape_feature(points: np.ndarray, alpha: float,
                        projection: str = "best_fit",
                        color: Optional[np.ndarray] = None) -> Optional[VectorFeature]:
    coords2d = project_to_plane(points, projection)
    edges = alpha_shape_edges(coords2d, alpha)
    pts3d = np.asarray(points, dtype=np.float32)[:, :3]
    return edges_to_wireframe_feature(
        pts3d, edges, "alpha_shape", _ALPHA_COLOR if color is None else color
    )
