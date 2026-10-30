"""
Tests for core.services.outline_cleanup and core.services.boundary_extraction.

Every cluster here is synthetic with an exactly known outline and surface, so
the oracle is the truth itself: the right number of vertices, every corner
where it belongs, heights on the surface. The scenes follow the experiment
that settled the design (``experiments/cluster_outline``, not tracked): a
rectangle, an L, a pitched roof whose gable peaks matter only in height, a
wall stood vertical, a 30 deg roof face, stray points, and a curved island.

Tolerance is 5 cm in plan and height throughout, as in the experiment.
Corner limits are 8-10 cm, not 5: the outline follows the cluster's
outermost points (2-3 cm inside the true edge on each side, so ~4 cm at a
corner) and the rebuilt corner adds a little more. Broken corner repair or
a stray pulling the outline shows up as 15 cm to a metre. Each scene uses a
fixed seed; the limits were checked over ten seeds.

Needs CuPy and an NVIDIA GPU: every neighbour search runs on ``GridKNN``,
which has no CPU path by design.

Run:
    python unit_test/outline_cleanup_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from core.entities.vector_feature import VectorFeature
from core.services import boundary_extraction as be
from core.services import outline_cleanup as oc

TOL = 0.05          # clean-up tolerance, plan and height (m)
ALL_METHODS = list(be.METHODS)


# ----------------------------------------------------------------------
# Synthetic scenes (a trimmed copy of experiments/cluster_outline/scenes.py)
# ----------------------------------------------------------------------

def _densify(ring, step=0.005, closed=True):
    ring = np.asarray(ring, float)
    pairs = zip(ring, np.roll(ring, -1, axis=0)) if closed else zip(ring[:-1], ring[1:])
    out = []
    for a, b in pairs:
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (np.arange(n)[:, None] / n) * (b - a))
    if not closed:
        out.append(ring[-1:])
    return np.vstack(out)


def _arc(c, r, n, start, end):
    t = np.linspace(start, end, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def _rect(w, h):
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def _lshape():
    return np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 8], [0, 8]], float)


def _stadium(length=12.0, r=1.5, n=180):
    right = _arc((length / 2 - r, 0), r, n, -np.pi / 2, np.pi / 2)
    left = _arc((-(length / 2 - r), 0), r, n, np.pi / 2, 3 * np.pi / 2)
    return np.vstack([right, [[length / 2 - r, r]], left, [[-(length / 2 - r), -r]]])


def _gentle(x, y):
    return 10.0 + 0.02 * x + 0.01 * y


def _gable(x, y, eave=3.0, half=3.0, pitch=30.0):
    return eave + np.tan(np.radians(pitch)) * (half - np.abs(y - half))


def _face30(x, y):
    return 5.0 + np.tan(np.radians(30)) * y


def _standing(yaw_deg):
    """Local x along the wall (yawed), local y straight up, local z = normal."""
    a = np.radians(yaw_deg)
    along = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.vstack([along, up, np.cross(along, up)]), np.array([5.0, 3.0, 2.0])


class Scene:
    """Points sampled uniformly inside ``outline`` (local x, y) at ``density``
    per m2, height ``zfun(x, y)``, 1 cm edge noise, 5 mm height noise, then
    placed in the world by ``placement`` (rows of R = local axes in world)."""

    def __init__(self, outline, zfun, density=400, seed=0, edge_noise=0.01,
                 z_noise=0.005, outliers=0.0, placement=None):
        self.outline = np.asarray(outline, float)
        self.zfun = zfun
        self.truth = _densify(self.outline)
        rng = np.random.default_rng(seed)
        mn, mx = self.outline.min(0), self.outline.max(0)
        n = int(density * (mx - mn).prod())
        xy = rng.uniform(mn, mx, (n, 2))
        xy = xy[MplPath(self.outline).contains_points(xy)]
        xy += rng.normal(0, edge_noise, xy.shape)
        z = zfun(xy[:, 0], xy[:, 1]) + rng.normal(0, z_noise, len(xy))
        pts = [np.column_stack([xy, z])]
        self.n_strays = 0
        if outliers:
            # ``outliers`` lone strays 0.4-0.6 m straight out from edge points
            # spread evenly round the edge, so no two are neighbours (the
            # outline is CCW, so (dy, -dx) of the edge points outward).
            m = int(outliers)
            i = (np.arange(m) * len(self.truth)) // m + rng.integers(0, len(self.truth) // (4 * m), m)
            tangent = np.roll(self.truth, -1, axis=0)[i] - np.roll(self.truth, 1, axis=0)[i]
            normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            oxy = self.truth[i] + rng.uniform(0.4, 0.6, (m, 1)) * normal
            self.n_strays = len(oxy)
            pts.append(np.column_stack([oxy, zfun(oxy[:, 0], oxy[:, 1])]))
        self.R, self.t = placement if placement is not None else (np.eye(3), np.zeros(3))
        self.points = self.to_world(np.vstack(pts)).astype(np.float32)

    def to_world(self, local):
        return np.asarray(local, float) @ self.R + self.t

    def to_local(self, world):
        return (np.asarray(world, float) - self.t) @ self.R.T

    def corners3d(self):
        c = self.outline
        return self.to_world(np.column_stack([c, self.zfun(c[:, 0], c[:, 1])]))

    def surface_error(self, world):
        loc = self.to_local(world)
        return loc[:, 2] - self.zfun(loc[:, 0], loc[:, 1])

    def plan_error(self, verts):
        """Worst plan distance between the outline and the true edge, both
        ways (outline -> truth and truth -> outline), in local x, y."""
        loc = self.to_local(verts)[:, :2]
        dense = _densify(loc, 0.005)
        a, _ = cKDTree(self.truth).query(dense)
        b, _ = cKDTree(dense).query(self.truth)
        return float(max(a.max(), b.max()))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _outline(scene, method="grid_outline", **kw):
    feat, info = be.outline_feature(scene.points, method, TOL, **kw)
    assert feat is not None, f"{method}: no outline ({info})"
    return feat.geometry["vertices"].astype(np.float64), feat, info


def _segments_cross(ring):
    """True if two non-adjacent edges of a closed 2D ring cross."""
    n = len(ring)
    a, b = ring, np.roll(ring, -1, axis=0)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p, r = a[i], b[i] - a[i]
            q, s = a[j], b[j] - a[j]
            rxs = r[0] * s[1] - r[1] * s[0]
            if rxs == 0:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
            u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
            if 0 < t < 1 and 0 < u < 1:
                return True
    return False


def _corner_errors(scene, verts):
    """3D distance from every true corner to its nearest outline vertex."""
    d, _ = cKDTree(verts).query(scene.corners3d())
    return d


def _area(ring2d):
    return oc.ring_area(np.asarray(ring2d, float))


# ----------------------------------------------------------------------
# Clean-up on known shapes
# ----------------------------------------------------------------------

def test_rectangle_every_method(seed=1):
    """10 x 4 m, 400 pts/m2, 1 cm edge noise, gentle slope: four corners.

    Grid outline and alpha shape give exactly 4 vertices on every seed
    tried. The convex, concave and k-nearest hulls keep an extra kink or
    two on some seeds at this density (up to 8 vertices), so for them the
    count is bounded rather than exact; a broken clean-up gives far more.

    K-nearest is the least steady method: its walk zig-zags up to ~10 cm
    inward between edge points, and now and then a dip survives the clean-up
    as a dent next to a corner (about one seed in five, up to ~20 cm off).
    Its corner limit is looser for that reason, not because the clean-up is
    allowed to be worse."""
    scene = Scene(_rect(10, 4), _gentle, seed=seed)
    for method in ALL_METHODS:
        verts, _feat, _info = _outline(scene, method)
        corner = _corner_errors(scene, verts)
        loc2d = scene.to_local(verts)[:, :2]
        herr = np.abs(scene.surface_error(verts))
        if method in ("grid_outline", "alpha_shape"):
            assert len(verts) == 4, f"{method}: {len(verts)} vertices, want 4"
        else:
            assert 4 <= len(verts) <= 8, f"{method}: {len(verts)} vertices, want 4-8"
        corner_limit = 0.25 if method == "knn_concave_hull" else 0.08
        assert corner.max() < corner_limit, f"{method}: corner off by {corner.max():.3f} m"
        assert _area(loc2d) > 0, f"{method}: ring is clockwise"
        assert not _segments_cross(loc2d), f"{method}: ring crosses itself"
        assert herr.max() < 0.01, f"{method}: height off by {herr.max():.3f} m"
        print(f"  rectangle {method:17s}: {len(verts)} v, corners <= {100 * corner.max():.1f} cm, "
              f"heights <= {100 * herr.max():.2f} cm")


def test_l_shape(seed=1):
    """Grid outline gives exactly 6 vertices on every seed tried. Concave
    hull and alpha shape do on most seeds but keep an extra kink or two on
    some (concave up to ~12), so their count is bounded rather than exact;
    every true corner must still be matched closely."""
    scene = Scene(_lshape(), _gentle, seed=seed)
    for method in ("grid_outline", "concave_hull", "alpha_shape"):
        verts, _f, _i = _outline(scene, method)
        corner = _corner_errors(scene, verts)
        if method == "grid_outline":
            assert len(verts) == 6, f"{method}: {len(verts)} vertices, want 6"
        else:
            assert 6 <= len(verts) <= 12, f"{method}: {len(verts)} vertices, want 6-12"
        assert corner.max() < 0.09, f"{method}: corner off by {corner.max():.3f} m"
        assert not _segments_cross(scene.to_local(verts)[:, :2])
        print(f"  L shape {method:17s}: {len(verts)} v, corners <= {100 * corner.max():.1f} cm")

    verts, _f, _i = _outline(scene, "convex_hull")
    loc2d = scene.to_local(verts)[:, :2]
    assert len(verts) < 6, f"convex hull: {len(verts)} vertices"
    # Convex: every turn goes the same (CCW) way.
    e1 = np.roll(loc2d, -1, axis=0) - loc2d
    e2 = np.roll(e1, -1, axis=0)
    turns = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    assert np.all(turns > 0), "convex hull outline is not convex"
    # It covers the inner corner (4, 4) instead of following it.
    assert MplPath(loc2d).contains_point((4.0, 4.0)), "convex hull does not cover the inner corner"
    print(f"  L shape convex_hull      : {len(verts)} v, convex, covers the inner corner")


def test_pitched_roof_keeps_gable_peaks(seed=0):
    """Two 30 deg faces: the peaks sit on straight plan edges and matter only
    in height. 3D weeding must keep them; 2D-only weeding loses them (and
    the eave line then misses each peak by 1.7 m in height).

    The peak heights run a few cm low: the 16 neighbours of a vertex on the
    ridge straddle both faces, and one plane through them sits under the
    ridge. Hence 6 cm on the peak, against 1 cm on a single face."""
    scene = Scene(_rect(8, 6), _gable, seed=seed)
    verts, _f, _i = _outline(scene, "grid_outline")
    peaks = scene.to_world(np.array([[0, 3, _gable(0, 3)], [8, 3, _gable(8, 3)]]))
    d, nearest = cKDTree(verts).query(peaks)
    herr = np.abs(scene.surface_error(verts))
    assert len(verts) == 6, f"pitched roof: {len(verts)} vertices, want 6"
    assert d.max() < 0.10, f"a gable peak is {d.max():.3f} m from any vertex"
    peak_h = np.abs(verts[nearest, 2] - peaks[:, 2])
    assert peak_h.max() < 0.06, f"gable peak height off by {peak_h.max():.3f} m"
    assert herr.max() < 0.05, f"a vertex is {herr.max():.3f} m off the roof"
    assert _corner_errors(scene, verts).max() < 0.10
    print(f"  pitched roof: 6 v, peaks within {100 * d.max():.1f} cm "
          f"(height {100 * peak_h.max():.1f} cm)")

    # The same with height ignored: the peaks go.
    flat_feat, _ = be.outline_feature(scene.points, "grid_outline", TOL, height_tol=1e9)
    flat = flat_feat.geometry["vertices"]
    assert len(flat) == 4, f"2D-only weeding kept {len(flat)} vertices"
    print(f"  pitched roof, 2D-only weeding (height_tol huge): {len(flat)} v - peaks lost")


def test_vertical_wall_is_outlined_on_the_wall(seed=1):
    """A pentagon gable wall 10 m long, 4.5 m high, turned 30 deg, standing."""
    outline = np.array([[0, 0], [10, 0], [10, 3], [5, 4.5], [0, 3]], float)
    scene = Scene(outline, lambda x, y: 0.0 * x, seed=seed, placement=_standing(30))
    verts, _f, _i = _outline(scene, "grid_outline", projection="best_fit")
    corner = _corner_errors(scene, verts)
    off_wall = np.abs(scene.surface_error(verts))
    assert len(verts) == 5, f"wall: {len(verts)} vertices, want 5"
    assert corner.max() < 0.10, f"wall corner off by {corner.max():.3f} m"
    assert off_wall.max() < 0.01, f"a vertex is {off_wall.max():.3f} m off the wall plane"
    assert not _segments_cross(scene.to_local(verts)[:, :2])
    print(f"  vertical wall: 5 v on the wall (off by <= {100 * off_wall.max():.2f} cm), "
          f"corners <= {100 * corner.max():.1f} cm")


def test_sloped_face_heights_are_unbiased(seed=5):
    """A single 30 deg roof face. Flattened onto its own plane (best_fit) the
    face is level, so heights are easy; flattened onto XY the slope stays in
    the heights and a vertex on the edge has all its neighbours on the inner
    side. Their median is dragged uphill (5-7 cm worst, 3.4 cm RMS along the
    edge); the plane fitted through them and read at the vertex is not."""
    scene = Scene(_rect(8, 5), _face30, seed=seed)
    for projection in ("best_fit", "xy"):
        verts, _f, _i = _outline(scene, "grid_outline", projection=projection)
        herr = np.abs(scene.surface_error(verts))
        assert herr.max() < 0.01, f"{projection}: height off by {herr.max():.3f} m on a 30 deg face"
        print(f"  30 deg face, {projection:8s}: outline heights <= {100 * herr.max():.2f} cm")

    # And vertex_heights directly on the true edge, in XY.
    prep = oc.Prepared(scene.points, "xy")
    try:
        edge3d = scene.to_world(np.column_stack(
            [scene.truth, scene.zfun(scene.truth[:, 0], scene.truth[:, 1])]))
        uv, true_h = prep.frame.to_2d(edge3d)
        h = oc.vertex_heights(uv, prep.uv, prep.h, index=prep.index)
    finally:
        prep.release()
    rms = float(np.sqrt(np.mean((h - true_h) ** 2)))
    assert rms < 0.015, f"vertex_heights RMS {rms:.4f} m along the edge"
    print(f"  30 deg face, vertex_heights on the true edge: RMS {100 * rms:.2f} cm")


def test_stray_points_do_not_pull_the_outline(seed=6):
    """20 lone points 0.4-0.6 m outside the edge are all dropped and the
    outline stays on the edge. Without dropping them the convex hull reaches
    out to them, so the scene does test something. The stray test also
    drops the odd real point (about 1 in 2000-8000): allowed, a few."""
    scene = Scene(_rect(10, 4), _gentle, seed=seed, outliers=20)
    verts, _f, info = _outline(scene, "grid_outline")
    err = scene.plan_error(verts)
    n_real = len(scene.points) - scene.n_strays
    assert scene.n_strays == 20
    assert 20 <= info["strays"] <= 20 + max(3, n_real // 1000), (info["strays"], n_real)
    assert len(verts) == 4, f"{len(verts)} vertices, want 4"
    assert err < 0.08, f"outline {err:.3f} m from the true edge"
    pulled, _f, _i = _outline(scene, "convex_hull", drop_strays=False)
    assert scene.plan_error(pulled) > 0.3, "strays kept should pull the convex hull out"
    print(f"  {scene.n_strays} strays 0.4-0.6 m out: {info['strays']} dropped, "
          f"outline within {100 * err:.1f} cm, {len(verts)} v "
          f"(kept, they pull the convex hull {100 * scene.plan_error(pulled):.0f} cm out)")


def test_strays_close_to_the_edge_are_dropped(seed=0):
    """Strays only 0.2 m outside a dense edge. Judged against their closest
    few neighbours they look ordinary (edge points look sparse too, and
    strays vouch for each other), so they used to survive and the grid's
    gap closing joined them to the outline (8 vertices, 13 cm out). Judged
    against a wide neighbourhood they are dropped."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0, 0], [10, 4], (16000, 2))
    strays = np.column_stack([np.linspace(0.5, 9.5, 10), np.full(10, -0.2)])
    pts = np.column_stack([np.vstack([xy, strays]), np.zeros(16010)]).astype(np.float32)
    feat, info = be.outline_feature(pts, "grid_outline", 0.05)
    verts = feat.geometry["vertices"]
    assert info["strays"] >= 10, f"only {info['strays']} of 10 strays dropped"
    assert len(verts) == 4, f"{len(verts)} vertices, want 4"
    assert verts[:, 1].min() > -0.06, f"outline pulled to y = {verts[:, 1].min():.3f}"
    print(f"  10 strays 0.2 m out: {info['strays']} dropped, {len(verts)} v, "
          f"lowest y {verts[:, 1].min():.3f} m")


def test_curved_island_keeps_its_round_ends(seed=7):
    scene = Scene(_stadium(), _gentle, seed=seed)
    verts, _f, _i = _outline(scene, "grid_outline")
    loc = scene.to_local(verts)[:, :2]
    err = scene.plan_error(verts)
    ends = (np.abs(loc[:, 0]) > 4.6).sum()     # vertices on the round ends (past x = 4.5)
    assert ends >= 6, f"round ends kept only {ends} vertices"
    assert len(verts) > 8, f"island collapsed to {len(verts)} vertices"
    assert err < 0.10, f"island outline {err:.3f} m from the truth"
    print(f"  island: {len(verts)} v ({ends} on the round ends), within {100 * err:.1f} cm")


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

def _dist_to_polyline(p, poly, closed):
    dense = _densify(poly, 0.001, closed=closed)
    d, _ = cKDTree(dense).query(p)
    return d


def test_weed_polyline():
    rng = np.random.default_rng(8)
    # A noisy closed rectangle, 3D, plus an open wiggly line.
    rect = _densify(_rect(6, 3), 0.1)
    rect = np.column_stack([rect + rng.normal(0, 0.01, rect.shape),
                            2.0 + rng.normal(0, 0.005, len(rect))])
    t = np.linspace(0, 10, 200)
    line = np.column_stack([t, 0.3 * np.sin(t), 0.01 * t])
    for poly, closed, ptol, htol in ((rect, True, 0.05, 0.05), (line, False, 0.02, 0.02),
                                     (line, False, 0.2, 0.2)):
        w = oc.weed_polyline(poly, ptol, htol, closed=closed).astype(np.float64)
        assert len(w) < len(poly)
        # Never moved: every output row is an input row, in the input's order.
        rows = cKDTree(poly).query(w)
        assert np.all(rows[0] < 1e-5), "a vertex moved"
        if closed:
            rolled = np.roll(rows[1], -int(np.argmin(rows[1])))
            assert np.all(np.diff(rolled) > 0), "order changed"
        else:
            assert np.all(np.diff(rows[1]) > 0), "order changed"
            assert rows[1][0] == 0 and rows[1][-1] == len(poly) - 1, "an end was dropped"
        # Every dropped vertex is within tolerance of the weeded line (3D
        # distance bounds plan and height together: allow both).
        d = _dist_to_polyline(poly, w, closed)
        assert d.max() <= np.hypot(ptol, htol) + 1e-6, f"dropped vertex {d.max():.3f} m away"
        print(f"  weed {'closed' if closed else 'open'} {ptol}/{htol}: "
              f"{len(poly)} -> {len(w)} v, max deviation {100 * d.max():.1f} cm")

    # A gable peak on a straight plan edge: kept with a small height
    # tolerance, dropped with a huge one.
    # Symmetric (both gable ends), so the best-fit plane it is weeded in is
    # level and "height" is Z.
    y = np.linspace(0, 6, 13)
    ring_xy = np.vstack([np.column_stack([np.linspace(0, 8, 17)[:-1], np.zeros(16)]),
                         np.column_stack([np.full(12, 8.0), y[:-1]]),
                         np.column_stack([np.linspace(8, 0, 17)[:-1], np.full(16, 6.0)]),
                         np.column_stack([np.zeros(12), y[::-1][:-1]])])
    gable_poly = np.column_stack([ring_xy, _gable(ring_xy[:, 0], ring_xy[:, 1])])
    peaks = np.array([[8.0, 3.0, _gable(8.0, 3.0)], [0.0, 3.0, _gable(0.0, 3.0)]])
    kept = oc.weed_polyline(gable_poly, 0.05, 0.05, closed=True)
    near = cKDTree(kept).query(peaks)[0]
    assert len(kept) == 6 and near.max() < 1e-4, f"gable peaks weeded out ({len(kept)} v)"
    lost = oc.weed_polyline(gable_poly, 0.05, 1e9, closed=True)
    assert len(lost) == 4 and cKDTree(lost).query(peaks)[0].min() > 1e-3, \
        f"2D weeding should drop the peaks ({len(lost)} v)"
    print(f"  gable peak kept with height_tol 5 cm ({len(kept)} v), "
          f"dropped when height is ignored ({len(lost)} v)")

    # Too few vertices to weed: unchanged.
    tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-4, 0]], float)
    assert np.array_equal(oc.weed_polyline(tri, 1.0, 1.0, closed=True), tri.astype(np.float32))
    seg = np.array([[0, 0, 0], [5, 5, 5]], float)
    assert np.array_equal(oc.weed_polyline(seg, 1.0, 1.0, closed=False), seg.astype(np.float32))
    print("  3 closed / 2 open vertices come back unchanged")


def test_plane_frame():
    rng = np.random.default_rng(9)
    pts = rng.normal(0, 1, (500, 3)) * [5, 2, 0.1] + [100, 200, 30]
    for mode in ("best_fit", "xy", "xz", "yz"):
        f = oc.PlaneFrame(pts, mode)
        uv, h = f.to_2d(pts)
        back = f.to_3d(uv, h)
        assert np.abs(back - pts).max() < 1e-4, f"{mode}: round trip off"
        # Right-handed, orthonormal: a CCW triangle stays CCW in 2D when
        # viewed from the frame's +normal side.
        ax = f.axes
        assert np.allclose(ax @ ax.T, np.eye(3), atol=1e-9), f"{mode}: axes not orthonormal"
        assert np.isclose(np.linalg.det(ax), 1.0), f"{mode}: frame is left-handed"
        tri = np.array([[0, 0], [1, 0], [0, 1]], float)
        world = f.to_3d(tri, np.zeros(3)).astype(np.float64)
        n = np.cross(world[1] - world[0], world[2] - world[0])
        assert n @ ax[2] > 0, f"{mode}: CCW ring flips"
        uv2, _ = f.to_2d(world)
        assert _area(uv2) > 0
    # A vertical wall: best-fit normal is horizontal.
    wall = np.column_stack([rng.random(800) * 10, rng.normal(0, 0.005, 800), rng.random(800) * 4])
    yaw = np.radians(30)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    f = oc.PlaneFrame(wall @ Rz.T, "best_fit")
    assert abs(f.axes[2][2]) < 0.01, f"wall normal has Z {f.axes[2][2]:.3f}"
    # A flat-ish cluster's normal points up.
    assert oc.PlaneFrame(pts, "best_fit").axes[2][2] > 0.99
    try:
        oc.PlaneFrame(pts, "diagonal")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown projection should raise ValueError")
    print("  round trips, right-handed frames, wall normal horizontal, flat normal up")


def test_prepared():
    try:
        oc.Prepared(np.random.default_rng(0).random((7, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("fewer than 8 points should raise ValueError")
    scene = Scene(_rect(10, 4), _gentle, seed=10, outliers=20)
    prep = oc.Prepared(scene.points)
    try:
        assert prep.strays >= 0.8 * scene.n_strays, (prep.strays, scene.n_strays)
        assert len(prep.uv) == len(scene.points) - prep.strays
        # 400 pts/m2 is ~5 cm apart; nearest neighbour typically ~2.5 cm.
        assert 0.01 < prep.spacing < 0.05, prep.spacing
        assert prep.sparse_scale >= np.median(prep.local)
        assert prep.index is not None
    finally:
        prep.release()
    assert prep.index is None
    kept = oc.Prepared(scene.points, drop_strays=False)
    kept.release()
    assert kept.strays == 0 and len(kept.uv) == len(scene.points)
    print(f"  < 8 points refused; {prep.strays} of {scene.n_strays} strays dropped; "
          f"spacing {100 * prep.spacing:.1f} cm")


def test_clean_ring_3d_versus_2d_weeding():
    """``clean_ring`` straight on a dense ring with known heights: the gable
    peaks stay with height_tol = tolerance and go with a huge height_tol."""
    raw = _densify(_rect(8, 6), 0.05)
    raw = raw + np.random.default_rng(11).normal(0, 0.005, raw.shape)

    def height_at(xy):
        return _gable(xy[:, 0], xy[:, 1])

    ring, h = oc.clean_ring(raw, TOL, height_at)
    assert len(ring) == 6, f"clean_ring with 3D weeding: {len(ring)} vertices"
    assert _area(ring) > 0 and len(h) == len(ring)
    ring2, _h2 = oc.clean_ring(raw, TOL, height_at, height_tol=1e9)
    assert len(ring2) == 4, f"clean_ring with height ignored: {len(ring2)} vertices"
    print(f"  clean_ring: 6 v with height, {len(ring2)} v when height is ignored")


def test_outline_feature_contract():
    scene = Scene(_rect(6, 3), _gentle, seed=12)
    feat, info = be.outline_feature(scene.points, "grid_outline", TOL, cell=None)
    assert isinstance(feat, VectorFeature)
    assert feat.geometry_type == "polyline"
    assert feat.geometry["closed"] is True
    assert feat.geometry["vertices"].dtype == np.float32
    assert feat.geometry["vertices"].shape[1] == 3
    for key in ("vertices", "raw_vertices", "strays", "spacing"):
        assert key in info, f"info lacks {key}"
    assert info["vertices"] == len(feat.geometry["vertices"])
    assert info["raw_vertices"] > info["vertices"]
    # None means "the spacing-based default": same answer as leaving it out.
    plain, _ = be.outline_feature(scene.points, "grid_outline", TOL)
    assert np.array_equal(plain.geometry["vertices"], feat.geometry["vertices"])
    for method, key in (("concave_hull", "max_edge"), ("alpha_shape", "alpha"),
                        ("knn_concave_hull", "k")):
        a, _ = be.outline_feature(scene.points, method, TOL, **{key: None})
        b, _ = be.outline_feature(scene.points, method, TOL)
        assert np.array_equal(a.geometry["vertices"], b.geometry["vertices"]), method
    print(f"  polyline, closed, float32; info {sorted(info)}; None params = defaults")


def test_dp3_closed_does_not_depend_on_the_start():
    raw = _densify(_rect(8, 6), 0.05)
    h = _gable(raw[:, 0], raw[:, 1])
    base = set(map(tuple, np.round(raw[oc.dp3_closed(raw, h, TOL, TOL)], 6)))
    for shift in (17, 101, 333):
        r, hh = np.roll(raw, shift, axis=0), np.roll(h, shift)
        idx = oc.dp3_closed(r, hh, TOL, TOL)
        assert np.all(np.diff(idx) > 0), "indices not in ring order"
        assert set(map(tuple, np.round(r[idx], 6))) == base, f"start {shift} changes the result"
    assert len(base) == 6, len(base)
    print("  dp3_closed: same 6 vertices whatever the ring's start")


if __name__ == "__main__":
    try:
        import cupy
        cupy.zeros(1)
    except Exception as exc:
        print(f"Outline clean-up tests need CuPy and a GPU ({type(exc).__name__}); skipped.")
        sys.exit(0)

    t0 = time.perf_counter()
    test_rectangle_every_method()
    test_l_shape()
    test_pitched_roof_keeps_gable_peaks()
    test_vertical_wall_is_outlined_on_the_wall()
    test_sloped_face_heights_are_unbiased()
    test_stray_points_do_not_pull_the_outline()
    test_strays_close_to_the_edge_are_dropped()
    test_curved_island_keeps_its_round_ends()
    test_weed_polyline()
    test_plane_frame()
    test_prepared()
    test_clean_ring_3d_versus_2d_weeding()
    test_outline_feature_contract()
    test_dp3_closed_does_not_depend_on_the_start()
    print(f"\nAll outline clean-up tests passed ({time.perf_counter() - t0:.1f} s).")
