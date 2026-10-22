"""
Tracing edges - where two surfaces meet - with the line tracer.

The Kalman line tracer reads a line as a thin bunch of points (a cable). An
edge is not that: it is where two surfaces meet - a building corner, a roof
ridge, a kerb's top or bottom, a wall on a floor. This tracer keeps
everything else of the line tracer (prediction, coasting across gaps, each
line traced once, joining, stop reasons) and reads the line differently:

- **The reading.** At each step, up to four planes - surfaces - are fitted
  (RANSAC) to the points in a short window round the prediction; where the
  right two cross gives the edge's direction. The window is then seen in
  cross-section, cut square to that direction, up to four straight lines are
  fitted there, and the edge is where two of them meet: of the pairs, the
  one whose meeting point is nearest the prediction and whose shape matches
  the edge picked (which way each surface leaves the corner) is read. So on
  a kerb, where road, face and footpath are all in view, the top edge and
  the bottom edge are told apart. Cut even a few degrees off square, the
  section would smear each surface and read centimetres off.
- **Fits.** Each plane or line is refitted on one unbroken run of its
  points: on a low step the most points lie on a slanted "ghost" across it.
- **The start.** At the pick, two planes are fitted to the cloud round it;
  where they cross gives the edge's direction - the picks need not line up
  along it - and a first reading there learns the edge's shape and the
  noise.
- **Gaps.** Where a surface is hidden or missing, the reading finds no pair
  and the tracer coasts on its prediction, as it does for a cable through a
  tree, until both surfaces are seen again where the edge was predicted.
- **The line** is a smooth curve through the readings (a step apart);
  straight pieces between them would cut the corner of a tight bend.

General: nothing here is about buildings or kerbs - any angle between the
surfaces from ``min_angle_deg`` up, convex or concave, straight or curved.
"""

import numpy as np
from scipy.spatial import cKDTree

from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.line_tracer_base import (
    _COVER_RADII,
    _GATE_SIGMAS,
    _MIN_LINE_POINTS,
    _ON_LINE,
    _lateral,
)

__all__ = ["EdgeLineTracer"]

_RANSAC_TRIES = 200
_MAX_SURFACES = 4
# A fitted surface needs this share of the window's points (and at least
# _MIN_SURFACE_POINTS).
_MIN_SURFACE_SHARE = 0.015
_MIN_SURFACE_POINTS = 8
# Surface fit tolerance = this many noise sigmas.
_FIT_SIGMAS = 3.0
# A reading is cut again, square to the edge's measured direction, when that
# differs from the slice's by more than _REALIGN_MIN_DEG (and less than
# _REALIGN_MAX_DEG); at most _REALIGN_ROUNDS times.
_REALIGN_MIN_DEG = 1.0
_REALIGN_MAX_DEG = 45.0
_REALIGN_ROUNDS = 2
# Points each surface needs behind and ahead of the prediction.
_MIN_EACH_SIDE = 3
# Two surfaces' directions may differ from the learned shape by this much.
_SHAPE_TOLERANCE_DEG = 35.0


class EdgeLineTracer(KalmanLineTracer):
    """
    Trace edges where two surfaces meet.

    Settings as ``KalmanLineTracer``, plus:
        reach: how far across the edge (m) each reading looks - far enough to
            see a good piece of both surfaces, not so far it sees the next
            edge's surfaces as well. Default: half the window.
        min_angle_deg: the shallowest turn between the two surfaces that is
            still an edge (180 - the angle between them for a convex edge).
    """

    def __init__(self, all_points, reach=None, min_angle_deg=15.0, **settings):
        super().__init__(all_points, **settings)
        self.reach = float(reach) if reach else self.window / 2
        self.min_angle_deg = float(min_angle_deg)
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # One edge                                                           #
    # ------------------------------------------------------------------ #

    def _trace_from(self, remaining, lines):
        owner = self._owner
        mini = self._dense_start(remaining)
        pts = self.all_points[mini].astype(float)
        anchor = pts[int(np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1)))]
        start = self._edge_start(anchor, pts)
        if start is None:
            return self._set_aside(remaining, mini)
        point, heading, cfg = start
        fwd, end_f = self._track(point, heading, cfg, cfg.get("bend"))
        back, end_b = self._track(point, -heading, cfg, cfg.get("bend"))
        if end_f[0] == "end":
            fwd = self._edge_tail(fwd, cfg)
        if end_b[0] == "end":
            back = self._edge_tail(back, cfg)
        centerline = np.vstack([back[::-1], fwd[1:]]) if len(back) > 1 else fwd
        # Readings are a step apart; on a tight bend straight pieces between
        # them cut the corner (8 mm on a 4 m radius at 0.5 m steps).
        centerline = _smooth(centerline, self.step / 4)
        ends = [end_b, end_f]
        radius = cfg["radius"]
        pieces = self._cut_same_line(centerline, radius, lines)
        took = 0
        for piece in pieces:
            members = self._points_near(piece, radius)
            members = members[owner[members] < 0]
            if len(members) < _MIN_LINE_POINTS:
                continue
            owner[members] = len(lines)
            lines.append(dict(centerline=piece, radius=radius,
                              ends=ends if len(pieces) == 1 else [_ON_LINE, _ON_LINE]))
            self._index_lines(lines)
            took += len(members)
        # One group of clicks is one edge. Clicks land a few cm round it, on
        # either surface - and on a kerb or a ledge near the next edge too;
        # left-over clicks would only start more lines along the same edge,
        # or on its neighbour.
        if took > 0:
            return remaining[:0]
        covered = owner[remaining] >= 0
        if lines:
            covered |= self._near_lines(self.all_points[remaining], lines)
        if not covered.any():
            rel = self.all_points[remaining].astype(float) - point
            a = rel @ heading
            off = np.linalg.norm(rel - np.outer(a, heading), axis=1)
            covered |= (np.abs(a) < 2 * self.window) & (off < _COVER_RADII * radius)
            covered |= remaining == remaining[np.argmin(np.linalg.norm(rel, axis=1))]
        return remaining[~covered & (owner[remaining] < 0)]

    def _edge_start(self, anchor, picks):
        """Direction from two planes fitted round the pick (where they cross),
        then a first reading there for the edge's position, shape and noise.
        Returns ``(edge point, heading, cfg)`` or None."""
        idx = self._ball(anchor, self.reach)
        if len(idx) < 3 * _MIN_SURFACE_POINTS:
            return None
        q = self.all_points[idx].astype(float)
        planes = _two_planes(q, self._rng)
        if planes is None:
            return None
        (n1, _c1), (n2, _c2), sigma = planes
        heading = np.cross(n1, n2)
        if np.linalg.norm(heading) < np.sin(np.radians(self.min_angle_deg)):
            return None
        heading /= np.linalg.norm(heading)
        if len(picks) >= 2:                      # point it along the picks
            c = picks - picks.mean(0)
            if np.ptp(c @ heading) > 0 and (c @ heading)[np.argmax(np.abs(c @ heading))] < 0:
                heading = -heading
        cfg = dict(radius=max(3 * sigma, 0.005), point_sigma=max(sigma, 1e-3),
                   gate_min=max(3 * sigma, 0.005), gate_max=self.reach / 2,
                   gate_max_coast=self.reach, shape=None, axis=heading.copy())
        reading = self._section(anchor, heading, np.zeros(3), cfg, gate=self.reach / 2)
        if reading is None:
            return None
        point, shape, sigma2, _d = reading
        cfg["shape"] = shape
        aligned = self._aligned_section(point, heading, np.zeros(3), cfg, gate=self.reach / 2)
        if aligned is not None:
            point, shape, sigma2, d = aligned
            heading = d if d @ heading > 0 else -d
        cfg["shape"] = shape
        cfg["point_sigma"] = max(sigma2, 1e-3)
        cfg["radius"] = cfg["gate_min"] = max(3 * sigma2, 0.005)
        # The bend at the start: readings a window ahead and behind, and the
        # circle through the three. Started straight, a tracer loses a tight
        # curve in its first window.
        heading, bend = self._start_bend(point, heading, cfg)
        cfg["bend"] = bend
        return point, heading, cfg

    def _start_bend(self, point, heading, cfg):
        """Heading and curvature vector at *point* from readings one window
        ahead and behind (straight when they cannot be read)."""
        found = []
        for sign in (-1.0, 1.0):
            r = self._aligned_section(point + sign * self.window * heading, heading,
                                      np.zeros(3), cfg, gate=self.reach / 2)
            if r is None:
                return heading, None
            found.append(r[0])
        a, b, c = found[0], point, found[1]
        ab, cb = a - b, c - b
        # circle through a, b, c: curvature vector at b
        cross = np.cross(ab, cb)
        denom = 2 * np.linalg.norm(cross) ** 2
        if denom < 1e-12:
            return _unit(c - a), np.zeros(3)
        centre = b + (np.dot(ab, ab) * np.cross(cb, cross) + np.dot(cb, cb) * np.cross(cross, ab)) / denom
        to_c = centre - b
        k = to_c / max(np.dot(to_c, to_c), 1e-12)
        if np.linalg.norm(k) > self.max_curvature:
            k = k * self.max_curvature / np.linalg.norm(k)
        t = _unit(c - a)
        return (t if t @ heading > 0 else -t), k - (k @ t) * t

    # ------------------------------------------------------------------ #
    # The reading                                                        #
    # ------------------------------------------------------------------ #

    def _read(self, state, cfg, coasting=False, strict=False):
        p, t, k = state.position, state.tangent, state.curvature
        sig = state.lateral_sigma()
        gmax = cfg["gate_max_coast"] if coasting else cfg["gate_max"]
        g = float(np.clip(_GATE_SIGMAS * np.hypot(sig, cfg["point_sigma"]),
                          cfg["gate_min"], gmax))
        reading = self._aligned_section(p, t, k, cfg, gate=g)
        if reading is None:
            return None, "none"
        point, _shape, _s, _d = reading
        # Ran onto an edge already traced: join it, never trace it twice.
        near = self._ball(point, cfg["radius"])
        if len(near):
            taken = self._owner[near] >= 0
            if taken.mean() > 0.5:
                ids = self._owner[near][taken]
                return None, f"joined:{int(np.bincount(np.minimum(ids, 2**20)).argmax())}"
        return point, "ok"

    def _aligned_section(self, p, t, k, cfg, gate):
        """A reading cut square to the edge. Cut at an angle - a few degrees
        off on a curve, after coasting - the slice smears each surface into
        a band, the fitted lines take a part of it, and the corner is read
        centimetres off (or a smeared, dense surface takes every fit and the
        other is never found). So a first look fits planes in 3-D, which no
        angle smears; where the right two cross is the edge's direction, and
        the reading is cut square to that. Returns ``(point, shape, sigma,
        direction)``."""
        d = self._edge_heading(p, t, k, cfg, gate)
        if d is None:
            return None
        for _ in range(_REALIGN_ROUNDS):
            r = self._section(p, d, k - (k @ d) * d, cfg, gate)
            if r is None or float(r[3] @ d) > np.cos(np.radians(_REALIGN_MIN_DEG)):
                break
            d = r[3]
        return None if r is None else (r[0], r[1], r[2], r[3])

    def _edge_heading(self, p, t, k, cfg, gate):
        """The edge's direction near *p*: up to four planes fitted to the
        window's points (bend taken out) and, of the pairs that cross within
        _REALIGN_MAX_DEG of *t* and *gate* of *p*, the one nearest that
        matches the edge's shape. None if no pair does."""
        half = self.window / 2
        span = self.reach + gate
        idx = self._ball(p, np.hypot(half, span))
        if len(idx) < 2 * _MIN_SURFACE_POINTS:
            return None
        a, lat = _lateral(self.all_points[idx].astype(float) - p, t)
        lat = lat - 0.5 * np.outer(a ** 2, k)
        keep = (np.abs(a) < half) & (np.linalg.norm(lat, axis=1) < span)
        xyz = np.outer(a[keep], t) + lat[keep]
        tol = _FIT_SIGMAS * cfg["point_sigma"]
        planes = _fit_planes(xyz, tol, self._rng, t)
        # A plane on a sliver (a surface just ending, at a gap) can tilt
        # freely about it: each must run along most of the window.
        planes = [pl for pl in planes
                  if np.ptp(np.percentile(pl[2] @ t, [5, 95])) > half]
        ref = cfg.get("axis", t)
        best = None
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                (n1, c1, q1), (n2, c2, q2) = planes[i], planes[j]
                d = np.cross(n1, n2)
                if np.linalg.norm(d) < np.sin(np.radians(self.min_angle_deg)):
                    continue
                d = _unit(d)
                d = d if d @ t >= 0 else -d
                if d @ t < np.cos(np.radians(_REALIGN_MAX_DEG)):
                    continue
                # the point of the crossing nearest p, and which way each
                # plane leaves it (seen in the section square to d)
                x = np.linalg.solve(np.stack([n1, n2, d]), [n1 @ c1, n2 @ c2, 0.0])
                dist = float(np.linalg.norm(x))
                if dist > gate:
                    continue
                e1, e2 = _section_axes(d if d @ ref >= 0 else -d)
                away = []
                for n, q in ((n1, q1), (n2, q2)):
                    w = np.cross(d, n)
                    along = (q - x) @ w
                    lo, hi = np.percentile(along, 5), np.percentile(along, 95)
                    far, near, sign = (hi, -lo, 1.0) if hi >= -lo else (-lo, hi, -1.0)
                    if far < 3 * tol or near > 0.25 * far:
                        break
                    w2 = np.array([w @ e1, w @ e2]) * sign
                    away.append(w2 / max(np.linalg.norm(w2), 1e-12))
                if len(away) < 2:
                    continue
                mismatch = _shape_mismatch(tuple(away), cfg.get("shape"))
                if mismatch is None:
                    continue
                key = dist + mismatch * self.reach / np.pi
                if best is None or key < best[0]:
                    best = (key, d)
        return None if best is None else best[1]

    def _section(self, p, t, k, cfg, gate):
        """The edge in the cross-section at *p* looking along *t*: fit up to
        four surfaces, read where the right two meet. Returns ``(3-D edge
        point, shape, noise sigma, edge direction)`` or None."""
        half = self.window / 2
        span = self.reach + gate
        idx = self._ball(p, np.hypot(half, span))
        if len(idx) < 2 * _MIN_SURFACE_POINTS:
            return None
        a, lat = _lateral(self.all_points[idx].astype(float) - p, t)
        lat = lat - 0.5 * np.outer(a ** 2, k)
        # Axes from the edge's own direction, not the way the trace travels:
        # turned round, axes built from the travel direction come out
        # mirrored and the learned shape no longer matches.
        ref = cfg.get("axis", t)
        e1, e2 = _section_axes(t if t @ ref >= 0 else -t)
        uv = np.stack([lat @ e1, lat @ e2], 1)
        keep = (np.abs(a) < half) & (np.linalg.norm(uv, axis=1) < span)
        uv, a, lat = uv[keep], a[keep], lat[keep]
        if len(uv) < 2 * _MIN_SURFACE_POINTS:
            return None
        tol = _FIT_SIGMAS * cfg["point_sigma"]
        surfaces = _fit_lines(uv, tol, self._rng)
        best = None
        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                pair = _refit_pair(surfaces[i], surfaces[j], tol)
                corner = _corner(*pair, self.min_angle_deg, tol)
                if corner is None:
                    continue
                x, d1, d2 = corner
                mismatch = _shape_mismatch((d1, d2), cfg.get("shape"))
                if mismatch is None:
                    continue
                dist = float(np.linalg.norm(x))
                if dist > gate:
                    continue
                key = dist + mismatch * self.reach / np.pi
                if best is None or key < best[0]:
                    sig = np.sqrt(np.mean(np.concatenate([surfaces[i][3], surfaces[j][3]]) ** 2))
                    best = (key, x, (d1, d2), sig, pair)
        if best is None:
            return None
        _key, x, shape, sig, pair = best
        if not _continues_both_ways(uv, a, x, shape, tol, self.reach):
            return None
        return p + x[0] * e1 + x[1] * e2, shape, float(sig), \
            _edge_direction(np.outer(a, t) + lat, uv, pair, tol, t)

    def _edge_tail(self, path, cfg):
        """Carry an ended trace on to where the edge really ends: a reading
        needs the edge on both sides of it, so the trace stops up to half a
        window short. One-sided reading at the last point: the furthest
        point still at the corner, ahead, along the last heading."""
        if len(path) < 2:
            return path
        tip, heading = path[-1], path[-1] - path[-2]
        norm = float(np.linalg.norm(heading))
        if norm < 1e-9:
            return path
        heading = heading / norm
        half = self.window / 2
        idx = self._ball(tip + heading * half / 2, np.hypot(half / 2, self.reach))
        if len(idx) == 0:
            return path
        a, lat = _lateral(self.all_points[idx].astype(float) - tip, heading)
        near = (a > 0) & (a < half) & (np.linalg.norm(lat, axis=1) < 3 * cfg["radius"])
        if not near.any():
            return path
        return np.vstack([path, tip + heading * float(a[near].max())])

    # The thin-line searches for a kink or the far side of a gap do not
    # apply: an edge is found again by its reading where it was predicted.
    def _find_kink(self, kt, last_good, gaps, cfg):
        return None

    def _bridge(self, kt, last_good, a_from, cfg):
        return None

    def _line_radius(self, line, seed_radius):
        return seed_radius


# --------------------------------------------------------------------- #
# Geometry                                                              #
# --------------------------------------------------------------------- #

def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _smooth(path, spacing):
    """A smooth curve through the vertices (centripetal Catmull-Rom), with
    points about *spacing* apart; the vertices stay on it."""
    if len(path) < 3:
        return path
    ext = np.vstack([2 * path[0] - path[1], path, 2 * path[-1] - path[-2]])
    out = [path[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        t1 = np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        if min(t1, t2 - t1, t3 - t2) < 1e-9:
            out.append(p2)
            continue
        n = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / spacing)))
        u = (t1 + (t2 - t1) * np.arange(1, n + 1) / n)[:, None]
        a1 = ((t1 - u) * p0 + u * p1) / t1
        a2 = ((t2 - u) * p1 + (u - t1) * p2) / (t2 - t1)
        a3 = ((t3 - u) * p2 + (u - t2) * p3) / (t3 - t2)
        b1 = ((t2 - u) * a1 + u * a2) / t2
        b2 = ((t3 - u) * a2 + (u - t1) * a3) / (t3 - t1)
        out.extend(((t2 - u) * b1 + (u - t1) * b2) / (t2 - t1))
    return np.array(out)


def _section_axes(t):
    """Two axes across the edge: e2 as near 'up' as the edge allows."""
    up = np.array([0.0, 0.0, 1.0])
    if abs(t @ up) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    e2 = up - (up @ t) * t
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, t)
    return e1, e2


def _fit_lines(uv, tol, rng):
    """Up to _MAX_SURFACES straight lines in the section, one after another
    (RANSAC, refitted). Each: (point, unit direction, inlier coords,
    residuals, inlier indices into uv)."""
    out = []
    rest, ids = uv, np.arange(len(uv))
    need = max(_MIN_SURFACE_POINTS, int(_MIN_SURFACE_SHARE * len(uv)))
    for _ in range(_MAX_SURFACES):
        if len(rest) < need:
            break
        n = len(rest)
        i = rng.integers(0, n, _RANSAC_TRIES)
        j = rng.integers(0, n, _RANSAC_TRIES)
        best, best_in = None, None
        for a, b in zip(i, j):
            d = rest[b] - rest[a]
            norm = np.linalg.norm(d)
            if norm < 2 * tol:
                continue
            nrm = np.array([-d[1], d[0]]) / norm
            inl = np.abs((rest - rest[a]) @ nrm) < tol
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (rest[a], d / norm), inl
        if best_in is None or best_in.sum() < need:
            break
        # Refit on one unbroken run of the inliers. Where a step is only a
        # few tolerances high (a low kerb), the most inliers lie on a slanted
        # line across it - a piece of the top, a piece of the bottom; the
        # biggest run is a piece of one real surface, and its own fit is that
        # surface.
        inl = best_in
        for _r in range(3):
            pts = rest[_largest_run(rest, inl, best[1], tol)]
            if len(pts) < need:
                break
            c = pts.mean(0)
            v = np.linalg.eigh((pts - c).T @ (pts - c))[1][:, -1]
            best = (c, v)
            inl = np.abs((rest - c) @ np.array([-v[1], v[0]])) < tol
        c, v = best
        inl = _largest_run(rest, inl, v, tol)
        if inl.sum() < need:
            break
        res = (rest - c) @ np.array([-v[1], v[0]])
        out.append((c, v, rest[inl], res[inl], ids[inl]))
        rest, ids = rest[~inl], ids[~inl]
    return out


def _largest_run(uv, inl, v, tol):
    """The inliers in the longest unbroken run along direction *v* (2-D or
    3-D points): a break is a gap wider than 3 tolerances and 10 typical
    spacings."""
    idx = np.flatnonzero(inl)
    if len(idx) < 2:
        return inl
    along = uv[idx] @ v
    order = np.argsort(along)
    gaps = np.diff(along[order])
    cut = np.flatnonzero(gaps > max(3 * tol, 10 * float(np.median(gaps))))
    bounds = np.concatenate([[0], cut + 1, [len(idx)]])
    k = int(np.argmax(np.diff(bounds)))
    out = np.zeros_like(inl)
    out[idx[order[bounds[k]:bounds[k + 1]]]] = True
    return out


def _refit_pair(s1, s2, tol, rounds=3):
    """Refit two surfaces jointly: each of their points goes to whichever
    line it is nearer, both are refitted, a few times. Fitted one after the
    other, the first takes the second's points near the corner - and on a
    shallow edge that moves the corner."""
    pts = np.vstack([s1[2], s2[2]])
    ids = np.concatenate([s1[4], s2[4]])
    lines = [(s1[0], s1[1]), (s2[0], s2[1])]
    for _ in range(rounds):
        d = np.stack([np.abs((pts - c) @ np.array([-v[1], v[0]])) for c, v in lines], 1)
        which = np.argmin(d, 1)
        new = []
        for k in range(2):
            q = pts[(which == k) & (d[:, k] < tol)]
            if len(q) < _MIN_SURFACE_POINTS:
                return s1, s2
            c = q.mean(0)
            v = np.linalg.eigh((q - c).T @ (q - c))[1][:, -1]
            new.append((c, v if v @ lines[k][1] >= 0 else -v))
        lines = new
    out = []
    for c, v in lines:
        nrm = np.array([-v[1], v[0]])
        res = (pts - c) @ nrm
        inl = np.abs(res) < tol
        out.append((c, v, pts[inl], res[inl], ids[inl]))
    return out[0], out[1]


def _corner(s1, s2, min_angle_deg, tol):
    """Where two fitted surfaces meet, if they form an edge: they turn by at
    least min_angle_deg, and each surface's points run up to the meeting
    point (it is an end of both). Returns (point, away1, away2) - the
    directions each surface leaves the corner - or None."""
    c1, v1, p1 = s1[:3]
    c2, v2, p2 = s2[:3]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(cross) < np.sin(np.radians(min_angle_deg)):
        return None
    # c1 + s v1 = c2 + u v2
    s = ((c2 - c1)[0] * v2[1] - (c2 - c1)[1] * v2[0]) / cross
    x = c1 + s * v1
    away = []
    for c, v, p in ((c1, v1, p1), (c2, v2, p2)):
        along = (p - x) @ v
        lo, hi = np.percentile(along, 5), np.percentile(along, 95)
        # The corner must be at one end of the surface's points: little of
        # it on the near side, against the surface's own length (a kerb face
        # may be only a few cm), which must be more than its noise.
        far, near, sign = (hi, -lo, 1.0) if hi >= -lo else (-lo, hi, -1.0)
        if far < 3 * tol or near > 0.25 * far:
            return None
        away.append(sign * v)
    return x, away[0], away[1]


def _edge_direction(xyz, uv, pair, tol, t, rounds=2):
    """Where the two surfaces' planes cross. Each plane is fitted in 3-D
    (*xyz*: the slice's points, bend taken out), first to its points away
    from the other surface - near the corner the other surface's points are
    within the tolerance too - then again to all the slice's points near
    that plane: points picked by a line in a slice cut at an angle lean the
    plane towards the slice. *t* if unclear."""
    planes = []
    for me, other in ((pair[0], pair[1]), (pair[1], pair[0])):
        c, v = other[0], other[1]
        far = np.abs((uv[me[4]] - c) @ np.array([-v[1], v[0]])) > 2 * tol
        planes.append(xyz[me[4][far]])
    fitted = []
    for q in planes:
        if len(q) < _MIN_SURFACE_POINTS:
            return t
        fitted.append(_plane(q))
    for _ in range(rounds):
        new = []
        for k in range(2):
            (n, c), (n2, c2) = fitted[k], fitted[1 - k]
            near = (np.abs((xyz - c) @ n) < tol) & (np.abs((xyz - c2) @ n2) > 2 * tol)
            if near.sum() < _MIN_SURFACE_POINTS:
                return t
            new.append(_plane(xyz[near]))
        fitted = new
    d = np.cross(fitted[0][0], fitted[1][0])
    if np.linalg.norm(d) < 1e-6:
        return t
    d = _unit(d)
    return d if d @ t >= 0 else -d


def _fit_planes(xyz, tol, rng, t, most=_MAX_SURFACES):
    """Up to *most* planes, one after another (RANSAC, refitted on one
    unbroken run of their inliers across the edge direction *t*, as
    _fit_lines; with no *t*, across both of the plane's own axes). Each:
    (unit normal, centre, inlier points)."""
    out = []
    rest = xyz
    need = max(_MIN_SURFACE_POINTS, int(_MIN_SURFACE_SHARE * len(xyz)))
    for _ in range(most):
        if len(rest) < need:
            break
        tri = rng.integers(0, len(rest), (_RANSAC_TRIES, 3))
        nrm = np.cross(rest[tri[:, 1]] - rest[tri[:, 0]], rest[tri[:, 2]] - rest[tri[:, 0]])
        size = np.linalg.norm(nrm, axis=1)
        ok = size > 1e-9
        if not ok.any():
            break
        nrm, base = nrm[ok] / size[ok, None], rest[tri[ok, 0]]
        counts = [int(np.sum(np.abs((rest - b) @ n) < tol)) for n, b in zip(nrm, base)]
        k = int(np.argmax(counts))
        if counts[k] < need:
            break
        n, c = nrm[k], base[k]
        inl = np.abs((rest - c) @ n) < tol
        for _r in range(3):
            run = _plane_run(rest, inl, n, t, tol)
            if run.sum() < need:
                break
            n, c = _plane(rest[run])
            inl = np.abs((rest - c) @ n) < tol
        inl = _plane_run(rest, inl, n, t, tol)
        if inl.sum() < need:
            break
        out.append((n, c, rest[inl]))
        rest = rest[~inl]
    return out


def _plane_run(xyz, inl, n, t, tol):
    """_largest_run for a plane's inliers: across *t* in the plane, or
    with no *t* along each of the inliers' two in-plane axes."""
    if t is not None:
        return _largest_run(xyz, inl, _unit(np.cross(t, n)), tol)
    if inl.sum() < 3:
        return inl
    q = xyz[inl]
    axes = np.linalg.eigh(np.cov((q - q.mean(0)).T))[1]
    for k in (2, 1):
        inl = _largest_run(xyz, inl, axes[:, k], tol)
    return inl


def _plane(q):
    c = q.mean(0)
    return np.linalg.eigh(np.cov((q - c).T))[1][:, 0], c


def _continues_both_ways(uv, a, x, shape, tol, reach):
    """Both surfaces run on behind AND ahead of the prediction, near the
    corner (within half the reach of it) - else a reading half past the
    edge's end still finds it in the points behind, and the line overshoots."""
    for d in shape:
        nrm = np.array([-d[1], d[0]])
        rel = uv - x
        along = rel @ d
        on = (np.abs(rel @ nrm) < tol) & (along > -tol) & (along < reach / 2)
        if min(np.sum(on & (a < 0)), np.sum(on & (a > 0))) < _MIN_EACH_SIDE:
            return False
    return True


def _shape_mismatch(dirs, shape):
    """How far the two surfaces' away-directions are from the learned
    shape (radians, best pairing); 0 when nothing is learned yet; None when
    beyond tolerance."""
    if shape is None:
        return 0.0
    ang = lambda a, b: float(np.arccos(np.clip(a @ b, -1, 1)))
    m = min(ang(dirs[0], shape[0]) + ang(dirs[1], shape[1]),
            ang(dirs[0], shape[1]) + ang(dirs[1], shape[0]))
    if m > 2 * np.radians(_SHAPE_TOLERANCE_DEG):
        return None
    return m


def _two_planes(q, rng):
    """The two surfaces round a pick that meet: planes fitted one after
    another (as _fit_planes; tolerance from the first plane's own noise),
    and of those the pair at the largest angle - at a kerb, road and
    footpath are parallel and only the face meets either. Returns ((n1, c1),
    (n2, c2), noise sigma) or None."""
    # The noise, from small patches (each mostly on one surface): a fixed
    # first tolerance is too coarse for a low step and too fine for a noisy
    # scan.
    k = min(16, len(q))
    near = cKDTree(q).query(q[rng.choice(len(q), min(60, len(q)), replace=False)], k)[1]
    local = [np.sqrt(max(np.linalg.eigvalsh(np.cov(q[nb].T))[0], 0.0)) for nb in near]
    first = _fit_planes(q, max(_FIT_SIGMAS * float(np.median(local)), 0.003), rng, None, most=1)
    if not first:
        return None
    n, c, inl = first[0]
    res = (inl - c) @ n
    sigma = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 1e-3)
    planes = _fit_planes(q, max(_FIT_SIGMAS * sigma, 0.003), rng, None)
    if len(planes) < 2:
        return None
    pairs = [(i, j) for i in range(len(planes)) for j in range(i + 1, len(planes))]
    i, j = max(pairs, key=lambda ij: np.linalg.norm(np.cross(planes[ij[0]][0], planes[ij[1]][0])))
    return planes[i][:2], planes[j][:2], sigma
