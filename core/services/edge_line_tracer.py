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
from core.services.linear_region_grower import GrownLine
from core.services.line_tracer_base import (
    _COVER_RADII,
    _GATE_SIGMAS,
    _MIN_LINE_POINTS,
    _ON_LINE,
    _lateral,
)

__all__ = ["EdgeLineTracer", "GrownEdge", "MARK_LOWERED", "MARK_BRIDGED"]


class GrownEdge(GrownLine):
    """A traced edge in the grower's format, plus ``marks``: the stretches
    not read as a normal edge, ``[(MARK_LOWERED or MARK_BRIDGED, (M, 3))]``."""
    marks = ()

_RANSAC_TRIES = 200
# RANSAC guesses are scored in batches, all at once, against at most
# _RANSAC_SCORE_POINTS of the points; it stops once the best so far would be
# found with _RANSAC_CONFIDENCE (the usual adaptive rule). The chosen surface
# is then refitted to all the points.
_RANSAC_BATCH = 40
_RANSAC_SCORE_POINTS = 512
_RANSAC_CONFIDENCE = 0.999
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
# Readings taken either side of the start (half-windows each way).
_START_STEPS = 3
# Where an edge fades out - a kerb lowered at a driveway or a pram ramp - its
# face shrinks to nothing and the surfaces meet at a few degrees. Such a weak
# fold is read only when the normal reading fails, only this close to the
# prediction (as a share of Look Across), and for at most Max Gap.
_WEAK_MIN_ANGLE_DEG = 2.0
_WEAK_GATE = 0.25
# Dynamic step: a prediction one step ahead may miss a curve by about
# step^2 * bend / 2; the step is shortened to keep that under _STEP_MISS (m),
# down to step / _STEP_SHRINK at most. After a reading that came in further
# than _STEP_SURPRISE gates from its prediction, half a step.
_STEP_MISS = 0.01
_STEP_SHRINK = 5.0
_STEP_SURPRISE = 0.5
# Kinds of stretch on a traced edge (``line["marks"]``).
MARK_LOWERED = "lowered"        # read from a weak fold: the edge fades out
MARK_BRIDGED = "bridged"        # nothing read: coasted across and found again


class EdgeLineTracer(KalmanLineTracer):
    """
    Trace edges where two surfaces meet.

    Settings as ``KalmanLineTracer``, plus:
        reach: how far across the edge (m) each reading looks - far enough to
            see a good piece of both surfaces, not so far it sees the next
            edge's surfaces as well. Default: half the window.
        min_angle_deg: the shallowest turn between the two surfaces that is
            still an edge (180 - the angle between them for a convex edge).
        follow_lowered: where an edge that opens upward (a kerb's bottom)
            fades out - a driveway, a pram ramp - follow the gentle fold
            that carries it on, instead of coasting across. Marked
            ``MARK_LOWERED`` on the line.
        dynamic_step: shorten the step where the edge turns (and after a
            reading off its prediction) - ``step`` is then the longest.
    """

    # One click near an edge is enough: the surfaces round it give the
    # edge's direction, and the corner nearest the click is the edge.
    MIN_GROUP_PICKS = 1

    def __init__(self, all_points, reach=None, min_angle_deg=15.0,
                 follow_lowered=False, dynamic_step=True, **settings):
        super().__init__(all_points, **settings)
        self.reach = float(reach) if reach else self.window / 2
        self.min_angle_deg = float(min_angle_deg)
        self.follow_lowered = bool(follow_lowered)
        self.dynamic_step = bool(dynamic_step)
        self._surprised = False
        self._last_step = self.step
        self._rng = np.random.default_rng(0)
        self._last_direction = None
        self._kinds = []                 # per accepted reading: "ok" / MARK_LOWERED
        self._weak_run = 0.0

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
        kinds_f = self._vertex_kinds(len(fwd))
        back, end_b = self._track(point, -heading, cfg, cfg.get("bend"))
        kinds_b = self._vertex_kinds(len(back))
        if end_f[0] == "end":
            fwd = self._edge_tail(fwd, cfg)
            kinds_f += ["ok"] * (len(fwd) - len(kinds_f))
        if end_b[0] == "end":
            back = self._edge_tail(back, cfg)
            kinds_b += ["ok"] * (len(back) - len(kinds_b))
        if len(back) > 1:
            centerline = np.vstack([back[::-1], fwd[1:]])
            kinds = kinds_b[::-1] + kinds_f[1:]
        else:
            centerline, kinds = fwd, kinds_f
        marks = self._marks(centerline, kinds)
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
                              ends=ends if len(pieces) == 1 else [_ON_LINE, _ON_LINE],
                              marks=marks if piece is pieces[0] else []))
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
                   surface_sigma=max(sigma, 1e-3),
                   gate_min=max(3 * sigma, 0.005), gate_max=self.reach / 2,
                   gate_max_coast=self.reach, shape=None, axis=heading.copy())
        # Cut square to the edge as every reading is: a start a few cm off
        # the edge (one click beside it) with the planes' direction a little
        # off smears a sparse surface into bands.
        reading = self._aligned_section(anchor, heading, np.zeros(3), cfg, gate=self.reach / 2)
        if reading is None:
            return None
        point, shape, sigma2, d = reading
        heading = d if d @ heading > 0 else -d
        cfg["shape"] = shape
        aligned = self._aligned_section(point, heading, np.zeros(3), cfg, gate=self.reach / 2)
        if aligned is not None:
            point, shape, sigma2, d = aligned
            heading = d if d @ heading > 0 else -d
        cfg["shape"] = shape
        cfg["point_sigma"] = cfg["surface_sigma"] = max(sigma2, 1e-3)
        cfg["radius"] = cfg["gate_min"] = max(3 * sigma2, 0.005)
        # The edge's own points: within the noise of it - and, in a sparse
        # scan, within the spacing of the points (else too few are near
        # enough for the line to count).
        spacing = float(np.median(cKDTree(q).query(q, 2)[0][:, 1]))
        cfg["radius"] = max(cfg["radius"], 2 * spacing)
        # Heading, bend and how far single readings scatter, from a short run
        # of readings either side. Started straight, a tracer loses a tight
        # curve in its first window; told readings are as exact as the
        # scanner, it swings on every wobble of a rounded kerb top.
        heading, bend, read_sigma = self._start_run(point, heading, cfg)
        cfg["bend"] = bend
        cfg["point_sigma"] = max(read_sigma, cfg["surface_sigma"])
        cfg["gate_min"] = max(3 * cfg["point_sigma"], 0.005)
        return point, heading, cfg

    def _start_run(self, point, heading, cfg):
        """Readings stepped out _START_STEPS half-windows either way from
        *point*, each aimed along the direction the last one measured. A
        curve fitted through them (along their arc length) gives the heading
        and curvature vector at *point*, and its residuals how far one
        reading strays. ``(heading, bend or None, reading sigma)``."""
        found = {0: point}
        for sign in (-1.0, 1.0):
            p, d = point, sign * heading
            for k in range(1, _START_STEPS + 1):
                r = self._aligned_section(p + d * self.window / 2, d, np.zeros(3), cfg,
                                          gate=self.reach / 2)
                if r is None:
                    break
                p, d = r[0], (r[3] if r[3] @ d > 0 else -r[3])
                found[sign * k] = p
        pts = np.array([found[k] for k in sorted(found)])
        if len(pts) < 3:
            return heading, None, cfg["surface_sigma"]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        arc -= arc[sorted(found).index(0)]
        coef = np.polyfit(arc, pts, 2)                    # rows: s^2, s, 1
        t = _unit(coef[1])
        t = t if t @ heading > 0 else -t
        k = 2 * coef[0] / max(np.dot(coef[1], coef[1]), 1e-12)
        k = k - (k @ t) * t
        if np.linalg.norm(k) > self.max_curvature:
            k = k * self.max_curvature / np.linalg.norm(k)
        if len(pts) < 5:
            return t, k, cfg["surface_sigma"]
        res = pts - (np.outer(arc ** 2, coef[0]) + np.outer(arc, coef[1]) + coef[2])
        res = res - np.outer(res @ t, t)
        sigma = float(np.sqrt(np.sum(res ** 2) / (len(pts) - 3) / 2))
        return t, k, sigma

    # ------------------------------------------------------------------ #
    # The reading                                                        #
    # ------------------------------------------------------------------ #

    def _read(self, state, cfg, coasting=False, strict=False):
        p, t, k = state.position, state.tangent, state.curvature
        sig = state.lateral_sigma()
        gmax = cfg["gate_max_coast"] if coasting else cfg["gate_max"]
        g = float(np.clip(_GATE_SIGMAS * np.hypot(sig, cfg["point_sigma"]),
                          cfg["gate_min"], gmax))
        reading, kind = self._aligned_section(p, t, k, cfg, gate=g), "ok"
        if reading is None and self.follow_lowered:
            reading, kind = self._weak_reading(p, t, k, cfg, g), MARK_LOWERED
        if reading is None:
            return None, "none"
        point, _shape, _s, self._last_direction = reading
        self._surprised = float(np.linalg.norm(point - p)) > _STEP_SURPRISE * g
        # Ran onto an edge already traced: join it, never trace it twice.
        near = self._ball(point, cfg["radius"])
        if len(near):
            taken = self._owner[near] >= 0
            if taken.mean() > 0.5:
                ids = self._owner[near][taken]
                return None, f"joined:{int(np.bincount(np.minimum(ids, 2**20)).argmax())}"
        if kind == MARK_LOWERED:
            self._weak_run += self._last_step
            # a weak fold gives no reliable direction
            self._last_direction = None
        else:
            self._weak_run = 0.0
        self._kinds.append(kind)
        return point, "ok"

    def _weak_reading(self, p, t, k, cfg, gate):
        """Where the edge fades out (a lowered kerb), the fold between the
        surfaces that carry on: two surfaces meeting at only a few degrees,
        one leaving the edge as the learned one does, the other at least to
        the same side. Only near the prediction, for at most Max Gap in a
        row, and only for an edge that opens upward - a kerb's bottom, a
        wall's foot: at a ramp its fold runs on along the road. A kerb's top
        has no such fold to follow and is bridged (coasted across)."""
        if self._weak_run >= self.max_gap or not _opens_upward(cfg.get("shape")):
            return None
        r = self._section(p, t, k, cfg, _WEAK_GATE * self.reach, weak=True)
        return None if r is None else (r[0], r[1], r[2], t)

    def _track(self, start, heading, cfg, bend=None):
        self._kinds, self._weak_run, self._surprised = [], 0.0, False
        return super()._track(start, heading, cfg, bend)

    def _next_step(self, kt, gaps):
        """The step for the next reading: shorter where the edge turns (a
        prediction one step ahead would miss the curve by about
        step^2 x bend / 2 - kept under _STEP_MISS), and half after a reading
        that came in off its prediction. Coasting keeps the full step."""
        ds = self.step
        if self.dynamic_step and gaps == 0:
            bend = float(np.linalg.norm(kt.state.curvature))
            if bend > 1e-9:
                ds = min(ds, float(np.sqrt(2 * _STEP_MISS / bend)))
            if self._surprised:
                ds = min(ds, self.step / 2)
            ds = max(ds, self._min_step())
        self._last_step = ds
        return ds

    def _min_step(self):
        return self.step / _STEP_SHRINK if self.dynamic_step else self.step

    def _vertex_kinds(self, n):
        """Kind of each vertex of the path ``_track`` just returned: the
        start, then one per accepted reading (a join adds one more)."""
        kinds = ["ok"] + list(self._kinds)
        return (kinds + ["ok"] * n)[:n]

    def _marks(self, path, kinds):
        """The stretches of *path* not read as a normal edge: read from a
        weak fold (lowered), or coasted across with nothing read (bridged -
        readings further apart than a step and a half). ``[(kind, (M, 3))]``,
        consecutive pieces of one kind joined."""
        out = []
        for i in range(len(path) - 1):
            if MARK_LOWERED in (kinds[i], kinds[i + 1]):
                kind = MARK_LOWERED
            elif np.linalg.norm(path[i + 1] - path[i]) > 1.5 * self.step:
                kind = MARK_BRIDGED
            else:
                continue
            if out and out[-1][0] == kind and out[-1][2] == i:
                out[-1][1].append(path[i + 1])
                out[-1][2] = i + 1
            else:
                out.append([kind, [path[i], path[i + 1]], i + 1])
        return [(kind, _smooth(np.array(pts), self.step / 4)) for kind, pts, _ in out]

    def _to_grown_line(self, label, line):
        grown = GrownEdge(*super()._to_grown_line(label, line))
        grown.marks = line.get("marks", [])
        return grown

    def _aligned_section(self, p, t, k, cfg, gate):
        """``_aligned_once`` over the full window - and, when that fails,
        over a half and then a quarter of it. On a tight curve whose bend is
        not known yet (a start on a kerb return) or is changing, a long
        slice cut along the predicted bend smears the edge; a short one
        hardly bends. The full window first: on a straight it reads best."""
        for half in (self.window / 2, self.window / 4, self.window / 8):
            r = self._aligned_once(p, t, k, cfg, gate, half)
            if r is not None:
                return r
        return None

    def _aligned_once(self, p, t, k, cfg, gate, half):
        """A reading cut square to the edge. Cut at an angle - a few degrees
        off on a curve, after coasting - the slice smears each surface into
        a band, the fitted lines take a part of it, and the corner is read
        centimetres off (or a smeared, dense surface takes every fit and the
        other is never found). So a first look fits planes in 3-D, which no
        angle smears; where the right two cross is the edge's direction, and
        the reading is cut square to that. Returns ``(point, shape, sigma,
        direction)``."""
        d = self._edge_heading(p, t, k, cfg, gate, half)
        if d is None:
            # No two planes cross here: perhaps a step whose face is hidden.
            r = self._section(p, t, k, cfg, gate, step_only=True, half=half)
            return None if r is None else (r[0], r[1], r[2], r[3])
        for _ in range(_REALIGN_ROUNDS):
            r = self._section(p, d, k - (k @ d) * d, cfg, gate, half=half)
            if r is None or float(r[3] @ d) > np.cos(np.radians(_REALIGN_MIN_DEG)):
                break
            d = r[3]
        return None if r is None else (r[0], r[1], r[2], r[3])

    def _edge_heading(self, p, t, k, cfg, gate, half=None):
        """The edge's direction near *p*: up to four planes fitted to the
        window's points (bend taken out) and, of the pairs that cross within
        _REALIGN_MAX_DEG of *t* and *gate* of *p*, the one nearest that
        matches the edge's shape. None if no pair does."""
        half = self.window / 2 if half is None else half
        span = self.reach + gate
        idx = self._ball(p, np.hypot(half, span))
        if len(idx) < 2 * _MIN_SURFACE_POINTS:
            return None
        a, lat = _lateral(self.all_points[idx].astype(float) - p, t)
        lat = lat - 0.5 * np.outer(a ** 2, k)
        keep = (np.abs(a) < half) & (np.linalg.norm(lat, axis=1) < span)
        xyz = np.outer(a[keep], t) + lat[keep]
        tol = _FIT_SIGMAS * cfg["surface_sigma"]
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

    def _section(self, p, t, k, cfg, gate, step_only=False, weak=False, half=None):
        """The edge in the cross-section at *p* looking along *t*: fit up to
        four surfaces, read where the right two meet - or, where none meet,
        where the upper of two parallel surfaces ends at a step whose face is
        hidden (``_step_edge``; only that when *step_only*). With *weak*, a
        fold of a few degrees where the edge fades out (``_weak_reading``).
        Returns ``(3-D edge point, shape, noise sigma, edge direction)`` or
        None. *half*: the slice's half-length (default half the window)."""
        half = self.window / 2 if half is None else half
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
        tol = _FIT_SIGMAS * cfg["surface_sigma"]
        if weak:
            fold = _weak_fold(uv, a, tol, gate, cfg.get("shape"),
                              self.min_angle_deg, self.reach)
            if fold is None:
                return None
            x, shape, sig = fold
            return p + x[0] * e1 + x[1] * e2, shape, sig, t
        surfaces = _fit_lines(uv, tol, self._rng)
        best = None
        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                if step_only:
                    break
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
            # A hidden face hides an upward-opening edge's foot: only a top
            # (downward-opening) edge is read from a step.
            if _opens_upward(cfg.get("shape")):
                return None
            step = _step_edge(surfaces, a, tol, gate, cfg.get("shape"),
                              self.min_angle_deg, self.reach)
            if step is None:
                return None
            x, shape, sig = step
            return p + x[0] * e1 + x[1] * e2, shape, sig, t
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

    def _measured_direction(self):
        # Where the two surfaces' planes cross: the edge's direction at each
        # reading. Given to the tracker with the point, the bend follows a
        # curve as it tightens and eases - from the point alone it lags, and
        # past a kerb's curve it kept turning off the straight after it.
        return self._last_direction

    def _ball(self, centre, radius):
        # In row order: the GPU query's order depends on its memory state,
        # and the random surface fits sample by position in the list - the
        # same edge must trace the same way every time.
        return np.sort(super()._ball(centre, radius))

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
        found = _ransac(rest, tol, rng, 2)
        if found is None:
            break
        base, nrm = found
        best_in = np.abs((rest - base) @ nrm) < tol
        if best_in.sum() < need:
            break
        best = (base, np.array([nrm[1], -nrm[0]]))
        # Refit on one unbroken run of the inliers. Where a step is only a
        # few tolerances high (a low kerb), the most inliers lie on a slanted
        # line across it - a piece of the top, a piece of the bottom; the
        # biggest run is a piece of one real surface, and its own fit is that
        # surface.
        inl = best_in
        for _r in range(3):
            run = _largest_run(rest, inl, best[1], tol)
            pts = rest[run]
            if len(pts) < need:
                break
            c = pts.mean(0)
            v = np.linalg.eigh((pts - c).T @ (pts - c))[1][:, -1]
            best = (c, v)
            new = np.abs((rest - c) @ np.array([-v[1], v[0]])) < tol
            settled = np.array_equal(new, inl)
            inl = new
            if settled:                  # refitting again changes nothing
                break
        c, v = best
        inl = _largest_run(rest, inl, v, tol)
        if inl.sum() < need:
            break
        res = (rest - c) @ np.array([-v[1], v[0]])
        out.append((c, v, rest[inl], res[inl], ids[inl]))
        rest, ids = rest[~inl], ids[~inl]
    return out


def _ransac(pts, tol, rng, m):
    """The best line (m=2, 2-D points) or plane (m=3, 3-D) through *m* random
    points: guesses scored in batches against a sample of the points,
    stopping once the best so far would have been drawn with
    _RANSAC_CONFIDENCE, or at _RANSAC_TRIES. Returns (a point on it, unit
    normal) or None."""
    n = len(pts)
    sub = pts if n <= _RANSAC_SCORE_POINTS else pts[rng.choice(n, _RANSAC_SCORE_POINTS, replace=False)]
    best, best_count, tries = None, -1, 0
    while tries < _RANSAC_TRIES:
        pick = rng.integers(0, n, (_RANSAC_BATCH, m))
        tries += _RANSAC_BATCH
        if m == 2:
            d = pts[pick[:, 1]] - pts[pick[:, 0]]
            size = np.linalg.norm(d, axis=1)
            ok = size >= 2 * tol
            nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        else:
            nrm = np.cross(pts[pick[:, 1]] - pts[pick[:, 0]], pts[pick[:, 2]] - pts[pick[:, 0]])
            size = np.linalg.norm(nrm, axis=1)
            ok = size > 1e-9
        if not ok.any():
            continue
        nrm = nrm[ok] / size[ok, None]
        base = pts[pick[ok, 0]]
        counts = (np.abs(sub @ nrm.T - np.einsum("ij,ij->i", base, nrm)) < tol).sum(0)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best, best_count = (base[k], nrm[k]), int(counts[k])
        share = best_count / len(sub)
        if share >= 1.0:
            break
        if share > 0:
            enough = np.log(1 - _RANSAC_CONFIDENCE) / np.log(1 - share ** m)
            if tries >= enough:
                break
    return best


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
        found = _ransac(rest, tol, rng, 3)
        if found is None:
            break
        c, n = found
        inl = np.abs((rest - c) @ n) < tol
        if inl.sum() < need:
            break
        for _r in range(3):
            run = _plane_run(rest, inl, n, t, tol)
            if run.sum() < need:
                break
            n, c = _plane(rest[run])
            new = np.abs((rest - c) @ n) < tol
            settled = np.array_equal(new, inl)
            inl = new
            if settled:                  # refitting again changes nothing
                break
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


def _step_edge(surfaces, a, tol, gate, shape, min_angle_deg, reach):
    """The top of a step whose face is hidden - a kerb scanned from the
    footpath, where the face and the road next to it lie in the kerb's
    shadow. Two parallel surfaces, apart by more than the noise; one (the
    upper) ends, the other carries on beyond a gap. A shadow always falls on
    the LOWER surface, past a convex edge, so the upper surface's end is a
    real edge - the step's top. (Its foot is what the shadow hides: a
    concave edge is never read this way.) Taken only when it matches the
    learned *shape* (the upper surface leaving the edge, the face going down
    to the lower one). Returns ``(point, shape, sigma)`` or None."""
    best = None
    for A in surfaces:
        for B in surfaces:
            if A is B:
                continue
            cA, vA, pA = A[0], A[1], A[2]
            if abs(vA[0] * B[1][1] - vA[1] * B[1][0]) >= np.sin(np.radians(min_angle_deg)):
                continue
            nA = np.array([-vA[1], vA[0]])
            h = float((B[0] - cA) @ nA)
            if not 3 * tol < abs(h) < reach:
                continue
            alA, alB = (pA - cA) @ vA, (B[2] - cA) @ vA
            if np.median(alB) > np.median(alA):          # B lies past A's + end
                end, sign = np.percentile(alA, 99), 1.0
                gap = np.percentile(alB, 2) - end
            else:
                end, sign = np.percentile(alA, 1), -1.0
                gap = end - np.percentile(alB, 98)
            if not -tol < gap < reach / 2:
                continue
            if abs(end - np.percentile(alA, 50 - 49 * sign)) < 3 * tol:
                continue
            away = -sign * vA
            face = np.sign(h) * nA
            mismatch = _shape_mismatch((away, face), shape)
            if mismatch is None:
                continue
            x = cA + end * vA
            dist = float(np.linalg.norm(x))
            if dist > gate:
                continue
            # the upper surface runs on behind and ahead, near its end
            near = np.abs(alA - end) < reach / 2
            aa = a[A[4]][near]
            if min(np.sum(aa < 0), np.sum(aa > 0)) < _MIN_EACH_SIDE:
                continue
            key = dist + mismatch * reach / np.pi
            if best is None or key < best[0]:
                best = (key, x, (away, face), float(np.sqrt(np.mean(A[3] ** 2))))
    return None if best is None else best[1:]


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


def _opens_upward(shape):
    """Whether the learned edge opens upward - both surfaces leave it with
    the space between them above (a kerb's bottom, a wall on a floor), not
    below (a kerb's top). Section coordinates: the second axis is as near
    up as the edge allows."""
    return shape is not None and float((shape[0] + shape[1])[1]) > 0


def _weak_fold(uv, a, tol, gate, shape, min_angle_deg, reach):
    """A gentle fold in the section - a ramp meeting the road - where line
    fitting sees one slightly bent surface. The points within *reach* of
    the prediction, turned so they run roughly level, are fitted with two
    straight pieces joined at a hinge; the hinge placed (1 cm steps, within
    *gate*) where they fit best, points beyond 3 *tol* of the fit dropped and
    refitted once. A fold of _WEAK_MIN_ANGLE_DEG up to *min_angle_deg* whose
    sides match the learned *shape* (``_weak_mismatch``), with points both
    behind and ahead. Returns ``(point, shape, sigma)`` or None."""
    m = np.linalg.norm(uv, axis=1) < reach
    if m.sum() < 4 * _MIN_SURFACE_POINTS:
        return None
    q, aa = uv[m], a[m]
    c = q.mean(0)
    v = np.linalg.eigh(np.cov((q - c).T))[1][:, -1]
    nrm = np.array([-v[1], v[0]])
    x, y = (q - c) @ v, (q - c) @ nrm
    x0s = np.arange(-gate, gate + 1e-9, 0.01) - c @ v       # hinge places near the prediction
    keep = np.ones(len(x), bool)
    best = None
    for _round in range(2):
        best = None
        for h in x0s:
            A = np.c_[np.ones(len(x)), x, np.maximum(0.0, x - h)]
            coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
            r = y - A @ coef
            cost = float(np.mean(r[keep] ** 2))
            if best is None or cost < best[0]:
                best = (cost, h, coef, r)
        keep = np.abs(best[3]) < tol
        if keep.sum() < 4 * _MIN_SURFACE_POINTS:
            return None
    _cost, h, (a0, b, cc), r = best
    if min(np.sum(keep & (x < h)), np.sum(keep & (x > h))) < 2 * _MIN_SURFACE_POINTS:
        return None
    turn = np.degrees(abs(np.arctan(b + cc) - np.arctan(b)))
    if not _WEAK_MIN_ANGLE_DEG <= turn < min_angle_deg:
        return None
    hinge = c + h * v + (a0 + b * h) * nrm
    if np.linalg.norm(hinge) > gate:
        return None
    left = _unit(-v - b * nrm)                  # leaving the hinge on each side
    right = _unit(v + (b + cc) * nrm)
    mismatch = _weak_mismatch((left, right), shape)
    if mismatch is None:
        return None
    for side in (x < h, x > h):                 # both sides run on behind and ahead
        near = keep & side & (np.abs(x - h) < reach / 2)
        if min(np.sum(near & (aa < 0)), np.sum(near & (aa > 0))) < _MIN_EACH_SIDE:
            return None
    return hinge, (left, right), float(np.sqrt(np.mean(r[keep] ** 2)))


def _weak_mismatch(dirs, shape):
    """_shape_mismatch for a weak fold (radians, or None): one surface leaves
    the edge as one learned surface does; the other - nearly carrying it on,
    where the learned face stood steep - need only leave to the same side as
    the learned face."""
    if shape is None:
        return None
    ang = lambda a, b: float(np.arccos(np.clip(a @ b, -1, 1)))
    best = None
    for kept, other in ((dirs[0], dirs[1]), (dirs[1], dirs[0])):
        for i in (0, 1):
            m = ang(kept, shape[i])
            if m <= np.radians(_SHAPE_TOLERANCE_DEG) and other @ shape[1 - i] > 0:
                best = m if best is None else min(best, m)
    return best


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
