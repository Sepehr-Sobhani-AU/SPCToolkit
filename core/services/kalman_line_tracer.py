"""
Kalman tracing of linear features from seed groups.

The Kalman Trace growth mode of Linear Region Growing. Where the axis march
fits a fresh line in each window, this keeps a running estimate of the line -
position, direction and bend, with how sure it is of each
(``core.services.path_tracker``, extended Kalman) - and uses it to:

- **read the line where it is predicted to be**: each step takes the points in
  a short window around the prediction and fits the line's position there;
- **coast through clutter and gaps**: where the window is crowded (vegetation,
  a pole) or empty, it carries on along the prediction instead of following
  whatever points are there;
- **find the far side**: after a gap, a search from the last good point for a
  continuation that lands where the prediction says, within its uncertainty,
  and is about as thick as the line;
- **find a kink**: where the line turns at a support, a short search from
  points around the bend for where it leaves.

Size is measured, not set: each line's radius comes from its own points -
the median distance from the centre line plus two robust standard deviations
of that distance - first at the seed, then again along the whole traced line.
That is the APPARENT width (true size plus scanner noise), which is what a
window has to hold, whether a 2 cm cable or a 30 cm pipe.

Lines are traced once. Points already on a line - this run's or an existing
result's (*blocked*) - are seen but never taken: a trace that runs onto one
stops and joins it there; a gap is bridged onto one only when it passes the
same checks as any continuation; a turn never goes onto one. A line whose end
joins another line's end is merged into it. Where nothing matches, the line
ends, and that end is a stop the extension tools can revisit.

The engine is general. What it does assume about a feature is a setting:
``open_air`` (crowded points are clutter to coast through - right for a
cable or a pipe in air, wrong for a kerb on the ground), the gap it may cross,
the turn it may take at a kink, the bend it may follow. Presets of those for
particular features belong on top of this, not in it.

Neighbour questions go to the shape query service (GPU), as the axis march's
do; the filter itself is a handful of 9x9 products per step and stays on the
CPU (see ``path_tracker``).
"""

import copy

import numpy as np
from scipy.spatial import cKDTree

from core.services.path_tracker import PathTracker, CURVED
from core.services.query_shapes import Cylinder, Sphere
from core.services.linear_region_grower import (
    GrownLine,
    MarchStop,
    STOP_EMPTY_SPACE,
    _default_shape_query,
)


# Owner id written for points of lines that existed before this run (the
# *blocked* mask). Any id >= 0 is "on a line"; these can be joined but are
# never merged with this run's lines.
_EXISTING = np.iinfo(np.int32).max

# Fewest points to measure a line from (radius, direction).
_MIN_PROFILE_POINTS = 12
# Fewest points a traced line must take to be kept.
_MIN_LINE_POINTS = 10
# Radius = median distance + this many robust standard deviations.
_RADIUS_SDS = 2.0
# A continuation's radius must be within this factor of the line's.
_THICKNESS_RATIO = 2.5
# Gate = this many sigmas of the prediction's sideways uncertainty.
_GATE_SIGMAS = 3.0
# Widest gate (x radius) while reading normally / while coasting.
_GATE_MAX_RADII = 5.0
_COAST_GATE_MAX_RADII = 16.0
# Readings needed in a window: plain / just after clutter, before trusting.
_MIN_READ_POINTS = 3
_STRICT_READ_POINTS = 6
_CONFIRM_STEPS = 2
# Clutter: points 1.5-4 radii off the line, per point on it.
_RING_INNER, _RING_OUTER = 1.5, 4.0
_CLUTTER_RATIO = 0.5
_STRICT_CLUTTER_RATIO = 0.1
# Bridge search: window and ray budget.
_BRIDGE_WINDOWS = 4.0          # search window, in measuring windows
_BRIDGE_SLICES = 8
_BRIDGE_MIN_SPAN = 6
_BRIDGE_RAYS = 400
# Kink search: window and its slicing.
_KINK_WINDOWS = 3.0
_KINK_SLICES = 10
_KINK_MIN_SPAN = 6
_KINK_MIN_TURN_DEG = 2.0
# Uncertainty when (re)starting at a measured point and heading.
_RESTART_DIR_SIGMA = 0.03      # rad
# Points of a seed group within this many radii of a traced line are covered
# by it - also the few just outside its 96% radius, which would otherwise be
# traced again as a second line.
_COVER_RADII = 2.0
# Resampling step (m) for comparing and claiming along centerlines.
_SAMPLE = 0.1
# Finer resampling for distances to traced lines: a point's distance to the
# nearest sample is within half of this of its distance to the line.
_FINE_SAMPLE = 0.02


class KalmanLineTracer:
    """
    Trace linear features through a point cloud with a Kalman filter.

    Parameters:
        all_points: ``(N, 3)`` cloud.
        shape_query: ``ShapeQueryService`` (defaults to the app's). GPU.
        step: metres between readings.
        window: length (m) of the window each reading is taken from.
        max_gap: the longest stretch (m) coasted with no reading before the
            line is ended.
        max_turn_deg: the sharpest kink (deg) the line may take at a support.
        bridge_cone_deg: cone (deg) round the heading searched for the far
            side of a gap.
        min_bend_radius: the tightest bend (m) the line may follow.
        flexibility: how fast the bend may change ((1/m) per sqrt(m)).
        open_air: True when the line stands free of other points (cable,
            pipe): points crowding round it are clutter, coasted through. False
            for a line on or along a surface (kerb, edge).
        max_length: cap (m) on one direction of one trace.
    """

    def __init__(self, all_points, shape_query=None, step=0.5, window=1.0,
                 max_gap=20.0, max_turn_deg=30.0, bridge_cone_deg=15.0,
                 min_bend_radius=50.0, flexibility=0.02, open_air=True,
                 max_length=5000.0):
        self.all_points = np.asarray(all_points)
        self.shape_query = shape_query if shape_query is not None \
            else _default_shape_query()
        self.step = float(step)
        self.window = float(max(window, step))
        self.max_gap = float(max_gap)
        self.max_turn_deg = float(max_turn_deg)
        self.bridge_cone_deg = float(bridge_cone_deg)
        self.max_curvature = 1.0 / max(float(min_bend_radius), 1e-6)
        self.flexibility = float(flexibility)
        self.open_air = bool(open_air)
        self.max_length = float(max_length)
        self._owner = None
        self._cancel_event = None
        # Fine samples of the lines traced so far, for distances to them.
        self._line_tree = None
        self._line_of_sample = None

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def grow_lines(self, seed_groups, *, progress_cb=None, cancel_event=None,
                   blocked=None) -> list:
        """Trace every line the seed groups lie on, each once.

        Same contract as ``LinearRegionGrower.grow_lines``: *seed_groups* are
        index arrays (typically DBSCAN groups of the picked points), largest
        traced first; *blocked* marks points of existing lines, seen but never
        taken; *progress_cb(done, total, message)*; *cancel_event.is_set()*
        stops early with what was traced. Returns a list of ``GrownLine``.

        A group may hold more than one line (lines touching inside the
        selection): after tracing from its densest spot, its seeds the line
        did not cover are traced next.
        """
        n = len(self.all_points)
        owner = np.full(n, -1, dtype=np.int32)
        if blocked is not None:
            owner[np.asarray(blocked, dtype=bool)] = _EXISTING
        self._owner = owner
        self._cancel_event = cancel_event
        self._line_tree = None
        self._line_of_sample = None
        groups = sorted((np.asarray(g, dtype=np.intp) for g in seed_groups
                         if np.asarray(g).size >= 2), key=len, reverse=True)
        lines = []          # dicts: centerline, radius, ends
        try:
            for gi, group in enumerate(groups):
                remaining = group[owner[group] < 0]
                while len(remaining) >= _MIN_LINE_POINTS and not self._cancelled():
                    remaining = self._trace_from(remaining, lines)
                if progress_cb is not None:
                    progress_cb(gi + 1, len(groups),
                                f"Kalman tracing - {len(lines)} line(s), "
                                f"{len(groups) - gi - 1} seed group(s) left")
                if self._cancelled():
                    break
            lines = self._merge_end_joins(lines)
            return [self._to_grown_line(k, line) for k, line in enumerate(lines)]
        finally:
            self._owner = None
            self._cancel_event = None
            self._line_tree = None
            self._line_of_sample = None

    def release(self) -> None:
        """Free the shape query service's GPU state for this cloud."""
        self.shape_query.release_all()

    # ------------------------------------------------------------------ #
    # One line                                                           #
    # ------------------------------------------------------------------ #

    def _trace_from(self, remaining, lines):
        """Trace one line from the densest spot of *remaining*, keep what is
        new of it, and return the seeds still uncovered."""
        owner = self._owner
        mini = self._dense_start(remaining)
        prof = line_profile(self.all_points[mini])
        if prof is None:                 # too few points here to measure
            return remaining[~np.isin(remaining, mini)]
        cfg = self._cfg(prof)
        centerline, ends = self._trace_both_ways(mini, cfg)
        pieces = self._cut_same_line(centerline, prof["radius"], lines)
        took = 0
        for piece in pieces:
            radius = self._line_radius(piece, prof["radius"])
            members = self._points_near(piece, radius)
            members = members[owner[members] < 0]
            if len(members) < _MIN_LINE_POINTS:
                continue
            owner[members] = len(lines)
            lines.append(dict(centerline=piece, radius=radius,
                              ends=ends if len(pieces) == 1 else [_END, _END]))
            self._index_lines(lines)
            took += len(members)
        covered = owner[remaining] >= 0
        if lines:
            covered |= self._near_lines(self.all_points[remaining], lines)
        if took == 0 or not covered.any():
            # Nothing traced from this spot: set its neighbourhood aside.
            covered |= np.linalg.norm(self.all_points[remaining]
                                      - self.all_points[mini].mean(0), axis=1) \
                < 2 * self.window
        return remaining[~covered & (owner[remaining] < 0)]

    def _dense_start(self, idx):
        """The seeds within two windows of the densest spot of *idx*: a short
        straight piece to start from."""
        pts = self.all_points[idx].astype(float)
        reach = 2 * self.window
        sub = (np.arange(len(idx)) if len(idx) <= 2000 else
               np.random.default_rng(0).choice(len(idx), 2000, replace=False))
        counts = np.array([len(x) for x in
                           cKDTree(pts).query_ball_point(pts[sub], reach / 3)])
        centre = pts[sub[int(np.argmax(counts))]]
        return idx[np.linalg.norm(pts - centre, axis=1) <= reach]

    def _cfg(self, prof):
        r = prof["radius"]
        return dict(
            radius=r,
            point_sigma=prof["sigma"],
            gate_min=r,
            gate_max=_GATE_MAX_RADII * r,
            gate_max_coast=max(_COAST_GATE_MAX_RADII * r, 2 * r),
        )

    def _trace_both_ways(self, mini, cfg):
        """Forward from the back end of the start piece through it and on;
        backward from the same point straight out. Joined into one
        centerline, back half reversed - no overlap to trim."""
        seed = self.all_points[mini].astype(float)
        c = seed - seed.mean(0)
        ax = np.linalg.eigh(c.T @ c)[1][:, -1]
        start = seed[int(np.argmin(c @ ax))]
        fwd, end_f = self._track(start, ax, cfg)
        back, end_b = self._track(start, -ax, cfg)
        line = np.vstack([back[::-1], fwd[1:]]) if len(back) > 1 else fwd
        return line, [end_b, end_f]

    # ------------------------------------------------------------------ #
    # The tracker                                                        #
    # ------------------------------------------------------------------ #

    def _track(self, start, heading, cfg):
        """Trace one way from *start* along *heading*. Returns the centerline
        vertices and how it ended: ``("end", None)`` or ``("joined", id)``."""
        kt = PathTracker(model=CURVED, point_sigma=cfg["point_sigma"],
                         curvature_noise=self.flexibility)
        start = np.asarray(start, dtype=float)
        self._restart(kt, start, heading, cfg)
        path = [start]
        last_good = kt.state
        gaps = travelled = 0.0
        retried = clutter_seen = False
        next_bridge = self._coast_after()
        for _ in range(int(3 * self.max_length / self.step)):
            if travelled >= self.max_length or self._cancelled():
                break
            kt.predict(self.step)
            self._clamp(kt)
            travelled += self.step
            z, why = self._read(kt.state, cfg, coasting=gaps >= self._coast_after(),
                                strict=clutter_seen)
            if z is not None and clutter_seen and not self._confirmed(kt, z, cfg):
                z, why = None, "unconfirmed"
            if z is not None:
                kt.update(z)
                self._clamp(kt)
                last_good = kt.state
                gaps, retried, clutter_seen = 0.0, False, False
                path.append(kt.state.position)
                continue
            if why.startswith("joined:"):
                other = int(why.split(":")[1])
                # Running straight onto a traced line, this line ends there.
                # After coasting it is a join only if the other line CONTINUES
                # this one; else it is an unrelated line the coast reached.
                if gaps <= self.window or self._continues(kt.state, other, cfg):
                    path.append(kt.state.position)
                    return np.array(path), ("joined", other)
                return np.array(path), _END
            gaps += self.step
            clutter_seen |= why == "clutter"
            if gaps >= 2 * self.step and not retried and why != "clutter":
                retried = True
                turned = self._find_kink(kt, last_good, gaps, cfg)
                if turned is not None:
                    anchor, new_t = turned
                    if np.linalg.norm(anchor - last_good.position) > 1e-6:
                        path.append(anchor)
                    self._restart(kt, anchor, new_t, cfg)
                    travelled -= gaps
                    gaps = 0.0
                    continue
            if gaps < self._coast_after():
                next_bridge = self._coast_after()
            if gaps >= next_bridge and why != "clutter":
                next_bridge = gaps + self.window
                found = self._bridge(kt, last_good, gaps - self.window, cfg)
                if found is not None:
                    pos, new_t, joins = found
                    path.append(pos)
                    if joins is not None:
                        # The far side is a traced line and it matches: the
                        # same line - join it there.
                        return np.array(path), ("joined", joins)
                    self._restart(kt, pos, new_t, cfg)
                    last_good = kt.state
                    gaps, retried, clutter_seen = 0.0, False, False
                    continue
            if gaps > self.max_gap:
                break          # nothing matching within reach: the line ends
        return np.array(path), _END

    def _coast_after(self):
        return 3 * self.window

    def _restart(self, kt, pos, heading, cfg):
        """Begin (again) at a measured point and heading, with a bend prior
        no looser than the bend the line may follow."""
        kt.start(pos, heading)
        sig = np.full(kt.n, cfg["point_sigma"])
        sig[3:6] = _RESTART_DIR_SIGMA
        sig[6:9] = self.max_curvature
        kt._state.P = np.diag(sig ** 2)
        kt._history[-1] = kt._history[-1]._replace(
            prior=kt._state.copy(), filtered=kt._state.copy())

    def _clamp(self, kt):
        """No bend tighter than min_bend_radius."""
        k = kt._state.x[6:9]
        m = np.linalg.norm(k)
        if m > self.max_curvature:
            kt._state.x[6:9] = k * self.max_curvature / m

    def _read(self, state, cfg, coasting=False, strict=False):
        """One reading: the line's position in the window round the
        prediction. Returns ``(point, "ok")`` or ``(None, why)`` - "none",
        "sparse", "clutter" or "joined:<id>".

        Picks the cluster of points both dense and close to the prediction
        (two lines can share a wide gate), re-centres on it, then refuses it
        if it is another line's or (open air) crowded by other points.
        """
        p, t, k = state.position, state.tangent, state.curvature
        sig = state.lateral_sigma()
        r = cfg["radius"]
        gmax = cfg["gate_max_coast"] if coasting else cfg["gate_max"]
        g = float(np.clip(_GATE_SIGMAS * np.hypot(sig, cfg["point_sigma"]),
                          cfg["gate_min"], gmax))
        half = self.window / 2
        ring = r * _RING_OUTER
        idx = self._ball(p, np.hypot(half, g + ring))
        if len(idx) < _MIN_READ_POINTS:
            return None, "none"
        a, lat = _lateral(self.all_points[idx].astype(float) - p, t)
        lat = lat - 0.5 * np.outer(a ** 2, k)          # follow the predicted bend
        keep = np.abs(a) < half
        idx, lat = idx[keep], lat[keep]
        cand = lat[np.linalg.norm(lat, axis=1) < g]
        if len(cand) < _MIN_READ_POINTS:
            return None, "none"
        sub = cand if len(cand) <= 1500 else \
            cand[np.random.default_rng(0).choice(len(cand), 1500, replace=False)]
        dens = (((sub[:, None] - sub[None]) ** 2).sum(-1) < r * r).sum(1)
        s = max(sig, r)
        centre = sub[int(np.argmax(dens * np.exp(-0.5 * (sub ** 2).sum(1) / s ** 2)))]
        for _ in range(3):              # re-centre on the line's axis
            core = lat[np.linalg.norm(lat - centre, axis=1) < r]
            if len(core) == 0:
                break
            centre = core.mean(0)
        dist = np.linalg.norm(lat - centre, axis=1)
        in_core = dist < r
        if in_core.sum() < (_STRICT_READ_POINTS if strict else _MIN_READ_POINTS):
            return None, "sparse"
        taken = self._owner[idx[in_core]] >= 0
        if taken.mean() > 0.5:
            other = int(np.bincount(self._owner[idx[in_core]][taken]).argmax()) \
                if self._owner[idx[in_core]][taken].max() < _EXISTING else _EXISTING
            return None, f"joined:{other}"
        if self.open_air:
            n_ring = int(np.sum((dist >= _RING_INNER * r) & (dist < ring)))
            ratio = _STRICT_CLUTTER_RATIO if strict else _CLUTTER_RATIO
            if n_ring > ratio * in_core.sum():
                return None, "clutter"
        return p + lat[in_core].mean(0), "ok"

    def _confirmed(self, kt, z, cfg):
        """After clutter, a reading counts only if the next readings agree
        with it: a branch in a tree gives one good-looking window, a line
        coming out of the tree a run of them. Tried on a copy."""
        trial = copy.deepcopy(kt)
        trial.update(z)
        self._clamp(trial)
        for _ in range(_CONFIRM_STEPS):
            trial.predict(self.step)
            self._clamp(trial)
            z2, _why = self._read(trial.state, cfg, strict=True)
            if z2 is None:
                return False
            trial.update(z2)
            self._clamp(trial)
        return True

    def _continues(self, state, other, cfg):
        """Whether line *other*, met at *state*, continues the traced line:
        running within the bridge cone of the predicted heading, and about as
        thick."""
        idx = self._ball(state.position, 2 * self.window)
        idx = idx[self._owner[idx] == other] if len(idx) else idx
        if len(idx) < _MIN_PROFILE_POINTS:
            return False
        pts = self.all_points[idx].astype(float)
        c = pts - pts.mean(0)
        d = np.linalg.eigh(c.T @ c)[1][:, -1]
        if np.degrees(np.arccos(min(1.0, abs(float(d @ state.tangent))))) > self.bridge_cone_deg:
            return False
        prof = line_profile(pts)
        return prof is not None and _similar(prof["radius"], cfg["radius"])

    def _find_kink(self, kt, last_good, gaps, cfg):
        """Where the line turns at a support. The bend lies between the last
        good reading and where readings stopped, so the search is tried from
        points along the prediction there; the best kept. A turn never goes
        onto a traced line. Returns ``(anchor, new heading)`` or None."""
        positions, tangents, _s = kt.predict_path(gaps, self.step / 2, state=last_good)
        best = None
        for pos, tan in zip(positions, tangents):
            found = self._ray_search(pos, tan, 0.2, cfg, length=_KINK_WINDOWS * self.window,
                                     cone_deg=self.max_turn_deg, slices=_KINK_SLICES,
                                     min_span=_KINK_MIN_SPAN, floor=0.2, may_join=False)
            if found is not None and (best is None or found["span"] > best[0]["span"]):
                best = (found, pos)
        if best is None:
            return None
        found, anchor = best
        turn = np.degrees(np.arccos(np.clip(found["direction"] @ last_good.tangent, -1, 1)))
        return (anchor, found["direction"]) if turn > _KINK_MIN_TURN_DEG else None

    def _bridge(self, kt, last_good, a_from, cfg):
        """Find where the line comes out of a gap or clutter, from the last
        good point. The continuation must land where the line was predicted
        to be, within the prediction's own uncertainty - which grows with the
        gap, so a long gap allows more offset but only along the predicted
        course. Returns ``(point, heading, joined id or None)`` or None."""
        def predict(dist):
            dist = max(dist, 1e-3)
            pos, _tan, sig = kt.predict_path(dist, dist, state=last_good)
            return pos[-1], sig[-1]

        found = self._ray_search(last_good.position, last_good.tangent, a_from, cfg,
                                 length=_BRIDGE_WINDOWS * self.window,
                                 cone_deg=self.bridge_cone_deg, slices=_BRIDGE_SLICES,
                                 min_span=_BRIDGE_MIN_SPAN, floor=0.5, may_join=True)
        if found is None:
            return None
        t = last_good.tangent
        start = found["start"]
        pos, sigma = predict(float((start - last_good.position) @ t))
        off = start - pos
        off = off - (off @ t) * t
        if np.linalg.norm(off) > _GATE_SIGMAS * np.hypot(sigma, cfg["point_sigma"]) + cfg["radius"]:
            return None
        return start, found["direction"], found["joins"]

    def _ray_search(self, anchor, t, a_from, cfg, length, cone_deg, slices, min_span,
                    floor, may_join):
        """Rays from *anchor* within *cone_deg* of *t*, through points in the
        window [a_from, a_from + length] ahead. A ray scores by the clean
        slices of the window it covers - points within the line's radius,
        and (open air) no crowd round them. The line itself, collinear with
        its own exit, covers the window; a ray crossing another line covers
        a slice or two. The found piece must be about as thick as the line.
        Traced lines are never re-traced: their points are left out unless
        *may_join*, and a found piece on one is reported as a join.

        Returns dict(start, direction, span, joins) or None.
        """
        r = cfg["radius"]
        a_from = max(a_from, floor)
        tan = np.tan(np.radians(cone_deg))
        mid = anchor + t * (a_from + length / 2)
        idx = self._ball(mid, np.hypot(length / 2, (a_from + length) * tan + r))
        if len(idx) < _MIN_READ_POINTS:
            return None
        rel = self.all_points[idx].astype(float) - anchor
        a = rel @ t
        lat = np.linalg.norm(rel - np.outer(a, t), axis=1)
        keep = (a >= a_from) & (a <= a_from + length) & (lat <= a * tan + r)
        if not may_join:
            keep &= self._owner[idx] < 0
        idx, rel = idx[keep], rel[keep]
        if len(rel) < _MIN_READ_POINTS:
            return None
        pick = rel if len(rel) <= _BRIDGE_RAYS else \
            rel[np.random.default_rng(0).choice(len(rel), _BRIDGE_RAYS, replace=False)]
        dirs = np.vstack([t, pick / np.linalg.norm(pick, axis=1)[:, None]])
        sq = np.einsum("ij,ij->i", rel, rel)
        best = None
        for lo in range(0, len(dirs), 64):
            d = dirs[lo:lo + 64]
            proj = d @ rel.T
            dist = np.sqrt(np.maximum(sq[None] - proj ** 2, 0))
            al = proj - a_from
            in_win = (al >= 0) & (al <= length)
            sl = np.clip((al / (length / slices)).astype(int), 0, slices - 1)
            inb = (dist < r) & in_win
            ring = (dist >= r * _RING_INNER) & (dist < r * _RING_OUTER) & in_win
            for i in range(len(d)):
                nb = np.bincount(sl[i][inb[i]], minlength=slices)
                clean = nb >= 1
                if self.open_air:
                    nr = np.bincount(sl[i][ring[i]], minlength=slices)
                    clean &= nr <= _CLUTTER_RATIO * np.maximum(nb, 1)
                key = (int(clean.sum()), float(d[i] @ t))
                if best is None or key > best[0]:
                    best = (key, d[i], inb[i])
        (span, _), ray, band = best
        if span < min_span:
            return None
        pts = rel[band]
        c = pts.mean(0)
        v = np.linalg.eigh((pts - c).T @ (pts - c))[1][:, -1]
        direction = v if v @ ray > 0 else -v
        prof = line_profile(anchor + pts)
        if prof is not None and not _similar(prof["radius"], r):
            return None
        band_idx = idx[band]
        taken = self._owner[band_idx] >= 0
        joins = None
        if taken.mean() > 0.5:
            ids = self._owner[band_idx][taken]
            joins = _EXISTING if ids.max() == _EXISTING else int(np.bincount(ids).argmax())
        start = anchor + c + direction * float(((pts - c) @ direction).min())
        return dict(start=start, direction=direction, span=span, joins=joins)

    # ------------------------------------------------------------------ #
    # Claiming and joining                                               #
    # ------------------------------------------------------------------ #

    def _line_radius(self, line, seed_radius):
        """The traced line's radius along all of it: the seed's densest spot
        (often nearest the scanner, least noisy) reads too tight further on.
        Distances of the points within three seed radii of the centerline,
        median + 2 robust sd; never below the seed's own."""
        idx = self._points_near(line, 3 * seed_radius)
        if len(idx) < _MIN_PROFILE_POINTS:
            return seed_radius
        d = cKDTree(_densify(line, _FINE_SAMPLE)).query(self.all_points[idx])[0]
        return max(seed_radius, float(np.median(d) + _RADIUS_SDS * _robust_sd(d)))

    def _points_near(self, line, radius):
        """Rows within *radius* of the polyline *line*."""
        rows = []
        for a, b in zip(line[:-1], line[1:]):
            seg = b - a
            length = float(np.linalg.norm(seg))
            if length < 1e-9:
                continue
            start = a - seg / length * radius
            rows.append(self.shape_query.points_in(
                self.all_points, Cylinder(start, seg, radius, length + 2 * radius)))
        if not rows:
            return np.empty(0, dtype=np.intp)
        idx = np.unique(np.concatenate(rows)).astype(np.intp)
        d = cKDTree(_densify(line, _FINE_SAMPLE)).query(self.all_points[idx])[0]
        return idx[d <= radius]

    def _index_lines(self, lines):
        """Rebuild the lookup of fine samples along every traced line."""
        samples = [_densify(np.asarray(l["centerline"], dtype=float), _FINE_SAMPLE)
                   for l in lines]
        self._line_tree = cKDTree(np.vstack(samples))
        self._line_of_sample = np.concatenate(
            [np.full(len(x), k, dtype=np.intp) for k, x in enumerate(samples)])

    def _nearest_line(self, pts):
        """Distance from each of *pts* to the nearest traced line, and which."""
        d, i = self._line_tree.query(np.asarray(pts, dtype=float))
        return d, self._line_of_sample[i]

    def _near_lines(self, pts, lines):
        """Mask of *pts* within _COVER_RADII radii of the nearest traced line."""
        if self._line_tree is None:
            return np.zeros(len(pts), dtype=bool)
        d, k = self._nearest_line(pts)
        radii = np.array([l["radius"] for l in lines])
        return d < _COVER_RADII * radii[k]

    def _cut_same_line(self, line, radius, lines):
        """Pieces of *line* not lying on a traced line - closer than the sum
        of the two radii (their tubes overlap) for longer than a window is
        the same line, drawn once."""
        if self._line_tree is None or len(line) < 2:
            return [line]
        dense = _densify(line, _SAMPLE)
        d, k = self._nearest_line(dense)
        radii = np.array([l["radius"] for l in lines])
        near = d < radius + radii[k]
        cut = np.zeros(len(dense), dtype=bool)
        for a, b in _runs(near):
            if (b - a) * _SAMPLE > self.window:
                cut[a:b] = True
        keep = max(1, int(round(self.step / _SAMPLE)))
        return [dense[a:b][::keep] if b - a > keep else dense[a:b]
                for a, b in _runs(~cut) if b - a >= 2]

    def _merge_end_joins(self, lines):
        """A line whose end joined another of this run's lines AT ITS END is
        the same line: merge them. A join onto the middle of a line (a
        junction) stays two lines meeting there."""
        owner = self._owner
        alive = list(range(len(lines)))
        merged = True
        while merged:
            merged = False
            for i in alive:
                for side in (0, 1):
                    kind, j = lines[i]["ends"][side]
                    if kind != "joined" or j == i or j not in alive:
                        continue
                    tip = lines[i]["centerline"][0 if side == 0 else -1]
                    other = lines[j]["centerline"]
                    d0 = np.linalg.norm(other[0] - tip)
                    d1 = np.linalg.norm(other[-1] - tip)
                    if min(d0, d1) > self.window:
                        continue
                    a = lines[i]["centerline"] if side == 1 else lines[i]["centerline"][::-1]
                    b = other if d0 <= d1 else other[::-1]
                    far_i = lines[i]["ends"][1 - side]
                    far_j = lines[j]["ends"][1] if d0 <= d1 else lines[j]["ends"][0]
                    lines[i] = dict(centerline=np.vstack([a, b]),
                                    radius=max(lines[i]["radius"], lines[j]["radius"]),
                                    ends=[far_i, far_j])
                    owner[owner == j] = i
                    alive.remove(j)
                    merged = True
                    break
                if merged:
                    break
        remap = {old: new for new, old in enumerate(alive)}
        relabel = np.full(len(lines) + 1, -1, dtype=np.int32)
        for old, new in remap.items():
            relabel[old] = new
        mine = (owner >= 0) & (owner < _EXISTING)
        owner[mine] = relabel[owner[mine]]
        out = []
        for i in alive:
            ends = [(kind, remap.get(j, j) if kind == "joined" else j)
                    for kind, j in lines[i]["ends"]]
            out.append(dict(lines[i], ends=ends))
        return out

    def _to_grown_line(self, label, line):
        """The line in the grower's format. An end where the line ended is a
        stop (the extension tools revisit it); an end where it joined another
        line is a junction, not a stop."""
        c = np.asarray(line["centerline"], dtype=float)
        stops, end_cylinders = [], []
        for side, (kind, _j) in enumerate(line["ends"]):
            if kind != "end" or len(c) < 2:
                continue
            tip = c[0] if side == 0 else c[-1]
            prev = c[1] if side == 0 else c[-2]
            direction = tip - prev
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
            stops.append(MarchStop(tip.copy(), direction, STOP_EMPTY_SPACE))
            end_cylinders.append((STOP_EMPTY_SPACE,
                                  (tip.copy(), direction, line["radius"], self.window)))
        return GrownLine(np.flatnonzero(self._owner == label).astype(np.intp),
                         c, [], end_cylinders, stops, [])

    def _ball(self, centre, radius):
        return self.shape_query.points_in(
            self.all_points, Sphere(np.asarray(centre, dtype=float), float(radius))
        ).astype(np.intp)

    def _cancelled(self):
        return self._cancel_event is not None and self._cancel_event.is_set()


# --------------------------------------------------------------------------- #
# Measuring a line                                                            #
# --------------------------------------------------------------------------- #

_END = ("end", None)


def line_profile(pts, chunk=None):
    """Measure a line from its own points: radius, per-axis noise sigma and
    point spacing. None when there are too few points.

    Measured on short pieces along it, so a bend or sag is not read as
    thickness. radius = median distance from the centre line + 2 robust
    standard deviations of that distance: ~96% of a solid line's points, and
    for a hollow pipe (points on a ring) the pipe radius plus its jitter.
    Robust (median absolute deviation), so a few stray points do not widen it.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < _MIN_PROFILE_POINTS:
        return None
    c = pts - pts.mean(0)
    axis = np.linalg.eigh(c.T @ c)[1][:, -1]
    pts = pts[np.argsort(c @ axis)]
    size = chunk or max(10, len(pts) // 8)
    dist, across, spacing = [], [], []
    for lo in range(0, len(pts) - size + 1, size):
        ch = pts[lo:lo + size]
        cc = ch - ch.mean(0)
        v = np.linalg.eigh(cc.T @ cc)[1]
        along = cc @ v[:, -1]
        length = float(np.ptp(along))
        if length <= 0:
            continue
        lat = cc - np.outer(along, v[:, -1])
        dist.extend(np.linalg.norm(lat, axis=1))
        across.extend(lat @ v[:, 1])
        across.extend(lat @ v[:, 0])
        spacing.append(length / (len(ch) - 1))
    if not dist:
        return None
    dist = np.array(dist)
    radius = float(np.median(dist) + _RADIUS_SDS * _robust_sd(dist))
    return dict(radius=max(radius, 1e-3), sigma=max(_robust_sd(across), 1e-4),
                spacing=float(np.median(spacing)))


def _robust_sd(x):
    """Standard deviation from the median absolute deviation."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _similar(radius, reference):
    return 1.0 / _THICKNESS_RATIO <= radius / max(reference, 1e-9) <= _THICKNESS_RATIO


def _lateral(rel, t):
    a = rel @ t
    return a, rel - np.outer(a, t)


def _densify(line, step):
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.array(out)


def _runs(mask):
    d = np.diff(np.r_[0, np.asarray(mask).astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
