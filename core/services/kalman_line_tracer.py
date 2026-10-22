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

from core.services.path_tracker import PathTracker, CURVED
from core.services.linear_region_grower import (
    STOP_CLUTTER,
    STOP_EMPTY_SPACE,
    STOP_STEP_CAP,
    STOP_UNRELATED_LINE,
)
from core.services.line_tracer_base import (
    LineTracerBase,
    line_profile,
    _GATE_SIGMAS,
    _CONFIRM_STEPS,
    _BRIDGE_WINDOWS,
    _BRIDGE_SLICES,
    _BRIDGE_MIN_SPAN,
    _KINK_WINDOWS,
    _KINK_SLICES,
    _KINK_MIN_SPAN,
    _KINK_MIN_TURN_DEG,
    _RESTART_DIR_SIGMA,
)

__all__ = ["KalmanLineTracer", "line_profile"]


class KalmanLineTracer(LineTracerBase):
    """
    Trace linear features with a Kalman filter. Settings: see
    ``LineTracerBase``.
    """

    def _track(self, start, heading, cfg, bend=None):
        """Trace one way from *start* along *heading*. Returns the centerline
        vertices and how it ended: ``("end", None)`` or ``("joined", id)``."""
        kt = PathTracker(model=CURVED, point_sigma=cfg["point_sigma"],
                         curvature_noise=self.flexibility)
        start = np.asarray(start, dtype=float)
        self._restart(kt, start, heading, cfg, bend)
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
                return np.array(path), ("end", STOP_UNRELATED_LINE)
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
            # A coast keeps the last bend; on a tight curve that carries it
            # round in a circle and back over its own line. A line does not
            # turn back across a gap: past a right angle it has ended.
            if gaps > self.max_gap or kt.state.tangent @ last_good.tangent < 0:
                # Nothing matching within reach: the line ends - in clutter
                # or in open space.
                return np.array(path), ("end", STOP_CLUTTER if clutter_seen else STOP_EMPTY_SPACE)
        return np.array(path), ("end", STOP_STEP_CAP)

    def _coast_after(self):
        return 3 * self.window

    def _restart(self, kt, pos, heading, cfg, bend=None):
        """Begin (again) at a measured point and heading, with a bend prior
        no looser than the bend the line may follow."""
        kt.start(pos, heading, bend)
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
