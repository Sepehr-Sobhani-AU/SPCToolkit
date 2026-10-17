"""
Kalman tracking of a 1-D path through 3-D space.

A ``PathTracker`` keeps a running estimate of where a line is, which way it is
heading and (in the curved model) how it is bending — together with how sure it
is of each — and refines it one measured step at a time:

- **predict(ds)**: carry the estimate ``ds`` metres further along the path,
  growing the uncertainty as it goes.
- **update(point, direction=None)**: correct the estimate with a measured point
  on the path (and, optionally, its measured direction), weighing the
  measurement against the prediction by how much each is trusted.
- **smooth()**: once the whole path has been seen, one backward pass
  (Rauch-Tung-Striebel) so every step also benefits from what came after it.
- **is_consistent(point)**: whether a candidate point is where the track
  expects the line to be (Mahalanobis gate) — for rejecting clutter.
- **predict_path(distance)**: where the line should go next, with the
  sideways uncertainty at each sample — e.g. across a gap.

Two motion models, one machinery:

- ``"straight"`` — state: position + direction. The motion is linear, so this
  is the plain Kalman filter.
- ``"curved"`` — state: position + unit tangent + curvature vector, propagated
  along an exact circular arc of that curvature. The motion is non-linear, so
  this is the extended Kalman filter: each step is linearised (its Jacobian
  taken numerically) around the current estimate. Carries a bend across a gap
  where the straight model would shoot off along the tangent.

Distances are arc length along the path, not time. Standalone on purpose: plain
numpy (+ scipy for the gate threshold), no Qt, no ``global_variables`` and no
other project imports, so it can be tested on its own and reused elsewhere.

**CPU and float64, deliberately.** The state is 6 or 9 numbers; one step is a
handful of 9x9 matrix products, microseconds on the CPU. A GPU's per-call cost
(transfer + launch) is far larger than that work, and steps cannot run in
parallel — each needs the previous one — so a GPU would be slower, not faster.
It would only pay for many tracks at once (batched) or a particle filter.
float64 because a filter's covariance degrades as rounding accumulates over
hundreds of steps; at this size it costs nothing. (This is numerical stability,
not large-coordinate precision.)
"""

from collections import namedtuple

import numpy as np
from scipy.stats import chi2


STRAIGHT = "straight"
CURVED = "curved"

_STATE_SIZE = {STRAIGHT: 6, CURVED: 9}

# Below this turn angle (rad) over one step the arc is propagated with its
# Taylor expansion — the exact form divides by the curvature.
_SMALL_TURN = 1e-6

# Finite-difference step for the extended model's Jacobian, relative to the
# size of each state entry (with this as the floor).
_JACOBIAN_EPS = 1e-6

# Starting uncertainty for what the first measurement does not tell us.
_INITIAL_DIRECTION_SIGMA = 0.5        # rad
_INITIAL_CURVATURE_SIGMA = 1.0        # 1/m


class TrackState:
    """The tracker's estimate at one point along the path: state vector ``x``
    and its covariance ``P`` (float64). Read through the properties."""

    __slots__ = ("x", "P", "model")

    def __init__(self, x, P, model):
        self.x = np.asarray(x, dtype=np.float64)
        self.P = np.asarray(P, dtype=np.float64)
        self.model = model

    @property
    def position(self):
        return self.x[0:3].copy()

    @property
    def tangent(self):
        """Unit direction of travel."""
        t = self.x[3:6]
        return t / max(np.linalg.norm(t), 1e-12)

    @property
    def curvature(self):
        """Curvature vector (1/m, pointing to the centre of the bend); zero for
        the straight model."""
        return self.x[6:9].copy() if self.model == CURVED else np.zeros(3)

    @property
    def position_cov(self):
        return self.P[0:3, 0:3].copy()

    def lateral_sigma(self):
        """One-sigma sideways position uncertainty (m): the larger spread of
        the position covariance across the direction of travel."""
        t = self.tangent
        across = np.eye(3) - np.outer(t, t)
        cov = across @ self.P[0:3, 0:3] @ across
        return float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))

    def reversed(self):
        """The same estimate, travelling the other way. Position and curvature
        vector are unchanged (curvature is a second derivative, so reversing
        the path leaves it alone); the tangent flips."""
        flip = np.ones(len(self.x))
        flip[3:6] = -1.0
        return TrackState(self.x * flip, self.P * np.outer(flip, flip), self.model)

    def copy(self):
        return TrackState(self.x.copy(), self.P.copy(), self.model)


# One step of history, kept for the smoother.
#   prior:    the estimate predicted for this step, before its measurement.
#   F:        the Jacobian that carried the previous step's estimate here.
#   filtered: the estimate after this step's measurement (or = prior without).
_Step = namedtuple("_Step", ["prior", "F", "filtered"])


class PathTracker:
    """
    Kalman / extended Kalman tracker for a path through 3-D space.

    Parameters:
        model: ``"straight"`` (Kalman: position + direction) or ``"curved"``
            (extended Kalman: + curvature, exact arcs).
        point_sigma: How far off the true path a measured point may be (m,
            one sigma). Larger = trust the measurements less, smooth more.
        direction_sigma: Same for a measured direction (rad), when given.
        direction_noise: Straight model — how freely the direction may wander
            per metre travelled (rad per sqrt(m)). Larger = follows turns
            faster, smooths less.
        curvature_noise: Curved model — how freely the curvature may change
            per metre travelled ((1/m) per sqrt(m)). Larger = follows changes
            of bend faster, smooths less.
    """

    def __init__(self, model=CURVED, point_sigma=0.01,
                 direction_sigma=np.radians(3.0), direction_noise=0.05,
                 curvature_noise=0.05):
        if model not in _STATE_SIZE:
            raise ValueError(f"unknown model '{model}' "
                             f"(use '{STRAIGHT}' or '{CURVED}')")
        self.model = model
        self.n = _STATE_SIZE[model]
        self.point_sigma = float(point_sigma)
        self.direction_sigma = float(direction_sigma)
        self.direction_noise = float(direction_noise)
        self.curvature_noise = float(curvature_noise)
        self._state = None
        self._history = []

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    @property
    def state(self):
        """The current estimate (a copy)."""
        if self._state is None:
            raise RuntimeError("tracker not started — call start() first")
        return self._state.copy()

    def start(self, point, direction, curvature=None):
        """Begin a track at *point*, heading along *direction*.

        The first point is taken as measured (``point_sigma``); direction and
        curvature start loose, so the first few steps settle them.
        """
        t = _unit(direction)
        x = np.zeros(self.n)
        x[0:3] = point
        x[3:6] = t
        sig = np.empty(self.n)
        sig[0:3] = self.point_sigma
        sig[3:6] = _INITIAL_DIRECTION_SIGMA
        if self.model == CURVED:
            if curvature is not None:
                k = np.asarray(curvature, dtype=np.float64)
                x[6:9] = k - (k @ t) * t
            sig[6:9] = _INITIAL_CURVATURE_SIGMA
        state = TrackState(x, np.diag(sig ** 2), self.model)
        self._state = state
        self._history = [_Step(state.copy(), np.eye(self.n), state.copy())]
        return self.state

    def predict(self, ds):
        """Carry the estimate *ds* metres further along the path.

        Records a new step; ``update`` then corrects it. Returns the predicted
        estimate.
        """
        prior, F = self._propagate(self._state_or_raise(), float(ds))
        self._state = prior
        self._history.append(_Step(prior.copy(), F, prior.copy()))
        return self.state

    def update(self, point, direction=None):
        """Correct the current estimate with a measured *point* (and optional
        unit *direction*). Returns the corrected estimate."""
        state = self._state_or_raise()
        z, h, H, R = self._measurement(state, point, direction)
        y = z - h
        S = H @ state.P @ H.T + R
        K = np.linalg.solve(S, H @ state.P).T          # P H^T S^-1
        x = state.x + K @ y
        # Joseph form: stays symmetric and positive under rounding.
        I_KH = np.eye(self.n) - K @ H
        P = I_KH @ state.P @ I_KH.T + K @ R @ K.T
        corrected = TrackState(x, 0.5 * (P + P.T), self.model)
        self._enforce_constraints(corrected)
        self._state = corrected
        if self._history:
            last = self._history[-1]
            self._history[-1] = _Step(last.prior, last.F, corrected.copy())
        return self.state

    def mahalanobis(self, point):
        """Squared Mahalanobis distance of *point* from where the current
        estimate expects the path to be — position uncertainty plus
        measurement noise both accounted for."""
        state = self._state_or_raise()
        y = np.asarray(point, dtype=np.float64) - state.x[0:3]
        S = state.P[0:3, 0:3] + np.eye(3) * self.point_sigma ** 2
        return float(y @ np.linalg.solve(S, y))

    def is_consistent(self, point, probability=0.99):
        """Whether *point* lies where the track expects the path, at the given
        confidence (chi-square gate, 3 degrees of freedom)."""
        return self.mahalanobis(point) <= chi2.ppf(probability, 3)

    def smooth(self):
        """Rauch-Tung-Striebel backward pass over the steps so far.

        Returns one ``TrackState`` per step (the start included), each now
        informed by the whole track. The last equals the forward estimate —
        nothing comes after it. Does not change the tracker's own state.
        """
        if not self._history:
            return []
        steps = self._history
        smoothed = [None] * len(steps)
        smoothed[-1] = steps[-1].filtered.copy()
        for k in range(len(steps) - 2, -1, -1):
            filt = steps[k].filtered
            nxt = steps[k + 1]
            # C = P_k F^T (P_prior_{k+1})^-1
            C = np.linalg.solve(nxt.prior.P.T, (filt.P @ nxt.F.T).T).T
            x = filt.x + C @ (smoothed[k + 1].x - nxt.prior.x)
            P = filt.P + C @ (smoothed[k + 1].P - nxt.prior.P) @ C.T
            state = TrackState(x, 0.5 * (P + P.T), self.model)
            self._enforce_constraints(state)
            smoothed[k] = state
        return smoothed

    def predict_path(self, distance, step=0.1, state=None):
        """Where the path should go next, without changing the tracker.

        Returns ``(positions, tangents, lateral_sigma)`` sampled every *step*
        metres over *distance*, starting at the current estimate (or *state*).
        ``lateral_sigma`` is the one-sigma sideways uncertainty at each sample:
        it grows the further the prediction reaches.
        """
        current = (self._state_or_raise() if state is None else state).copy()
        n = max(1, int(np.ceil(distance / step)))
        ds = distance / n
        positions = [current.position]
        tangents = [current.tangent]
        sigmas = [current.lateral_sigma()]
        for _ in range(n):
            current, _F = self._propagate(current, ds)
            positions.append(current.position)
            tangents.append(current.tangent)
            sigmas.append(current.lateral_sigma())
        return np.array(positions), np.array(tangents), np.array(sigmas)

    # ------------------------------------------------------------------ #
    # Models                                                             #
    # ------------------------------------------------------------------ #

    def _state_or_raise(self):
        if self._state is None:
            raise RuntimeError("tracker not started — call start() first")
        return self._state

    def _propagate(self, state, ds):
        """Prior estimate *ds* further on, and the Jacobian that took it there."""
        if self.model == STRAIGHT:
            F = np.eye(6)
            F[0:3, 3:6] = np.eye(3) * ds
            x = F @ state.x                    # linear: plain Kalman
        else:
            x = _arc_motion(state.x, ds)
            F = self._numeric_jacobian(state.x, ds)
        P = F @ state.P @ F.T + self._process_noise(ds)
        prior = TrackState(x, 0.5 * (P + P.T), self.model)
        self._enforce_constraints(prior)
        return prior, F

    def _numeric_jacobian(self, x, ds):
        """d(arc motion)/d(state), by central differences (extended Kalman)."""
        F = np.empty((self.n, self.n))
        for j in range(self.n):
            h = _JACOBIAN_EPS * max(1.0, abs(x[j]))
            up = x.copy()
            down = x.copy()
            up[j] += h
            down[j] -= h
            F[:, j] = (_arc_motion(up, ds) - _arc_motion(down, ds)) / (2.0 * h)
        return F

    def _process_noise(self, ds):
        """How much the path may change over *ds*: white noise driving the
        highest derivative the model carries (direction, or curvature),
        integrated down to position."""
        ds = abs(ds)
        if self.model == STRAIGHT:
            q = self.direction_noise ** 2
            block = q * np.array([[ds ** 3 / 3.0, ds ** 2 / 2.0],
                                  [ds ** 2 / 2.0, ds]])
        else:
            q = self.curvature_noise ** 2
            block = q * np.array([[ds ** 5 / 20.0, ds ** 4 / 8.0, ds ** 3 / 6.0],
                                  [ds ** 4 / 8.0, ds ** 3 / 3.0, ds ** 2 / 2.0],
                                  [ds ** 3 / 6.0, ds ** 2 / 2.0, ds]])
        return np.kron(block, np.eye(3))

    def _measurement(self, state, point, direction):
        """``(z, h(x), H, R)`` for a measured point and optional direction."""
        H_point = np.zeros((3, self.n))
        H_point[:, 0:3] = np.eye(3)
        z = [np.asarray(point, dtype=np.float64)]
        h = [state.x[0:3]]
        H = [H_point]
        r = [self.point_sigma ** 2] * 3
        if direction is not None:
            t = state.x[3:6]
            norm = max(np.linalg.norm(t), 1e-12)
            u = t / norm
            d = _unit(direction)
            if d @ u < 0:                    # a direction measured backwards
                d = -d
            H_dir = np.zeros((3, self.n))
            # d(t/|t|)/dt — the along-track part carries no information.
            H_dir[:, 3:6] = (np.eye(3) - np.outer(u, u)) / norm
            z.append(d)
            h.append(u)
            H.append(H_dir)
            r += [self.direction_sigma ** 2] * 3
        return (np.concatenate(z), np.concatenate(h), np.vstack(H),
                np.diag(r))

    def _enforce_constraints(self, state):
        """Keep the tangent unit length and the curvature across it — the
        filter's linear correction does not respect either by itself."""
        t = state.x[3:6]
        norm = np.linalg.norm(t)
        if norm > 1e-12:
            t = t / norm
            state.x[3:6] = t
        if self.model == CURVED:
            k = state.x[6:9]
            state.x[6:9] = k - (k @ t) * t


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("direction has zero length")
    return vec / norm


def _arc_motion(x, ds):
    """Carry a curved-model state *ds* along the circular arc it describes."""
    p, t, k = x[0:3], x[3:6], x[6:9]
    kappa = np.linalg.norm(k)
    theta = kappa * ds
    if abs(theta) < _SMALL_TURN:
        # Taylor: the exact form below divides by kappa.
        return np.concatenate([p + t * ds + 0.5 * k * ds * ds, t + k * ds, k])
    n = k / kappa
    s, c = np.sin(theta), np.cos(theta)
    p_new = p + t * (s / kappa) + n * ((1.0 - c) / kappa)
    t_new = t * c + n * s
    k_new = kappa * (n * c - t * s)
    return np.concatenate([p_new, t_new, k_new])


# --------------------------------------------------------------------------- #
# Convenience: track a whole polyline                                          #
# --------------------------------------------------------------------------- #

# The result of tracking one polyline.
#   filtered:  (M, 3) forward estimates at each vertex.
#   smoothed:  (M, 3) smoothed estimates at each vertex.
#   tangents:  (M, 3) smoothed unit directions.
#   states:    list of smoothed TrackState (one per vertex).
#   end_state: the forward estimate at the last vertex — the one to predict
#              onward from (smoothing changes nothing at the end).
#   tracker:   the PathTracker, for predict_path() from the end.
TrackResult = namedtuple("TrackResult", ["filtered", "smoothed", "tangents",
                                         "states", "end_state", "tracker"])


def track_polyline(points, model=CURVED, **settings):
    """Track the ordered vertices of a polyline, forward then smoothed.

    Each vertex is one position measurement; the distance stepped between them
    is their straight-line spacing. The start direction comes from the first
    few vertices. Repeated (zero-length) vertices are skipped. *settings* go to
    ``PathTracker``. Returns a ``TrackResult``, or None for fewer than two
    distinct vertices.
    """
    pts = np.asarray(points, dtype=np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return None

    head = pts[:min(len(pts), 4)]
    direction = head[-1] - head[0]
    tracker = PathTracker(model=model, **settings)
    tracker.start(pts[0], direction)

    filtered = [tracker.state.position]
    for prev, cur in zip(pts[:-1], pts[1:]):
        tracker.predict(np.linalg.norm(cur - prev))
        filtered.append(tracker.update(cur).position)

    states = tracker.smooth()
    return TrackResult(
        np.array(filtered),
        np.array([s.position for s in states]),
        np.array([s.tangent for s in states]),
        states,
        tracker.state,
        tracker,
    )
