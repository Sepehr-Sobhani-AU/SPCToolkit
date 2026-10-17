# path_tracker_test.py
#
# The Kalman / extended Kalman path tracker on made-up lines, where the true
# path is known exactly: smoothing must beat the raw noise, the curved model
# must learn a bend and carry it across a gap, the gate must tell a point on
# the track from one off it, and the smoother must agree with the forward
# filter where nothing comes after.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.path_tracker import (
    PathTracker, track_polyline, STRAIGHT, CURVED,
)


def _line(noise, seed=0, spacing=0.25, length=20.0):
    rng = np.random.default_rng(seed)
    x = np.arange(0, length, spacing)
    truth = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    return truth + rng.normal(0, noise, truth.shape)


def _arc(R, noise, seed=0, spacing=0.25, sweep=np.pi):
    rng = np.random.default_rng(seed)
    t = np.arange(0, sweep, spacing / R)
    truth = np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1)
    return truth + rng.normal(0, noise, truth.shape)


def _off_line(p):
    return np.sqrt((p[:, 1:] ** 2).sum(axis=1))


def _off_arc(p, R):
    return np.abs(np.linalg.norm(p[:, :2], axis=1) - R)


def _rms(v):
    return float(np.sqrt(np.mean(v ** 2)))


def test_smoothing_beats_the_noise_on_a_straight_line():
    meas = _line(noise=0.02)
    raw = _rms(_off_line(meas))
    for model in (STRAIGHT, CURVED):
        smoothed = _rms(_off_line(track_polyline(meas, model=model,
                                                 point_sigma=0.02).smoothed))
        print(f"straight line, {model}: raw {raw * 1000:.1f} mm -> "
              f"smoothed {smoothed * 1000:.1f} mm")
        assert smoothed < 0.6 * raw, f"{model} smoothing did not reduce the noise"


def test_curved_model_learns_the_bend():
    R = 2.0
    meas = _arc(R, noise=0.01)
    raw = _rms(_off_arc(meas, R))
    straight = track_polyline(meas, model=STRAIGHT, point_sigma=0.01)
    curved = track_polyline(meas, model=CURVED, point_sigma=0.01)
    s_err = _rms(_off_arc(straight.smoothed, R))
    c_err = _rms(_off_arc(curved.smoothed, R))
    kappa = np.linalg.norm(curved.end_state.curvature)
    print(f"arc R={R}: raw {raw * 1000:.1f} mm, straight {s_err * 1000:.1f} mm, "
          f"curved {c_err * 1000:.1f} mm; curvature {kappa:.3f} (true {1 / R})")
    assert abs(kappa - 1 / R) < 0.05, f"curvature {kappa:.3f}, true {1 / R}"
    assert c_err < 0.6 * raw, "curved model did not smooth the arc"
    assert c_err < s_err, "curved model no better than straight on a curve"


def test_curved_prediction_carries_the_bend_across_a_gap():
    """Track half an arc, then predict 1.5 m on: the curved model follows the
    bend, the straight one shoots off along the tangent."""
    R = 2.0
    meas = _arc(R, noise=0.01)
    half = meas[:len(meas) // 2]
    errors = {}
    for model in (STRAIGHT, CURVED):
        result = track_polyline(half, model=model, point_sigma=0.01)
        positions, _tangents, sigma = result.tracker.predict_path(1.5, 0.1)
        errors[model] = float(_off_arc(positions[-1:], R)[0])
        print(f"gap 1.5 m, {model}: lands {errors[model] * 100:.1f} cm off the arc, "
              f"sideways sigma {sigma[-1] * 100:.1f} cm")
        assert np.all(np.diff(sigma) >= -1e-12), "uncertainty must grow ahead"
    assert errors[CURVED] < 0.10, f"curved prediction {errors[CURVED]:.3f} m off"
    assert errors[STRAIGHT] > 5 * errors[CURVED], "setup wrong: straight should miss"


def test_gate_tells_on_track_from_off_track():
    result = track_polyline(_line(noise=0.01), model=CURVED, point_sigma=0.01)
    tracker = result.tracker
    tracker.predict(0.25)
    ahead = tracker.state.position
    on = ahead + np.array([0.0, 0.01, 0.0])
    off = ahead + np.array([0.0, 0.30, 0.0])
    print(f"gate: on-track d2={tracker.mahalanobis(on):.1f}, "
          f"0.3 m off d2={tracker.mahalanobis(off):.1f}")
    assert tracker.is_consistent(on)
    assert not tracker.is_consistent(off)


def test_reversed_state_predicts_behind_the_start():
    """Reverse the smoothed start state and predict backwards: on an arc it
    must continue the same circle behind where tracking began."""
    R = 2.0
    meas = _arc(R, noise=0.005, sweep=np.pi / 2)
    result = track_polyline(meas, model=CURVED, point_sigma=0.005)
    back = result.states[0].reversed()
    positions, _t, _s = result.tracker.predict_path(1.0, 0.1, state=back)
    err = float(_off_arc(positions[-1:], R)[0])
    behind = positions[-1, 1] < 0          # the arc started at y=0, going +y
    print(f"backwards 1 m: {err * 100:.1f} cm off the arc, behind start: {behind}")
    assert behind and err < 0.05


def test_smoother_agrees_with_the_filter_at_the_end():
    result = track_polyline(_arc(2.0, noise=0.01), model=CURVED, point_sigma=0.01)
    assert np.allclose(result.smoothed[-1], result.filtered[-1])
    assert np.allclose(result.states[-1].x, result.end_state.x)
    print("smoother: last step equals the forward estimate")


def test_direction_measurement_and_constraints():
    """A measured direction is used (backwards ones flipped), and the state
    stays well formed: unit tangent, curvature across it, float64."""
    tracker = PathTracker(model=CURVED, point_sigma=0.01)
    tracker.start([0, 0, 0], [1, 0, 0])
    for i in range(1, 20):
        tracker.predict(0.25)
        tracker.update([0.25 * i, 0, 0], direction=[-1, 0.0, 0])   # backwards
    s = tracker.state
    assert s.x.dtype == np.float64
    assert abs(np.linalg.norm(s.tangent) - 1) < 1e-9
    assert s.tangent[0] > 0.99, "a backwards direction measurement flipped the track"
    assert abs(s.curvature @ s.tangent) < 1e-9
    print("direction measurement and constraints hold")


def test_repeated_vertices_and_tiny_input():
    pts = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    result = track_polyline(pts, model=STRAIGHT)
    assert len(result.smoothed) == 3, "repeated vertices should be skipped"
    assert track_polyline(np.zeros((3, 3))) is None
    print("repeated vertices skipped, degenerate input returns None")


if __name__ == "__main__":
    test_smoothing_beats_the_noise_on_a_straight_line()
    test_curved_model_learns_the_bend()
    test_curved_prediction_carries_the_bend_across_a_gap()
    test_gate_tells_on_track_from_off_track()
    test_reversed_state_predicts_behind_the_start()
    test_smoother_agrees_with_the_filter_at_the_end()
    test_direction_measurement_and_constraints()
    test_repeated_vertices_and_tiny_input()
    print("\nAll path tracker tests passed.")
