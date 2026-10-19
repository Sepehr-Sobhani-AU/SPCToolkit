import sys, numpy as np
sys.path.insert(0, ".")
from core.services.path_tracker import track_polyline, PathTracker
rng = np.random.default_rng(0)


def arc_err(p, R):
    return np.abs(np.linalg.norm(p[:, :2], axis=1) - R)


# straight noisy line, vertex spacing 0.25 m, noise 2 cm
x = np.arange(0, 20, 0.25)
truth = np.stack([x, 0 * x, 0 * x], 1)
meas = truth + rng.normal(0, 0.02, truth.shape)
for model in ("straight", "curved"):
    r = track_polyline(meas, model=model, point_sigma=0.02)
    print(f"line {model}: raw rms {np.sqrt((meas[:, 1:] ** 2).sum(1).mean()):.4f}  "
          f"smoothed rms {np.sqrt((r.smoothed[:, 1:] ** 2).sum(1).mean()):.4f}")

# arc R=2, noise 1 cm
R = 2.0
t = np.arange(0, np.pi, 0.25 / R)
truth = np.stack([R * np.cos(t), R * np.sin(t), 0 * t], 1)
meas = truth + rng.normal(0, 0.01, truth.shape)
for model in ("straight", "curved"):
    r = track_polyline(meas, model=model, point_sigma=0.01)
    print(f"arc {model}: raw rms {np.sqrt((arc_err(meas, R) ** 2).mean()):.4f} "
          f"smoothed rms {np.sqrt((arc_err(r.smoothed, R) ** 2).mean()):.4f} "
          f"curvature end {np.linalg.norm(r.end_state.curvature):.3f} (true {1 / R})")
    half = len(meas) // 2
    r2 = track_polyline(meas[:half], model=model, point_sigma=0.01)
    pos, tan, sig = r2.tracker.predict_path(1.5, 0.1)
    print(f"   predict 1.5 m: end radial error {arc_err(pos[-1:], R)[0]:.3f} m, "
          f"lateral sigma {sig[-1]:.3f}")
