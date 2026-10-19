"""Synthetic lines unlike the test wires: does the tracker adapt to the line
it is given? Not tuned on - a check only.

Each scene: a 120 m line with a gentle sag (radius 400 m), a 5 m gap at
40 m, a clutter blob over the line at 80 m, and a parallel neighbour line
offset sideways. Seed = the first 5 m. Pass = the trace reaches within 1 m
of the far end and stays on the line (<= 2% of vertices off it).
"""
import ktrack
from ktrack import *

rng = np.random.default_rng(7)
L, R_SAG = 120.0, 400.0


def curve(s):
    """Points on the true line at arc lengths s (sag in z)."""
    return np.stack([s, np.zeros_like(s), (s - L / 2) ** 2 / (2 * R_SAG)], 1)


def line_points(per_m, kind, size, offset=np.zeros(3)):
    n = int(L * per_m)
    s = np.sort(rng.uniform(0, L, n))
    s = s[(s < 40) | (s > 45)]                       # the gap
    base = curve(s) + offset
    if kind == "wire":                               # solid, gaussian across
        lat = rng.normal(0, size, (len(s), 2))
    else:                                            # pipe: points on a tube surface
        a = rng.uniform(0, 2 * np.pi, len(s))
        lat = np.stack([np.cos(a), np.sin(a)], 1) * size + rng.normal(0, 0.005, (len(s), 2))
    return base + np.stack([np.zeros(len(s)), lat[:, 0], lat[:, 1]], 1), s


def scene(per_m, kind, size, neighbour):
    pts, s = line_points(per_m, kind, size)
    other, _ = line_points(per_m, kind, size, offset=np.array([0, neighbour, 0]))
    blob = rng.normal(0, 1.5, (3000, 3)) + curve(np.array([80.0]))[0]
    P_ = np.vstack([pts, other, blob]).astype(np.float32)
    seed = np.flatnonzero(np.arange(len(P_)) < len(pts))[s < 5]
    return P_, seed


SCENES = {
    "A thin dense wire": (50, "wire", 0.008, 0.5),
    "B thick pipe":      (40, "pipe", 0.10, 0.8),
    "C sparse thin wire": (5, "wire", 0.01, 0.5),
    "D thick sparse":     (8, "wire", 0.05, 0.8),
}


def judge(path, size):
    s = np.clip(path[:, 0], 0, L)
    off = np.linalg.norm(path - curve(s), axis=1)
    reached = path[:, 0].max() > L - 1.0
    stray = np.mean(off > max(0.15, 4 * size))
    return reached, stray


if __name__ == "__main__":
    for name, (per_m, kind, size, nb) in SCENES.items():
        P_, seed = scene(per_m, kind, size, nb)
        ktrack.P = P_
        ktrack._TREE = None
        out = []
        for scaled in (False, True):
            cfg = dict(DEFAULTS); cfg["auto_scale"] = scaled; cfg["max_len"] = 200
            path, ev = track(seed, +1, cfg)
            reached, stray = judge(path, size)
            ok = reached and stray <= 0.02
            out.append(f"{'measured' if scaled else 'fixed'}: {'PASS' if ok else 'fail'} "
                       f"(reached {path[:, 0].max():5.1f} m, off-line {stray:.0%})")
        prof = seed_profile(P_[seed])
        print(f"{name:20s} | measured thickness {prof['thickness']*100:4.1f} cm, spacing {prof['spacing']*100:4.1f} cm"
              f" | {out[0]} | {out[1]}", flush=True)
