"""General synthetic scenes for line tracers (not tuned to any one feature).

Each scene: points, seed indices, the true line (a function giving the
distance of a point to it), and the settings that suit that kind of line -
as a preset would. A trace passes when it reaches the far end of the true
line (or, for "ends", does NOT reach the unrelated line) and stays on it.

Usage: python scenes.py <tracer: kalman|particle> <worktree root> [scene,...]
"""
import sys
import time
import numpy as np


def wire(x, sd, rng):
    return np.stack([x, rng.normal(0, sd, len(x)), rng.normal(0, sd, len(x))], 1)


def straight_dist(c, y=0.0, z=0.0):
    return np.hypot(c[:, 1] - y, c[:, 2] - z)


def make_scenes(seed=11):
    rng = np.random.default_rng(seed)
    S = {}

    # A: thin line, gap, clutter blob on it, neighbour 0.5 m away
    x = np.sort(rng.uniform(0, 60, 2400)); x = x[(x < 25) | (x > 29)]
    line = wire(x, 0.008, rng)
    nb = wire(np.sort(rng.uniform(0, 60, 2400)), 0.008, rng) + [0, 0.5, 0]
    blob = rng.normal(0, 1.0, (3000, 3)) + [45, 0, 0]
    S["A thin, gap, clutter"] = dict(pts=np.vstack([line, nb, blob]), seed=np.flatnonzero(x < 3),
                                    dist=straight_dist, far=60, settings={})

    # B: hollow pipe, radius 0.15 m
    x = np.sort(rng.uniform(0, 40, 3000)); a = rng.uniform(0, 2 * np.pi, len(x))
    pipe = np.stack([x, 0.15 * np.cos(a), 0.15 * np.sin(a)], 1) + rng.normal(0, 0.004, (len(x), 3))
    S["B hollow pipe r=15cm"] = dict(pts=pipe, seed=np.flatnonzero(x < 3), dist=straight_dist,
                                     far=40, settings={})

    # C: sparse thin line, 5 points per metre
    x = np.sort(rng.uniform(0, 60, 300))
    S["C sparse 5/m"] = dict(pts=wire(x, 0.01, rng), seed=np.flatnonzero(x < 6),
                             dist=straight_dist, far=60, settings=dict(window=3.0, step=1.0))

    # D: tight curve - a quarter circle of radius 4 m (a kerb round a corner)
    R = 4.0
    th = np.sort(rng.uniform(0, np.pi / 2, 1500))
    arc = np.stack([R * np.sin(th), R - R * np.cos(th), np.zeros_like(th)], 1) + rng.normal(0, 0.008, (len(th), 3))
    def arc_dist(c, R=R):
        return np.hypot(np.linalg.norm(c[:, :2] - [0, R], axis=1) - R, c[:, 2])
    S["D curve R=4m"] = dict(pts=arc, seed=np.flatnonzero(th < 0.25), dist=arc_dist,
                             far_point=np.array([R, R, 0]), settings=dict(min_bend_radius=2.0, flexibility=0.5))

    # E: kink of 25 deg at x = 20 (a support)
    x1 = np.sort(rng.uniform(0, 20, 800)); s2 = np.sort(rng.uniform(0, 20, 800))
    ang = np.radians(25)
    seg2 = np.stack([20 + s2 * np.cos(ang), s2 * np.sin(ang), np.zeros_like(s2)], 1)
    kink = np.vstack([wire(x1, 0.008, rng), seg2 + rng.normal(0, 0.008, seg2.shape)])
    def kink_dist(c, ang=ang):
        d1 = straight_dist(c)
        d1 = np.where(c[:, 0] <= 20, d1, np.inf)
        rel = c - [20, 0, 0]; u = np.array([np.cos(ang), np.sin(ang), 0])
        al = np.clip(rel @ u, 0, None)
        d2 = np.linalg.norm(rel - al[:, None] * u, axis=1)
        return np.minimum(d1, d2)
    S["E kink 25deg"] = dict(pts=kink, seed=np.flatnonzero(np.arange(len(kink)) < 100),
                             dist=kink_dist, far_point=np.array([20 + 20 * np.cos(ang), 20 * np.sin(ang), 0]),
                             settings={})

    # F: crossing line at 60 deg through the middle: must go straight on
    x = np.sort(rng.uniform(0, 40, 1600))
    main = wire(x, 0.008, rng)
    s = np.sort(rng.uniform(-10, 10, 800)); u = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0])
    cross = np.array([20, 0, 0]) + s[:, None] * u + rng.normal(0, 0.008, (len(s), 3))
    S["F crossing 60deg"] = dict(pts=np.vstack([main, cross]), seed=np.flatnonzero(x < 3),
                                 dist=straight_dist, far=40, settings={})

    # G: close parallel line 0.15 m away (thin lines): must stay on its own
    x = np.sort(rng.uniform(0, 40, 1600)); x2 = np.sort(rng.uniform(0, 40, 1600))
    S["G parallel 15cm"] = dict(pts=np.vstack([wire(x, 0.008, rng), wire(x2, 0.008, rng) + [0, 0.15, 0]]),
                                seed=np.flatnonzero(x < 3), dist=straight_dist, far=40, settings={})

    # H: line ends at x = 20; an unrelated line crosses its path 12 m on
    x = np.sort(rng.uniform(0, 20, 800))
    s = np.sort(rng.uniform(-5, 5, 400))
    other = np.stack([np.full_like(s, 32), s, np.full_like(s, 0.6)], 1) + rng.normal(0, 0.008, (len(s), 3))
    S["H ends, unrelated beyond"] = dict(pts=np.vstack([wire(x, 0.008, rng), other]),
                                         seed=np.flatnonzero(x < 3), dist=straight_dist,
                                         must_not_pass=21.0, settings={})
    return S


def judge(sc, line):
    c = np.asarray(line.centerline, dtype=float)
    off = sc["dist"](c)
    stray = float(np.mean(off > 0.25))
    if "must_not_pass" in sc:
        ok = c[:, 0].max() < sc["must_not_pass"]
        return ok, f"reached x {c[:, 0].max():.1f} (must stop before {sc['must_not_pass']})"
    if "far" in sc:
        reach = c[:, 0].max(); ok = reach > sc["far"] - 1.0
    else:
        reach = -float(np.linalg.norm(c - sc["far_point"], axis=1).min()); ok = -reach < 1.0
    ok = ok and stray <= 0.02
    return ok, f"reach {reach:.1f}, off-line {stray:.0%}"


if __name__ == "__main__":
    which, root = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    from core.services.kalman_line_tracer import KalmanLineTracer
    from core.services.particle_line_tracer import ParticleLineTracer
    cls = {"kalman": KalmanLineTracer, "particle": ParticleLineTracer}[which]
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    passed = 0
    scenes = make_scenes()
    for name, sc in scenes.items():
        if only and name[0] not in only:
            continue
        pts = sc["pts"].astype(np.float32)
        t0 = time.time()
        tr = cls(pts, **sc["settings"])
        lines = tr.grow_lines([sc["seed"]])
        tr.release()
        # the line through the seed
        line = max(lines, key=lambda l: np.isin(sc["seed"], l.indices).sum()) if lines else None
        if line is None:
            print(f"{name:26s} FAIL  no line"); continue
        ok, info = judge(sc, line)
        passed += ok
        print(f"{name:26s} {'PASS' if ok else 'FAIL'}  {info}  ({len(lines)} line(s), {time.time()-t0:.1f}s)", flush=True)
    print(f"{which}: {passed}/{len(scenes) if not only else len(only)} passed")
