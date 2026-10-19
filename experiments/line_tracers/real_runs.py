"""Real-data robustness: trace each test line from one seed piece, check it
reaches the other piece (>= half of its points on the same line), both ways.
Seeds grouped as the plugin does (DBSCAN 0.10 m). One setting changed at a
time. Usage: python real_runs.py <tracer> <worktree root>"""
import sys
import time
import numpy as np
from sklearn.cluster import DBSCAN

which, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.particle_line_tracer import ParticleLineTracer
from core.services.shape_query import ShapeQueryService

D = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit/experiments/kalman_tracker/data"
P = np.load(f"{D}/points.npy")
SEEDS = dict(np.load(f"{D}/seeds.npz"))
cls = {"kalman": KalmanLineTracer, "particle": ParticleLineTracer}[which]
service = ShapeQueryService()

RUNS = [(t, d, f"{t}_{a}", f"{t}_{b}") for t in ("Test1", "Test2")
        for d, a, b in (("fwd", "Start", "End"), ("back", "End", "Start"))]


def groups_of(idx):
    lab = DBSCAN(eps=0.10, min_samples=2).fit_predict(P[idx])
    return [idx[lab == k] for k in range(lab.max() + 1) if np.sum(lab == k) >= 2]


def run(settings):
    out = []
    for t, d, a, b in RUNS:
        tr = cls(P, shape_query=service, **settings)
        lines = tr.grow_lines(groups_of(SEEDS[a]))
        line = max(lines, key=lambda l: np.isin(SEEDS[a], l.indices).sum()) if lines else None
        ok = line is not None and np.isin(SEEDS[b], line.indices).mean() >= 0.5
        out.append("OK " if ok else "x  ")
    service.release_all()
    return out


grid = [{}] + [{k: v} for k, vs in {
    "step": [0.25, 1.0], "window": [0.6, 2.0], "max_gap": [10.0, 30.0],
    "max_turn_deg": [15.0, 45.0], "bridge_cone_deg": [8.0, 25.0],
    "min_bend_radius": [20.0, 200.0], "flexibility": [0.005, 0.1],
    "open_air": [False]}.items() for v in vs]
if which == "particle":
    grid += [{"particles": 100}, {"particles": 600}, {"seed": 1}, {"seed": 2}]
total = ok = 0
print("runs:", [f"{t}-{d}" for t, d, _, _ in RUNS], flush=True)
for g in grid:
    t0 = time.time()
    r = run(g)
    total += len(r); ok += sum(x == "OK " for x in r)
    print(f"{str(g or 'defaults'):30s}", " ".join(r), f" {time.time()-t0:.0f}s", flush=True)
print(f"{which}: passed {ok}/{total}")
