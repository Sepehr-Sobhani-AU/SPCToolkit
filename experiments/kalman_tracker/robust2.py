from ktrack import *
import itertools
# Forward (start->end) and backward (end->start) runs.
RUNS = []
for t in TESTS:
    RUNS.append((t, "fwd", f"{t}_Start", f"{t}_End"))
    RUNS.append((t, "back", f"{t}_End", f"{t}_Start"))
def run(cfg):
    res = []
    for t, d, a, b in RUNS:
        s = P[SEEDS[a]].astype(float); e = P[SEEDS[b]].mean(0)
        c = s - s.mean(0); _, v = np.linalg.eigh(c.T @ c)
        sign = 1 if v[:, -1] @ (e - s.mean(0)) > 0 else -1
        path, ev = track(SEEDS[a], sign, cfg)
        mem = np.unique(np.concatenate([np.array(x, dtype=np.intp) for x in tree().query_ball_point(path, 0.04)]))
        hit = np.isin(SEEDS[b], mem).sum() > 0
        prog = (path - s.mean(0)) @ ((e - s.mean(0)) / np.linalg.norm(e - s.mean(0)))
        res.append("OK " if hit else f"x{prog.max():.0f}")
    return res
grid = [{}] + [{k: v} for k, vs in {
    "step": [0.25, 0.75, 1.5], "point_sigma": [0.005, 0.05], "tight": [0.03, 0.1],
    "gate_max": [0.15, 0.4], "bridge_win": [3.0, 6.0], "bridge_max_deg": [10.0, 20.0],
    "clutter_ratio": [0.3, 1.0], "max_gap": [10.0, 30.0], "min_pts": [2, 6],
    "curvature_noise": [0.002, 0.1], "max_curvature": [0.01, 0.05], "slab": [0.6, 1.5],
    "model": ["straight"]}.items() for v in vs]
if __name__ != "__main__": grid = []
total = ok = 0
print("runs:", [f"{t}-{d}" for t, d, _, _ in RUNS])
for g in grid:
    cfg = dict(DEFAULTS); cfg.update(g)
    r = run(cfg); total += len(r); ok += sum(x == "OK " for x in r)
    print(f"{str(g or 'defaults'):32s}", " ".join(r), flush=True)
print(f"passed {ok}/{total}")
