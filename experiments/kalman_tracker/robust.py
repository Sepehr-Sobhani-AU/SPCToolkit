from ktrack import *
def onwire(path):
    d, _ = tree().query(path)
    return f"{np.mean(d < 0.05)*100:.0f}% of vertices within 5 cm of a point, median {np.median(d)*100:.1f} cm"
for t in TESTS:
    print(t, onwire(np.load(f"{S}/path_{t}_v3.npy")))
grid = [{}, {"step": 0.3}, {"step": 1.0}, {"point_sigma": 0.03}, {"curvature_noise": 0.05},
        {"curvature_noise": 0.005}, {"min_pts": 5}, {"max_curvature": 0.03}, {"tight": 0.08},
        {"model": "straight"}]
for g in grid:
    cfg = dict(DEFAULTS); cfg.update(g)
    out = []
    for t in TESTS:
        path, ev = track(SEEDS[f"{t}_Start"], sign_for(t), cfg)
        mem = np.unique(np.concatenate([np.array(x, dtype=np.intp) for x in tree().query_ball_point(path, 0.04)]))
        sc = score(t, mem)
        out.append(f"{t}: {'REACHED' if sc['reached'] else 'no'} {sc['progress']}")
    print(g or "defaults", " | ".join(out), flush=True)
