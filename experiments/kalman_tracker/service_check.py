"""Run core.services.kalman_line_tracer (the plugin's engine) outside the app.
Usage: python service_check.py <worktree root> <seeds: masks|Seeds> [seed_eps]"""
import sys, time, pickle
import numpy as np
root = sys.argv[1]
sys.path.insert(0, root)
from sklearn.cluster import DBSCAN
from core.services.kalman_line_tracer import KalmanLineTracer
D = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit/experiments/kalman_tracker/data"
P = np.load(f"{D}/points.npy")
if sys.argv[2] == "masks":
    S = dict(np.load(f"{D}/seeds.npz")); seeds = np.unique(np.concatenate(list(S.values())))
else:
    old = np.load(f"{D}/cables_seeded_clean.npz")["labels"]; seeds = np.flatnonzero(old >= 0); S = {}
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
lab = DBSCAN(eps=eps, min_samples=2).fit_predict(P[seeds])
groups = [seeds[lab == k] for k in range(lab.max() + 1) if np.sum(lab == k) >= 2]
print(f"{len(seeds)} seeds -> {len(groups)} groups (eps {eps})", flush=True)
t0 = time.time()
tr = KalmanLineTracer(P)
lines = tr.grow_lines(groups, progress_cb=lambda d, t, m: print(f"  {m}", flush=True) if d % max(1, t // 10) == 0 else None)
tr.release()
print(f"{len(lines)} lines in {time.time()-t0:.0f}s, {sum(len(l.indices) for l in lines)} points")
for k, l in enumerate(lines[:12]):
    L = np.linalg.norm(np.diff(l.centerline, axis=0), axis=1).sum()
    on = {a: int(np.isin(v, l.indices).sum()) for a, v in S.items() if np.isin(v, l.indices).any()}
    print(f"  line {k+1}: {L:6.1f} m, {len(l.indices)} pts, stops {len(l.stops)}, seeds {on}")
np.savez(f"{D}/service_{sys.argv[2]}.npz", labels=np.concatenate([np.full(len(l.indices), k) for k, l in enumerate(lines)]),
         idx=np.concatenate([l.indices for l in lines]), **{f"line_{k}": l.centerline for k, l in enumerate(lines)})
