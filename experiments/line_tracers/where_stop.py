import sys, numpy as np
from sklearn.cluster import DBSCAN
which, root, a, b = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, root)
from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.particle_line_tracer import ParticleLineTracer
from core.services.line_tracer_base import line_profile
D = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit/experiments/kalman_tracker/data"
P = np.load(f"{D}/points.npy"); S = dict(np.load(f"{D}/seeds.npz"))
idx = S[a]; lab = DBSCAN(eps=0.10, min_samples=2).fit_predict(P[idx])
groups = [idx[lab == k] for k in range(lab.max() + 1) if np.sum(lab == k) >= 2]
print("groups", [len(g) for g in groups], "seed profile", {k: round(v, 4) for k, v in line_profile(P[idx]).items()})
cls = {"kalman": KalmanLineTracer, "particle": ParticleLineTracer}[which]
tr = cls(P); lines = tr.grow_lines(groups); tr.release()
s0 = P[S[a]].mean(0); e0 = P[S[b]].mean(0); u = (e0 - s0) / np.linalg.norm(e0 - s0)
for l in lines:
    c = l.centerline; al = (c - s0) @ u
    print(f"line {len(c)} vertices, along {al.min():.1f}..{al.max():.1f} of {np.linalg.norm(e0 - s0):.1f} m, covers {b}: {np.isin(S[b], l.indices).mean():.0%}")
