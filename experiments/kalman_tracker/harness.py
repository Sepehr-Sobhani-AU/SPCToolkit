"""Shared test harness: load cloud + seeds, score a trace against the end seed."""
import sys, time, numpy as np
sys.path.insert(0, "/media/sepehr/Data2/OneDrive/AI/SPCToolkit")
from scipy.spatial import cKDTree
S = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
P = np.load(f"{S}/points.npy")
SEEDS = dict(np.load(f"{S}/seeds.npz"))
TESTS = ("Test1", "Test2")

def score(test, members, centerline=None):
    s = P[SEEDS[f"{test}_Start"]].astype(float); e = P[SEEDS[f"{test}_End"]].astype(float)
    sc, ec = s.mean(0), e.mean(0)
    total = np.linalg.norm(ec - sc); u = (ec - sc) / total
    m = P[np.asarray(members, dtype=np.intp)].astype(float)
    tree = cKDTree(e)
    d, _ = tree.query(m)
    end_hits = np.isin(SEEDS[f"{test}_End"], members).sum()
    prog = ((m - sc) @ u)
    far = m[np.argmax(prog)]
    return dict(reached=bool(end_hits > 0), end_seed_points_claimed=int(end_hits),
                closest_to_end_m=round(float(d.min()), 2),
                progress=f"{prog.max():.1f}/{total:.1f} m",
                furthest_point=far.round(2).tolist(), n_members=len(m))
