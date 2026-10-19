"""Trace cables from given seed groups (the 'Seeds' cloud of Test2.pcdtk).

Each seed group is one cluster label of the Seeds cloud, mapped onto the main
cloud by exact position. For each group: find its two ends (extremes along
its axis), take the local direction there (PCA of the last few metres), and
trace from end A inward through the group and on beyond end B, plus from
end A outward. Then the same cleanup as the automatic run (postfilter).
Writes data/cables_seeded.npz (same format as cables.npz).
"""
import pickle
from find_cables import *

ROOT = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit"
END_REACH = 3.0      # metres of seed used for the direction at each end

d = pickle.load(open(f"{ROOT}/Test_Data/Test2.pcdtk", "rb"))
nodes = d["data_nodes"].data_nodes
roots = {(n.alias or n.params): n for n in nodes.values() if n.parent_uid is None}
cloud = np.asarray(roots["Test Data"].data.points)
assert np.array_equal(cloud, P), "Test2 cloud differs from the cached cloud"
sp = np.asarray(roots["Seeds"].data.points)
sl = np.asarray(roots["Seeds"].data.attributes["cluster_labels"])
dist, idx = tree().query(sp)
assert dist.max() < 1e-6, f"seed points not in the cloud (max {dist.max()})"
groups = [idx[sl == k] for k in np.unique(sl) if k >= 0]
print(f"{len(groups)} seed groups, {len(idx)} seed points", flush=True)


def end_frame(pts, axis, sign):
    """Point at the *sign* end of pts along axis, and the local direction
    there pointing INTO the group."""
    a = (pts - pts.mean(0)) @ axis * sign
    tip = pts[np.argmax(a)]
    near = pts[np.linalg.norm(pts - tip, axis=1) < END_REACH]
    c = near - near.mean(0)
    loc = np.linalg.eigh(c.T @ c)[1][:, -1] if len(near) >= 3 else axis
    inward = -loc if loc @ (axis * sign) > 0 else loc
    return tip, inward


cfg = dict(DEFAULTS); cfg["max_len"] = 1500.0
t0 = time.time()
lines = []
for k, g in enumerate(groups):
    pts = P[g].astype(float)
    c = pts - pts.mean(0)
    axis = np.linalg.eigh(c.T @ c)[1][:, -1]
    a_tip, a_in = end_frame(pts, axis, +1)
    through, _ = track(None, 0, cfg, start=a_tip, ax=a_in)      # A -> through group -> beyond B
    outward, _ = track(None, 0, cfg, start=a_tip, ax=-a_in)     # A -> outward
    line = np.vstack([outward[::-1], through[1:]])
    lines.append(line)
    print(f"seed group {k + 1:2d}: {len(g):6d} seed pts -> {path_length(line):6.0f} m traced  [{time.time()-t0:.0f}s]", flush=True)
np.savez(f"{S}/cables_seeded.npz", labels=np.full(len(P), -2, dtype=np.int32),
         **{f"line_{i}": l for i, l in enumerate(lines)})
