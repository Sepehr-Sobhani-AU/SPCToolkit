import sys, pickle, numpy as np
sys.path.insert(0, ".")
d = pickle.load(open("Test_Data/Test1.pcdtk", "rb"))
nodes = d["data_nodes"].data_nodes
root = [n for n in nodes.values() if n.parent_uid is None][0]
P = np.asarray(root.data.points, dtype=np.float64)
seeds = {n.alias: P[n.data.mask] for n in nodes.values() if n is not root}
print("attributes:", list(root.data.attributes) if root.data.attributes else None,
      "intensity:", root.data.intensity is not None, "colors:", root.data.colors is not None,
      "normals:", root.data.normals is not None)
for t in ("Test1", "Test2"):
    s, e = seeds[f"{t}-Start"], seeds[f"{t}-End"]
    gap = e.mean(0) - s.mean(0)
    print(f"\n{t}: start->end {np.linalg.norm(gap):.1f} m (horiz {np.linalg.norm(gap[:2]):.1f}, dz {gap[2]:+.1f})")
    for name, q in (("Start", s), ("End", e)):
        c = q - q.mean(0); w, v = np.linalg.eigh(c.T @ c / len(q)); ax = v[:, -1]
        ang = np.degrees(np.arccos(abs(ax @ gap) / np.linalg.norm(gap)))
        print(f"  {name}: {len(q)} pts, axis {ax.round(2)}, spread sd {np.sqrt(w[::-1]).round(3)} m, {ang:.0f} deg off the straight start-end line")
