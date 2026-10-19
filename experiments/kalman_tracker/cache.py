import sys, pickle, numpy as np
sys.path.insert(0, ".")
S = sys.argv[1]
d = pickle.load(open("Test_Data/Test1.pcdtk", "rb"))
nodes = d["data_nodes"].data_nodes
root = [n for n in nodes.values() if n.parent_uid is None][0]
np.save(f"{S}/points.npy", np.asarray(root.data.points, dtype=np.float32))
np.savez(f"{S}/seeds.npz", **{n.alias.replace("-", "_"): np.flatnonzero(n.data.mask)
                              for n in nodes.values() if n is not root})
