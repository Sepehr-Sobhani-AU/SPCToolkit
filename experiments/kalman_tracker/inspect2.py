import sys, pickle, numpy as np
sys.path.insert(0, ".")
d = pickle.load(open("Test_Data/Test1.pcdtk", "rb"))
nodes = d["data_nodes"].data_nodes
root = [n for n in nodes.values() if n.parent_uid is None][0]
pc = root.data
P = np.asarray(pc.points)
print("root vars:", {k: type(v).__name__ for k, v in vars(root).items()})
print("pc vars:", [k for k in vars(pc)])
print("extent:", P.min(0), P.max(0))
for uid, n in nodes.items():
    if n is root: continue
    m = n.data
    print("\n", uid, {k: (v.shape if hasattr(v, "shape") else v) for k, v in vars(n).items() if k != "data"})
    print("  masks vars:", {k: (getattr(v, "shape", None), getattr(v, "dtype", type(v).__name__)) for k, v in vars(m).items()})
    for k, v in vars(m).items():
        if hasattr(v, "dtype") and v.dtype == bool:
            sel = P[v]; print(f"  {k}: {v.sum()} pts, extent {np.ptp(sel,0).round(2)}, centre {sel.mean(0).round(2)}")
        elif isinstance(v, (list, dict)):
            print(f"  {k}: {type(v).__name__} len {len(v)}", (list(v)[:3] if isinstance(v, dict) else [getattr(x,'shape',x) for x in v[:3]]))
