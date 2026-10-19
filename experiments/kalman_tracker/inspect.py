import sys, pickle, numpy as np
sys.path.insert(0, ".")
d = pickle.load(open("Test_Data/Test1.pcdtk", "rb"))
print("keys:", list(d) if isinstance(d, dict) else type(d))
nodes = d["data_nodes"].data_nodes
print("visibility:", d.get("tree_visibility"))
kids = {}
for uid, n in nodes.items():
    kids.setdefault(str(n.parent_uid) if n.parent_uid else None, []).append(str(uid))
def show(uid, depth):
    n = nodes.get(uid) or nodes.get(__import__("uuid").UUID(uid))
    data = n.data
    info = type(data).__name__
    for a in ("points", "labels"):
        v = getattr(data, a, None)
        if v is not None and hasattr(v, "shape"): info += f" {a}={v.shape}"
    if a == "labels" and hasattr(data, "labels"):
        lab = data.labels; u = np.unique(lab)
        info += f" labels: {len(u)} unique, min {u.min()}, max {u.max()}"
    tr = getattr(data, "line_traces", None)
    if isinstance(tr, dict): info += f" line_traces: {len(tr.get('lines', []))} lines"
    print("  " * depth + f"- {n.params!s:.60} [{n.data_type}] {info}")
    for k in kids.get(uid, []): show(k, depth + 1)
for r in kids[None]: show(r, 0)
