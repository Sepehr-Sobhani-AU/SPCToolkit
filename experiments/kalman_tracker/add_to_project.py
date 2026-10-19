"""Add the traced cables (data/cables.npz) to a COPY of the test project.

Builds the same branches Linear Region Growing does — a Clusters result
(one label per line, UNASSIGNED for the rest, traces carried on it) under the
root cloud, with a centerlines wireframe under that — and saves the project
under a new name. The original file is not touched.
"""
import os
import sys
import pickle
import numpy as np

ROOT = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit"
sys.path.insert(0, ROOT)
from core.entities.data_node import DataNode
from core.entities.clusters import Clusters
from core.services.linear_region_grower import (
    GrownLine, MarchStop, lines_to_traces, centerlines_to_vector_feature,
)

SRC = f"{ROOT}/Test_Data/Test1.pcdtk"
DST = f"{ROOT}/Test_Data/Test1_cables.pcdtk"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cables.npz")
PARAMS = {"growth_mode": "Axis Trace", "tracker": "Kalman (experimental)",
          "cylinder_radius": 0.05, "ransac_threshold": 0.05}

if os.path.exists(DST):
    sys.exit(f"{DST} already exists - not overwriting")

project = pickle.load(open(SRC, "rb"))
nodes = project["data_nodes"]
root = [n for n in nodes.data_nodes.values() if n.parent_uid is None][0]

cab = np.load(DATA)
labels = cab["labels"].astype(np.int32)
n_lines = len([k for k in cab.files if k.startswith("line_")])
lines = []
for k in range(n_lines):
    c = cab[f"line_{k}"].astype(np.float32)
    # The march stops are where each end of the trace gave up.
    stops = [MarchStop(c[0], (c[0] - c[1]) / max(np.linalg.norm(c[0] - c[1]), 1e-9), "too_few_points"),
             MarchStop(c[-1], (c[-1] - c[-2]) / max(np.linalg.norm(c[-1] - c[-2]), 1e-9), "too_few_points")]
    lines.append(GrownLine(np.flatnonzero(labels == k), c, [], [], stops))

clusters = Clusters(labels=labels,
                    cluster_names={k: f"Cable {k + 1}" for k in range(n_lines)},
                    line_traces=lines_to_traces(lines, PARAMS), tint_locked=False)
for k in range(n_lines):
    clusters.locked_clusters.setdefault(k, set()).add("select")
clusters.set_random_color()

result = DataNode("linear_region_growing", alias="power_cables_kalman", data=clusters,
                  data_type="cluster_labels", parent_uid=root.uid,
                  depends_on=[root.uid], tags=["linear_region_growing", PARAMS])
result.memory_size = f"{labels.nbytes / 1e6:.2f} MB"
result_uid = nodes.add_node(result)

vf = centerlines_to_vector_feature([l.centerline for l in lines])
vf.cluster_reference = str(result_uid)
cl = DataNode("centerlines", alias="centerlines", data=vf, data_type="vector_feature",
              parent_uid=result_uid, depends_on=[root.uid], tags=["linear_region_growing", PARAMS])
cl.memory_size = "0.00 MB"
cl_uid = nodes.add_node(cl)

vis = dict(project.get("tree_visibility") or {})
vis[str(root.uid)] = False
vis[str(result_uid)] = True
vis[str(cl_uid)] = True
project["tree_visibility"] = vis

with open(DST, "wb") as f:
    pickle.dump(project, f)
print(f"saved {DST}: {n_lines} cables, {np.sum(labels >= 0)} points labelled")
