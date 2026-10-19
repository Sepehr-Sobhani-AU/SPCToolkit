from harness import *
from core.services.linear_region_grower import LinearRegionGrower
import json
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
base = dict(ransac_threshold=0.03, cylinder_radius=0.03, cylinder_length=0.5,
            reach_factor=3.0, min_points=5, min_angle_deg=5.0, max_angle_deg=20.0,
            max_steps=3000)
base.update(cfg)
g = LinearRegionGrower(P, **base)
for t in TESTS:
    t0 = time.time()
    mem = g.grow(SEEDS[f"{t}_Start"])
    stops = [(st.reason, st.tip.round(1).tolist()) for st in g.march_stops()]
    print(t, f"{time.time()-t0:.0f}s", score(t, mem), stops, flush=True)
g.release()
