import sys, numpy as np
which, root, key = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
import scenes
from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.particle_line_tracer import ParticleLineTracer
from core.services.line_tracer_base import line_profile
sc = [v for k, v in scenes.make_scenes().items() if k[0] == key][0]
pts = sc["pts"].astype(np.float32)
cls = {"kalman": KalmanLineTracer, "particle": ParticleLineTracer}[which]
tr = cls(pts, **sc["settings"])
prof = line_profile(pts[sc["seed"]])
print("seed profile:", {k: round(v, 4) for k, v in prof.items()}, "seed n", len(sc["seed"]))
lines = tr.grow_lines([sc["seed"]]); tr.release()
for l in lines:
    c = l.centerline
    print(f"line: {len(c)} vertices, first {c[0].round(2)}, last {c[-1].round(2)}, stops {[s.reason for s in l.stops]}")
    print("  every 5th vertex:", c[::5].round(2).tolist()[:14])
