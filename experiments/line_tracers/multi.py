import sys, numpy as np
root = sys.argv[1]
sys.path.insert(0, root)
import scenes
from core.services.particle_line_tracer import ParticleLineTracer
from core.services.kalman_line_tracer import KalmanLineTracer
tot = {}
for scene_seed in (11, 12, 13):
    S = scenes.make_scenes(scene_seed)
    for name, sc in S.items():
        pts = sc["pts"].astype(np.float32)
        runs = [("kalman", KalmanLineTracer, {})] + [(f"particle s{s}", ParticleLineTracer, {"seed": s}) for s in (0, 1, 2)]
        for label, cls, extra in runs:
            tr = cls(pts, **sc["settings"], **extra)
            lines = tr.grow_lines([sc["seed"]]); tr.release()
            line = max(lines, key=lambda l: np.isin(sc["seed"], l.indices).sum()) if lines else None
            ok = line is not None and scenes.judge(sc, line)[0]
            key = "kalman" if label == "kalman" else "particle"
            tot.setdefault(key, {}).setdefault(name, []).append(ok)
for key, d in tot.items():
    print(key, " | ".join(f"{n[0]} {sum(v)}/{len(v)}" for n, v in d.items()), "| total", sum(sum(v) for v in d.values()), "/", sum(len(v) for v in d.values()))
