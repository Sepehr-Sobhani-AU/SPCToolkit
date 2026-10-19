import sys, numpy as np
root = sys.argv[1]; keys = sys.argv[2]
sys.path.insert(0, root)
import scenes
from core.services.particle_line_tracer import ParticleLineTracer
for scene_seed in (11, 12, 13):
    S = scenes.make_scenes(scene_seed)
    for name, sc in S.items():
        if name[0] not in keys: continue
        pts = sc["pts"].astype(np.float32)
        for s in (0, 1, 2):
            tr = ParticleLineTracer(pts, **sc["settings"], seed=s)
            lines = tr.grow_lines([sc["seed"]]); tr.release()
            line = max(lines, key=lambda l: np.isin(sc["seed"], l.indices).sum())
            ok, info = scenes.judge(sc, line)
            if not ok:
                c = line.centerline; off = sc["dist"](c)
                bad = np.flatnonzero(off > 0.25)
                print(f"{name[:1]} scene{scene_seed} s{s}: {info}; first off-line vertex at x {c[bad[0], 0]:.1f} y {c[bad[0], 1]:+.2f}" if len(bad) else f"{name[:1]} scene{scene_seed} s{s}: {info}; x range {c[:,0].min():.1f}..{c[:,0].max():.1f}")
