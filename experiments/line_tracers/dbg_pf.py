import sys, numpy as np
root, key, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, root)
import scenes
import core.services.particle_line_tracer as pf
sc = [v for k, v in scenes.make_scenes().items() if k[0] == key][0]
pts = sc["pts"].astype(np.float32)
tr = pf.ParticleLineTracer(pts, **sc["settings"])
orig = tr._score
count = [0]
def score(p, t, k, cfg):
    s, seen, joined = orig(p, t, k, cfg)
    count[0] += 1
    if count[0] <= steps:
        km = np.linalg.norm(k, axis=1)
        print(f"step {count[0]:3d} mean p {p.mean(0).round(2)} spread {np.linalg.norm(p - p.mean(0), axis=1).max():.2f} "
              f"seen {seen.sum():3d} span max {s.max():.2f} |k| med {np.median(km):.3f} radius {cfg['radius']:.3f}")
    return s, seen, joined
tr._score = score
lines = tr.grow_lines([sc["seed"]]); tr.release()
print([ (l.centerline[0].round(2), l.centerline[-1].round(2)) for l in lines])
