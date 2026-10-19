import sys, numpy as np
root = sys.argv[1]; key = sys.argv[2]; scene_seed = int(sys.argv[3]); s = int(sys.argv[4])
sys.path.insert(0, root)
import scenes
import core.services.particle_line_tracer as pf
sc = [v for k, v in scenes.make_scenes(scene_seed).items() if k[0] == key][0]
pts = sc["pts"].astype(np.float32)
tr = pf.ParticleLineTracer(pts, **sc["settings"], seed=s)
o_score = tr._score
def score(p, t, k, cfg):
    sc_, seen, j = o_score(p, t, k, cfg)
    x = p.mean(0)[0]
    if x > 3:
        print(f"x {x:6.2f} y-mean {p[:,1].mean():+.3f} y-sd {p[:,1].std():.3f} seen {seen.sum():3d} best {sc_.max():.2f} r {cfg['radius']:.3f}")
    return sc_, seen, j
tr._score = score
lines = tr.grow_lines([sc["seed"]]); tr.release()
