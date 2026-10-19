import sys, numpy as np
root = sys.argv[1]; key = sys.argv[2]; scene_seed = int(sys.argv[3]); s = int(sys.argv[4])
sys.path.insert(0, root)
import scenes
import core.services.particle_line_tracer as pf
sc = [v for k, v in scenes.make_scenes(scene_seed).items() if k[0] == key][0]
pts = sc["pts"].astype(np.float32)
tr = pf.ParticleLineTracer(pts, **sc["settings"], seed=s)
o_prop, o_est = tr._propose, tr._estimate
last = {"y": 0.0}
def est(p, t, w, r):
    e = o_est(p, t, w, r)
    if abs(e[0][1] - last["y"]) > 0.1:
        print(f"  estimate jumps y {last['y']:+.2f} -> {e[0][1]:+.2f} at x {e[0][0]:.2f}; particle y spread {p[:,1].min():+.2f}..{p[:,1].max():+.2f}, weight max {w.max():.2f}")
    last["y"] = e[0][1]
    return e
def prop(p, t, k, anchor, heading, gap, rng, cfg):
    r = o_prop(p, t, k, anchor, heading, gap, rng, cfg)
    moved = (np.abs(r[0] - p).sum(1) > 1e-9)
    if moved.any():
        print(f"  PROPOSE at gap {gap:.1f} from x {anchor[0]:.2f}: re-aimed {moved.sum()} to y {r[0][moved][:,1].mean():+.2f} x {r[0][moved][:,0].mean():.2f}")
    return r
tr._propose, tr._estimate = prop, est
lines = tr.grow_lines([sc["seed"]]); tr.release()
