import sys, numpy as np
root, key = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import scenes
import core.services.particle_line_tracer as pf
sc = [v for k, v in scenes.make_scenes().items() if k[0] == key][0]
pts = sc["pts"].astype(np.float32)
tr = pf.ParticleLineTracer(pts, **sc["settings"])
o_score, o_prop, o_ray = tr._score, tr._propose, tr._ray_search
def score(p, t, k, cfg):
    s, seen, j = o_score(p, t, k, cfg)
    print(f"  score at x {p.mean(0)[0]:6.2f} y {p.mean(0)[1]:+.2f} seen {seen.sum():3d} best {s.max():.2f}")
    return s, seen, j
def prop(p, t, k, anchor, heading, gap, rng, cfg):
    r = o_prop(p, t, k, anchor, heading, gap, rng, cfg)
    moved = (np.abs(r[0] - p).sum(1) > 1e-9).sum()
    print(f"  PROPOSE gap {gap:.1f} anchor x {anchor[0]:.2f} heading {heading.round(2)} -> re-aimed {moved}")
    return r
def ray(anchor, t, a_from, cfg, **kw):
    r = o_ray(anchor, t, a_from, cfg, **kw)
    print(f"    ray from x {anchor[0]:.2f} cone {kw['cone_deg']:.0f}: {None if r is None else (r['direction'].round(2), r['span'])}")
    return r
tr._score, tr._propose, tr._ray_search = score, prop, ray
lines = tr.grow_lines([sc["seed"]]); tr.release()
