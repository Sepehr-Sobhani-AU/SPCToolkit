import ktrack, synthetic as sy
from ktrack import *
for nm, sp in sy.SCENES.items():
    P_, seed = sy.scene(*sp)
    if nm.startswith("C"): break
ktrack.P = P_; ktrack._TREE = None
ob, orq = ktrack.bridge_search, ktrack.reacquire
def bs(anchor, t, a_from, cfg, **kw):
    r = ob(anchor, t, a_from, cfg, **kw)
    if "W" not in kw:
        print(f"  BRIDGE from x {anchor[0]:.1f} y {anchor[1]:+.3f} a_from {a_from:.1f} sig {kw.get('anchor_sigma',0):.3f}:",
              None if r is None else f"land x {r[0][0]:.1f} y {r[0][1]:+.3f} span {r[2]}")
    return r
def rq(o, t, cfg, **kw):
    r = orq(o, t, cfg, **kw); print(f"  KINK from x {o[0]:.1f} y {o[1]:+.3f}:", None if r is None else f"dir {r[0].round(3)} span {r[1]}"); return r
ktrack.bridge_search, ktrack.reacquire = bs, rq
om = ktrack.measure
def lm(state, c, coasting=False, strict=False):
    r = om(state, c, coasting, strict)
    if 38 < state.position[0] < 62: print(f"x {state.position[0]:6.2f} y {state.position[1]:+.3f} sig {state.lateral_sigma():.3f} {r[2]} gate {r[1]:.3f}")
    return r
ktrack.measure = lm
c = P_[seed] - P_[seed].mean(0); axis = np.linalg.eigh(c.T @ c)[1][:, -1]
path, ev = track(seed, 1 if axis[0] > 0 else -1, dict(DEFAULTS, max_len=70))
print(ev)
