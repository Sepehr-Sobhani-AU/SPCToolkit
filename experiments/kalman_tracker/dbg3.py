import ktrack
from ktrack import *
T = sys.argv[1]; lo, hi = float(sys.argv[2]), float(sys.argv[3])
orig = ktrack.measure
s0 = P[SEEDS[f"{T}_Start"]].mean(0).astype(float); e0 = P[SEEDS[f"{T}_End"]].mean(0).astype(float)
u = e0 - s0; u[2] = 0; u /= np.linalg.norm(u); h = np.cross([0, 0, 1], u)
def logged(state, cfg, coasting=False, strict=False):
    z, g, why = orig(state, cfg, coasting, strict)
    q = state.position - s0
    if lo < q @ u < hi:
        print(f"al {q@u:6.1f} side {q@h:+.2f} z {q[2]:+.2f} tz {state.tangent[2]:+.3f} k {np.linalg.norm(state.curvature):.4f} sig {state.lateral_sigma():.3f} g {g:.3f} {why}"
              + (f" corr {np.linalg.norm(z-state.position):.3f}" if z is not None else ""))
    return z, g, why
ktrack.measure = logged
orig_re = ktrack.reacquire
def re_logged(o, t, cfg):
    r = orig_re(o, t, cfg); print("   REACQ", (o - s0) @ u, None if r is None else (r[0].round(3), r[1])); return r
ktrack.reacquire = re_logged
cfg = dict(DEFAULTS); cfg.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
path, ev = track(SEEDS[f"{T}_End"], -sign_for(T), cfg); np.save(f"{S}/path_back.npy", path)
print(ev)
