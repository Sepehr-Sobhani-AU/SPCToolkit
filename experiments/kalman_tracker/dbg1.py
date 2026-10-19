import ktrack
from ktrack import *
orig = ktrack.measure
s0 = P[SEEDS["Test1_Start"]].mean(0).astype(float); e0 = P[SEEDS["Test1_End"]].mean(0).astype(float)
u = e0 - s0; u[2] = 0; u /= np.linalg.norm(u)
def logged(state, cfg):
    z, g, why = orig(state, cfg)
    al = (state.position - s0) @ u
    if 84 < al < 105:
        print(f"al {al:6.1f} z {state.position[2]-s0[2]:+.2f} tz {state.tangent[2]:+.3f} k {np.linalg.norm(state.curvature):.4f} sig {state.lateral_sigma():.3f} g {g:.3f} {why}"
              + (f" corr {np.linalg.norm(z-state.position):.3f}" if z is not None else ""))
    return z, g, why
ktrack.measure = logged
cfg = dict(DEFAULTS)
path, ev = track(SEEDS["Test1_Start"], sign_for("Test1"), cfg)
print(ev)
