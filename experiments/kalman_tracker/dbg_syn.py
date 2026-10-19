import sys, synthetic as sy, ktrack
from ktrack import *
spec = sy.SCENES[sys.argv[1]]
P_, seed = sy.scene(*spec)
ktrack.P = P_; ktrack._TREE = None
cfg, prof = scaled_cfg(dict(DEFAULTS, max_len=float(sys.argv[2])), P_[seed])
print({k: round(v, 3) for k, v in cfg.items() if k in ("tight", "gate_min", "gate_max", "slab", "step", "reacq_len", "bridge_win", "coast_after")})
orig = ktrack.measure
def lm(state, c, coasting=False, strict=False):
    r = orig(state, c, coasting, strict); print(f"x {state.position[0]:6.2f} sig {state.lateral_sigma():.3f} gate {r[1]:.3f} {r[2]}"); return r
ktrack.measure = lm
c = P_[seed] - P_[seed].mean(0); axis = np.linalg.eigh(c.T @ c)[1][:, -1]
path, ev = ktrack.track(seed, 1 if axis[0] > 0 else -1, dict(DEFAULTS, max_len=float(sys.argv[2])))
print(ev)
