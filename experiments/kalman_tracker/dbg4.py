import ktrack
from ktrack import *
s0 = P[SEEDS["Test1_Start"]].mean(0).astype(float); e0 = P[SEEDS["Test1_End"]].mean(0).astype(float)
u = e0 - s0; u[2] = 0; u /= np.linalg.norm(u)
cap = {}
orig = ktrack.reacquire
def hook(o, t, cfg):
    cap["o"], cap["t"] = o.copy(), t.copy(); return orig(o, t, cfg)
ktrack.reacquire = hook
cfg = dict(DEFAULTS); cfg.update(step=0.75, max_len=60)
track(SEEDS["Test1_End"], -sign_for("Test1"), cfg)
o, t = cap["o"], cap["t"]
print("anchor al", (o - s0) @ u, "t", t.round(3))
# what does the wire look like ahead of the anchor: nearest points per 0.5 m along t
idx = tree().query_ball_point(o + 2.5 * t, 3.5)
rel = P[idx].astype(float) - o
a = rel @ t; lat = rel - np.outer(a, t)
for lo in np.arange(0, 5, 0.5):
    m = (a >= lo) & (a < lo + 0.5)
    if m.any():
        L = lat[m]; n = np.linalg.norm(L, axis=1); i = np.argmin(n)
        print(f"a {lo:.1f}: n {m.sum():4d} nearest lateral {n[i]:.3f} (z {L[i][2]:+.3f}) angle {np.degrees(np.arctan2(n[i], lo+0.25)):.1f}")
for over in [dict(), dict(reacq_min_span=4), dict(reacq_len=5.0), dict(reacq_radius=0.08), dict(reacq_max_deg=45)]:
    c = dict(cfg); c.update(over)
    r = orig(o, t, c)
    print(over, None if r is None else (np.degrees(np.arccos(r[0] @ t)).round(1), r[1]))
