from harness import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Corridor view along the straight start->end line: x = distance along, y = side offset / height.
t = sys.argv[1]; a0, a1 = float(sys.argv[2]), float(sys.argv[3]); half = float(sys.argv[4])
s = P[SEEDS[f"{t}_Start"]].mean(0).astype(float); e = P[SEEDS[f"{t}_End"]].mean(0).astype(float)
u = e - s; u[2] = 0; u /= np.linalg.norm(u); h = np.cross([0, 0, 1], u)
Q = P.astype(float) - s
al = Q @ u; sd = Q @ h
import os
zl = [float(x) for x in os.environ.get("ZWIN", "-1e9 1e9").split()]
m = (al > a0) & (al < a1) & (np.abs(sd) < half) & (Q[:, 2] > zl[0]) & (Q[:, 2] < zl[1])
fig, ax = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
ax[0].scatter(al[m], sd[m], s=0.3, c=Q[m, 2], cmap="turbo"); ax[0].set_ylabel("side (m)")
ax[1].scatter(al[m], Q[m, 2], s=0.3, c=sd[m], cmap="turbo", vmin=-half, vmax=half); ax[1].set_ylabel("height (m)")
for name, c in (("Start", "red"), ("End", "magenta")):
    q = P[SEEDS[f"{t}_{name}"]].astype(float) - s
    ax[0].scatter(q @ u, q @ h, s=4, c=c); ax[1].scatter(q @ u, q[:, 2], s=4, c=c)
for extra in sys.argv[6:]:
    q = np.load(extra).astype(float) - s
    ax[0].plot(q @ u, q @ h, "k-", lw=1.2); ax[1].plot(q @ u, q[:, 2], "k-", lw=1.2)
ax[1].set_xlabel("along start->end (m)"); fig.suptitle(f"{t} corridor {a0}..{a1} m, +-{half} m"); fig.tight_layout()
fig.savefig(sys.argv[5], dpi=80)
