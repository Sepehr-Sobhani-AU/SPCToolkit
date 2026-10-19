"""Local top + side view of points and centerlines around given spots."""
from find_cables import *
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cab = np.load(f"{S}/" + os.environ.get("CAB", "cables_seeded_clean_v2.npz"))
lines = [cab[f"line_{k}"] for k in range(len([f for f in cab.files if f.startswith("line_")]))]
spots = [tuple(map(float, s.split(","))) for s in sys.argv[2:]]
R = float(os.environ.get("R", "12"))
fig, axs = plt.subplots(2, len(spots), figsize=(6 * len(spots), 9))
cm = plt.get_cmap("tab10")
for j, sp in enumerate(spots):
    c = np.array(sp)
    m = np.all(np.abs(P - c) < [R, R, 4], axis=1)
    q = P[m]
    for i, (a, b) in enumerate([(0, 1), (0, 2)]):
        ax = axs[i, j]
        ax.scatter(q[:, a], q[:, b], s=0.3, c="0.7")
        for k, l in enumerate(lines):
            mm = np.all(np.abs(l - c) < [R, R, 4], axis=1)
            if mm.any():
                ax.plot(l[mm, a], l[mm, b], "-", lw=1.2, color=cm(k % 10), label=str(k + 1))
        ax.set_aspect("equal" if i == 0 else "auto"); ax.legend(fontsize=7)
        ax.set_title(f"{'top' if i == 0 else 'side x-z'} @ {sp}")
fig.tight_layout(); fig.savefig(f"{S}/{sys.argv[1]}", dpi=65)
