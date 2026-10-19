from harness import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cab = np.load(f"{S}/cables.npz"); lab = cab["labels"]
lines = [cab[f"line_{k}"] for k in range(len([f for f in cab.files if f.startswith("line_")]))]
sub = np.random.default_rng(0).choice(len(P), 600000, replace=False)
fig, ax = plt.subplots(1, 2, figsize=(20, 16))
for a in ax:
    a.scatter(P[sub, 0], P[sub, 1], s=0.05, c="0.8")
    a.set_aspect("equal")
cm = plt.get_cmap("tab20")
for k, l in enumerate(lines):
    n = np.sum(lab == k); L = np.linalg.norm(np.diff(l, axis=0), axis=1).sum()
    ax[0].plot(l[:, 0], l[:, 1], lw=0.8, color=cm(k % 20))
    if k < 1 or n / max(L, 1) > 400: ax[0].annotate(str(k + 1), l[len(l)//2, :2], fontsize=8)
# right: zoom on a busy stretch
ax[1].set_xlim(250, 330); ax[1].set_ylim(360, 520)
for k, l in enumerate(lines):
    ax[1].plot(l[:, 0], l[:, 1], lw=1, color=cm(k % 20))
fig.tight_layout(); fig.savefig(f"{S}/plan_all.png", dpi=70)
dens = [(k + 1, int(np.sum(lab == k)), round(float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()))) for k, l in enumerate(lines)]
print("pts/m top:", sorted(dens, key=lambda d: -d[1] / max(d[2], 1))[:8])
l = lines[0]; print("line1 bbox", l.min(0).round(1), l.max(0).round(1), "z of its points", np.percentile(P[lab == 0][:, 2], [5, 50, 95]).round(1))
