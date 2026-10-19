from find_cables import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cab = np.load(f"{S}/cables.npz")
ids = [int(x) for x in sys.argv[1].split(",")]
fig, axs = plt.subplots(1, len(ids), figsize=(4 * len(ids), 4.4))
for ax, k in zip(np.atleast_1d(axs), ids):
    piece = densify(cab[f"line_{k-1}"], 0.1)
    S1, U1 = [], []
    for i in range(2, len(piece) - 2, 10):
        d = piece[i + 2] - piece[i - 2]; d /= np.linalg.norm(d)
        side = np.cross(d, [0, 0, 1]); side /= np.linalg.norm(side); up = np.cross(side, d)
        r = tree().query_ball_point(piece[i], 0.6)
        rel = P[r].astype(float) - piece[i]
        m = np.abs(rel @ d) < 0.05
        S1 += list(rel[m] @ side); U1 += list(rel[m] @ up)
    ax.hist2d(S1, U1, bins=60, range=[[-0.6, 0.6], [-0.6, 0.6]], cmap="magma_r", cmin=1)
    ax.set_title(f"line {k}"); ax.set_aspect("equal")
fig.suptitle("cross-sections (m): side vs up, all along the line"); fig.tight_layout()
fig.savefig(f"{S}/xsec.png", dpi=70)
