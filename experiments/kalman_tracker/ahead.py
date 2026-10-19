from harness import *
# Profile what lies ahead of each stop along the start->end heading of the stop's local direction.
import json
stops = {"Test1": ([284.1, 490.0, 30.4]), "Test2": ([265.5, 480.7, 22.7])}
tree = None
for t, tip in stops.items():
    s = P[SEEDS[f"{t}_Start"]].astype(float)
    c = s - s.mean(0); _, v = np.linalg.eigh(c.T @ c); ax = v[:, -1]
    e = P[SEEDS[f"{t}_End"]].mean(0)
    if ax @ (e - s.mean(0)) < 0: ax = -ax
    tip = np.array(tip)
    box = np.all(np.abs(P - (tip + 20 * ax)) < 25, axis=1)
    Q = P[box].astype(float) - tip
    along = Q @ ax; perp = Q - np.outer(along, ax)
    horiz = np.cross(ax, [0, 0, 1]); horiz /= np.linalg.norm(horiz)
    side = perp @ horiz; up = perp[:, 2]
    print(f"\n{t} stop at {tip}, heading {ax.round(3)}")
    print(" along(m) | pts within 3cm | 10cm | 30cm | nearest lateral offset of pts in 1m slice (side, up)")
    for a in range(-2, 30, 1):
        sl = (along >= a) & (along < a + 1)
        r = np.hypot(side[sl], up[sl])
        i = np.argmin(r) if sl.any() else None
        print(f" {a:4d}     | {np.sum(r<0.03):5d} | {np.sum(r<0.1):5d} | {np.sum(r<0.3):5d} |",
              (round(side[sl][i],3), round(up[sl][i],3)) if i is not None else "-")
