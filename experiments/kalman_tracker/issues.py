"""Measure the two reported problems in a result (data/<CAB>):
1. the same line drawn twice - stretches where two centerlines coincide;
2. long reaches through empty space that end on another line."""
from find_cables import *
import os
cab = np.load(f"{S}/" + os.environ.get("CAB", "cables_clean.npz"))
lines = [cab[f"line_{k}"] for k in range(len([f for f in cab.files if f.startswith("line_")]))]
dense = [densify(l, 0.1) for l in lines]
owner = np.concatenate([np.full(len(d), k) for k, d in enumerate(dense)])
allp = np.vstack(dense); t = cKDTree(allp)

def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

# 1. coincident stretches, by how close
bins = {"< 1 cm": 0.01, "1-3 cm": 0.03, "3-15 cm": 0.15}
for label, r in bins.items():
    n, total = 0, 0.0
    for k, d in enumerate(dense):
        dd, ii = t.query(d, k=8, distance_upper_bound=r)
        other = np.array([any(owner[j] != k for j, x in zip(row_i, row_d) if np.isfinite(x)) for row_i, row_d in zip(ii, dd)])
        for a, b in runs(other):
            if (b - a) * 0.1 >= 1.0: n += 1; total += (b - a) * 0.1
    print(f"1. stretches >= 1 m where another centerline is within {label:7s}: {n:3d} stretches, {total:6.0f} m (counted from both lines)")

# 2. empty reaches: stretches with no cloud point within 5 cm, longer than 5 m
print("\n2. reaches through empty space (no point within 5 cm) longer than 5 m:")
rows = []
for k, d in enumerate(dense):
    has = np.array([len(x) > 0 for x in tree().query_ball_point(d, 0.05)])
    for a, b in runs(~has):
        L = (b - a) * 0.1
        if L < 5: continue
        at_end = a == 0 or b == len(d)
        # what does the reach lead to: another line within 15 cm just after it?
        nxt = d[min(b, len(d) - 1)]
        near = [owner[j] for j in t.query_ball_point(nxt, 0.15) if owner[j] != k]
        rows.append((k + 1, round(L, 1), "line end" if at_end else "inside", sorted(set(int(x) + 1 for x in near))))
print(f"   {len(rows)} reaches; {sum(1 for r in rows if r[3])} of them arrive on another line")
for r in sorted(rows, key=lambda r: -r[1])[:15]:
    print("   line", r[0], f"{r[1]:5.1f} m", r[2], "-> arrives on line(s)" if r[3] else "", r[3] or "")
