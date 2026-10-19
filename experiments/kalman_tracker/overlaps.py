"""Where does one kept centerline run on / end on another one?"""
from find_cables import *
import os
cab = np.load(f"{S}/" + os.environ.get("CAB", "cables_seeded_clean.npz"))
lines = [cab[f"line_{k}"] for k in range(len([f for f in cab.files if f.startswith("line_")]))]
dense = [densify(l, 0.1) for l in lines]
owner = np.concatenate([np.full(len(d), k) for k, d in enumerate(dense)])
allp = np.vstack(dense)
t = cKDTree(allp)
report = []
for k, d in enumerate(dense):
    # distance from each sample of line k to the nearest sample of ANY OTHER line
    near = t.query_ball_point(d, 0.15)
    other = np.array([any(owner[j] != k for j in n) for n in near])
    if not other.any():
        continue
    dd = np.diff(np.r_[0, other.astype(int), 0])
    for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
        ks = sorted({int(owner[j]) for n in near[a:b] for j in n if owner[j] != k})
        where = "start" if a == 0 else ("end" if b == len(d) else "middle")
        report.append((k + 1, round((b - a) * 0.1, 1), where, [x + 1 for x in ks], d[a].round(1).tolist()))
runs = [r for r in report if r[1] >= 1.0]
print(f"{len(lines)} lines; {len(runs)} stretches >= 1 m within 15 cm of another line")
from collections import Counter
print("by position:", Counter(r[2] for r in runs))
print("length histogram:", np.histogram([r[1] for r in runs], bins=[1, 2, 5, 10, 50, 1000])[0])
for r in sorted(runs, key=lambda r: -r[1])[:25]:
    print("  line", r[0], "len", r[1], "m at", r[2], "with", r[3], "@", r[4])

# Separation and point support inside each long overlap stretch
labels = cab["labels"]
print("\nline  len  median-gap  pts-near-this-line(own/other/unlabelled)  empty-samples")
for r in sorted(runs, key=lambda r: -r[1])[:14]:
    k = r[0] - 1
    d = dense[k]
    near = t.query_ball_point(d, 0.15)
    gaps, own, oth, unl, empty = [], 0, 0, 0, 0
    for i, n in enumerate(near):
        o = [j for j in n if owner[j] != k]
        if not o:
            continue
        gaps.append(np.min(np.linalg.norm(allp[o] - d[i], axis=1)))
        c = tree().query_ball_point(d[i], 0.05)
        if not c: empty += 1
        lab = labels[c]
        own += np.sum(lab == k); oth += np.sum((lab >= 0) & (lab != k)); unl += np.sum(lab < 0)
    print(f"{r[0]:4d} {r[1]:5.1f}   {np.median(gaps)*100:5.1f} cm   {own}/{oth}/{unl}   {empty}/{len(gaps)}")
