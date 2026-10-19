"""Clean the traced lines (data/cables.npz -> data/cables_clean.npz).

Lines are taken in the order they were traced; each sees what the earlier
kept ones claimed:
- Cut a line where it runs on wire an earlier line already holds (> 2 m),
  keep pieces with at least MIN_FREE m on unclaimed wire.
- Drop a piece that is not a wire in open air: points crowd round it
  (ring/core ratio) or it is far denser than a wire.
"""
from find_cables import *

CORE = 0.05
RING = (0.08, 0.25)
SAMPLE = 0.1
MAX_CLAIMED_RUN = 2.0
MIN_FREE = 10.0
MAX_RING_RATIO = 0.5
MAX_PTS_PER_M = 400


def runs(mask):
    """(start, end) index pairs of the True runs in *mask*."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


cab = np.load(f"{S}/cables.npz")
n_lines = len([f for f in cab.files if f.startswith("line_")])
claimed = np.zeros(len(P), dtype=bool)
kept, labels = [], np.full(len(P), -2, dtype=np.int32)
log = {"dup": 0, "clutter": 0, "dense": 0, "short": 0}
for k in range(n_lines):
    dense = densify(cab[f"line_{k}"], SAMPLE)
    core = tree().query_ball_point(dense, CORE)
    n_core = np.array([len(c) for c in core])
    n_claimed = np.array([claimed[c].sum() if c else 0 for c in core])
    on_old = (n_core > 0) & (n_claimed > 0.5 * n_core)
    # cut out long runs on old wire; short ones (a crossing) are kept
    cut = np.zeros(len(dense), dtype=bool)
    for a, b in runs(on_old):
        if (b - a) * SAMPLE > MAX_CLAIMED_RUN:
            cut[a:b] = True
    for a, b in runs(~cut):
        piece = dense[a:b]
        free = (n_core[a:b] > 0) & ~on_old[a:b]
        if free.sum() * SAMPLE < MIN_FREE:
            log["short" if not on_old[a:b].any() else "dup"] += 1
            continue
        mem = np.unique(np.fromiter((i for c in core[a:b] for i in c), dtype=np.intp))
        mem = mem[~claimed[mem]]
        ring_pts = tree().query_ball_point(piece[::5], RING[1])
        rel_counts = [np.sum(np.linalg.norm(P[r] - p, axis=1) > RING[0]) if r else 0
                      for p, r in zip(piece[::5], ring_pts)]
        ring_ratio = sum(rel_counts) / max(sum(n_core[a:b][::5]), 1)
        length = (b - a) * SAMPLE
        if ring_ratio > MAX_RING_RATIO:
            log["clutter"] += 1
            continue
        if len(mem) / length > MAX_PTS_PER_M:
            log["dense"] += 1
            continue
        labels[mem] = len(kept)
        claimed[mem] = True
        kept.append(piece[::5])          # 0.5 m vertices
        print(f"line {k + 1:3d} -> cable {len(kept):3d}: {length:5.0f} m, "
              f"{len(mem):6d} pts, ring {ring_ratio:.2f}", flush=True)
np.savez(f"{S}/cables_clean.npz", labels=labels, **{f"line_{i}": l for i, l in enumerate(kept)})
print(f"{len(kept)} cables kept; dropped {log}; {int((labels >= 0).sum())} points labelled")
