"""Trace lines from a set of selected seed points (general: any line).

1. DBSCAN the seeds into objects. The distance is taken from the seeds
   themselves: a few times their typical neighbour gap.
2. Each object, largest first: start at its densest spot, measure the line
   there (radius = median distance + 2 robust sd), trace both ways.
   Points already on a line are seen but never taken: a trace that runs onto
   one stops and joins it; a gap is never bridged onto one.
3. The line takes the points within its radius. Seeds it covers are done;
   seeds of the object it did not reach are traced next (an object may hold
   more than one line where lines touch).
4. Safety: a stretch of a new line closer to an existing line than their
   radii is the same line and is cut out.

Usage: python trace_seeds.py <project.pcdtk> <seed source> <out.npz>
  seed source: "masks"  -> every mask branch under the cloud, together
               "<name>" -> a root point cloud of that name (e.g. Seeds)
"""
import os
import sys
import pickle
from sklearn.cluster import DBSCAN
import ktrack
from find_cables import *

ROOT = "/media/sepehr/Data2/OneDrive/AI/SPCToolkit"
NEIGHBOUR_GAPS = 4.0      # DBSCAN eps = this many typical seed neighbour gaps
MIN_SAMPLES = 3
MIN_LINE_POINTS = 10


def load_seeds(project, source):
    d = pickle.load(open(f"{ROOT}/Test_Data/{project}", "rb"))
    nodes = d["data_nodes"].data_nodes
    cloud = [n for n in nodes.values() if n.parent_uid is None and (n.alias or n.params) == "Test Data"][0]
    assert np.array_equal(np.asarray(cloud.data.points), P), "cloud differs from the cached cloud"
    if source == "masks":
        masks = [n.data.mask for n in nodes.values() if n.parent_uid == cloud.uid and n.data_type == "masks"]
        return np.flatnonzero(np.any(masks, axis=0))
    src = [n for n in nodes.values() if n.parent_uid is None and (n.alias or n.params) == source][0]
    dist, idx = tree().query(np.asarray(src.data.points))
    assert dist.max() < 1e-6
    return np.unique(idx)


def cluster_seeds(seed_idx):
    pts = P[seed_idx].astype(float)
    gap = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
    eps = NEIGHBOUR_GAPS * gap
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(pts)
    groups = [seed_idx[lab == k] for k in range(lab.max() + 1)]
    groups.sort(key=len, reverse=True)
    return groups, eps, int(np.sum(lab < 0))


def dense_start(idx, window):
    """The group points within *window* of the densest spot of *idx*."""
    pts = P[idx].astype(float)
    sub = np.arange(len(idx)) if len(idx) <= 2000 else \
        np.random.default_rng(0).choice(len(idx), 2000, replace=False)
    t = cKDTree(pts)
    counts = np.array([len(x) for x in t.query_ball_point(pts[sub], window / 3)])
    centre = pts[sub[np.argmax(counts)]]
    return idx[t.query_ball_point(centre, window)]


def trace_one(mini, cfg):
    """Both halves from the back end of a small seed: forward through it and
    on, backward straight out. No overlap to trim."""
    seed = P[mini].astype(float)
    c = seed - seed.mean(0)
    ax = np.linalg.eigh(c.T @ c)[1][:, -1]
    start = seed[np.argmin(c @ ax)]
    fwd, ev_f = track(None, 0, cfg, start=start, ax=ax)
    back, ev_b = track(None, 0, cfg, start=start, ax=-ax)
    line = np.vstack([back[::-1], fwd[1:]]) if len(back) > 1 else fwd
    end_f = ev_f[-1] if ev_f and ev_f[-1][1] in ("joined", "end") else (None, "end", "")
    end_b = ev_b[-1] if ev_b and ev_b[-1][1] in ("joined", "end") else (None, "end", "")
    return line, (end_b, end_f)


def cut_same_line(line, radius, kept):
    """Pieces of *line* not lying on an existing line (closer than the two
    radii) for longer than a window - the same line, drawn once."""
    if not kept:
        return [line]
    dense = densify(line, 0.1)
    near = np.zeros(len(dense), dtype=bool)
    for other, r_other in kept:
        dd, _ = cKDTree(densify(other, 0.1)).query(dense)
        near |= dd < max(radius, r_other)
    cut = np.zeros(len(dense), dtype=bool)
    d = np.diff(np.r_[0, near.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) * 0.1 > DEFAULTS["slab"]:
            cut[a:b] = True
    d = np.diff(np.r_[0, (~cut).astype(int), 0])
    return [dense[a:b][::5] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= 2]


if __name__ == "__main__":
    project, source, out = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    seeds = load_seeds(project, source)
    groups, eps, noise = cluster_seeds(seeds)
    print(f"{len(seeds)} seed points -> {len(groups)} objects (eps {eps*100:.1f} cm, {noise} noise)", flush=True)
    owner = np.full(len(P), -1, dtype=np.int32)
    ktrack.OWNER = owner
    cfg0 = dict(DEFAULTS, max_len=1500.0)
    kept, ends, radii = [], [], []
    for gi, g in enumerate(groups):
        remaining = g[owner[g] < 0]
        while len(remaining) >= MIN_LINE_POINTS:
            mini = dense_start(remaining, 2 * cfg0["slab"])
            prof = line_profile(P[mini])
            if prof is None:
                break
            cfg = cfg_for_line(cfg0, prof)
            line, line_ends = trace_one(mini, cfg)
            pieces = cut_same_line(line, prof["radius"], list(zip(kept, radii)))
            took = 0
            for piece in pieces:
                if len(piece) < 2:
                    continue
                hits = tree().query_ball_point(densify(piece, 0.1), prof["radius"])
                mem = np.unique(np.fromiter((i for h in hits for i in h), dtype=np.intp))
                mem = mem[owner[mem] < 0]
                if len(mem) < MIN_LINE_POINTS:
                    continue
                owner[mem] = len(kept)
                kept.append(piece); radii.append(prof["radius"])
                ends.append(line_ends if len(pieces) == 1 else ((None, "end", ""), (None, "end", "")))
                took += len(mem)
            covered = owner[remaining] >= 0
            if took == 0 or not covered.any():
                # nothing traced from this spot: set its neighbourhood aside
                covered = np.linalg.norm(P[remaining] - P[mini].mean(0), axis=1) < 2 * cfg0["slab"]
            remaining = remaining[~covered & (owner[remaining] < 0)]
        print(f"object {gi + 1:3d}: {len(g):6d} seeds -> {len(kept)} lines so far  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(f"{S}/{out}", labels=owner, radii=np.array(radii),
             ends=np.array([[e[0][1], e[1][1]] for e in ends], dtype=object),
             joined_to=np.array([[e[0][2] if e[0][1] == "joined" else -1,
                                  e[1][2] if e[1][1] == "joined" else -1] for e in ends]),
             **{f"line_{i}": l for i, l in enumerate(kept)})
    n_join = sum(1 for e in ends for x in e if x[1] == "joined")
    print(f"{len(kept)} lines, {int((owner >= 0).sum())} points; ends: {n_join} joined another line, "
          f"{2 * len(ends) - n_join} ended  [{time.time()-t0:.0f}s]")
