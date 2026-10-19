"""Find and trace every wire in the test cloud with the Kalman tracker (scratch).

1. Seeds: 1 m voxels whose points form a thin, roughly level line in open
   air (PCA: long, <3 cm across, slope under ~27 deg, above the ground).
2. Trace from each unclaimed seed, both ways, with ktrack.track.
3. Keep lines long enough to be wires; points within 5 cm of a kept line
   are claimed by it (first line wins), and seeds they cover are dropped.
Saves data/cables.npz: one centerline per line + per-point labels.
"""
from scipy.ndimage import minimum_filter
import ktrack
from ktrack import *

VOXEL = 1.0
MIN_PTS = 6
MAX_LATERAL_SD = 0.03
MIN_ALONG_SD = 0.2
MAX_SLOPE = 0.45            # |tz| of the voxel's axis
MIN_HEIGHT = 0.5            # above the estimated ground (m)
MIN_LINE_LEN = 15.0         # shorter traces are not kept (branches, edges)
MEMBER_RADIUS = 0.05


def ground_height(P, cell=2.0, window=5):
    """Rough ground: lowest point per 2 m cell, then the lowest over 10 m."""
    xy = np.floor((P[:, :2] - P[:, :2].min(0)) / cell).astype(np.int64)
    shape = xy.max(0) + 1
    g = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(g, (xy[:, 0], xy[:, 1]), P[:, 2])
    g = minimum_filter(g, size=window, mode="nearest")
    return P[:, 2] - g[xy[:, 0], xy[:, 1]]


def seed_voxels(P, hag):
    keys = np.floor(P / VOXEL).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    X = P.astype(np.float64) - (keys * VOXEL)        # local coords keep precision
    s = np.stack([np.bincount(inv, X[:, i], len(uniq)) for i in range(3)], 1)
    ss = np.empty((len(uniq), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            ss[:, i, j] = ss[:, j, i] = np.bincount(inv, X[:, i] * X[:, j], len(uniq))
    n = cnt[:, None]
    ok = cnt >= MIN_PTS
    mean = s[ok] / n[ok]
    cov = ss[ok] / n[ok][:, :, None] - mean[:, :, None] * mean[:, None, :]
    w, v = np.linalg.eigh(cov)
    axis = v[:, :, 2]
    good = ((np.sqrt(np.maximum(w[:, 2], 0)) >= MIN_ALONG_SD)
            & (np.sqrt(np.maximum(w[:, 1], 0)) <= MAX_LATERAL_SD)
            & (np.abs(axis[:, 2]) <= MAX_SLOPE))
    vox_ids = np.flatnonzero(ok)[good]
    # height above ground of each seed voxel: median of its points
    hv = np.zeros(len(uniq)); np.add.at(hv, inv, hag); hv /= cnt
    vox_ids = vox_ids[hv[vox_ids] >= MIN_HEIGHT]
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)]
    return [order[starts[i]:starts[i + 1]] for i in vox_ids]


def densify(path, step=0.1):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(L / step)))
        out.extend(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.array(out)


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


def members_of(path):
    dense = densify(path)
    hits = tree().query_ball_point(dense, MEMBER_RADIUS)
    return np.unique(np.fromiter((i for h in hits for i in h), dtype=np.intp))


def trace_both_ways(seed_idx, cfg):
    """One line through the seed: back half reversed + forward half."""
    fwd, ev_f = track(seed_idx, +1, cfg)
    back, ev_b = track(seed_idx, -1, cfg)
    # both start at an end of the seed; join them through the seed
    line = np.vstack([back[::-1], fwd])
    return line, ev_f + ev_b


if __name__ == "__main__":
    t0 = time.time()
    hag = ground_height(P)
    seeds = seed_voxels(P, hag)
    print(f"{len(seeds)} seed voxels  [{time.time()-t0:.0f}s]", flush=True)
    cfg = dict(DEFAULTS)
    cfg["max_len"] = 1500.0
    label = np.full(len(P), -2, dtype=np.int32)
    lines, rejected = [], 0
    # densest seeds first: the clearest wire stretches
    seeds.sort(key=len, reverse=True)
    for k, sd in enumerate(seeds):
        if np.mean(label[sd] >= 0) > 0.3:
            continue                      # already on a traced line
        line, ev = trace_both_ways(sd, cfg)
        mem = members_of(line)
        mem = mem[label[mem] < 0]         # first line wins
        L = path_length(line)
        if L < MIN_LINE_LEN or len(mem) < 50:
            rejected += 1
            label[sd] = -3                # don't retry this seed's points
            continue
        label[mem] = len(lines)
        lines.append(line)
        print(f"line {len(lines):3d}: {L:6.0f} m, {len(mem):6d} pts, seed {k}/{len(seeds)}  [{time.time()-t0:.0f}s]", flush=True)
    label[label == -3] = -2
    np.savez(f"{S}/cables.npz", labels=label, **{f"line_{i}": l for i, l in enumerate(lines)})
    print(f"{len(lines)} lines kept, {rejected} short traces dropped, "
          f"{np.sum(label >= 0)} points claimed  [{time.time()-t0:.0f}s]")
