"""
Outline clean-up — the shared half of every cluster-outline method.

An outline method (``core.services.boundary_extraction``) produces a *raw*
closed ring in 2D: the cluster flattened onto a plane, traced at whatever
detail the method gives — a grid staircase, a ring of edge points. This
service does the work that is the same for all of them and turns that ring
into a clean, closed **3D polyline** with straight segments only:

- ``Prepared``        — flatten the cluster onto its plane, measure each
                        point's local spacing, drop stray points.
- ``clean_ring``      — smooth the raw ring, give every vertex of the dense
                        ring its height, weed in 3D, straighten each run with
                        a line fit, rebuild corners where the lines meet.
- ``weed_polyline``   — the separate "weed vertices" tool: drop vertices of
                        an existing 3D polyline that a straight line already
                        explains, within a plan and a height tolerance.
- ``vertex_heights``  — height of a vertex from its nearest cluster points.
- ``build_outline``   — points -> clean 3D polyline, one call.

Decisions behind it (synthetic tests in ``experiments/cluster_outline``):

- **Weeding is 3D and comes after heights.** A vertex that only matters for
  its height — a gable peak on a straight plan edge — is kept. Weeding in 2D
  first collapsed a pitched roof to its four eave corners.
- **The outline follows the cluster's outermost points**; it is not pushed
  out to an estimated true surface edge.
- **Projection is the cluster's best-fit plane** by default, so a wall is
  outlined on the wall. Height is the offset off that plane — for a cluster
  lying roughly flat, the points' Z.
- **Heights come from a small plane fitted through the nearest points and
  read at the vertex**, not from their median. A vertex sits on the edge, so
  all its neighbours lie on the inner side: on a slope their median is biased
  (3 cm RMS on a 30 deg roof at 400 pts/m2; the plane fit gives 1 cm).

Pure geometry, no GUI. Every neighbour search runs on the GPU
(``core.services.grid_knn``).
"""

import numpy as np

from core.services.grid_knn import GridKNN, free_gpu_memory

# Neighbours that measure a point's local spacing (distance to the k-th).
SPACING_K = 6

# Stray test: a point is a stray when its distance to its STRAY_K-th
# neighbour is more than STRAY_RATIO times the median of that distance over
# its STRAY_REF_K nearest points. The wide reference matters: strays sit near
# other strays, and judged only against its closest few a stray is vouched
# for by them. Measured on a cluster with 0.5% strays up to 0.5 m out: 98% of
# those more than 10 cm out caught, about 1 real point in 2000 dropped.
STRAY_K = 3
STRAY_REF_K = 32
STRAY_RATIO = 2.5

# A flagged point is kept when the unflagged points around it surround it —
# no empty gap wider than this (deg) between the directions to its nearest
# STRAY_GAP_K of them (after thinning, see _rescue_surrounded). A stray outside
# an edge has them all on one side (the cluster's); a sparse point where
# density drops sharply looks isolated against the dense side, but has sparse
# points beyond it too. Flagged points do not count, so strays near other
# strays cannot vouch for each other.
STRAY_MAX_GAP_DEG = 180.0
STRAY_GAP_K = 12

# Points whose neighbours are gathered at once while preparing a cluster;
# bounds the (block x 32) neighbour scratch to ~400 MB on the host.
PREPARE_BLOCK = 1_000_000

# Nearest points a vertex takes its height from.
HEIGHT_K = 16

# A turn sharper than this (deg) makes the raw ring double back: a spike.
SPIKE_TURN_DEG = 150.0

# Bends gentler than this (deg) are not corners: never rebuilt as one, and
# the sides either side may be merged into one line.
MIN_CORNER_DEG = 20.0


# --------------------------------------------------------------------------- #
# Neighbour search                                                             #
# --------------------------------------------------------------------------- #

def knn(query, k, reference=None, index=None):
    """(distances, indices) of the k nearest ``reference`` points for every 2D
    query point (default reference: the queries themselves, self included).
    Pass a prebuilt ``GridKNN`` as ``index`` to reuse it."""
    built = index is None
    if built:
        index = GridKNN(query if reference is None else reference)
    try:
        d, i = index.query(query, k)
    finally:
        if built:
            index.release()
    return d.astype(np.float64), i


# --------------------------------------------------------------------------- #
# Plane frame and the prepared cluster                                         #
# --------------------------------------------------------------------------- #

_FIXED_AXES = {
    "xy": np.eye(3),
    "xz": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
    "yz": np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]]),
}


class PlaneFrame:
    """The 2D frame a cluster is flattened into.

    ``best_fit`` is the cluster's own plane (PCA); height is the offset along
    its normal. ``xy`` / ``xz`` / ``yz`` are fixed views (height = the third
    axis). All three axes form a right-handed frame, so a CCW ring stays CCW.
    """

    def __init__(self, points, mode="best_fit"):
        pts = np.asarray(points, dtype=np.float64)[:, :3]
        self.origin = pts.mean(axis=0)
        if mode == "best_fit":
            _, _, vt = np.linalg.svd(pts - self.origin, full_matrices=False)
            normal = vt[2] if vt[2][2] >= 0 else -vt[2]
            self.axes = np.vstack([vt[0], np.cross(normal, vt[0]), normal])
        elif mode in _FIXED_AXES:
            self.axes = _FIXED_AXES[mode]
        else:
            raise ValueError(f"Unknown projection {mode!r}")

    def to_2d(self, points):
        """(N,3) -> (N,2) plane coordinates and (N,) heights."""
        local = (np.asarray(points, dtype=np.float64)[:, :3] - self.origin) @ self.axes.T
        return local[:, :2], local[:, 2]

    def to_3d(self, uv, h):
        """Plane coordinates + heights -> (N,3) float32 world points."""
        local = np.column_stack([uv, h])
        return (local @ self.axes + self.origin).astype(np.float32)


def _rescue_surrounded(uv, keep, ds):
    """Flagged points (``~keep``) whose unflagged surroundings enclose them.

    The test runs against the unflagged points thinned to one per cell of
    half the flagged points' typical scale: next to a sharp density drop a
    point's nearest raw neighbours all lie on the dense side, thinned they
    spread round it."""
    flagged = np.flatnonzero(~keep)
    good = np.flatnonzero(keep)
    if len(good) < 3:
        return np.zeros(len(uv), dtype=bool)
    cell = 0.5 * float(np.median(ds[flagged]))
    ij = np.floor((uv[good] - uv[good].min(axis=0)) / max(cell, 1e-9)).astype(np.int64)
    _, first = np.unique(ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1], return_index=True)
    thin = uv[good[first]]
    _, nn = knn(uv[flagged], min(STRAY_GAP_K, len(thin)), reference=thin)
    out = np.zeros(len(uv), dtype=bool)
    out[flagged[_surrounded(uv[flagged], thin[nn])]] = True
    return out


def _surrounded(points, neighbours, max_gap_deg=STRAY_MAX_GAP_DEG):
    """For each point (M,2) and its neighbours (M,k,2): True when the
    neighbours' directions leave no empty gap wider than ``max_gap_deg``."""
    rel = neighbours - points[:, None, :]
    ang = np.sort(np.arctan2(rel[..., 1], rel[..., 0]), axis=1)
    gaps = np.diff(np.concatenate([ang, ang[:, :1] + 2 * np.pi], axis=1), axis=1)
    return np.degrees(gaps.max(axis=1)) <= max_gap_deg


class Prepared:
    """A cluster ready for an outline method.

    Attributes:
        frame: the ``PlaneFrame``.
        uv, h: plane coordinates and heights of the kept points.
        local: each kept point's local spacing (distance to its 6th neighbour).
        local_ref: the same judged by its neighbours (median of theirs) — an
            isolated or edge point does not make its area look sparse.
        spacing: typical nearest-neighbour distance.
        sparse_scale: a scale that still bridges the sparse parts of an
            uneven cluster (95th percentile of ``local``).
        strays: how many stray points were dropped.
        index: GPU ``GridKNN`` over ``uv``, reused by every later query —
            call ``release()`` when done.
    """

    def __init__(self, points, projection="best_fit", drop_strays=True):
        if len(points) < SPACING_K + 2:
            raise ValueError(f"An outline needs at least {SPACING_K + 2} points.")
        self.frame = PlaneFrame(points, projection)
        uv, h = self.frame.to_2d(points)
        n = len(uv)
        k = min(STRAY_REF_K, n)
        d1 = np.empty(n)
        dk = np.empty(n)
        ref = np.empty(n)
        keep = np.ones(n, dtype=bool)
        full = GridKNN(uv)
        try:
            # Two passes over blocks: the stray test needs every point's
            # STRAY_K-th distance before any point's reference can be read.
            ds = np.empty(n)
            for a in range(0, n, PREPARE_BLOCK):
                d, _ = knn(uv[a:a + PREPARE_BLOCK], k, index=full)
                d1[a:a + len(d)] = d[:, 1]
                dk[a:a + len(d)] = d[:, min(SPACING_K, k - 1)]
                ds[a:a + len(d)] = d[:, min(STRAY_K, k - 1)]
            for a in range(0, n, PREPARE_BLOCK):
                _, nn = knn(uv[a:a + PREPARE_BLOCK], k, index=full)
                ref[a:a + len(nn)] = np.median(dk[nn[:, 1:SPACING_K + 1]], axis=1)
                if drop_strays:
                    wide = np.median(ds[nn[:, 1:]], axis=1)
                    keep[a:a + len(nn)] = ds[a:a + len(nn)] <= STRAY_RATIO * np.maximum(wide, 1e-12)
            if drop_strays and not keep.all():
                keep |= _rescue_surrounded(uv, keep, ds)
        finally:
            full.release()
        self.uv, self.h, self.local = uv[keep], h[keep], dk[keep]
        self.local_ref = ref[keep]
        self.strays = int((~keep).sum())
        nn1 = d1[keep]
        nn1 = nn1[nn1 > 0]
        self.spacing = float(np.median(nn1)) if len(nn1) else 0.0
        self.sparse_scale = float(np.percentile(self.local, 95))
        self.index = GridKNN(self.uv)

    def release(self):
        if self.index is not None:
            self.index.release()
            self.index = None


# --------------------------------------------------------------------------- #
# Ring helpers                                                                 #
# --------------------------------------------------------------------------- #

def ring_area(ring):
    """Signed area of a closed 2D ring (positive = counter-clockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def make_ccw(ring):
    return ring if ring_area(ring) >= 0 else ring[::-1].copy()


def _dedupe(ring, eps=1e-9):
    keep = np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1) > eps
    return ring[keep] if keep.sum() >= 3 else ring


def _despike(ring, max_turn_deg):
    """Remove vertices where the ring nearly doubles back on itself (a spike
    out or a notch in). Repeats until none are left."""
    cos_lim = np.cos(np.radians(max_turn_deg))
    ring = ring.copy()
    while len(ring) > 3:
        a = ring - np.roll(ring, 1, axis=0)
        b = np.roll(ring, -1, axis=0) - ring
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        ok = (na > 0) & (nb > 0)
        cos_t = np.ones(len(ring))
        cos_t[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
        spikes = cos_t < cos_lim
        if not spikes.any():
            break
        # Drop every other spike per pass so neighbours are re-judged.
        drop = spikes & ~np.roll(spikes, 1)
        if not drop.any():
            drop = spikes
        ring = ring[~drop]
    return ring


def smooth_ring(ring, length, scale_at=None):
    """Resample the ring evenly and smooth it with a Gaussian of ``length``
    along the ring — or, with ``scale_at``, of max(length, local scale), so
    sparse stretches are smoothed more. Removes grid staircases and point-to-
    point jitter; corners get rounded and are rebuilt afterwards."""
    from scipy.ndimage import gaussian_filter1d

    closed = np.vstack([ring, ring[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(16, int(s[-1] / (length / 4.0)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    xy = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])
    ds = s[-1] / n
    lengths = np.full(n, float(length))
    if scale_at is not None:
        lengths = np.maximum(lengths, scale_at(xy))
    # Smooth once per length bucket (x1.25 apart) and pick per sample.
    buckets = np.round(np.log(lengths / length) / np.log(1.25)).astype(int)
    out = np.empty_like(xy)
    for b in np.unique(buckets):
        sigma = length * 1.25 ** b / ds
        sel = buckets == b
        out[sel, 0] = gaussian_filter1d(xy[:, 0], sigma, mode="wrap")[sel]
        out[sel, 1] = gaussian_filter1d(xy[:, 1], sigma, mode="wrap")[sel]
    return out


def _fit_line(pts):
    """Total-least-squares line: (point on line, unit direction)."""
    c = pts.mean(axis=0)
    if len(pts) < 2:
        return c, np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _project(p, line):
    c, d = line
    return c + np.dot(p - c, d) * d


def _span(a, b, n):
    """Ring indices a..b inclusive, wrapping."""
    return np.arange(a, b + 1) if b >= a else np.r_[np.arange(a, n), np.arange(0, b + 1)]


def _refine_corners(raw, idx, min_corner_deg, trim=0.0):
    """Straighten each weeded segment with a line fitted to all the raw ring
    vertices it replaced, then put each corner where neighbouring lines meet.
    Gentle bends (lines nearly parallel) keep the weeded vertex, moved onto
    the two lines' average."""
    n, m = len(raw), len(idx)
    lines = []
    for j in range(m):
        seg = raw[_span(idx[j], idx[(j + 1) % m], n)]
        if trim > 0 and len(seg) > 6:
            # Leave out the rounded ends next to corners.
            ends = np.minimum(np.linalg.norm(seg - seg[0], axis=1),
                              np.linalg.norm(seg - seg[-1], axis=1))
            inner = seg[ends > trim]
            if len(inner) >= 3:
                seg = inner
        lines.append(_fit_line(seg))
    sin_lim = np.sin(np.radians(min_corner_deg))
    out = np.empty((m, 2))
    for j in range(m):
        prev, cur = lines[j - 1], lines[j]
        v = raw[idx[j]]
        fallback = 0.5 * (_project(v, prev) + _project(v, cur))
        d1, d2 = prev[1], cur[1]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < sin_lim:
            out[j] = fallback
            continue
        diff = cur[0] - prev[0]
        x = prev[0] + ((diff[0] * d2[1] - diff[1] * d2[0]) / cross) * d1
        # Neighbouring weeded vertices bound how far a corner may move.
        reach = 0.5 * min(np.linalg.norm(v - raw[idx[j - 1]]),
                          np.linalg.norm(v - raw[idx[(j + 1) % m]]))
        out[j] = x if np.linalg.norm(x - v) <= reach else fallback
    return out


def _straight_run(ring, start, step, need, tol):
    """Walk from vertex ``start`` in direction ``step`` (+1/-1) until the path
    is ``need`` long. Returns the far point if every vertex passed lies within
    ``tol`` of the straight line start->far, else None."""
    m = len(ring)
    p = ring[start]
    total, k = 0.0, start
    passed = []
    for _ in range(m - 1):
        nk = (k + step) % m
        total += np.linalg.norm(ring[nk] - ring[k])
        k = nk
        if total >= need:
            break
        passed.append(ring[k])
    else:
        return None
    far = ring[k]
    if passed:
        d = far - p
        L = np.linalg.norm(d)
        if L == 0:
            return None
        rel = np.asarray(passed) - p
        if (np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L).max() > tol:
            return None
    return far


def _self_intersects(ring):
    """True if any two non-adjacent edges cross (O(n^2), fine for outlines)."""
    n = len(ring)
    if n < 4:
        return False
    a = ring
    b = np.roll(ring, -1, axis=0)
    for i in range(n):
        p, r = a[i], b[i] - a[i]
        q = a[i + 2:] if i > 0 else a[2:n - 1]
        s = (b[i + 2:] if i > 0 else b[2:n - 1]) - q
        if len(q) == 0:
            continue
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
            u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / rxs
        if ((rxs != 0) & (t > 0) & (t < 1) & (u > 0) & (u < 1)).any():
            return True
    return False


def _cut_chamfers(ring, short_len, min_corner_deg, straight_tol, min_side, max_chain=6):
    """Smoothing rounds every corner and weeding keeps the rounding as a few
    short edges. Replace such a short chain between two long sides that meet
    at a real corner with the point where those sides meet.

    A cut corner sits between two long straight sides — 2x the chain and at
    least ``min_side``, straight within ``straight_tol`` — so short edges along
    a real curve fail the test and stay. Every start and chain length is
    tried, shortest chains first; a cut that would fold the ring is skipped.
    ``short_len``: one length per vertex (a chain starting there must be
    shorter than it)."""
    ring = np.asarray(ring, dtype=np.float64).copy()
    short_len = np.asarray(short_len, dtype=np.float64).copy()
    sin_lim = np.sin(np.radians(min_corner_deg))
    changed = True
    while changed and len(ring) > 3:
        changed = False
        m = len(ring)
        L = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        cands = []
        for j in range(m):
            total = 0.0
            for r in range(1, min(max_chain, m - 3) + 1):
                total += L[(j + r - 1) % m]
                if total >= short_len[j]:
                    break
                cands.append((total, j, r))
        for chain_len, j, r in sorted(cands):
            a, b = ring[j], ring[(j + r) % m]
            need = max(2.0 * chain_len, min_side)
            p0 = _straight_run(ring, j, -1, need, straight_tol)
            q1 = _straight_run(ring, (j + r) % m, +1, need, straight_tol)
            if p0 is None or q1 is None:
                continue
            d1, d2 = a - p0, q1 - b
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 == 0 or n2 == 0:
                continue
            d1, d2 = d1 / n1, d2 / n2
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < sin_lim:
                continue
            diff = b - p0
            x = p0 + ((diff[0] * d2[1] - diff[1] * d2[0]) / cross) * d1
            if np.linalg.norm(x - 0.5 * (a + b)) > 2.0 * chain_len:
                continue
            drop = [(j + i) % m for i in range(1, r + 1)]
            trial = ring.copy()
            trial[j] = x
            trial = np.delete(trial, drop, axis=0)
            if _self_intersects(trial):
                continue
            ring = trial
            short_len = np.delete(short_len, drop)
            changed = True
            break
    return ring


# --------------------------------------------------------------------------- #
# 3D weeding                                                                   #
# --------------------------------------------------------------------------- #

def _dp3_open(xy, h, plan_tol, h_tol):
    """Douglas-Peucker on an open chain in plan + height: a vertex stays if
    dropping it would move the line by more than ``plan_tol`` in plan OR
    ``h_tol`` in height. ``plan_tol`` may be one value per vertex. Returns the
    kept indices."""
    n = len(xy)
    plan_tol = np.broadcast_to(np.asarray(plan_tol, dtype=np.float64), (n,))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = xy[j] - xy[i]
        L2 = float(seg @ seg)
        rel = xy[i + 1:j] - xy[i]
        t = np.clip(rel @ seg / L2, 0, 1) if L2 > 0 else np.zeros(len(rel))
        dplan = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        dh = np.abs(h[i + 1:j] - (h[i] + t * (h[j] - h[i])))
        score = np.maximum(dplan / np.maximum(plan_tol[i + 1:j], 1e-12),
                           dh / max(h_tol, 1e-12))
        k = int(np.argmax(score))
        if score[k] > 1.0:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return np.flatnonzero(keep)


def dp3_closed(xy, h, plan_tol, h_tol):
    """Closed-ring version of ``_dp3_open``. Anchors are the two vertices
    farthest apart in plan, so the result does not depend on where the ring
    happens to start. Returns kept indices in ring order."""
    n = len(xy)
    if n <= 4:
        return np.arange(n)
    plan_tol = np.broadcast_to(np.asarray(plan_tol, dtype=np.float64), (n,))
    c = xy.mean(axis=0)
    a = int(np.argmax(np.linalg.norm(xy - c, axis=1)))
    b = (int(np.argmax(np.linalg.norm(xy - xy[a], axis=1))) - a) % n
    rx, rh, rt = np.roll(xy, -a, axis=0), np.roll(h, -a), np.roll(plan_tol, -a)
    first = _dp3_open(rx[:b + 1], rh[:b + 1], rt[:b + 1], h_tol)
    second = _dp3_open(np.vstack([rx[b:], rx[:1]]), np.r_[rh[b:], rh[:1]],
                       np.r_[rt[b:], rt[:1]], h_tol) + b
    idx = np.unique(np.concatenate([first, second[:-1]]))
    return np.sort((idx + a) % n)


def _merge_straight(ring, h, idx, plan_tol, h_tol, min_corner_deg, window):
    """Robust weeding on the dense ring. ``idx`` are the kept vertices (ring
    indices). Drop a kept vertex when ONE line through all the dense ring
    points of its two sides fits them within tolerance — plan offset, and
    height offset from a straight height profile — after averaging the
    SIGNED offsets over ``window`` ring samples. Noise flips side and
    averages out; a curve's bulge stays on one side and does not, so kinks
    from noise go while curves keep their vertices (Douglas-Peucker alone
    judges by the single worst wiggle). Gentle turns only (below
    ``min_corner_deg``), so corners are never merged."""
    n = len(ring)
    idx = list(idx)
    plan_tol = np.broadcast_to(np.asarray(plan_tol, dtype=np.float64), (n,))
    cos_lim = np.cos(np.radians(min_corner_deg))
    w = max(1, int(window))
    while len(idx) > 3:
        best = None
        m = len(idx)
        for j in range(m):
            a, v, b = idx[j - 1], idx[j], idx[(j + 1) % m]
            d1, d2 = ring[v] - ring[a], ring[b] - ring[v]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 == 0 or n2 == 0 or (d1 @ d2) / (n1 * n2) < cos_lim:
                continue
            sp = _span(a, b, n)
            c, d = _fit_line(ring[sp])
            rel = ring[sp] - c
            off = rel[:, 0] * d[1] - rel[:, 1] * d[0]
            t = rel @ d
            A = np.column_stack([t, np.ones(len(t))])
            coef, *_ = np.linalg.lstsq(A, h[sp], rcond=None)
            dh = h[sp] - A @ coef
            kern = np.ones(min(w, len(sp))) / min(w, len(sp))
            off_s = np.convolve(off, kern, mode="valid")
            dh_s = np.convolve(dh, kern, mode="valid")
            tol_s = np.convolve(plan_tol[sp], kern, mode="valid")
            score = max(np.abs(off_s / tol_s).max(), np.abs(dh_s).max() / h_tol)
            if score <= 1.0 and (best is None or score < best[0]):
                best = (score, j)
        if best is None:
            break
        del idx[best[1]]
    return np.array(idx)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def vertex_heights(ring_uv, cloud_uv, cloud_h, k=HEIGHT_K, index=None):
    """Height of every vertex from its k nearest cluster points (in plane).

    A small plane is fitted through the neighbours and read AT the vertex. A
    vertex sits on the cluster's edge, so all its neighbours lie on the inner
    side; on a slope their plain median would be biased (uphill at an eave),
    the plane carries the slope out to the edge. The plane is trusted only
    next to its points — neighbours not all on a line, and the vertex no
    farther from the nearest one than half their spread (a convex-hull vertex
    can hang over empty space); otherwise the median is used."""
    ring_uv = np.asarray(ring_uv, dtype=np.float64)
    dist, nn = knn(ring_uv, k, reference=cloud_uv, index=index)
    hs = cloud_h[nn]
    out = np.median(hs, axis=1)
    rel = cloud_uv[nn] - ring_uv[:, None, :]            # vertex at the origin
    A = np.concatenate([rel, np.ones(rel.shape[:2] + (1,))], axis=2)
    AtA = np.einsum("mki,mkj->mij", A, A)
    Atb = np.einsum("mki,mk->mi", A, hs)
    ok = (np.linalg.cond(AtA) < 1e8) & (dist[:, 0] <= 0.5 * dist[:, -1])
    if ok.any():
        out[ok] = np.linalg.solve(AtA[ok], Atb[ok][..., None])[:, 2, 0]
    return out


def clean_ring(ring, tolerance, height_at, height_tol=None, smooth=None,
               scale_at=None, min_corner_deg=MIN_CORNER_DEG):
    """Clean a raw 2D ring into an outline with few, well-placed vertices and
    give every vertex its height. Returns (ring (M,2) CCW, heights (M,)).

    Args:
        ring: raw closed ring (N,2) in plane coordinates.
        tolerance: how far the outline may stray from the raw ring in plan.
        height_at: fn(xy) -> height at each position (from the cluster).
        height_tol: the same in height (default ``tolerance``).
        smooth: smoothing length along the ring (default ``tolerance``);
            at least the raw ring's own step (e.g. the grid cell).
        scale_at: optional fn(xy) -> local point gap; where points are sparse,
            smoothing and plan tolerance grow to it (wiggles smaller than the
            gap between points are noise, not shape).
    """
    height_tol = tolerance if height_tol is None else height_tol
    ring = make_ccw(_dedupe(np.asarray(ring, dtype=np.float64)))
    if len(ring) >= 4:
        ring = _despike(ring, SPIKE_TURN_DEG)
    smooth = tolerance if smooth is None else smooth
    if smooth > 0 and len(ring) >= 4:
        ring = make_ccw(smooth_ring(ring, smooth, scale_at))
    if len(ring) < 4:
        return ring, height_at(ring)

    def plan_tol(xy):
        return tolerance if scale_at is None else np.maximum(tolerance, 0.5 * scale_at(xy))

    # 1. Heights on the dense ring, then 3D weeding.
    h = height_at(ring)
    idx = dp3_closed(ring, h, plan_tol(ring), height_tol)
    if len(idx) < 3:
        return ring[idx], h[idx]
    # 2. Merge sides one line explains (robust to wiggles), then straighten
    #    each run and rebuild corners.
    step_len = np.median(np.linalg.norm(np.diff(ring, axis=0), axis=1))
    idx = _merge_straight(ring, h, idx, plan_tol(ring), height_tol, min_corner_deg,
                          window=smooth / max(step_len, 1e-12))
    out = _refine_corners(ring, idx, min_corner_deg, trim=2.0 * smooth)
    if _self_intersects(out):
        out = ring[idx]                 # refinement folded the ring: undo it
    # Merge nearly-collinear pieces so each side reaches step 3 whole.
    keep = dp3_closed(out, height_at(out), plan_tol(out), height_tol)
    out = out[keep]
    # 3. Corners rounded by smoothing. Gaussian rounding reaches ~3 smoothing
    #    lengths each side, so a chain shorter than 6 between two real sides
    #    is a cut corner. Both sides must be straight for 8 smoothing lengths
    #    (within 2x tolerance: surviving vertices sit just over 1x by
    #    definition), so a real curve is never mistaken for a rounded corner.
    scale = smooth if scale_at is None else np.maximum(smooth, scale_at(out))
    out = _cut_chamfers(out, np.broadcast_to(6.0 * scale, (len(out),)),
                        max(min_corner_deg, 30.0), straight_tol=2.0 * tolerance,
                        min_side=8.0 * smooth)
    # 4. Heights at the final positions, and a last 3D weed.
    h = height_at(out)
    keep = dp3_closed(out, h, plan_tol(out), height_tol)
    out, h = out[keep], h[keep]
    if ring_area(out) < 0:
        out, h = out[::-1].copy(), h[::-1].copy()
    return out, h


def weed_polyline(vertices, plan_tol, height_tol, closed=True,
                  projection="best_fit", return_indices=False):
    """The weeding tool: drop vertices from an existing 3D polyline that a
    straight line already explains, within ``plan_tol`` across the line and
    ``height_tol`` in height. Vertices are never moved.

    ``projection`` sets what "plan" and "height" mean. ``best_fit`` uses the
    polyline's own plane — right for an outline (a wall outline is weeded
    across the wall), but for a nearly straight open line (a cable, a traced
    edge) that plane is poorly defined and can stand vertical; use ``xy``
    there, so height is Z. Returns the kept vertices (float32), or their
    row indices with ``return_indices``."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) <= (3 if closed else 2):
        idx = np.arange(len(v))
    else:
        uv, h = PlaneFrame(v, projection).to_2d(v)
        if closed:
            idx = dp3_closed(uv, h, plan_tol, height_tol)
        else:
            idx = _dp3_open(uv, h, plan_tol, height_tol)
    return idx if return_indices else v[idx].astype(np.float32)


def build_outline(points, raw_ring_fn, tolerance, height_tol=None,
                  projection="best_fit", drop_strays=True):
    """A cluster's clean, closed 3D outline.

    Args:
        points: (N,3) cluster points.
        raw_ring_fn: fn(prep: Prepared) -> (raw ring (M,2) or None, step) —
            the outline method; ``step`` is its own step size (the clean-up
            smooths at least twice that).
        tolerance: plan tolerance (m); ``height_tol`` defaults to it.
        projection: ``best_fit`` (default), ``xy``, ``xz`` or ``yz``.
        drop_strays: ignore isolated stray points.

    Returns:
        (vertices (M,3) float32, info dict with ``raw_vertices``, ``strays``
        and ``spacing``), or (None, info) when the method found no ring.
    """
    prep = Prepared(points, projection, drop_strays=drop_strays)
    try:
        raw, step = raw_ring_fn(prep)
        info = {"raw_vertices": 0 if raw is None else len(raw),
                "strays": prep.strays, "spacing": prep.spacing}
        if raw is None or len(raw) < 3:
            return None, info

        def scale_at(xy):
            _, nn = knn(xy, 10, index=prep.index)
            return np.median(prep.local[nn], axis=1)

        def height_at(xy):
            return vertex_heights(xy, prep.uv, prep.h, index=prep.index)

        # A raw ring is ragged by about one of its own steps, correlated over
        # a few steps: smooth over two.
        ring, h = clean_ring(raw, tolerance, height_at, height_tol=height_tol,
                             smooth=max(tolerance, 2.0 * step), scale_at=scale_at)
        return prep.frame.to_3d(ring, h), info
    finally:
        prep.release()
        free_gpu_memory()
