"""Experimental Kalman-driven line tracker (scratch). Not project code."""
import json
from harness import *
from core.services.path_tracker import PathTracker, CURVED, STRAIGHT

_TREE = None


def tree():
    global _TREE
    if _TREE is None:
        t0 = time.time()
        _TREE = cKDTree(P, leafsize=32, balanced_tree=False, compact_nodes=False)
        print(f"[kdtree {time.time()-t0:.0f}s]", flush=True)
    return _TREE


DEFAULTS = dict(
    model=CURVED, point_sigma=0.015, curvature_noise=0.02, direction_noise=0.05,
    step=0.5,            # metres per predict/update
    slab=1.0,            # length of the measurement slab centred on the prediction
    gate_min=0.04, gate_max=0.25, gate_sigmas=3.0,
    tight=0.05,          # refine: keep points this close to the median offset
    min_pts=3,
    max_spread=0.06,     # lateral sd above this = clutter, not a wire
    max_gap=20.0,        # coast at most this far without a measurement
    reacq_len=3.0, reacq_max_deg=30.0, reacq_radius=0.05, reacq_slices=10,
    reacq_min_span=6,
    max_len=1000.0,
    gate_max_coast=0.8, coast_after=3.0, ring_factor=4.0, ring_inner=1.5, clutter_ratio=0.5,
    restart_dir_sigma=0.03, restart_k_sigma=0.02, max_curvature=0.02,
    strict_min_pts=6, confirm_steps=2,
    bridge_win=4.0, bridge_max_deg=15.0, bridge_rays=400, bridge_slices=8,
    bridge_min_span=6, bridge_every=1.0, bridge_back=1.0, kink_step=0.25, strict_clutter_ratio=0.1,
)


def restart(kt, pos, t, cfg):
    """Begin (again) at *pos* heading *t*, with a measured heading and a
    wire-like bend prior instead of PathTracker's loose defaults."""
    kt.start(pos, t)
    n = kt.n
    sig = np.full(n, cfg["point_sigma"])
    sig[3:6] = cfg["restart_dir_sigma"]
    if n == 9:
        sig[6:9] = cfg["restart_k_sigma"]
    kt._state.P = np.diag(sig ** 2)
    kt._history[-1] = kt._history[-1]._replace(prior=kt._state.copy(), filtered=kt._state.copy())


def clamp(kt, cfg):
    if kt.n == 9:
        k = kt._state.x[6:9]
        m = np.linalg.norm(k)
        if m > cfg["max_curvature"]:
            kt._state.x[6:9] = k * cfg["max_curvature"] / m


def _lateral(rel, t):
    a = rel @ t
    return a, rel - np.outer(a, t)


def measure(state, cfg, coasting=False, strict=False):
    """Lateral correction of the prediction from points in a slab around it.

    Picks the tight cluster of points that is both dense and close to the
    prediction (not the median: two wires can share a wide gate). Refuses
    when the cluster is not a clean wire: points crowd round it (clutter).
    """
    p, t, k = state.position, state.tangent, state.curvature
    sig = state.lateral_sigma()
    gmax = cfg["gate_max_coast"] if coasting else cfg["gate_max"]
    g = float(np.clip(cfg["gate_sigmas"] * np.hypot(sig, cfg["point_sigma"]),
                      cfg["gate_min"], gmax))
    half = cfg["slab"] / 2
    ring = cfg["tight"] * cfg["ring_factor"]
    idx = tree().query_ball_point(p, np.hypot(half, g + ring))
    if len(idx) < cfg["min_pts"]:
        return None, g, "none"
    q = P[idx].astype(float)
    a, lat = _lateral(q - p, t)
    lat = lat - 0.5 * np.outer(a ** 2, k)          # follow the predicted bend
    in_slab = np.abs(a) < half
    lat = lat[in_slab]
    r = np.linalg.norm(lat, axis=1)
    cand = lat[r < g]
    if len(cand) < cfg["min_pts"]:
        return None, g, "none"
    sub = cand if len(cand) <= 1500 else cand[np.random.default_rng(0).choice(len(cand), 1500, replace=False)]
    d2 = np.sum((sub[:, None] - sub[None]) ** 2, axis=2)
    dens = (d2 < cfg["tight"] ** 2).sum(1)
    s = max(sig, cfg["tight"])
    best = sub[np.argmax(dens * np.exp(-0.5 * np.sum(sub ** 2, 1) / s ** 2))]
    # Re-centre on the wire (a few mean-shift steps): the densest sample may
    # sit on the wire's edge, and the clutter ring must be drawn round its axis.
    centre = best
    for _ in range(3):
        core = lat[np.linalg.norm(lat - centre, axis=1) < cfg["tight"]]
        if len(core) == 0:
            break
        centre = core.mean(0)
    dist = np.linalg.norm(lat - centre, axis=1)
    core = lat[dist < cfg["tight"]]
    if len(core) < (cfg["strict_min_pts"] if strict else cfg["min_pts"]):
        return None, g, "sparse"
    # Clutter = points crowding round the wire, beyond its own spread.
    n_ring = int(np.sum((dist >= cfg["ring_inner"] * cfg["tight"]) & (dist < ring)))
    if n_ring > (cfg["strict_clutter_ratio"] if strict else cfg["clutter_ratio"]) * len(core):
        return None, g, "clutter"
    centre = core.mean(0)
    spread = float(np.sqrt(np.mean(np.sum((core - centre) ** 2, axis=1))))
    if spread > cfg["max_spread"]:
        return None, g, "clutter"
    return p + centre, g, "ok"


def reacquire(origin, t, cfg):
    """Where the line goes from *origin* after a bend: the same clean-wire
    ray search as the bridge, anchored at the bend, over a short window and a
    wider cone. Returns (direction, span) or None."""
    r = bridge_search(origin, t, 0.2, cfg, W=cfg["reacq_len"], max_deg=cfg["reacq_max_deg"],
                      n_sl=cfg["reacq_slices"], min_span=cfg["reacq_min_span"], a_floor=0.2)
    return None if r is None else (r[1], r[2])


def bridge_search(anchor, t, a_from, cfg, W=None, max_deg=None, n_sl=None, min_span=None, a_floor=0.5):
    """Find where the line comes out of a gap or clutter.

    Rays from the last good point *anchor*, within bridge_max_deg of its
    heading *t*, through points in a window [a_from, a_from + bridge_win]
    along it. A ray scores by the CLEAN slices of the window it covers: band
    points present (within reacq_radius) and no crowd round them. A ray that
    crosses a neighbouring wire at an angle covers a slice or two; the wire
    itself, collinear with its own exit, covers the window. Returns
    (point, direction, span) on the found wire, or None.
    """
    W = cfg["bridge_win"] if W is None else W
    r = cfg["reacq_radius"]
    a_from = max(a_from, a_floor)
    tan = np.tan(np.radians(cfg["bridge_max_deg"] if max_deg is None else max_deg))
    mid = anchor + t * (a_from + W / 2)
    idx = tree().query_ball_point(mid, np.hypot(W / 2, (a_from + W) * tan + r))
    if len(idx) < cfg["min_pts"]:
        return None
    rel = P[idx].astype(float) - anchor
    a = rel @ t
    lat = np.linalg.norm(rel - np.outer(a, t), axis=1)
    keep = (a >= a_from) & (a <= a_from + W) & (lat <= a * tan + r)
    rel = rel[keep]
    if len(rel) < cfg["min_pts"]:
        return None
    rng = np.random.default_rng(0)
    pick = rel if len(rel) <= cfg["bridge_rays"] else rel[rng.choice(len(rel), cfg["bridge_rays"], replace=False)]
    dirs = np.vstack([t, pick / np.linalg.norm(pick, axis=1)[:, None]])
    n_sl = cfg["bridge_slices"] if n_sl is None else n_sl
    best = None
    sq = np.einsum("ij,ij->i", rel, rel)
    for lo in range(0, len(dirs), 64):
        d = dirs[lo:lo + 64]
        proj = d @ rel.T
        dist = np.sqrt(np.maximum(sq[None] - proj ** 2, 0))
        al = proj - a_from * 1.0
        sl = np.clip((al / (W / n_sl)).astype(int), 0, n_sl - 1)
        inb = (dist < r) & (al >= 0) & (al <= W)
        ring = (dist >= r * 1.5) & (dist < r * 4) & (al >= 0) & (al <= W)
        for i in range(len(d)):
            nb = np.bincount(sl[i][inb[i]], minlength=n_sl)
            nr = np.bincount(sl[i][ring[i]], minlength=n_sl)
            clean = (nb >= 1) & (nr <= cfg["clutter_ratio"] * np.maximum(nb, 1))
            span = int(clean.sum())
            key = (span, float(d[i] @ t))
            if best is None or key > best[0]:
                best = (key, d[i], inb[i])
    (span, _), ray, band = best
    if span < (cfg["bridge_min_span"] if min_span is None else min_span):
        return None
    pts = rel[band]
    c = pts.mean(0)
    _, v = np.linalg.eigh((pts - c).T @ (pts - c))
    ax = v[:, -1] if v[:, -1] @ ray > 0 else -v[:, -1]
    # where the found wire starts in the window
    start = c + ax * float(((pts - c) @ ax).min())
    return anchor + start, ax, span


def confirmed(kt, z, cfg):
    """After clutter, a reading counts only if the next confirm_steps readings
    agree with it - a branch in a tree gives one good-looking slab, a wire
    coming out of the tree gives a run of them. Tried on a copy."""
    import copy
    trial = copy.deepcopy(kt)
    trial.update(z)
    clamp(trial, cfg)
    for _ in range(cfg["confirm_steps"]):
        trial.predict(cfg["step"])
        clamp(trial, cfg)
        z2, _g, _why = measure(trial.state, cfg, strict=True)
        if z2 is None:
            return False
        trial.update(z2)
        clamp(trial, cfg)
    return True


def track(seed_idx, direction_sign, cfg, log=None):
    seed = P[seed_idx].astype(float)
    c = seed - seed.mean(0)
    _, v = np.linalg.eigh(c.T @ c)
    ax = v[:, -1] * direction_sign
    a = c @ ax
    start = seed[np.argmin(a)]
    kt = PathTracker(model=cfg["model"], point_sigma=cfg["point_sigma"],
                     curvature_noise=cfg["curvature_noise"], direction_noise=cfg["direction_noise"])
    restart(kt, start, ax, cfg)
    path, gaps, events = [start], 0.0, []
    last_good = kt.state
    travelled = 0.0
    retried = False          # one restart per last good point, no loops
    clutter_seen = False
    for _ in range(int(3 * cfg["max_len"] / cfg["step"])):
        if travelled >= cfg["max_len"]:
            break
        kt.predict(cfg["step"])
        clamp(kt, cfg)
        travelled += cfg["step"]
        z, g, why = measure(kt.state, cfg, coasting=gaps >= cfg["coast_after"], strict=clutter_seen)
        if z is not None and clutter_seen and not confirmed(kt, z, cfg):
            z, why = None, "unconfirmed"
        if z is not None:
            kt.update(z)
            clamp(kt, cfg)
            last_good = kt.state
            gaps, retried, clutter_seen = 0.0, False, False
            path.append(kt.state.position)
            continue
        gaps += cfg["step"]
        clutter_seen |= why == "clutter"
        # Lost the line in open space (not clutter): maybe a kink. Re-find it
        # from the last good point, once.
        if gaps >= cfg["step"] * 2 and not retried and why != "clutter":
            retried = True
            # The bend (a pole) lies somewhere between the last good reading
            # and where readings stopped: try the search from points slid
            # forward along the prediction, keep the best.
            positions, tangents, _s = kt.predict_path(gaps, cfg["kink_step"], state=last_good)
            found, anchor = None, None
            for pos, tan in zip(positions, tangents):
                r = reacquire(pos, tan, cfg)
                if r is not None and (found is None or r[1] > found[1]):
                    found, anchor = r, pos
            if found is not None:
                new_t, span = found
                turn = np.degrees(np.arccos(np.clip(new_t @ last_good.tangent, -1, 1)))
                if turn > 2.0:
                    events.append((round(travelled, 1), why, f"turn {turn:.1f} deg span {span}"))
                    if np.linalg.norm(anchor - last_good.position) > 1e-6:
                        path.append(anchor)
                    restart(kt, anchor, new_t, cfg)
                    travelled -= gaps
                    gaps = 0.0
                    continue
        if gaps < cfg["coast_after"]:
            next_bridge = cfg["coast_after"]
        if gaps >= next_bridge and why != "clutter":
            next_bridge = gaps + cfg["bridge_every"]
            found = bridge_search(last_good.position, last_good.tangent,
                                  gaps - cfg["bridge_back"], cfg)
            if found is not None:
                pos, new_t, span = found
                events.append((round(travelled, 1), "bridge", f"span {span}, gap {np.linalg.norm(pos - last_good.position):.1f} m"))
                path.append(pos)
                restart(kt, pos, new_t, cfg)
                last_good = kt.state
                gaps, retried, clutter_seen = 0.0, False, False
                continue
        if gaps > cfg["max_gap"]:
            events.append((round(travelled, 1), "stop", why))
            break
    return np.array(path), events


def sign_for(t):
    e = P[SEEDS[f"{t}_End"]].mean(0)
    s = P[SEEDS[f"{t}_Start"]].astype(float)
    c = s - s.mean(0); _, v = np.linalg.eigh(c.T @ c)
    return 1 if v[:, -1] @ (e - s.mean(0)) > 0 else -1


if __name__ == "__main__":
    cfg = dict(DEFAULTS)
    if len(sys.argv) > 1:
        cfg.update(json.loads(sys.argv[1]))
    tests = sys.argv[2].split(",") if len(sys.argv) > 2 else TESTS
    for t in tests:
        e = P[SEEDS[f"{t}_End"]].mean(0)
        s = P[SEEDS[f"{t}_Start"]].astype(float)
        c = s - s.mean(0); _, v = np.linalg.eigh(c.T @ c)
        sign = 1 if v[:, -1] @ (e - s.mean(0)) > 0 else -1   # only to pick which march to report first
        t0 = time.time()
        path, events = track(SEEDS[f"{t}_Start"], sign, cfg)
        # score: members = points within 3 cm of path vertices
        mem = np.unique(np.concatenate([np.array(x, dtype=np.intp) for x in tree().query_ball_point(path, 0.04)]))
        np.save(f"{S}/path_{t}.npy", path)
        print(t, f"{time.time()-t0:.0f}s", score(t, mem), flush=True)
        for ev in events[-12:]:
            print("   ", ev)
        print("    events:", len(events))
