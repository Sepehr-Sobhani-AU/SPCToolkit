"""
Generic linear-feature region growing.

Grows a 1-D linear feature (cable, pipe, rail, kerb, edge) outward from seed
points using one of three strategies, all built on the shared RANSAC line
engine (``core/services/ransac.fit``):

- ``"axis_trace"`` — fit a line to the seeds, march a search cylinder along its
  axis, refit each step, and stop on a large direction change (e.g. a pole),
  too few points, or empty space. Raw points only; best for isolated thin
  features (cables, pipes, rails).

- ``"linearity_connected"`` — breadth-first neighbour expansion over a KD-tree,
  accepting a neighbour only if its *precomputed* per-point linearity is above a
  threshold. Best for edges/kerbs embedded in a surface, where an axis cylinder
  would leak into the surface. Requires a per-point linearity array (consumed
  from upstream eigenvalues — never computed here).

- ``"hybrid"`` — the axis-trace march with the linearity gate additionally
  applied to candidate points; combines directional ordering with
  surface-leak resistance.

This is an *orchestrator* over RANSAC, per ``DECISIONS.md`` 2026-05-26: it calls
``fit`` and owns the iteration policy. It is not a RANSAC variant.
"""

from collections import deque, namedtuple

import numpy as np
from scipy.spatial import cKDTree

from core.services.ransac import fit
from core.services.geometry_utils import unit, perp_basis, principal_axis
from core.entities.vector_feature import VectorFeature


# Growth modes
AXIS_TRACE = "axis_trace"
LINEARITY_CONNECTED = "linearity_connected"
HYBRID = "hybrid"

_LINEARITY_MODES = (LINEARITY_CONNECTED, HYBRID)

# A cluster is considered consumed by a grown line (and dropped from the pool)
# when at least this fraction of its points ended up in that line. Clusters that
# a line merely crosses (a few shared points at an intersection) stay below this
# and get their own growth pass.
_CONSUME_FRAC = 0.5

# Minimum inlier ratio for the one-off RANSAC line fit that seeds the initial
# march heading (a loose gate — the march re-fits every step afterwards).
_SEED_FIT_MIN_INLIER_RATIO = 0.2

# When bridging a gap, land the tip this fraction of a cylinder_length *before*
# the nearest far point, so the next step's fit window captures the whole cluster.
_GAP_LANDING_FRAC = 0.1

# Centerline vertex de-duplication tolerance: max(length * REL, ABS) metres.
_DEDUPE_REL_TOL = 1e-3
_DEDUPE_ABS_TOL = 1e-6

# One grown, possibly-joined linear feature.
#   indices:       (K,) point indices into all_points belonging to the line.
#   centerline:    (M, 3) float32 ordered polyline down the line, or None (no
#                  centerline for the linearity-connected mode / empty growth).
#   cylinders:     list of (tip, direction, radius, length) search cylinders.
#   end_cylinders: list of (stop_reason, cylinder) for each march direction —
#                  the last cylinder at each end of the line plus WHY growth
#                  stopped there (a STOP_* key). Up to two per line (one per end).
GrownLine = namedtuple(
    "GrownLine", ["indices", "centerline", "cylinders", "end_cylinders"]
)


class LinearRegionGrower:
    """
    Trace a linear feature through a point cloud from seed points.

    Parameters:
        all_points: ``(N, 3)`` array — the full point cloud.
        kdtree: Pre-built ``cKDTree`` for *all_points* (built on demand if None).
        mode: One of ``AXIS_TRACE``, ``LINEARITY_CONNECTED``, ``HYBRID``.
        ransac_threshold: RANSAC line inlier distance threshold (m).
        max_iterations: Max RANSAC hypotheses tried per line fit.
        cylinder_radius: Axis-trace search cylinder radius (m).
        cylinder_length: Axis-trace search cylinder length per step (m).
        overlap: Fraction (0–0.9) each step's cylinder overlaps the previous; the
            tip advances by (1 - overlap) of a cylinder length per step.
        reach_factor: Search reach as a multiple of *cylinder_length* (>= 1). The
            march looks this far ahead for the next points, so a short fit window
            still bridges gaps in fragmented features. 1.0 = no bridging.
        min_points: Stop the axis march if fewer points fall in the cylinder.
        max_angle_deg: Max direction change per step before stopping (m).
        max_steps: Safety cap on axis-march steps per direction.
        linearity: ``(N,)`` per-point linearity, required for the linearity
            modes. Consumed from upstream eigenvalues; never computed here.
        linearity_threshold: Minimum linearity to accept a point.
        neighbor_radius: Radius for linearity-connected neighbour queries (m).
            If None, a k-NN query with *neighbor_k* is used instead.
        neighbor_k: k for k-NN neighbour queries when *neighbor_radius* is None.
    """

    def __init__(
        self,
        all_points: np.ndarray,
        kdtree: cKDTree = None,
        mode: str = AXIS_TRACE,
        ransac_threshold: float = 0.03,
        max_iterations: int = 100,
        cylinder_radius: float = 0.03,
        cylinder_length: float = 0.5,
        overlap: float = 0.0,
        reach_factor: float = 3.0,
        min_points: int = 5,
        max_angle_deg: float = 20.0,
        max_steps: int = 500,
        linearity: np.ndarray = None,
        linearity_threshold: float = 0.4,
        neighbor_radius: float = None,
        neighbor_k: int = 16,
    ):
        self.all_points = np.asarray(all_points)
        self.kdtree = kdtree if kdtree is not None else cKDTree(self.all_points)
        self.mode = mode

        self.ransac_threshold = ransac_threshold
        self.max_iterations = max_iterations
        self.cylinder_radius = cylinder_radius
        self.cylinder_length = cylinder_length
        self.overlap = overlap
        # Search reach as a multiple of the fit window. reach >= cylinder_length,
        # so a short window (curve fidelity) can still look far enough ahead to
        # bridge gaps in fragmented features. 1.0 disables bridging.
        self.reach_factor = max(1.0, reach_factor)
        self.min_points = min_points
        self.max_angle_cos = np.cos(np.radians(max_angle_deg))
        self.max_steps = max_steps

        # Debug geometry recorded during axis-trace marching, for visualization.
        # Accumulates across grow() calls: (tip, direction, radius, length) per
        # search cylinder, and (p0, p1) per centerline segment. debug_end_cylinders
        # holds only the last cylinder of each march direction — the stop point at
        # each end of the line.
        self.debug_cylinders = []
        self.debug_lines = []
        self.debug_end_cylinders = []

        # Optional cooperative-cancel handle (any object with .is_set()), set for
        # the duration of a grow_lines() call so the march can bail out promptly
        # on a long trace. None disables the check. The engine stays GUI-free —
        # the caller owns the event (e.g. global_variables.global_cancel_event).
        self._cancel_event = None

        self.linearity = None if linearity is None else np.asarray(linearity)
        self.linearity_threshold = linearity_threshold
        self.neighbor_radius = neighbor_radius
        self.neighbor_k = neighbor_k

        if self.mode in _LINEARITY_MODES and self.linearity is None:
            raise ValueError(
                f"growth mode '{self.mode}' requires a per-point linearity "
                "array; none was provided (compute eigenvalues upstream)."
            )
        if self.linearity is not None and len(self.linearity) != len(self.all_points):
            raise ValueError(
                "linearity array length "
                f"({len(self.linearity)}) does not match number of points "
                f"({len(self.all_points)})."
            )

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def grow(self, seed_indices: np.ndarray) -> np.ndarray:
        """
        Grow a single linear feature from *seed_indices*.

        Returns an array of point indices belonging to the feature (indices
        into ``self.all_points``), always including the seeds.
        """
        seed_indices = np.asarray(seed_indices, dtype=np.intp)
        if seed_indices.size == 0:
            return seed_indices
        if self.mode == LINEARITY_CONNECTED:
            return self._grow_linearity_connected(seed_indices)
        # AXIS_TRACE and HYBRID both march along the axis; HYBRID adds the gate.
        return self._grow_axis_trace(
            seed_indices, use_linearity_gate=(self.mode == HYBRID)
        )

    def grow_lines(self, seed_groups, *, progress_cb=None, cancel_event=None) -> list:
        """
        Grow one line per physical feature, greedily, largest cluster first.

        *seed_groups* is a list of index arrays (into ``all_points``), one per
        cluster of the picked seeds (typically DBSCAN of the user's selection).
        A single physical line is often split by clustering into several
        clusters; this method recovers it without any merge heuristic:

            1. Grow a line from the LARGEST remaining cluster.
            2. Drop every remaining cluster that growth consumed (most of its
               points ended up in the grown line) — those clusters were just
               pieces of this same line.
            3. Repeat until no clusters remain.

        Because a cluster is removed only when the growth actually reached it,
        parallel neighbouring lines are never fused (growth never crosses to
        them) and a line split into many clusters comes out whole (the march
        walks through all of them).

        *progress_cb*, if given, is called ``progress_cb(done, total, message)``
        after each line — ``done`` lines finished, ``total`` the initial cluster
        count (an upper bound; consumption shrinks it), ``message`` a short
        status string. *cancel_event*, if given, is any object with
        ``.is_set()``; it is polled before each line and inside the march, and
        when set the method stops early and returns the lines grown so far
        (partial result). Both default off, so existing callers are unaffected.

        Returns a list of ``GrownLine``, one per physical line.
        """
        pool = [np.asarray(g, dtype=np.intp) for g in seed_groups
                if np.asarray(g).size >= 2]
        lines = []
        total = len(pool)
        self._cancel_event = cancel_event

        try:
            while pool:
                if cancel_event is not None and cancel_event.is_set():
                    break

                # Largest remaining cluster seeds the next line — the most seed
                # points give the steadiest starting axis.
                s = int(np.argmax([c.size for c in pool]))
                seed_cluster = pool.pop(s)

                # Grow in isolation so this line keeps only its own debug geometry.
                self.debug_cylinders = []
                self.debug_lines = []
                self.debug_end_cylinders = []
                grown = self.grow(seed_cluster)
                if grown.size == 0:
                    continue

                centerline = self._join_centerline(list(self.debug_lines))
                lines.append(GrownLine(np.array(sorted(set(int(i) for i in grown)),
                                                dtype=np.intp),
                                       centerline, list(self.debug_cylinders),
                                       list(self.debug_end_cylinders)))

                if progress_cb is not None:
                    progress_cb(len(lines), total,
                                f"Growing lines — {len(lines)} traced, "
                                f"{len(pool)} seed group(s) left")

                # Drop clusters this line consumed. A cluster is part of the line if
                # most of its points were grown into it; clusters merely crossed
                # (a few shared points at an intersection) are kept for their own
                # pass. The seed cluster is always consumed (grow includes seeds),
                # so the loop always shrinks.
                pool = [c for c in pool
                        if float(np.mean(np.isin(c, grown))) < _CONSUME_FRAC]
        finally:
            self._cancel_event = None
        return lines

    def _join_centerline(self, segments):
        """Turn recorded ``(p0, p1)`` centerline segments into one ordered
        polyline (``(M, 3)`` float32), or None if there is nothing to draw.

        Segments from several joined groups (and both march directions) are
        pooled, de-duplicated, then chained nearest-to-nearest from one end so
        the result runs continuously down the whole line, bridging any gap
        between originally-separate groups.
        """
        if not segments:
            return None
        pts = []
        for p0, p1 in segments:
            pts.append(np.asarray(p0, dtype=np.float64))
            pts.append(np.asarray(p1, dtype=np.float64))
        uniq = self._dedupe(np.asarray(pts))
        if len(uniq) < 2:
            return None
        return self._order_polyline(uniq).astype(np.float32)

    def _dedupe(self, pts):
        """Drop near-coincident vertices (consecutive segments share an end)."""
        tol = max(self.cylinder_length * _DEDUPE_REL_TOL, _DEDUPE_ABS_TOL)
        keys = np.round(pts / tol).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(keep)]

    @staticmethod
    def _order_polyline(pts):
        """Order points into a single chain: start at one PCA-axis extreme,
        then repeatedly hop to the nearest unused point (handles curves and
        bridges gaps between joined groups)."""
        axis = LinearRegionGrower._principal_axis(pts)
        proj = (pts - pts.mean(axis=0)) @ axis
        n = len(pts)
        used = np.zeros(n, dtype=bool)
        cur = int(np.argmin(proj))
        used[cur] = True
        order = [cur]
        for _ in range(n - 1):
            d = np.linalg.norm(pts - pts[cur], axis=1)
            d[used] = np.inf
            cur = int(np.argmin(d))
            used[cur] = True
            order.append(cur)
        return pts[order]

    # ------------------------------------------------------------------ #
    # Axis-trace (cylinder march)                                        #
    # ------------------------------------------------------------------ #

    def _fit_line(self, points, min_inlier_ratio):
        return fit(
            points,
            "line",
            self.ransac_threshold,
            max_iterations=self.max_iterations,
            min_inlier_ratio=min_inlier_ratio,
        )

    def _grow_axis_trace(self, seed_indices, use_linearity_gate=False) -> np.ndarray:
        seed_indices = np.asarray(seed_indices, dtype=np.intp)
        seed_pts = self.all_points[seed_indices]
        if len(seed_pts) < 2:
            return seed_indices

        # Order the seeds along the feature by their dominant (PCA) axis. A
        # RANSAC line is the WRONG tool here: seeds spanning a curved feature are
        # not collinear, so the fit fails and nothing grows. The PCA axis always
        # returns and is used ONLY to locate the span centre + anchor the march;
        # the march re-fits locally every step, so the seed body obeys the same
        # cylinder rule as the growth instead of being one straight line fit.
        axis = self._principal_axis(seed_pts)
        projections = (seed_pts - seed_pts.mean(axis=0)) @ axis

        anchor, start_dir = self._seed_anchor_and_direction(seed_pts, projections, axis)

        collected = set(int(i) for i in seed_indices)
        # March outward from the span centre in both directions. Each pass steps
        # a locally-refit cylinder, so it traverses half the seed body, reaches
        # the far end, and continues past it; the two passes share the anchor and
        # tile into one continuous chain of aligned cylinders / centerline.
        collected |= self._march(anchor, start_dir, use_linearity_gate)
        collected |= self._march(anchor, -start_dir, use_linearity_gate)
        return np.array(sorted(collected), dtype=np.intp)

    @staticmethod
    def _principal_axis(pts):
        """Unit dominant axis of *pts* (eigenvector of the largest mean-centred
        covariance eigenvalue). Always defined, even when the points are not
        collinear. Thin delegate to ``geometry_utils.principal_axis`` (kept as a
        staticmethod for callers/tests that reference it here)."""
        return principal_axis(pts)

    def _seed_anchor_and_direction(self, seed_pts, projections, axis):
        """Start point + initial heading for the axis march.

        Anchors at the seed nearest the middle of the span, then takes the
        heading from a line fit to the dense *cloud* within one cylinder length
        of that anchor — a true local tangent that does not depend on how
        sparsely the user picked seeds. Falls back to the seed *axis* if the
        local fit is unavailable. The two marches (start_dir and -start_dir)
        cover both directions, so the sign of the heading does not matter.
        """
        mid = 0.5 * (float(projections.min()) + float(projections.max()))
        anchor = seed_pts[int(np.argmin(np.abs(projections - mid)))].copy()

        direction = axis
        nbr = self.kdtree.query_ball_point(anchor, self.cylinder_length)
        if len(nbr) >= 2:
            local_model, _ = self._fit_line(
                self.all_points[np.asarray(nbr, dtype=np.intp)],
                min_inlier_ratio=_SEED_FIT_MIN_INLIER_RATIO,
            )
            if local_model is not None:
                direction = local_model.direction
        return anchor, unit(direction)

    def _march(self, tip, direction, use_linearity_gate) -> set:
        """Drive the cylinder march in one direction from *tip*.

        Each step: search the reach-tube ahead (``_query_tube``); if the near
        fit window holds enough points, fit the local axis (``_fit_step``),
        collect its members (``_collect_members``), record the output geometry on
        the midpoint chain (``_record_step``) and advance the search tip
        (``_advance_tip``); otherwise try to bridge a gap (``_bridge_gap``). Stops
        on a sharp bend, an empty tube, no band continuation, or the step cap,
        recording why at this end of the line.

        Output geometry (centerline + cylinders) is built on the MIDPOINTS of the
        per-step fitted lines (band centroids), connected in order, so it forms
        one continuous chain with matched vertices. The final segment is extended
        from the last midpoint to the END of the last fitted line, so the line
        reaches the feature's end instead of stopping half a window short.
        """
        collected = set()
        start_n = len(self.debug_cylinders)
        # The fit window is one cylinder_length; the search reach looks further so
        # a short window can still bridge gaps in fragmented features.
        reach = self.cylinder_length * self.reach_factor
        reach_half = reach / 2.0
        reach_radius = np.sqrt(self.cylinder_radius ** 2 + reach_half ** 2)

        current_tip = np.asarray(tip, dtype=float).copy()
        current_dir = np.asarray(direction, dtype=float).copy()

        # Output geometry hangs off the fitted-line midpoints. prev_mid starts at
        # the (on-feature) march anchor, so the first segment ties the line back to
        # it. last_end is the far end of the most recent fitted line (used to
        # extend the final segment); just_bridged suppresses the cylinder over a
        # bridged gap.
        prev_mid = current_tip.copy()
        last_end = None
        just_bridged = False

        # Why this direction stopped. Defaults to the step cap: if the loop runs
        # all max_steps without breaking, that is the reason. Otherwise each break
        # below sets its own reason before stopping.
        reason = STOP_STEP_CAP

        for _ in range(self.max_steps):
            if self._cancel_event is not None and self._cancel_event.is_set():
                break  # cooperative cancel — keep whatever was collected so far
            tube = self._query_tube(
                current_tip, current_dir, reach, reach_half, reach_radius,
                use_linearity_gate,
            )
            if tube is None:  # nothing within reach / nothing in the tube
                reason = STOP_EMPTY_SPACE
                break
            candidate_idx, pts, along, perp_dist, tube_mask = tube

            # The near fit window is the first cylinder_length of the tube.
            near_mask = tube_mask & (along <= self.cylinder_length)

            if np.count_nonzero(near_mask) >= self.min_points:
                band = pts[near_mask]
                fit_dir, fit_point = self._fit_step(band, current_dir)

                # Stop on a genuine (measured) bend of the fitted axis.
                if np.dot(fit_dir, current_dir) < self.max_angle_cos:
                    reason = STOP_SHARP_BEND  # direction turned more than max_angle
                    break

                near_idx = candidate_idx[near_mask]
                collected.update(
                    self._collect_members(band, near_idx, fit_point, fit_dir)
                )
                # Record on the midpoint chain. A segment spanning a just-bridged
                # gap gets no cylinder (nothing was searched inside the gap).
                self._record_step(prev_mid, fit_point,
                                  record_cylinder=not just_bridged)
                prev_mid = fit_point
                just_bridged = False
                # Far end of this fitted line, for extending the final segment.
                last_end = fit_point + float(
                    ((band - fit_point) @ fit_dir).max()) * fit_dir
                current_tip = self._advance_tip(current_tip, fit_point, fit_dir)
                current_dir = fit_dir

            else:
                # Too few points hug the heading in the near window: an empty gap,
                # a sparse/scattered patch, or a genuine turn. Keep any near
                # on-axis points (they belong to the line) and try to bridge to a
                # real continuation further along the SAME heading.
                on_axis_mask = tube_mask & (perp_dist < self.ransac_threshold)
                near_on_axis = near_mask & on_axis_mask
                collected.update(int(i) for i in candidate_idx[near_on_axis])

                next_tip = self._bridge_gap(current_tip, current_dir, along, tube_mask)
                if next_tip is None:
                    reason = STOP_TOO_FEW_POINTS  # no band continuation within reach
                    break
                current_tip = next_tip
                just_bridged = True

        # Extend the final segment from the last midpoint to the end of the last
        # fitted line, so the centerline (and its tube) reach the feature's end
        # rather than stopping at the last midpoint, half a window short.
        if last_end is not None:
            self._record_step(prev_mid, last_end)

        # The last cylinder searched in this direction is the stop point at this
        # end of the line — the march ended just beyond it. Record it together
        # with WHY it stopped (reason), so ends can be split by reason and drawn.
        if len(self.debug_cylinders) > start_n:
            self.debug_end_cylinders.append((reason, self.debug_cylinders[-1]))

        return collected

    def _query_tube(self, tip, direction, reach, reach_half, reach_radius,
                    use_linearity_gate):
        """Search the reach-tube ahead of *tip* and project candidates onto the
        current heading.

        Returns ``(candidate_idx, pts, along, perp_dist, tube_mask)`` for the
        points found ahead, or ``None`` when nothing is within reach or nothing
        falls inside the tube (the caller stops with STOP_EMPTY_SPACE). The tube
        is searched out to the full reach — not just the fit window — so points
        across a gap are visible.
        """
        centre = tip + reach_half * direction
        candidate_idx = self.kdtree.query_ball_point(centre, reach_radius)
        if not candidate_idx:
            return None

        candidate_idx = np.asarray(candidate_idx, dtype=np.intp)
        pts = self.all_points[candidate_idx]

        # Project onto the CURRENT axis. perp_dist is each point's distance to
        # that axis; it gates the tube radius and (tighter) on-feature status.
        vecs = pts - tip
        along = vecs @ direction
        perp = vecs - np.outer(along, direction)
        perp_dist = np.linalg.norm(perp, axis=1)

        tube_mask = (along > 0) & (along <= reach) & (perp_dist < self.cylinder_radius)
        if use_linearity_gate:
            tube_mask &= self.linearity[candidate_idx] >= self.linearity_threshold
        if not np.any(tube_mask):
            return None
        return candidate_idx, pts, along, perp_dist, tube_mask

    def _fit_step(self, band, current_dir):
        """Fit the per-step axis (direction + centre) from the FULL near band.

        Uses PCA of the whole band (within cylinder_radius), NOT a thin
        ransac_threshold stripe. This is the crux of not drifting off the line:
        on a band WIDER than the threshold, gating the fit by the threshold
        selects a diagonal stripe whose PCA follows the stripe's tilt, so the
        axis walks off to one side. PCA of the whole band returns its true long
        axis (the along-extent dwarfs the lateral scatter) and its centroid is
        the band centre, so the tip re-centres each step. (Plain
        variance-weighted PCA already supplies the "fit length" term and
        discounts a compact off-axis stub; density normalization was measured to
        make that worse — see DECISIONS 2026-07-09.) The axis is sign-aligned to
        the current heading. Returns ``(fit_dir, fit_point)``.
        """
        fit_dir = self._principal_axis(band)
        if np.dot(fit_dir, current_dir) < 0:
            fit_dir = -fit_dir
        fit_point = band.mean(axis=0)  # the band centre
        return fit_dir, fit_point

    def _collect_members(self, band, near_idx, fit_point, fit_dir):
        """Near-band points within ransac_threshold of the fitted axis (the
        line members contributed by this step)."""
        rel = band - fit_point
        perp_m = np.linalg.norm(rel - np.outer(rel @ fit_dir, fit_dir), axis=1)
        return [int(i) for i in near_idx[perp_m < self.ransac_threshold]]

    def _record_step(self, prev_mid, mid, record_cylinder=True):
        """Record output geometry on the MIDPOINT chain: a centerline segment from
        the previous midpoint to *mid*, and (unless this segment spans a bridged
        gap) a search cylinder over it.

        Both hang off the fitted-line midpoints (band centroids), which sit on the
        points and are shared vertex-for-vertex by consecutive steps — so the
        centerline is one continuous chain and the cylinders abut end-to-end,
        instead of each starting off to one side on its own freshly-fitted line.
        """
        self.debug_lines.append((prev_mid.copy(), mid.copy()))
        if not record_cylinder:
            return
        seg = mid - prev_mid
        length = float(np.linalg.norm(seg))
        if length > 1e-9:
            self.debug_cylinders.append(
                (prev_mid.copy(), seg / length, self.cylinder_radius, length)
            )

    def _advance_tip(self, current_tip, fit_point, fit_dir):
        """Advance the SEARCH tip one step along the fitted axis, re-centred onto
        it (the foot of the tip on the axis) so the next window sits on the
        points. Drives the search only; output geometry is recorded separately on
        the midpoint chain by ``_record_step``."""
        base = fit_point + float((current_tip - fit_point) @ fit_dir) * fit_dir
        step = self.cylinder_length * (1.0 - self.overlap)
        return base + step * fit_dir

    def _bridge_gap(self, current_tip, current_dir, along, tube_mask):
        """Jump the tip toward the nearest far-band continuation within the tube.

        Requires at least min_points in the far reach zone WITHIN THE TUBE
        (cylinder_radius of the current heading). The tube gate keeps the bridge
        on the SAME line longitudinally — a parallel neighbour beyond the radius
        is excluded — and a real bend (points turned off the heading) finds none
        within the tube. Returns the next tip (landed just inside the fit window
        of the nearest far point, keeping the heading since there is no data in
        the gap to re-fit), or ``None`` when there is no continuation (the caller
        stops with STOP_TOO_FEW_POINTS).
        """
        far_band = tube_mask & (along > self.cylinder_length)
        if np.count_nonzero(far_band) < self.min_points:
            return None
        a_next = float(along[far_band].min())
        jump = a_next - _GAP_LANDING_FRAC * self.cylinder_length
        next_tip = current_tip + jump * current_dir
        return next_tip

    # ------------------------------------------------------------------ #
    # Linearity-connected (neighbour BFS gated by linearity)             #
    # ------------------------------------------------------------------ #

    def _grow_linearity_connected(self, seed_indices) -> np.ndarray:
        n = len(self.all_points)
        visited = np.zeros(n, dtype=bool)
        in_region = np.zeros(n, dtype=bool)
        thr = self.linearity_threshold
        use_radius = self.neighbor_radius is not None and self.neighbor_radius > 0

        queue = deque()
        for i in seed_indices:
            i = int(i)
            visited[i] = True
            in_region[i] = True
            queue.append(i)

        while queue:
            i = queue.popleft()
            if use_radius:
                nbrs = self.kdtree.query_ball_point(
                    self.all_points[i], self.neighbor_radius
                )
            else:
                _, nbrs = self.kdtree.query(self.all_points[i], k=self.neighbor_k + 1)
                nbrs = np.atleast_1d(nbrs)

            for j in nbrs:
                j = int(j)
                if j >= n or visited[j]:  # cKDTree returns n for missing neighbours
                    continue
                visited[j] = True
                if self.linearity[j] >= thr:
                    in_region[j] = True
                    queue.append(j)

        return np.where(in_region)[0].astype(np.intp)


# --------------------------------------------------------------------------- #
# Debug geometry -> renderable branches                                        #
#                                                                              #
# The grower records raw search cylinders and centerline segments during the   #
# axis-trace march. These helpers convert that raw geometry into render-only    #
# VectorFeature objects so a plugin can add them as ordinary, fully            #
# controllable tree branches (wireframe, drawn through the standard set_lines   #
# path) rather than as an ad-hoc viewer overlay. They are pure: no GUI access.  #
# --------------------------------------------------------------------------- #

_CYLINDER_COLOR = np.array([0.1, 0.7, 1.0], dtype=np.float32)      # light blue
_CENTERLINE_COLOR = np.array([1.0, 0.9, 0.1], dtype=np.float32)    # yellow

# Why an axis march stopped at an end. One key per break condition in _march.
# Each maps to (human label, RGB colour) so ends can be split into one coloured
# branch per reason and named in the summary.
STOP_TOO_FEW_POINTS = "too_few_points"
STOP_SHARP_BEND = "sharp_bend"
STOP_EMPTY_SPACE = "empty_space"
STOP_STEP_CAP = "step_cap"

STOP_REASONS = {
    STOP_TOO_FEW_POINTS: ("too few points", np.array([1.0, 0.1, 0.1], dtype=np.float32)),   # red
    STOP_SHARP_BEND:     ("sharp bend",      np.array([1.0, 0.5, 0.0], dtype=np.float32)),   # orange
    STOP_EMPTY_SPACE:    ("empty space",     np.array([1.0, 0.0, 1.0], dtype=np.float32)),   # magenta
    STOP_STEP_CAP:       ("step cap",        np.array([1.0, 1.0, 1.0], dtype=np.float32)),   # white
}


def cylinders_to_vector_feature(cylinders, color=None, n_segments=12,
                                symbol_type="Search Cylinders"):
    """Wireframe VectorFeature: two end rings + a few longitudinals per cylinder."""
    if not cylinders:
        return None
    verts = []
    edges = []
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    long_step = max(1, n_segments // 4)

    for tip, direction, radius, length in cylinders:
        tip = np.asarray(tip, dtype=np.float64)
        d = unit(direction)
        u, v = perp_basis(d)
        ring = radius * np.array([np.cos(a) * u + np.sin(a) * v for a in angles])

        base0 = len(verts)
        verts.extend(tip + ring)
        top0 = len(verts)
        verts.extend(tip + length * d + ring)

        for i in range(n_segments):
            j = (i + 1) % n_segments
            edges.append([base0 + i, base0 + j])   # base ring
            edges.append([top0 + i, top0 + j])     # top ring
        for i in range(0, n_segments, long_step):
            edges.append([base0 + i, top0 + i])    # longitudinal

    return _wireframe_vector_feature(
        symbol_type, verts, edges,
        _CYLINDER_COLOR if color is None else color,
    )


def centerlines_to_vector_feature(centerlines, color=None):
    """One VectorFeature holding several joined centerlines.

    *centerlines* is a list of ordered ``(M, 3)`` polylines (one per grown
    line). Each becomes its own connected edge chain in a single mesh — so the
    separate lines draw connected within themselves, with no spurious edge
    bridging one line to the next. Returns None if nothing is drawable.
    """
    verts = []
    edges = []
    for cl in centerlines:
        if cl is None or len(cl) < 2:
            continue
        base = len(verts)
        verts.extend(np.asarray(cl, dtype=np.float64))
        for i in range(len(cl) - 1):
            edges.append([base + i, base + i + 1])
    if not edges:
        return None
    return _wireframe_vector_feature(
        "Centerlines", verts, edges,
        _CENTERLINE_COLOR if color is None else color,
    )


def segments_to_vector_feature(segments, color=None):
    """Wireframe VectorFeature from a list of (p0, p1) centerline segments."""
    if not segments:
        return None
    verts = []
    edges = []
    for p0, p1 in segments:
        i = len(verts)
        verts.append(np.asarray(p0, dtype=np.float64))
        verts.append(np.asarray(p1, dtype=np.float64))
        edges.append([i, i + 1])

    return _wireframe_vector_feature(
        "Centerlines", verts, edges,
        _CENTERLINE_COLOR if color is None else color,
    )


def _wireframe_vector_feature(symbol_type, verts, edges, color):
    vertices = np.asarray(verts, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int32)
    dims = (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float32)
    geometry = {"vertices": vertices, "faces": [], "edges": edges}
    return VectorFeature(
        symbol_type=symbol_type,
        geometry_type="mesh",
        geometry=geometry,
        transform_matrix=np.eye(4),
        dimensions=dims,
        color=np.asarray(color, dtype=np.float32),
    )


def debug_vector_features(grower, show_cylinders, show_lines):
    """Return ``[(branch_name, VectorFeature), ...]`` for the requested overlays.

    Pure: builds render-only geometry from the grower's recorded debug data; the
    caller adds each as an ordinary, controllable tree branch.
    """
    out = []
    if show_cylinders:
        vf = cylinders_to_vector_feature(grower.debug_cylinders)
        if vf is not None:
            out.append(("search_cylinders", vf))
    if show_lines:
        vf = segments_to_vector_feature(grower.debug_lines)
        if vf is not None:
            out.append(("centerlines", vf))
    return out
