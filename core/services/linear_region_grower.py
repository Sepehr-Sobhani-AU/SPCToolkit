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
# when at least this fraction of its points fall in the region that line SWEPT.
# Clusters that a line merely crosses (a few shared points at an intersection)
# stay below this and get their own growth pass.
#
# The test is against the swept region, NOT the line's member points: membership
# is gated at ransac_threshold while the march searches out to cylinder_radius,
# so a march can drive straight through a cluster and claim none of its points
# (any point further off-axis than the threshold). Testing membership there
# leaves the cluster in the pool, and it regrows the same feature as a duplicate
# line lying on top of the first one.
_CONSUME_FRAC = 0.5

# A freshly grown line is discarded as a duplicate when at least this fraction of
# its points lie in the region already swept by the lines kept before it. This is
# the backstop for the consume test above: that test runs BEFORE a cluster grows,
# so it cannot catch a cluster that legitimately survived (e.g. it sits past the
# point where the first line stopped) and then retraces the feature backwards —
# each march runs in both directions from its anchor.
_DUPLICATE_FRAC = 0.5

# Minimum inlier ratio for the one-off RANSAC line fit that seeds the initial
# march heading (a loose gate — the march re-fits every step afterwards).
_SEED_FIT_MIN_INLIER_RATIO = 0.2

# When bridging a gap, land the tip this fraction of a cylinder_length *before*
# the nearest far point, so the next step's fit window captures the whole cluster.
_GAP_LANDING_FRAC = 0.1

# How far past the last recorded vertex a claimed point must lie before the
# centerline is carried out to it (metres). Just above the de-duplication
# tolerance, so this never adds a vertex that would be collapsed anyway.
_TAIL_MIN_GAIN = 1e-3

# Centerline vertex de-duplication tolerance: max(length * REL, ABS) metres.
_DEDUPE_REL_TOL = 1e-3
_DEDUPE_ABS_TOL = 1e-6

# The corridor ahead of a stop: this many times further than the march's own
# search reach, and this many times wider than its tube. Deliberately more
# generous than the march — it already failed inside its own tube, so anything
# it could have used is by definition NOT there; what the user picks lives just
# outside, a few metres further on or a little off-axis.
#
# It answers "is there anything worth extending into?" for ranking AND defines
# what the extension window offers for picking, so widening it costs on both
# sides: the offer grows into whatever is around (canopy, walls) far faster than
# it grows along the feature. At the defaults it is ~9 m long and 0.6 m wide.
_RANK_REACH_FACTOR = 3.0
_RANK_RADIUS_FACTOR = 3.0

# Safety margin on the reach an extension is granted to arrive at the user's
# picks: the landing lands slightly short of the target, so the reach must
# clear the pick distance rather than exactly meet it.
_REACH_MARGIN = 1.2

# One grown, possibly-joined linear feature.
#   indices:       (K,) point indices into all_points belonging to the line.
#   centerline:    (M, 3) float32 ordered polyline down the line, or None (no
#                  centerline for the linearity-connected mode / empty growth).
#   cylinders:     list of (tip, direction, radius, length) search cylinders.
#   end_cylinders: list of (stop_reason, cylinder) for each march direction —
#                  the last cylinder at each end of the line plus WHY growth
#                  stopped there (a STOP_* key). Up to two per line (one per end).
#                  Display geometry; ``stops`` is the machine-readable version.
#   stops:         list of MarchStop — where and why each end gave up, the handle
#                  the guided-extension workflow steps through.
GrownLine = namedtuple(
    "GrownLine", ["indices", "centerline", "cylinders", "end_cylinders", "stops"],
    defaults=((),),
)

# Where and why one march direction gave up.
#   tip:       the search tip when it stopped — the point the march was looking
#              ahead from, i.e. the end of what it managed to trace.
#   direction: the heading it was travelling on (unit).
#   reason:    a STOP_* key.
#
# Distinct from ``end_cylinders``, which holds midpoint-CHAIN geometry for
# drawing and is recorded only when at least one step succeeded. A march that
# dies on its first step produces no cylinder but still produces a MarchStop —
# and those ends are exactly the ones worth re-seeding.
MarchStop = namedtuple("MarchStop", ["tip", "direction", "reason"])

# The outcome of continuing a trace past one of its stops.
#   line:    the extended GrownLine (picks always included — see
#            extend_from_stop; the user picking a point IS the statement that it
#            belongs to the line).
#   stop:    the line's NEW frontier at this end — where the caller should look
#            next, whether growth marched on or only the picks were adopted.
#   marched: True when the march itself carried the trace past the picks; False
#            when nothing but the picks landed. The caller needs the difference
#            to tell the user "the trace advanced" apart from "your points were
#            added, now pick further".
Extension = namedtuple("Extension", ["line", "stop", "marched"])


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

        # Point indices the last grow() SWEPT — every point inside a step's fit
        # window, whether or not it became a member of the line. Membership is
        # gated at ransac_threshold, the search at the wider cylinder_radius, so
        # the swept region (not the members) is what "growth reached here" means.
        # Held as a list of per-step index arrays; read via swept_indices().
        self._swept_chunks = []

        # Where each march direction of the last grow() gave up: one MarchStop
        # per direction. Read via march_stops(). These are what the guided
        # extension workflow steps through — see march_stops().
        self._march_stops = []

        # Optional cooperative-cancel handle (any object with .is_set()), set for
        # the duration of a grow_lines() call so the march can bail out promptly
        # on a long trace. None disables the check. The engine stays GUI-free —
        # the caller owns the event (e.g. global_variables.global_cancel_event).
        self._cancel_event = None

        # Point indices the march may bridge a gap to even when too few of them
        # to satisfy min_points — the points the user picked during a guided
        # extension. Set for the duration of extend_from_stop(); None otherwise,
        # which leaves ordinary growth exactly as strict as it has always been.
        self._bridge_targets = None

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

    def grow(self, seed_indices: np.ndarray, only_direction=None) -> np.ndarray:
        """
        Grow a single linear feature from *seed_indices*.

        Returns an array of point indices belonging to the feature (indices
        into ``self.all_points``), always including the seeds.

        *only_direction*, when given, restricts the axis march to the single
        heading pointing that way instead of both. Used when continuing an
        existing trace, where the opposite march would only re-walk line that is
        already traced: it costs a second full march and leaves redundant
        vertices in the joined centerline (measured: 37 vertices vs 26 for the
        same feature). The result is equivalent either way — the union and the
        centerline join both absorb the retrace — so this is about cost, not
        correctness.
        """
        seed_indices = np.asarray(seed_indices, dtype=np.intp)
        self._swept_chunks = []
        self._march_stops = []
        if seed_indices.size == 0:
            return seed_indices
        if self.mode == LINEARITY_CONNECTED:
            region = self._grow_linearity_connected(seed_indices)
            # No march, so no wider search region: the region IS what was swept.
            self._swept_chunks = [region]
            return region
        # AXIS_TRACE and HYBRID both march along the axis; HYBRID adds the gate.
        return self._grow_axis_trace(
            seed_indices, use_linearity_gate=(self.mode == HYBRID),
            only_direction=only_direction,
        )

    def swept_indices(self) -> np.ndarray:
        """Unique point indices the last ``grow()`` swept — everything that fell
        inside a step's fit window, members and non-members alike.

        Wider than the returned line (membership is gated at ransac_threshold,
        the search at cylinder_radius), and the right set for asking "did this
        growth already reach that cluster?" — see ``_CONSUME_FRAC``.
        """
        if not self._swept_chunks:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate(self._swept_chunks)).astype(np.intp)

    def march_stops(self) -> list:
        """The ``MarchStop`` for each direction of the last ``grow()`` — where
        and why the trace gave up.

        This is the handle for continuing a short trace: re-seed with the points
        around a stop plus points the user picks beyond it, and call ``grow()``
        again. Empty for the linearity-connected mode (it does not march).
        """
        return list(self._march_stops)

    # ------------------------------------------------------------------ #
    # Guided extension — continue a short trace from one of its stops     #
    #                                                                    #
    # A trace that gave up early is not re-grown from scratch: the user   #
    # picks a few points beyond the stop, those are unioned with the      #
    # line's own points around the stop, and the SAME grow() runs on the  #
    # combined seed body. The existing points carry the established       #
    # heading; the picks say where the feature goes. Growth is never      #
    # loosened to reach further — the picks are the evidence that the     #
    # feature continues. See LINEAR_REGION_GROWING.md.                    #
    # ------------------------------------------------------------------ #

    def unclaimed_ahead(self, stop, claimed_mask) -> np.ndarray:
        """Indices of points lying ahead of *stop* that no line has claimed.

        Two jobs, one answer: it ranks the stops (how much is there worth going
        after?) and it is exactly the set the extension window offers the user
        for picking. Keeping them the same set is the point — the number the
        window reports and the points it lights up are then the same thing, and
        a wider offer only ever adds points nobody counted.

        Searched wider and further than the march itself (see
        ``_RANK_REACH_FACTOR`` / ``_RANK_RADIUS_FACTOR``): the march already
        established there was nothing usable inside its own tube, so anything
        worth extending into is by definition outside it. At the defaults that
        is a corridor about 9 m long and 0.6 m wide.

        Deliberately not routed through ``_query_tube`` — that gates on
        ``cylinder_radius``, the very limit this needs to exceed.
        """
        reach = self.cylinder_length * self.reach_factor * _RANK_REACH_FACTOR
        radius = self.cylinder_radius * _RANK_RADIUS_FACTOR
        half = reach / 2.0

        tip = np.asarray(stop.tip, dtype=float)
        direction = unit(np.asarray(stop.direction, dtype=float))
        candidate_idx = self.kdtree.query_ball_point(
            tip + half * direction, np.sqrt(radius ** 2 + half ** 2)
        )
        if not candidate_idx:
            return np.empty(0, dtype=np.intp)

        candidate_idx = np.asarray(candidate_idx, dtype=np.intp)
        vecs = self.all_points[candidate_idx] - tip
        along = vecs @ direction
        perp_dist = np.linalg.norm(vecs - np.outer(along, direction), axis=1)

        ahead = (along > 0) & (along <= reach) & (perp_dist < radius)
        ahead &= ~np.asarray(claimed_mask, dtype=bool)[candidate_idx]
        return candidate_idx[ahead]

    def rank_stops(self, stops, claimed_mask) -> list:
        """Order *stops* by how many unclaimed points lie ahead of each, most
        first, so genuine feature ends sink to the bottom of the queue.

        Returns ``[(stop, n_ahead), ...]``. With up to two ends per line, thirty
        features produce sixty stops and most are real ends — without ranking the
        user would step through all of them to find the few worth extending.
        """
        scored = [(stop, int(self.unclaimed_ahead(stop, claimed_mask).size))
                  for stop in stops]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def stop_seed_indices(self, stop, line_indices) -> np.ndarray:
        """The line's own points around *stop* — the traced body a re-seed
        inherits its heading from.

        Union these with the user's fresh picks and hand the result to
        ``grow()``: PCA orders the combined body, the anchor lands in its middle,
        and the march traverses it under the same cylinder rule as any other
        growth (see ``_grow_axis_trace``). One search reach of already-traced
        line is enough body to fix the direction.
        """
        line_indices = np.asarray(line_indices, dtype=np.intp)
        if line_indices.size == 0:
            return line_indices
        radius = self.cylinder_length * self.reach_factor
        dist = np.linalg.norm(
            self.all_points[line_indices] - np.asarray(stop.tip, dtype=float), axis=1
        )
        return line_indices[dist <= radius]

    def _pick_bounded_search(self, stop, picks):
        """How far and how wide an extension must search to actually ARRIVE at
        the user's *picks*. Returns ``(reach_factor, cylinder_radius)``.

        The normal search is tuned to bridge incidental gaps blindly, so it is
        deliberately short and narrow — a re-seed under it stops dead at the very
        hole the user just pointed across. The picks are explicit evidence that
        the feature continues there, so the search is authorised to travel far
        enough and wide enough to see the furthest one, and no further. Both
        relaxations are bounded by the user's own gesture rather than by a guess.

        The width matters as much as the distance, and less obviously: the tube
        is aimed along the heading the march *drifted* to, and any angular error
        is amplified by how far it reaches. A heading 2 degrees off — routine
        after a few re-fits — misses by 0.2 m at 6 m and by 0.85 m at 25 m. Grant
        the reach without the width and the tube sails straight past the points
        the user picked, which is exactly the failure this method exists to stop.
        """
        if picks.size == 0:
            return self.reach_factor, self.cylinder_radius

        tip = np.asarray(stop.tip, dtype=float)
        direction = unit(np.asarray(stop.direction, dtype=float))
        vecs = self.all_points[picks] - tip

        needed = float(np.linalg.norm(vecs, axis=1).max()) * _REACH_MARGIN
        along = vecs @ direction
        perp = float(np.linalg.norm(vecs - np.outer(along, direction), axis=1).max())
        return (max(self.reach_factor, needed / self.cylinder_length),
                max(self.cylinder_radius, perp * _REACH_MARGIN))

    def rollback_stop(self, line, stop, n_steps):
        """Discard the last *n_steps* of march from the end of *line* at *stop*,
        returning ``(trimmed_line, new_stop)``.

        A march often stops because the last step or two went wrong, not because
        the feature ended: the fit window caught a neighbouring object, the axis
        drifted off the centre, and the heading that came out of it points
        somewhere the feature never went. Extending from that tip inherits the
        bad heading and repeats the mistake. Rolling the end back to before the
        damage gives the re-seed a clean body and a clean direction.

        Trimming is done on the centerline by ARC LENGTH, not by slicing the
        recorded cylinder list: cylinders accumulate across both march directions
        and any earlier extensions, so their order no longer identifies "the last
        few of this end". Arc length along the polyline does, and it follows
        curves correctly.

        Returns ``None`` when the rollback would consume the whole line (nothing
        left to re-seed from) or when there is no centerline to trim.
        """
        if n_steps <= 0:
            return line, stop
        centerline = line.centerline
        if centerline is None or len(centerline) < 2:
            return None

        # Orient the polyline so the end being trimmed is last.
        pts = np.asarray(centerline, dtype=float)
        tip = np.asarray(stop.tip, dtype=float)
        if np.linalg.norm(pts[0] - tip) < np.linalg.norm(pts[-1] - tip):
            pts = pts[::-1]

        # The march advances by this much per step, so N steps is this far back.
        distance = n_steps * self.cylinder_length * (1.0 - self.overlap)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if distance >= seg.sum():
            return None  # would trim away the whole line

        # Walk back from the trimmed end until *distance* of arc is consumed,
        # then cut partway along the segment we land in.
        walked = 0.0
        cut_i = len(pts) - 1
        for i in range(len(seg) - 1, -1, -1):
            if walked + seg[i] >= distance:
                cut_i = i
                break
            walked += seg[i]
        remainder = distance - walked
        frac = 0.0 if seg[cut_i] <= 1e-12 else remainder / seg[cut_i]
        cut_point = pts[cut_i + 1] + (pts[cut_i] - pts[cut_i + 1]) * frac

        kept = np.vstack([pts[:cut_i + 1], cut_point[None, :]])
        if len(kept) < 2:
            return None

        heading = unit(kept[-1] - kept[-2])  # still pointing outward
        new_stop = MarchStop(cut_point, heading, stop.reason)

        # Drop the points the trimmed stretch had collected. Scoped to the
        # neighbourhood of the cut so the far side of a curved or doubled-back
        # line is never caught by the same half-space test.
        indices = np.asarray(line.indices, dtype=np.intp)
        rel = self.all_points[indices] - cut_point
        near = np.linalg.norm(rel, axis=1) <= distance + self.cylinder_length
        keep = ~(near & ((rel @ heading) > 0))

        return GrownLine(
            indices[keep],
            kept.astype(np.float32),
            [c for c in line.cylinders
             if not self._beyond(c[0], cut_point, heading, distance)],
            list(line.end_cylinders),
            [s for s in line.stops if s is not stop] + [new_stop],
        ), new_stop

    def _beyond(self, point, cut_point, heading, span):
        """Whether *point* lies past the cut, within the trimmed neighbourhood."""
        rel = np.asarray(point, dtype=float) - cut_point
        return bool(np.linalg.norm(rel) <= span + self.cylinder_length
                    and rel @ heading > 0)

    def splice_centerline(self, centerline, segments):
        """Re-join an existing polyline with freshly marched *segments*.

        A polyline's consecutive vertex pairs are valid input segments, so this
        is just ``_join_centerline`` over the union — which already pools,
        dedupes and re-chains from a PCA extreme, so it needs no help ordering an
        extension relative to what it extends.
        """
        pooled = list(segments)
        if centerline is not None and len(centerline) >= 2:
            pts = np.asarray(centerline, dtype=float)
            pooled.extend(zip(pts[:-1], pts[1:]))
        return self._join_centerline(pooled)

    def _pick_segments(self, stop, picks):
        """Centerline segments running from *stop* out through the user's
        *picks*, plus the frontier they leave behind.

        Returns ``(segments, new_stop)``, or ``(None, None)`` when no pick lies
        ahead of the stop. The picks are binned along the heading by
        cylinder_length and reduced to one centroid per bin — the same band
        centroids the march itself builds its centerline from, so a hand-walked
        stretch and a marched one produce the same kind of polyline instead of a
        zigzag through every individual point the user happened to click.
        """
        tip = np.asarray(stop.tip, dtype=float)
        direction = unit(np.asarray(stop.direction, dtype=float))
        pts = self.all_points[picks]
        along = (pts - tip) @ direction

        ahead = along > 0
        if not np.any(ahead):
            return None, None
        pts, along = pts[ahead], along[ahead]

        bins = np.floor(along / max(self.cylinder_length, 1e-9)).astype(np.int64)
        centroids = np.array([pts[bins == b].mean(axis=0)
                              for b in np.unique(bins)])

        chain = np.vstack([tip[None, :], centroids])
        segments = list(zip(chain[:-1], chain[1:]))

        # The frontier is the far end of what the user just claimed, aimed the
        # way their picks were going — so the next round of picking continues
        # from there rather than from the stop that was already dealt with.
        heading = unit(chain[-1] - chain[-2]) if len(chain) > 1 else direction
        return segments, MarchStop(chain[-1], heading, STOP_PICKED_END)

    def extend_from_stop(self, stop, line, extra_indices):
        """Continue *line* past *stop*, guided by the user's *extra_indices*.

        Returns an ``Extension`` (see above), or None when there was nothing to
        work with at all — no picks and too small a body to re-grow from.

        **The picks are always adopted.** A point the user picked is a point they
        looked at and judged to be on this line; the engine's opinion about
        whether it could have got there by itself does not override that. So the
        march is the *bonus*: it runs, and whatever it reaches is spliced in, but
        if it stops dead the picks still join the line and the frontier still
        advances to them. That is what makes the workflow always able to make
        progress — worst case the user walks the feature themselves, a few points
        at a time, which is exactly the situation (a long occlusion, a cable
        through canopy) where nothing automatic was ever going to work.

        Three things are relaxed for this growth and nothing else: the march runs
        only outward (``only_direction``), its search is opened up just far and
        wide enough to arrive at the furthest pick (``_pick_bounded_search``),
        and it may bridge a gap onto the picks however few of them there are
        (``_bridge_gap``). Everything after the landing — the fit window, the
        angle gate, membership — is the ordinary growth every other line gets.
        """
        picks = np.asarray(extra_indices, dtype=np.intp)
        seeds = np.union1d(self.stop_seed_indices(stop, line.indices), picks)
        if seeds.size < 2:
            return None

        # Grow in isolation so the spliced geometry is only this extension's.
        self.debug_cylinders = []
        self.debug_lines = []
        self.debug_end_cylinders = []

        saved = (self.reach_factor, self.cylinder_radius)
        try:
            self.reach_factor, self.cylinder_radius = \
                self._pick_bounded_search(stop, picks)
            self._bridge_targets = picks
            grown = self.grow(seeds, only_direction=stop.direction)
        finally:
            self.reach_factor, self.cylinder_radius = saved
            self._bridge_targets = None

        # Did the MARCH contribute, or did it stop immediately and hand back the
        # seeds it was given? grow() always returns the seeds, so counting points
        # cannot tell the two apart — and the user would be told the trace
        # advanced while it stood still.
        marched = np.setdiff1d(grown, seeds, assume_unique=False).size > 0

        segments = list(self.debug_lines)
        new_stops = self.march_stops()
        if not marched:
            # Nothing but the picks landed. Walk the centerline out through them
            # anyway and put the frontier at their far end, so the next round of
            # picking carries on from there instead of re-offering this stop.
            pick_segments, frontier = self._pick_segments(stop, picks)
            if pick_segments is None:
                return None
            segments, new_stops = pick_segments, [frontier]

        # The stop just extended from is gone; the new frontier replaces it.
        # Everything else the line already had is kept. end_cylinders accumulate
        # (display only) — the caller rebuilds stop markers from ``stops``.
        remaining = [s for s in line.stops if s is not stop]
        extended = GrownLine(
            np.union1d(np.asarray(line.indices, dtype=np.intp), grown),
            self.splice_centerline(line.centerline, segments),
            list(line.cylinders) + list(self.debug_cylinders),
            list(line.end_cylinders) + list(self.debug_end_cylinders),
            remaining + new_stops,
        )
        return Extension(extended, new_stops[-1] if new_stops else stop, marched)

    def grow_lines(self, seed_groups, *, progress_cb=None, cancel_event=None) -> list:
        """
        Grow one line per physical feature, greedily, largest cluster first.

        *seed_groups* is a list of index arrays (into ``all_points``), one per
        cluster of the picked seeds (typically DBSCAN of the user's selection).
        A single physical line is often split by clustering into several
        clusters; this method recovers it without any merge heuristic:

            1. Grow a line from the LARGEST remaining cluster.
            2. Discard that line if it mostly retraces ground the lines kept so
               far already swept (``_DUPLICATE_FRAC``).
            3. Drop every remaining cluster that growth consumed (most of its
               points fall in the region swept so far) — those clusters were
               just pieces of this same line.
            4. Repeat until no clusters remain.

        Because a cluster is removed only when the growth actually reached it,
        parallel neighbouring lines are never fused (growth never crosses to
        them) and a line split into many clusters comes out whole (the march
        walks through all of them).

        Steps 2–3 both test against the SWEPT region accumulated over the kept
        lines, not their member points — see ``_CONSUME_FRAC`` and
        ``_DUPLICATE_FRAC`` for why each is needed to keep one physical feature
        from coming back as two lines drawn on top of each other.

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

        # Everything the KEPT lines have swept, as a per-point flag (cheap to
        # test, and cumulative — a cluster half-covered by one line and half by
        # the next is still recognised as already grown).
        swept_mask = np.zeros(len(self.all_points), dtype=bool)

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

                # Retracing a kept line? A cluster can legitimately survive the
                # consume test below (e.g. it lies past the point where the first
                # line stopped) and then march BACK over that line — each march
                # runs both ways from its anchor. Drop the duplicate rather than
                # drawing a second centerline on top of the first.
                duplicate = float(np.mean(swept_mask[grown])) >= _DUPLICATE_FRAC

                # Mark this growth's sweep whether or not the line is kept, then
                # drop the clusters it consumed. A cluster is part of an
                # already-grown feature if most of its points fall in the swept
                # region; clusters merely crossed (a few shared points at an
                # intersection) stay for their own pass. A discarded duplicate
                # still swept real ground — any cluster in there would only
                # retrace the same feature again. The loop always shrinks: this
                # iteration's seed cluster was popped whatever the outcome.
                swept_mask[self.swept_indices()] = True
                pool = [c for c in pool
                        if float(np.mean(swept_mask[c])) < _CONSUME_FRAC]
                if duplicate:
                    continue

                centerline = self._join_centerline(list(self.debug_lines))
                lines.append(GrownLine(np.array(sorted(set(int(i) for i in grown)),
                                                dtype=np.intp),
                                       centerline, list(self.debug_cylinders),
                                       list(self.debug_end_cylinders),
                                       self.march_stops()))

                if progress_cb is not None:
                    progress_cb(len(lines), total,
                                f"Growing lines — {len(lines)} traced, "
                                f"{len(pool)} seed group(s) left")
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

    def _grow_axis_trace(self, seed_indices, use_linearity_gate=False,
                         only_direction=None) -> np.ndarray:
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
        directions = (start_dir, -start_dir)
        if only_direction is not None:
            # Keep the locally-fitted heading (a better tangent than the caller's
            # reference) but only the sign that travels the requested way.
            ref = unit(np.asarray(only_direction, dtype=float))
            directions = (start_dir if np.dot(start_dir, ref) >= 0 else -start_dir,)
        for direction in directions:
            grown, stop = self._march(anchor, direction, use_linearity_gate)
            collected |= grown
            self._march_stops.append(stop)
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

    def _march(self, tip, direction, use_linearity_gate):
        """Drive the cylinder march in one direction from *tip*.

        Returns ``(collected, MarchStop)`` — the point indices grown in this
        direction, and where/why the march gave up.

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
                    # Keep the points still hugging the CURRENT heading before
                    # giving up — they are on this line even though the window's
                    # fitted axis turned away (the turn is usually caused by
                    # something else entering the window). Mirrors the
                    # too-few-points branch below; without it a bend throws away
                    # up to a full window of genuine members.
                    #
                    # Deliberately NOT recorded as swept: a bend often means a
                    # second feature crosses here, and marking its neighbourhood
                    # swept would consume that feature's seed group.
                    collected.update(
                        int(i) for i in candidate_idx[near_mask
                                                      & (perp_dist < self.ransac_threshold)]
                    )
                    reason = STOP_SHARP_BEND  # direction turned more than max_angle
                    break

                near_idx = candidate_idx[near_mask]
                # The whole fit window was swept, not just the points close
                # enough to the axis to become members.
                self._swept_chunks.append(near_idx)
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

                next_tip = self._bridge_gap(current_tip, current_dir, along,
                                            tube_mask, candidate_idx)
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
            prev_mid = last_end

        # And on to the furthest point this march actually claimed, if that lies
        # further still. Geometry is only recorded when a window FITS, but the
        # else branch above claims near on-axis points and bridges onward without
        # fitting anything — so a march that ends by creeping along a sparse
        # continuation holds points its centerline never reaches, and a march
        # that never fits a single window (the whole continuation sparse) draws
        # no centerline at all. Measured: 4 points claimed out to x=13.4 with the
        # centerline still ending at x=10.0.
        #
        # Only ever LENGTHENS the chain along the heading already travelled — it
        # cannot move or reroute a segment the march fitted.
        if collected:
            idx = np.fromiter(collected, dtype=np.intp, count=len(collected))
            reach_out = (self.all_points[idx] - prev_mid) @ current_dir
            if reach_out.size and float(reach_out.max()) > _TAIL_MIN_GAIN:
                tail = self.all_points[idx[int(np.argmax(reach_out))]]
                # No cylinder: nothing was fitted here, so drawing a search
                # window over it would claim more than the march did.
                self._record_step(prev_mid, tail, record_cylinder=False)

        # The last cylinder searched in this direction is the stop point at this
        # end of the line — the march ended just beyond it. Record it together
        # with WHY it stopped (reason), so ends can be split by reason and drawn.
        if len(self.debug_cylinders) > start_n:
            self.debug_end_cylinders.append((reason, self.debug_cylinders[-1]))

        # current_tip is where the search was looking when it gave up, and
        # current_dir the heading it was on — both valid even when no step ever
        # succeeded (the anchor and its initial heading), which is the case the
        # end_cylinders record above cannot represent.
        return collected, MarchStop(current_tip.copy(), current_dir.copy(), reason)

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

    def _bridge_gap(self, current_tip, current_dir, along, tube_mask,
                    candidate_idx=None):
        """Jump the tip toward the nearest far-band continuation within the tube.

        Requires at least min_points in the far reach zone WITHIN THE TUBE
        (cylinder_radius of the current heading). The tube gate keeps the bridge
        on the SAME line longitudinally — a parallel neighbour beyond the radius
        is excluded — and a real bend (points turned off the heading) finds none
        within the tube. Returns the next tip (landed just inside the fit window
        of the nearest far point, keeping the heading since there is no data in
        the gap to re-fit), or ``None`` when there is no continuation (the caller
        stops with STOP_TOO_FEW_POINTS).

        During a guided extension (``_bridge_targets`` set) the user's picks can
        satisfy the landing on their own, however few of them there are. The
        min_points gate exists to stop the march from bridging blindly onto noise
        — a handful of stray returns beyond a gap is not evidence of a feature.
        A human who looked at the cloud and pointed at those points IS that
        evidence, and the gate must not overrule them: sparse continuations
        (a thin cable through foliage, a couple of returns per metre) are the
        common case this workflow exists to recover.
        """
        far_band = tube_mask & (along > self.cylinder_length)
        if np.count_nonzero(far_band) < self.min_points:
            if candidate_idx is None or self._bridge_targets is None:
                return None
            far_band &= np.isin(candidate_idx, self._bridge_targets)
            if not np.any(far_band):
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
# Not a march outcome: the end of a stretch the user claimed by picking, where
# growth could not carry on by itself. It marks the frontier to keep picking
# from, so it reads differently in the review window from a march that failed.
STOP_PICKED_END = "picked_end"

STOP_REASONS = {
    STOP_TOO_FEW_POINTS: ("too few points", np.array([1.0, 0.1, 0.1], dtype=np.float32)),   # red
    STOP_SHARP_BEND:     ("sharp bend",      np.array([1.0, 0.5, 0.0], dtype=np.float32)),   # orange
    STOP_EMPTY_SPACE:    ("empty space",     np.array([1.0, 0.0, 1.0], dtype=np.float32)),   # magenta
    STOP_STEP_CAP:       ("step cap",        np.array([1.0, 1.0, 1.0], dtype=np.float32)),   # white
    STOP_PICKED_END:     ("end of your picks", np.array([0.0, 1.0, 1.0], dtype=np.float32)), # cyan
}


# --------------------------------------------------------------------------- #
# Stop records — MarchStop <-> plain data for the project file                 #
#                                                                             #
# A result branch carries its stops so a short trace can be continued in a     #
# later session. Persisted as plain dicts, not MarchStop tuples: the records   #
# outlive the session inside the project file, so they must not depend on this #
# module's class staying importable at the same path.                          #
# --------------------------------------------------------------------------- #

def lines_to_traces(lines, params, resolved=()) -> dict:
    """Pack grown *lines* into the plain ``Clusters.line_traces`` structure.

    A line's label is its index in *lines*, matching the cluster labels the
    plugin writes. Point indices are deliberately NOT stored — they are
    recoverable from those labels, and duplicating them would let the two
    disagree. *resolved* is a set of ``(label, reason, tip_tuple)`` keys for
    stops the user has already dismissed as genuine feature ends.
    """
    packed = []
    for label, line in enumerate(lines):
        centerline = ([] if line.centerline is None
                      else [[float(v) for v in pt] for pt in line.centerline])
        packed.append({
            "label": int(label),
            "centerline": centerline,
            # Search cylinders travel with the trace so a session that reopens a
            # saved result can REBUILD the cylinder wireframe after extending it,
            # instead of either leaving a branch that stops where the original
            # run did or replacing it with the extension's few cylinders alone.
            # Eight floats per step — a rounding error next to the point data.
            "cylinders": [
                {
                    "tip": [float(v) for v in tip],
                    "direction": [float(v) for v in direction],
                    "radius": float(radius),
                    "length": float(length),
                }
                for tip, direction, radius, length in line.cylinders
            ],
            "stops": [
                {
                    "tip": [float(v) for v in stop.tip],
                    "direction": [float(v) for v in stop.direction],
                    "reason": stop.reason,
                    "resolved": stop_key(label, stop) in resolved,
                }
                for stop in line.stops
            ],
        })
    return {"params": dict(params), "lines": packed}


def traces_to_lines(traces, labels) -> list:
    """Rebuild ``GrownLine`` objects from persisted *traces* plus the cluster
    *labels* array the point indices come from.

    End cylinders are not persisted: they are display leftovers of the original
    run that only ever accumulate, and stop markers are drawn from ``stops``
    instead. Search cylinders are, so an extension can rebuild the whole
    wireframe rather than just its own contribution — traces written before that
    was stored simply come back with none, and the branch is left as it was.
    """
    lines = []
    for entry in traces.get("lines", []):
        label = int(entry["label"])
        centerline = np.asarray(entry["centerline"], dtype=np.float32)
        cylinders = [
            (np.asarray(rec["tip"], dtype=float),
             np.asarray(rec["direction"], dtype=float),
             float(rec["radius"]), float(rec["length"]))
            for rec in entry.get("cylinders", [])
        ]
        lines.append(GrownLine(
            np.where(np.asarray(labels) == label)[0].astype(np.intp),
            centerline if len(centerline) >= 2 else None,
            cylinders, [],
            [record_to_stop(rec) for rec in entry.get("stops", [])],
        ))
    return lines


def resolved_stop_keys(traces) -> set:
    """The ``stop_key`` set for stops already dismissed as genuine ends, so a
    reopened session does not walk the user back through them."""
    return {
        stop_key(int(entry["label"]), record_to_stop(rec))
        for entry in traces.get("lines", [])
        for rec in entry.get("stops", [])
        if rec.get("resolved")
    }


def record_to_stop(record) -> MarchStop:
    """Rebuild a usable ``MarchStop`` from one persisted stop record."""
    return MarchStop(
        np.asarray(record["tip"], dtype=float),
        np.asarray(record["direction"], dtype=float),
        record["reason"],
    )


def stop_key(label, stop):
    """Hashable identity for a stop, stable across a save/load round trip.

    Stops have no id of their own and are rebuilt as fresh objects on load, so
    identity has to come from the values. Tips are rounded to the millimetre —
    far below any real feature detail, and enough to survive float32 storage.
    """
    return (int(label), stop.reason,
            tuple(round(float(v), 3) for v in stop.tip))


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
