"""What kind of line is being traced: the assumptions that are NOT measured.

Everything about the line's SIZE (thickness, point spacing, and the windows,
steps and point counts that follow from them) is measured from the seed -
see ``ktrack.seed_profile`` / ``ktrack.scaled_cfg``. What remains is what the
seed cannot show: how sharply the line may bend beyond it, how long a gap it
may cross, whether it hangs in open air. Those are properties of the KIND of
feature, collected here as one named preset each, with the reason for every
number.

Only one preset so far, deliberately: presets are added when a real test set
for that kind of feature exists, not ahead of it. A kerb or building edge
would also need a different reading of the points (a step on a surface, not
a thin line in air), not just different numbers.
"""

OVERHEAD_WIRE = dict(
    name="Overhead wire",

    # -- Bending -----------------------------------------------------------
    # A hanging wire sags gently between supports: radius >= 50 m.
    max_curvature=0.02,          # 1/m, tightest bend the tracker may follow
    curvature_noise=0.02,        # 1/m per sqrt(m): how fast the bend may change
    restart_k_sigma=0.02,        # 1/m: bend uncertainty when (re)starting
    # At a pole the wire kinks: up to ~30 deg (seen ~28 deg on Test1).
    reacq_max_deg=30.0,

    # -- Gaps and occlusion -----------------------------------------------
    # Coast this far with no reading (a tree, a hole in the scan; the
    # Test2 tree hid 17 m).
    max_gap=20.0,
    # After a coast the wire may reappear this far off the prediction (m),
    # and within this cone (deg) from the last good point.
    gate_max_coast=0.8,
    bridge_max_deg=15.0,

    # -- Surroundings ------------------------------------------------------
    # A wire hangs in open air: points crowding round it are clutter
    # (vegetation, a pole), not wire. Ring = 1.5..3 core radii round it;
    # wider rings also catch a neighbouring line as clutter.
    ring_inner=1.5,
    ring_factor=3.0,
    clutter_ratio=0.5,           # ring points per core point = clutter
    strict_clutter_ratio=0.1,    # ... just after clutter, before trusting again

    # -- Finding wires without seeds (find_cables.py) -----------------------
    find_max_thickness=0.03,     # m, one-axis sd across a seed voxel
    find_max_slope=0.45,         # |vertical part| of the direction: not poles/stays
    find_min_height=0.5,         # m above the estimated ground
    find_min_length=15.0,        # m, shorter traces are not kept

    # -- Cleaning up (postfilter.py) ----------------------------------------
    # Separate wires here are >= ~30 cm apart; a stretch traced through a
    # gap or a tree can sit ~10 cm off its wire.
    retrace_dist=0.12,           # m: closer than this to a kept line = same wire
    max_retrace_run=1.0,         # m: shorter contacts are crossings, kept
    min_free_length=10.0,        # m of new wire a piece needs to be kept
    max_crowding=1.0,            # ring/core points round a kept line
)

PRESETS = {"overhead_wire": OVERHEAD_WIRE}
