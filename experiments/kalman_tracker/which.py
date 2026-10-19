import ktrack
from robust2 import *
fails = [{"S_slab_points": 24}, {"S_slab_cores": 12}, {"S_slab_cores": 30}, {"S_steps_per_window": 6},
         {"S_min_share": 0.15}, {"curvature_noise": 0.1}, {"max_curvature": 0.01}, {"max_curvature": 0.05}]
for label, use_dir, spw in [("now", True, None), ("no direction readings", False, None),
                            ("half-window steps", True, 2), ("both", False, 2)]:
    ktrack.kt_use_direction = use_dir
    ok = tot = 0
    for f in fails:
        c = dict(DEFAULTS); c.update(f)
        if spw and "S_steps_per_window" not in f:
            c["S_steps_per_window"] = spw
        r = run(c); ok += sum(x == "OK " for x in r); tot += len(r)
    print(f"{label:24s}: {ok}/{tot} on the 8 failing settings", flush=True)
