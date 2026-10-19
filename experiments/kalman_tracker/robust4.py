from robust2 import *
rng = np.random.default_rng(42)
ranges = {"step": (0.25, 1.5), "point_sigma": (0.005, 0.05), "tight": (0.03, 0.1), "gate_max": (0.15, 0.4),
          "bridge_win": (3.0, 6.0), "bridge_max_deg": (6.0, 15.0), "clutter_ratio": (0.3, 1.0),
          "curvature_noise": (0.002, 0.1), "max_curvature": (0.01, 0.05), "slab": (0.6, 1.5)}
total = ok = 0
for i in range(25):
    g = {k: round(float(rng.uniform(*r)), 3) for k, r in ranges.items()}
    g["min_pts"] = int(rng.integers(2, 7)); g["max_gap"] = 25.0; g["bridge_max_deg"] = 15.0
    cfg = dict(DEFAULTS); cfg.update(g)
    r = run(cfg); total += len(r); ok += sum(x == "OK " for x in r)
    print(i, " ".join(r), "" if all(x == "OK " for x in r) else g, flush=True)
print(f"random combos passed {ok}/{total}")
