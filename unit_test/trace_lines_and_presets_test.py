# trace_lines_and_presets_test.py
#
# Trace Lines (the line tracer as its own tool) and settings presets in the
# parameter dialog. Checks: the tool's few settings, worded without "Kalman";
# the seed grouping distance it works out from the picks (a sparse line stays
# one group, two lines stay two); it grows with the line tracer; presets save,
# list, load and delete, and loading one fills the dialog; the shipped
# "Overhead cable" preset matches the tool's settings.
import sys
import os
import tempfile
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

import services.preset_service as presets
from core.services.kalman_line_tracer import KalmanLineTracer

lrg = importlib.import_module("plugins.020_Points.020_Clustering.040_linear_region_growing_plugin")
tl = importlib.import_module("plugins.020_Points.020_Clustering.045_trace_lines_plugin")

app = QApplication.instance() or QApplication(sys.argv)


def test_settings_are_few_and_plain():
    schema = tl.TraceLinesPlugin().get_parameters()
    assert set(schema) == {"max_gap", "max_kink", "min_bend_radius", "open_air",
                           "window", "show_lines"}, set(schema)
    for name, info in schema.items():
        text = f"{info.get('label', '')} {info.get('description', '')}".lower()
        assert "kalman" not in text, f"'{name}' mentions Kalman"
    assert "growth_mode" in lrg.LinearRegionGrowingPlugin().get_parameters()
    assert "Kalman Trace" not in lrg.LinearRegionGrowingPlugin().get_parameters()["growth_mode"]["options"]
    print(f"Trace Lines: {len(schema)} settings, none mention Kalman")


def test_seed_grouping_from_the_picks():
    rng = np.random.default_rng(0)
    # sparse picks along one line, unevenly spaced
    x = np.sort(rng.uniform(0, 6, 90))
    one = np.stack([x, rng.normal(0, 0.01, 90), rng.normal(0, 0.01, 90)], 1)
    two = np.vstack([one, one + [0, 1.0, 0]])            # a second line 1 m away
    eps1 = lrg._auto_seed_eps(one)
    eps2 = lrg._auto_seed_eps(two)
    from sklearn.cluster import DBSCAN
    n1 = DBSCAN(eps=eps1, min_samples=2).fit_predict(one).max() + 1
    n2 = DBSCAN(eps=eps2, min_samples=2).fit_predict(two).max() + 1
    assert n1 == 1, f"one sparse line split into {n1} groups (eps {eps1:.3f})"
    assert n2 == 2, f"two lines 1 m apart grouped into {n2} (eps {eps2:.3f})"
    print(f"seed grouping: one line -> 1 group (eps {eps1 * 100:.0f} cm), two lines -> 2")


def test_grows_with_the_line_tracer():
    pts = np.zeros((10, 3), dtype=np.float32)
    engine, grower = tl.TraceLinesPlugin()._make_growers(
        pts, {"window": 2.0, "max_gap": 7.0, "max_kink": 10.0,
              "min_bend_radius": 4.0, "open_air": False}, None, None)
    assert isinstance(engine, KalmanLineTracer)
    assert engine.window == 2.0 and engine.step == 1.0 and engine.max_gap == 7.0
    assert engine.max_turn_deg == 10.0 and abs(engine.max_curvature - 0.25) < 1e-9
    assert engine.open_air is False
    assert hasattr(grower, "unclaimed_ahead")          # extension still uses the march
    print("Trace Lines grows with the line tracer; extension keeps the axis march")


def test_preset_files():
    with tempfile.TemporaryDirectory() as root:
        assert presets.list_presets("p", root) == []
        presets.save_preset("p", "Wet / day: 2", {"a": 1.5, "b": True}, root)
        presets.save_preset("p", "alpha", {"a": 2.0}, root)
        assert presets.list_presets("p", root) == ["alpha", "Wet / day: 2"]
        assert presets.load_preset("p", "Wet / day: 2", root) == {"a": 1.5, "b": True}
        presets.save_preset("p", "alpha", {"a": 3.0}, root)          # replace
        assert presets.load_preset("p", "alpha", root) == {"a": 3.0}
        presets.delete_preset("p", "alpha", root)
        assert presets.list_presets("p", root) == ["Wet / day: 2"]
    print("presets: save, list, load, replace, delete (odd names kept in the file)")


def test_dialog_loads_a_preset():
    from gui.dialog_boxes.dynamic_dialog import DynamicDialog
    schema = tl.TraceLinesPlugin().get_parameters()
    with tempfile.TemporaryDirectory() as root:
        saved_root = presets.PRESETS_ROOT
        presets.PRESETS_ROOT = root
        try:
            presets.save_preset("trace_lines", "Kerb", {
                "max_gap": 3.0, "max_kink": 45.0, "min_bend_radius": 3.0,
                "open_air": False, "window": 0.5, "not_a_setting": 1})
            dialog = DynamicDialog("Trace Lines Parameters", schema, plugin_name="trace_lines")
            assert dialog.preset_combo.findText("Kerb") >= 0
            dialog.preset_combo.setCurrentText("Kerb")
            dialog._load_preset()
            got = dialog.get_parameters()
            assert got["max_gap"] == 3.0 and got["max_kink"] == 45.0
            assert got["min_bend_radius"] == 3.0 and got["open_air"] is False
            assert got["window"] == 0.5 and got["show_lines"] is False   # not in preset: kept
            assert "not_a_setting" not in got
        finally:
            presets.PRESETS_ROOT = saved_root
    plain = DynamicDialog("Plain", schema)
    assert plain.preset_combo is None                   # no plugin name, no presets
    print("dialog: a preset fills the settings; unknown keys ignored, missing kept")


def test_shipped_overhead_cable_preset():
    shipped = presets.load_preset("trace_lines", "Overhead cable")
    schema = tl.TraceLinesPlugin().get_parameters()
    assert set(shipped) == set(schema), set(shipped) ^ set(schema)
    print(f"'Overhead cable' preset: {shipped}")


if __name__ == "__main__":
    test_settings_are_few_and_plain()
    test_seed_grouping_from_the_picks()
    test_grows_with_the_line_tracer()
    test_preset_files()
    test_dialog_loads_a_preset()
    test_shipped_overhead_cable_preset()
    print("\nAll Trace Lines and preset tests passed.")
