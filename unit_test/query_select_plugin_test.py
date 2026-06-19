"""
Tests for attribute-query selection (ArcGIS-style "Select By Attributes").

Covers the GUI-free core: ``services.attribute_query.evaluate_query`` over
geometry / normals / attribute fields with AND vs OR combiners, the ``between``
operator, NaN handling, and malformed queries. Also exercises the plugin's
``execute`` end-to-end (query -> Masks) since branch creation downstream relies
on a correct boolean mask of the right length.

The plugin module lives under a digit-prefixed directory that is not a valid
Python package name, so it is loaded directly from its file path with importlib.
"""

import sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import importlib.util
import unittest

import numpy as np

from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from core.entities.masks import Masks
from services.attribute_query import evaluate_query, evaluate_clause

_PLUGIN_PATH = os.path.join(
    REPO, "plugins", "030_Selection", "020_query_select_plugin.py")
_spec = importlib.util.spec_from_file_location("query_select_plugin_mod", _PLUGIN_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QuerySelectPlugin = _mod.QuerySelectPlugin


def _cloud():
    """A 5-point cloud with height, a normal-Z field, and a scalar attribute."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 4.0],
    ], dtype=np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (5, 1))
    normals[0] = [1.0, 0.0, 0.0]  # nz = 0 for the first point
    pc = PointCloud(points=points, normals=normals)
    pc.add_attribute("values", np.array([10.0, 20.0, np.nan, 40.0, 50.0],
                                        dtype=np.float32))
    return pc


class TestEvaluateQuery(unittest.TestCase):

    def test_single_condition_z(self):
        pc = _cloud()
        mask = evaluate_query(pc, [{"field": "z", "op": ">", "value": 1.5}], "AND")
        np.testing.assert_array_equal(mask, [False, False, True, True, True])

    def test_and_combines_conditions(self):
        pc = _cloud()
        clauses = [
            {"field": "z", "op": ">=", "value": 1.0},
            {"field": "z", "op": "<=", "value": 3.0},
        ]
        mask = evaluate_query(pc, clauses, "AND")
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_or_combines_conditions(self):
        pc = _cloud()
        clauses = [
            {"field": "z", "op": "<", "value": 0.5},   # point 0
            {"field": "z", "op": ">", "value": 3.5},   # point 4
        ]
        mask = evaluate_query(pc, clauses, "OR")
        np.testing.assert_array_equal(mask, [True, False, False, False, True])

    def test_between_operator(self):
        pc = _cloud()
        mask = evaluate_query(
            pc, [{"field": "z", "op": "between", "value": 1.0, "value2": 3.0}], "AND")
        np.testing.assert_array_equal(mask, [False, True, True, True, False])
        # Order of the two bounds must not matter.
        mask_rev = evaluate_query(
            pc, [{"field": "z", "op": "between", "value": 3.0, "value2": 1.0}], "AND")
        np.testing.assert_array_equal(mask_rev, mask)

    def test_attribute_field_and_nan(self):
        pc = _cloud()
        # values = [10, 20, NaN, 40, 50]; NaN must never satisfy a comparison.
        mask = evaluate_query(
            pc, [{"field": "attr:values", "op": ">", "value": 15.0}], "AND")
        np.testing.assert_array_equal(mask, [False, True, False, True, True])

    def test_normal_z_field(self):
        pc = _cloud()
        mask = evaluate_query(pc, [{"field": "nz", "op": ">", "value": 0.5}], "AND")
        np.testing.assert_array_equal(mask, [False, True, True, True, True])

    def test_missing_field_skipped(self):
        pc = _cloud()
        # 'intensity' is absent -> that clause resolves to None and is skipped,
        # leaving the z clause to decide.
        clauses = [
            {"field": "intensity", "op": ">", "value": 0.0},
            {"field": "z", "op": ">", "value": 2.5},
        ]
        mask = evaluate_query(pc, clauses, "AND")
        np.testing.assert_array_equal(mask, [False, False, False, True, True])

    def test_no_usable_clause_selects_nothing(self):
        pc = _cloud()
        # Non-numeric value -> unusable -> empty selection, not everything.
        mask = evaluate_query(pc, [{"field": "z", "op": ">", "value": "abc"}], "AND")
        self.assertFalse(mask.any())
        self.assertEqual(mask.shape[0], 5)

    def test_non_numeric_value_clause_is_none(self):
        pc = _cloud()
        self.assertIsNone(
            evaluate_clause(pc, {"field": "z", "op": ">", "value": ""}))


class TestQuerySelectPlugin(unittest.TestCase):

    def setUp(self):
        self.plugin = QuerySelectPlugin()
        self.node = DataNode(params="A", data=_cloud(), data_type="point_cloud")

    def test_execute_returns_masks(self):
        params = {
            "new_branch_name": "Tall points",
            "combiner": "AND",
            "clauses": [{"field": "z", "op": ">=", "value": 2.0}],
        }
        result, result_type, deps = self.plugin.execute(self.node, params)
        self.assertEqual(result_type, "masks")
        self.assertIsInstance(result, Masks)
        np.testing.assert_array_equal(result.mask, [False, False, True, True, True])
        self.assertEqual(deps, [self.node.uid])

    def test_execute_no_conditions_raises(self):
        with self.assertRaises(ValueError):
            self.plugin.execute(self.node, {"clauses": [], "combiner": "AND"})

    def test_execute_no_match_raises(self):
        params = {"combiner": "AND",
                  "clauses": [{"field": "z", "op": ">", "value": 100.0}]}
        with self.assertRaises(ValueError):
            self.plugin.execute(self.node, params)


if __name__ == "__main__":
    unittest.main()
