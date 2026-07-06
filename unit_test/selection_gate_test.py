"""
Tests for the selection gate's pure logic (``application/selection_gate.py``):

- ``selection_kind()`` normalises ``requires_selection()`` into a kind or
  ``None``, tolerating the legacy boolean contract.
- ``selection_present()`` reports whether points / branches are currently
  selected, reading the global singletons.

The ``SelectionPrompt`` dialog needs Qt + a display, so it is exercised
manually, not here. Importing the module only needs PyQt5 installed (no
``QApplication`` and no widgets are created below).
"""

import sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import types
import unittest

from config.config import global_variables
from application.selection_gate import (
    selection_kind, selection_present, POINTS, BRANCHES, EITHER,
)


class _Req:
    """Minimal stand-in plugin exposing a fixed ``requires_selection()`` value."""

    def __init__(self, value):
        self._value = value

    def requires_selection(self):
        return self._value


def _plugin_returning(value):
    """A plugin *class* (selection_kind instantiates it) with a fixed value."""
    return lambda: _Req(value)


class SelectionKindTest(unittest.TestCase):

    def test_none_for_no_requirement(self):
        self.assertIsNone(selection_kind(_plugin_returning(None)))
        self.assertIsNone(selection_kind(_plugin_returning(False)))
        self.assertIsNone(selection_kind(_plugin_returning("")))
        self.assertIsNone(selection_kind(None))

    def test_legacy_true_is_points(self):
        self.assertEqual(selection_kind(_plugin_returning(True)), POINTS)

    def test_explicit_kinds(self):
        self.assertEqual(selection_kind(_plugin_returning("points")), POINTS)
        self.assertEqual(selection_kind(_plugin_returning("branches")), BRANCHES)
        self.assertEqual(selection_kind(_plugin_returning("either")), EITHER)
        # case / whitespace tolerant
        self.assertEqual(selection_kind(_plugin_returning("  Points ")), POINTS)

    def test_unknown_truthy_falls_back_to_points(self):
        self.assertEqual(selection_kind(_plugin_returning("seeds")), POINTS)

    def test_missing_or_raising_hook_is_none(self):
        class NoHook:
            pass
        self.assertIsNone(selection_kind(lambda: NoHook()))

        class Boom:
            def requires_selection(self):
                raise RuntimeError("boom")
        self.assertIsNone(selection_kind(lambda: Boom()))


class SelectionPresentTest(unittest.TestCase):

    def setUp(self):
        self._viewer = global_variables.global_pcd_viewer_widget
        self._controller = global_variables.global_application_controller

    def tearDown(self):
        global_variables.global_pcd_viewer_widget = self._viewer
        global_variables.global_application_controller = self._controller

    def _set(self, picked=None, polygons=None, branches=None):
        global_variables.global_pcd_viewer_widget = types.SimpleNamespace(
            picked_points_indices=picked or [],
            _selection_polygons=polygons or [],
        )
        global_variables.global_application_controller = types.SimpleNamespace(
            selected_branches=branches or [],
        )

    def test_none_kind_is_always_satisfied(self):
        self._set()  # nothing selected
        self.assertTrue(selection_present(None))

    def test_points_present(self):
        self._set(picked=[3, 7])
        self.assertTrue(selection_present(POINTS))
        self.assertFalse(selection_present(BRANCHES))

    def test_points_via_polygon(self):
        self._set(polygons=[("poly",)])
        self.assertTrue(selection_present(POINTS))

    def test_branches_present(self):
        self._set(branches=["uid-1"])
        self.assertTrue(selection_present(BRANCHES))
        self.assertFalse(selection_present(POINTS))

    def test_either(self):
        self._set()  # nothing
        self.assertFalse(selection_present(EITHER))
        self._set(branches=["uid-1"])
        self.assertTrue(selection_present(EITHER))
        self._set(picked=[1])
        self.assertTrue(selection_present(EITHER))

    def test_nothing_selected(self):
        self._set()
        self.assertFalse(selection_present(POINTS))
        self.assertFalse(selection_present(BRANCHES))


if __name__ == "__main__":
    unittest.main()
