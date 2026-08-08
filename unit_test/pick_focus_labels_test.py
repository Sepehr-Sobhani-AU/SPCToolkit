# pick_focus_labels_test.py
#
# The extension window makes points pickable by giving them a real cluster
# label, because a label already carries both halves of what is wanted: a -1
# point is drawn dark grey AND refused by the viewer's picking filters. These
# tests pin the two things that would silently break that:
#
#   1. WHICH label the candidates get (it decides every other cluster's colour), and
#   2. that the viewer resolves a rendered point to the right row of the labels
#      array — which it does not do by subtracting an offset once LOD is on.
#
# No Qt: the picking filters live in a mixin that only needs numpy and the
# global_variables singleton, and the colours live on Clusters.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types

import numpy as np

from core.entities.clusters import Clusters
from gui.widgets.pcd_viewer._branch_helpers import BranchSelectionMixin
from gui.widgets.pcd_viewer._data_management import DataManagementMixin
from config.config import global_variables


_UID = "result-uid"


class _StubViewer(DataManagementMixin, BranchSelectionMixin):
    """Just enough viewer to exercise the picking filters.

    Both mixins are real — only the state they read is faked, so the tests bind
    to the code that actually decides what can be picked.
    """

    def __init__(self, n_rendered, sample_indices=None):
        self._visible_branches = [_UID]
        self._branch_offsets_cache = {_UID: (0, n_rendered)}
        self._branch_sample_indices = {_UID: sample_indices}
        # Emphasis state, as _init_data would leave it. The rendered slice is
        # only ever measured for its length here.
        self._branch_vertices = {_UID: np.zeros((n_rendered, 6), dtype=np.float32)}
        self._branch_emphasis = {}
        self._emphasis_rows_cache = {}

    def update(self):
        pass


def _install_branch(labels, locked=None):
    """Point the controller singleton at one cluster branch holding *labels*."""
    node = types.SimpleNamespace(
        data_type="cluster_labels",
        data=types.SimpleNamespace(labels=np.asarray(labels),
                                   locked_clusters=locked or {}),
    )
    global_variables.global_application_controller = types.SimpleNamespace(
        selected_branches=[_UID],
        get_node=lambda uid: node,
    )
    return node


def _cluster_colours(labels):
    """{label: rgb} as Clusters would paint them."""
    clusters = Clusters(labels=np.asarray(labels))
    clusters.set_random_color()
    return {int(l): clusters.colors[np.asarray(labels) == l][0]
            for l in sorted(set(labels))}


def test_candidate_label_leaves_other_clusters_alone():
    """Colours are handed out in sorted label order, so where the candidate label
    SORTS decides whether the traced lines keep their colours. One above the
    highest appends and disturbs nothing; anything sorting first shifts every
    line's colour each time candidates appear and vanish — the cables would
    change colour every time the user stepped to another stop."""
    base = [-1, -1, 0, 0, 1, 1]
    before = _cluster_colours(base)
    appended = _cluster_colours(base + [2, 2])       # max + 1
    prepended = _cluster_colours(base + [-2, -2])    # sorts before everything

    print("line colours with candidates = max+1: " + ", ".join(
        f"{l}:{'same' if np.allclose(before[l], appended[l]) else 'CHANGED'}"
        for l in (0, 1)))
    print("line colours with candidates = -2:    " + ", ".join(
        f"{l}:{'same' if np.allclose(before[l], prepended[l]) else 'CHANGED'}"
        for l in (0, 1)))

    for label in (0, 1):
        assert np.allclose(before[label], appended[label]), \
            f"line {label} changed colour when the candidate cluster appeared"
    assert not np.allclose(before[0], prepended[0]), \
        "setup wrong: a label sorting first was expected to shift the colours"


def test_labelling_candidates_makes_them_selectable():
    """The regression this whole feature turns on. Unclaimed points are -1 and
    the viewer refuses them, so nothing can be picked to extend into. Giving the
    candidates a label of their own is what lifts that."""
    labels = np.array([0, 0, -1, -1, -1])
    _install_branch(labels)
    viewer = _StubViewer(len(labels))

    before = [viewer._is_point_selectable(i) for i in range(len(labels))]
    assert before == [True, True, False, False, False], \
        f"unclaimed points were expected to be unselectable, got {before}"

    labels[3] = 7                       # promoted to the candidate cluster
    after = [viewer._is_point_selectable(i) for i in range(len(labels))]
    kept = viewer._filter_selection(np.arange(len(labels)))
    print(f"selectable before: {before}\n"
          f"selectable after:  {after}\npolygon select keeps: {kept}")

    assert after[3], "the candidate point is still refused"
    assert not after[2] and not after[4], \
        "labelling one point made its noise neighbours selectable too"
    assert set(kept.tolist()) == {0, 1, 3}, \
        f"polygon select disagrees with click select: {kept}"


def test_a_cluster_lock_still_wins():
    """A lock is the user's own statement about a cluster. Being made a candidate
    must not quietly unlock it."""
    labels = np.array([0, 0, 7, -1])
    _install_branch(labels, locked={7: {"select"}})
    viewer = _StubViewer(len(labels))

    print(f"locked candidate selectable: {viewer._is_point_selectable(2)}, "
          f"polygon keeps {viewer._filter_selection(np.arange(4))}")
    assert not viewer._is_point_selectable(2), \
        "a cluster locked against selection was selectable as a candidate"
    assert 2 not in viewer._filter_selection(np.arange(4)).tolist(), \
        "polygon select ignored the lock"


def test_labels_are_read_through_the_lod_subsample():
    """Under LOD the viewer holds points[indices], so rendered row k is cloud row
    indices[k]. Reading labels[k] instead consults an unrelated point, and the
    viewer ends up deciding what may be picked from the wrong data."""
    # 10 cloud points; only the candidate at cloud row 6 should be pickable.
    labels = np.full(10, -1, dtype=np.int64)
    labels[6] = 7
    _install_branch(labels)

    sample = np.array([0, 2, 4, 6, 8])          # every second point rendered
    viewer = _StubViewer(len(sample), sample_indices=sample)

    selectable = [i for i in range(len(sample)) if viewer._is_point_selectable(i)]
    print(f"rendered rows {list(range(len(sample)))} -> cloud rows "
          f"{sample.tolist()}; selectable rendered rows: {selectable}")
    assert selectable == [3], \
        f"expected only rendered row 3 (cloud row 6) to be pickable, got {selectable}"
    assert viewer.cloud_index(_UID, 3) == 6
    assert viewer.cloud_indices(_UID, np.arange(5)).tolist() == sample.tolist()


def test_emphasis_is_resolved_through_the_lod_subsample():
    """Emphasis is given in SOURCE rows — the space cluster labels live in — and
    has to be translated to rendered rows to be drawn. Storing rendered rows
    instead would rot the moment LOD changed the subsample, quietly enlarging
    and fading whichever points had inherited those row numbers."""
    sample = np.array([0, 2, 4, 6, 8])           # every second cloud point drawn
    viewer = _StubViewer(len(sample), sample_indices=sample)

    # Cloud rows 4 and 6 are offered; 0 and 8 are clutter; 2 is neither. Cloud
    # row 5 is offered as well but was not drawn at all, so it cannot appear.
    viewer.set_point_emphasis(_UID, emphasised=[4, 5, 6], faded=[0, 6, 8])
    emphasised, opaque = viewer._emphasis_draw_groups(_UID, len(sample))
    print(f"cloud rows drawn {sample.tolist()}: emphasised rendered "
          f"{emphasised.tolist()}, opaque {opaque.tolist()} (the faded ones — "
          f"rendered rows 0 and 4 — are covered by the whole-branch pass)")

    assert emphasised.tolist() == [2, 3], \
        f"cloud rows 4 and 6 are rendered rows 2 and 3, got {emphasised.tolist()}"
    assert opaque.tolist() == [1], \
        f"only rendered row 1 is neither offered nor clutter, got {opaque.tolist()}"
    assert 3 in emphasised.tolist() and 3 not in opaque.tolist(), \
        "a row named as both must come out emphasised"

    viewer.set_point_emphasis(_UID)
    assert viewer._emphasis_draw_groups(_UID, len(sample)) is None, \
        "clearing the emphasis must put the branch back to a single draw pass"


def test_offering_and_withdrawing_leaves_the_branch_as_it_was():
    """Stepping between stops re-offers a different set of points without
    rewriting the whole branch, so withdrawing an offer has to restore exactly
    what was there — labels AND colours. Anything left behind would show up as a
    stray cluster in the tree and, eventually, in a DXF layer.

    Calls the window's own methods unbound, so the test binds to the real code
    rather than a copy of it (constructing the dialog would need a QApplication).
    """
    from plugins.dialogs.line_extension_window import LineExtensionWindow as W

    labels = np.array([0, 0, -1, -1, -1, -1], dtype=np.int32)
    clusters = Clusters(labels=labels.copy())
    clusters.set_random_color()
    before_labels = clusters.labels.copy()
    before_colors = clusters.colors.copy()

    offered = np.array([3, 5], dtype=np.intp)
    window = types.SimpleNamespace(
        marked_indices=None,
        marked_label=None,
        _candidate_indices=lambda: offered,
    )
    # Bind the real helper the marking methods call, so this still exercises the
    # window's own code rather than a copy of it.
    window._forget_candidate_naming = lambda c: W._forget_candidate_naming(window, c)

    W._mark_candidates(window, clusters)
    marked = clusters.labels[offered].copy()
    print(f"offered {offered.tolist()} -> label {marked.tolist()}, "
          f"colour {clusters.colors[offered][0].tolist()}")
    assert set(marked.tolist()) == {1}, \
        f"candidates should take one label above the highest in use, got {marked}"
    assert np.allclose(clusters.colors[offered], [1.0, 1.0, 0.0]), \
        "candidates were not painted the candidate colour"

    W._unmark_candidates(window, clusters)
    print(f"after withdrawing: labels {clusters.labels.tolist()}, "
          f"marked_indices={window.marked_indices}")
    assert np.array_equal(clusters.labels, before_labels), \
        "withdrawing the offer left stray labels behind"
    assert np.allclose(clusters.colors, before_colors), \
        "withdrawing the offer left the candidate colour behind"
    assert window.marked_indices is None


def test_full_resolution_still_maps_one_to_one():
    """With no subsample the rendered row IS the cloud row — the translation must
    not disturb the ordinary case."""
    labels = np.array([-1, 7, -1])
    _install_branch(labels)
    viewer = _StubViewer(len(labels))          # sample_indices None

    assert viewer.cloud_index(_UID, 2) == 2
    assert [viewer._is_point_selectable(i) for i in range(3)] == [False, True, False]
    print("no LOD: rendered rows map to themselves")


if __name__ == "__main__":
    test_candidate_label_leaves_other_clusters_alone()
    test_labelling_candidates_makes_them_selectable()
    test_a_cluster_lock_still_wins()
    test_labels_are_read_through_the_lod_subsample()
    test_emphasis_is_resolved_through_the_lod_subsample()
    test_offering_and_withdrawing_leaves_the_branch_as_it_was()
    test_full_resolution_still_maps_one_to_one()
    print("\nAll pick-focus label tests passed.")
