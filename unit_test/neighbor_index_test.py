"""
Tests for core.services.neighbor_index.

The whole point of this class is that an algorithm can stop building a KD-tree
and notice no difference in its answers, so ``cKDTree`` is the oracle for
almost every test here: same indices, same distances, same shapes, same
behaviour at the edges. Anywhere the two disagree, the algorithm above them
would grow a different feature, and that is not a trade anyone agreed to.

Run:
    python unit_test/neighbor_index_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

from core.services.neighbor_index import NeighborIndex

_RNG = np.random.default_rng(11)


def _uniform(n=20_000, extent=10.0):
    """Evenly spread points — the easy case, and the one nothing may break."""
    return (_RNG.random((n, 3)) * extent).astype(np.float32)


def _clustered(n=20_000):
    """Blobs on a lattice. Real survey data is clustered, not uniform, and
    clustering is what makes a uniform grid work hardest: most cells are empty
    and the occupied ones hold far more than the average suggests."""
    centres = _RNG.integers(0, 5, (n, 3)) * 3.0
    return (centres + _RNG.normal(0, 0.4, (n, 3))).astype(np.float32)


def _flat(n=5_000):
    """A single plane. One axis has no extent at all, which is the shape that
    used to divide by zero and drop every point into cell 0."""
    return np.column_stack([
        _RNG.random(n) * 10.0, _RNG.random(n) * 10.0, np.zeros(n),
    ]).astype(np.float32)


CLOUDS = [("uniform", _uniform()), ("clustered", _clustered()),
          ("flat", _flat())]


def test_ball_queries_match_a_kdtree():
    """``query_ball_point`` returns exactly what ``cKDTree`` returns.

    Radii spanning well under a cell to well over one, so the answer comes from
    a single cell, from a column, and from a box of many — the three shapes the
    lookup has separate code for.
    """
    for name, points in CLOUDS:
        tree = cKDTree(points)
        index = NeighborIndex(points)
        centres = points[_RNG.integers(0, len(points), 150)]

        biggest = 0
        for centre in centres:
            for radius in (0.01, 0.05, 0.3, 1.0, 4.0):
                want = np.sort(np.asarray(tree.query_ball_point(centre, radius),
                                          dtype=np.intp))
                got = np.sort(index.query_ball_point(centre, radius))
                assert np.array_equal(want, got), (name, centre, radius)
                biggest = max(biggest, want.size)
        print(f"  {name}: 750 ball queries match the tree exactly "
              f"(largest {biggest:,} points)")


def test_nearest_queries_match_a_kdtree():
    """``query`` matches ``cKDTree`` in distances, indices and shapes.

    Both call shapes: one point in gives scalars at k=1 and a (k,) pair
    otherwise; a batch gives (m,) or (m, k). The growers rely on the single-point
    form and ``selection_gate`` on the batch, so neither may drift.
    """
    for name, points in CLOUDS:
        tree = cKDTree(points)
        index = NeighborIndex(points)
        singles = points[_RNG.integers(0, len(points), 100)].astype(np.float64)

        for centre in singles:
            for k in (1, 2, 17):
                want_d, want_i = tree.query(centre, k=k)
                got_d, got_i = index.query(centre, k=k)
                assert np.shape(want_d) == np.shape(got_d), (name, k)
                assert np.allclose(want_d, got_d), (name, k, centre)
                assert np.array_equal(np.atleast_1d(want_i),
                                      np.atleast_1d(got_i)), (name, k, centre)

        batch = (_RNG.random((300, 3)) * 10.0).astype(np.float64)
        for k in (1, 4):
            want_d, want_i = tree.query(batch, k=k)
            got_d, got_i = index.query(batch, k=k)
            assert np.shape(want_d) == np.shape(got_d), (name, k)
            assert np.allclose(want_d, got_d), (name, k)
            assert np.array_equal(want_i, got_i), (name, k)
        print(f"  {name}: single and batched k-NN match the tree, k = 1, 2, 4, 17")


def test_a_missing_neighbour_is_reported_the_way_a_tree_reports_it():
    """Asking for more neighbours than exist, or capping the distance.

    ``cKDTree`` answers both with distance ``inf`` and index ``n``, and the
    grower's breadth-first walk rejects a neighbour with exactly ``j >= n``. Get
    this wrong and it reads past the end of its own arrays.
    """
    points = _uniform(n=5)
    tree, index = cKDTree(points), NeighborIndex(points)

    want_d, want_i = tree.query(points[0], k=9)
    got_d, got_i = index.query(points[0], k=9)
    assert np.allclose(want_d, got_d)
    assert np.array_equal(want_i, got_i)
    assert got_i[-1] == index.n, "a missing neighbour must be reported as n"

    batch = (_RNG.random((50, 3)) * 10.0).astype(np.float64)
    want_d, want_i = tree.query(batch, k=3, distance_upper_bound=0.5)
    got_d, got_i = index.query(batch, k=3, distance_upper_bound=0.5)
    assert np.allclose(want_d, got_d)
    assert np.array_equal(want_i, got_i)
    print("  too-few neighbours and a distance cap both answer like the tree")


def test_the_ordered_copy_holds_the_same_points():
    """The cell-ordered copy must be the cloud, rearranged and nothing else.

    It is what every query actually reads — the original array is never touched
    after the build — so a wrong row here would be a wrong answer everywhere,
    and one that no distance test could catch because the distances would all
    be self-consistent.
    """
    points = _clustered(n=5_000)
    index = NeighborIndex(points)

    assert index._xyz.shape == (len(points), 3)
    assert index._xyz.dtype == np.float32
    assert np.array_equal(index._xyz, points[index.grid.order, :3])

    # And the pairing that matters: row i of the copy is cloud row order[i].
    sample = _RNG.integers(0, len(points), 200)
    assert np.array_equal(index._xyz[sample],
                          points[index.grid.order[sample], :3])
    print(f"  the ordered copy is the cloud rearranged "
          f"({index._xyz.nbytes / 2 ** 20:.1f} MB at {len(points):,} points)")


def test_cell_sizing_is_a_choice_and_a_ceiling():
    """``points_per_cell``, ``cell_size`` and ``max_cells`` do what they say.

    The ceiling is the one that matters in anger: a query radius that is tiny
    against the site would otherwise divide the bounding box into an offset
    table larger than the cloud.
    """
    points = _uniform(n=40_000, extent=100.0)

    coarse = NeighborIndex(points, points_per_cell=1_000)
    fine = NeighborIndex(points, points_per_cell=4)
    assert coarse.grid.n_cells < fine.grid.n_cells
    assert coarse.grid.n_cells <= 40_000 // 1_000

    sized = NeighborIndex(points, cell_size=5.0)
    assert np.all(sized.grid.step <= 5.0 + 1e-6), sized.grid.step

    capped = NeighborIndex(points, cell_size=0.001, max_cells=10_000)
    assert capped.grid.n_cells <= 10_000, capped.grid.n_cells

    # However it was sized, it still answers correctly.
    tree = cKDTree(points)
    for index in (coarse, fine, sized, capped):
        for centre in points[_RNG.integers(0, len(points), 20)]:
            want = np.sort(np.asarray(tree.query_ball_point(centre, 2.0),
                                      dtype=np.intp))
            assert np.array_equal(want, np.sort(index.query_ball_point(centre, 2.0)))
    print(f"  sizing: {coarse.grid.n_cells:,} / {fine.grid.n_cells:,} cells, "
          f"cell_size honoured, ceiling held at {capped.grid.n_cells:,}")


def test_degenerate_clouds():
    """Empty, single, and every-point-identical.

    An empty cloud has no grid at all, and both queries have to answer rather
    than raise — a grower handed an empty branch should stop, not crash.
    """
    empty = NeighborIndex(np.empty((0, 3), dtype=np.float32))
    assert empty.grid is None
    assert empty.query_ball_point([0, 0, 0], 1.0).size == 0
    assert empty.query([0, 0, 0]) == (np.inf, 0)
    distances, indices = empty.query(np.zeros((2, 3)), k=3)
    assert np.all(np.isinf(distances)) and np.all(indices == 0)

    one = NeighborIndex(np.zeros((1, 3), dtype=np.float32))
    assert one.query_ball_point([0, 0, 0], 1.0).tolist() == [0]
    assert one.query([0, 0, 0]) == (0.0, 0)

    same = NeighborIndex(np.zeros((100, 3), dtype=np.float32))
    assert same.query_ball_point([0, 0, 0], 0.1).size == 100
    distances, indices = same.query([0, 0, 0], k=3)
    assert np.allclose(distances, 0.0) and len(set(indices.tolist())) == 3
    print("  empty, single and all-identical clouds answer without raising")


def test_a_query_outside_the_cloud_still_finds_the_nearest():
    """Far outside the bounding box, in every direction.

    The home cell is clamped to the edge of the grid, so the widening has to
    keep going until it can prove nothing nearer exists — and the proof has to
    account for a box that has already run off the side.
    """
    points = _clustered(n=8_000)
    tree, index = cKDTree(points), NeighborIndex(points)

    far = np.array([
        [-500.0, 0.0, 0.0], [500.0, 0.0, 0.0], [0.0, -500.0, 0.0],
        [0.0, 500.0, 0.0], [0.0, 0.0, -500.0], [0.0, 0.0, 500.0],
        [-500.0, -500.0, -500.0], [500.0, 500.0, 500.0],
    ])
    for centre in far:
        want_d, want_i = tree.query(centre, k=3)
        got_d, got_i = index.query(centre, k=3)
        assert np.allclose(want_d, got_d), centre
        assert np.array_equal(want_i, got_i), centre
        assert index.query_ball_point(centre, 1.0).size == 0
    print("  8 queries far outside the cloud find the same nearest as the tree")


def test_non_finite_coordinates_are_never_returned():
    """One NaN point must not become everybody's nearest neighbour.

    A NaN distance sorts as small under ``argpartition`` rather than being
    skipped, so without care the bad point wins every query and the real
    neighbour is never seen.
    """
    points = _uniform(n=2_000)
    points[7] = np.nan
    points[9] = np.inf

    index = NeighborIndex(points)
    for centre in points[_RNG.integers(0, len(points), 50)]:
        if not np.all(np.isfinite(centre)):
            continue
        assert 7 not in index.query_ball_point(centre, 3.0).tolist()
        assert 9 not in index.query_ball_point(centre, 3.0).tolist()
        _distances, indices = index.query(centre, k=5)
        assert 7 not in indices.tolist() and 9 not in indices.tolist()
    print("  a NaN and an inf point are never returned by either query")


def test_the_cpu_and_gpu_builds_answer_identically():
    """Whichever backend numbers the cells, the answers are the same.

    Skipped without CuPy, which is normal on a machine with no NVIDIA card.
    """
    try:
        import cupy
        cupy.zeros(1)
    except Exception as exc:
        print(f"  skipped, no working CuPy ({type(exc).__name__})")
        return

    from plugins.backends.grid_backends import CuPyGrid, NumpyGrid

    points = _clustered(n=30_000)
    on_cpu = NeighborIndex(points, backend=NumpyGrid())
    on_gpu = NeighborIndex(points, backend=CuPyGrid())
    assert on_cpu.grid.shape == on_gpu.grid.shape

    for centre in points[_RNG.integers(0, len(points), 100)]:
        assert np.array_equal(np.sort(on_cpu.query_ball_point(centre, 0.5)),
                              np.sort(on_gpu.query_ball_point(centre, 0.5)))
        cpu_d, cpu_i = on_cpu.query(centre, k=9)
        gpu_d, gpu_i = on_gpu.query(centre, k=9)
        assert np.allclose(cpu_d, gpu_d) and np.array_equal(cpu_i, gpu_i)
    print("  CPU and GPU builds answer 100 queries identically")


if __name__ == "__main__":
    test_ball_queries_match_a_kdtree()
    test_nearest_queries_match_a_kdtree()
    test_a_missing_neighbour_is_reported_the_way_a_tree_reports_it()
    test_the_ordered_copy_holds_the_same_points()
    test_cell_sizing_is_a_choice_and_a_ceiling()
    test_degenerate_clouds()
    test_a_query_outside_the_cloud_still_finds_the_nearest()
    test_non_finite_coordinates_are_never_returned()
    test_the_cpu_and_gpu_builds_answer_identically()
    print("\nAll neighbour index tests passed.")
