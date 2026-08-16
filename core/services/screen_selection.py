"""
Screen-Space Selection

One implementation of "which of these 3D points land inside this shape on
screen". Used by the lasso (select and deselect), by the full-resolution
re-test plugins run, and by zoom-to-window.

Two things make it fast enough for 100M+ point clouds:

**It works in blocks.** The old code ran each stage over the whole cloud before
starting the next — cast every point to float64, then project every point, then
test every point — so every stage's scratch array was alive at once. At 170M
that came to 19 GB of scratch against 4 GB of actual points, which does not fit
in RAM and swaps. Here a block of ~8M points is carried all the way to a yes/no
before the next block starts, so scratch stays around 200 MB whatever the cloud
size. Only the answer grows with the cloud, and that is one byte per point.

**It works in float32.** Screen positions need pixel accuracy, not 16 digits.
Measured over 20M points, the float32 answer differed from float64 on a single
point, which sat 1.7e-06 pixels from the lasso boundary.

float32 is safe here because the project shifts clouds to the origin on import:
every importer subtracts ``min_bound`` and records it as
``PointCloud.translation`` (see the LAS/e57/npy/semantickitti importers and
``services/file_manager``), and export and Identify Point add it back through
``find_root_translation``. Coordinates therefore reach the viewer starting at
zero, never at survey magnitude, and nothing here has to defend against a
cancellation that cannot happen.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Points carried through the pipeline at once. Sized so the scratch arrays stay
# in the hundreds of MB rather than scaling with the cloud.
DEFAULT_BLOCK = 8_000_000

# Layout of the array returned by screen_coeffs(). Kept as one flat float32
# array because it is passed straight into a CUDA kernel as scalars.
#   0- 3  ax bx cx dx   numerator of screen X (viewport scale folded in)
#   4- 7  ay by cy dy   numerator of screen Y (viewport scale folded in)
#   8-11  aw bw cw dw   clip W (depth); W <= 0 means behind the camera
#  12-13  tx ty         viewport centre offset
N_COEFFS = 14


def screen_coeffs(mv, proj, viewport) -> np.ndarray:
    """Fold model-view, projection and viewport into coefficients for float32 use.

    The viewer's projection is::

        clip   = [x, y, z, 1] @ (mv @ proj)
        ndc    = clip[:2] / clip[3]
        screen = ndc * half_viewport + viewport_centre     (Y flipped)

    Written that way it needs a 4x4 matrix product per point and a float64
    intermediate. Collapsing the two matrices and folding in the viewport
    leaves nine multiply-adds per point on small float32 numbers.

    Args:
        mv: 4x4 model-view matrix as OpenGL reports it (column-major, so it
            appears transposed in numpy — hence row-vector multiplication).
        proj: 4x4 projection matrix, same convention.
        viewport: (vp_x, vp_y, vp_w, vp_h).

    Returns:
        (14,) float32 array laid out as described in the module constants.
    """
    # Everything here is 4x4 or smaller, so collapsing it in float64 costs
    # nothing and keeps the coefficients exact.
    m = np.asarray(mv, dtype=np.float64) @ np.asarray(proj, dtype=np.float64)

    vp_x, vp_y, vp_w, vp_h = (float(v) for v in viewport)
    half_w, half_h = 0.5 * vp_w, 0.5 * vp_h

    out = np.empty(N_COEFFS, dtype=np.float32)
    for slot, (col, scale) in enumerate(((0, half_w), (1, half_h), (3, 1.0))):
        column = np.array([m[0, col], m[1, col], m[2, col], m[3, col]])
        out[slot * 4: slot * 4 + 4] = column * scale

    out[12] = half_w + vp_x
    out[13] = half_h + vp_y
    return out


def project_block(block, coeffs):
    """Screen X/Y and an in-front-of-camera mask for one block of points.

    Args:
        block: (N, >=3) float32 array; only the first three columns are read.
        coeffs: the (14,) array from ``screen_coeffs``.

    Returns:
        (screen_x, screen_y, valid) — float32, float32, bool, each (N,).
        Screen coordinates are Qt widget coordinates: top-left origin, Y down,
        the same space ``QMouseEvent.pos()`` is in. Entries where *valid* is
        False hold no meaningful position.
    """
    c = coeffs
    u = block[:, 0]
    v = block[:, 1]
    t = block[:, 2]

    w = u * c[8] + v * c[9] + t * c[10] + c[11]
    valid = w > 0
    # Division by a non-positive w is meaningless, and dividing by exactly zero
    # would raise. Substitute a harmless denominator and let `valid` mask the
    # results out; np.where allocates far less than a masked divide would.
    safe_w = np.where(valid, w, np.float32(1.0))

    screen_x = (u * c[0] + v * c[1] + t * c[2] + c[3]) / safe_w + c[12]
    screen_y = c[13] - (u * c[4] + v * c[5] + t * c[6] + c[7]) / safe_w
    return screen_x, screen_y, valid


def polygon_bounds(polygon):
    """(min_x, max_x, min_y, max_y) of a polygon, as float32."""
    poly = np.asarray(polygon, dtype=np.float32)
    return (float(poly[:, 0].min()), float(poly[:, 0].max()),
            float(poly[:, 1].min()), float(poly[:, 1].max()))


def _resolve_backend(backend):
    """The selection backend to use: the caller's, else the registry's."""
    if backend is not None:
        return backend
    try:
        from config.config import global_variables
        registry = global_variables.global_backend_registry
        if registry is not None:
            return registry.get_selection()
    except Exception as exc:                       # registry not up yet (tests)
        logger.debug(f"Selection backend registry unavailable ({exc}); using NumPy")
    from plugins.backends.selection_backends import NumpySelection
    return NumpySelection()


def select_in_polygon(points, polygon, mv, proj, viewport,
                      block=DEFAULT_BLOCK, backend=None):
    """Boolean mask over *points* marking those inside *polygon* on screen.

    Args:
        points: (N, >=3) array of world coordinates. Only xyz is read, so the
            viewer's interleaved xyz+rgb buffer can be passed straight in.
        polygon: (M, 2) screen-space vertices in Qt widget coordinates.
        mv, proj, viewport: the camera the polygon was drawn against.
        block: points carried through at once.
        backend: override the registry's choice (used by the tests to compare
            the CPU and GPU paths against each other).

    Returns:
        (N,) boolean mask. Points behind the camera are never selected.
    """
    points = np.asarray(points)
    n = len(points)
    out = np.zeros(n, dtype=bool)
    poly = np.asarray(polygon, dtype=np.float32)
    if n == 0 or len(poly) < 3:
        return out

    coeffs = screen_coeffs(mv, proj, viewport)
    bounds = polygon_bounds(poly)
    impl = _resolve_backend(backend)

    for start in range(0, n, block):
        stop = min(start + block, n)
        impl.points_in_polygon(points[start:stop], coeffs, poly, bounds,
                               out[start:stop])
    return out


def select_in_rect(points, rect, mv, proj, viewport,
                   block=DEFAULT_BLOCK, backend=None):
    """Boolean mask over *points* marking those inside a screen rectangle.

    A rectangle is a four-sided polygon, so this shares every part of the
    polygon path rather than repeating the projection with its own inline test.

    Args:
        rect: (left, top, right, bottom) in Qt widget coordinates.
        Other arguments as ``select_in_polygon``.
    """
    left, top, right, bottom = (float(v) for v in rect)
    polygon = np.array([(left, top), (right, top),
                        (right, bottom), (left, bottom)], dtype=np.float32)
    return select_in_polygon(points, polygon, mv, proj, viewport,
                             block=block, backend=backend)
