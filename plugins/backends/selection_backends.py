"""
Screen-Space Selection Backend Implementations

GPU (CuPy) and CPU (NumPy) backends for testing which points fall inside a
polygon drawn on screen. Both receive one block of points at a time from
``core.services.screen_selection``; neither decides how the cloud is split up.

The GPU backend is the one that matters. Projecting and testing a point is a
handful of multiply-adds, so an array-at-a-time implementation spends its time
moving intermediates rather than computing — every stage writes a full-size
temporary that the next stage reads back. Fusing the whole thing into one
kernel, so a point is loaded once and produces its answer before the next point
starts, measured 44x faster than the array-at-a-time code on a 20-corner lasso
over 20M points (5.70 s to 0.13 s).
"""

import logging
import numpy as np
from .base import ScreenSelectionBackend

logger = logging.getLogger(__name__)

# The fused CUDA kernel. Compiled once, on first use, then cached by CuPy.
#
# `raw` on every parameter is required to pass an explicit `size=`, which is
# what lets one thread handle one point of an interleaved (N, 6) buffer rather
# than one float of it.
#
# The crossing rule is deliberately identical to the NumPy version below,
# including the `>=` comparisons and the skipping of horizontal edges, so the
# two backends agree point for point rather than merely almost.
_POLYGON_KERNEL_SOURCE = r'''
    const int base = i * stride;
    const float u = pts[base];
    const float v = pts[base + 1];
    const float t = pts[base + 2];

    const float w = u * c[8] + v * c[9] + t * c[10] + c[11];
    bool result = false;

    if (w > 0.0f) {
        const float inv = 1.0f / w;
        const float sx = (u * c[0] + v * c[1] + t * c[2] + c[3]) * inv + c[12];
        const float sy = c[13] - (u * c[4] + v * c[5] + t * c[6] + c[7]) * inv;

        // Reject against the polygon's bounding box first. A lasso covers a
        // fraction of the screen, so this discards most points for four
        // comparisons instead of running the full edge loop on them.
        if (sx >= bminx && sx <= bmaxx && sy >= bminy && sy <= bmaxy) {
            int crossings = 0;
            for (int k = 0; k < m; ++k) {
                const int kn = (k + 1 == m) ? 0 : k + 1;
                const float x1 = poly[2 * k],  y1 = poly[2 * k + 1];
                const float x2 = poly[2 * kn], y2 = poly[2 * kn + 1];
                const float dy = y2 - y1;
                if (dy != 0.0f) {
                    const bool above1 = (sy >= y1);
                    const bool above2 = (sy >= y2);
                    if (above1 != above2) {
                        const float xint = x1 + (sy - y1) / dy * (x2 - x1);
                        if (xint > sx) ++crossings;
                    }
                }
            }
            result = (crossings & 1) != 0;
        }
    }
    inside[i] = result;
'''

_kernel = None


def _polygon_kernel():
    """The compiled kernel, built on first use so importing needs no GPU."""
    global _kernel
    if _kernel is None:
        import cupy as cp
        _kernel = cp.ElementwiseKernel(
            in_params=('raw float32 pts, int32 stride, raw float32 c, '
                       'raw float32 poly, int32 m, '
                       'float32 bminx, float32 bmaxx, '
                       'float32 bminy, float32 bmaxy'),
            out_params='raw bool inside',
            operation=_POLYGON_KERNEL_SOURCE,
            name='points_in_polygon',
        )
    return _kernel


class CuPySelection(ScreenSelectionBackend):
    """GPU screen-space selection using a single fused CuPy kernel."""

    @property
    def name(self) -> str:
        return "CuPy"

    @property
    def is_gpu(self) -> bool:
        return True

    @staticmethod
    def _required_mb(block) -> int:
        """GPU memory for one block: the points, plus a byte per point out."""
        return int(block.nbytes // (1024 * 1024)) + int(len(block) // (1024 * 1024)) + 8

    def points_in_polygon(self, block, coeffs, polygon, bounds, out):
        """Test one block of points; see ScreenSelectionBackend."""
        from infrastructure.memory_manager import MemoryManager

        required_mb = self._required_mb(block)
        if not MemoryManager.can_use_gpu(required_mb):
            logger.info(f"Selection falling back to CPU (needs {required_mb} MB VRAM)")
            return NumpySelection().points_in_polygon(
                block, coeffs, polygon, bounds, out)

        import cupy as cp
        try:
            block = np.ascontiguousarray(block, dtype=np.float32)
            pts_gpu = cp.asarray(block)
            coeffs_gpu = cp.asarray(coeffs)
            poly_gpu = cp.asarray(np.ascontiguousarray(polygon, dtype=np.float32))
            inside_gpu = cp.empty(len(block), dtype=cp.bool_)

            _polygon_kernel()(
                pts_gpu, np.int32(block.shape[1]), coeffs_gpu,
                poly_gpu, np.int32(len(polygon)),
                np.float32(bounds[0]), np.float32(bounds[1]),
                np.float32(bounds[2]), np.float32(bounds[3]),
                inside_gpu, size=len(block),
            )

            out[:] = cp.asnumpy(inside_gpu)
            del pts_gpu, coeffs_gpu, poly_gpu, inside_gpu
            return out

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM during selection: {e}, falling back to CPU")
            MemoryManager.cleanup()
            return NumpySelection().points_in_polygon(
                block, coeffs, polygon, bounds, out)


class NumpySelection(ScreenSelectionBackend):
    """CPU screen-space selection using NumPy.

    For machines with no NVIDIA card or no CuPy. Still far better than the code
    it replaces, because it works in float32 and rejects against the polygon's
    bounding box before running the edge loop — on a typical lasso that leaves
    the loop with a small fraction of the block.
    """

    @property
    def name(self) -> str:
        return "NumPy"

    @property
    def is_gpu(self) -> bool:
        return False

    def points_in_polygon(self, block, coeffs, polygon, bounds, out):
        """Test one block of points; see ScreenSelectionBackend."""
        from core.services.screen_selection import project_block

        screen_x, screen_y, valid = project_block(
            np.asarray(block, dtype=np.float32), coeffs)

        # Bounding-box reject, folded into the same pass as the camera test.
        min_x, max_x, min_y, max_y = bounds
        np.logical_and(valid, screen_x >= min_x, out=valid)
        np.logical_and(valid, screen_x <= max_x, out=valid)
        np.logical_and(valid, screen_y >= min_y, out=valid)
        np.logical_and(valid, screen_y <= max_y, out=valid)

        candidates = np.flatnonzero(valid)
        out[:] = False
        if candidates.size == 0:
            return out

        # Only the survivors reach the edge loop, so its scratch arrays are
        # sized by what is near the lasso rather than by the whole block.
        cand_x = screen_x[candidates]
        cand_y = screen_y[candidates]
        crossings = np.zeros(candidates.size, dtype=np.int32)
        above_1 = np.empty(candidates.size, dtype=bool)
        above_2 = np.empty(candidates.size, dtype=bool)

        poly = np.asarray(polygon, dtype=np.float32)
        m = len(poly)
        for k in range(m):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % m]
            dy = y2 - y1
            if dy == 0:
                continue

            np.greater_equal(cand_y, y1, out=above_1)
            np.greater_equal(cand_y, y2, out=above_2)
            np.not_equal(above_1, above_2, out=above_1)     # straddles the ray
            if not above_1.any():
                continue

            straddling = above_1
            x_intersect = x1 + (cand_y[straddling] - y1) / dy * (x2 - x1)
            crossings[straddling] += x_intersect > cand_x[straddling]

        out[candidates] = (crossings & 1) != 0
        return out
