"""
Spatial Grid Backend Implementations

GPU (CuPy) and CPU (NumPy) backends for numbering points by the grid cell they
fall in. Both receive one block of points at a time from
``core.services.spatial_grid``; neither decides how the cloud is split up.

This used to live on ``ScreenSelectionBackend`` in ``selection_backends.py``,
because the viewer's pick grid was the first thing that needed it. It is not a
screen operation — it is a spatial index any service can build — so it now has
its own backend kind and its own file, and the viewer is just one caller.

Cell width follows the *out* array's dtype: ``uint8`` for a grid of at most 256
cells (the viewer's, one byte per point) and ``int32`` for anything finer (what
an algorithm wants, where cells are sized to the query radius rather than to the
whole cloud). Both kernels are compiled on first use and cached by CuPy.
"""

import logging
import numpy as np

from .base import SpatialGridBackend

logger = logging.getLogger(__name__)

# One thread per point. `raw` on every parameter is required to pass an explicit
# `size=`, which is what lets a thread handle one point of an interleaved (N, 6)
# buffer rather than one float of it.
#
# %(out_type)s is the only thing that differs between the two compiled versions.
_CELL_KERNEL_SOURCE = r'''
    const int base = i * stride;
    const float fx = (pts[base]     - lox) * invx;
    const float fy = (pts[base + 1] - loy) * invy;
    const float fz = (pts[base + 2] - loz) * invz;
    // NaN converts to an implementation-defined int, which would put the GPU
    // and CPU paths in different cells. Send it to cell 0 explicitly, matching
    // _cell_ids_numpy. Every comparison with NaN is false, so isnan() is the
    // only way to catch it.
    int ix = isnan(fx) ? 0 : (int)fx;
    int iy = isnan(fy) ? 0 : (int)fy;
    int iz = isnan(fz) ? 0 : (int)fz;
    ix = ix < 0 ? 0 : (ix >= nx ? nx - 1 : ix);
    iy = iy < 0 ? 0 : (iy >= ny ? ny - 1 : iy);
    iz = iz < 0 ? 0 : (iz >= nz ? nz - 1 : iz);
    out[i] = (%(out_type)s)((ix * ny + iy) * nz + iz);
'''

# out dtype -> (CUDA out_params type, C cast). Anything not listed is refused
# rather than silently truncated.
_OUT_TYPES = {
    np.dtype(np.uint8): ("uint8", "unsigned char"),
    np.dtype(np.uint16): ("uint16", "unsigned short"),
    np.dtype(np.int32): ("int32", "int"),
}

_cell_kernels = {}


def _cell_id_kernel(out_dtype):
    """The compiled cell-numbering kernel for *out_dtype*, built on first use."""
    key = np.dtype(out_dtype)
    if key not in _cell_kernels:
        cuda_type, cast = _OUT_TYPES[key]
        import cupy as cp
        _cell_kernels[key] = cp.ElementwiseKernel(
            in_params=('raw float32 pts, int32 stride, '
                       'float32 lox, float32 loy, float32 loz, '
                       'float32 invx, float32 invy, float32 invz, '
                       'int32 nx, int32 ny, int32 nz'),
            out_params=f'raw {cuda_type} out',
            operation=_CELL_KERNEL_SOURCE % {"out_type": cast},
            name=f'spatial_grid_cell_ids_{cuda_type}',
        )
    return _cell_kernels[key]


def _cell_ids_numpy(block, lo, inv_step, shape, out):
    """Shared CPU implementation — also the GPU backend's fallback.

    Multiplies by the reciprocal cell size rather than dividing, and reuses two
    scratch buffers across the three axes, so a block costs three passes and no
    repeated allocation. Accumulates in int32 and narrows on the way out, so the
    same code serves a uint8 grid and a wider one.
    """
    n_pts = len(block)
    idx = np.zeros(n_pts, dtype=np.int32)
    cell_f = np.empty(n_pts, dtype=np.float32)
    cell_i = np.empty(n_pts, dtype=np.int32)

    for axis, n_cells in enumerate(int(v) for v in shape):
        np.subtract(block[:, axis], lo[axis], out=cell_f)
        np.multiply(cell_f, inv_step[axis], out=cell_f)
        np.floor(cell_f, out=cell_f)
        # Clamp: a point exactly on the far face of the bounding box would
        # otherwise index one cell past the end of this axis. This also folds
        # +/-inf into the edge cells.
        np.clip(cell_f, 0, n_cells - 1, out=cell_f)
        # NaN survives both floor and clip, and casting it to int is undefined —
        # it warns on the CPU and would differ from the GPU. Send those points
        # to cell 0, which is where the plain scan effectively left them: never
        # the nearest point to anything, because every comparison with NaN is
        # False.
        np.nan_to_num(cell_f, copy=False, nan=0.0)
        np.copyto(cell_i, cell_f, casting='unsafe')
        idx *= n_cells
        idx += cell_i

    np.copyto(out, idx, casting='unsafe')
    return out


def _block_bounds_numpy(block):
    """Shared CPU bounding box for one block — also the GPU backend's fallback.

    Takes a plain ``min``/``max`` first and only redoes the block with
    ``where=isfinite`` when the answer comes back non-finite. The masked form is
    the one that survives a NaN, but masked reductions are ~1.7x slower than
    plain ones (measured 3.30 s against 1.89 s over 50M points), and real clouds
    almost never contain one — so paying for it on every block was paying for a
    case that virtually never happens.
    """
    xyz = block[:, :3]
    lo = np.min(xyz, axis=0)
    hi = np.max(xyz, axis=0)
    if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
        return lo.astype(np.float32), hi.astype(np.float32)

    # A NaN spread through, or the block really does reach infinity. Redo it
    # ignoring the non-finite values. `initial=` keeps this quiet when a whole
    # column is bad, where nanmin would warn once per block.
    finite = np.isfinite(xyz)
    lo = np.min(xyz, axis=0, where=finite, initial=np.inf)
    hi = np.max(xyz, axis=0, where=finite, initial=-np.inf)
    return lo.astype(np.float32), hi.astype(np.float32)


def _argsort_numpy(cell_ids):
    """Shared CPU bucket order — also the GPU backend's fallback.

    ``kind="stable"`` on a small integer type is a radix pass, so this is a
    counting sort in all but name.
    """
    return np.argsort(cell_ids, kind="stable").astype(np.int32, copy=False)


def _check_dtype(out):
    """Refuse an out array the kernels cannot represent exactly."""
    key = np.dtype(out.dtype)
    if key not in _OUT_TYPES:
        raise TypeError(
            f"cell_ids needs a {sorted(str(k) for k in _OUT_TYPES)} out array, "
            f"not {key}")
    return key


class CuPyGrid(SpatialGridBackend):
    """GPU cell numbering using a single fused CuPy kernel."""

    @property
    def name(self) -> str:
        return "CuPy"

    @property
    def is_gpu(self) -> bool:
        return True

    @staticmethod
    def _required_mb(block, out) -> int:
        """GPU memory for one block: the points, plus the cell number per point."""
        return (int(block.nbytes // (1024 * 1024))
                + int((len(block) * out.dtype.itemsize) // (1024 * 1024)) + 8)

    def cell_ids(self, block, lo, inv_step, shape, out):
        """Number one block of points by cell; see SpatialGridBackend."""
        from infrastructure.memory_manager import MemoryManager

        dtype = _check_dtype(out)
        required_mb = self._required_mb(block, out)
        if not MemoryManager.can_use_gpu(required_mb):
            return _cell_ids_numpy(block, lo, inv_step, shape, out)

        import cupy as cp
        try:
            block = np.ascontiguousarray(block, dtype=np.float32)
            pts_gpu = cp.asarray(block)
            out_gpu = cp.empty(len(block), dtype=dtype)

            _cell_id_kernel(dtype)(
                pts_gpu, np.int32(block.shape[1]),
                np.float32(lo[0]), np.float32(lo[1]), np.float32(lo[2]),
                np.float32(inv_step[0]), np.float32(inv_step[1]),
                np.float32(inv_step[2]),
                np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                out_gpu, size=len(block),
            )

            out[:] = cp.asnumpy(out_gpu)
            del pts_gpu, out_gpu
            return out

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM numbering grid cells: {e}, falling back to CPU")
            MemoryManager.cleanup()
            return _cell_ids_numpy(block, lo, inv_step, shape, out)

    def block_bounds(self, block):
        """Bounding box of one block; see SpatialGridBackend."""
        from infrastructure.memory_manager import MemoryManager

        required_mb = int(block.nbytes // (1024 * 1024)) + 8
        if not MemoryManager.can_use_gpu(required_mb):
            return _block_bounds_numpy(block)

        import cupy as cp
        try:
            xyz = cp.asarray(np.ascontiguousarray(block[:, :3], dtype=np.float32))
            finite = cp.isfinite(xyz)
            if bool(cp.all(finite)):
                lo = cp.asnumpy(xyz.min(axis=0))
                hi = cp.asnumpy(xyz.max(axis=0))
            else:
                # Same fallback as the CPU path: replace the bad values with
                # something that loses every comparison rather than spreading.
                lo = cp.asnumpy(cp.where(finite, xyz, cp.float32(np.inf)).min(axis=0))
                hi = cp.asnumpy(cp.where(finite, xyz, cp.float32(-np.inf)).max(axis=0))
            del xyz, finite
            return lo.astype(np.float32), hi.astype(np.float32)

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM measuring the bounding box: {e}, "
                           f"falling back to CPU")
            MemoryManager.cleanup()
            return _block_bounds_numpy(block)

    def argsort(self, cell_ids):
        """Bucket order for the whole index; see SpatialGridBackend.

        Not blocked, because a sort cannot be: the answer depends on every
        element. So this is the one place where the whole index has to fit on
        the card at once. CuPy's argsort returns int64, so the requirement is
        the index plus 8 bytes per point plus the radix temporary — roughly
        4 GB at 168M points, which is why the pre-check matters here more than
        anywhere else.
        """
        from infrastructure.memory_manager import MemoryManager

        n = len(cell_ids)
        required_mb = int((cell_ids.nbytes + n * 12) // (1024 * 1024)) + 64
        if not MemoryManager.can_use_gpu(required_mb):
            logger.info(f"Grid sort staying on the CPU (needs ~{required_mb} MB VRAM)")
            return _argsort_numpy(cell_ids)

        import cupy as cp
        try:
            ids_gpu = cp.asarray(cell_ids)
            order_gpu = cp.argsort(ids_gpu, kind="stable").astype(cp.int32)
            order = cp.asnumpy(order_gpu)
            del ids_gpu, order_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return order

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM sorting the grid: {e}, falling back to CPU")
            MemoryManager.cleanup()
            return _argsort_numpy(cell_ids)


class NumpyGrid(SpatialGridBackend):
    """CPU cell numbering. Also the fallback whenever the GPU path declines."""

    @property
    def name(self) -> str:
        return "NumPy"

    @property
    def is_gpu(self) -> bool:
        return False

    def cell_ids(self, block, lo, inv_step, shape, out):
        """Number one block of points by cell; see SpatialGridBackend."""
        _check_dtype(out)
        return _cell_ids_numpy(np.asarray(block, dtype=np.float32),
                               lo, inv_step, shape, out)

    def block_bounds(self, block):
        """Bounding box of one block; see SpatialGridBackend."""
        return _block_bounds_numpy(np.asarray(block))

    def argsort(self, cell_ids):
        """Bucket order for the whole index; see SpatialGridBackend."""
        return _argsort_numpy(cell_ids)
