import logging
import traceback
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class DataManagementMixin:
    """Point cloud data loading and management for PCDViewerWidget."""

    def _init_data(self):
        """Initialize point cloud data attributes."""
        self.points = None
        self.vbo = None
        self._kdtree = None  # Spatial index for fast point picking

        # Line geometry (e.g. mesh wireframes, CAD polylines). Independent of point data.
        self.line_vertices = None  # Nx3 float32
        self.line_indices = None   # (2*M,) uint32 — flattened edge endpoint indices
        self.line_colors = None    # Nx3 float32 per-vertex colors, or None for uniform gray

        # Initialize list to store indices of picked points
        self.picked_points_indices = []

    def _release_point_data(self):
        """Release current point cloud data and associated resources."""
        from infrastructure.memory_manager import MemoryManager

        if self.vbo is not None:
            try:
                self.vbo.delete()
            except Exception as e:
                logger.warning(f"  Error deleting VBO: {e}")
            self.vbo = None
        self.points = None
        self._kdtree = None
        MemoryManager.cleanup()

    def _clear_display(self):
        """Clear the display: release data and trigger repaint."""
        self._release_point_data()
        self._release_line_data()
        self.update()

    def _release_line_data(self):
        """Drop any stored line geometry."""
        self.line_vertices = None
        self.line_indices = None
        self.line_colors = None

    def set_lines(self, vertices: np.ndarray, edges: np.ndarray,
                  colors: np.ndarray = None):
        """
        Set wireframe line geometry to be rendered in the widget.

        Line data is independent of point data — both can be displayed
        simultaneously. When no point data is present, camera bounds are
        recomputed from the line vertices so zoom_to_extent() works.

        Args:
            vertices: Nx3 float32 array of vertex positions, or None to clear.
            edges: Mx2 integer array of vertex-index pairs defining line segments.
            colors: Nx3 float32 per-vertex RGB colors in [0, 1], or None for
                uniform gray (0.85, 0.85, 0.85).
        """
        if vertices is None or edges is None or len(edges) == 0:
            self._release_line_data()
            self.update()
            return

        assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be Nx3"
        assert vertices.dtype == np.float32, f"vertices must be float32, not {vertices.dtype}"
        assert edges.ndim == 2 and edges.shape[1] == 2, "edges must be Mx2"

        self._release_line_data()

        self.line_vertices = vertices
        self.line_indices = edges.reshape(-1).astype(np.uint32, copy=False)
        if colors is not None:
            assert colors.shape == vertices.shape, "colors must match vertices shape Nx3"
            self.line_colors = np.asarray(colors, dtype=np.float32)

        # Compute camera bounds from lines only when no point data exists.
        if self.points is None:
            min_bounds = np.min(vertices, axis=0)
            max_bounds = np.max(vertices, axis=0)
            self.center = (min_bounds + max_bounds) / 2.0
            self.size = max_bounds - min_bounds
            self.max_extent = float(np.max(self.size)) or 1.0

        self.update()

    def set_points(self, points: np.ndarray, colors: np.ndarray = None):
        """
        Set the point cloud data to be visualised in the widget.

        This method accepts point coordinates and optional colour information. It also initialises
        the vertex buffer object (VBO) with the given data.

        Args:
            points (numpy.ndarray): A Nx3 array of point coordinates (x, y, z). Must be of type float32.
            colors (numpy.ndarray, optional): A Nx3 array of RGB colour values corresponding to each point.
                Must be of type float32. If not provided, all points will be rendered with a default colour.

        Raises:
            AssertionError: If the `points` array does not have the correct shape or data type.
            AssertionError: If the `colors` array is provided but does not have the correct shape or data type.
        """
        logger.debug("PCDViewerWidget.set_points() called")

        if points is None:
            logger.debug("  Clearing display")
            self._clear_display()
            return

        # Validate inputs
        logger.debug(f"  Points: {points.shape}, {points.nbytes / 1024 / 1024:.1f} MB")
        assert points.shape[1] == 3, "Points array must have shape Nx3"
        assert points.dtype == np.float32, f"Points array must be float32, not {points.dtype}"

        if colors is not None:
            logger.debug(f"  Colors: {colors.shape}, {colors.nbytes / 1024 / 1024:.1f} MB")
            assert colors.shape[0] == points.shape[0], "Points and colors must have same length"
            assert colors.shape[1] == 3, "Colors array must have shape Nx3"
            assert colors.dtype == np.float32, "Colors array must be float32"
        else:
            logger.debug("  No colors provided, using white")
            colors = np.ones_like(points, dtype=np.float32)

        try:
            # Release existing point data before allocating new arrays
            self._release_point_data()

            # Create combined array (pre-allocated for efficiency)
            num_points = len(points)
            self.points = np.empty((num_points, 6), dtype=np.float32)
            self.points[:, :3] = points
            self.points[:, 3:] = colors
            logger.debug(f"  Combined array: {self.points.shape}, {self.points.nbytes / 1024 / 1024:.1f} MB")

            # Build spatial index for fast point picking
            logger.debug("  Building KDTree spatial index...")
            self._kdtree = cKDTree(self.points[:, :3])
            logger.debug("  KDTree built")

            self.update()
            logger.debug("  set_points() completed")

        except MemoryError as e:
            logger.error(f"  MEMORY ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            logger.error(traceback.format_exc())
            raise

    def set_point_vertices(self, vertices: np.ndarray):
        """
        Set point vertex data directly. More memory efficient than set_points().

        This method accepts vertex data as an Nx6 array (position + color per vertex)
        and assigns it directly without creating intermediate arrays.

        Args:
            vertices (numpy.ndarray): Nx6 array of point vertices where columns 0-2
                are XYZ position and columns 3-5 are RGB color. Must be float32.

        Raises:
            AssertionError: If the array does not have shape Nx6 or is not float32.
        """
        logger.debug("PCDViewerWidget.set_point_vertices() called")

        if vertices is None:
            logger.debug("  Clearing display")
            self._clear_display()
            return

        # Validate input
        logger.debug(f"  Vertices: {vertices.shape}, {vertices.nbytes / 1024 / 1024:.1f} MB")
        if vertices.shape[1] != 6:
            raise ValueError(f"Vertices must have shape Nx6, got {vertices.shape}")
        if vertices.dtype != np.float32:
            raise ValueError(f"Vertices must be float32, not {vertices.dtype}")

        try:
            # Release existing point data before assigning new array
            self._release_point_data()

            # Direct assignment - no copying needed
            self.points = vertices
            logger.debug(f"  Assigned {len(vertices):,} point vertices directly")

            # Build spatial index for fast point picking
            logger.debug("  Building KDTree spatial index...")
            self._kdtree = cKDTree(self.points[:, :3])
            logger.debug("  KDTree built")

            self.update()
            logger.debug("  set_point_vertices() completed")

        except MemoryError as e:
            logger.error(f"  MEMORY ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            logger.error(traceback.format_exc())
            raise
