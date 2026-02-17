import logging
import traceback
import numpy as np
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import QOpenGLWidget, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluProject, gluUnProject
from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric, gluQuadricDrawStyle
from OpenGL.GLU import gluSphere
from OpenGL.GLU import GLU_FILL
from OpenGL.arrays import vbo

from config.config import global_variables
from application.lod_manager import LODManager

# Get logger for this module
logger = logging.getLogger(__name__)


class PCDViewerWidget(QOpenGLWidget):
    """
    A Qt-based OpenGL widget for viewing point cloud data (PCD).

    This class provides an interactive visualisation environment for point clouds, allowing for operations such as
    rotation, panning, zooming, and point selection. It leverages OpenGL for efficient rendering of large point
    clouds and supports advanced features such as axis symbol display, point picking, and modifying the centre of
    rotation.

    Hotkeys:
        - Left Click: Rotate around the X and Y axes.
        - CTRL + Left Click: Rotate around the Z-axis.
        - Right/Middle Click: Pan along the X and Y axes.
        - CTRL + Right/Middle Click: Pan along the Z-axis.
        - Mouse Wheel: Zoom in and out.
        - CTRL + R: Reset the camera view to its default state.
        - F: Zoom to extent (fit all visible points in viewport).
        - SHIFT + Left Click: Select a point in the point cloud.
        - SHIFT + Right Click: Deselect a point in the point cloud.
        - P: Enter polygon selection mode (click vertices, right-click/double-click to close and select).
        - Shift + P: Enter polygon deselect mode (draw polygon to remove points from selection).
        - CTRL + SHIFT + Right Click: Deselect all selected points from the clicked cluster (by color).
        - ESC: Cancel polygon mode (if active), or deselect all selected points after confirmation.
        - + / =: Increase point size (geometric scaling x1.2).
        - -: Decrease point size (geometric scaling /1.2, min 0.5).

    Attributes:
    -----------
        The following attributes represent internal state variables that are used in the widget's logic and operations. They are not intended to be directly accessed or modified by users of this class, but instead are used internally to maintain the state of the viewer and perform necessary computations.

        - center (numpy.ndarray): Centre of the current point cloud.
        - size (numpy.ndarray): Size of the bounding box of the point cloud.
        - max_extent (float): Maximum extent of the point cloud's bounding box.
        - fov (float): Field of view (in degrees) of the perspective camera.
        - near_plane (float): Distance to the near clipping plane.
        - far_plane (float): Distance to the far clipping plane.
        - points (numpy.ndarray): Point cloud data including positions and optional colours.
        - vbo (vbo.VBO): Vertex buffer object for efficient rendering of point cloud data.
        - last_mouse_pos (QPoint): Last recorded mouse position for interaction.
        - rot_x (float): Rotation angle around the X-axis.
        - rot_y (float): Rotation angle around the Y-axis.
        - rot_z (float): Rotation angle around the Z-axis.
        - pan_x (float): Panning offset along the X-axis.
        - pan_y (float): Panning offset along the Y-axis.
        - pan_z (float): Panning offset along the Z-axis.
        - is_rotating (bool): Flag to indicate if the widget is in rotation mode.
        - is_rotating_z (bool): Flag to indicate if the widget is rotating around the Z-axis.
        - is_panning (bool): Flag to indicate if the widget is in panning mode.
        - is_panning_z (bool): Flag to indicate if the widget is panning along the Z-axis.
        - show_axis (bool): Flag to indicate whether the axis symbol should be displayed.
        - picked_points_indices (list of int): List of indices of picked points from the point cloud.
        - model_view_matrix (numpy.ndarray): Model-view matrix for the current OpenGL context.
        - projection_matrix (numpy.ndarray): Projection matrix for the current OpenGL context.
        - viewport (numpy.ndarray): Viewport settings for the OpenGL context.
        - camera_distance (float): Current distance of the camera from the point cloud.
        - axis_timer (QTimer or None): Timer for hiding the axis symbol after panning.

    """

    @property
    def point_size(self):
        """
        float: Size of the rendered points. Can be set to adjust point cloud visibility.
        """
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        if value > 0:
            self._point_size = value
            self.update()

    @property
    def axis_line_length(self):
        """
        float: Length of the axis lines for the axis symbol. Can be set to adjust axis visibility.
        """
        return self._axis_line_length

    @axis_line_length.setter
    def axis_line_length(self, value):
        if value > 0:
            self._axis_line_length = value
            self.update()

    @property
    def axis_line_width(self):
        """
        float: Width of the axis lines for the axis symbol. Can be set to adjust axis visibility.
        """
        return self._axis_line_width

    @axis_line_width.setter
    def axis_line_width(self, value):
        if value > 0:
            self._axis_line_width = value
            self.update()

    @property
    def picking_point_threshold_factor(self):
        """
        float: Threshold factor for determining if a point can be picked. Can be adjusted for selection precision.
        """
        return self._picking_point_threshold_factor

    @picking_point_threshold_factor.setter
    def picking_point_threshold_factor(self, value):
        if value > 0:
            self._picking_point_threshold_factor = value

    @property
    def picked_point_highlight_color(self):
        """
        tuple of float: RGB colour for highlighting picked points. Can be set to customise highlight colour.
        """
        return self._picked_point_highlight_color

    @picked_point_highlight_color.setter
    def picked_point_highlight_color(self, value):
        if len(value) == 3 and all(0.0 <= v <= 1.0 for v in value):
            self._picked_point_highlight_color = value
            self.update()

    @property
    def picked_point_highlight_size(self):
        """
        float: Size factor for rendering picked points as highlighted spheres. Can be set to adjust highlight size.
        """
        return self._picked_point_highlight_size

    @picked_point_highlight_size.setter
    def picked_point_highlight_size(self, value):
        if value > 0:
            self._picked_point_highlight_size = value
            self.update()

    @property
    def zoom_min_factor(self):
        """
        float: Minimum zoom factor limit. Can be adjusted to control zoom limits.
        """
        return self._zoom_min_factor

    @zoom_min_factor.setter
    def zoom_min_factor(self, value):
        if value > 0:
            self._zoom_min_factor = value

    @property
    def zoom_max_factor(self):
        """
        float: Maximum zoom factor limit. Can be adjusted to control zoom limits.
        """
        return self._zoom_max_factor

    @zoom_max_factor.setter
    def zoom_max_factor(self, value):
        if value > self._zoom_min_factor:
            self._zoom_max_factor = value

    @property
    def zoom_sensitivity(self):
        """
        float: Sensitivity of the zoom operation. Can be adjusted to control zoom speed.
        """
        return self._zoom_sensitivity

    @zoom_sensitivity.setter
    def zoom_sensitivity(self, value):
        if value > 0:
            self._zoom_sensitivity = value

    @property
    def pan_sensitivity(self):
        """
        float: Sensitivity of the panning operation. Can be adjusted to control panning speed.
        """
        return self._pan_sensitivity

    @pan_sensitivity.setter
    def pan_sensitivity(self, value):
        if value > 0:
            self._pan_sensitivity = value

    @property
    def rotate_sensitivity(self):
        """
        float: Sensitivity of the rotation operation. Can be adjusted to control rotation speed.
        """
        return self._rotate_sensitivity

    @rotate_sensitivity.setter
    def rotate_sensitivity(self, value):
        if value > 0:
            self._rotate_sensitivity = value

    @property
    def pixel_threshold(self):
        """
        int: Threshold (in pixels) for selecting/deselecting points in screen space. Can be set to adjust selection accuracy.
        """
        return self._pixel_threshold

    @pixel_threshold.setter
    def pixel_threshold(self, value):
        if value > 0:
            self._pixel_threshold = value

    @property
    def default_camera_distance(self):
        """
        float: Default distance of the camera from the point cloud. Can be set to adjust the initial camera view.
        """
        return self._default_camera_distance

    @default_camera_distance.setter
    def default_camera_distance(self, value):
        self._default_camera_distance = value

    @property
    def default_zoom_factor(self):
        """
        float: Default zoom factor for the camera. Can be set to adjust the initial zoom level.
        """
        return self._default_zoom_factor

    @default_zoom_factor.setter
    def default_zoom_factor(self, value):
        self._default_zoom_factor = value

    @property
    def default_rot_x(self):
        """
        float: The current value of the default rotation around the X-axis.
        """
        return self._default_rot_x

    @default_rot_x.setter
    def default_rot_x(self, value):
        self._default_rot_x = value

    @property
    def default_rot_y(self):
        """
        float: The current value of the default rotation around the Y-axis.
        """
        return self._default_rot_y

    @default_rot_y.setter
    def default_rot_y(self, value):
        self._default_rot_y = value

    @property
    def default_rot_z(self):
        """
        float: The current value of the default rotation around the Z-axis.
        """
        return self._default_rot_z

    @default_rot_z.setter
    def default_rot_z(self, value):
        self._default_rot_z = value

    @property
    def default_pan_x(self):
        """
        float: The current value of the default pan along the X-axis.
        """
        return self._default_pan_x

    @default_pan_x.setter
    def default_pan_x(self, value):
        self._default_pan_x = value

    @property
    def default_pan_y(self):
        """
        float: The current value of the default pan along the Y-axis.
        """
        return self._default_pan_y

    @default_pan_y.setter
    def default_pan_y(self, value):
        self._default_pan_y = value

    @property
    def default_pan_z(self):
        """
        float: The current value of the default pan along the Z-axis.
        """
        return self._default_pan_z

    @default_pan_z.setter
    def default_pan_z(self, value):
        self._default_pan_z = value

    @property
    def lod_info(self) -> dict:
        """Current LOD state for debugging."""
        controller = global_variables.global_application_controller
        rc = controller.rendering_coordinator if controller else None
        full = rc.total_visible_points if rc else 0
        rendered = len(self.points) if self.points is not None else 0
        return {
            "sample_rate": f"{self._current_sample_rate:.1%}",
            "point_budget": rc._point_budget if rc else 0,
            "full_points": full,
            "rendered_points": rendered,
            "reduction": f"{(1 - rendered/full)*100:.1f}%" if full > 0 else "0%"
        }

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set focus policy to ensure the widget receives keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Initialize variables for zoom to extent
        self.center = np.array([0.0, 0.0, 0.0])
        self.size = np.array([1.0, 1.0, 1.0])
        self.max_extent = None
        self.fov = 60.0  # Field of view in degrees
        self.near_plane = 0.1
        self.far_plane = 1000.0

        # TODO: Need to control visibility of several point clouds
        # New attribute to control visibility of the point cloud
        self.visible = True

        # Point cloud data
        self.points = None
        self.vbo = None
        self._kdtree = None  # Spatial index for fast point picking

        # Interaction variables
        self.last_mouse_pos = None

        self.rot_x = 0.0  # Rotation around X-axis
        self.rot_y = 0.0  # Rotation around Y-axis
        self.rot_z = 0.0  # Rotation around Z-axis

        self.pan_x = 0.0  # Panning along X-axis
        self.pan_y = 0.0  # Panning along Y-axis
        self.pan_z = 0.0  # Panning along Z-axis

        # Interaction flags
        self.is_rotating = False
        self.is_rotating_z = False  # New flag for Z-axis rotation
        self.is_panning = False
        self.is_panning_z = False  # New flag for Z-axis panning
        self.show_axis = False  # Flag to show/hide the axis symbol

        # Initialize list to store indices of picked points
        self.picked_points_indices = []

        # Initialize matrices used in the pick_point method
        self.model_view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.viewport = np.array([0, 0, self.width(), self.height()], dtype=np.int32)

        # Setting properties as attributes that can be adjusted
        self._point_size = 0.5
        self._POINT_SIZE_MIN = 0.5
        self._POINT_SIZE_MAX = 20.0
        self._POINT_SIZE_FACTOR = 1.2  # geometric scaling per keystroke
        self._axis_line_length = 5
        self._axis_line_width = 5
        self._picking_point_threshold_factor = 1.0
        self._picked_point_highlight_color = (1.0, 0.0, 0.0)
        self._picked_point_highlight_size = 1
        self._zoom_min_factor = 0.02
        self._zoom_max_factor = 1
        self._zoom_sensitivity = 1
        self._pan_sensitivity = 1
        self._rotate_sensitivity = 1
        self._pixel_threshold = 5

        # Properties for the attributes
        self._default_camera_distance = 100
        self._default_zoom_factor = self._zoom_max_factor
        self._default_rot_x = 0.0
        self._default_rot_y = 0.0
        self._default_rot_z = 0.0
        self._default_pan_x = 0.0
        self._default_pan_y = 0.0
        self._default_pan_z = 0.0

        # Initialize default values
        self.camera_distance = self._default_camera_distance
        self.zoom_factor = self._default_zoom_factor

        # Timer for hiding the axis symbol after panning
        self.axis_timer = None

        # Polygon selection mode state
        self._polygon_mode = False        # Whether polygon selection mode is active
        self._polygon_deselect_mode = False  # Whether polygon is for deselection (vs selection)
        self._polygon_vertices = []       # List of (x, y) tuples in Qt widget coordinates

        # Stored polygons + matrices for full-resolution re-testing by plugins.
        # Each entry is a (polygon, mv, proj, viewport) tuple.
        self._selection_polygons = []

        # Per-branch index ranges in combined vertex array: uid -> (start, end)
        self._branch_offsets = {}

        # LOD state (for triggering DataManager re-render)
        self._current_sample_rate: float = 1.0
        self._lod_enabled: bool = True  # Dynamic LOD for large point clouds

        # create a slot connection for branches_visibility_status emitted from tree_structure_widget

    def initializeGL(self):
        """
        Initialise the OpenGL context for the widget.

        This method sets up the OpenGL environment, including clearing the background colour, enabling depth testing,
        and setting up blending options for transparency.

        Raises:
            ValueError: If the point cloud data (points) is not set before initialisation.
        """

        # OpenGL initialization
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
        from infrastructure.memory_manager import MemoryManager

        logger.debug("PCDViewerWidget.set_points() called")

        if points is None:
            # Clear display and release memory
            logger.debug("  Clearing display")
            if self.vbo is not None:
                try:
                    self.vbo.delete()
                except Exception as e:
                    logger.warning(f"  Error deleting VBO: {e}")
                self.vbo = None
            self.points = None
            self._kdtree = None  # Clear spatial index
            MemoryManager.cleanup()
            self.update()
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
            # Release existing memory before allocating new arrays
            if self.vbo is not None:
                try:
                    self.vbo.delete()
                except Exception:
                    pass
                self.vbo = None
            self.points = None
            MemoryManager.cleanup()

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
            import traceback
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
        from infrastructure.memory_manager import MemoryManager

        logger.debug("PCDViewerWidget.set_point_vertices() called")

        if vertices is None:
            # Clear display and release memory
            logger.debug("  Clearing display")
            if self.vbo is not None:
                try:
                    self.vbo.delete()
                except Exception as e:
                    logger.warning(f"  Error deleting VBO: {e}")
                self.vbo = None
            self.points = None
            self._kdtree = None  # Clear spatial index
            MemoryManager.cleanup()
            self.update()
            return

        # Validate input
        logger.debug(f"  Vertices: {vertices.shape}, {vertices.nbytes / 1024 / 1024:.1f} MB")
        if vertices.shape[1] != 6:
            raise ValueError(f"Vertices must have shape Nx6, got {vertices.shape}")
        if vertices.dtype != np.float32:
            raise ValueError(f"Vertices must be float32, not {vertices.dtype}")

        try:
            # Release existing memory before assigning new array
            if self.vbo is not None:
                try:
                    self.vbo.delete()
                except Exception:
                    pass
                self.vbo = None
            self.points = None
            MemoryManager.cleanup()

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

    def draw_axis_symbol(self, position):
        """
        Draw the axis symbol at the specified position.

        This method renders a 3D axis symbol, consisting of X, Y, and Z axes, at the given position in the point cloud
        space. The X-axis is rendered in red, the Y-axis in green, and the Z-axis in blue. This symbol is used to help
        users orient themselves within the point cloud.

        Args:
            position (tuple or list or numpy.ndarray): A 3-element array representing the (x, y, z) position where
                the axis symbol should be drawn.
        """

        glPushMatrix()

        glLineWidth(self.axis_line_width)
        glTranslatef(position[0], position[1], position[2])

        glBegin(GL_LINES)
        # X-axis in red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(self.axis_line_length, 0.0, 0.0)

        # Y-axis in green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, self.axis_line_length, 0.0)

        # Z-axis in blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, self.axis_line_length)
        glEnd()

        glLineWidth(self.axis_line_width)

        glPopMatrix()

    def paintGL(self):
        """
        Render the point cloud and other visual elements.

        This method is called whenever the widget needs to be repainted. It clears the colour and depth buffers, sets
        up the projection and model-view matrices, and renders the point cloud data, picked points, and optionally the
        axis symbol. The rendering includes applying transformations for panning, zooming, and rotation.

        If no point cloud data is set, the method returns without rendering anything.
        """

        if self.points is None or self.max_extent is None:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w = self.width()
        h = self.height()
        aspect = w / h if h != 0 else 1
        gluPerspective(self.fov * self.zoom_factor, aspect, max(self.near_plane, 0.1), self.far_plane)

        # Update model-view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Move the camera back
        camera_distance = self.camera_distance * self.zoom_factor
        glTranslatef(0.0, 0.0, -camera_distance)

        # Apply panning
        glTranslatef(self.pan_x, self.pan_y, self.pan_z)

        # Translate to the center of rotation
        glTranslatef(self.center[0], self.center[1], self.center[2])

        # Apply rotations around the origin
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glRotatef(self.rot_z, 0.0, 0.0, 1.0)

        # Translate back from the center
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])

        # Render the point cloud
        self.render_point_cloud()

        # Render picked points
        self.render_picked_points()

        # Store matrices for picking
        self.model_view_matrix = glGetDoublev(GL_MODELVIEW_MATRIX).copy()
        self.projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX).copy()
        self.viewport = glGetIntegerv(GL_VIEWPORT).copy()

        # Draw the point of view (POV) sphere
        if self.show_axis:
            # Draw axis symbol at the center of rotation
            self.draw_axis_symbol(self.center)

        # Draw polygon selection overlay (2D on top of scene)
        self.render_polygon_overlay()

    def render_point_cloud(self):
        """
        Render the point cloud data using OpenGL VBO.
        """
        if self.points is None:
            return

        try:
            # Create VBO if needed
            if self.vbo is None:
                num_points = len(self.points)
                data_size_mb = self.points.nbytes / (1024 * 1024)
                logger.debug(f"  Creating VBO: {num_points:,} points, {data_size_mb:.1f} MB")
                self.vbo = vbo.VBO(self.points)
                logger.debug("  VBO created")

            glPointSize(self.point_size)

            # Enable client states
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            # Bind the VBO
            self.vbo.bind()

            # Set pointers to the VBO data
            stride = 6 * self.points.itemsize
            glVertexPointer(3, GL_FLOAT, stride, self.vbo)
            glColorPointer(3, GL_FLOAT, stride, self.vbo + 12)

            # Draw all points
            glDrawArrays(GL_POINTS, 0, len(self.points))

            # Disable client states
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

            # Unbind the VBO to prevent issues with other rendering
            self.vbo.unbind()

        except Exception as e:
            logger.error(f"  ERROR in render_point_cloud(): {e}")
            logger.error(f"  Points shape: {self.points.shape if self.points is not None else 'None'}")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise

    def render_picked_points(self):
        """
        Render the picked points in the point cloud.

        This method highlights the picked points by drawing spheres at their positions. The colour and size of the
        spheres are determined by the `picked_point_highlight_color` and `picked_point_highlight_size` attributes.
        The purpose of this method is to visually distinguish the picked points from the rest of the point cloud.

        If no points have been picked, the method returns without rendering anything.
        """

        # Highlight picked points by drawing larger points
        if self.picked_points_indices:
            # Filter out invalid indices
            max_idx = len(self.points) - 1
            valid = [i for i in self.picked_points_indices if i <= max_idx]
            if len(valid) != len(self.picked_points_indices):
                self.picked_points_indices[:] = valid

            if valid:
                positions = self.points[valid, :3]
                highlight_size = self.point_size * self.picked_point_highlight_size * 5
                glPointSize(highlight_size)
                glColor3f(*self.picked_point_highlight_color)
                glBegin(GL_POINTS)
                for pos in positions:
                    glVertex3f(pos[0], pos[1], pos[2])
                glEnd()

    def resizeGL(self, w, h):
        """
        Handle the resizing of the OpenGL viewport.

        This method is called whenever the widget is resized. It updates the OpenGL viewport to match the new widget
        dimensions and adjusts the projection matrix to maintain the correct aspect ratio.

        Args:
            w (int): The new width of the widget.
            h (int): The new height of the widget.
        """

        if h == 0:
            h = 1
        aspect = w / h
        glViewport(0, 0, w, h)

        # Update stored viewport
        self.viewport = np.array([0, 0, w, h], dtype=np.int32)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov * self.zoom_factor, aspect, max(self.near_plane, 0.1), self.far_plane)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousePressEvent(self, event):
        """
        Handle mouse press events for interaction with the point cloud.

        This method processes mouse press events to initiate interactions such as rotation, panning, and point
        selection. Depending on the mouse button and modifier keys pressed, the method determines the type of
        interaction (e.g., rotating, panning, or selecting points).

        Args:
            event (QMouseEvent): The mouse event containing details such as the button pressed and the mouse position.
        """

        # Polygon mode: left-click adds vertex, right-click closes polygon
        if self._polygon_mode:
            if event.button() == Qt.LeftButton:
                self._polygon_vertices.append((event.x(), event.y()))
                self.update()
                return
            elif event.button() == Qt.RightButton:
                if self._polygon_deselect_mode:
                    self._close_polygon_and_deselect()
                else:
                    self._close_polygon_and_select()
                return

        self.last_mouse_pos = event.pos()
        modifiers = event.modifiers()

        if (modifiers & Qt.ShiftModifier) and (modifiers & Qt.ControlModifier):
            if event.button() == Qt.RightButton:
                # Ctrl + Shift + Right Click: Deselect cluster by color
                self.deselect_cluster_at(event.pos())
        elif modifiers & Qt.ShiftModifier:
            if event.button() == Qt.LeftButton:
                # Shift + Left Click: Select a point
                self.pick_point(event.pos(), select=True)
            elif event.button() == Qt.RightButton:
                # Shift + Right Click: Deselect a point
                self.pick_point(event.pos(), select=False)
        elif modifiers & Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                # Ctrl + Left Click: Rotate around Z-axis
                self.setCursor(Qt.OpenHandCursor)
                self.is_rotating_z = True
            elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
                # Ctrl + Right/Middle Click: Pan along Z-axis
                self.setCursor(Qt.ClosedHandCursor)
                self.is_panning_z = True

            self.show_axis = True  # Show axis when panning
        else:
            if event.button() == Qt.LeftButton:
                # Rotate around X and Y axes
                self.setCursor(Qt.OpenHandCursor)
                self.is_rotating = True
            elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
                # Pan along X and Y axes
                self.setCursor(Qt.ClosedHandCursor)
                self.is_panning = True

            self.show_axis = True  # Show axis when panning

    def mouseDoubleClickEvent(self, event):
        """
        Handle mouse double-click events for updating the centre of rotation.

        This method processes double-click events to update the centre of rotation of the point cloud view. When the
        user double-clicks on a point, the centre of rotation is moved to that point, making it the new focal point
        for subsequent rotations.

        Args:
            event (QMouseEvent): The mouse event containing details such as the button pressed and the mouse position.
        """

        # Polygon mode: double-click closes polygon
        if self._polygon_mode:
            if event.button() == Qt.LeftButton:
                if self._polygon_deselect_mode:
                    self._close_polygon_and_deselect()
                else:
                    self._close_polygon_and_select()
            return

        if event.button() == Qt.LeftButton:
            # Update the rotation center on double left-click
            self.update_rotation_center(event.pos())

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events for ending interactions.

        This method processes mouse release events to end interactions such as rotation or panning. When the user
        releases the mouse button, any ongoing rotation or panning is stopped, and the cursor is reset to its default
        state. The axis symbol is also hidden after the interaction ends.

        Args:
            event (QMouseEvent): The mouse event containing details such as the button released and the mouse position.
        """
        # Suppress release events in polygon mode
        if self._polygon_mode:
            return

        if event.button() == Qt.LeftButton:
            self.is_rotating = False
            self.is_rotating_z = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.is_panning_z = False
            self.setCursor(Qt.ArrowCursor)

        self.show_axis = False  # Hide axis when panning stops
        self.update()

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events for updating the interaction state.

        This method processes mouse movement events to update the state of ongoing interactions such as rotation and
        panning. Depending on the interaction mode (e.g., rotating or panning), it adjusts the rotation angles or
        panning offsets based on the mouse movement distance.

        Args:
            event (QMouseEvent): The mouse event containing details such as the current mouse position.
        """

        # Suppress drag in polygon mode
        if self._polygon_mode:
            return

        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if self.is_rotating:
            # Rotate around X and Y axes
            # / 10 is a scaling factor to adjust the rotation sensitivity
            self.rot_x += dy * self.rotate_sensitivity / 10
            self.rot_y += dx * self.rotate_sensitivity / 10
        elif self.is_rotating_z:
            # Rotate around Z-axis
            self.rot_z += dx * self.rotate_sensitivity / 10  # Horizontal movement affects Z rotation
        elif self.is_panning:
            # Pan along X and Y axes
            # / 10000 is a scaling factor to adjust the panning sensitivity
            self.pan_x += dx * self.pan_sensitivity / 10000 * self.camera_distance
            self.pan_y -= dy * self.pan_sensitivity / 10000 * self.camera_distance
        elif self.is_panning_z:
            # Pan along Z-axis
            self.pan_z += dy * self.pan_sensitivity / 10000 * self.camera_distance  # Vertical movement affects Z panning

        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming the view.

        This method processes mouse wheel events to adjust the zoom factor of the camera. The zoom sensitivity is
        controlled by the `zoom_sensitivity` attribute, and the resulting zoom factor is clamped between the
        `zoom_min_factor` and `zoom_max_factor` attributes to prevent excessive zooming in or out.

        Args:
            event (QWheelEvent): The wheel event containing details such as the direction and magnitude of the scroll.
        """

        delta = event.angleDelta().y()
        # / 1000 is a scaling factor to be able to set the defult_zoom_sensitivity to a reasonable value
        #
        zoom_step = delta * self.zoom_sensitivity / 1000
        self.zoom_factor *= (1 + zoom_step)
        self.zoom_factor = max(self.zoom_min_factor, min(self.zoom_factor, self.zoom_max_factor))  # Limit zoom factor

        # Show the axis symbol after zooming
        self.show_axis = True

        # Stop existing timer if any to prevent memory leak
        if self.axis_timer is not None:
            self.axis_timer.stop()

        # Set a timer to hide the axis symbol after zooming is done
        self.axis_timer = QTimer(self)
        self.axis_timer.setSingleShot(True)
        self.axis_timer.timeout.connect(self.hide_axis_after_zoom)
        self.axis_timer.start(500)  # 500 milliseconds delay to hide the axis symbol

        # Check if LOD needs to be updated
        self._on_zoom_changed()

        self.update()

    def hide_axis_after_zoom(self):
        """Hide the axis symbol after zooming is completed."""
        self.show_axis = False
        self.update()

    def _on_zoom_changed(self):
        """Called after zoom changes to potentially update LOD.

        Note: LOD only DECREASES detail when zooming out, never increases.
        This prevents OOM when zooming in on large datasets.
        """
        if not self._lod_enabled:
            return

        controller = global_variables.global_application_controller
        if controller is None or controller.rendering_coordinator is None:
            return

        rc = controller.rendering_coordinator
        if rc.total_visible_points <= 0:
            return

        total_points = rc.total_visible_points
        point_budget = rc._point_budget

        # Calculate max safe rate (never exceed budget)
        max_safe_rate = min(1.0, point_budget / total_points) if total_points > 0 else 1.0

        new_rate = LODManager.compute_sample_rate(
            total_points,
            self.camera_distance,
            self.zoom_factor,
            self.max_extent or 1.0,
            point_budget
        )

        # Cap at max safe rate
        new_rate = min(new_rate, max_safe_rate)

        # IMPORTANT: Only re-render if DECREASING detail (zooming out)
        # Never increase detail dynamically - it risks OOM on large datasets
        # Users who want more detail should adjust point_budget or hide branches
        if new_rate < self._current_sample_rate - 0.05:
            logger.debug(f"LOD: {self._current_sample_rate:.1%} -> {new_rate:.1%} (zoom out)")
            self._current_sample_rate = new_rate
            main_window = global_variables.global_main_window
            if main_window and hasattr(main_window, 'render_visible_with_lod'):
                main_window.render_visible_with_lod(new_rate)

    def closeEvent(self, event):
        """
        Handle the close event for cleaning up resources.

        This method is called when the widget is about to be closed. It unbinds and deletes the vertex buffer object
        (VBO) if it exists, ensuring that all allocated resources are properly released before the widget is closed.

        Args:
            event (QCloseEvent): The close event containing details about the widget being closed.
        """

        # Make the OpenGL context current
        self.makeCurrent()

        if self.vbo is not None:
            self.vbo.unbind()
            self.vbo.delete()
            self.vbo = None  # Remove the reference to the VBO

        # Now, delete OpenGL resources explicitly
        self.deleteOpenGLResources()

        # Call the parent class's closeEvent
        super().closeEvent(event)

    def deleteOpenGLResources(self):
        """
        Clean up OpenGL resources.

        This method is called during closeEvent to ensure all OpenGL resources
        are properly released. Override this method to add additional cleanup
        for any custom OpenGL resources.
        """
        # Additional OpenGL resource cleanup can be added here if needed
        pass

    def keyPressEvent(self, event):
        """
        Handle key press events for interaction with the point cloud.

        This method processes key press events to allow specific actions, such as deselecting all picked points. If
        the Escape key is pressed, a confirmation dialog is displayed, and if confirmed, all selected points are
        deselected.

        Args:
            event (QKeyEvent): The key event containing details such as the key pressed.
        """

        if event.key() == Qt.Key_P:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+P: Enter polygon deselect mode
                if self._polygon_mode:
                    self.exit_polygon_mode()
                else:
                    self.enter_polygon_deselect_mode()
            else:
                # P: Toggle polygon selection mode
                if self._polygon_mode:
                    self.exit_polygon_mode()
                else:
                    self.enter_polygon_mode()
        elif event.key() == Qt.Key_Escape:
            if self._polygon_mode:
                # Cancel polygon mode without selecting
                self.exit_polygon_mode()
            else:
                # Create a confirmation dialog box
                reply = QMessageBox.question(self, 'Confirmation', 'Deselect all selected points?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # Clear all selected points if confirmed
                    self.picked_points_indices.clear()
                    self._selection_polygons.clear()
                    self.update()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key_F:
            self.zoom_to_extent()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            new_size = min(self._point_size * self._POINT_SIZE_FACTOR, self._POINT_SIZE_MAX)
            self.point_size = new_size
        elif event.key() == Qt.Key_Minus:
            new_size = max(self._point_size / self._POINT_SIZE_FACTOR, self._POINT_SIZE_MIN)
            self.point_size = new_size
        elif event.key() == Qt.Key_C:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("cut_cluster")
        elif event.key() == Qt.Key_M:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("merge_clusters")
        elif event.key() == Qt.Key_Delete:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("remove_clusters")

        # Ensure the parent class handles other key events
        super(PCDViewerWidget, self).keyPressEvent(event)

    def pick_point(self, mouse_pos, select=True):
        """
        Handle point picking or deselecting points in the point cloud.

        This method is used to pick or deselect points in the point cloud based on a mouse click position. It uses
        OpenGL to project the clicked point onto the screen space and determines whether a point in the point cloud
        is close enough to be picked or deselected. If `select` is True, the method attempts to pick a point;
        otherwise, it attempts to deselect a point.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
            select (bool, optional): A flag indicating whether to select (True) or deselect (False) the point. Defaults
                to True.
        """

        if select:
            self.select_point_at(mouse_pos)
        else:
            self.deselect_point_at(mouse_pos)

    def project_to_screen(self, point_3d):
        """
        Project a 3D point onto the screen space.

        This method takes a 3D point in the world coordinate system and projects it onto the 2D screen space using the
        current model-view matrix, projection matrix, and viewport settings. The resulting screen coordinates can be
        used for tasks such as selecting or highlighting points in the point cloud.

        Args:
            point_3d (numpy.ndarray): A 3-element array representing the (x, y, z) coordinates of the point to be
                projected.

        Returns:
            numpy.ndarray: A 3-element array representing the (x, y, z) coordinates of the projected point in screen
            space.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Use gluProject to project the point
        screen_pos = gluProject(
            point_3d[0], point_3d[1], point_3d[2],
            modelview, projection, viewport
        )
        return np.array(screen_pos)

    def draw_sphere(self, position, radius, slices=16, stacks=16):
        """
        Draw a sphere at the specified position.

        This method uses OpenGL to render a sphere at the given 3D position in the point cloud space. The sphere
        is often used to highlight specific points, such as those that have been picked by the user. The appearance
        of the sphere can be customised using the radius, slices, and stacks parameters.

        Args:
            position (tuple or list or numpy.ndarray): A 3-element array representing the (x, y, z) coordinates of
                the sphere's centre.
            radius (float): The radius of the sphere to be drawn.
            slices (int, optional): The number of subdivisions around the Z-axis (similar to lines of longitude).
                Defaults to 16.
            stacks (int, optional): The number of subdivisions along the Z-axis (similar to lines of latitude).
                Defaults to 16.
        """

        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)  # Use GLU_LINE for wireframe
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def select_point_at(self, mouse_pos):
        """
        Select a point in the point cloud at the given mouse position.

        This method is used to select a point in the point cloud based on the mouse click position in widget coordinates.
        It reads the depth buffer to get the depth value at the mouse position and then unprojects the screen coordinates
        to world coordinates. The closest point to the unprojected coordinates is selected if it lies within a specified
        threshold.

        The distance threshold used for selecting points is defined by the `picking_point_threshold_factor` attribute.
        You can adjust this attribute to control how close a point must be to be considered selectable.

        If a point is successfully selected, its index is added to the `picked_points_indices` attribute, which stores the
        indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Get the window coordinates
        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()  # Invert Y coordinate

        # Read the depth value at the mouse position
        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        # Handle cases where the depth value is 1.0 (background)
        if win_z == 1.0:
            return

        # Unproject the window coordinates to get the world coordinates
        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        pick_point = np.array(world_coords[:3])

        # Set a threshold to determine if a point is close enough to be considered picked
        threshold = self.max_extent * self.picking_point_threshold_factor

        # Find the closest point using KDTree for O(log n) performance
        if self._kdtree is not None:
            min_distance, min_distance_index = self._kdtree.query(pick_point, k=1)
        else:
            # Fallback to brute-force if KDTree not available
            distances = np.linalg.norm(self.points[:, :3] - pick_point, axis=1)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

        if min_distance < threshold:
            # Filter: only allow selection within the selected branch
            if not self._is_index_in_selected_branch(min_distance_index):
                return
            # Filter: skip points in clusters locked against selection
            if self._is_point_selection_locked(min_distance_index):
                return
            # Filter: skip noise points (cluster label == -1)
            if self._is_noise_point(min_distance_index):
                return
            # Add the index to the list of picked points
            if min_distance_index not in self.picked_points_indices:
                self.picked_points_indices.append(min_distance_index)

    def deselect_point_at(self, mouse_pos):
        """
        Deselect a point in the point cloud at the given mouse position.

        This method is used to deselect a previously selected point in the point cloud based on the mouse click position
        in widget coordinates. It projects the picked points to screen space and determines the closest point to the
        mouse click position. If the closest point lies within a specified pixel threshold, it is deselected.

        The pixel threshold used for deselecting points is defined by the `pixel_threshold` attribute. You can adjust
        this attribute to control how close a point must be to be considered for deselection.

        If a point is successfully deselected, its index is removed from the `picked_points_indices` attribute, which
        stores the indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        viewport = self.viewport

        # Convert mouse position to screen space coordinates
        click_x = mouse_pos.x()
        click_y = viewport[3] - mouse_pos.y()  # Invert Y coordinate (adjust if necessary)

        # Project picked points to screen space
        screen_positions = []
        for index in self.picked_points_indices:
            point_3d = self.points[index, :3]
            screen_pos = self.project_to_screen(point_3d)
            screen_positions.append((index, screen_pos))

        # Find the closest picked point to the mouse click
        min_distance = float('inf')
        closest_index = None
        for index, screen_pos in screen_positions:
            dx = screen_pos[0] - click_x
            dy = screen_pos[1] - click_y
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
                closest_index = index

        # Define a threshold in pixels (e.g., radius of the sphere in screen space)
        # TODO: Not sure if the pixel threshold is appropriate for all cases
        pixel_threshold = self.pixel_threshold

        if min_distance <= pixel_threshold:
            # Remove the point from picked points
            self.picked_points_indices.remove(closest_index)
            # Invalidate stored polygons so plugins fall back to coordinate matching
            self._selection_polygons.clear()

    def deselect_cluster_at(self, mouse_pos):
        """
        Deselect all selected points that share the same color as the clicked point.

        This removes an entire cluster from the selection by matching the clicked point's
        RGB color against the colors of all currently selected points. Points with matching
        colors are removed from picked_points_indices.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """
        if not self.picked_points_indices:
            return

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Get the window coordinates
        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()

        # Read the depth value at the mouse position
        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        if win_z == 1.0:
            return

        # Unproject to world coordinates
        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        pick_point = np.array(world_coords[:3])

        # Find the closest point using KDTree
        threshold = self.max_extent * self.picking_point_threshold_factor
        if self._kdtree is not None:
            min_distance, clicked_index = self._kdtree.query(pick_point, k=1)
        else:
            distances = np.linalg.norm(self.points[:, :3] - pick_point, axis=1)
            clicked_index = np.argmin(distances)
            min_distance = distances[clicked_index]

        if min_distance >= threshold:
            return

        # Get the color of the clicked point
        target_color = self.points[clicked_index, 3:6]

        # Vectorized removal of all selected points matching this color
        selected = np.array(self.picked_points_indices, dtype=np.int64)
        colors = self.points[selected, 3:6]
        matches = np.all(colors == target_color, axis=1)
        self.picked_points_indices[:] = selected[~matches].tolist()
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.update()

    def update_rotation_center(self, mouse_pos):
        """
        Update the centre of rotation based on the given mouse position.

        This method is used to update the centre of rotation of the point cloud view based on a double-click event
        at a specific mouse position. It reads the depth value at the mouse position, unprojects the screen coordinates
        to world coordinates, and sets the centre of rotation to the closest point in the point cloud if it lies within
        a specified threshold.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Get the window coordinates
        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()  # Invert Y coordinate

        # Read the depth value at the mouse position
        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        # Handle cases where the depth value is 1.0 (background)
        if win_z == 1.0:
            # No depth information; do not update center
            return

        # Unproject the window coordinates to get the world coordinates
        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        click_point = np.array(world_coords[:3])

        # Set a threshold to determine if a point is close enough
        threshold = self.max_extent * self.picking_point_threshold_factor

        # Find the closest point using KDTree for O(log n) performance
        if self._kdtree is not None:
            min_distance, min_distance_index = self._kdtree.query(click_point, k=1)
        else:
            # Fallback to brute-force if KDTree not available
            distances = np.linalg.norm(self.points[:, :3] - click_point, axis=1)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

        if min_distance < threshold:
            # Save old center before updating
            old_center = self.center.copy()
            new_center = self.points[min_distance_index, :3].copy()

            # Adjust pan to compensate for center change (prevents view shift)
            # The transformation is: pan + center, so to keep the same view:
            # new_pan + new_center = old_pan + old_center
            # new_pan = old_pan + (old_center - new_center)
            self.pan_x += old_center[0] - new_center[0]
            self.pan_y += old_center[1] - new_center[1]
            self.pan_z += old_center[2] - new_center[2]

            # Update the center to the new point
            self.center = new_center

    def reset_view(self):
        """
        Reset the camera to its default position and orientation.

        This method restores the default settings for the camera, including zoom, rotation, and panning,
        allowing the user to return to the initial view of the point cloud.
        Specifically, the following attributes are reset:

        - `zoom_factor`: Set to `default_zoom_factor` to restore the original zoom level.
        - `rot_x`, `rot_y`, `rot_z`: Rotation angles around the X, Y, and Z axes are reset to default values, default_rot_x, etc.
        - `pan_x`, `pan_y`, `pan_z`: Panning offsets along the X, Y, and Z axes are reset to default values, default_pan_x.
        - `camera_distance`: Set to `default_camera_distance` to restore the original distance from the point cloud.

        After resetting these parameters, the view is updated to reflect the changes.
        """

        self.zoom_factor = self.default_zoom_factor
        self.rot_x = self.default_rot_x
        self.rot_y = self.default_rot_y
        self.rot_z = self.default_rot_z
        self.pan_x = self.default_pan_x
        self.pan_y = self.default_pan_y
        self.pan_z = self.default_pan_z
        self.camera_distance = self.default_camera_distance

        self.center[0] = -self.default_pan_x
        self.center[1] = -self.default_pan_y
        self.center[2] = -self.default_pan_z

        self.update()

    def zoom_to_extent(self, preserve_rotation=False):
        """
        Zoom the camera to fit all visible points in the viewport.

        This method calculates the bounding box of the currently displayed point cloud
        and adjusts the camera parameters to frame the entire dataset optimally.
        It recenters the view on the data and calculates an appropriate camera distance.

        Args:
            preserve_rotation: If True, keeps current rotation angles (camera angle).
                             If False, resets rotation to default values.

        The following parameters are adjusted:
        - `center`: Set to the center of the point cloud bounding box
        - `camera_distance`: Calculated based on bounding box size and FOV
        - `zoom_factor`: Reset to 1.0
        - `rot_x`, `rot_y`, `rot_z`: Reset to default (or preserved if preserve_rotation=True)
        - `pan_x`, `pan_y`, `pan_z`: Adjusted to center the view
        - Also updates default values for reset_view functionality

        After adjusting parameters, the view is updated to reflect the changes.
        """
        logger.debug("zoom_to_extent() called")

        # Check if points are available
        if self.points is None or len(self.points) == 0:
            logger.debug("  No points to zoom to")
            return

        logger.debug(f"  Processing {len(self.points)} points")

        try:
            # Calculate bounding box of visible points
            logger.debug("  Calculating bounding box...")
            min_bounds = np.min(self.points[:, :3], axis=0)
            max_bounds = np.max(self.points[:, :3], axis=0)

            # Calculate center, size, and maximum extent
            self.center = (min_bounds + max_bounds) / 2.0
            self.size = max_bounds - min_bounds
            self.max_extent = np.max(self.size)

            logger.debug(f"  Center: {self.center}")
            logger.debug(f"  Size: {self.size}")
            logger.debug(f"  Max extent: {self.max_extent}")

            # Avoid division by zero for degenerate cases
            if self.max_extent == 0:
                self.max_extent = 1.0

            # Calculate optimal camera distance based on FOV and bounding box
            half_fov_rad = np.radians(self.fov / 2)
            self.default_camera_distance = self.max_extent / (2 * np.tan(half_fov_rad)) * 1.2  # 20% padding
            self.camera_distance = self.default_camera_distance

            logger.debug(f"  Camera distance: {self.camera_distance}")

            # Update panning offsets to align the center with the view
            self.pan_x = -self.center[0]
            self.pan_y = -self.center[1]
            self.pan_z = -self.center[2]

            # Save these values as defaults for reset_view
            self.default_pan_x = self.pan_x
            self.default_pan_y = self.pan_y
            self.default_pan_z = self.pan_z

            # Reset rotation to default values (unless preserving rotation)
            if not preserve_rotation:
                self.rot_x = self.default_rot_x
                self.rot_y = self.default_rot_y
                self.rot_z = self.default_rot_z

            # Reset zoom factor
            self.zoom_factor = 1.0
            self.default_zoom_factor = 1.0

            # Show axis briefly for orientation
            self.show_axis = True

            # Note: Do NOT reset _current_sample_rate here - it was set by DataManager
            # during render to respect the point budget. Resetting would cause
            # _on_zoom_changed to trigger an unnecessary (and potentially OOM) re-render.

            # Update the view
            logger.debug("  Updating view...")
            self.update()

            # Set a timer to hide the axis after a short time
            if hasattr(self, 'axis_timer') and self.axis_timer is not None:
                self.axis_timer.stop()

            self.axis_timer = QTimer(self)
            self.axis_timer.setSingleShot(True)
            self.axis_timer.timeout.connect(self.hide_axis_after_zoom)
            self.axis_timer.start(500)  # 500ms delay

            logger.debug("  zoom_to_extent() completed")

        except Exception as e:
            logger.error(f"  ERROR in zoom_to_extent(): {e}")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise

    def show_point_cloud(self, visible=True):
        """Show or hide the point cloud by setting visibility and updating the view."""
        if self.visible != visible:
            self.visible = visible
            self.update()

    # ── Branch-scoped selection helpers ─────────────────────────────────

    def set_branch_offsets(self, offsets: dict):
        """Store per-branch index ranges for selection filtering."""
        self._branch_offsets = offsets

    def _is_index_in_selected_branch(self, index: int) -> bool:
        """Check if a point index belongs to one of the selected branches."""
        if not self._branch_offsets:
            return True
        controller = global_variables.global_application_controller
        if controller is None:
            return True
        selected = controller.selected_branches
        if not selected:
            return True
        for uid in selected:
            rng = self._branch_offsets.get(uid)
            if rng and rng[0] <= index < rng[1]:
                return True
        return False

    def _get_selected_branch_index_range(self):
        """Return (start, end) tuples for all selected branches, or None if no filtering."""
        if not self._branch_offsets:
            return None
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        selected = controller.selected_branches
        if not selected:
            return None
        ranges = []
        for uid in selected:
            rng = self._branch_offsets.get(uid)
            if rng:
                ranges.append(rng)
        return ranges if ranges else None

    def _get_cluster_lock_info(self, uid):
        """Get (labels, locked_clusters) for a cluster_labels branch, or None."""
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        node = controller.get_node(uid)
        if node is None or node.data_type != "cluster_labels":
            return None
        clusters = node.data
        if not getattr(clusters, 'locked_clusters', None):
            return None
        # Check if any cluster is locked against selection
        has_select_lock = any("select" in locks for locks in clusters.locked_clusters.values())
        if not has_select_lock:
            return None
        return clusters.labels, clusters.locked_clusters

    def _is_point_selection_locked(self, index: int) -> bool:
        """Check if a single point index belongs to a cluster locked against selection."""
        if not self._branch_offsets:
            return False
        for uid, (start, end) in self._branch_offsets.items():
            if start <= index < end:
                info = self._get_cluster_lock_info(uid)
                if info is None:
                    return False
                labels, locked = info
                local_idx = index - start
                if local_idx < len(labels):
                    cid = int(labels[local_idx])
                    return "select" in locked.get(cid, set())
                return False
        return False

    def _filter_selection_locked(self, indices):
        """Filter out indices belonging to clusters locked against selection."""
        import numpy as np
        if not self._branch_offsets:
            return indices
        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            info = self._get_cluster_lock_info(uid)
            if info is None:
                continue
            labels, locked = info
            locked_ids = {cid for cid, locks in locked.items() if "select" in locks}
            for i, idx in enumerate(indices):
                if start <= idx < end:
                    local_idx = idx - start
                    if local_idx < len(labels) and int(labels[local_idx]) in locked_ids:
                        keep_mask[i] = False
        return indices[keep_mask]

    def _get_cluster_labels(self, uid):
        """Get cluster labels array for a cluster_labels branch, or None."""
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        node = controller.get_node(uid)
        if node is None or node.data_type != "cluster_labels":
            return None
        return getattr(node.data, 'labels', None)

    def _is_noise_point(self, index: int) -> bool:
        """Check if a point index is a noise point (cluster label == -1)."""
        if not self._branch_offsets:
            return False
        for uid, (start, end) in self._branch_offsets.items():
            if start <= index < end:
                labels = self._get_cluster_labels(uid)
                if labels is None:
                    return False
                local_idx = index - start
                if local_idx < len(labels):
                    return int(labels[local_idx]) == -1
                return False
        return False

    def _filter_noise_points(self, indices):
        """Filter out indices that are noise points (cluster label == -1)."""
        if not self._branch_offsets:
            return indices
        keep_mask = np.ones(len(indices), dtype=bool)
        for uid, (start, end) in self._branch_offsets.items():
            labels = self._get_cluster_labels(uid)
            if labels is None:
                continue
            for i, idx in enumerate(indices):
                if start <= idx < end:
                    local_idx = idx - start
                    if local_idx < len(labels) and int(labels[local_idx]) == -1:
                        keep_mask[i] = False
        return indices[keep_mask]

    # ── Polygon Selection Mode ──────────────────────────────────────────

    def enter_polygon_mode(self):
        """Activate polygon selection mode. User clicks to add vertices."""
        if self.points is None:
            return
        self._polygon_mode = True
        self._polygon_vertices = []
        self.setCursor(Qt.CrossCursor)
        self.update()

    def enter_polygon_deselect_mode(self):
        """Activate polygon deselect mode. User draws a polygon to remove points from selection."""
        if self.points is None:
            return
        self._polygon_mode = True
        self._polygon_deselect_mode = True
        self._polygon_vertices = []
        self.setCursor(Qt.CrossCursor)
        self.update()

    def exit_polygon_mode(self):
        """Deactivate polygon selection mode and restore normal cursor."""
        self._polygon_mode = False
        self._polygon_deselect_mode = False
        self._polygon_vertices = []
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _close_polygon_and_select(self):
        """Close the polygon and select all 3D points whose screen projections fall inside it."""
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        # OpenGL's glGetDoublev returns column-major matrices. In numpy (row-major)
        # these appear transposed: mv_np = ModelView^T, proj_np = Projection^T.
        # Use row-vector multiplication: clip_row = point_row @ MV^T @ P^T
        mv = np.array(self.model_view_matrix, dtype=np.float64)   # 4x4 (M^T)
        proj = np.array(self.projection_matrix, dtype=np.float64)  # 4x4 (P^T)

        # Append polygon + matrices for full-resolution re-testing by plugins
        self._selection_polygons.append((
            polygon.copy(), mv.copy(), proj.copy(), tuple(self.viewport)
        ))

        # Get all 3D points as homogeneous coordinates (row vectors)
        pts_3d = self.points[:, :3].astype(np.float64)  # (N, 3)
        n = pts_3d.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        pts_homo = np.hstack([pts_3d, ones])  # (N, 4)

        # Row-vector projection: clip = pts @ MV^T @ P^T  -> (N, 4)
        clip = pts_homo @ mv @ proj  # (N, 4)

        w = clip[:, 3]
        # Filter points behind camera (w <= 0)
        valid_mask = w > 0

        # NDC coordinates
        ndc_x = np.zeros(n, dtype=np.float64)
        ndc_y = np.zeros(n, dtype=np.float64)
        ndc_x[valid_mask] = clip[valid_mask, 0] / w[valid_mask]
        ndc_y[valid_mask] = clip[valid_mask, 1] / w[valid_mask]

        # Viewport transform: NDC -> Qt widget coordinates (top-left origin, Y down)
        vp = self.viewport
        vp_x, vp_y, vp_w, vp_h = vp[0], vp[1], vp[2], vp[3]
        screen_x = (ndc_x + 1.0) * 0.5 * vp_w + vp_x
        screen_y = (1.0 - ndc_y) * 0.5 * vp_h + vp_y  # Flip Y for Qt coords

        # Point-in-polygon test (ray casting)
        inside = self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        # Get indices of selected points
        new_indices = np.where(inside)[0]

        # Filter to selected branch range(s)
        branch_ranges = self._get_selected_branch_index_range()
        if branch_ranges is not None and new_indices.size > 0:
            mask = np.zeros(new_indices.size, dtype=bool)
            for start, end in branch_ranges:
                mask |= (new_indices >= start) & (new_indices < end)
            new_indices = new_indices[mask]

        # Filter out points in clusters locked against selection
        if new_indices.size > 0:
            new_indices = self._filter_selection_locked(new_indices)

        # Filter out noise points (cluster label == -1)
        if new_indices.size > 0:
            new_indices = self._filter_noise_points(new_indices)

        # Avoid duplicates
        if new_indices.size > 0:
            existing = set(self.picked_points_indices)
            for idx in new_indices:
                idx_int = int(idx)
                if idx_int not in existing:
                    self.picked_points_indices.append(idx_int)
                    existing.add(idx_int)

        self.exit_polygon_mode()

    def retest_polygon_selection(self, points_3d):
        """
        Re-test arbitrary 3D points against all stored polygon selections.

        Uses the stored polygons and camera matrices from polygon selections,
        so it works correctly even if the camera has moved.  Returns the union
        of all polygon tests (a point inside ANY stored polygon is True).

        Args:
            points_3d: (N, 3) float array of 3D world coordinates.

        Returns:
            (N,) boolean mask (True = inside any polygon), or None if no polygons stored.
        """
        if not self._selection_polygons:
            return None

        pts = np.asarray(points_3d, dtype=np.float64)
        n = pts.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        pts_homo = np.hstack([pts, ones])  # (N, 4)

        combined_mask = np.zeros(n, dtype=bool)

        for polygon, mv, proj, viewport in self._selection_polygons:
            vp_x, vp_y, vp_w, vp_h = viewport

            clip = pts_homo @ mv @ proj  # (N, 4)
            w = clip[:, 3]
            valid_mask = w > 0

            ndc_x = np.zeros(n, dtype=np.float64)
            ndc_y = np.zeros(n, dtype=np.float64)
            ndc_x[valid_mask] = clip[valid_mask, 0] / w[valid_mask]
            ndc_y[valid_mask] = clip[valid_mask, 1] / w[valid_mask]

            screen_x = (ndc_x + 1.0) * 0.5 * vp_w + vp_x
            screen_y = (1.0 - ndc_y) * 0.5 * vp_h + vp_y

            combined_mask |= self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        return combined_mask

    def _close_polygon_and_deselect(self):
        """Close the polygon and remove all points inside it from the current selection."""
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        if not self.picked_points_indices:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)

        # Only project the currently selected points (not all points)
        selected_indices = np.array(self.picked_points_indices, dtype=np.int64)
        pts_3d = self.points[selected_indices, :3].astype(np.float64)
        n = pts_3d.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        pts_homo = np.hstack([pts_3d, ones])

        clip = pts_homo @ mv @ proj

        w = clip[:, 3]
        valid_mask = w > 0

        ndc_x = np.zeros(n, dtype=np.float64)
        ndc_y = np.zeros(n, dtype=np.float64)
        ndc_x[valid_mask] = clip[valid_mask, 0] / w[valid_mask]
        ndc_y[valid_mask] = clip[valid_mask, 1] / w[valid_mask]

        vp = self.viewport
        vp_x, vp_y, vp_w, vp_h = vp[0], vp[1], vp[2], vp[3]
        screen_x = (ndc_x + 1.0) * 0.5 * vp_w + vp_x
        screen_y = (1.0 - ndc_y) * 0.5 * vp_h + vp_y

        inside = self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        # Remove points that fall inside the polygon from the selection
        indices_to_remove = set(int(selected_indices[i]) for i in np.where(inside)[0])
        self.picked_points_indices[:] = [i for i in self.picked_points_indices if i not in indices_to_remove]
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.exit_polygon_mode()

    @staticmethod
    def _points_in_polygon(screen_x, screen_y, polygon, valid_mask):
        """
        Vectorized ray-casting point-in-polygon test.

        Args:
            screen_x: (N,) array of screen X coordinates
            screen_y: (N,) array of screen Y coordinates
            polygon: (M, 2) array of polygon vertices in screen coords
            valid_mask: (N,) boolean mask for points in front of camera

        Returns:
            (N,) boolean array — True for points inside the polygon
        """
        n = len(screen_x)
        m = len(polygon)
        inside = np.zeros(n, dtype=bool)

        # Ray casting: count crossings for each point
        crossings = np.zeros(n, dtype=np.int32)
        for i in range(m):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % m]

            # For each edge, test all valid points simultaneously
            # A ray from (px, py) going right crosses edge if:
            #   1. The edge straddles the ray's y-level
            #   2. The intersection x is to the right of px
            cond1 = (y1 <= screen_y) & (y2 > screen_y)
            cond2 = (y2 <= screen_y) & (y1 > screen_y)
            straddle = cond1 | cond2

            # Compute x-intersection of the edge with the horizontal ray
            # x_intersect = x1 + (screen_y - y1) / (y2 - y1) * (x2 - x1)
            test = straddle & valid_mask
            if not np.any(test):
                continue

            dy = y2 - y1
            if dy == 0:
                continue

            t = (screen_y[test] - y1) / dy
            x_intersect = x1 + t * (x2 - x1)

            crosses = x_intersect > screen_x[test]
            crossings[test] += crosses.astype(np.int32)

        # Odd number of crossings = inside
        inside = (crossings % 2 == 1) & valid_mask
        return inside

    def render_polygon_overlay(self):
        """Draw the polygon as a 2D overlay on top of the 3D scene."""
        if not self._polygon_mode or len(self._polygon_vertices) == 0:
            return

        w = self.width()
        h = self.height()

        # Save all GL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # Switch to 2D orthographic projection matching Qt widget coords
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)  # Top-left origin, Y-down (Qt convention)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        verts = self._polygon_vertices

        # Choose color based on mode: red for deselect, blue for select
        if self._polygon_deselect_mode:
            fill_color = (1.0, 0.2, 0.2, 0.15)
            edge_color = (1.0, 0.2, 0.2, 0.8)
        else:
            fill_color = (0.2, 0.4, 1.0, 0.15)
            edge_color = (0.2, 0.4, 1.0, 0.8)

        # Draw semi-transparent filled polygon (if >= 3 vertices)
        if len(verts) >= 3:
            glColor4f(*fill_color)
            glBegin(GL_POLYGON)
            for x, y in verts:
                glVertex2f(x, y)
            glEnd()

        # Draw polygon edge lines
        glColor4f(*edge_color)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for x, y in verts:
            glVertex2f(x, y)
        glEnd()

        # Draw closing edge (dotted visual hint) if >= 3 vertices
        if len(verts) >= 3:
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(1, 0x00FF)
            glBegin(GL_LINES)
            glVertex2f(verts[-1][0], verts[-1][1])
            glVertex2f(verts[0][0], verts[0][1])
            glEnd()
            glDisable(GL_LINE_STIPPLE)

        # Draw vertex dots
        glColor4f(1.0, 1.0, 0.0, 1.0)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        for x, y in verts:
            glVertex2f(x, y)
        glEnd()

        # Restore GL state
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glPopAttrib()
