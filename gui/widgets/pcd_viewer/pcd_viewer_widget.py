from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt

from ._properties import ViewerPropertiesMixin
from ._gl_rendering import GLRenderingMixin
from ._data_management import DataManagementMixin
from ._mouse_input import MouseInputMixin
from ._key_input import KeyInputMixin
from ._point_picking import PointPickingMixin
from ._camera_control import CameraControlMixin
from ._branch_helpers import BranchSelectionMixin
from ._polygon_selection import PolygonSelectionMixin
from ._zoom_window import ZoomWindowMixin
from ._overlay_rendering import OverlayRenderingMixin


class PCDViewerWidget(
    ViewerPropertiesMixin,
    GLRenderingMixin,
    DataManagementMixin,
    MouseInputMixin,
    KeyInputMixin,
    PointPickingMixin,
    CameraControlMixin,
    BranchSelectionMixin,
    PolygonSelectionMixin,
    ZoomWindowMixin,
    OverlayRenderingMixin,
    QOpenGLWidget,
):
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
        - Z: Toggle zoom window mode (drag a rectangle to zoom into that region).
        - ESC: Cancel zoom window / polygon mode (if active), or deselect all selected points after confirmation.
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

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set focus policy to ensure the widget receives keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Order matters: _init_properties sets defaults used by _init_camera
        self._init_properties()
        self._init_camera()
        self._init_data()
        self._init_mouse_state()
        self._init_branch_state()
        self._init_polygon_state()
        self._init_zoom_window_state()
