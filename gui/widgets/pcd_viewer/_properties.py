import logging

from config.config import global_variables

logger = logging.getLogger(__name__)


class ViewerPropertiesMixin:
    """Property getters/setters for PCDViewerWidget."""

    def _init_properties(self):
        """Initialize property backing fields."""
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

        # Internal constants for interaction scaling
        self._ROTATION_SENSITIVITY_DIVISOR = 10
        self._PAN_SENSITIVITY_DIVISOR = 10000
        self._ZOOM_WHEEL_DIVISOR = 1000
        self._AXIS_DISPLAY_DURATION_MS = 500
        self._CAMERA_DISTANCE_PADDING = 1.2
        self._MIN_ZOOM_WINDOW_SIZE_PX = 10
        self._LOD_RATE_CHANGE_THRESHOLD = 0.05
        self._PICKED_POINT_SIZE_MULTIPLIER = 5

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
