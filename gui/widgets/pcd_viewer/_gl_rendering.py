import logging
import traceback
import numpy as np
from OpenGL.GL import (
    glClearColor, glEnable, glClear,
    GL_DEPTH_TEST, GL_VERTEX_PROGRAM_POINT_SIZE, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glMatrixMode, glLoadIdentity, glTranslatef, glRotatef,
    GL_PROJECTION, GL_MODELVIEW,
    glGetDoublev, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX,
    glGetIntegerv, GL_VIEWPORT,
    glPointSize, glEnableClientState, glDisableClientState,
    GL_VERTEX_ARRAY, GL_COLOR_ARRAY,
    glVertexPointer, glColorPointer, glDrawArrays, glDrawElements,
    GL_FLOAT, GL_POINTS, GL_UNSIGNED_INT,
    glColor3f, glBegin, glEnd, glVertex3f,
    GL_LINES, glLineWidth,
    glPushMatrix, glPopMatrix,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric, gluQuadricDrawStyle
from OpenGL.GLU import gluSphere
from OpenGL.GLU import GLU_FILL
from OpenGL.arrays import vbo

logger = logging.getLogger(__name__)


class GLRenderingMixin:
    """OpenGL initialization, painting, and resource management for PCDViewerWidget."""

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

    def paintGL(self):
        """
        Render the point cloud and other visual elements.

        This method is called whenever the widget needs to be repainted. It clears the colour and depth buffers, sets
        up the projection and model-view matrices, and renders the point cloud data, picked points, and optionally the
        axis symbol. The rendering includes applying transformations for panning, zooming, and rotation.

        If no point cloud data is set, the method returns without rendering anything.
        """

        has_points = self.points is not None
        has_lines = self.line_vertices is not None and self.line_indices is not None
        if (not has_points and not has_lines) or self.max_extent is None:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w = self.width()
        h = self.height()
        aspect = w / h if h != 0 else 1
        # Cap FOV at base value so zooming out (zoom_factor > 1) only moves
        # the camera back without widening FOV, preventing fisheye distortion.
        effective_fov = self.fov * min(self.zoom_factor, 1.0)
        # Dynamic far plane: ensure it covers the camera distance + scene extent
        effective_distance = self.camera_distance * self.zoom_factor
        far_plane = max(self.far_plane, effective_distance + (self.max_extent or 0) * 2)
        gluPerspective(effective_fov, aspect, max(self.near_plane, 0.1), far_plane)

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

        # Render line geometry (e.g. mesh wireframes)
        self.render_lines()

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

        # Draw zoom window overlay (2D rubber band rectangle)
        self.render_zoom_window_overlay()

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

    def render_lines(self):
        """
        Render stored line geometry as GL_LINES using client-side vertex arrays.

        One glDrawElements call walks the index buffer in C — no Python
        iteration — so dense wireframes (drape meshes with tens of thousands of
        edges) stay interactive during navigation.
        """
        if self.line_vertices is None or self.line_indices is None:
            return

        glLineWidth(1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.line_vertices)

        colors = self.line_colors
        if colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, colors)
        else:
            glColor3f(0.85, 0.85, 0.85)

        glDrawElements(GL_LINES, self.line_indices.size, GL_UNSIGNED_INT,
                       self.line_indices)

        if colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

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
                highlight_size = self.point_size * self.picked_point_highlight_size * self._PICKED_POINT_SIZE_MULTIPLIER
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
        effective_fov = self.fov * min(self.zoom_factor, 1.0)
        effective_distance = self.camera_distance * self.zoom_factor
        far_plane = max(self.far_plane, effective_distance + (self.max_extent or 0) * 2)
        gluPerspective(effective_fov, aspect, max(self.near_plane, 0.1), far_plane)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
