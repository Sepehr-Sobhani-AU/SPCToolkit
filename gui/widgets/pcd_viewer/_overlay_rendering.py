import logging

from OpenGL.GL import (
    glPushAttrib, glPopAttrib, GL_ALL_ATTRIB_BITS,
    glMatrixMode, glPushMatrix, glPopMatrix, glLoadIdentity,
    GL_PROJECTION, GL_MODELVIEW,
    glOrtho,
    glDisable, glEnable, GL_DEPTH_TEST, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glColor4f,
    glBegin, glEnd, glVertex2f,
    GL_POLYGON, GL_LINE_STRIP, GL_LINES, GL_POINTS, GL_QUADS, GL_LINE_LOOP,
    glLineWidth, glPointSize,
    glEnable as glEnableStipple, GL_LINE_STIPPLE, glLineStipple,
    glDisable as glDisableStipple,
)

logger = logging.getLogger(__name__)


class OverlayRenderingMixin:
    """2D overlay rendering for polygon selection and zoom window."""

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

    def render_zoom_window_overlay(self):
        """Draw the zoom window rubber band rectangle as a 2D overlay."""
        if not self._zoom_window_mode:
            return
        if self._zoom_window_start is None or self._zoom_window_current is None:
            return

        x1, y1 = self._zoom_window_start
        x2, y2 = self._zoom_window_current

        w = self.width()
        h = self.height()

        glPushAttrib(GL_ALL_ATTRIB_BITS)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Semi-transparent green fill
        glColor4f(0.2, 1.0, 0.2, 0.12)
        glBegin(GL_QUADS)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()

        # Green border
        glColor4f(0.2, 1.0, 0.2, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glPopAttrib()
