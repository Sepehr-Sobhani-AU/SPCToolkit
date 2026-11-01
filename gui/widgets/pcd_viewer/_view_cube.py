"""View cube: a small labelled cube in the viewer's top-right corner.

It turns with the camera, so it always shows which way the view is facing.
Double-clicking one of its faces turns the camera to look straight at that side
of the scene — Top, Bottom, Front, Back, Left or Right — with a short animation.
The point the view turns around (the centre of rotation) stays where it is.

It is a see-through child widget laid over the 3D view rather than something
drawn into the GL scene: Qt5 composes child widgets over a QOpenGLWidget, and
that lets the labels be drawn with QPainter and the cube take its own clicks
without touching the GL drawing code.

Mouse handling: presses, drags and releases are passed straight on to the
viewer (``event.ignore()``), so dragging over the cube orbits the view like
anywhere else. Only a double-click on a face is kept. A double-click on the
cube's empty corners is passed on too, so it still sets the centre of rotation.

Orientation: paintGL turns the scene by Rx(rot_x) · Ry(rot_y) · Rz(rot_z), and
the cube uses the same rotation. Camera looks down -Z of eye space, so a face
is visible when its turned normal has a positive Z.
"""

import math

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QPainter, QPolygonF, QColor, QPen, QBrush, QFont, QTransform
from PyQt5.QtWidgets import QWidget

# Each face: label, outward normal, the world directions that are screen-right
# and screen-up when looking straight at it, and the camera angles
# (rot_x, rot_z) of that view — rot_y is always 0. World is Z-up; Front looks
# from -Y towards +Y.
_FACES = [
    ("TOP",    (0, 0, 1),  (1, 0, 0),  (0, 1, 0),  (0.0, 0.0)),
    ("BOTTOM", (0, 0, -1), (1, 0, 0),  (0, -1, 0), (-180.0, 0.0)),
    ("FRONT",  (0, -1, 0), (1, 0, 0),  (0, 0, 1),  (-90.0, 0.0)),
    ("BACK",   (0, 1, 0),  (-1, 0, 0), (0, 0, 1),  (-90.0, 180.0)),
    ("RIGHT",  (1, 0, 0),  (0, 1, 0),  (0, 0, 1),  (-90.0, -90.0)),
    ("LEFT",   (-1, 0, 0), (0, -1, 0), (0, 0, 1),  (-90.0, 90.0)),
]

_SIZE_PX = 120          # widget is a square this wide
_MARGIN_PX = 10         # gap to the viewer's top and right edges
_CUBE_SCALE = 0.30      # half a cube edge, as a fraction of the widget size
_ANIM_MS = 300          # how long a turn to a face takes
_ANIM_STEP_MS = 15
_LABEL_BOX = 100.0      # face label drawn in a 100 x 100 box mapped onto the face


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrap(deg):
    """Angle brought into (-180, 180]."""
    deg = math.fmod(deg, 360.0)
    if deg > 180.0:
        deg -= 360.0
    elif deg <= -180.0:
        deg += 360.0
    return deg


class ViewCube(QWidget):
    """The view cube overlay. Created by the viewer; reads and sets its camera
    angles directly."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self._viewer = viewer
        self._hover = None          # label of the face under the cursor
        self._faces = []            # [(label, QPolygonF, facing)] from the last paint
        self._last_angles = None

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_from = None
        self._anim_to = None
        self._anim_t = 0.0

        self.setFixedSize(_SIZE_PX, _SIZE_PX)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.place()

    # ------------------------------------------------------------ placement
    def place(self):
        """Keep the cube in the viewer's top-right corner."""
        self.move(max(self._viewer.width() - _SIZE_PX - _MARGIN_PX, 0), _MARGIN_PX)

    def sync(self):
        """Repaint if the camera has turned since the last paint. Called from
        the viewer's paintGL; cheap when nothing changed."""
        v = self._viewer
        angles = (v.rot_x, v.rot_y, v.rot_z)
        if angles != self._last_angles:
            self._last_angles = angles
            self.update()

    # ------------------------------------------------------------ geometry
    def _projected_faces(self):
        """Visible faces as ``(label, QPolygonF, facing)``, far ones first.

        *facing* is how squarely the face looks at the camera (0..1), used for
        shading. Corner order is top-left, top-right, bottom-right, bottom-left
        of the face's own label, so the label box maps straight onto it.
        """
        v = self._viewer
        R = _rot_x(v.rot_x) @ _rot_y(v.rot_y) @ _rot_z(v.rot_z)
        half = _SIZE_PX * _CUBE_SCALE
        cx = cy = _SIZE_PX / 2.0

        out = []
        for label, n, u, w, _view in _FACES:
            n_eye = R @ np.asarray(n, dtype=float)
            if n_eye[2] <= 1e-3:
                continue                                     # facing away
            c, u, w = np.asarray(n, float), np.asarray(u, float), np.asarray(w, float)
            corners = (c - u + w, c + u + w, c + u - w, c - u - w)
            poly = QPolygonF()
            for p in corners:
                e = R @ p
                poly.append(QPointF(cx + half * e[0], cy - half * e[1]))
            out.append((float((R @ c)[2]), label, poly, float(n_eye[2])))

        out.sort(key=lambda t: t[0])
        return [(label, poly, facing) for _z, label, poly, facing in out]

    def _face_at(self, pos):
        point = QPointF(pos)
        for label, poly, _facing in reversed(self._faces):   # nearest first
            if poly.containsPoint(point, Qt.OddEvenFill):
                return label
        return None

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event):
        self._faces = self._projected_faces()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        font = QFont(self.font())
        font.setBold(True)
        font.setPixelSize(20)
        label_box = QPolygonF([QPointF(0, 0), QPointF(_LABEL_BOX, 0),
                               QPointF(_LABEL_BOX, _LABEL_BOX), QPointF(0, _LABEL_BOX)])

        for label, poly, facing in self._faces:
            shade = int(150 + 80 * facing)                   # lit by the camera
            fill = QColor(120, 170, 235, 235) if label == self._hover \
                else QColor(shade, shade, shade, 215)
            painter.setTransform(QTransform())
            painter.setPen(QPen(QColor(60, 60, 60), 1.2))
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(poly)

            # Label laid on the face, so it tilts and turns with it. Faces seen
            # nearly edge-on are skipped — their text is unreadable anyway.
            if facing < 0.25:
                continue
            t = QTransform()
            if QTransform.quadToQuad(label_box, poly, t):
                painter.setTransform(t)
                painter.setFont(font)
                painter.setPen(QColor(30, 30, 30))
                painter.drawText(QRectF(0, 0, _LABEL_BOX, _LABEL_BOX),
                                 Qt.AlignCenter, label)
        painter.end()

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, event):
        event.ignore()                      # dragging over the cube orbits as usual

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        # Only hover reaches here: a drag belongs to the viewer, which took the
        # press. Not passed on, so hovering does not redraw the whole cloud.
        face = self._face_at(event.pos())
        if face != self._hover:
            self._hover = face
            self.update()

    def leaveEvent(self, _event):
        if self._hover is not None:
            self._hover = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        face = self._face_at(event.pos()) if event.button() == Qt.LeftButton else None
        if face is None:
            event.ignore()                  # viewer sets the centre of rotation
            return
        self.turn_to(face)

    # ------------------------------------------------------------ turning
    def turn_to(self, label):
        """Turn the camera to look straight at the named face, animated."""
        view = next((f[4] for f in _FACES if f[0] == label), None)
        if view is None:
            return
        v = self._viewer
        start = (_wrap(v.rot_x), _wrap(v.rot_y), _wrap(v.rot_z))
        rx, rz = view
        # Z takes the short way round, so Front -> Right is a quarter turn.
        end = (rx, 0.0, start[2] + _wrap(rz - start[2]))

        v.rot_x, v.rot_y, v.rot_z = start
        self._anim_from, self._anim_to, self._anim_t = start, end, 0.0
        self._anim_timer.start(_ANIM_STEP_MS)

    def _anim_step(self):
        self._anim_t = min(self._anim_t + _ANIM_STEP_MS / _ANIM_MS, 1.0)
        t = self._anim_t
        k = t * t * (3.0 - 2.0 * t)          # ease in and out
        a, b = self._anim_from, self._anim_to
        v = self._viewer
        v.rot_x = a[0] + (b[0] - a[0]) * k
        v.rot_y = a[1] + (b[1] - a[1]) * k
        v.rot_z = a[2] + (b[2] - a[2]) * k
        v.update()
        if t >= 1.0:
            self._anim_timer.stop()
