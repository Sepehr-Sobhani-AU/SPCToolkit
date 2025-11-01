"""
ProcessOverlayWidget - A semi-transparent overlay that blocks user interaction during processing.

This widget is designed to be positioned over other widgets to prevent user interaction
while long-running operations are in progress. It displays a message to inform the user
what operation is currently running.
"""

from PyQt5 import QtWidgets, QtCore, QtGui


class ProcessOverlayWidget(QtWidgets.QWidget):
    """
    A semi-transparent overlay widget that blocks interactions during processing.

    Features:
    - Semi-transparent background
    - Centered message label
    - Blocks all mouse/keyboard events to underlying widget
    - Can be positioned over any widget or manually positioned
    """

    def __init__(self, parent=None):
        """
        Initialize the ProcessOverlayWidget.

        Args:
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)

        # Set up the widget properties
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)  # Block mouse events

        # Initially hidden
        self.hide()

        # Set up the UI
        self._setup_ui()

    def _setup_ui(self):
        """Set up the overlay UI components."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container widget for the message (to allow proper background styling)
        self.container = QtWidgets.QWidget(self)
        self.container.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 180);
                border-radius: 10px;
            }
        """)

        # Container layout
        container_layout = QtWidgets.QVBoxLayout(self.container)
        container_layout.setContentsMargins(20, 20, 20, 20)

        # Message label
        self.message_label = QtWidgets.QLabel("Processing...")
        self.message_label.setAlignment(QtCore.Qt.AlignCenter)
        self.message_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14pt;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        self.message_label.setWordWrap(True)

        container_layout.addWidget(self.message_label)

        # Add container to main layout with centering
        layout.addStretch()
        layout.addWidget(self.container, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()

    def position_over(self, target_widget):
        """
        Position this overlay to cover the target widget.

        Args:
            target_widget: The widget to position over
        """
        if target_widget is None:
            return

        # Get the target widget's geometry in global coordinates
        target_rect = target_widget.geometry()

        # If target has a parent, we need to map to our parent's coordinate system
        if target_widget.parent() and self.parent():
            # Map target's position to global coordinates
            global_pos = target_widget.mapToGlobal(QtCore.QPoint(0, 0))
            # Map back to our parent's coordinates
            local_pos = self.parent().mapFromGlobal(global_pos)

            self.setGeometry(
                local_pos.x(),
                local_pos.y(),
                target_widget.width(),
                target_widget.height()
            )
        else:
            # Fallback to direct geometry
            self.setGeometry(target_rect)

        # Raise to top of stacking order
        self.raise_()

    def set_geometry(self, x, y, width, height):
        """
        Manually set the overlay geometry.

        Args:
            x: X position
            y: Y position
            width: Width
            height: Height
        """
        self.setGeometry(x, y, width, height)
        self.raise_()

    def show_processing(self, message="Processing..."):
        """
        Show the overlay with a processing message.

        Args:
            message: The message to display
        """
        self.message_label.setText(message)
        self.show()
        self.raise_()

        # Force immediate UI update
        QtWidgets.QApplication.processEvents()

    def hide_processing(self):
        """Hide the overlay."""
        self.hide()

        # Force immediate UI update
        QtWidgets.QApplication.processEvents()

    def paintEvent(self, event):
        """
        Custom paint event to draw the semi-transparent background.

        Args:
            event: The paint event
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw semi-transparent background over entire widget
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 100))

        super().paintEvent(event)
