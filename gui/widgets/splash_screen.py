# gui/widgets/splash_screen.py
"""
Professional splash screen for SPCToolkit.

Displays app branding, hardware detection status, and loading progress.
Auto-closes when the main application is ready.
"""

from PyQt5.QtWidgets import (
    QSplashScreen, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QApplication, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QLinearGradient, QPen


class SplashScreen(QWidget):
    """
    Professional splash screen with hardware info and loading progress.
    """

    # Minimum time to display splash screen (milliseconds)
    MIN_DISPLAY_TIME_MS = 20000

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Window size
        self.setFixedSize(500, 380)

        # Center on screen
        self._center_on_screen()

        # Status tracking
        self._status_text = "Initializing..."
        self._progress = 0

        # Track when splash was shown
        self._show_time = None

        # Setup UI
        self._setup_ui()

    def show(self):
        """Override show to track display time."""
        import time
        self._show_time = time.time()
        super().show()

    def _center_on_screen(self):
        """Center the splash screen on the primary display."""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """Setup the splash screen UI."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Container frame with rounded corners and background
        self.container = QFrame()
        self.container.setObjectName("splashContainer")
        self.container.setStyleSheet("""
            #splashContainer {
                background-color: #1a1a2e;
                border-radius: 15px;
                border: 2px solid #16213e;
            }
        """)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(30, 25, 30, 25)
        container_layout.setSpacing(15)

        # === Header Section ===
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)

        # Logo placeholder (colored box with icon-like design)
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(70, 70)
        self.logo_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 12px;
            font-size: 28px;
            color: white;
        """)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setText("◈")  # Unicode point cloud-like symbol

        # Title section
        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)

        self.title_label = QLabel("SPCToolkit")
        self.title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 26px;
            font-weight: bold;
            letter-spacing: 1px;
        """)

        self.subtitle_label = QLabel("Smart Point Cloud Toolkit")
        self.subtitle_label.setStyleSheet("""
            color: #8892b0;
            font-size: 13px;
            letter-spacing: 0.5px;
        """)

        self.version_label = QLabel("Version 1.0.0")
        self.version_label.setStyleSheet("""
            color: #64ffda;
            font-size: 11px;
        """)

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        title_layout.addWidget(self.version_label)

        header_layout.addWidget(self.logo_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        container_layout.addLayout(header_layout)

        # === Separator ===
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #16213e; max-height: 1px;")
        container_layout.addWidget(separator)

        # === Hardware Info Section ===
        hw_section = QVBoxLayout()
        hw_section.setSpacing(8)

        hw_title = QLabel("System Configuration")
        hw_title.setStyleSheet("""
            color: #ccd6f6;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 0.5px;
        """)
        hw_section.addWidget(hw_title)

        # Hardware info labels (will be updated dynamically)
        self.os_label = self._create_info_label("Operating System", "Detecting...")
        self.gpu_label = self._create_info_label("GPU", "Detecting...")
        self.mode_label = self._create_info_label("Compute Mode", "Detecting...")

        hw_section.addWidget(self.os_label)
        hw_section.addWidget(self.gpu_label)
        hw_section.addWidget(self.mode_label)

        container_layout.addLayout(hw_section)

        # === Separator ===
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet("background-color: #16213e; max-height: 1px;")
        container_layout.addWidget(separator2)

        # === Progress Section ===
        progress_section = QVBoxLayout()
        progress_section.setSpacing(8)

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            color: #8892b0;
            font-size: 11px;
        """)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #16213e;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #64ffda);
                border-radius: 3px;
            }
        """)
        self.progress_bar.setValue(0)

        progress_section.addWidget(self.status_label)
        progress_section.addWidget(self.progress_bar)

        container_layout.addLayout(progress_section)

        # === Continue Button ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.continue_button = QPushButton("Continue")
        self.continue_button.setFixedSize(120, 36)
        self.continue_button.setEnabled(False)  # Disabled until loading complete
        self.continue_button.setCursor(Qt.PointingHandCursor)
        self.continue_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
            QPushButton:pressed {
                background: #5a67d8;
            }
            QPushButton:disabled {
                background: #2d3748;
                color: #718096;
            }
        """)

        button_layout.addWidget(self.continue_button)
        button_layout.addStretch()

        container_layout.addLayout(button_layout)

        # === Footer ===
        container_layout.addStretch()

        footer_layout = QHBoxLayout()

        self.author_label = QLabel("Developed by Sepehr")
        self.author_label.setStyleSheet("""
            color: #4a5568;
            font-size: 10px;
        """)

        self.year_label = QLabel("© 2025")
        self.year_label.setStyleSheet("""
            color: #4a5568;
            font-size: 10px;
        """)

        footer_layout.addWidget(self.author_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.year_label)

        container_layout.addLayout(footer_layout)

        main_layout.addWidget(self.container)

    def _create_info_label(self, title: str, value: str) -> QLabel:
        """Create a styled info label with title and value."""
        label = QLabel(f"<span style='color: #8892b0;'>{title}:</span> "
                      f"<span style='color: #ccd6f6;'>{value}</span>")
        label.setStyleSheet("font-size: 12px;")
        return label

    def set_hardware_info(self, os_name: str, gpu_name: str, mode: str):
        """Update the hardware information display."""
        self.os_label.setText(
            f"<span style='color: #8892b0;'>Operating System:</span> "
            f"<span style='color: #ccd6f6;'>{os_name}</span>"
        )

        # GPU with icon
        if gpu_name and gpu_name != "None":
            gpu_display = f"<span style='color: #64ffda;'>✓</span> {gpu_name}"
        else:
            gpu_display = "<span style='color: #f56565;'>✗</span> No GPU detected"

        self.gpu_label.setText(
            f"<span style='color: #8892b0;'>GPU:</span> "
            f"<span style='color: #ccd6f6;'>{gpu_display}</span>"
        )

        # Mode with color coding
        mode_colors = {
            "FULL_GPU": "#64ffda",      # Teal/green
            "PARTIAL_GPU": "#ffd93d",   # Yellow
            "CPU_ONLY": "#f56565"       # Red
        }
        mode_color = mode_colors.get(mode, "#ccd6f6")

        self.mode_label.setText(
            f"<span style='color: #8892b0;'>Compute Mode:</span> "
            f"<span style='color: {mode_color}; font-weight: bold;'>{mode}</span>"
        )

    def set_status(self, status: str):
        """Update the status message."""
        self._status_text = status
        self.status_label.setText(status)

    def set_progress(self, value: int):
        """Update the progress bar (0-100)."""
        self._progress = min(100, max(0, value))
        self.progress_bar.setValue(self._progress)

    def finish(self, main_window):
        """
        Finish the splash screen and show the main window.
        Enables the Continue button for manual close, and auto-closes after MIN_DISPLAY_TIME_MS.

        Args:
            main_window: The main application window to show
        """
        import time

        self._main_window = main_window
        self.set_status("Ready! Click Continue to start.")
        self.set_progress(100)

        # Enable the continue button and connect it
        self.continue_button.setEnabled(True)
        self.continue_button.clicked.connect(self._on_continue_clicked)

        # Calculate remaining time to meet minimum display time
        if self._show_time is not None:
            elapsed_ms = (time.time() - self._show_time) * 1000
            remaining_ms = max(0, self.MIN_DISPLAY_TIME_MS - elapsed_ms)
        else:
            remaining_ms = self.MIN_DISPLAY_TIME_MS

        # Auto-close after remaining time (if user hasn't clicked Continue)
        self._auto_close_timer = QTimer.singleShot(int(remaining_ms), self._auto_close)

    def _on_continue_clicked(self):
        """Handle Continue button click."""
        self._close_and_show(self._main_window)

    def _auto_close(self):
        """Auto-close after minimum display time."""
        if self.isVisible():
            self._close_and_show(self._main_window)

    def _close_and_show(self, main_window):
        """Close splash and show main window."""
        self.close()
        main_window.show()
