# gui/dialogs/data_generation_progress_dialog.py
"""
Data Generation Progress Dialog - Shows real-time progress for training data generation.
"""

import time
from PyQt5 import QtWidgets, QtCore, QtGui


class DataGenerationProgressDialog(QtWidgets.QDialog):
    """
    Dialog to display data generation progress.

    Features:
    - Overall progress bar
    - Class-by-class progress
    - Sample counts
    - Time tracking
    - Status messages
    """

    # Signal emitted when user clicks cancel
    cancel_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, total_steps=100):
        super().__init__(parent)

        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.cancelled = False
        self.complete = False

        self._setup_ui()
        self._setup_window_flags()

    def _setup_window_flags(self):
        """Set window flags to prevent closing but allow minimizing."""
        # Remove close button, keep minimize and window stays on top
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.WindowMinimizeButtonHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.CustomizeWindowHint
        )

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Training Data Generation Progress")
        self.setMinimumWidth(600)
        self.setMinimumHeight(350)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title_label = QtWidgets.QLabel("Training Data Generation")
        title_font = QtGui.QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Separator
        line1 = QtWidgets.QFrame()
        line1.setFrameShape(QtWidgets.QFrame.HLine)
        line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(line1)

        # Current operation label
        self.operation_label = QtWidgets.QLabel("Initializing...")
        operation_font = QtGui.QFont()
        operation_font.setPointSize(10)
        self.operation_label.setFont(operation_font)
        main_layout.addWidget(self.operation_label)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(25)
        main_layout.addWidget(self.progress_bar)

        main_layout.addSpacing(10)

        # Statistics group
        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QVBoxLayout()

        # Current class
        self.current_class_label = QtWidgets.QLabel("Current Class: --")
        stats_layout.addWidget(self.current_class_label)

        # Processed samples
        self.processed_label = QtWidgets.QLabel("Processed Samples: 0")
        stats_layout.addWidget(self.processed_label)

        # Skipped samples
        self.skipped_label = QtWidgets.QLabel("Skipped Samples: 0")
        stats_layout.addWidget(self.skipped_label)

        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)

        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(line2)

        # Time info
        time_layout = QtWidgets.QVBoxLayout()

        self.elapsed_time_label = QtWidgets.QLabel("Time Elapsed: 0m 0s")
        time_layout.addWidget(self.elapsed_time_label)

        self.est_time_label = QtWidgets.QLabel("Est. Time Left: --")
        time_layout.addWidget(self.est_time_label)

        main_layout.addLayout(time_layout)

        main_layout.addStretch()

        # Status label
        self.status_label = QtWidgets.QLabel("Waiting to start...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_font = QtGui.QFont()
        status_font.setItalic(True)
        self.status_label.setFont(status_font)
        main_layout.addWidget(self.status_label)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        # Cancel button (shown during processing)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setMinimumWidth(120)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        button_layout.addWidget(self.cancel_button)

        # Close button (hidden initially, shown after completion)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setMinimumWidth(120)
        self.close_button.clicked.connect(self.accept)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

    def set_operation(self, message: str):
        """
        Update the current operation message.

        Args:
            message: Operation description (e.g., "Scanning input directory...")
        """
        self.operation_label.setText(message)
        QtWidgets.QApplication.processEvents()

    def update_progress(self, current: int, total: int, current_class: str = None,
                       processed_count: int = 0, skipped_count: int = 0):
        """
        Update progress information.

        Args:
            current: Current step number
            total: Total number of steps
            current_class: Name of class being processed (optional)
            processed_count: Number of samples processed
            skipped_count: Number of samples skipped
        """
        self.current_step = current
        self.total_steps = total

        # Update progress bar
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{current} / {total} ({progress}%)")
        else:
            self.progress_bar.setValue(0)

        # Update current class
        if current_class:
            self.current_class_label.setText(f"Current Class: {current_class}")

        # Update counts
        self.processed_label.setText(f"Processed Samples: {processed_count}")
        self.skipped_label.setText(f"Skipped Samples: {skipped_count}")

        # Update time estimates
        elapsed = time.time() - self.start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        self.elapsed_time_label.setText(f"Time Elapsed: {elapsed_min}m {elapsed_sec}s")

        # Estimate remaining time
        if current > 0:
            avg_time_per_step = elapsed / current
            remaining_steps = total - current
            est_remaining = avg_time_per_step * remaining_steps
            est_min = int(est_remaining // 60)
            est_sec = int(est_remaining % 60)
            self.est_time_label.setText(f"Est. Time Left: {est_min}m {est_sec}s")

        # Force update
        QtWidgets.QApplication.processEvents()

    def set_status(self, message: str):
        """
        Update status message.

        Args:
            message: Status message
        """
        self.status_label.setText(message)
        QtWidgets.QApplication.processEvents()

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Cancel Generation",
            "Are you sure you want to cancel data generation?\n\n"
            "Progress will be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.cancelled = True
            self.status_label.setText("Cancelling... Please wait.")
            self.cancel_button.setEnabled(False)
            self.cancel_requested.emit()
            QtWidgets.QApplication.processEvents()

    def mark_complete(self, success: bool = True, message: str = None):
        """
        Mark generation as complete.

        Args:
            success: Whether generation completed successfully
            message: Optional completion message
        """
        self.complete = True

        if message:
            self.status_label.setText(message)
        elif success:
            self.status_label.setText("Generation completed successfully!")
        else:
            self.status_label.setText("Generation cancelled.")

        # Set progress to 100%
        self.progress_bar.setValue(100)

        # Hide cancel button, show close button
        self.cancel_button.setVisible(False)
        self.close_button.setVisible(True)

        QtWidgets.QApplication.processEvents()

    def closeEvent(self, event):
        """Override close event to prevent closing during generation."""
        if self.complete or self.cancelled:
            event.accept()
        else:
            # Only minimize during generation
            event.ignore()
            self.showMinimized()