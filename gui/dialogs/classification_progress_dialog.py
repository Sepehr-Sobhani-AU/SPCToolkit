# gui/dialogs/classification_progress_dialog.py
"""
Progress dialog for PointNet cluster classification.

Shows real-time progress as clusters are being classified.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QTextEdit, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ClassificationProgressDialog(QDialog):
    """
    Progress dialog for cluster classification.

    Shows:
    - Overall progress (X/Y clusters)
    - Progress bar
    - Current cluster's predicted class and confidence
    - Log of recent classifications
    - Cancel button
    """

    def __init__(self, parent, total_clusters: int):
        """
        Initialize the classification progress dialog.

        Args:
            parent: Parent widget
            total_clusters: Total number of clusters to classify
        """
        super().__init__(parent)
        self.setWindowTitle("PointNet Cluster Classification")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.total_clusters = total_clusters
        self.cancelled = False
        self.classification_complete = False

        # Track statistics in real-time
        self.stats = {
            'classified': 0,
            'skipped_small': 0,
            'skipped_low_confidence': 0
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()

        # Title label
        title_label = QLabel("Classifying Clusters with PointNet")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.total_clusters)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Current cluster details
        self.details_label = QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setStyleSheet("QLabel { color: #0066cc; font-weight: bold; }")
        layout.addWidget(self.details_label)

        # Log area
        log_label = QLabel("Classification Log:")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # Statistics label
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_progress(
        self,
        current: int,
        cluster_id: int,
        predicted_class: str,
        confidence: float
    ):
        """
        Update progress with current cluster classification.

        Args:
            current: Current cluster index (1-indexed)
            cluster_id: Cluster ID being classified
            predicted_class: Predicted class name
            confidence: Prediction confidence (0-1)
        """
        # Update progress bar
        self.progress_bar.setValue(current)

        # Update status
        percent = (current / self.total_clusters) * 100
        self.status_label.setText(
            f"Processing cluster {current}/{self.total_clusters} ({percent:.1f}%)"
        )

        # Update statistics based on classification result
        if predicted_class == "Skipped (too small)":
            self.stats['skipped_small'] += 1
        elif "Unclassified" in predicted_class or "low confidence" in predicted_class:
            self.stats['skipped_low_confidence'] += 1
        else:
            self.stats['classified'] += 1

        # Update statistics display in real-time
        self.update_statistics(self.stats)

        # Update details
        if predicted_class == "Skipped (too small)":
            self.details_label.setText(
                f"Cluster #{cluster_id}: {predicted_class}"
            )
            self.details_label.setStyleSheet("QLabel { color: #999999; font-weight: bold; }")
            # Don't add to log - just update the count at the bottom
        else:
            confidence_percent = confidence * 100
            self.details_label.setText(
                f"Cluster #{cluster_id}: {predicted_class} (confidence: {confidence_percent:.1f}%)"
            )

            # Color based on confidence
            if confidence >= 0.8:
                color = "#00aa00"  # Green - high confidence
            elif confidence >= 0.5:
                color = "#0066cc"  # Blue - medium confidence
            else:
                color = "#ff6600"  # Orange - low confidence (unclassified)

            self.details_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")

            # Add to log (all non-skipped classifications)
            log_entry = f"Cluster {cluster_id}: {predicted_class} ({confidence:.2f})"
            self.log_text.append(log_entry)

        # Keep UI responsive
        QApplication.processEvents()

    def update_statistics(self, stats: dict):
        """
        Update statistics display.

        Args:
            stats: Dictionary with classification statistics
        """
        stats_text = (
            f"Classified: {stats.get('classified', 0)} | "
            f"Skipped (small): {stats.get('skipped_small', 0)} | "
            f"Unclassified (low confidence): {stats.get('skipped_low_confidence', 0)}"
        )
        self.stats_label.setText(stats_text)

    def classification_completed(self, stats: dict):
        """
        Mark classification as complete.

        Args:
            stats: Final classification statistics
        """
        self.classification_complete = True

        if self.cancelled:
            self.status_label.setText("Classification Cancelled")
        else:
            self.status_label.setText("Classification Complete!")

        self.update_statistics(stats)

        # Show close button, hide cancel button
        self.cancel_button.setVisible(False)
        self.close_button.setVisible(True)

    def on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling classification...")
        QApplication.processEvents()
