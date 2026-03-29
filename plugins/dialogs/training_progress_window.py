# gui/dialogs/training_progress_window.py
"""
Training Progress Window - Shows real-time PointNet training progress with GPU monitoring.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for projection='3d'
from PyQt5 import QtWidgets, QtCore, QtGui

# Use pynvml for fork-free GPU monitoring (subprocess.run forks, which is unsafe
# when another thread has an active CUDA context — causes segfaults).
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False


class TrainingProgressWindow(QtWidgets.QDialog):
    """
    Window to display training progress with GPU monitoring.

    Features:
    - Epoch progress
    - Training/validation metrics
    - GPU usage monitoring
    - Time tracking
    - Cancel button to stop training
    """

    def __init__(self, parent=None, total_epochs=100):
        super().__init__(parent)

        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = time.time()
        self.training_cancelled = False
        self.training_complete = False

        # History for plotting
        self.epochs_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_miou_history = []
        self.val_miou_history = []
        self.lr_history = []

        # Best values tracking
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.best_train_miou = 0.0
        self.best_val_miou = 0.0

        # Whether to show mIoU instead of accuracy (set on first call with mIoU data)
        self.show_miou = False

        self._setup_ui()
        self._setup_window_flags()
        self._setup_gpu_timer()

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
        self.setWindowTitle("PointNet Training Progress")
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)

        # Main layout (horizontal split)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Left panel - metrics and info
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(15)

        # Right panel - plots
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)

        # Epoch info
        epoch_layout = QtWidgets.QHBoxLayout()
        self.epoch_label = QtWidgets.QLabel("Epoch: 0 / 100")
        epoch_font = QtGui.QFont()
        epoch_font.setPointSize(10)
        self.epoch_label.setFont(epoch_font)
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addStretch()
        left_panel.addLayout(epoch_layout)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        left_panel.addWidget(self.progress_bar)

        left_panel.addSpacing(10)

        # Training Configuration section
        config_group = QtWidgets.QGroupBox()
        config_layout = QtWidgets.QVBoxLayout()

        self.loss_function_label = QtWidgets.QLabel("Loss Function:  -")
        config_layout.addWidget(self.loss_function_label)

        self.lr_scheduler_label = QtWidgets.QLabel("LR Scheduler:   -")
        config_layout.addWidget(self.lr_scheduler_label)

        # Learning rate (moved from Training Metrics)
        self.learning_rate_label = QtWidgets.QLabel("Learning Rate:     -.------")
        config_layout.addWidget(self.learning_rate_label)

        # 3D training progress mini-graph (x=Epoch, y=LR, z=mIoU)
        self._lr_formatter = FuncFormatter(lambda x, _: f"{x:.0e}")
        self.lr_figure = Figure(figsize=(3, 2.0), dpi=100)
        self.lr_canvas = FigureCanvas(self.lr_figure)
        self.lr_canvas.setMinimumHeight(160)
        self.lr_canvas.setMaximumHeight(220)
        self.ax_lr = self.lr_figure.add_subplot(111, projection='3d')
        self.ax_lr.set_xlabel('Epoch', fontsize=6, labelpad=1)
        self.ax_lr.set_ylabel('LR', fontsize=6, labelpad=1)
        self.ax_lr.set_zlabel('mIoU', fontsize=6, labelpad=1)
        self.ax_lr.tick_params(axis='both', labelsize=5, pad=0)
        self.ax_lr.tick_params(axis='z', labelsize=5, pad=0)
        self.ax_lr.yaxis.set_major_formatter(self._lr_formatter)
        self.ax_lr.view_init(elev=25, azim=-60)
        self.lr_figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.lr_canvas.draw()
        config_layout.addWidget(self.lr_canvas)

        config_group.setLayout(config_layout)
        left_panel.addWidget(config_group)

        # Metrics section
        metrics_group = QtWidgets.QGroupBox()
        metrics_layout = QtWidgets.QVBoxLayout()

        # Training loss
        self.train_loss_label = QtWidgets.QLabel("Training Loss:      -.---- (Best: -.----)")
        metrics_layout.addWidget(self.train_loss_label)

        # Validation loss
        self.val_loss_label = QtWidgets.QLabel("Validation Loss:    -.---- (Best: -.----)")
        metrics_layout.addWidget(self.val_loss_label)

        metrics_layout.addSpacing(5)

        # Training accuracy
        self.train_acc_label = QtWidgets.QLabel("Training Accuracy:  --.-%  (Best: --.-%)")
        metrics_layout.addWidget(self.train_acc_label)

        # Validation accuracy
        self.val_acc_label = QtWidgets.QLabel("Validation Accuracy: --.-% (Best: --.-%)")
        metrics_layout.addWidget(self.val_acc_label)

        # Training mIoU (hidden initially, shown for segmentation)
        self.train_miou_label = QtWidgets.QLabel("Training mIoU:      --.-%  (Best: --.-%)")
        self.train_miou_label.setVisible(False)
        metrics_layout.addWidget(self.train_miou_label)

        # Validation mIoU (hidden initially, shown for segmentation)
        self.val_miou_label = QtWidgets.QLabel("Validation mIoU:    --.-%  (Best: --.-%)")
        self.val_miou_label.setVisible(False)
        metrics_layout.addWidget(self.val_miou_label)

        metrics_group.setLayout(metrics_layout)
        left_panel.addWidget(metrics_group)

        # GPU section
        gpu_group = QtWidgets.QGroupBox()
        gpu_layout = QtWidgets.QVBoxLayout()

        self.gpu_usage_label = QtWidgets.QLabel("GPU Usage:       --%")
        gpu_layout.addWidget(self.gpu_usage_label)

        self.gpu_memory_label = QtWidgets.QLabel("GPU Memory:      -.-- / -.-- GB")
        gpu_layout.addWidget(self.gpu_memory_label)

        self.gpu_temp_label = QtWidgets.QLabel("GPU Temperature: --°C")
        gpu_layout.addWidget(self.gpu_temp_label)

        gpu_group.setLayout(gpu_layout)
        left_panel.addWidget(gpu_group)

        left_panel.addStretch()

        # Time info (near bottom)
        time_layout = QtWidgets.QVBoxLayout()
        time_layout.setSpacing(2)

        self.elapsed_time_label = QtWidgets.QLabel("Time Elapsed:    --m --s")
        time_layout.addWidget(self.elapsed_time_label)

        self.est_time_label = QtWidgets.QLabel("Est. Time Left:  --m --s")
        time_layout.addWidget(self.est_time_label)

        left_panel.addLayout(time_layout)

        # Status label at bottom
        self.status_label = QtWidgets.QLabel("Waiting to start...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_font = QtGui.QFont()
        status_font.setItalic(True)
        self.status_label.setFont(status_font)
        left_panel.addWidget(self.status_label)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        # Cancel button (shown during training)
        self.cancel_button = QtWidgets.QPushButton("Cancel Training")
        self.cancel_button.setMinimumWidth(150)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        button_layout.addWidget(self.cancel_button)

        # Close button (hidden initially, shown after training)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setMinimumWidth(150)
        self.close_button.clicked.connect(self.accept)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)

        button_layout.addStretch()
        left_panel.addLayout(button_layout)

        # Right panel - Learning curves
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # Create two subplots - Loss on top, Accuracy on bottom
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_acc = self.figure.add_subplot(212)

        # Configure loss plot (top)
        self.ax_loss.set_title('Training & Validation Loss', fontsize=10, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch', fontsize=9)
        self.ax_loss.set_ylabel('Loss', fontsize=9)
        self.ax_loss.grid(True, alpha=0.3)

        # Configure accuracy plot (bottom)
        self.ax_acc.set_title('Training & Validation Accuracy', fontsize=10, fontweight='bold')
        self.ax_acc.set_xlabel('Epoch', fontsize=9)
        self.ax_acc.set_ylabel('Accuracy', fontsize=9)
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.set_ylim([0, 1])

        self.figure.tight_layout()
        self.canvas.draw()  # Initial draw to show empty plots

        right_panel.addWidget(self.canvas)

        # Add panels to main layout
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_panel, stretch=2)

    def _setup_gpu_timer(self):
        """Set up timer to update GPU stats every 2 seconds."""
        self.gpu_timer = QtCore.QTimer(self)
        self.gpu_timer.timeout.connect(self._update_gpu_stats)
        self.gpu_timer.start(2000)  # Update every 2 seconds

    def _update_gpu_stats(self):
        """Update GPU statistics using pynvml (no fork — safe with concurrent CUDA)."""
        if _nvml_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                mem_used_gb = mem.used / (1024 ** 3)
                mem_total_gb = mem.total / (1024 ** 3)

                self.gpu_usage_label.setText(f"GPU Usage:       {util.gpu}%")
                self.gpu_memory_label.setText(
                    f"GPU Memory:      {mem_used_gb:.2f} / {mem_total_gb:.2f} GB "
                    f"({mem.used / mem.total * 100:.0f}%)")
                self.gpu_temp_label.setText(f"GPU Temperature: {temp}°C")
            except Exception:
                self._set_gpu_unavailable()
        else:
            self._set_gpu_unavailable()

    def _set_gpu_unavailable(self):
        """Set GPU labels to unavailable state."""
        self.gpu_usage_label.setText("GPU Usage:       N/A")
        self.gpu_memory_label.setText("GPU Memory:      N/A")
        self.gpu_temp_label.setText("GPU Temperature: N/A")

    def _update_plots(self):
        """Update the learning curve plots with current history."""
        # Clear previous plots
        self.ax_loss.clear()
        self.ax_acc.clear()

        # Reconfigure loss plot (top)
        self.ax_loss.set_title('Training & Validation Loss', fontsize=10, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch', fontsize=9)
        self.ax_loss.set_ylabel('Loss', fontsize=9)
        self.ax_loss.grid(True, alpha=0.3)

        # Reconfigure second plot (bottom) — mIoU for segmentation, accuracy otherwise
        if self.show_miou:
            self.ax_acc.set_title('Training & Validation mIoU', fontsize=10, fontweight='bold')
            self.ax_acc.set_xlabel('Epoch', fontsize=9)
            self.ax_acc.set_ylabel('mIoU', fontsize=9)
        else:
            self.ax_acc.set_title('Training & Validation Accuracy', fontsize=10, fontweight='bold')
            self.ax_acc.set_xlabel('Epoch', fontsize=9)
            self.ax_acc.set_ylabel('Accuracy', fontsize=9)
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.set_ylim([0, 1])

        # Plot loss curves (top)
        if len(self.epochs_history) > 0:
            self.ax_loss.plot(self.epochs_history, self.train_loss_history, 'b-',
                            label='Training', linewidth=1)
            self.ax_loss.plot(self.epochs_history, self.val_loss_history, 'r-',
                            label='Validation', linewidth=1)
            self.ax_loss.legend(loc='upper right', fontsize=9)

        # Plot second metric curves (bottom)
        if len(self.epochs_history) > 0:
            if self.show_miou:
                self.ax_acc.plot(self.epochs_history, self.train_miou_history, 'b-',
                               label='Training', linewidth=1)
                self.ax_acc.plot(self.epochs_history, self.val_miou_history, 'r-',
                               label='Validation', linewidth=1)
            else:
                self.ax_acc.plot(self.epochs_history, self.train_acc_history, 'b-',
                               label='Training', linewidth=1)
                self.ax_acc.plot(self.epochs_history, self.val_acc_history, 'r-',
                               label='Validation', linewidth=1)
            self.ax_acc.legend(loc='lower right', fontsize=9)

        # Update 3D mini-chart (x=Epoch, y=LR, z=mIoU)
        self.ax_lr.clear()
        self.ax_lr.set_xlabel('Epoch', fontsize=6, labelpad=1)
        self.ax_lr.set_ylabel('LR', fontsize=6, labelpad=1)
        self.ax_lr.set_zlabel('mIoU', fontsize=6, labelpad=1)
        self.ax_lr.tick_params(axis='both', labelsize=5, pad=0)
        self.ax_lr.tick_params(axis='z', labelsize=5, pad=0)
        self.ax_lr.yaxis.set_major_formatter(self._lr_formatter)
        self.ax_lr.view_init(elev=25, azim=-60)
        n = min(len(self.lr_history), len(self.train_miou_history))
        if n > 0:
            epochs = self.epochs_history[:n]
            lrs = self.lr_history[:n]
            mious = self.train_miou_history[:n]
            self.ax_lr.plot(epochs, lrs, mious, color='green', linewidth=1.5)
            # Ribbon: drop lines from each point to the floor (z=0)
            for i in range(n):
                self.ax_lr.plot([epochs[i], epochs[i]], [lrs[i], lrs[i]],
                               [0, mious[i]], color='green', alpha=0.2, linewidth=0.8)
        self.lr_canvas.draw()

        # Adjust layout and refresh canvas
        try:
            self.figure.tight_layout()
        except (ValueError, RuntimeError):
            pass  # tight_layout can fail if axis bounding boxes aren't ready
        self.canvas.draw()

    def update_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                     learning_rate=None, train_miou=None, val_miou=None):
        """
        Update the window with new epoch information.

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Training loss value
            train_acc: Training accuracy (0-1)
            val_loss: Validation loss value
            val_acc: Validation accuracy (0-1)
            learning_rate: Current learning rate (optional)
            train_miou: Training mIoU (optional, enables mIoU mode for segmentation)
            val_miou: Validation mIoU (optional, enables mIoU mode for segmentation)
        """
        self.current_epoch = epoch

        # Switch to mIoU mode on first call with mIoU data
        if train_miou is not None and val_miou is not None and not self.show_miou:
            self.show_miou = True
            self.train_acc_label.setVisible(False)
            self.val_acc_label.setVisible(False)
            self.train_miou_label.setVisible(True)
            self.val_miou_label.setVisible(True)

        # Update epoch label and progress bar
        self.epoch_label.setText(f"Epoch: {epoch} / {self.total_epochs}")
        progress_pct = (epoch / self.total_epochs) * 100
        self.progress_bar.setValue(int(progress_pct * 10))
        self.progress_bar.setFormat(f"{progress_pct:.1f}%")

        # Update best values
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # Update loss labels
        self.train_loss_label.setText(f"Training Loss:      {train_loss:.4f} (Best: {self.best_train_loss:.4f})")
        self.val_loss_label.setText(f"Validation Loss:    {val_loss:.4f} (Best: {self.best_val_loss:.4f})")

        # Update mIoU or accuracy labels
        if self.show_miou and train_miou is not None and val_miou is not None:
            if train_miou > self.best_train_miou:
                self.best_train_miou = train_miou
            if val_miou > self.best_val_miou:
                self.best_val_miou = val_miou
            self.train_miou_label.setText(f"Training mIoU:      {train_miou*100:.1f}%  (Best: {self.best_train_miou*100:.1f}%)")
            self.val_miou_label.setText(f"Validation mIoU:    {val_miou*100:.1f}%  (Best: {self.best_val_miou*100:.1f}%)")
            self.train_miou_history.append(train_miou)
            self.val_miou_history.append(val_miou)
        else:
            self.train_acc_label.setText(f"Training Accuracy:  {train_acc*100:.1f}% (Best: {self.best_train_acc*100:.1f}%)")
            self.val_acc_label.setText(f"Validation Accuracy: {val_acc*100:.1f}% (Best: {self.best_val_acc*100:.1f}%)")

        # Update learning rate if provided
        if learning_rate is not None:
            self.learning_rate_label.setText(f"Learning Rate:     {learning_rate:.6f}")
            self.lr_history.append(learning_rate)

        # Update time estimates
        elapsed = time.time() - self.start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        self.elapsed_time_label.setText(f"Time Elapsed:    {elapsed_min}m {elapsed_sec}s")

        # Estimate remaining time
        if epoch > 0:
            avg_time_per_epoch = elapsed / epoch
            remaining_epochs = self.total_epochs - epoch
            est_remaining = avg_time_per_epoch * remaining_epochs
            est_min = int(est_remaining // 60)
            est_sec = int(est_remaining % 60)
            self.est_time_label.setText(f"Est. Time Left:  {est_min}m {est_sec}s")

        # Update status
        self.status_label.setText(f"Training in progress...")

        # Append to history for plotting
        self.epochs_history.append(epoch)
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        # Update learning curve plots
        self._update_plots()

    def training_started(self):
        """Mark training as started."""
        self.start_time = time.time()
        self.status_label.setText("Training started...")

    def set_training_config(self, loss_function=None, lr_scheduler=None):
        """Set training configuration labels shown in the UI.

        Args:
            loss_function: Name of the loss function (e.g. "Focal + Class Weights")
            lr_scheduler: Name of the LR scheduler (e.g. "OneCycleLR")
        """
        if loss_function:
            self.loss_function_label.setText(f"Loss Function:  {loss_function}")
        if lr_scheduler:
            self.lr_scheduler_label.setText(f"LR Scheduler:   {lr_scheduler}")

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        # Show confirmation dialog
        reply = QtWidgets.QMessageBox.question(
            self,
            "Cancel Training",
            "Are you sure you want to cancel training?\n\n"
            "Progress will be lost and the best model so far will be saved.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.training_cancelled = True
            self.status_label.setText("Cancelling training... Please wait.")
            self.cancel_button.setEnabled(False)
            QtWidgets.QApplication.processEvents()

    def training_completed(self, best_val_metric, cancelled=False):
        """
        Mark training as completed.

        Args:
            best_val_metric: Best validation metric achieved (accuracy or mIoU)
            cancelled: Whether training was cancelled by user
        """
        self.training_complete = True
        self.gpu_timer.stop()

        metric_name = "mIoU" if self.show_miou else "accuracy"
        if cancelled:
            self.status_label.setText(f"Training cancelled. Best {metric_name}: {best_val_metric*100:.1f}%")
        else:
            self.status_label.setText(f"Training completed! Best {metric_name}: {best_val_metric*100:.1f}%")

        # Hide cancel button, show close button
        self.cancel_button.setVisible(False)
        self.close_button.setVisible(True)

        # Restore close button on title bar so user can close via X or Close button
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.WindowMinimizeButtonHint |
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowTitleHint
        )
        self.show()

        QtWidgets.QApplication.processEvents()

    def save_snapshot(self, file_path):
        """Save a screenshot of the entire dialog to an image file.

        Args:
            file_path: Output file path (e.g. '/path/to/training_progress.png')
        """
        pixmap = self.grab()
        pixmap.save(file_path, 'PNG')
        print(f"Training progress snapshot saved to: {file_path}")

    def closeEvent(self, event):
        """Override close event to prevent closing during training."""
        if self.training_complete or self.training_cancelled:
            # Allow closing if training is complete or cancelled
            self.gpu_timer.stop()
            event.accept()
        else:
            # Only minimize during training
            event.ignore()
            self.showMinimized()
