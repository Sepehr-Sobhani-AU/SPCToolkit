"""
Train PointNet Model Plugin

Trains a PointNet model for cluster classification using training data from a specified directory.
The directory should contain subfolders where each subfolder name is a class label and contains .npy files.
"""

import os
import json
import csv
import numpy as np
from typing import Dict, Any
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

from PyQt5 import QtWidgets

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.pointnet_model import PointNetClassifier
from gui.dialogs.training_progress_window import TrainingProgressWindow


class TrainPointNetPlugin(ActionPlugin):
    """
    Action plugin for training PointNet model for cluster classification.
    """

    # Class variables to store last used parameters
    last_params = {
        "training_data_dir": "training_data",
        "output_dir": "models",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "val_split": 0.2,
        "use_tnet": True,
        "early_stopping_patience": 20,
        "repetitions": 1
    }

    def get_name(self) -> str:
        return "train_pointnet_model"

    def _get_next_run_number(self, output_dir: str) -> int:
        """
        Get the next run number by scanning existing folders.

        Args:
            output_dir: Base output directory

        Returns:
            Next sequential run number
        """
        if not os.path.exists(output_dir):
            return 1

        # Find all folders matching run_XXX pattern
        run_numbers = []
        for item in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, item)) and item.startswith('run_'):
                try:
                    # Extract number from run_001, run_002, etc.
                    parts = item.split('_')
                    if len(parts) >= 2:
                        run_num = int(parts[1])
                        run_numbers.append(run_num)
                except (ValueError, IndexError):
                    continue

        return max(run_numbers) + 1 if run_numbers else 1

    def _write_to_tracking_csv(self, output_dir: str, training_data: Dict[str, Any]):
        """
        Write training results to central tracking CSV file.

        Args:
            output_dir: Base output directory
            training_data: Dictionary containing all training information
        """
        csv_path = os.path.join(output_dir, 'training_history.csv')

        # Define CSV columns
        fieldnames = [
            'folder_name', 'timestamp', 'run_number', 'epochs', 'batch_size',
            'learning_rate', 'val_split', 'use_tnet', 'early_stopping_patience',
            'repetitions', 'random_seed', 'best_val_acc', 'final_val_acc',
            'epochs_completed', 'training_samples', 'validation_samples',
            'num_classes', 'was_cancelled'
        ]

        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(csv_path)

        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if new file
            if not file_exists:
                writer.writeheader()

            # Write data row
            writer.writerow(training_data)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "training_data_dir": {
                "type": "directory",
                "default": self.last_params["training_data_dir"],
                "label": "Training Data Directory",
                "description": "Directory containing class subfolders with .npy files"
            },
            "output_dir": {
                "type": "directory",
                "default": self.last_params["output_dir"],
                "label": "Output Directory",
                "description": "Directory to save trained model"
            },
            "epochs": {
                "type": "int",
                "default": self.last_params["epochs"],
                "min": 1,
                "max": 1000,
                "label": "Epochs",
                "description": "Number of training epochs"
            },
            "batch_size": {
                "type": "int",
                "default": self.last_params["batch_size"],
                "min": 1,
                "max": 128,
                "label": "Batch Size",
                "description": "Training batch size"
            },
            "learning_rate": {
                "type": "float",
                "default": self.last_params["learning_rate"],
                "min": 0.0000001,
                "max": 0.1,
                "decimals": 7,
                "label": "Learning Rate",
                "description": "Initial learning rate"
            },
            "val_split": {
                "type": "float",
                "default": self.last_params["val_split"],
                "min": 0.1,
                "max": 0.5,
                "label": "Validation Split",
                "description": "Fraction of data to use for validation"
            },
            "use_tnet": {
                "type": "bool",
                "default": self.last_params["use_tnet"],
                "label": "Use T-Net",
                "description": "Use spatial and feature transformation networks"
            },
            "early_stopping_patience": {
                "type": "int",
                "default": self.last_params["early_stopping_patience"],
                "min": 1,
                "max": 100,
                "label": "Early Stopping Patience",
                "description": "Number of epochs with no improvement before stopping training"
            },
            "repetitions": {
                "type": "int",
                "default": self.last_params["repetitions"],
                "min": 1,
                "max": 100,
                "label": "Training Repetitions",
                "description": "Number of times to train with different random initializations (to find best result)"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the training process.

        Args:
            main_window: The main application window
            params: Parameters from the dialog
        """
        # Get parameters
        data_dir = params['training_data_dir'].strip()
        output_dir = params['output_dir'].strip()
        epochs = int(params['epochs'])
        batch_size = int(params['batch_size'])
        learning_rate = float(params['learning_rate'])
        val_split = float(params['val_split'])
        use_tnet = params['use_tnet']
        early_stopping_patience = int(params['early_stopping_patience'])
        repetitions = int(params['repetitions'])

        # Store parameters for next time (session persistence)
        TrainPointNetPlugin.last_params = {
            "training_data_dir": data_dir,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_split": val_split,
            "use_tnet": use_tnet,
            "early_stopping_patience": early_stopping_patience,
            "repetitions": repetitions
        }

        # Validate directories
        if not os.path.exists(data_dir):
            QMessageBox.critical(
                main_window,
                "Invalid Directory",
                f"Training data directory does not exist:\n{data_dir}"
            )
            return

        # Track results from all repetitions
        all_run_results = []
        best_overall_accuracy = 0.0
        best_overall_run = None

        # Get starting run number for this session
        base_run_number = self._get_next_run_number(output_dir)

        print(f"\n{'='*80}")
        print(f"Starting {repetitions} training run(s) with different random initializations")
        print(f"Starting from run #{base_run_number}")
        print(f"{'='*80}")

        # Loop through repetitions
        for rep_index in range(repetitions):
            run_number = base_run_number + rep_index

            print(f"\n{'='*80}")
            print(f"TRAINING RUN #{run_number} ({rep_index + 1}/{repetitions})")
            print(f"{'='*80}")

            # Set random seed using current timestamp for full randomness
            import time
            random_seed = int(time.time() * 1000) % (2**32)  # Use milliseconds, keep within 32-bit range
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
            print(f"Random seed: {random_seed}")

            # Small delay to ensure different seeds for consecutive runs
            time.sleep(0.01)

            # Create unique output directory with simple naming
            # Format: run_XXX_YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"run_{run_number:03d}_{timestamp}"

            # Combine base output dir with unique folder
            unique_output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(unique_output_dir, exist_ok=True)

            print(f"Model will be saved to: {unique_output_dir}")

            try:
                # Disable UI during training
                main_window.disable_menus()
                main_window.disable_tree()

                # Show progress overlay
                main_window.tree_overlay.show_processing("Loading training data...")

                # Load training data
                data, labels, class_mapping, metadata = self.load_training_data(data_dir)

                num_samples, num_points, num_features = data.shape
                num_classes = len(class_mapping)

                # Show data info
                class_counts = {class_mapping[i]: np.sum(labels == i) for i in range(num_classes)}
                info_msg = f"Loaded {num_samples} samples from {num_classes} classes:\n"
                for class_name, count in class_counts.items():
                    info_msg += f"  {class_name}: {count} samples\n"
                info_msg += f"\nShape: ({num_points} points, {num_features} features)"

                print("\n" + "="*80)
                print("PointNet Training Started")
                print("="*80)
                print(info_msg)

                # Split train/validation
                main_window.tree_overlay.show_processing(f"Splitting data ({val_split*100:.0f}% validation)...")

                X_train, X_val, y_train, y_val = train_test_split(
                    data, labels,
                    test_size=val_split,
                    random_state=42,
                    stratify=labels
                )

                print(f"\nTraining samples: {len(X_train)}")
                print(f"Validation samples: {len(X_val)}")

                # Compute class weights
                class_weights_array = compute_class_weight(
                    'balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )
                class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

                print("\nClass weights:")
                for class_id, weight in class_weights.items():
                    print(f"  {class_mapping[class_id]}: {weight:.3f}")

                # Create model
                main_window.tree_overlay.show_processing("Creating PointNet model...")

                print(f"\nModel configuration:")
                print(f"  Input: ({num_points}, {num_features})")
                print(f"  Classes: {num_classes}")
                print(f"  T-Net: {use_tnet}")

                classifier = PointNetClassifier(
                    num_points=num_points,
                    num_features=num_features,
                    num_classes=num_classes,
                    use_tnet=use_tnet
                )
                classifier.class_mapping = class_mapping
                classifier.compile_model(learning_rate=learning_rate)

                # Setup callbacks
                model_checkpoint = keras.callbacks.ModelCheckpoint(
                    os.path.join(unique_output_dir, 'pointnet_best.keras'),
                    monitor='val_sparse_categorical_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )

                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_sparse_categorical_accuracy',
                    patience=early_stopping_patience,
                    mode='max',
                    verbose=1,
                    restore_best_weights=True
                )

                reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-8,
                    verbose=1
                )

                # Create training progress window
                progress_window = TrainingProgressWindow(parent=main_window, total_epochs=epochs)

                # Update window title to show run number if multiple runs
                if repetitions > 1:
                    progress_window.setWindowTitle(f"PointNet Training Progress - Run #{run_number} ({rep_index + 1}/{repetitions})")

                progress_window.show()
                progress_window.training_started()

                # Center the window on screen
                screen_geometry = QtWidgets.QApplication.desktop().screenGeometry()
                x = (screen_geometry.width() - progress_window.width()) // 2
                y = (screen_geometry.height() - progress_window.height()) // 2
                progress_window.move(x, y)

                # Custom callback to update progress window and handle cancellation
                class ProgressWindowCallback(keras.callbacks.Callback):
                    def __init__(self, progress_window):
                        super().__init__()
                        self.progress_window = progress_window

                    def on_epoch_end(self, epoch, logs=None):
                        train_loss = logs.get('loss', 0)
                        train_acc = logs.get('sparse_categorical_accuracy', 0)
                        val_loss = logs.get('val_loss', 0)
                        val_acc = logs.get('val_sparse_categorical_accuracy', 0)

                        # Get current learning rate
                        from tensorflow.keras import backend as K
                        current_lr = float(K.get_value(self.model.optimizer.learning_rate))

                        self.progress_window.update_epoch(
                            epoch + 1,  # 1-indexed
                            train_loss,
                            train_acc,
                            val_loss,
                            val_acc,
                            learning_rate=current_lr
                        )

                        # Check if user requested cancellation
                        if self.progress_window.training_cancelled:
                            print("\nUser requested training cancellation. Stopping...")
                            self.model.stop_training = True

                progress_callback = ProgressWindowCallback(progress_window)
                callbacks = [model_checkpoint, early_stopping, reduce_lr, progress_callback]

                # Train model
                main_window.tree_overlay.show_processing(f"Training model...")

                print(f"\nTraining configuration:")
                print(f"  Epochs: {epochs}")
                print(f"  Batch size: {batch_size}")
                print(f"  Learning rate: {learning_rate}")
                print("-"*80)

                history = classifier.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                )

                # Save final model
                main_window.tree_overlay.show_processing("Saving model...")

                final_model_path = os.path.join(unique_output_dir, 'pointnet_final.keras')
                classifier.save(final_model_path)

                # Save class mapping
                mapping_path = os.path.join(unique_output_dir, 'class_mapping.json')
                with open(mapping_path, 'w') as f:
                    json.dump(class_mapping, f, indent=2)

                # Save training metadata
                training_metadata = {
                    'folder_name': folder_name,
                    'timestamp': timestamp,
                    'num_points': int(num_points),
                    'num_features': int(num_features),
                    'num_classes': int(num_classes),
                    'class_mapping': class_mapping,
                    'training_samples': int(len(X_train)),
                    'validation_samples': int(len(X_val)),
                    'epochs_completed': len(history.history['loss']),
                    'best_val_accuracy': float(max(history.history['val_sparse_categorical_accuracy'])),
                    'final_val_accuracy': float(history.history['val_sparse_categorical_accuracy'][-1]),
                    'use_tnet': use_tnet,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'early_stopping_patience': early_stopping_patience,
                    'validation_split': val_split,
                    'random_seed': random_seed,
                    'run_number': run_number,
                    'total_repetitions': repetitions,
                    'source_metadata': metadata
                }

                metadata_path = os.path.join(unique_output_dir, 'training_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(training_metadata, f, indent=2)

                # Check if training was cancelled
                was_cancelled = progress_window.training_cancelled

                # Mark training as completed in progress window
                progress_window.training_completed(
                    training_metadata['best_val_accuracy'],
                    cancelled=was_cancelled
                )

                # Close the progress window immediately after completion
                try:
                    progress_window.close()
                    print(f"\nProgress window for Run #{run_number} closed.")
                except:
                    pass  # Ignore errors if window already closed

                # Print results
                print("\n" + "="*80)
                if was_cancelled:
                    print("Training Cancelled by User")
                else:
                    print("Training Complete!")
                print("="*80)
                print(f"Best validation accuracy: {training_metadata['best_val_accuracy']:.4f}")
                print(f"Epochs completed: {training_metadata['epochs_completed']}")
                print(f"Models saved to: {unique_output_dir}/")
                print("="*80)

                # Show completion message (only for single run, summary shown for multiple)
                if repetitions == 1:
                    if was_cancelled:
                        QMessageBox.information(
                            main_window,
                            "Training Cancelled",
                            f"Training was cancelled by user.\n\n"
                            f"Best validation accuracy: {training_metadata['best_val_accuracy']:.2%}\n"
                            f"Epochs completed: {training_metadata['epochs_completed']}\n\n"
                            f"Best model saved to:\n{unique_output_dir}/\n\n"
                            f"Training logged to: training_history.csv"
                        )
                    else:
                        QMessageBox.information(
                            main_window,
                            "Training Complete",
                            f"PointNet training completed successfully!\n\n"
                            f"Best validation accuracy: {training_metadata['best_val_accuracy']:.2%}\n"
                            f"Epochs completed: {training_metadata['epochs_completed']}\n\n"
                            f"Models saved to:\n{unique_output_dir}/\n\n"
                            f"Training logged to: training_history.csv"
                        )

                # Track this run's results
                run_result = {
                    'run_number': run_number,
                    'random_seed': random_seed,
                    'best_val_accuracy': training_metadata['best_val_accuracy'],
                    'epochs_completed': training_metadata['epochs_completed'],
                    'output_dir': unique_output_dir,
                    'was_cancelled': was_cancelled
                }
                all_run_results.append(run_result)

                # Update best overall
                if training_metadata['best_val_accuracy'] > best_overall_accuracy:
                    best_overall_accuracy = training_metadata['best_val_accuracy']
                    best_overall_run = run_number

                # Write to central tracking CSV
                csv_data = {
                    'folder_name': folder_name,
                    'timestamp': timestamp,
                    'run_number': run_number,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'val_split': val_split,
                    'use_tnet': use_tnet,
                    'early_stopping_patience': early_stopping_patience,
                    'repetitions': repetitions,
                    'random_seed': random_seed,
                    'best_val_acc': training_metadata['best_val_accuracy'],
                    'final_val_acc': training_metadata['final_val_accuracy'],
                    'epochs_completed': training_metadata['epochs_completed'],
                    'training_samples': training_metadata['training_samples'],
                    'validation_samples': training_metadata['validation_samples'],
                    'num_classes': training_metadata['num_classes'],
                    'was_cancelled': was_cancelled
                }
                self._write_to_tracking_csv(output_dir, csv_data)
                print(f"Training results logged to: {os.path.join(output_dir, 'training_history.csv')}")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"\nERROR during training run {run_number}:\n{error_msg}")

                # Mark progress window as complete so it can be closed
                try:
                    if 'progress_window' in locals():
                        progress_window.training_complete = True
                        progress_window.cancel_button.setVisible(False)
                        progress_window.close_button.setVisible(True)
                        progress_window.status_label.setText("Training failed - see error message")
                except:
                    pass

                QMessageBox.critical(
                    main_window,
                    f"Training Error (Run {run_number}/{repetitions})",
                    f"An error occurred during training run {run_number}:\n\n{str(e)}\n\n"
                    f"See console for full traceback."
                )

            finally:
                # Re-enable UI
                main_window.tree_overlay.hide_processing()
                main_window.enable_menus()
                main_window.enable_tree()

        # Print summary of all runs
        if repetitions > 1 and len(all_run_results) > 0:
            print(f"\n{'='*80}")
            print(f"TRAINING SUMMARY - {len(all_run_results)} RUN(S) COMPLETED")
            print(f"{'='*80}")
            for result in all_run_results:
                status = "CANCELLED" if result['was_cancelled'] else "COMPLETE"
                best_marker = " ⭐ BEST" if result['run_number'] == best_overall_run else ""
                print(f"Run #{result['run_number']}: {result['best_val_accuracy']:.2%} ({result['epochs_completed']} epochs) [{status}]{best_marker}")
                print(f"  Seed: {result['random_seed']}")
                print(f"  Path: {result['output_dir']}")
            print(f"\nBest run: #{best_overall_run} with {best_overall_accuracy:.2%} validation accuracy")
            print(f"\nAll results saved to: {os.path.join(output_dir, 'training_history.csv')}")
            print(f"{'='*80}")

            # Show summary message box
            summary_msg = f"Completed {len(all_run_results)} training run(s)\n\n"
            summary_msg += f"BEST RESULT: Run #{best_overall_run} - {best_overall_accuracy:.2%}\n\n"
            summary_msg += "All runs:\n"
            for result in all_run_results:
                status = "✓" if not result['was_cancelled'] else "✗"
                best_marker = " <- BEST" if result['run_number'] == best_overall_run else ""
                summary_msg += f"{status} Run #{result['run_number']}: {result['best_val_accuracy']:.2%}{best_marker}\n"
            summary_msg += f"\nTracking file: training_history.csv\n"
            summary_msg += f"\nAll training windows have been closed automatically."

            QMessageBox.information(
                main_window,
                "Training Runs Complete",
                summary_msg
            )

    def load_training_data(self, data_dir):
        """
        Load training data from directory structure.

        Expected structure:
            data_dir/
                ClassA/
                    sample1.npy
                    sample2.npy
                ClassB/
                    sample1.npy
                metadata.json (optional)

        Args:
            data_dir: Root directory containing class subdirectories

        Returns:
            Tuple of (data, labels, class_mapping, metadata)
        """
        # Try to load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Find all class directories
        class_dirs = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                class_dirs.append(item)

        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {data_dir}")

        # Sort class names for consistent ordering
        class_dirs = sorted(class_dirs)

        # Create class mapping
        class_mapping = {i: class_name for i, class_name in enumerate(class_dirs)}

        # Load all samples
        all_data = []
        all_labels = []

        for class_id, class_name in class_mapping.items():
            class_dir = os.path.join(data_dir, class_name)

            # Find all .npy files
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]

            if len(npy_files) == 0:
                print(f"WARNING: No .npy files found in {class_dir}")
                continue

            for npy_file in npy_files:
                filepath = os.path.join(class_dir, npy_file)
                try:
                    sample = np.load(filepath)
                    all_data.append(sample)
                    all_labels.append(class_id)
                except Exception as e:
                    print(f"ERROR loading {filepath}: {e}")

        if len(all_data) == 0:
            raise ValueError("No valid samples loaded!")

        # Convert to numpy arrays
        data = np.array(all_data, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        return data, labels, class_mapping, metadata
