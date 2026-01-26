# core/metrics.py
"""
Per-class metrics for evaluating models on imbalanced data.

When dealing with imbalanced datasets, overall accuracy is misleading.
This module provides per-class metrics (precision, recall, F1) and
macro-averaged metrics that treat all classes equally.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class PerClassMetrics:
    """
    Track per-class precision, recall, F1, and confusion matrix.

    For imbalanced data, macro-averaged F1 is typically more meaningful
    than overall accuracy, as it treats all classes equally regardless
    of their sample counts.

    Example:
        >>> class_names = {0: 'vegetation', 1: 'building', 2: 'person'}
        >>> metrics = PerClassMetrics(class_names)
        >>>
        >>> # During validation
        >>> for batch_data, batch_labels in val_loader:
        ...     predictions = model(batch_data).argmax(dim=1)
        ...     metrics.update(predictions, batch_labels)
        >>>
        >>> # After validation
        >>> print(metrics.format_report())
        >>> print(f"Macro F1: {metrics.get_macro_f1():.4f}")
    """

    def __init__(self, class_names: Dict[int, str]):
        """
        Initialize metrics tracker.

        Args:
            class_names: Dictionary mapping class IDs (0, 1, 2, ...) to class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self):
        """Reset all counters for a new epoch/evaluation."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self._total_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: Predicted class indices (batch_size,)
            targets: True class indices (batch_size,)
        """
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()

        # Update confusion matrix
        for pred, target in zip(preds, targs):
            if 0 <= target < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[target, pred] += 1

        self._total_samples += len(preds)

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get the confusion matrix.

        Returns:
            Array of shape (num_classes, num_classes) where entry [i, j]
            is the count of samples with true class i predicted as class j.
        """
        return self.confusion_matrix.copy()

    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, F1, and support.

        Returns:
            Dictionary mapping class names to their metrics:
            {
                'class_name': {
                    'precision': float,
                    'recall': float,
                    'f1': float,
                    'accuracy': float,  # Per-class accuracy
                    'support': int      # Number of true samples
                }
            }
        """
        metrics = {}

        for class_id, class_name in self.class_names.items():
            # True positives: diagonal element
            tp = self.confusion_matrix[class_id, class_id]

            # False positives: sum of column minus diagonal
            fp = self.confusion_matrix[:, class_id].sum() - tp

            # False negatives: sum of row minus diagonal
            fn = self.confusion_matrix[class_id, :].sum() - tp

            # Support: total true samples of this class (row sum)
            support = self.confusion_matrix[class_id, :].sum()

            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp + 1e-8)

            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn + 1e-8)

            # F1: Harmonic mean of precision and recall
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # Per-class accuracy: TP / support
            accuracy = tp / (support + 1e-8)

            metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'support': int(support)
            }

        return metrics

    def get_overall_accuracy(self) -> float:
        """
        Get overall accuracy (correct predictions / total predictions).

        Note: This metric is misleading for imbalanced data!
        A model predicting only the majority class can achieve high accuracy.
        """
        correct = np.trace(self.confusion_matrix)  # Sum of diagonal
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        return float(correct / total)

    def get_macro_f1(self) -> float:
        """
        Get macro-averaged F1 score.

        This is the mean of per-class F1 scores, treating all classes equally.
        Recommended metric for imbalanced data as it doesn't favor majority classes.
        """
        metrics = self.get_per_class_metrics()
        f1_scores = [m['f1'] for m in metrics.values()]
        return float(np.mean(f1_scores))

    def get_weighted_f1(self) -> float:
        """
        Get weighted F1 score (weighted by class support).

        Each class's F1 is weighted by its sample count.
        Less affected by poor performance on tiny classes.
        """
        metrics = self.get_per_class_metrics()
        f1_scores = [m['f1'] for m in metrics.values()]
        supports = [m['support'] for m in metrics.values()]

        total_support = sum(supports)
        if total_support == 0:
            return 0.0

        weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
        return float(weighted_f1)

    def get_macro_precision(self) -> float:
        """Get macro-averaged precision."""
        metrics = self.get_per_class_metrics()
        precisions = [m['precision'] for m in metrics.values()]
        return float(np.mean(precisions))

    def get_macro_recall(self) -> float:
        """Get macro-averaged recall."""
        metrics = self.get_per_class_metrics()
        recalls = [m['recall'] for m in metrics.values()]
        return float(np.mean(recalls))

    def format_report(self, sort_by: str = 'name') -> str:
        """
        Format a classification report string.

        Args:
            sort_by: How to sort classes: 'name', 'f1', 'support'

        Returns:
            Formatted string report suitable for printing
        """
        metrics = self.get_per_class_metrics()

        # Sort classes
        if sort_by == 'f1':
            sorted_items = sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        elif sort_by == 'support':
            sorted_items = sorted(metrics.items(), key=lambda x: x[1]['support'], reverse=True)
        else:  # 'name'
            sorted_items = sorted(metrics.items(), key=lambda x: x[0])

        # Build report
        lines = []
        lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 62)

        for class_name, m in sorted_items:
            lines.append(
                f"{class_name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['support']:>10d}"
            )

        lines.append("-" * 62)
        lines.append(f"{'Overall Accuracy:':<20} {self.get_overall_accuracy():>10.4f}")
        lines.append(f"{'Macro Precision:':<20} {self.get_macro_precision():>10.4f}")
        lines.append(f"{'Macro Recall:':<20} {self.get_macro_recall():>10.4f}")
        lines.append(f"{'Macro F1:':<20} {self.get_macro_f1():>10.4f}")
        lines.append(f"{'Weighted F1:':<20} {self.get_weighted_f1():>10.4f}")
        lines.append(f"{'Total Samples:':<20} {self._total_samples:>10d}")

        return "\n".join(lines)

    def get_worst_classes(self, n: int = 5, metric: str = 'f1') -> Dict[str, Dict[str, float]]:
        """
        Get the n worst-performing classes.

        Useful for identifying which minority classes need attention.

        Args:
            n: Number of classes to return
            metric: Metric to sort by ('f1', 'precision', 'recall')

        Returns:
            Dictionary of worst-performing classes with their metrics
        """
        metrics = self.get_per_class_metrics()
        sorted_items = sorted(metrics.items(), key=lambda x: x[1][metric])
        return dict(sorted_items[:n])

    def get_best_classes(self, n: int = 5, metric: str = 'f1') -> Dict[str, Dict[str, float]]:
        """
        Get the n best-performing classes.

        Args:
            n: Number of classes to return
            metric: Metric to sort by ('f1', 'precision', 'recall')

        Returns:
            Dictionary of best-performing classes with their metrics
        """
        metrics = self.get_per_class_metrics()
        sorted_items = sorted(metrics.items(), key=lambda x: x[1][metric], reverse=True)
        return dict(sorted_items[:n])

    def to_dict(self) -> Dict:
        """
        Convert all metrics to a dictionary for JSON serialization.

        Returns:
            Dictionary containing all metrics and the confusion matrix
        """
        return {
            'per_class': self.get_per_class_metrics(),
            'overall_accuracy': self.get_overall_accuracy(),
            'macro_precision': self.get_macro_precision(),
            'macro_recall': self.get_macro_recall(),
            'macro_f1': self.get_macro_f1(),
            'weighted_f1': self.get_weighted_f1(),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'total_samples': self._total_samples
        }


def validate_with_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_mapping: Dict[int, str]
) -> Tuple[float, float, PerClassMetrics]:
    """
    Validate model and compute per-class metrics.

    Convenience function that runs validation and returns comprehensive metrics.

    Args:
        model: PyTorch model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        class_mapping: Dictionary mapping class IDs to names

    Returns:
        Tuple of (average_loss, overall_accuracy, PerClassMetrics instance)

    Example:
        >>> val_loss, val_acc, metrics = validate_with_metrics(
        ...     model, val_loader, criterion, device, class_mapping
        ... )
        >>> print(f"Val Loss: {val_loss:.4f}, Macro F1: {metrics.get_macro_f1():.4f}")
        >>> print(metrics.format_report())
    """
    model.eval()
    total_loss = 0.0

    # Initialize metrics tracker
    metrics = PerClassMetrics(class_mapping)

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            # Accumulate loss
            total_loss += loss.item() * batch_data.size(0)

            # Get predictions and update metrics
            predictions = torch.argmax(logits, dim=1)
            metrics.update(predictions, batch_labels)

    # Calculate averages
    avg_loss = total_loss / metrics._total_samples if metrics._total_samples > 0 else 0.0
    accuracy = metrics.get_overall_accuracy()

    return avg_loss, accuracy, metrics
