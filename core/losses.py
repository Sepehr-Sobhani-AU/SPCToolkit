# core/losses.py
"""
Custom loss functions for handling class imbalance in PointNet training.

Provides:
- FocalLoss: Down-weights easy examples, focuses on hard ones
- compute_class_weights: Multiple methods for computing per-class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    The key insight is that easy examples (high p) contribute less to the loss
    due to the (1-p)^gamma term, allowing the model to focus on hard examples.

    Args:
        alpha: Per-class weights (tensor, list, or ndarray of shape [num_classes]).
               If None, no class weighting is applied.
        gamma: Focusing parameter (default: 2.0).
               - gamma=0: Equivalent to weighted cross-entropy
               - gamma=2: Standard choice, good balance
               - gamma=5: Very aggressive focus on hard examples
        reduction: How to reduce the loss across the batch.
                   'mean' (default), 'sum', or 'none'

    Example:
        >>> # Compute class weights from training data
        >>> class_counts = [1000, 50, 25]  # Imbalanced
        >>> weights = compute_class_weights(class_counts, method='effective_num')
        >>> criterion = FocalLoss(alpha=weights, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Register alpha as buffer so it moves to device with model
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes)
            targets: Ground truth class indices of shape (batch_size,)

        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum',
            tensor of shape (batch_size,) if reduction='none')
        """
        # Get probabilities via softmax
        probs = F.softmax(logits, dim=1)

        # Get the probability of the true class for each sample
        batch_size = targets.size(0)
        device = logits.device
        p_t = probs[torch.arange(batch_size, device=device), targets]

        # Compute focal weight: (1 - p)^gamma
        # This down-weights well-classified examples (high p_t)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p)
        # Add small epsilon to prevent log(0)
        ce_loss = -torch.log(p_t + 1e-8)

        # Apply focal weighting
        loss = focal_weight * ce_loss

        # Apply per-class alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss combining effective number weighting with focal loss.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019

    The effective number of samples accounts for the diminishing returns of having
    more samples: E_n = (1 - beta^n) / (1 - beta), where n is the sample count.

    Args:
        samples_per_class: Array/list of sample counts per class
        beta: Hyperparameter for effective number calculation (default: 0.9999)
              Higher beta = more aggressive re-weighting
        gamma: Focal loss gamma parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'

    Example:
        >>> class_counts = [9691, 5868, 11, 1]  # Your SemanticKITTI distribution
        >>> criterion = ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        samples_per_class: Union[np.ndarray, List[int]],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.beta = beta

        # Compute effective number of samples
        samples = np.array(samples_per_class, dtype=np.float32)
        effective_num = 1.0 - np.power(beta, samples)
        weights = (1.0 - beta) / (effective_num + 1e-8)

        # Normalize weights so they sum to num_classes (mean = 1)
        weights = weights / np.sum(weights) * len(samples)

        self.register_buffer('weights', torch.FloatTensor(weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced focal loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes)
            targets: Ground truth class indices of shape (batch_size,)

        Returns:
            Loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)

        # Get p_t for each sample
        batch_size = targets.size(0)
        device = logits.device
        p_t = probs[torch.arange(batch_size, device=device), targets]

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy
        ce_loss = -torch.log(p_t + 1e-8)

        # Class-balanced weight
        class_weights = self.weights[targets]

        # Combine all weights
        loss = class_weights * focal_weight * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights(
    class_counts: Union[np.ndarray, List[int]],
    method: str = 'effective_num'
) -> np.ndarray:
    """
    Compute per-class weights for loss functions.

    Args:
        class_counts: Array of sample counts per class
        method: Weight calculation method:
            - 'balanced': sklearn style, n_samples / (n_classes * count)
            - 'inverse': Simple inverse frequency, 1 / count
            - 'inverse_sqrt': Less aggressive, 1 / sqrt(count)
            - 'effective_num': Class-Balanced Loss (CVPR 2019), recommended

    Returns:
        Normalized weight array (mean = 1.0)

    Example:
        >>> counts = [9691, 5868, 2879, 11, 1]
        >>> weights = compute_class_weights(counts, method='effective_num')
        >>> print(weights)  # Tiny classes get much higher weights
    """
    counts = np.array(class_counts, dtype=np.float32)

    # Ensure no zero counts (add small epsilon)
    counts = np.maximum(counts, 1e-8)

    if method == 'balanced':
        # sklearn style: n_samples / (n_classes * count)
        n_samples = counts.sum()
        n_classes = len(counts)
        weights = n_samples / (n_classes * counts)

    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / counts

    elif method == 'inverse_sqrt':
        # Less aggressive - inverse square root
        weights = 1.0 / np.sqrt(counts)

    elif method == 'effective_num':
        # Class-Balanced Loss (CVPR 2019)
        # Uses effective number of samples instead of raw counts
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Choose from: balanced, inverse, inverse_sqrt, effective_num")

    # Normalize so mean weight is 1.0
    # This keeps the loss magnitude similar to unweighted case
    weights = weights / np.mean(weights)

    return weights


def get_loss_function(
    loss_type: str,
    class_counts: Optional[Union[np.ndarray, List[int]]] = None,
    alpha_method: str = 'effective_num',
    gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to create a loss function based on configuration.

    Args:
        loss_type: Type of loss function:
            - 'ce': Standard cross-entropy (no class weighting)
            - 'weighted_ce': Weighted cross-entropy
            - 'focal': Focal loss with class weights
            - 'cb_focal': Class-balanced focal loss
        class_counts: Sample counts per class (required for weighted losses)
        alpha_method: Method for computing class weights
                      ('balanced', 'inverse', 'inverse_sqrt', 'effective_num')
        gamma: Focal loss gamma parameter

    Returns:
        PyTorch loss module

    Example:
        >>> class_counts = [9691, 5868, 11, 1]
        >>> criterion = get_loss_function(
        ...     'focal', class_counts, alpha_method='effective_num', gamma=2.0
        ... )
    """
    loss_type = loss_type.lower().replace(' ', '_').replace('-', '_')

    if loss_type == 'ce' or loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce' or loss_type == 'weighted_cross_entropy':
        if class_counts is None:
            raise ValueError("class_counts required for weighted cross-entropy")
        weights = compute_class_weights(class_counts, method=alpha_method)
        return nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))

    elif loss_type == 'focal' or loss_type == 'focal_loss':
        alpha = None
        if class_counts is not None:
            alpha = compute_class_weights(class_counts, method=alpha_method)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'cb_focal' or loss_type == 'class_balanced_focal':
        if class_counts is None:
            raise ValueError("class_counts required for class-balanced focal loss")
        return ClassBalancedFocalLoss(class_counts, gamma=gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from: ce, weighted_ce, focal, cb_focal")
