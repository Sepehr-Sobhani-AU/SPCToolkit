"""RANSAC infrastructure.

Single-cloud ``fit()`` contract with pluggable models, samplers, and
scorers. See ``core/services/RANSAC.md`` for the full design.
"""

from .base import RansacModel, Sampler, Scorer
from .engine import fit, fit_many, register_model
from .primitives.line import LineModel
from .primitives.plane import PlaneModel
from .samplers import UniformSampler
from .scorers import InlierCountScorer, MSACScorer

__all__ = [
    "fit",
    "fit_many",
    "register_model",
    "RansacModel",
    "Sampler",
    "Scorer",
    "LineModel",
    "PlaneModel",
    "UniformSampler",
    "MSACScorer",
    "InlierCountScorer",
]
