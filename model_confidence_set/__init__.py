"""
model-confidence-set.

This package provides a Python implementation of the Model Confidence Set (MCS) procedure (Hansen, Lunde, and Nason, 2011), a statistical method for comparing and selecting models based on their performance. It allows users to identify a set of models that are statistically indistinguishable from the best model, given a statistical confidence level.
"""
from .core import ModelConfidenceSet

__version__ = "0.1.0"
__author__ = "Jonathan Chassot"
__all__ = ["ModelConfidenceSet"]