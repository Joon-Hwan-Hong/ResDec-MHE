"""Prediction head modules."""

from src.models.heads.bayesian_head import BayesianPredictionHead
from src.models.heads.deterministic_head import DeterministicPredictionHead

__all__ = ["BayesianPredictionHead", "DeterministicPredictionHead"]
