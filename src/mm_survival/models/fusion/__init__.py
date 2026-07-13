"""Fusion modules for multimodal survival models."""

from .concat import ConcatFusion
from .factory import build_fusion, list_fusion_methods
from .gated import ModalityGatedConcatFusion, ScalarModalityGate
from .lowrank_bilinear import LowRankBilinearFusion, PairwiseLowRankBilinearFusion

__all__ = [
    "ConcatFusion",
    "ScalarModalityGate",
    "ModalityGatedConcatFusion",
    "LowRankBilinearFusion",
    "PairwiseLowRankBilinearFusion",
    "build_fusion",
    "list_fusion_methods",
]
