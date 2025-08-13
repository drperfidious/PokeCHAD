"""Definitions of weight search space for evolutionary tuner.

Bounds kept generous but finite to avoid runaway explosion.
"""
from __future__ import annotations
from typing import Dict, Tuple

# Inclusive min, max for each tunable weight
WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    'expected_mult': (0.0, 3.0),
    'go_first_bonus': (0.0, 2.0),
    'opp_dmg_penalty': (0.0, 3.0),
    'survival_bonus': (0.0, 3.0),
    'accuracy_mult': (0.0, 2.0),
    'effectiveness_mult': (0.0, 2.5),
    'ko_bonus': (0.0, 3.0),
    'switch_outgoing_mult': (0.0, 3.0),
    'switch_incoming_penalty': (0.0, 3.0),
}

DEFAULT_MUTATION_SCALE: Dict[str, float] = {
    k: (bounds[1]-bounds[0]) * 0.15 for k, bounds in WEIGHT_BOUNDS.items()
}

SEARCH_KEYS = list(WEIGHT_BOUNDS.keys())

__all__ = ['WEIGHT_BOUNDS','DEFAULT_MUTATION_SCALE','SEARCH_KEYS']

