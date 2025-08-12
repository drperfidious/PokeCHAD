
# speed_priority_rules.py
"""
Shared priority and speed rules (Gen 9) for turn order prediction.

This module exposes small, declarative tables + helpers that `turn_order.py`
imports. Keeping them here lets you swap/update pieces using Showdown data later.

Exports
-------
- to_id(s) -> str
- PRIORITY_BOOST_ABILITIES: id -> PriorityBoost(amount:int, applies:callable(ctx)->bool)
- CONDITIONAL_PRIORITY_MOVES: move_id -> callable(ctx)->int  (priority delta)
- PRIORITY_BLOCKERS: [callable(ctx)->bool]  (Psychic Terrain, Dazzling-line, Prankster vs Dark)
- PRECEDENCE_OVERRIDES: id -> PrecedenceOverride(always_first, always_last, chance_first, only_if?)
- SPEED_MULTIPLIER_ITEMS: id -> float (e.g., 1.5 for Choice Scarf)
- SPEED_MULTIPLIER_ABILITIES: id -> callable(ctx)->float
- SPEED_MULTIPLIER_FIELD: { 'tailwind': callable(ctx)->float }
- STATUS_SPEED_MULTIPLIER: callable(ctx)->float (paralysis, Quick Feet cancel)
- stage_multiplier(stage:int) -> (num, den)
- chain_mul(base:int, muls: list[tuple|float]) -> int
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional, Union

# --------------------------- utils ---------------------------

def to_id(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def stage_multiplier(stage: int) -> Tuple[int, int]:
    s = max(-6, min(6, int(stage)))
    if s >= 0:
        return 2 + s, 2
    return 2, 2 - s

FP = 4096

def _to_fp(x: float) -> int:
    return int(round(x * FP))

def chain_mul(base: int, muls: List[Union[Tuple[int,int], float]]) -> int:
    """PS-like chained multipliers with fixed-point rounding."""
    val = int(base)
    for m in muls:
        if isinstance(m, tuple):
            num, den = m
            # staged integer ratio first
            val = (val * int(num)) // max(1, int(den))
        else:
            # float multiplier via fixed-point
            val = (val * _to_fp(float(m))) // FP
        if val < 1:
            val = 1
    return val

# --------------------------- priority boosts ---------------------------

@dataclass(frozen=True)
class PriorityBoost:
    amount: int
    applies: Callable[[Dict[str, Any]], bool]

# Ability-based priority increases
PRIORITY_BOOST_ABILITIES: Dict[str, PriorityBoost] = {
    # +1 to Status moves
    "prankster": PriorityBoost(
        amount=1,
        applies=lambda ctx: str(ctx.get("category","")).lower() == "status"
    ),
    # +3 to healing/draining moves
    "triage": PriorityBoost(
        amount=3,
        applies=lambda ctx: bool(ctx.get("is_healing_or_drain", False))
    ),
    # +1 to Flying-type moves at full HP
    "galewings": PriorityBoost(
        amount=1,
        applies=lambda ctx: str(ctx.get("type","")).lower() == "flying" and bool(ctx.get("user_hp_is_full", False))
    ),
}

# Move-specific *conditional* priority deltas
CONDITIONAL_PRIORITY_MOVES: Dict[str, Callable[[Dict[str, Any]], int]] = {
    # Grassy Glide: +1 only in Grassy Terrain
    "grassyglide": lambda ctx: 1 if str(ctx.get("terrain","")).lower() == "grassy" else 0,
}

# --------------------------- priority blockers ---------------------------

# Each blocker receives a context dict containing:
#   'priority_after_mods' (int), 'terrain' (str), 'target_is_grounded' (bool),
#   'opponent_has_priority_block_ability' (bool), 'is_prankster_applied' (bool),
#   'target_is_dark' (bool)
def _block_psychic_terrain(ctx: Dict[str, Any]) -> bool:
    return (
        int(ctx.get("priority_after_mods", 0)) > 0
        and str(ctx.get("terrain","")).lower() == "psychic"
        and bool(ctx.get("target_is_grounded", True))
    )

def _block_dazzling_line(ctx: Dict[str, Any]) -> bool:
    return int(ctx.get("priority_after_mods", 0)) > 0 and bool(ctx.get("opponent_has_priority_block_ability", False))

def _block_prankster_vs_dark(ctx: Dict[str, Any]) -> bool:
    return int(ctx.get("priority_after_mods", 0)) > 0 and bool(ctx.get("is_prankster_applied", False)) and bool(ctx.get("target_is_dark", False))

PRIORITY_BLOCKERS = [_block_psychic_terrain, _block_dazzling_line, _block_prankster_vs_dark]

# --------------------------- precedence overrides ---------------------------

@dataclass(frozen=True)
class PrecedenceOverride:
    always_first: bool = False
    always_last: bool = False
    chance_first: float = 0.0
    only_if: Optional[Callable[[Dict[str, Any]], bool]] = None

# Items/abilities that alter *within-bracket* execution precedence
PRECEDENCE_OVERRIDES: Dict[str, PrecedenceOverride] = {
    # Random "move first" within bracket
    "quickclaw": PrecedenceOverride(chance_first=0.20),
    "quickdraw": PrecedenceOverride(chance_first=0.30, only_if=lambda ctx: str(ctx.get("category","")).lower() in {"physical","special"}),
    # Force last
    "laggingtail": PrecedenceOverride(always_last=True),
    "fullincense": PrecedenceOverride(always_last=True),
    "stall": PrecedenceOverride(always_last=True),  # ability
    # Mycelium Might: status moves go last in bracket
    "myceliummight": PrecedenceOverride(always_last=True, only_if=lambda ctx: str(ctx.get("category","")).lower() == "status"),
    # Custap Berry (engine can pass custap_active in ctx when threshold met)
    "custapberry": PrecedenceOverride(always_first=True, only_if=lambda ctx: bool(ctx.get("custap_active", False))),
}

# --------------------------- speed multipliers ---------------------------

SPEED_MULTIPLIER_ITEMS: Dict[str, float] = {
    "choicescarf": 1.5,
    "ironball": 0.5,
    "machobrace": 0.5,
    "poweranklet": 0.5,
    "powerband": 0.5,
    "powerbelt": 0.5,
    "powerbracer": 0.5,
    "powerlens": 0.5,
    "powerweight": 0.5,
    "quickpowder": 2.0,  # Ditto only (handled by caller via is_ditto_untransformed)
}

def _is_sun(weather: Optional[str]) -> bool:
    return str(weather or "").lower() in {"sun", "harshsunlight", "desolateland"}

def _is_rain(weather: Optional[str]) -> bool:
    return str(weather or "").lower() in {"rain", "heavyrain", "primordialsea"}

def _is_sand(weather: Optional[str]) -> bool:
    return str(weather or "").lower() in {"sand", "sandstorm"}

def _is_hail_or_snow(weather: Optional[str]) -> bool:
    return str(weather or "").lower() in {"hail", "snow"}

def _is_electric(terrain: Optional[str]) -> bool:
    return str(terrain or "").lower() == "electric"

# Ability -> multiplier function
def _chlorophyll(ctx: Dict[str, Any]) -> float: return 2.0 if _is_sun(ctx.get("weather")) else 1.0
def _swiftswim(ctx: Dict[str, Any]) -> float: return 2.0 if _is_rain(ctx.get("weather")) else 1.0
def _sandrush(ctx: Dict[str, Any]) -> float: return 2.0 if _is_sand(ctx.get("weather")) else 1.0
def _slushrush(ctx: Dict[str, Any]) -> float: return 2.0 if _is_hail_or_snow(ctx.get("weather")) else 1.0
def _surgesurfer(ctx: Dict[str, Any]) -> float: return 2.0 if _is_electric(ctx.get("terrain")) else 1.0
def _quickfeet(ctx: Dict[str, Any]) -> float: return 1.5 if bool(ctx.get("user_is_statused", False)) else 1.0
def _unburden(ctx: Dict[str, Any]) -> float: return 2.0 if bool(ctx.get("user_unburden_active", False)) else 1.0
def _slowstart(ctx: Dict[str, Any]) -> float: return 0.5 if bool(ctx.get("user_slowstart_active", False)) else 1.0
def _proto_quark_speed(ctx: Dict[str, Any]) -> float:
    return 1.5 if (bool(ctx.get("user_protosynthesis_speed", False)) or bool(ctx.get("user_quarkdrive_speed", False))) else 1.0

SPEED_MULTIPLIER_ABILITIES: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "chlorophyll": _chlorophyll,
    "swiftswim": _swiftswim,
    "sandrush": _sandrush,
    "slushrush": _slushrush,
    "surgesurfer": _surgesurfer,
    "quickfeet": _quickfeet,
    "unburden": _unburden,
    "slowstart": _slowstart,
    "protosynthesis": _proto_quark_speed,
    "quarkdrive": _proto_quark_speed,
}

SPEED_MULTIPLIER_FIELD: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "tailwind": lambda ctx: 2.0 if bool(ctx.get("side_tailwind_active", False)) else 1.0
}

# Paralysis / Quick Feet cancel
def STATUS_SPEED_MULTIPLIER(ctx: Dict[str, Any]) -> float:
    if bool(ctx.get("user_is_paralyzed", False)) and not bool(ctx.get("user_has_quickfeet", False)):
        return 0.5
    return 1.0
