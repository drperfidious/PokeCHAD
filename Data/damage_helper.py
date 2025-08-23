"""
Damage calculation module for Pokemon battles (Gen 9 order & fixed-point modifiers)
-----------------------------------------------------------------------------------

Implements a faithful damage pipeline with fixed-point (1/4096) modifiers similar
to Pokémon Showdown's "chainModify" approach. The order of modifiers follows the
modern (Gen 6+) order used by Smogon/Showdown:

  1) Targets (spread)
  2) Weather
  3) Critical
  4) Random (0.85 .. 1.00, uniform discrete)
  5) STAB
  6) Type effectiveness
  7) Burn (physical only)
  8) Other (items/abilities/field)
  9) Screens (Reflect/Light Screen/Aurora Veil)
  10) Terrain

We also expose extension points to inject item/ability modifiers without coupling
this module to specific dex data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# We intentionally do not import poke-env here to keep this module importable in isolation.
# Callers are expected to provide type charts and battle/field data via helper modules.

# ---------------------------- Fixed-point helpers --------------------------------

FP_BASE = 4096  # Showdown-style fixed point

def to_fp(x: float) -> int:
    """Convert a float multiplier to fixed-point (rounded)."""
    # Handle tuple case (weather function returns tuple but should be unpacked)
    if isinstance(x, (list, tuple)):
        # This should only happen if weather_fn result wasn't unpacked correctly
        x = x[0] if x else 1.0
    return int(round(float(x) * FP_BASE))

def chain_mul(base_fp: int, mods_fp: Iterable[int]) -> int:
    """Chain fixed-point multipliers with rounding down each step."""
    out = base_fp
    for m in mods_fp:
        out = (out * m) // FP_BASE
    return out

def apply_fp(damage: int, mods_fp: Iterable[int]) -> int:
    """Apply fixed-point modifiers to integer damage."""
    mult = chain_mul(FP_BASE, mods_fp)
    return max(1, (damage * mult) // FP_BASE)


# ---------------------------- Dataclasses ----------------------------------------

@dataclass
class CombatantState:
    level: int
    types: Sequence[str]
    atk: int
    def_: int
    spa: int
    spd: int
    spe: int = 0

    # Battle-time flags
    tera_type: Optional[str] = None
    terastallized: bool = False
    grounded: bool = True
    is_burned: bool = False

    # Optional knowledge (revealed / estimated)
    ability: Optional[str] = None
    item: Optional[str] = None

    # Stat stages (-6..+6). If None, treat as 0.
    atk_stage: int = 0
    def_stage: int = 0
    spa_stage: int = 0
    spd_stage: int = 0
    spe_stage: int = 0


@dataclass
class MoveContext:
    move_id: str
    name: str
    type: str
    category: str         # 'Physical' | 'Special' | 'Status'
    base_power: int
    is_spread: bool = False
    hits_multiple_targets_on_execution: bool = False  # whether spread mod applies

    # convenience flags (fill from PS data if desired)
    makes_contact: bool = False
    is_sound: bool = False
    is_punch: bool = False
    is_biting: bool = False

    # Multihit (1 or [min,max])
    multihit: Optional[Sequence[int]] = None


@dataclass
class FieldState:
    # keep in sync with battle_helper.FieldState for convenience
    weather: Optional[str] = None
    terrain: Optional[str] = None
    gravity: bool = False
    trick_room: bool = False
    is_doubles: bool = False
    targets_on_target_side: int = 1

    # Screens on *defender* side
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False


@dataclass
class DamageResult:
    min_damage: int
    max_damage: int
    rolls: List[int]
    effectiveness: float
    is_crit: bool
    applied_modifiers: Dict[str, float] = field(default_factory=dict)


# ---------------------------- Utility: stat stages --------------------------------

_STAGE_NUM = [2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8]  # index by stage + 6
_STAGE_DEN = [8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2]

def apply_stage(stat: int, stage: int, ignore_positive: bool=False, ignore_negative: bool=False) -> int:
    """Apply a standard stage multiplier to a stat."""
    if stage == 0:
        return stat
    s = max(-6, min(6, stage))
    if (s > 0 and ignore_positive) or (s < 0 and ignore_negative):
        return stat
    num = _STAGE_NUM[s + 6]
    den = _STAGE_DEN[s + 6]
    return (stat * num) // den


# ---------------------------- Core damage ----------------------------------------

def _base_damage(level: int, base_power: int, atk: int, deff: int) -> int:
    """Core pre-mod damage integer math: floor(floor(floor(2L/5+2)*BP*A/D)/50)+2"""
    if base_power <= 0:
        return 0
    t1 = (2 * level) // 5 + 2
    t2 = (t1 * base_power * atk) // max(1, deff)
    t3 = t2 // 50
    return t3 + 2


def calc_damage_range(
    attacker: CombatantState,
    defender: CombatantState,
    move: MoveContext,
    field: FieldState,
    get_type_chart: Callable[[], Dict[str, Dict[str, float]]],
    is_critical: bool = False,
    extra_modifiers: Optional[List[float]] = None,
    type_effectiveness_fn: Optional[Callable[[str, Sequence[str], Dict[str, Dict[str, float]], Optional[str]], float]] = None,
    stab_fn: Optional[Callable[[str, Sequence[str], Optional[str], Optional[str], bool], float]] = None,
    weather_fn: Optional[Callable[[str, Optional[str], Optional[str]], Tuple[float, bool]]] = None,
    terrain_fn: Optional[Callable[[str, bool, bool, Optional[str]], float]] = None,
    screen_fn: Optional[Callable[[str, object, bool, bool, int], float]] = None,
) -> DamageResult:
    """
    Calculate damage range (16 rolls) using realistic order and fixed-point modifiers.

    The helpers are injectable and default to no-op if not provided.
    """
    category = move.category.capitalize()
    is_physical = category == "Physical"
    is_special = category == "Special"

    # Apply stat stages; crit ignores attacker's negative and defender's positive stages
    eff_atk = attacker.atk if is_physical else attacker.spa
    eff_def = defender.def_ if is_physical else defender.spd
    # Special case: Body Press uses user's Defense for the attacking stat
    is_body_press = (move.move_id or '').lower() == 'bodypress'
    if is_body_press:
        # Use attacker's Defense with Defense stage as the attacking stat
        eff_atk = attacker.def_
        eff_atk = apply_stage(eff_atk, attacker.def_stage, ignore_negative=is_critical)
        # Defender uses normal Defense with Defense stage (physical defense)
        eff_def = apply_stage(eff_def, defender.def_stage, ignore_positive=is_critical)
    else:
        eff_atk = apply_stage(eff_atk, attacker.atk_stage if is_physical else attacker.spa_stage,
                              ignore_negative=is_critical)
        eff_def = apply_stage(eff_def, defender.def_stage if is_physical else defender.spd_stage,
                              ignore_positive=is_critical)

    # Core base damage
    base = _base_damage(attacker.level, move.base_power, eff_atk, max(1, eff_def))

    # Gather fixed-point modifiers in the correct order
    mods_fp: List[int] = []

    # 1) Targets (spread)
    if move.is_spread and move.hits_multiple_targets_on_execution and field.is_doubles:
        mods_fp.append(to_fp(0.75))

    # 2) Weather
    move_fails = False
    if weather_fn:
        wmult, move_fails = weather_fn(move.type, field.weather, move.move_id)
        if move_fails:
            return DamageResult(0, 0, [0]*16, 1.0, is_critical, applied_modifiers={"weather": 0.0})
        mods_fp.append(to_fp(wmult))

    # 3) Critical
    if is_critical:
        # Sniper handling can be an extra modifier (2.25 total) via extra_modifiers
        mods_fp.append(to_fp(1.5))

    # 4) Random (85..100%)
    rand_vals = [x / 100 for x in range(85, 101)]

    # 5) STAB
    if stab_fn:
        stab = stab_fn(
            move.type,
            attacker.types,
            attacker.tera_type,
            attacker.ability,
            terastallized=attacker.terastallized,
        )
        mods_fp.append(to_fp(stab))

    # 6) Type effectiveness
    type_chart = get_type_chart() if get_type_chart else {}
    if type_effectiveness_fn and type_chart:
        eff = type_effectiveness_fn(move.type, defender.types, type_chart, move.move_id, 
                                    defender.tera_type, defender.terastallized)
    else:
        eff = 1.0

    # 7) Burn (physical only)
    burn_mult = 1.0
    if is_physical and attacker.is_burned:
        # Ignored by Guts; Facade is handled by caller (we expose extra_modifiers for that case).
        if (attacker.ability or "").lower() != "guts":
            burn_mult = 0.5
            mods_fp.append(to_fp(burn_mult))

    # 8) Other (items/abilities/field) -> provided by caller
    if extra_modifiers:
        for m in extra_modifiers:
            mods_fp.append(to_fp(m))

    # 9) Screens
    if screen_fn:
        # Build a lightweight side view from FieldState
        class _Side:
            def __init__(self, r, ls, av):
                self.reflect = r
                self.light_screen = ls
                self.aurora_veil = av
        side = _Side(field.reflect, field.light_screen, field.aurora_veil)
        s = screen_fn(move.category, side, is_critical, field.is_doubles, field.targets_on_target_side)
        mods_fp.append(to_fp(s))

    # 10) Terrain
    if terrain_fn:
        t = terrain_fn(move.type, attacker.grounded, defender.grounded, field.terrain)
        mods_fp.append(to_fp(t))

    # Build 16 rolls
    rolls: List[int] = []
    for r in rand_vals:
        dmg = apply_fp(base, [to_fp(r)] + mods_fp)
        # Apply type effectiveness at the end (as the only float step) to preserve integer chain size
        final = max(0, int(dmg * eff))
        rolls.append(final)

    return DamageResult(min(rolls), max(rolls), rolls, eff, is_critical,
                        applied_modifiers={
                            "spread": 0.75 if (move.is_spread and move.hits_multiple_targets_on_execution and field.is_doubles) else 1.0,
                            "weather": wmult if weather_fn else 1.0,
                            "crit": 1.5 if is_critical else 1.0,
                            "stab": stab if stab_fn else 1.0,
                            "burn": burn_mult,
                            "terrain": t if terrain_fn else 1.0,
                        })
