"""
Battle helper module for Pokemon battle mechanics (Gen 9 accurate helpers)
----------------------------------------------------------------------------
This module provides:
- Type effectiveness helpers (Gen 9 chart, Freeze-Dry / Flying Press exceptions)
- Weather / Terrain / Screen / Spread modifiers
- STAB calculation including Terastallization + Adaptability rules
- Dataclasses to carry field/side state to the damage calculator

NOTE: We do not import poke-env here to let this file be imported without the lib.
If poke-env is available, you can pass in GenData.type_chart to the helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ----------------------------- Data containers ---------------------------------

@dataclass
class SideState:
    # Hazards
    stealth_rock: bool = False
    spikes: int = 0               # 0..3
    toxic_spikes: int = 0         # 0..2
    sticky_web: bool = False

    # Screens / team-wide effects
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False
    safeguard: bool = False
    tailwind: bool = False

    # Other common side conditions
    mist: bool = False
    craft_shield: bool = False     # placeholder for custom projects


@dataclass
class FieldState:
    # Battle-wide
    weather: Optional[str] = None      # 'sun', 'rain', 'sand', 'snow', 'desolateland', 'primordialsea', etc.
    terrain: Optional[str] = None      # 'electric', 'grassy', 'psychic', 'misty'

    gravity: bool = False
    trick_room: bool = False

    # Context
    is_doubles: bool = False
    # For screen reduction in doubles: if the move *actually* has multiple targets on the
    # target side at execution time, use 2/3; else 1/2. (We expose this value explicitly.)
    targets_on_target_side: int = 1

    # Side-specific (attacker perspective)
    attacker_side: SideState = field(default_factory=SideState)
    defender_side: SideState = field(default_factory=SideState)


# --------------------------- Type effectiveness --------------------------------

def type_effectiveness(
    move_type: str,
    defender_types: Sequence[str],
    type_chart: Dict[str, Dict[str, float]],
    move_id: Optional[str] = None,
) -> float:
    """Return the Gen 9 effectiveness multiplier for a move against defender types.

    Special cases:
      - Freeze-Dry (move_id == 'freezedry'): always super-effective vs Water (x2).
      - Flying Press (move_id == 'flyingpress'): combine Fighting *and* Flying.

    Args:
        move_type: Canonical type name, e.g., 'Water', 'Fire'. Case-insensitive.
        defender_types: Defender's types (1 or 2). Type names, case-insensitive.
        type_chart: Mapping Type -> vsType -> multiplier (e.g., GenData.type_chart).
        move_id: Showdown id for certain special-case moves.
    """
    if not defender_types:
        return 1.0

    mtype = move_type.capitalize()

    # Helper to multiply vs both defender types
    def mult_for(att_type: str) -> float:
        mult = 1.0
        for d in defender_types:
            mult *= type_chart.get(att_type, {}).get(d.capitalize(), 1.0)
        return mult

    # Special: Freeze-Dry
    if move_id == "freezedry":
        base = mult_for("Ice")
        if "Water" in [t.capitalize() for t in defender_types]:
            base *= 2.0
        return base

    # Special: Flying Press (Fighting+Flying stacked effectiveness)
    if move_id == "flyingpress":
        return mult_for("Fighting") * mult_for("Flying")

    # Normal case
    return mult_for(mtype)


# --------------------------- Modifiers (weather/terrain/screens/spread) ---------

def spread_modifier(is_doubles: bool, hits_multiple_targets: bool) -> float:
    """Return spread modifier: 0.75 for doubles when a move actually hits multiple targets."""
    return 0.75 if (is_doubles and hits_multiple_targets) else 1.0


def weather_modifier(move_type: str, weather: Optional[str], move_id: Optional[str] = None):
    """Return (modifier, move_fails) for weather.

    - Sun: Fire x1.5, Water x0.5 except Hydro Steam (x1.5).
    - Rain: Water x1.5, Fire x0.5.
    - Primordial Sea: Fire moves fail (modifier 0, move_fails True).
    - Desolate Land: Water moves fail (modifier 0, move_fails True).
    - Sand / Snow: no direct power mod.

    Returns:
        (multiplier, move_fails) where move_fails=True means the move does 0 damage.
    """
    if not weather:
        return 1.0, False

    w = weather.lower()
    t = move_type.capitalize()

    # Extremely harsh weathers first
    if w == "primordialsea":
        if t == "Fire":
            return 0.0, True
        if t == "Water":
            return 1.5, False  # Primordial Sea still boosts Water damage
        return 1.0, False
    if w == "desolateland":
        if t == "Water":
            return 0.0, True
        if t == "Fire":
            return 1.5, False
        return 1.0, False

    if w == "sun":
        if t == "Fire":
            return 1.5, False
        if t == "Water":
            # Hydro Steam exception (boosted instead of halved)
            if move_id == "hydrosteam":
                return 1.5, False
            return 0.5, False
        return 1.0, False

    if w == "rain":
        if t == "Water":
            return 1.5, False
        if t == "Fire":
            return 0.5, False
        return 1.0, False

    # sand / snow / delta_stream etc. -> handled elsewhere if needed
    return 1.0, False


def screen_modifier(
    category: str,
    defender_side: SideState,
    is_critical: bool,
    is_doubles: bool,
    targets_on_target_side: int,
) -> float:
    """Reflect / Light Screen / Aurora Veil reduction.

    - 0.5 in singles.
    - 2/3 in doubles *if* the executed move has multiple targets
      on that side; otherwise still 0.5.
    - Ignored by critical hits.
    - Aurora Veil applies to both categories (same reduction as screens).
    """
    if is_critical:
        return 1.0

    cat = (category or "").lower()
    has_screen = defender_side.aurora_veil or (cat == "physical" and defender_side.reflect) or (
        cat == "special" and defender_side.light_screen
    )
    if not has_screen:
        return 1.0

    if is_doubles and targets_on_target_side > 1:
        return 2.0 / 3.0
    return 0.5


def terrain_modifier(
    move_type: str,
    attacker_grounded: bool,
    target_grounded: bool,
    terrain: Optional[str],
) -> float:
    """Return terrain multiplier (exact Gen 8/9 factors).

    - Electric/Grassy/Psychic: ~1.3 for grounded attacker (exact 5325/4096).
    - Misty: halves Dragon vs *grounded* targets.
    """
    if not terrain:
        return 1.0

    t = terrain.lower()
    mtype = move_type.capitalize()

    if t in ("electric", "grassy", "psychic"):
        if attacker_grounded and (
            (t == "electric" and mtype == "Electric")
            or (t == "grassy" and mtype == "Grass")
            or (t == "psychic" and mtype == "Psychic")
        ):
            return 5325.0 / 4096.0
        return 1.0

    if t == "misty":
        if target_grounded and mtype == "Dragon":
            return 0.5
        return 1.0

    return 1.0


# --------------------------- STAB (with Tera + Adaptability) --------------------

def stab_multiplier(
    move_type: str,
    user_types: Sequence[str],
    tera_type: Optional[str] = None,
    ability: Optional[str] = None,
) -> float:
    """Compute STAB for move_type with Tera and Adaptability rules (Gen 9).

    Rules:
      - Normal STAB is 1.5 if move matches any of user's original types.
      - Terastallizing grants STAB for the Tera type.
      - If the Tera type equals an original type AND the move matches that type,
        STAB becomes 2.0 (not 1.5 * 1.5).
      - Adaptability (pre-Tera): STAB becomes 2.0 for matching original type.
      - Adaptability (Tera): applies only to the Tera type.
           * If Tera matches an original type and the move matches it -> 2.25
           * If Tera is a new type and the move matches it -> 2.0
           * Original types stay 1.5 under Adaptability when Tera is different.
    """
    mtype = move_type.capitalize()
    orig = {t.capitalize() for t in user_types if t}
    ter = tera_type.capitalize() if tera_type else None
    abil = (ability or "").lower()

    # If the move doesn't match Tera type:
    if ter is None or mtype != ter:
        if mtype in orig:
            return 2.0 if abil == "adaptability" else 1.5
        return 1.0

    # Move matches Tera type
    tera_matches_original = ter in orig
    if abil == "adaptability":
        return 2.25 if tera_matches_original else 2.0

    # No Adaptability
    return 2.0 if tera_matches_original else 1.5
