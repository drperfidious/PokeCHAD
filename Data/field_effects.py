
# field_effects.py
"""
Entry hazard & terrain/status helpers.

Key functions
-------------
- apply_entry_hazards_on_switch_in(state, side, field, type_chart, *, mutate=True) -> dict
    Applies Stealth Rock / Spikes / Toxic Spikes / Sticky Web upon switch-in,
    honoring groundedness, Heavy-Duty Boots, Magic Guard (damage only),
    Poison-type absorption, and Misty Terrain (status block).

Assumptions
-----------
- `side` is a simple dict or dataclass-like with attributes: stealth_rock(bool),
  spikes(int 0..3), toxic_spikes(int 0..2), sticky_web(bool).
- `state` is team_state.PokemonState (has types, item, ability, grounded, stats).
- `type_chart` is Showdown-style chart: type->vsType->multiplier.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from .team_state import PokemonState
from .stat_effects import augment_grounded, apply_stage_change

def _to_id(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def _type_effectiveness(att_type: str, defender_types, type_chart: Dict[str, Dict[str, float]]) -> float:
    mult = 1.0
    for t in defender_types:
        if not t: continue
        mult *= type_chart.get(att_type.capitalize(), {}).get(t.capitalize(), 1.0)
    return mult

def apply_entry_hazards_on_switch_in(
    state: PokemonState,
    side: Any,
    field: Dict[str, Any],
    type_chart: Dict[str, Dict[str, float]],
    *,
    mutate: bool = True
) -> Dict[str, Any]:
    res = {
        "damage_taken": 0,          # integer HP lost (if mutate=True), else 0
        "fraction_lost": 0.0,       # fraction of max HP
        "poisoned": False,
        "toxic_spikes_absorbed": False,
        "sticky_web_applied": False,
    }

    # Heavy-Duty Boots prevent *all* hazard effects (including Sticky Web / TSpikes status)
    item = _to_id(state.item or "")
    if item == "heavydutyboots":
        return res

    # Determine groundedness (Air Balloon, Gravity, Flying/Levitate, Iron Ball, etc.)
    grounded = state.grounded
    if grounded is None:
        grounded = augment_grounded(state, field)

    # Stealth Rock: 1/8 * Rock effectiveness
    if getattr(side, "stealth_rock", False):
        eff = _type_effectiveness("Rock", state.types, type_chart)
        frac = 1.0/8.0 * eff
        # Magic Guard prevents hazard *damage*, not status changes
        if _to_id(state.ability or "") == "magicguard":
            frac = 0.0
        res["fraction_lost"] += frac

    # Spikes: grounded only; 1/8, 1/6, 1/4 for 1/2/3 layers
    layers = int(getattr(side, "spikes", 0) or 0)
    if grounded and layers > 0:
        if _to_id(state.ability or "") != "magicguard":
            if layers == 1: res["fraction_lost"] += 1.0/8.0
            elif layers == 2: res["fraction_lost"] += 1.0/6.0
            else: res["fraction_lost"] += 1.0/4.0

    # Toxic Spikes: grounded non-Poisoned immune types; Poison-types absorb all layers
    tsl = int(getattr(side, "toxic_spikes", 0) or 0)
    if grounded and tsl > 0:
        types = {t for t in state.types if t}
        if "poison" in types:
            # Absorb and clear
            if hasattr(side, "toxic_spikes"):
                side.toxic_spikes = 0
            res["toxic_spikes_absorbed"] = True
        elif "steel" in types or "poison" in types:
            pass  # immune to poison
        else:
            # Misty Terrain prevents new status on grounded
            if (field.get("terrain") or "").lower() == "misty":
                pass
            else:
                res["poisoned"] = True  # caller decides psn vs tox based on layers & immunity
                # We don't mutate state.status here; battle engine should do it

    # Sticky Web: grounded only; Speed -1 (stage). Not blocked by Magic Guard.
    if grounded and getattr(side, "sticky_web", False):
        # Apply a stage drop with prevention checks (Clear Amulet, Clear Body, etc.)
        applied = apply_stage_change(state, {"spe": -1}, source=None, cause="stickyweb")
        res["sticky_web_applied"] = any(v != 0 for v in applied.values())

    # Mutate HP
    if mutate and res["fraction_lost"] > 0 and state.max_hp:
        dmg = int(state.max_hp * res["fraction_lost"] + 1e-6)
        res["damage_taken"] = max(1, dmg)  # hazard damage is at least 1 if any
        state.current_hp = max(0, (state.current_hp or state.max_hp) - res["damage_taken"])

    return res
