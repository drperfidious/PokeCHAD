
# stat_effects.py
"""
Stat change + passive multipliers engine (abilities & items), designed to
cooperate with team_state.PokemonState snapshots and Showdown JSON (optional).

It provides:
- apply_stage_change(state, deltas, source=..., dex=...) -> actual_applied_deltas
- on_switch_in(user, opponents, dex, context) -> mutates states for entry abilities/items
- compute_passive_multipliers(state, field, dex) -> dict per-STAT multiplicative mods
- augment_grounded(state, field, dex) -> sets state.grounded if determinable

Notes:
- This module mutates PokemonState.stats.boosts in-place for stages and may
  mutate item/ability fields (e.g., consume White Herb) via convenience helpers.
- We honor Simple (double stage magnitude) and Contrary (invert sign) when
  applying stage deltas.
- We honor "prevention/redirect" abilities/items: Clear Body/Full Metal Body/White Smoke,
  Mirror Armor, Hyper Cutter, Big Pecks, Keen Eye, Clear Amulet, Guard Dog.
- We trigger "on stat lowered" reactions: Defiant, Competitive, Adrenaline Orb item,
  Mirror Herb (copy boosts once), Opportunist (copy boosts), and Eject Pack desire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

# Light dependency on team_state for the dataclass
from .team_state import PokemonState, STATS

def _to_id(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

# ------------------------- Stage change application ------------------------------

_PREVENT_DROP_ALL = {"clearbody","fullmetalbody","whitesmoke"}
_PREVENT_DROP_ATK = {"hypercutter","guarddog"}   # guard dog also boosts on attempt
_PREVENT_DROP_DEF = {"bigpecks"}
_PREVENT_DROP_ACC = {"keeneye"}
_REFLECT_DROPS = {"mirrorarmor"}

_SIMPLE = "simple"
_CONTRARY = "contrary"
_DEFIANT = "defiant"
_COMPETITIVE = "competitive"
_OPPORTUNIST = "opportunist"

def _apply_simple_contrary(ability: str, deltas: Dict[str, int]) -> Dict[str, int]:
    out = deltas.copy()
    if ability == _SIMPLE:
        for k in out: out[k] *= 2
    if ability == _CONTRARY:
        for k in out: out[k] *= -1
    return out

def _clamp_stage(x: int) -> int:
    return max(-6, min(6, x))

def apply_stage_change(
    target: PokemonState,
    deltas: Dict[str, int],
    *,
    source: Optional[PokemonState] = None,
    cause: str = "",
    dex: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """
    Apply stage deltas to target, honoring prevention/redirect and reactive abilities/items.

    Returns the *actually applied* deltas after prevention/reflection and Simple/Contrary.
    May mutate target/item/ability (e.g., White Herb, Mirror Herb) and source (Mirror Armor reflect).
    """
    ability_t = _to_id(target.ability or "")
    item_t = _to_id(target.item or "")
    ability_s = _to_id(source.ability) if source and source.ability else ""

    # 1) Try to prevent or reflect drops (only for negative deltas from opponent)
    attempted_negative = any(v < 0 for v in deltas.values())
    reflected = False
    prevented = False
    if attempted_negative and source and source is not target:
        # Clear Amulet blocks all opposing-lowered stats
        if item_t == "clearamulet":
            prevented = True
        # Ability-based prevention
        if ability_t in _PREVENT_DROP_ALL:
            prevented = True
        # Slot-specific prevention
        if deltas.get("atk", 0) < 0 and ability_t in _PREVENT_DROP_ATK:
            prevented = True
        if deltas.get("def", 0) < 0 and ability_t in _PREVENT_DROP_DEF:
            prevented = True
        if deltas.get("accuracy", 0) < 0 and ability_t in _PREVENT_DROP_ACC:
            prevented = True
        # Mirror Armor reflects
        if ability_t in _REFLECT_DROPS and not prevented:
            # reflect to source, nothing applied to target
            reflected = True
            prevented = True
            # Recursively apply to source as if from target
            if source:
                apply_stage_change(source, deltas, source=target, cause=cause, dex=dex)

        # Guard Dog: prevents Attack drop and raises Attack
        if ability_t == "guarddog" and deltas.get("atk", 0) < 0:
            prevented = True
            # boost Atk by 1
            d = {"atk": 1}
            _apply_stage_internal(target, _apply_simple_contrary(ability_t, d))

    # 2) If prevented (and not reflected), trigger reactions and bail
    if prevented and not reflected:
        # Defiant/Competitive trigger if something *attempted* to lower a stat but was prevented?
        # In-game these trigger when stats are *actually lowered*. We therefore DO NOT trigger them here.
        return {k: 0 for k in deltas}

    # 3) Apply Simple/Contrary
    deltas2 = _apply_simple_contrary(ability_t, deltas)

    # 4) Apply and clamp
    applied = _apply_stage_internal(target, deltas2)

    # 5) White Herb if any negative after application
    if any(v < 0 for v in applied.values()) and item_t == "whiteherb":
        # cures *all* lowered stats by 1 stage
        for k in list(target.stats.boosts.keys()):
            if target.stats.boosts[k] < 0:
                target.stats.boosts[k] = min(0, target.stats.boosts[k] + 1)
        # Mark consumption (caller should clear item if they want to simulate item loss)
        target.item = None  # consumed

    # 6) Reactive triggers: Defiant/Competitive, Adrenaline Orb, Opportunist/Mirror Herb
    lowered = any(v < 0 for v in applied.values())
    if lowered and source and source is not target:
        if ability_t == _DEFIANT:
            _apply_stage_internal(target, _apply_simple_contrary(ability_t, {"atk": 2}))
        if ability_t == _COMPETITIVE:
            _apply_stage_internal(target, _apply_simple_contrary(ability_t, {"spa": 2}))
        # Adrenaline Orb (item): +1 Speed when holder's stats are lowered by an opponent
        if item_t == "adrenalineorb":
            _apply_stage_internal(target, {"spe": 1})
            target.item = None  # consumed

    # Opportunist (copies an opponent's stat increases when they increase)
    if source and source is not target:
        ability_u = _to_id(source.ability or "")
        # If the *source* had positive deltas applied, target with Opportunist will copy them
        if ability_t == _OPPORTUNIST and any(v > 0 for v in deltas.values()):
            plus = {k: v for k, v in deltas.items() if v > 0}
            if plus:
                _apply_stage_internal(target, _apply_simple_contrary(ability_t, plus))
        # Mirror Herb (item) copies opponent's boosts once
        if item_t == "mirrorherb" and any(v > 0 for v in deltas.values()):
            plus = {k: v for k, v in deltas.items() if v > 0}
            if plus:
                _apply_stage_internal(target, plus)
                target.item = None  # consumed

    return applied

def _apply_stage_internal(target: PokemonState, deltas: Dict[str, int]) -> Dict[str, int]:
    # Ensure all boost keys exist
    boosts = target.stats.boosts or {}
    for k in STATS + ("accuracy","evasion"):
        boosts[k] = int(boosts.get(k, 0))

    # Apply and clamp
    for k, dv in deltas.items():
        if k not in boosts:
            continue
        boosts[k] = _clamp_stage(boosts[k] + int(dv))

    target.stats.boosts = boosts
    return deltas

# ------------------------- Switch-in triggers ------------------------------------

_ONE_TIME_ON_SWITCH = {"intrepidsword","dauntlessshield"}  # once per battle

def on_switch_in(user: PokemonState, opponents: List[PokemonState]) -> None:
    """Handle common entry abilities/items that affect stages immediately.

    Applies:
      - Intimidate: lowers foe Atk by 1, with Mirror Armor bounce, Guard Dog +1 Atk, prevention abilities.
      - Download: +1 Atk or SpA based on lower opposing defensive stat.
      - Intrepid Sword / Dauntless Shield: +1 Atk/Def (once per battle).
    """
    abil = _to_id(user.ability or "")
    # Intimidate
    if abil == "intimidate":
        for opp in opponents:
            apply_stage_change(opp, {"atk": -1}, source=user, cause="intimidate")

    # Download
    if abil == "download" and opponents:
        # Compare sum or first opponent's current raw Def/SpD (we use the first active foe)
        opp = opponents[0]
        defn = int(opp.stats.raw.get("def", 0))
        spd = int(opp.stats.raw.get("spd", 0))
        if defn >= spd:
            apply_stage_change(user, {"spa": 1}, source=user, cause="download")
        else:
            apply_stage_change(user, {"atk": 1}, source=user, cause="download")

    # Intrepid Sword / Dauntless Shield
    if abil in _ONE_TIME_ON_SWITCH:
        flag = f"used_{abil}_once"
        # We'll tuck one-time flags into the state via volatiles set for simplicity
        if flag not in user.volatiles:
            if abil == "intrepidsword":
                apply_stage_change(user, {"atk": 1}, source=user, cause=abil)
            elif abil == "dauntlessshield":
                apply_stage_change(user, {"def": 1}, source=user, cause=abil)
            user.volatiles.add(flag)

# ------------------------- Passive multiplicative modifiers ----------------------

def compute_passive_multipliers(state: PokemonState, field: Dict[str, Any]) -> Dict[str, float]:
    """Return per-stat multiplicative modifiers from abilities/items that are NOT stage-based.

    Examples:
        - Huge Power/Pure Power: ×2 Attack
        - Gorilla Tactics: ×1.5 Attack (with move-lock; we only return the multiplier)
        - Hustle: ×1.5 Attack (accuracy penalty is handled elsewhere)
        - Guts: ignore burn Attack halving (handled in damage code), and ×1.5 Attack when statused
        - Marvel Scale: ×1.5 Defense when statused
        - Fluffy/Ice Scales: category-based damage halving are better treated in damage layer
        - Orichalcum Pulse: Attack ×1.33 in sun (we include as 1.33 here; also sets Sun elsewhere)
        - Supreme Overlord: damage mod; handle in damage layer
    """
    abil = _to_id(state.ability or "")
    item = _to_id(state.item or "")
    status = (state.status or "").lower()
    weather = (field.get("weather") or "").lower()

    mults = {"atk": 1.0, "def": 1.0, "spa": 1.0, "spd": 1.0, "spe": 1.0}

    if abil in {"hugepower","purepower"}:
        mults["atk"] *= 2.0
    if abil == "gorillatactics":
        mults["atk"] *= 1.5
    if abil == "hustle":
        mults["atk"] *= 1.5
    if abil == "guts" and status in {"brn","psn","tox","par","slp"}:
        mults["atk"] *= 1.5
    if abil == "marvelscale" and status in {"brn","psn","tox","par","slp"}:
        mults["def"] *= 1.5
    if abil == "orichalcumpulse" and weather == "sun":
        mults["atk"] *= 4/3

    # Items already handled elsewhere for stats: Choice Band/Specs/Scarf, Assault Vest, Eviolite, etc.
    # We don't multiply here to avoid double-counting. Keep this for ability-only multipliers.

    return mults

# ------------------------- Groundedness ------------------------------------------

def augment_grounded(state: PokemonState, field: Dict[str, Any]) -> bool:
    """Infer `state.grounded` when possible.

    Grounded if:
      - Not Flying-type, not Levitate, not on Air Balloon
      - Or under Gravity, or affected by Smack Down/Thousand Arrows, or holding Iron Ball
      - Ingrain grounds the user; Magnet Rise/Telekinesis remove groundedness (ignored during Gravity)
    """
    types = {t for t in state.types if t}
    abil = _to_id(state.ability or "")
    item = _to_id(state.item or "")
    vols = {v for v in (state.volatiles or set())}
    gravity = bool(field.get("gravity", False))

    grounded = True
    if gravity:
        grounded = True
    else:
        if "flying" in types or abil == "levitate" or item == "airballoon" or "magnetrise" in vols or "telekinesis" in vols:
            grounded = False
        if "smackdown" in vols or "thousandarrows" in vols or item == "ironball" or "ingrain" in vols:
            grounded = True

    state.grounded = grounded
    return grounded


# ------------------------- Switch-out cleanup ------------------------------------

def on_switch_out(state: PokemonState) -> None:
    """Reset volatile stages and common flags that clear on leaving the field."""
    # Reset stat stages
    state.stats.boosts = {k: 0 for k in (list(STATS) + ["accuracy","evasion"])}
    # Clear most volatiles; keep long-term battle notes (like used_intrepidsword_once) if you want
    state.volatiles = {v for v in state.volatiles if v.startswith("used_")}
    # End effects like Unburden (caller should track activation); we conservatively clear any hint flags
    # Item stays whatever the server says; if it was consumed it should already be None/consumed_item set
    return None
