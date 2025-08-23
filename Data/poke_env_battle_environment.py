"""
Pokemon environment battle integration module
---------------------------------------------
This wraps a poke-env `Battle` object into a compact, serializable snapshot and a
`FieldState` suitable for the damage calculator.

We purposely avoid importing poke-env at module import time. The functions will
use duck-typing against a `Battle` object at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# We mirror the calculator's light FieldState to avoid circular imports.
@dataclass
class FieldState:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    gravity: bool = False
    trick_room: bool = False
    is_doubles: bool = False
    targets_on_target_side: int = 1
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False


def _safe_lower(x: Any) -> Optional[str]:
    return x.lower() if isinstance(x, str) else None


def _extract_side_conditions(side) -> Dict[str, Any]:
    """Best-effort extraction of hazards/screens from a poke-env Side object."""
    out = {
        "stealth_rock": False,
        "spikes": 0,
        "toxic_spikes": 0,
        "sticky_web": False,
        "reflect": False,
        "light_screen": False,
        "aurora_veil": False,
        "tailwind": False,
        "safeguard": False,
        "mist": False,
    }
    if side is None:
        return out

    # poke-env represents side conditions via enums/sets; we use getattr defensively.
    # Handle case where side is already a dict (newer poke-env versions)
    if isinstance(side, dict):
        sconds = side
    else:
        sconds = getattr(side, "conditions", None) or getattr(side, "side_conditions", None)
    
    if isinstance(sconds, dict):
        # Newer poke-env stores counts for stackables
        # Handle both uppercase enum keys and lowercase underscore keys
        out["stealth_rock"] = bool(sconds.get("STEALTH_ROCK") or sconds.get("Stealth Rock") or sconds.get("stealth_rock"))
        out["spikes"] = int(sconds.get("SPIKES") or sconds.get("Spikes") or sconds.get("spikes") or 0)
        out["toxic_spikes"] = int(sconds.get("TOXIC_SPIKES") or sconds.get("Toxic Spikes") or sconds.get("toxic_spikes") or 0)
        out["sticky_web"] = bool(sconds.get("STICKY_WEB") or sconds.get("Sticky Web") or sconds.get("sticky_web"))
        out["reflect"] = bool(sconds.get("REFLECT") or sconds.get("Reflect") or sconds.get("reflect"))
        out["light_screen"] = bool(sconds.get("LIGHT_SCREEN") or sconds.get("Light Screen") or sconds.get("light_screen"))
        out["aurora_veil"] = bool(sconds.get("AURORA_VEIL") or sconds.get("Aurora Veil") or sconds.get("aurora_veil"))
        out["tailwind"] = bool(sconds.get("TAILWIND") or sconds.get("Tailwind") or sconds.get("tailwind"))
        out["safeguard"] = bool(sconds.get("SAFEGUARD") or sconds.get("Safeguard") or sconds.get("safeguard"))
        out["mist"] = bool(sconds.get("MIST") or sconds.get("Mist") or sconds.get("mist"))
    else:
        # Very old poke-env
        for name in ("Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web",
                     "Reflect", "Light Screen", "Aurora Veil", "Tailwind", "Safeguard", "Mist"):
            val = getattr(side, name.replace(" ", "_").lower(), None)
            if isinstance(val, bool):
                out[name.replace(" ", "_").lower()] = val
    return out


def to_field_state(battle) -> FieldState:
    """Convert a poke-env Battle to our FieldState."""
    # Weather / terrain are enums or strings; coerce to lowercase keys
    w = getattr(battle, "weather", None)
    weather = getattr(w, "name", w)
    t = getattr(battle, "terrain", None)
    terrain = getattr(t, "name", t)

    # Side conditions
    my_side = getattr(battle, "side_conditions", None) or getattr(battle, "side", None)
    opp_side = getattr(battle, "opponent_side_conditions", None) or getattr(battle, "opponent_side", None)

    my = _extract_side_conditions(my_side)
    opp = _extract_side_conditions(opp_side)

    # Active target count for doubles (rough estimate: 2 if both active; else 1)
    is_doubles = bool(getattr(battle, "is_doubles", False) or getattr(battle, "double_battle", False))
    targets_on_target_side = 2 if (is_doubles and len(getattr(battle, "opponent_active_pokemon", []) or []) == 2) else 1

    # Screens are on defender's side; we return defender=opp by default for convenience
    return FieldState(
        weather=_safe_lower(weather),
        terrain=_safe_lower(terrain),
        trick_room=bool(getattr(battle, "trick_room", False)),
        gravity=bool(getattr(battle, "gravity", False)),
        is_doubles=is_doubles,
        targets_on_target_side=targets_on_target_side,
        reflect=opp["reflect"],
        light_screen=opp["light_screen"],
        aurora_veil=opp["aurora_veil"],
    )


def snapshot(battle) -> Dict[str, Any]:
    """Dump a serializable, RL-friendly snapshot of a poke-env Battle.

    This intentionally includes both IDs and human-readable names for stability.
    """
    s = {
        "battle_tag": getattr(battle, "battle_tag", ""),
        "format": getattr(battle, "format", None),
        "turn": getattr(battle, "turn", None),
        "weather": _safe_lower(getattr(getattr(battle, "weather", None), "name", None)),
        "terrain": _safe_lower(getattr(getattr(battle, "terrain", None), "name", None)),
        "trick_room": bool(getattr(battle, "trick_room", False)),
        "gravity": bool(getattr(battle, "gravity", False)),
        "is_doubles": bool(getattr(battle, "is_doubles", False) or getattr(battle, "double_battle", False)),
        "my_team": {},
        "opp_team": {},
        "active_moves_ids": [],
        "active_switch_ids": [],
        "force_switch": bool(getattr(battle, "force_switch", False)),
        "can_tera": bool(getattr(battle, "can_tera", False)),
        "side_conditions": _extract_side_conditions(getattr(battle, "side_conditions", None) or getattr(battle, "side", None)),
        "opp_side_conditions": _extract_side_conditions(getattr(battle, "opponent_side_conditions", None) or getattr(battle, "opponent_side", None)),
    }

    # Determine active references for quick flagging
    try:
        _active_me = getattr(battle, "active_pokemon", None)
    except Exception:
        _active_me = None
    try:
        _active_opp = getattr(battle, "opponent_active_pokemon", None)
    except Exception:
        _active_opp = None

    # Team snapshots
    team = getattr(battle, "team", {}) or {}
    for sid, p in team.items():
        s["my_team"][sid] = {
            "species": getattr(p, "species", None),
            "types": getattr(p, "types", None),
            "level": getattr(p, "level", None),
            "hp_fraction": (getattr(p, "current_hp_fraction", None) or getattr(p, "hp_fraction", None)),
            "status": getattr(getattr(p, "status", None), "name", None),
            "boosts": getattr(p, "boosts", None),
            "item": getattr(p, "item", None),
            "ability": getattr(p, "ability", None),
            "revealed_moves": [getattr(m, "id", None) for m in (getattr(p, "moves", None) or {}).values()],
            "is_active": bool(p is _active_me),
        }

    opp_team = getattr(battle, "opponent_team", {}) or {}
    for sid, p in opp_team.items():
        s["opp_team"][sid] = {
            "species": getattr(p, "species", None),
            "types": getattr(p, "types", None),
            "level": getattr(p, "level", None),
            "hp_fraction": (getattr(p, "current_hp_fraction", None) or getattr(p, "hp_fraction", None)),
            "status": getattr(getattr(p, "status", None), "name", None),
            "boosts": getattr(p, "boosts", None),
            "item": getattr(p, "item", None),
            "ability": getattr(p, "ability", None),
            "revealed_moves": [getattr(m, "id", None) for m in (getattr(p, "moves", None) or {}).values()],
            "is_active": bool(p is _active_opp),
        }

    # Actions available
    try:
        s["active_moves_ids"] = [getattr(m, "id", None) for m in (getattr(battle, "available_moves", []) or [])]
        s["active_switch_ids"] = [getattr(p, "species", None) for p in (getattr(battle, "available_switches", []) or [])]
    except Exception:
        pass

    return s
