"""
Pokemon moves information and data management module
---------------------------------------------------
Lightweight wrapper around poke_env.data.GenData for moves and type chart access.
"""

from __future__ import annotations

# --- BEGIN UI TYPE-CHART NORMALIZER PATCH ---
_UI_TC_NORMALIZER_INSTALLED = True

_CANON_TYPES = ["NORMAL","FIRE","WATER","ELECTRIC","GRASS","ICE","FIGHTING","POISON","GROUND","FLYING","PSYCHIC","BUG","ROCK","GHOST","DRAGON","DARK","STEEL","FAIRY","STELLAR"]

def _norm_type_name(t):
    if not t: return None
    s = str(t).strip().upper()
    if s == "": return None
    return s

def _normalize_showdown_typechart(tc):
    """Accepts Showdown's {Type: {damageTaken:{...}}} or a flat multiplier table.
    Returns {ATTACKING_TYPE: {DEFENDING_TYPE: mult(float)}} with UPPERCASE keys.
    """
    if not isinstance(tc, dict):
        return {}
    out = {T:{} for T in _CANON_TYPES}
    for atk, row in tc.items():
        A = _norm_type_name(atk)
        if not A: 
            continue
        if isinstance(row, dict) and ("damageTaken" in row or "damage_taken" in row):
            taken = row.get("damageTaken") or row.get("damage_taken") or {}
            for dfd, code in taken.items():
                D = _norm_type_name(dfd)
                if not D: 
                    continue
                mult = 1.0
                try:
                    code = int(code)
                except Exception:
                    code = None
                if code == 1: mult = 0.5
                elif code == 2: mult = 2.0
                elif code == 3: mult = 0.0
                out[A][D] = float(mult)
        else:
            for dfd, mult in (row or {}).items():
                D = _norm_type_name(dfd)
                if not D:
                    continue
                try:
                    out[A][D] = float(mult)
                except Exception:
                    pass
    return out
# --- END UI TYPE-CHART NORMALIZER PATCH ---

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# Added by patch_think_showdown.py
import os
try:
    from tools.Data.showdown.ps_data_loader import load_showdown_dir  # noqa: F401
except Exception:
    load_showdown_dir = None

try:
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
except Exception:  # pragma: no cover - allow import without poke-env
    GenData = None
    def to_id_str(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())


@dataclass
class MoveInfo:
    id: str
    name: str
    type: Optional[str] = None
    category: Optional[str] = None
    base_power: Optional[int] = None
    accuracy: Optional[Union[int, float, bool]] = None
    priority: int = 0
    target: Optional[str] = None
    pp: Optional[int] = None
    flags: Dict[str, bool] = field(default_factory=dict)
    secondary: Optional[Dict[str, Any]] = None
    secondaries: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    volatile_status: Optional[str] = None
    boosts: Optional[Dict[str, int]] = None
    multihit: Optional[Union[int, List[int]]] = None
    drain: Optional[List[int]] = None
    recoil: Optional[List[int]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def makes_contact(self) -> bool:
        return bool(self.flags.get("contact"))


class MovesInfo:
    def __init__(self, gen_or_format: Union[int, str] = 9):
        if isinstance(gen_or_format, int):
            self._data = GenData.from_gen(gen_or_format) if GenData else None
        else:
            self._data = GenData.from_format(gen_or_format) if GenData else None
        # Load Showdown dex as a secondary source (items/abilities/moves/typechart)
        self._ps_dex: Dict[str, Dict[str, Any]] = {}
        try:
            if load_showdown_dir:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sd_dir = os.path.join(repo_root, "tools", "Data", "showdown")
                if os.path.isdir(sd_dir):
                    self._ps_dex = load_showdown_dir(sd_dir)
        except Exception:
            self._ps_dex = {}

    def gen(self) -> int:
        return getattr(self._data, "gen", 9)

    def exists(self, name_or_id: str) -> bool:
        mid = to_id_str(name_or_id)
        in_pokeenv = bool(getattr(self, "_data", None)) and (mid in getattr(self._data, "moves", {}))
        in_ps = bool(self._ps_dex) and (mid in self._ps_dex.get("moves", {}))
        return in_pokeenv or in_ps

    def all_ids(self) -> List[str]:
        if not self._data:
            return []
        return list(self._data.moves.keys())

    def raw(self, name_or_id: str) -> Dict[str, Any]:
        mid = to_id_str(name_or_id)
        m = None
        try:
            if getattr(self, "_data", None):
                m = self._data.moves.get(mid)
        except Exception:
            m = None
        if m is None and self._ps_dex:
            m = self._ps_dex.get("moves", {}).get(mid)
        if m is None:
            raise KeyError(f"Unknown move: {name_or_id} (normalized: {mid})")
        return m

    def get(self, name_or_id: str) -> MoveInfo:
        m = self.raw(name_or_id)
        mid = to_id_str(name_or_id)
        return MoveInfo(
            id=mid,
            name=m.get("name", mid),
            type=m.get("type"),
            category=m.get("category"),
            base_power=m.get("basePower") if "basePower" in m else m.get("base_power"),
            accuracy=m.get("accuracy"),
            priority=m.get("priority", 0),
            target=m.get("target"),
            pp=m.get("pp"),
            flags=m.get("flags", {}),
            secondary=m.get("secondary"),
            secondaries=m.get("secondaries"),
            status=m.get("status"),
            volatile_status=m.get("volatileStatus") or m.get("volatile_status"),
            boosts=m.get("boosts"),
            multihit=m.get("multihit"),
            drain=m.get("drain"),
            recoil=m.get("recoil"),
            raw=m,
        )

    def get_type_chart(self) -> Dict[str, Dict[str, float]]:
        """Return a normalized, uppercase type chart with caching (falls back to Showdown)."""
        try:
            tc = getattr(self, "_type_chart_cache", None)
            if tc is None:
                raw_tc = (
                    getattr(getattr(self, "_data", None), "type_chart", None)
                    or getattr(self, "type_chart", None)
                    or (self._ps_dex.get("typechart") if getattr(self, "_ps_dex", None) else {})
                    or {}
                )
                tc = _normalize_showdown_typechart(raw_tc)
                setattr(self, "_type_chart_cache", tc)
            return tc
        except Exception:
            return {}
