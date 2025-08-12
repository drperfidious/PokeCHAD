"""
Pokemon moves information and data management module
---------------------------------------------------
Lightweight wrapper around poke_env.data.GenData for moves and type chart access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

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

    @property
    def gen(self) -> int:
        return getattr(self._data, "gen", 9)

    def exists(self, name_or_id: str) -> bool:
        if not self._data:
            return False
        return to_id_str(name_or_id) in self._data.moves

    def all_ids(self) -> List[str]:
        if not self._data:
            return []
        return list(self._data.moves.keys())

    def raw(self, name_or_id: str) -> Dict[str, Any]:
        if not self._data:
            raise RuntimeError("poke_env not available")
        mid = to_id_str(name_or_id)
        m = self._data.moves.get(mid)
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
            base_power=m.get("basePower"),
            accuracy=m.get("accuracy"),
            priority=m.get("priority", 0),
            target=m.get("target"),
            pp=m.get("pp"),
            flags=m.get("flags", {}),
            secondary=m.get("secondary"),
            secondaries=m.get("secondaries"),
            status=m.get("status"),
            volatile_status=m.get("volatileStatus"),
            boosts=m.get("boosts"),
            multihit=m.get("multihit"),
            drain=m.get("drain"),
            recoil=m.get("recoil"),
            raw=m,
        )

    def get_type_chart(self) -> Dict[str, Dict[str, float]]:
        if not self._data:
            return {}
        return self._data.type_chart
