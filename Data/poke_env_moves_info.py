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

def _to_title(s: str) -> str:
    return s[:1] + s[1:].lower() if s else s

# Correct Showdown code mapping (defense-centric damageTaken): 0=neutral,1=weak(2x),2=resist(0.5x),3=immune(0x)
_SHOWDOWN_CODE_TO_MULT = {0:1.0, 1:2.0, 2:0.5, 3:0.0}
_ALT_SHOWDOWN_CODE_TO_MULT = {0:1.0, 1:0.5, 2:2.0, 3:0.0}  # fallback if first mapping clearly wrong

_DEF_CENTRIC_KEYS = {"DAMAGETAKEN","DAMAGE_TAKEN"}

# New: canonical expectations used to score candidate matrices
_EXPECTATIONS = [
    ('Fire','Grass',2.0),
    ('Fire','Water',0.5),
    ('Water','Fire',2.0),
    ('Dark','Fairy',0.5),
    ('Fairy','Dark',2.0),
    ('Fighting','Ghost',0.0),
    ('Ghost','Normal',0.0),
    ('Ground','Flying',0.0),
    ('Electric','Ground',0.0),
]

def _looks_like_attack_matrix(tc: dict) -> bool:
    """Heuristic: treat as attack->defense matrix if each row is a dict of numeric multipliers within plausible range."""
    if not tc: return False
    rows = list(tc.values())
    sample = None
    for r in rows:
        if isinstance(r, dict):
            sample = r; break
    if not isinstance(sample, dict):
        return False
    vals = list(sample.values())
    if not vals:
        return False
    # If any float not in {0,1,2,3} assume already multipliers
    if any(isinstance(v, (int,float)) and v not in (0,1,2,3) for v in vals):
        return True
    # If all rows contain any of the def-centric markers, not an attack matrix
    for r in rows:
        if isinstance(r, dict) and any(k.lower() == 'damagetaken' for k in r.keys()):
            return False
    # Ambiguous (all 0/1/2/3). Assume not attack matrix to avoid transposition errors.
    return False

def _normalize_showdown_typechart(tc):
    """Normalize Showdown type chart to attack->defense (Title-case keys).

    Handles both:
      * Defense-centric Showdown format: {DefType: {damageTaken: {AtkType: code}}}
      * Already attack-centric matrix: {AtkType: {DefType: mult(float)}}

    Previous buggy version misinterpreted orientation and swapped weakness/resist codes.
    """
    import logging
    log = logging.getLogger('typecalc')
    if not isinstance(tc, dict):
        return {}

    # Fast path: already attack matrix with float multipliers
    if _looks_like_attack_matrix(tc):
        out = {}
        for atk, row in tc.items():
            A = _norm_type_name(atk)
            if not A: continue
            At = _to_title(A)
            out.setdefault(At, {})
            if not isinstance(row, dict):
                continue
            for dfd, mult in row.items():
                D = _norm_type_name(dfd)
                if not D: continue
                Dt = _to_title(D)
                try:
                    out[At][Dt] = float(mult)
                except Exception:
                    pass
        return out

    # Defense-centric path
    atk_mat = { _to_title(a): {} for a in _CANON_TYPES }
    # We'll first collect raw codes per (atk,def) to allow adaptive mapping decision
    raw_codes = []  # list of (atk, def, code_int)
    # Also collect attack-centric codes if the input is actually attack->def matrix of codes
    raw_codes_attack_first = []  # list of (atk, def, code_int)
    for def_type, row in tc.items():
        D = _norm_type_name(def_type)
        if not D: continue
        if not isinstance(row, dict):
            continue
        # locate damageTaken structure or treat row as codes
        dmg_row = None
        for k, v in row.items():
            if isinstance(v, dict) and k.upper() in _DEF_CENTRIC_KEYS:
                dmg_row = v; break
        if dmg_row is None:
            # Maybe row itself is the damageTaken dict (codes). We don't yet know orientation.
            if all(isinstance(x, (int,float)) for x in row.values()):
                dmg_row = row
                # If this is actually attack-centric, then 'def_type' is attacker and keys are defender
                for maybe_def, code in row.items():
                    A = D  # attacker
                    DD = _norm_type_name(maybe_def)
                    if A in _CANON_TYPES and DD in _CANON_TYPES:
                        try: raw_codes_attack_first.append((A, DD, int(code)))
                        except Exception: raw_codes_attack_first.append((A, DD, 0))
        if not isinstance(dmg_row, dict):
            continue
        # Interpret as defense-centric map: keys are attacker types
        for atk_type, code in dmg_row.items():
            A = _norm_type_name(atk_type)
            if not A: continue
            if A not in _CANON_TYPES or D not in _CANON_TYPES:
                continue
            try:
                code_int = int(code)
            except Exception:
                code_int = 0
            raw_codes.append((A, D, code_int))
    # Build matrices for both code maps and both orientations, pick best by expectations
    def build_matrix(codes, code_map):
        m = { _to_title(a): {} for a in _CANON_TYPES }
        for A,D,ci in codes:
            if A in _CANON_TYPES and D in _CANON_TYPES:
                m[_to_title(A)][_to_title(D)] = code_map.get(ci, 1.0)
        return m
    cand_def_c1 = build_matrix(raw_codes, _SHOWDOWN_CODE_TO_MULT)
    cand_def_c2 = build_matrix(raw_codes, _ALT_SHOWDOWN_CODE_TO_MULT)
    cand_atk_c1 = build_matrix(raw_codes_attack_first, _SHOWDOWN_CODE_TO_MULT)
    cand_atk_c2 = build_matrix(raw_codes_attack_first, _ALT_SHOWDOWN_CODE_TO_MULT)

    def score(mat):
        sc = 0
        for a,d,exp in _EXPECTATIONS:
            got = mat.get(a, {}).get(d)
            if got == exp: sc += 1
        return sc
    candidates = [cand_def_c1, cand_def_c2, cand_atk_c1, cand_atk_c2]
    scores = [score(c) for c in candidates]
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    atk_mat = candidates[best_idx]
    # Drop empty rows
    out = { atk: row for atk, row in atk_mat.items() if any(v != 1.0 for v in row.values()) }
    if not out:
        out = atk_mat
    # Sanity and final clamp
    try:
        ok = sum(1 for a,d,exp in _EXPECTATIONS if out.get(a, {}).get(d) == exp)
        if ok < len(_EXPECTATIONS) - 1:
            import logging; logging.getLogger('typecalc').warning('Type chart mapping failed (%d/%d). Falling back to static canonical chart.', ok, len(_EXPECTATIONS))
            return _build_static_chart()
    except Exception:
        pass
    return out
# --- END UI TYPE-CHART NORMALIZER PATCH ---

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# Added by patch_think_showdown.py
import os
from Data.think_helpers import normalize_accuracy

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
            accuracy=normalize_accuracy(m.get('accuracy')),
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
        """Return a normalized, Title-case type chart with caching.
        By default we now use the canonical static Gen 9 chart to avoid mis-parsing
        dynamic sources that can flip weaknesses/resistances. To opt-in to dynamic
        parsing (Showdown/poke-env), set POKECHAD_ENABLE_DYNAMIC_TYPECHART=1.
        """
        try:
            import os, logging
            # Prefer static unless explicitly enabled
            if not os.getenv('POKECHAD_ENABLE_DYNAMIC_TYPECHART'):
                tc = _build_static_chart()
                setattr(self, '_type_chart_cache', tc)
                return tc

            tc = getattr(self, "_type_chart_cache", None)
            if tc is None:
                raw_tc = (
                    getattr(getattr(self, "_data", None), "type_chart", None)
                    or getattr(self, "type_chart", None)
                    or (self._ps_dex.get("typechart") if getattr(self, "_ps_dex", None) else {})
                    or {}
                )
                tc = _normalize_showdown_typechart(raw_tc)
                # Validate expectations (extended)
                ok = sum(1 for a,d,exp in _EXPECTATIONS if tc.get(a, {}).get(d) == exp)
                if ok < len(_EXPECTATIONS):
                    logging.getLogger('typecalc').warning('Dynamic type chart failed validation (%d/%d). Using static canonical chart.', ok, len(_EXPECTATIONS))
                    tc = _build_static_chart()
                setattr(self, "_type_chart_cache", tc)
            else:
                if tc and all(k.isupper() for k in tc.keys()):
                    converted = {}
                    for atk, row in tc.items():
                        converted[_to_title(atk)] = { _to_title(dfd): mult for dfd, mult in row.items() }
                    tc = converted
                    setattr(self, "_type_chart_cache", tc)
            return tc
        except Exception:
            return _build_static_chart()

_STATIC_CHART = None

def _build_static_chart():
    global _STATIC_CHART
    if _STATIC_CHART is not None:
        return _STATIC_CHART
    types = ["Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"]
    n=1.0; h=0.5; s=2.0; z=0.0
    eff = {t:{u:n for u in types} for t in types}
    eff["Normal"]["Rock"]=h; eff["Normal"]["Ghost"]=z; eff["Normal"]["Steel"]=h
    for u in ["Fire","Water","Rock","Dragon"]: eff["Fire"][u]=h
    for u in ["Grass","Ice","Bug","Steel"]: eff["Fire"][u]=s
    for u in ["Fire","Ground","Rock"]: eff["Water"][u]=s
    for u in ["Water","Grass","Dragon"]: eff["Water"][u]=h
    for u in ["Water","Flying"]: eff["Electric"][u]=s
    for u in ["Electric","Grass","Dragon"]: eff["Electric"][u]=h
    eff["Electric"]["Ground"]=z
    for u in ["Water","Ground","Rock"]: eff["Grass"][u]=s
    for u in ["Fire","Grass","Poison","Flying","Bug","Dragon","Steel"]: eff["Grass"][u]=h
    for u in ["Grass","Ground","Flying","Dragon"]: eff["Ice"][u]=s
    for u in ["Fire","Water","Ice","Steel"]: eff["Ice"][u]=h
    for u in ["Normal","Ice","Rock","Dark","Steel"]: eff["Fighting"][u]=s
    for u in ["Poison","Flying","Psychic","Bug","Fairy"]: eff["Fighting"][u]=h
    eff["Fighting"]["Ghost"]=z
    eff["Poison"]["Grass"]=s
    for u in ["Poison","Ground","Rock","Ghost"]: eff["Poison"][u]=h
    eff["Poison"]["Steel"]=z
    eff["Poison"]["Fairy"]=s
    for u in ["Fire","Electric","Poison","Rock","Steel"]: eff["Ground"][u]=s
    for u in ["Grass","Bug"]: eff["Ground"][u]=h
    eff["Ground"]["Flying"]=z
    for u in ["Grass","Fighting","Bug"]: eff["Flying"][u]=s
    for u in ["Electric","Rock","Steel"]: eff["Flying"][u]=h
    for u in ["Fighting","Poison"]: eff["Psychic"][u]=s
    for u in ["Psychic","Steel"]: eff["Psychic"][u]=h
    eff["Psychic"]["Dark"]=z
    for u in ["Grass","Psychic","Dark"]: eff["Bug"][u]=s
    for u in ["Fire","Fighting","Poison","Flying","Ghost","Steel","Fairy"]: eff["Bug"][u]=h
    for u in ["Fire","Ice","Flying","Bug"]: eff["Rock"][u]=s
    for u in ["Fighting","Ground","Steel"]: eff["Rock"][u]=h
    eff["Ghost"]["Ghost"]=s; eff["Ghost"]["Psychic"]=s
    eff["Ghost"]["Dark"]=h; eff["Ghost"]["Normal"]=z
    eff["Dragon"]["Dragon"]=s; eff["Dragon"]["Steel"]=h; eff["Dragon"]["Fairy"]=z
    eff["Dark"]["Psychic"]=s; eff["Dark"]["Ghost"]=s
    for u in ["Fighting","Dark","Fairy"]: eff["Dark"][u]=h
    for u in ["Rock","Ice","Fairy"]: eff["Steel"][u]=s
    for u in ["Fire","Water","Electric","Steel"]: eff["Steel"][u]=h
    for u in ["Fighting","Dragon","Dark"]: eff["Fairy"][u]=s
    for u in ["Fire","Poison","Steel"]: eff["Fairy"][u]=h
    _STATIC_CHART = eff
    return eff
