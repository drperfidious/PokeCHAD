
# team_state.py
"""
Structured state objects for ally and opponent Pokémon in poke-env battles.

See earlier docstring in the first version for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

try:
    # Core poke-env imports
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
    from poke_env.stats import compute_raw_stats
except Exception:  # pragma: no cover - allow type checking without poke-env
    GenData = Any  # type: ignore
    def to_id_str(x: str) -> str:  # type: ignore
        return (x or '').lower().replace(' ', '').replace('-', '')
    def compute_raw_stats(species: str, evs: List[int], ivs: List[int], level: int, nature: str, data: Any) -> List[int]:  # type: ignore
        raise RuntimeError("poke-env not installed: compute_raw_stats unavailable")

STATS = ("hp","atk","def","spa","spd","spe")
DEFAULT_IVS = [31,31,31,31,31,31]

def _species_id(name: str) -> str:
    return to_id_str(name)

def _get_species_entry(gendata: GenData, species: str) -> Dict[str, Any]:
    sid = _species_id(species)
    return getattr(gendata, "pokedex", {}).get(sid, {})

def _species_base_stats(gendata: GenData, species: str) -> Dict[str, int]:
    entry = _get_species_entry(gendata, species)
    bs = entry.get("baseStats", {})
    if isinstance(bs, dict) and all(k in bs for k in STATS):
        return {k: int(bs[k]) for k in STATS}
    return {k: 0 for k in STATS}

def _species_types(gendata: GenData, species: str) -> Tuple[Optional[str], Optional[str]]:
    entry = _get_species_entry(gendata, species)
    types = entry.get("types", [])
    t1 = to_id_str(types[0]) if len(types) >= 1 else None
    t2 = to_id_str(types[1]) if len(types) >= 2 else None
    return (t1, t2)

def _guess_level_from_format(battle_format: Optional[str]) -> int:
    fmt = (battle_format or "").lower()
    if "vgc" in fmt or "doubles" in fmt:
        return 50
    return 100

def _coerce_evs(evs: Optional[Sequence[int] or Dict[str,int]]) -> List[int]:
    if isinstance(evs, dict):
        return [int(evs.get(k, 0)) for k in STATS]
    if isinstance(evs, (list, tuple)) and len(evs) == 6:
        return [int(x) for x in evs]
    return [0,0,0,0,0,0]

def _coerce_ivs(ivs: Optional[Sequence[int] or Dict[str,int]]) -> List[int]:
    if isinstance(ivs, dict):
        return [int(ivs.get(k, 31)) for k in STATS]
    if isinstance(ivs, (list, tuple)) and len(ivs) == 6:
        return [int(x) for x in ivs]
    return DEFAULT_IVS[:]

@dataclass
class MoveSlot:
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    base_power: Optional[int] = None
    accuracy: Optional[float] = None
    pp: Optional[int] = None
    max_pp: Optional[int] = None
    priority: Optional[int] = None
    target: Optional[str] = None

    @staticmethod
    def from_pokenv_move(move_obj: Any) -> "MoveSlot":
        def pick(obj: Any, *names: str, default=None):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return default
        mid = pick(move_obj, "id", "move_id", default=None)
        if mid is None:
            mid = to_id_str(str(pick(move_obj, "name", default="unknown")))
        return MoveSlot(
            id=str(mid),
            name=pick(move_obj, "name", default=None),
            type=to_id_str(pick(move_obj, "type", default="")) or None,
            category=to_id_str(pick(move_obj, "category", default="")) or None,
            base_power=int(pick(move_obj, "base_power", "basePower", default=0)) or None,
            accuracy=float(pick(move_obj, "accuracy", default=100.0)) if pick(move_obj, "accuracy", default=None) is not None else None,
            pp=int(pick(move_obj, "pp", default=0)) or None,
            max_pp=int(pick(move_obj, "max_pp", "maxpp", default=0)) or None,
            priority=int(pick(move_obj, "priority", default=0)) or None,
            target=pick(move_obj, "target", default=None),
        )

@dataclass
class ResolvedStats:
    base: Dict[str, int]
    evs: Dict[str, int]
    ivs: Dict[str, int]
    nature: Optional[str]
    level: int
    raw: Dict[str, int]
    boosts: Dict[str, int] = field(default_factory=dict)
    @property
    def hp(self) -> Optional[int]: return self.raw.get("hp")
    @property
    def atk(self) -> Optional[int]: return self.raw.get("atk")
    @property
    def def_(self) -> Optional[int]: return self.raw.get("def")
    @property
    def spa(self) -> Optional[int]: return self.raw.get("spa")
    @property
    def spd(self) -> Optional[int]: return self.raw.get("spd")
    @property
    def spe(self) -> Optional[int]: return self.raw.get("spe")

@dataclass
class PokemonState:
    species: str
    nickname: Optional[str]
    types: Tuple[Optional[str], Optional[str]]
    ability: Optional[str]
    ability_options: Optional[List[str]] = None
    item: Optional[str] = None
    consumed_item: Optional[str] = None
    tera_type: Optional[str] = None
    max_hp: Optional[int] = None
    current_hp: Optional[int] = None
    hp_fraction: Optional[float] = None
    status: Optional[str] = None
    volatiles: Set[str] = field(default_factory=set)
    is_active: bool = False
    is_trapped: Optional[bool] = None
    has_priority_block: bool = False
    grounded: Optional[bool] = None
    stats: ResolvedStats = field(default_factory=lambda: ResolvedStats(base={k:0 for k in STATS}, evs={k:0 for k in STATS}, ivs={k:31 for k in STATS}, nature=None, level=100, raw={k:0 for k in STATS}))
    moves: List[MoveSlot] = field(default_factory=list)
    stat_history: List[dict] = field(default_factory=list)

    @property
    def hp(self) -> Optional[int]: return self.stats.hp if self.stats else None
    @property
    def atk(self) -> Optional[int]: return self.stats.atk if self.stats else None
    @property
    def def_(self) -> Optional[int]: return self.stats.def_ if self.stats else None
    @property
    def spa(self) -> Optional[int]: return self.stats.spa if self.stats else None
    @property
    def spd(self) -> Optional[int]: return self.stats.spd if self.stats else None
    @property
    def spe(self) -> Optional[int]: return self.stats.spe if self.stats else None

    @staticmethod
    def _safe_bool(x: Any) -> bool:
        return bool(x) if x is not None else False

    @classmethod
    def from_ally(cls, battle: Any, mon: Any, gen_data: GenData, team_hint: Optional[Dict[str, Any]] = None) -> "PokemonState":
        species = getattr(mon, "species", None) or getattr(mon, "name", "unknown")
        nickname = getattr(mon, "nickname", None) or getattr(mon, "name", None)
        types = _species_types(gen_data, species)
        tera_type = to_id_str(getattr(mon, "tera_type", None) or getattr(mon, "terastallized_type", None) or "") or None

        ability = to_id_str(getattr(mon, "ability", None) or "")
        item = to_id_str(getattr(mon, "item", None) or "") or None

        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None) or getattr(mon, "hp", None)
        hp_fraction = getattr(mon, "current_hp_fraction", None) or getattr(mon, "hp_fraction", None)
        status = to_id_str(getattr(mon, "status", None) or "") or None

        vol_raw = set()
        for attr in ("effects","volatile_statuses","volatiles","_effects"):
            v = getattr(mon, attr, None)
            if isinstance(v, dict):
                vol_raw.update(to_id_str(k) for k in v.keys())
            elif isinstance(v, (set,list,tuple)):
                vol_raw.update(to_id_str(x) for x in v)
        is_trapped = getattr(mon, "trapped", None)
        if is_trapped is None:
            trap_ids = {"trapped","partialtrapped","ingrain","jawlock","bind","clamp","whirlpool","firespin","sandtomb","octolock","snaptrap"}
            is_trapped = any(t in vol_raw for t in trap_ids)

        moves: List[MoveSlot] = []
        pkm_moves = getattr(mon, "moves", None)
        if isinstance(pkm_moves, dict):
            for mv in pkm_moves.values():
                try: moves.append(MoveSlot.from_pokenv_move(mv))
                except Exception: pass
        elif isinstance(pkm_moves, (list, tuple)):
            for mv in pkm_moves:
                try: moves.append(MoveSlot.from_pokenv_move(mv))
                except Exception: pass
        if len(moves) > 4: moves = moves[:4]

        boosts = {}
        raw_boosts = getattr(mon, "boosts", None)
        if isinstance(raw_boosts, dict):
            for k in STATS:
                if k in raw_boosts: boosts[k] = int(raw_boosts[k])

        evs = _coerce_evs(getattr(mon, "evs", None))
        ivs = _coerce_ivs(getattr(mon, "ivs", None))
        nature = to_id_str(getattr(mon, "nature", None) or "") or None
        level = int(getattr(mon, "level", 0) or 0)
        if (sum(evs) == 0 or sum(ivs) == 0 or not nature or level <= 0) and team_hint:
            evs = _coerce_evs(team_hint.get("evs"))
            ivs = _coerce_ivs(team_hint.get("ivs"))
            nature = to_id_str(team_hint.get("nature") or "") or nature
            level = int(team_hint.get("level") or 0) or level
        if level <= 0:
            level = _guess_level_from_format(getattr(battle, "battle_format", None) or getattr(battle, "format", None))
        if not nature:
            nature = "serious"

        base = _species_base_stats(gen_data, species)
        raw_stats_arr = compute_raw_stats(species=species, evs=evs, ivs=ivs, level=level, nature=nature, data=gen_data)
        raw = {k: int(raw_stats_arr[i]) for i, k in enumerate(STATS)}

        if max_hp is None: max_hp = raw["hp"]
        if current_hp is None and hp_fraction is not None and max_hp:
            current_hp = int(round(max_hp * float(hp_fraction)))
        # ## PATCH: default hp when unknown
        if current_hp is None and max_hp is not None:
            if (status or '').lower() == 'fnt':
                current_hp = 0
                hp_fraction = 0.0
            else:
                current_hp = int(max_hp)
                hp_fraction = 1.0

        stats = ResolvedStats(
            base=base,
            evs={k: evs[i] for i, k in enumerate(STATS)},
            ivs={k: ivs[i] for i, k in enumerate(STATS)},
            nature=nature, level=level, raw=raw, boosts=boosts,
        )

        is_active = False
        try: is_active = (mon is getattr(battle, "active_pokemon", None))
        except Exception: is_active = False

        return cls(
            species=str(species), nickname=nickname, types=types, ability=ability or None,
            item=item, tera_type=tera_type, max_hp=max_hp, current_hp=current_hp, hp_fraction=hp_fraction,
            status=status, volatiles=set(vol_raw), is_active=is_active,
            is_trapped=bool(is_trapped) if is_trapped is not None else None,
            has_priority_block=(ability in {'dazzling','queenlymajesty','armortail'}),
            grounded=None, stats=stats, moves=moves
        )

    @classmethod
    def from_opponent(cls, battle: Any, mon: Any, gen_data: GenData, ev_policy: str = "auto", default_level: Optional[int] = None) -> "PokemonState":
        species = getattr(mon, "species", None) or getattr(mon, "name", "unknown")
        nickname = getattr(mon, "nickname", None) or getattr(mon, "name", None)
        types = _species_types(gen_data, species)
        tera_type = to_id_str(getattr(mon, "tera_type", None) or getattr(mon, "terastallized_type", None) or "") or None

        ability = to_id_str(getattr(mon, "ability", None) or "") or None
        item = to_id_str(getattr(mon, "item", None) or "") or None

        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None) or getattr(mon, "hp", None)
        hp_fraction = getattr(mon, "current_hp_fraction", None) or getattr(mon, "hp_fraction", None)
        status = to_id_str(getattr(mon, "status", None) or "") or None

        vol_raw = set()
        for attr in ("effects","volatile_statuses","volatiles","_effects"):
            v = getattr(mon, attr, None)
            if isinstance(v, dict):
                vol_raw.update(to_id_str(k) for k in v.keys())
            elif isinstance(v, (set,list,tuple)):
                vol_raw.update(to_id_str(x) for x in v)
        is_trapped = getattr(mon, "trapped", None)
        if is_trapped is None:
            trap_ids = {"trapped","partialtrapped","ingrain","jawlock","bind","clamp","whirlpool","firespin","sandtomb","octolock","snaptrap"}
            is_trapped = any(t in vol_raw for t in trap_ids)

        boosts = {}
        raw_boosts = getattr(mon, "boosts", None)
        if isinstance(raw_boosts, dict):
            for k in STATS:
                if k in raw_boosts: boosts[k] = int(raw_boosts[k])

        level = int(getattr(mon, "level", 0) or 0)
        if level <= 0:
            level = int(default_level or _guess_level_from_format(getattr(battle, "battle_format", None) or getattr(battle, "format", None)))

        base = _species_base_stats(gen_data, species)
        main_offense = "atk" if base.get("atk", 0) >= base.get("spa", 0) else "spa"
        if ev_policy == "max_physical": main_offense = "atk"
        elif ev_policy == "max_special": main_offense = "spa"

        evs = [0,0,0,0,0,0]; ivs = DEFAULT_IVS[:]; nature = "serious"
        if ev_policy in ("auto","max_offense","max_physical","max_special"):
            if main_offense == "atk":
                evs = [4,252,0,0,0,252]; nature = "adamant"
            else:
                evs = [4,0,0,252,0,252]; nature = "modest"
        elif ev_policy == "balanced":
            evs = [252,0,0,0,4,252]; nature = "jolly"

        raw_stats_arr = compute_raw_stats(species=species, evs=evs, ivs=ivs, level=level, nature=nature, data=gen_data)
        raw = {k: int(raw_stats_arr[i]) for i, k in enumerate(STATS)}
        if max_hp is None: max_hp = raw["hp"]
        if current_hp is None and hp_fraction is not None and max_hp:
            current_hp = int(round(max_hp * float(hp_fraction)))
        # ## PATCH: default hp when unknown
        if current_hp is None and max_hp is not None:
            if (status or '').lower() == 'fnt':
                current_hp = 0
                hp_fraction = 0.0
            else:
                current_hp = int(max_hp)
                hp_fraction = 1.0

        abilities_dict = _get_species_entry(gen_data, species).get("abilities", {})
        ability_options = []
        if isinstance(abilities_dict, dict):
            for key in ("0","1","H","S"):
                if abilities_dict.get(key):
                    ability_options.append(to_id_str(abilities_dict.get(key)))

        moves: List[MoveSlot] = []
        pkm_moves = getattr(mon, "moves", None)
        if isinstance(pkm_moves, dict):
            for mv in pkm_moves.values():
                try: moves.append(MoveSlot.from_pokenv_move(mv))
                except Exception: pass
        elif isinstance(pkm_moves, (list, tuple)):
            for mv in pkm_moves:
                try: moves.append(MoveSlot.from_pokenv_move(mv))
                except Exception: pass
        if len(moves) > 4: moves = moves[:4]

        stats = ResolvedStats(base=base, evs={k: evs[i] for i,k in enumerate(STATS)}, ivs={k: ivs[i] for i,k in enumerate(STATS)},
                              nature=nature, level=level, raw=raw, boosts=boosts)

        is_active = False
        try: is_active = (mon is getattr(battle, "opponent_active_pokemon", None))
        except Exception: is_active = False

        return cls(
            species=str(species), nickname=nickname, types=types, ability=ability,
            ability_options=ability_options or None, item=item, tera_type=tera_type,
            max_hp=max_hp, current_hp=current_hp, hp_fraction=hp_fraction, status=status,
            volatiles=set(vol_raw), is_active=is_active,
            is_trapped=bool(is_trapped) if is_trapped is not None else None,
            has_priority_block=(ability in {'dazzling','queenlymajesty','armortail'}),
            grounded=None, stats=stats, moves=moves
        )

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["volatiles"] = sorted(list(self.volatiles))
        out["moves"] = [asdict(m) for m in self.moves]
        out["stats"]["raw"] = {k: int(v) for k, v in self.stats.raw.items()}
        out["stats"]["base"] = {k: int(v) for k, v in self.stats.base.items()}
        return out

@dataclass
class TeamState:
    ours: Dict[str, PokemonState] = field(default_factory=dict)
    opponent: Dict[str, PokemonState] = field(default_factory=dict)

    @classmethod
    def from_battle(cls, battle: Any, gen: int = 9, ev_policy: str = "auto") -> "TeamState":
        gendata = GenData.from_gen(gen)
        ours: Dict[str, PokemonState] = {}
        team_dict = getattr(battle, "team", {}) or {}
        for key, mon in team_dict.items():
            try: ours[str(key)] = PokemonState.from_ally(battle, mon, gendata)
            except Exception: pass

        opponents: Dict[str, PokemonState] = {}
        opp_dict = getattr(battle, "opponent_team", {}) or {}
        for key, mon in opp_dict.items():
            try: opponents[str(key)] = PokemonState.from_opponent(battle, mon, gendata, ev_policy=ev_policy)
            except Exception: pass

        return cls(ours=ours, opponent=opponents)
