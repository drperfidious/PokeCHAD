"""
Pokemon information and data management module
----------------------------------------------
Wraps poke_env.data.GenData for species data and exposes a stat estimator that
follows the in-game formulas (or defers to poke_env.stats.compute_raw_stats if
available).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
    from poke_env import stats as pe_stats
except Exception:  # pragma: no cover
    GenData = None
    pe_stats = None
    def to_id_str(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())


@dataclass
class PokemonStats:
    hp: int = 0
    atk: int = 0
    def_: int = 0
    spa: int = 0
    spd: int = 0
    spe: int = 0


@dataclass
class SpeciesInfo:
    id: str
    name: str
    types: List[str] = field(default_factory=list)
    abilities: Dict[Union[str, int], str] = field(default_factory=dict)
    base_stats: PokemonStats = field(default_factory=PokemonStats)
    weightkg: Optional[float] = None
    heightm: Optional[float] = None
    evos: List[str] = field(default_factory=list)
    prevo: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ------------------------- Core wrapper ------------------------------------------

class PokemonInfo:
    def __init__(self, gen_or_format: Union[int, str] = 9):
        if isinstance(gen_or_format, int):
            self._data = GenData.from_gen(gen_or_format) if GenData else None
        else:
            self._data = GenData.from_format(gen_or_format) if GenData else None

    @property
    def gen(self) -> int:
        return getattr(self._data, "gen", 9)

    def all_ids(self) -> List[str]:
        if not self._data:
            return []
        return list(self._data.pokedex.keys())

    def exists(self, name_or_id: str) -> bool:
        if not self._data:
            return False
        return to_id_str(name_or_id) in self._data.pokedex

    def raw(self, name_or_id: str) -> Dict[str, Any]:
        if not self._data:
            raise RuntimeError("poke_env not available")
        sid = to_id_str(name_or_id)
        s = self._data.pokedex.get(sid)
        if s is None:
            raise KeyError(f"Unknown species: {name_or_id} (normalized: {sid})")
        return s

    def get(self, name_or_id: str) -> SpeciesInfo:
        s = self.raw(name_or_id)
        sid = to_id_str(name_or_id)
        bs = s.get("baseStats", {})
        return SpeciesInfo(
            id=sid,
            name=s.get("name", sid),
            types=s.get("types", []),
            abilities=s.get("abilities", {}),
            base_stats=PokemonStats(
                hp=bs.get("hp", 0), atk=bs.get("atk", 0), def_=bs.get("def", 0),
                spa=bs.get("spa", 0), spd=bs.get("spd", 0), spe=bs.get("spe", 0),
            ),
            weightkg=s.get("weightkg"),
            heightm=s.get("heightm"),
            evos=s.get("evos", []),
            prevo=s.get("prevo"),
            raw=s,
        )

    # ----------- Stats (poke_env path or fallback) --------------------------------

    def compute_raw_stats(
        self,
        species: str,
        evs: Sequence[int],
        ivs: Sequence[int],
        level: int,
        nature: str,
    ) -> List[int]:
        """Delegate to poke_env.stats.compute_raw_stats when available, else fallback."""
        if pe_stats is not None:
            return pe_stats.compute_raw_stats(species, list(evs), list(ivs), level, nature, self._data)
        return self._fallback_compute_raw_stats(species, evs, ivs, level, nature)

    def _fallback_compute_raw_stats(
        self,
        species: str,
        evs: Sequence[int],
        ivs: Sequence[int],
        level: int,
        nature: str,
    ) -> List[int]:
        """In-game formula fallback using the local GenData (approx identical)."""
        s = self.get(species)
        bs = s.base_stats
        # Nature multipliers
        nat = (nature or "Serious").capitalize()
        inc = {"Lonely":"atk","Brave":"atk","Adamant":"atk","Naughty":"atk",
               "Bold":"def_","Relaxed":"def_","Impish":"def_","Lax":"def_",
               "Modest":"spa","Mild":"spa","Quiet":"spa","Rash":"spa",
               "Calm":"spd","Gentle":"spd","Sassy":"spd","Careful":"spd",
               "Timid":"spe","Hasty":"spe","Jolly":"spe","Naive":"spe"}
        dec = {"Bold":"atk","Relaxed":"spe","Impish":"spa","Lax":"spd",
               "Modest":"atk","Mild":"def_","Quiet":"spe","Rash":"spd",
               "Calm":"atk","Gentle":"def_","Sassy":"spe","Careful":"spa",
               "Timid":"atk","Hasty":"def_","Jolly":"spa","Naive":"spd",
               "Lonely":"def_","Brave":"spe","Adamant":"spa","Naughty":"spd"}
        mults = {"atk":1.0,"def_":1.0,"spa":1.0,"spd":1.0,"spe":1.0}
        if nat in inc: mults[inc[nat]] = 1.1
        if nat in dec: mults[dec[nat]] = 0.9

        def stat_non_hp(base: int, ev: int, iv: int) -> int:
            pre = ((2*base + iv + ev//4) * level) // 100 + 5
            return int(pre * mults_key)

        # compute
        ivs = list(ivs) if ivs else [31]*6
        evs = list(evs) if evs else [0]*6
        b = [bs.hp, bs.atk, bs.def_, bs.spa, bs.spd, bs.spe]
        # HP special case
        hp = ((2*b[0] + ivs[0] + evs[0]//4) * level) // 100 + level + 10
        # Others with nature
        defn = ((2*b[2] + ivs[2] + evs[2]//4) * level) // 100 + 5
        atk = int( (((2*b[1] + ivs[1] + evs[1]//4) * level) // 100 + 5) * mults["atk"] )
        spa = int( (((2*b[3] + ivs[3] + evs[3]//4) * level) // 100 + 5) * mults["spa"] )
        spd = int( (((2*b[4] + ivs[4] + evs[4]//4) * level) // 100 + 5) * mults["spd"] )
        spe = int( (((2*b[5] + ivs[5] + evs[5]//4) * level) // 100 + 5) * mults["spe"] )
        return [hp, atk, defn, spa, spd, spe]


# ------------------------- Estimation policy -------------------------------------

@dataclass
class StatEstimationPolicy:
    """Controls default IV/EV/nature assumptions for unknown opponents."""
    level: int = 50
    default_ivs: Sequence[int] = (31,31,31,31,31,31)
    default_evs: Sequence[int] = (0,0,0,0,0,0)
    default_nature: str = "Serious"   # neutral
    # If True, assume 252 in the attacker's relevant stat and Speed, and 4 HP by default
    max_offense_hint: bool = True


def estimate_stats(
    pinfo: PokemonInfo,
    species: str,
    policy: Optional[StatEstimationPolicy] = None,
    as_attacker: bool = False,
    special_attacker_hint: bool = False,
) -> PokemonStats:
    """Estimate raw stats for a species according to a policy.

    If `as_attacker=True`, we bias EVs toward Atk or SpA + Spe depending on hint.
    """
    pol = policy or StatEstimationPolicy()
    evs = list(pol.default_evs)
    if as_attacker and pol.max_offense_hint:
        if special_attacker_hint:
            evs = [4, 0, 0, 252, 0, 252]  # 4 HP / 252 SpA / 252 Spe
        else:
            evs = [4, 252, 0, 0, 0, 252]  # 4 HP / 252 Atk / 252 Spe
    ivs = list(pol.default_ivs)
    stats = pinfo.compute_raw_stats(species, evs, ivs, pol.level, pol.default_nature)
    return PokemonStats(*stats)


# ------------------------- Known item -> stat modifiers --------------------------

def _is_unevolved(pinfo: "PokemonInfo", species: str) -> bool:
    try:
        info = pinfo.get(species)
        # If it has evolutions, it's unevolved. Some species have branching/alt forms.
        return bool(info.evos)
    except Exception:
        return False


def apply_item_stat_modifiers(pinfo: "PokemonInfo", species: str, item: Optional[str], stats: PokemonStats) -> PokemonStats:
    """Apply stat-side item multipliers to a PokemonStats snapshot.

    Only items that *modify stats* (not damage multipliers) are applied here, to
    preserve correct damage-ordering elsewhere (e.g., Life Orb is a damage mod).
    """
    if not item:
        return stats
    it = item.lower()

    hp, atk, deff, spa, spd, spe = stats.hp, stats.atk, stats.def_, stats.spa, stats.spd, stats.spe

    # Choice items
    if it == "choiceband":
        atk = int(atk * 1.5)
    elif it == "choicespecs":
        spa = int(spa * 1.5)
    elif it == "choicescarf":
        spe = int(spe * 1.5)

    # Assault Vest
    if it == "assaultvest":
        spd = int(spd * 1.5)

    # Eviolite (only if unevolved)
    if it == "eviolite" and _is_unevolved(pinfo, species):
        deff = int(deff * 1.5)
        spd = int(spd * 1.5)

    # Species-specific doublers
    sid = to_id_str(species)
    if it == "thickclub" and sid in ("cubone", "marowak", "marowakalola", "marowakalolan"):
        atk = atk * 2
    if it == "lightball" and sid == "pikachu":
        atk = atk * 2
        spa = spa * 2
    if sid == "clamperl":
        if it == "deepseatooth":
            spa = spa * 2
        elif it == "deepseascale":
            spd = spd * 2

    return PokemonStats(hp, atk, deff, spa, spd, spe)
