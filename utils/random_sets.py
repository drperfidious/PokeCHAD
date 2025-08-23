from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse showdown dir discovery similar to type_effectiveness
_DEF_CANDIDATES = None

def _showdown_base_dir() -> Path:
    global _DEF_CANDIDATES
    if _DEF_CANDIDATES is None:
        root = Path(__file__).resolve().parent.parent
        _DEF_CANDIDATES = [
            Path(os.getenv('POKECHAD_SHOWDOWN_DIR', '')),
            root / 'tools' / 'Data' / 'showdown',
            root / 'showdown',
            root / 'tools' / 'showdown',
            root / 'Resources' / 'showdown',
        ]
    for c in _DEF_CANDIDATES:
        if not c:
            continue
        try:
            if (c / 'moves.json').exists() or (c / 'pokedex.json').exists():
                return c
        except Exception:
            continue
    # default to tools/Data/showdown
    return _DEF_CANDIDATES[1]


def _parse_gen_from_format(fmt: Optional[str]) -> int:
    if not fmt:
        return 9
    fmt = str(fmt).lower()
    if fmt.startswith('gen'):
        # extract digits after 'gen'
        num = ''
        for ch in fmt[3:]:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            try:
                return int(num)
            except Exception:
                pass
    return 9

# Cache: gen -> { species_id -> [set_obj, ...] }
_CACHE: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}


def _load_sets_for_gen(gen: int) -> Dict[str, List[Dict[str, Any]]]:
    if gen in _CACHE:
        return _CACHE[gen]
    base = _showdown_base_dir()
    # If the initially selected base doesn't have sets for this gen, try other candidates
    try:
        # Ensure candidates list is initialized
        _ = _showdown_base_dir()
        candidates = [c for c in (_DEF_CANDIDATES or []) if c]
    except Exception:
        candidates = [base]
    chosen_base = None
    for c in candidates:
        try:
            sets_path = c / 'sets' / f'gen{gen}.json'
            alt_path = c / f'gen{gen}_random_sets.json'
            if sets_path.exists() or alt_path.exists():
                chosen_base = c
                break
        except Exception:
            continue
    if chosen_base is None:
        chosen_base = base
    base = chosen_base
    p = base / 'sets' / f'gen{gen}.json'
    try:
        if not p.exists():
            # Some distributions flatten to random-sets.json
            alt = base / f'gen{gen}_random_sets.json'
            if alt.exists():
                p = alt
        data: Dict[str, Any] = {}
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
        # Normalize to id -> list[set]
        norm: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                sid = str(k).lower().replace(' ', '').replace('-', '')
                if isinstance(v, list):
                    norm[sid] = [x for x in v if isinstance(x, dict)]
                elif isinstance(v, dict):
                    # If the JSON nests under {"sets": [...]}
                    if 'sets' in v and isinstance(v['sets'], list):
                        norm[sid] = [x for x in v['sets'] if isinstance(x, dict)]
                    else:
                        # Treat as a single set definition
                        norm[sid] = [v]
        _CACHE[gen] = norm
        return norm
    except Exception:
        _CACHE[gen] = {}
        return {}


def is_random_format(fmt: Optional[str]) -> bool:
    f = (fmt or '').lower()
    return 'randombattle' in f or 'random' in f


@dataclass
class ProvisionalSet:
    moves: List[str]
    ability: Optional[str] = None
    item: Optional[str] = None
    level: Optional[int] = None
    nature: Optional[str] = None
    evs: Optional[Dict[str, int]] = None
    ivs: Optional[Dict[str, int]] = None


def _pick_moves_from_set(defn: Dict[str, Any], rng: Optional[random.Random] = None) -> List[str]:
    # Common layouts: {"moves": ["a","b","c","d"]} or {"moves": [["a","b"], ["c"], ...]}
    raw = defn.get('moves')
    pool: List[str] = []
    if isinstance(raw, list):
        if all(isinstance(x, str) for x in raw):
            pool = [x for x in raw]
        else:
            # assume list of options per slot
            choices: List[str] = []
            for slot in raw:
                if isinstance(slot, list) and slot:
                    choices.append((rng or random).choice(slot))
                elif isinstance(slot, str):
                    choices.append(slot)
            pool = choices
    # Fallbacks from other PS structures
    if not pool:
        for key in ('movePool', 'randomBattleMoves', 'randommoves'):
            alt = defn.get(key)
            if isinstance(alt, list) and alt and all(isinstance(x, str) for x in alt):
                pool = list(alt)
                break
    # Ensure ids
    pool = [str(m).lower().replace(' ', '').replace('-', '') for m in pool]
    # Cap at 4, random sample if larger
    if len(pool) > 4:
        (rng or random).shuffle(pool)
        pool = pool[:4]
    return pool


def choose_random_set(species: str, fmt_or_gen: Optional[str | int], *, seed: Optional[str|int]=None) -> Optional[ProvisionalSet]:
    if isinstance(fmt_or_gen, int):
        gen = fmt_or_gen
    else:
        gen = _parse_gen_from_format(fmt_or_gen)
    sets = _load_sets_for_gen(gen)
    if not sets:
        return None
    sid = str(species).lower().replace(' ', '').replace('-', '')
    arr = sets.get(sid)
    if not arr:
        # Try base species without form suffix
        if sid.endswith('alola') or sid.endswith('alolan'):
            base = sid.replace('alola', '').replace('alolan', '')
        else:
            base = sid.split('mega')[0].split('-')[0]
        arr = sets.get(base)
    if not arr:
        return None
    rng = random.Random()
    if seed is not None:
        try:
            rng.seed(str(seed))
        except Exception:
            pass
    cand = rng.choice(arr)
    moves = _pick_moves_from_set(cand, rng)
    ability = cand.get('ability') or cand.get('abilities')
    if isinstance(ability, list):
        ability = rng.choice(ability) if ability else None
    item = cand.get('item') or cand.get('items')
    if isinstance(item, list):
        item = rng.choice(item) if item else None
    # Optional extras
    level = cand.get('level')
    nature = cand.get('nature')
    evs = cand.get('evs') if isinstance(cand.get('evs'), dict) else None
    ivs = cand.get('ivs') if isinstance(cand.get('ivs'), dict) else None
    # Normalize ids
    ability = str(ability).lower().replace(' ', '').replace('-', '') if ability else None
    item = str(item).lower().replace(' ', '').replace('-', '') if item else None
    return ProvisionalSet(moves=moves or [], ability=ability or None, item=item or None, level=level, nature=nature, evs=evs, ivs=ivs)
