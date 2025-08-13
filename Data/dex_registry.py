"""dex_registry.py
Central lightweight registry wrapping the raw Showdown JS / JSON data files
(abilities, items, moves) so that battle logic can access richer passive data
without re-parsing each time.

Loads from the first existing directory among (relative to project root):
  - showdown/
  - Resources/showdown/
  - tools/Data/showdown/

Provides helper functions:
  get_item(id) -> dict | None
  get_ability(id) -> dict | None
  get_move(id) -> dict | None

All ids are normalized with _to_id (lowercase alphanumerics only).

We intentionally keep parsing extremely simple: JS exports are parsed via the
existing ps_data_loader when possible; otherwise we fall back to the local
loader here.
"""
from __future__ import annotations

from functools import lru_cache
import os, json, re
from typing import Dict, Any, Optional

_DEF_DIR_CANDIDATES = [
    os.path.join(os.getcwd(), 'showdown'),
    os.path.join(os.getcwd(), 'Resources', 'showdown'),
    os.path.join(os.getcwd(), 'tools', 'Data', 'showdown'),
]

_ID_RX = re.compile(r"[^a-z0-9]")

def _to_id(s: str) -> str:
    return _ID_RX.sub('', (s or '').lower())

def _strip_js_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"(^|\s)//.*?$", r"\1", src, flags=re.M)
    return src

def _parse_js_object_literal(text: str) -> Dict[str, Any]:
    # Extract the last object literal assignment
    m = re.search(r"=\s*({.*})\s*;?\s*$", text, flags=re.S)
    if not m:
        m = re.search(r"({.*})\s*;?\s*$", text, flags=re.S)
    if not m:
        return {}
    obj = m.group(1)
    obj = re.sub(r",(\s*[}\]])", r"\1", obj)  # trailing commas
    # quote bare keys safely
    obj = re.sub(r'([:{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', obj)
    try:
        data = json.loads(obj)
    except Exception:
        return {}
    return { _to_id(k): v for k, v in data.items() }

@lru_cache(maxsize=1)
def _load_all() -> Dict[str, Dict[str, Any]]:
    root = None
    for cand in _DEF_DIR_CANDIDATES:
        if os.path.isdir(cand):
            root = cand
            break
    if root is None:
        return {'items': {}, 'abilities': {}, 'moves': {}}

    def load_json(name: str) -> Dict[str, Any]:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            return {}
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return { _to_id(k): v for k, v in data.items() }
            return {}
        except Exception:
            return {}

    def load_js(name: str) -> Dict[str, Any]:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            return {}
        try:
            with open(p, 'r', encoding='utf-8') as f:
                txt = _strip_js_comments(f.read())
            return _parse_js_object_literal(txt)
        except Exception:
            return {}

    # Try both js and json variants
    items = load_js('items.js') or load_json('items.json')
    abilities = load_js('abilities.js') or load_json('abilities.json')
    moves = load_json('moves.json') or load_js('moves.js')  # moves.json usually large

    return {'items': items, 'abilities': abilities, 'moves': moves}

# Public API

def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    d = _load_all()['items']
    return d.get(_to_id(item_id))

def get_ability(ability_id: str) -> Optional[Dict[str, Any]]:
    d = _load_all()['abilities']
    return d.get(_to_id(ability_id))

def get_move(move_id: str) -> Optional[Dict[str, Any]]:
    d = _load_all()['moves']
    return d.get(_to_id(move_id))

def has_data() -> bool:
    data = _load_all()
    return bool(data['items'] or data['abilities'] or data['moves'])
