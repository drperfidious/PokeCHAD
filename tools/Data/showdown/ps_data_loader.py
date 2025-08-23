
# tools/Data/showdown/ps_data_loader.py
"""
Lightweight loader for Pokémon Showdown dex JSON (items, abilities, moves).

Usage
-----
Place exported JSON files from Pokémon Showdown under a directory, e.g.:

    data/showdown/
      abilities.json
      items.json
      moves.json

Then:

    from ps_data_loader import load_showdown_dir
    dex = load_showdown_dir("data/showdown")

`dex` will be a dict with keys: 'items', 'abilities', 'moves' mapping id -> dict.
We normalize ids by lowercasing and stripping non-alphanumerics (like Showdown ids).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
import re

def _strip_js_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"(^|\s)//.*?$", r"\1", src, flags=re.M)
    return src

def _ensure__to_id():
    # Some repos define _to_id already; if not, provide a compatible fallback.
    globals_ = globals()
    if "_to_id" not in globals_:
        def _to_id(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (s or "").lower())
        globals_["_to_id"] = _to_id

def _load_js(path: str) -> Dict[str, Any]:
    """Parse Showdown-style JS exports like `exports.Moves = { ... }`."""
    _ensure__to_id()
    with open(path, "r", encoding="utf-8") as f:
        src = _strip_js_comments(f.read())
    # Try to grab final object literal (covers `exports.X = {...};` and `const X = {...}`)
    m = re.search(r"=\s*({.*})\s*;?\s*$", src, flags=re.S)
    if not m:
        # Fallback: last {...} block
        m = re.search(r"({.*})\s*;?\s*$", src, flags=re.S)
    if not m:
        raise ValueError(f"Could not locate object literal in {path}")
    obj = m.group(1)
    # Remove trailing commas
    obj = re.sub(r",(\s*[\]}])", r"\1", obj)
    # Quote bare keys
    obj2 = re.sub(r"([{\[,]\s*)([A-Za-z0-9_]+)\s*:", r'\1"\2":', obj)
    import json as _json
    data = _json.loads(obj2)
    return { _to_id(k): v for k, v in data.items() }

def _to_id(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Some exports store arrays; Showdown uses id fields inside; normalize to dict
    if isinstance(data, list):
        out = {}
        for it in data:
            if isinstance(it, dict):
                sid = _to_id(it.get("id") or it.get("name") or "")
                if sid:
                    out[sid] = it
        return out
    if isinstance(data, dict):
        # If top-level is an array under data['items'], handle that as well
        if "items" in data and isinstance(data["items"], list):
            out = {}
            for it in data["items"]:
                if isinstance(it, dict):
                    sid = _to_id(it.get("id") or it.get("name") or "")
                    if sid:
                        out[sid] = it
            return out
        return { _to_id(k): v for k, v in data.items() }
    raise ValueError(f"Unsupported JSON schema in {path}")

def load_showdown_dir(root_dir: str) -> Dict[str, Dict[str, Any]]:
    root = os.path.abspath(root_dir)
    def find(name: str):
        for cand in (name, name.lower(), name.upper()):
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                return p
        # Try with .json inside
        for cand in (name + ".json", name.lower() + ".json", name.upper() + ".json"):
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                return p
        return None

    items_p = find("items") or find("items.json")
    abilities_p = find("abilities") or find("abilities.json")  
    moves_p = find("moves") or find("moves.json")
    pokedex_p = find("pokedex") or find("pokedex.json")

    dex = {"items": {}, "abilities": {}, "moves": {}, "pokedex": {}}
    if items_p:
        dex["items"] = _load_json(items_p)
    if abilities_p:
        dex["abilities"] = _load_json(abilities_p)
    if moves_p:
        dex["moves"] = _load_json(moves_p)
    if pokedex_p:
        dex["pokedex"] = _load_json(pokedex_p)

    return dex
