#!/usr/bin/env python3
"""
Patch the Think UI data mapping to use Play! Showdown .json/.js data as fallback.

- tools/Data/showdown/ps_data_loader.py:
    * add _strip_js_comments / _load_js helpers
    * include items/abilities/moves/typechart with .json or .js
- Data/poke_env_moves_info.py:
    * load Showdown dex once in MovesInfo.__init__
    * exists/raw/get/get_type_chart now fall back to Showdown data

Run from project root:
    python patch_think_showdown.py
"""

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PS_LOADER = ROOT / "tools" / "Data" / "showdown" / "ps_data_loader.py"
MOVES_INFO = ROOT / "Data" / "poke_env_moves_info.py"


def backup(path: Path):
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix(path.suffix + f".{ts}.bak")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {path} -> {bak.name}")


def ensure_imports_ps_loader(src: str) -> str:
    changed = False
    if not re.search(r'^\s*import\s+json\b', src, flags=re.M):
        src = re.sub(r'^(from\s+typing.*?\n)', r'\1import json\n', src, flags=re.S | re.M)
        changed = True
    if not re.search(r'^\s*import\s+re\b', src, flags=re.M):
        src = re.sub(r'^(from\s+typing.*?\n)', r'\1import re\n', src, flags=re.S | re.M)
        changed = True
    if not re.search(r'^\s*import\s+os\b', src, flags=re.M):
        src = re.sub(r'^(from\s+typing.*?\n)', r'\1import os\n', src, flags=re.S | re.M)
        changed = True
    if not re.search(r'from\s+typing\s+import\s+.*\bOptional\b', src):
        # Be gentle: extend existing typing import if present
        m = re.search(r'from\s+typing\s+import\s+([^\n]+)', src)
        if m:
            before = m.group(0)
            if "Optional" not in before:
                after = before.rstrip() + ", Optional"
                src = src.replace(before, after)
                changed = True
        else:
            src = re.sub(r'^(import\s+json.*?\n)', r'\1from typing import Optional\n', src, flags=re.S | re.M)
            changed = True
    if changed:
        print("[patch] ps_data_loader imports updated")
    return src


JS_HELPERS = r'''
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
'''.lstrip()

LOAD_DIR_FUNC = r'''
def load_showdown_dir(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load Showdown dex data (items, abilities, moves, typechart) from JSON or JS."""
    def find(*names: str) -> Optional[str]:
        for base, _, files in os.walk(root_dir):
            for n in names:
                if n in files:
                    return os.path.join(base, n)
        return None

    items_p     = find("items.json", "items.js", "items")
    abilities_p = find("abilities.json", "abilities.js", "abilities")
    moves_p     = find("moves.json", "moves.js", "moves")
    typechart_p = find("typechart.json", "typechart.js", "typechart")

    dex: Dict[str, Dict[str, Any]] = {"items": {}, "abilities": {}, "moves": {}, "typechart": {}}

    def load_any(p: str) -> Dict[str, Any]:
        if p.endswith(".json"):
            return _load_json(p)
        return _load_js(p)

    if items_p:
        dex["items"] = load_any(items_p)
    if abilities_p:
        dex["abilities"] = load_any(abilities_p)
    if moves_p:
        dex["moves"] = load_any(moves_p)
    if typechart_p:
        dex["typechart"] = load_any(typechart_p)

    return dex
'''.lstrip()


def patch_ps_data_loader():
    if not PS_LOADER.exists():
        print(f"[skip] {PS_LOADER} not found")
        return

    src = PS_LOADER.read_text(encoding="utf-8")
    orig = src

    src = ensure_imports_ps_loader(src)

    # Insert JS helpers if missing
    if "_load_js(" not in src:
        # Put after _load_json if present; else after imports
        ins_point = re.search(r"\ndef\s+_load_json\([^\)]*\):", src)
        if ins_point:
            idx = ins_point.end()
            # Insert helpers AFTER the _load_json function block
            # Find end of that function by dedenting
            block = re.search(r"\ndef\s+_load_json\([^\)]*\):[\s\S]*?(?=\n\w|\Z)", src)
            if block:
                insert_at = block.end()
                src = src[:insert_at] + "\n\n" + JS_HELPERS + src[insert_at:]
            else:
                # Fallback: insert after imports
                imp_block = re.search(r"(?:^|\n)(?:from\s+typing.*\n)(?:import.*\n)*", src)
                insert_at = imp_block.end() if imp_block else 0
                src = src[:insert_at] + "\n" + JS_HELPERS + src[insert_at:]
        else:
            # Fallback: insert after imports
            imp_block = re.search(r"(?:^|\n)(?:from\s+typing.*\n)(?:import.*\n)*", src)
            insert_at = imp_block.end() if imp_block else 0
            src = src[:insert_at] + "\n" + JS_HELPERS + src[insert_at:]
        print("[patch] added _strip_js_comments/_load_js to ps_data_loader.py")
    else:
        print("[info] _load_js already present in ps_data_loader.py")

    # Replace load_showdown_dir with our version that supports .js + typechart
    if "typechart" not in src or re.search(r"def\s+load_showdown_dir\(", src) and ".js" not in src:
        # Replace full function block
        src = re.sub(
            r"\ndef\s+load_showdown_dir\([^\)]*\):[\s\S]*?(?=\n\w|\Z)",
            "\n" + LOAD_DIR_FUNC,
            src,
            count=1,
        )
        print("[patch] replaced load_showdown_dir with JSON/JS-aware version")
    else:
        print("[info] load_showdown_dir already JSON/JS-aware")

    if src != orig:
        backup(PS_LOADER)
        PS_LOADER.write_text(src, encoding="utf-8")
        print(f"[done] patched {PS_LOADER}")
    else:
        print(f"[skip] no changes needed in {PS_LOADER}")


# ---------- poke_env_moves_info.py patching ----------

MOVESINFO_IMPORT = r'''
# Added by patch_think_showdown.py
import os
try:
    from tools.Data.showdown.ps_data_loader import load_showdown_dir  # noqa: F401
except Exception:
    load_showdown_dir = None
'''.lstrip()

METHOD_INIT = r'''
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
'''.rstrip()

METHOD_EXISTS = r'''
    def exists(self, name_or_id: str) -> bool:
        mid = to_id_str(name_or_id)
        in_pokeenv = bool(getattr(self, "_data", None)) and (mid in getattr(self._data, "moves", {}))
        in_ps = bool(self._ps_dex) and (mid in self._ps_dex.get("moves", {}))
        return in_pokeenv or in_ps
'''.rstrip()

METHOD_RAW = r'''
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
'''.rstrip()

METHOD_GET = r'''
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
'''.rstrip()

METHOD_TYPECHART = r'''
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
'''.rstrip()


def ensure_movesinfo_imports(src: str) -> str:
    changed = False
    if not re.search(r'^\s*import\s+os\b', src, flags=re.M):
        # Add after the last import line
        m = list(re.finditer(r'^(?:from\s+\S+\s+import\s+[^\n]+|import\s+\S+)\s*$', src, flags=re.M))
        if m:
            idx = m[-1].end()
            src = src[:idx] + "\n" + MOVESINFO_IMPORT + src[idx:]
        else:
            src = MOVESINFO_IMPORT + "\n" + src
        changed = True
    elif "load_showdown_dir" not in src:
        # os is present, but not our import
        m = list(re.finditer(r'^(?:from\s+\S+\s+import\s+[^\n]+|import\s+\S+)\s*$', src, flags=re.M))
        idx = m[-1].end() if m else 0
        src = src[:idx] + "\n" + MOVESINFO_IMPORT + src[idx:]
        changed = True

    if changed:
        print("[patch] added imports to poke_env_moves_info.py")
    return src


def replace_or_insert_method(src: str, class_name: str, method_def: str, method_name: str) -> str:
    # Find class block
    class_pat = re.compile(rf'(class\s+{class_name}\s*:\s*\n)([\s\S]*?)(?=\n\S)', re.M)
    m = class_pat.search(src)
    if not m:
        # Try until EOF
        class_pat = re.compile(rf'(class\s+{class_name}\s*:\s*\n)([\s\S]*)\Z', re.M)
        m = class_pat.search(src)
    if not m:
        print(f"[warn] class {class_name} not found; cannot place {method_name}")
        return src

    header, body = m.group(1), m.group(2)
    indent_match = re.search(r'^(\s+)def\s', body, flags=re.M)
    indent = indent_match.group(1) if indent_match else "    "

    # If method already present, replace its body
    meth_pat = re.compile(rf'^{indent}def\s+{method_name}\s*\(self[^\)]*\):[\s\S]*?(?=^{indent}def\s+\w|\Z)', re.M)
    if meth_pat.search(body):
        body = meth_pat.sub(textwrap_indent(method_def.strip() + "\n", indent), body, count=1)
        print(f"[patch] replaced {class_name}.{method_name}")
    else:
        # Insert near end of class
        body = body.rstrip() + "\n\n" + textwrap_indent(method_def.strip() + "\n", indent)
        print(f"[patch] inserted {class_name}.{method_name}")

    # Rebuild src
    start, end = m.span()
    return src[:start] + header + body + src[end:]


def textwrap_indent(text: str, indent: str) -> str:
    return "\n".join((indent + line if line.strip() else line) for line in text.splitlines())


def patch_moves_info():
    if not MOVES_INFO.exists():
        print(f"[skip] {MOVES_INFO} not found")
        return

    src = MOVES_INFO.read_text(encoding="utf-8")
    orig = src

    # Quick idempotency: if we already reference self._ps_dex, skip re-adding imports/methods
    already = "self._ps_dex" in src

    if not already:
        src = ensure_movesinfo_imports(src)

        # Replace/insert key methods inside MovesInfo
        src = replace_or_insert_method(src, "MovesInfo", METHOD_INIT, "__init__")
        src = replace_or_insert_method(src, "MovesInfo", METHOD_EXISTS, "exists")
        src = replace_or_insert_method(src, "MovesInfo", METHOD_RAW, "raw")
        src = replace_or_insert_method(src, "MovesInfo", METHOD_GET, "get")
        src = replace_or_insert_method(src, "MovesInfo", METHOD_TYPECHART, "get_type_chart")
    else:
        print("[info] poke_env_moves_info.py already references self._ps_dex; skipping method injection")

    if src != orig:
        backup(MOVES_INFO)
        MOVES_INFO.write_text(src, encoding="utf-8")
        print(f"[done] patched {MOVES_INFO}")
    else:
        print(f"[skip] no changes needed in {MOVES_INFO}")


def main():
    print("[start] Think UI Showdown fallback patch")
    patch_ps_data_loader()
    patch_moves_info()
    print("[done] All patches attempted.")


if __name__ == "__main__":
    main()
