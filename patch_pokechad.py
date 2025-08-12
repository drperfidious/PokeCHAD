#!/usr/bin/env python3
"""
One-shot text patcher for PokeCHAD (no git needed).

What it does:
  1) Data/poke_env_moves_info.py
     - Injects a robust, normalized get_type_chart() and helpers.
  2) Models/stockfish_model.py
     - Replaces direct calls to mi.get_type_chart() with a safe helper.
  3) Data/team_state.py
     - Merges bench (available_switches) into TeamState and copies revealed moves.
     - Ensures default HP values are sane when Showdown omits them.

All edits are idempotent and a .bak is written next to each file.
"""

import os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _write_backup_then(p, new_src):
    bak = p + ".bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(_read(p))
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_src)

# -------------------- 1) Normalize the Showdown type chart --------------------

def patch_moves_info_typechart():
    path = os.path.join(ROOT, "Data", "poke_env_moves_info.py")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    src = _read(path)

    # Inject helpers (once)
    if "_UI_TC_NORMALIZER_INSTALLED" not in src:
        # Insert helpers right after the imports block
        helpers = r'''
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
'''
        # Find the first import block
        m = re.search(r'^(?:from[^\n]*\n|import[^\n]*\n)+', src, flags=re.MULTILINE)
        insert_at = m.end() if m else 0
        src = src[:insert_at] + helpers + src[insert_at:]

    # Ensure MovesInfo.get_type_chart exists and uses the normalizer.
    if "class MovesInfo" in src:
        # 1a) Replace an existing get_type_chart, if any.
        def repl_get_tc(m):
            indent = m.group(1)
            body = f"""{indent}def get_type_chart(self):
{indent}    \"\"\"Return a normalized, uppercase type chart with caching.\"\"\"
{indent}    try:
{indent}        tc = getattr(self, "_type_chart_cache", None)
{indent}        if tc is None:
{indent}            raw_tc = getattr(getattr(self, "_data", None), "type_chart", None) or getattr(self, "type_chart", None) or {{}}
{indent}            tc = _normalize_showdown_typechart(raw_tc)
{indent}            setattr(self, "_type_chart_cache", tc)
{indent}        return tc
{indent}    except Exception:
{indent}        return {{}}
"""
            return body

        pat = re.compile(r'(^\s*)def\s+get_type_chart\s*\(self[^\)]*\)\s*:\s*[\s\S]*?(?=^\s*def\s|\Z)',
                         flags=re.MULTILINE)
        if pat.search(src):
            # Use a function replacement to avoid backslash-escape issues in the replacement
            src = pat.sub(lambda m: repl_get_tc(m), src, count=1)
        else:
            # 1b) Insert a new get_type_chart right after the class header
            m = re.search(r'^class\s+MovesInfo[^\n]*:\s*\n', src, flags=re.MULTILINE)
            if m:
                insert_at = m.end()
                method = repl_get_tc(type("M", (), {"group": lambda _self, _i=1: "    "})())
                src = src[:insert_at] + method + src[insert_at:]

    _write_backup_then(path, src)
    print(f"[ok] normalized type chart + get_type_chart() in {path}")

# -------------------- 2) Harden stockfish_model for type chart -------------------

def patch_stockfish_model_safe_get_tc():
    path = os.path.join(ROOT, "Models", "stockfish_model.py")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    src = _read(path)

    # Inject safe helper once
    if "_SAFE_GET_TYPE_CHART" not in src:
        helper = r'''
# --- BEGIN SAFE TYPE CHART HELPER ---
_SAFE_GET_TYPE_CHART = True
def _safe_get_type_chart(mi):
    try:
        return mi.get_type_chart()
    except Exception:
        pass
    # fall back to any raw chart we can find
    try:
        raw_tc = getattr(mi, "type_chart", None)
        if raw_tc is None:
            data = getattr(mi, "_data", None)
            raw_tc = getattr(data, "type_chart", None) if data is not None else None
    except Exception:
        raw_tc = None
    # try to normalize like in Data/poke_env_moves_info
    try:
        from Data.poke_env_moves_info import _normalize_showdown_typechart  # type: ignore
        return _normalize_showdown_typechart(raw_tc or {})
    except Exception:
        return raw_tc if isinstance(raw_tc, dict) else {}
# --- END SAFE TYPE CHART HELPER ---
'''
        # Put this after imports
        m = re.search(r'^(?:from[^\n]*\n|import[^\n]*\n)+', src, flags=re.MULTILINE)
        insert_at = m.end() if m else 0
        src = src[:insert_at] + helper + src[insert_at:]

    # Replace direct mi.get_type_chart() calls with _safe_get_type_chart(mi)
    src = re.sub(r'\bmi\.get_type_chart\(\)', '_safe_get_type_chart(mi)', src)

    _write_backup_then(path, src)
    print(f"[ok] stockfish_model uses _safe_get_type_chart() in {path}")

# --------------- 3) Merge bench into TeamState + default HP guard ---------------

def patch_team_state_merge_bench_and_hp():
    path = os.path.join(ROOT, "Data", "team_state.py")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    src = _read(path)

    # (a) default HP when Showdown omits it (run after current_hp calc)
    if "## PATCH: default hp when unknown" not in src:
        src = re.sub(
            r'(current_hp\s*=\s*int\(round\(max_hp\s*\*\s*float\(hp_fraction\)\)\)\s*\)\s*)',
            r'\1\n        # ## PATCH: default hp when unknown\n'
            r"        if current_hp is None and max_hp is not None:\n"
            r"            if (status or '').lower() == 'fnt':\n"
            r"                current_hp = 0\n"
            r"                hp_fraction = 0.0\n"
            r"            else:\n"
            r"                current_hp = int(max_hp)\n"
            r"                hp_fraction = 1.0\n",
            src
        )

    # (b) merge bench available_switches & copy revealed moves (once)
    if "# --- PATCH: merge available_switches" not in src:
        bench_block = r'''
        # --- PATCH: merge available_switches (bench) into state & copy revealed moves ---
        try:
            bench = getattr(battle, "available_switches", None) or []
            for p in bench:
                species = getattr(p, "species", None) or getattr(p, "name", None) or str(p)
                if not species:
                    continue
                # Find existing key by species (case-insensitive)
                match_key = None
                for k, ps in ours.items():
                    if str(ps.species or "").lower() == str(species).lower():
                        match_key = k
                        break
                if match_key is None:
                    # Create a new entry if we didn't see it in battle.team
                    try:
                        ours[f"p1: {species}"] = PokemonState.from_ally(battle, p, gendata)
                    except Exception:
                        pass
                    continue

                # Merge revealed moves from the poke-env Pokemon into our snapshot
                try:
                    bench_moves = getattr(p, "moves", None) or {}
                    slots = []
                    if isinstance(bench_moves, dict):
                        for mv in bench_moves.values():
                            try:
                                slots.append(MoveSlot.from_pokenv_move(mv))
                            except Exception:
                                pass
                    elif isinstance(bench_moves, (list, tuple)):
                        for mv in bench_moves:
                            try:
                                slots.append(MoveSlot.from_pokenv_move(mv))
                            except Exception:
                                pass
                    if slots and (not ours[match_key].moves or len(ours[match_key].moves) < len(slots)):
                        ours[match_key].moves = slots[:4]
                except Exception:
                    pass
        except Exception:
            pass
        # --- END PATCH ---
'''
        # Insert just before "return cls(ours=ours, opponent=opponents)"
        src = re.sub(
            r'(\n\s*return\s+cls\(ours=ours,\s*opponent=opponents\)\s*)',
            bench_block + r'\1',
            src, count=1
        )

    _write_backup_then(path, src)
    print(f"[ok] TeamState bench merge + default HP guard in {path}")

# --------------------------------- Runner ---------------------------------------

def main():
    patch_moves_info_typechart()
    patch_stockfish_model_safe_get_tc()
    patch_team_state_merge_bench_and_hp()
    print("[done] All patches applied.")

if __name__ == "__main__":
    main()
