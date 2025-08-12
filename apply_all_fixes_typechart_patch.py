#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time
from pathlib import Path

MODEL_CANDIDATES = ["Models/stockfish_model.py", "models/stockfish_model.py", "stockfish_model.py", "/workspace/Models/stockfish_model.py"]
UI_MAIN_CANDIDATES = ["UI/tk_stockfish_model_ui.py", "ui/tk_stockfish_model_ui.py", "tk_stockfish_model_ui.py", "/workspace/UI/tk_stockfish_model_ui.py"]
UI_PATCH_CANDIDATES = ["UI/ui_think_patch.py", "ui/ui_think_patch.py", "ui_think_patch.py", "/workspace/UI/ui_think_patch.py"]

MODEL_NEW = r"""
from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Data.battle_runtime import (
    get_state,
    predict_order_for_ids,
    estimate_damage,
    would_fail,
    apply_switch_in_effects,
)
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo

# ---------------- Weights ----------------
def _load_weights(path: str = os.path.join("Models", "weights.json")) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
        return {
            "go_first_bonus": float(w.get("go_first_bonus", 0.4)),
            "opp_dmg_penalty": float(w.get("opp_dmg_penalty", 1.0)),
            "survival_bonus": float(w.get("survival_bonus", 0.0)),
        }
    except Exception:
        return {"go_first_bonus": 0.4, "opp_dmg_penalty": 1.0, "survival_bonus": 0.0}

# ---------------- Helpers ----------------
def _acc_to_prob(acc) -> float:
    if acc is True or acc is None:
        return 1.0
    try:
        x = float(acc)
        return x / 100.0 if x > 1.0 else max(0.0, min(1.0, x))
    except Exception:
        return 1.0

def _hp_frac(ps) -> float:
    try:
        if ps.max_hp:
            return max(0.0, min(1.0, (ps.current_hp or ps.max_hp) / ps.max_hp))
    except Exception:
        pass
    return 1.0

def _safe_get_type_chart(mi: MovesInfo) -> Dict[str, Dict[str, float]]:
    \"\"\"Resilient getter that tolerates PS-style charts; callers shouldn't depend on shape.\"\"\"
    try:
        return mi.get_type_chart()
    except Exception:
        pass
    # fallback: raw attribute(s)
    try:
        raw_tc = getattr(mi, "type_chart", None)
        if raw_tc is None:
            data = getattr(mi, "_data", None)
            raw_tc = getattr(data, "type_chart", None) if data is not None else None
    except Exception:
        raw_tc = None
    # attempt to normalize via the data module if available
    try:
        from Data.poke_env_moves_info import _normalize_showdown_typechart  # type: ignore
        return _normalize_showdown_typechart(raw_tc or {})
    except Exception:
        return raw_tc if isinstance(raw_tc, dict) else {}

def _expected_damage_fraction(state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo) -> Tuple[float, Dict[str, Any]]:
    \"\"\"Return (expected fraction of defender max HP, damage dict).\"\"\"
    dmg = estimate_damage(state, atk_key, dfd_key, move_id, mi, is_critical=False)
    rolls = dmg.get("rolls") or []
    if not rolls:
        return 0.0, dmg
    dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
    max_hp = int(getattr(dfd, "max_hp", 0) or dfd.stats.raw.get("hp", 1) or 1)
    exp = float(sum(rolls)) / (len(rolls) * max_hp)
    return exp, dmg

# Fallback move dictionary for unknown opponents / candidates
_COMMON_STAB = {
    "normal": ["return","bodyslam","doubleedge"],
    "fire": ["flamethrower","fireblast","overheat"],
    "water": ["surf","hydropump","sparklingaria"],
    "electric": ["thunderbolt","discharge","thunder"],
    "grass": ["energyball","gigadrain","leafstorm"],
    "ice": ["icebeam","blizzard","freezedry"],
    "fighting": ["closecombat","superpower","drainpunch"],
    "poison": ["sludgebomb","gunkshot","poisonjab"],
    "ground": ["earthquake","earthpower","stompingtantrum"],
    "flying": ["hurricane","airslash","bravebird"],
    "psychic": ["psychic","psyshock","futuresight"],
    "bug": ["bugbuzz","utr","leechlife"],
    "rock": ["stoneedge","rockslide","powergem"],
    "ghost": ["shadowball","shadowclaw","poltergeist"],
    "dragon": ["dracometeor","dragonpulse","outrage"],
    "dark": ["darkpulse","crunch","knockoff"],
    "steel": ["flashcannon","ironhead","meteormash"],
    "fairy": ["moonblast","playrough","dazzlinggleam"],
}

def _opp_best_on_target(state, opp_key: str, target_key: str, mi: MovesInfo) -> float:
    \"\"\"Max expected fraction the opponent can deal to target with revealed or fallback STABs (accuracy-weighted).\"\"\"
    opp = state.team.opponent[opp_key]
    best = 0.0
    # Prefer revealed
    mv_list = [m for m in (opp.moves or []) if m and m.id]
    if mv_list:
        for mv in mv_list:
            # ignore pure status or 0 BP
            if (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
                continue
            frac, _ = _expected_damage_fraction(state, opp_key, target_key, mv.id, mi)
            best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
        return best
    # Fallback: pick two STAB candidates by type
    for t in opp.types or []:
        t = (t or "").lower()
        for mid in _COMMON_STAB.get(t, [])[:2]:
            try:
                frac, _ = _expected_damage_fraction(state, opp_key, target_key, mid, mi)
                # Pull accuracy from moves info when we synthesized the move id
                raw = mi.raw(mid) or {}
                best = max(best, frac * _acc_to_prob(raw.get("accuracy", None)))
            except Exception:
                continue
    # Generic coverage if types unknown
    for mid in ("icebeam","closecombat","earthquake"):
        try:
            frac, _ = _expected_damage_fraction(state, opp_key, target_key, mid, mi)
            raw = mi.raw(mid) or {}
            best = max(best, frac * _acc_to_prob(raw.get("accuracy", None)))
        except Exception:
            continue
    return best

def _our_best_on_target(state, my_key: str, opp_key: str, mi: MovesInfo, legal_moves) -> float:
    \"\"\"Best expected fraction we can deal *this turn* with legal damaging moves (accuracy-weighted).\"\"\"
    best = 0.0
    for mv in (legal_moves or []):
        mid = getattr(mv, "id", None) or getattr(mv, "move_id", None)
        if not mid:
            continue
        cat = str(getattr(mv, "category", "") or "").lower()
        bp  = int(getattr(mv, "base_power", 0) or getattr(mv, "basePower", 0) or 0)
        if cat == "status" or bp <= 0:
            continue
        frac, _ = _expected_damage_fraction(state, my_key, opp_key, str(mid), mi)
        best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
    return best

# ---------------- Public data container -----------------
@dataclass
class ChosenAction:
    kind: str  # 'move' | 'switch'
    move_id: Optional[str] = None
    switch_species: Optional[str] = None
    debug: Dict[str, Any] | None = None

# ----------------- Core engine --------------------------
class StockfishModel:
    def __init__(self, battle_format: str = 'gen9ou'):
        self.battle_format = battle_format
        self._W = _load_weights()
        self._depth = 1  # exposed to UI; not used for tree search in this patch

    # API for UI
    def set_depth(self, d: int):
        self._depth = int(d)

    def reload_weights(self, path: str = os.path.join("Models", "weights.json")):
        self._W = _load_weights(path)
        try:
            print(f"[weights] reloaded from {path}: {self._W}")
        except Exception:
            pass

    def choose_action(self, battle: Any) -> ChosenAction:
        \"\"\"Return ChosenAction and rich debug info for the UI + telemetry.\"\"\"
        state = get_state(battle)
        mi = MovesInfo(state.format or 9)
        _ = _safe_get_type_chart(mi)  # ensure chart is loaded (for downstream)

        # Active keys
        def _active_key(side: Dict[str, Any]) -> Optional[str]:
            for k, p in side.items():
                if getattr(p, 'is_active', False):
                    return k
            for k, p in side.items():
                if (getattr(p, 'status', None) or '').lower() == 'fnt':
                    continue
                chp = getattr(p, 'current_hp', None)
                if chp is None:
                    return k
                if chp > 0:
                    return k
            return None

        my_key = _active_key(state.team.ours) if getattr(state, 'team', None) else None
        opp_key = _active_key(state.team.opponent) if getattr(state, 'team', None) else None

        legal_moves = list(getattr(battle, 'available_moves', []) or [])
        legal_switches = list(getattr(battle, 'available_switches', []) or [])
        force_switch = bool(getattr(battle, 'force_switch', False)) or (not legal_moves and bool(legal_switches))

        # Opponent revealed moves
        opp_moves_known = []
        try:
            if opp_key:
                opp_ps = state.team.opponent[opp_key]
                opp_moves_known = [m.id for m in (opp_ps.moves or []) if getattr(m, "id", None)]
        except Exception:
            pass

        # ---- Evaluate MOVES ----
        moves_eval: List[Dict[str, Any]] = []
        first_probs: Dict[str, float] = {}

        if not force_switch and my_key and opp_key:
            opp_ps = state.team.opponent[opp_key]
            opp_max = int(getattr(opp_ps, 'max_hp', 0) or opp_ps.stats.raw.get('hp', 1) or 1)
            opp_hp_now = _hp_frac(opp_ps)

            for mv in legal_moves:
                mid = getattr(mv, 'id', None) or getattr(mv, 'move_id', None)
                name = getattr(mv, 'name', str(mid))
                if not mid:
                    continue

                # Would it fail due to protect/terrain/etc?
                try:
                    fail, why = would_fail(str(mid), my_key, opp_key, state, mi)
                except Exception:
                    fail, why = False, None
                if fail:
                    moves_eval.append({
                        'id': str(mid), 'name': name,
                        'score': 0.0,
                        'expected': 0.0, 'exp_dmg': 0.0,
                        'acc_mult': 0.0, 'acc': 0.0,
                        'effectiveness': 0.0, 'eff': 0.0,
                        'why_blocked': str(why or 'would fail')
                    })
                    continue

                # Damage & accuracy
                try:
                    exp_frac, dmg = _expected_damage_fraction(state, my_key, opp_key, str(mid), mi)
                except Exception as e:
                    exp_frac, dmg = 0.0, {}
                acc_p = _acc_to_prob(getattr(mv, 'accuracy', 1.0))

                # Order prediction
                try:
                    first_prob, _details = predict_order_for_ids(state, my_key, str(mid), opp_key, (opp_moves_known[0] if opp_moves_known else "tackle"), mi)
                except Exception:
                    first_prob = 0.5
                first_probs[str(mid)] = float(first_prob)

                # KO probability if we move first
                rolls = (dmg.get("rolls") or [])
                # threshold in absolute HP
                thr_abs = int(round(opp_hp_now * opp_max))
                ko_rolls = 0
                for r in rolls:
                    if int(r) >= max(1, thr_abs):
                        ko_rolls += 1
                p_ko_if_hit = (ko_rolls / max(1, len(rolls))) if rolls else 0.0
                p_ko_first  = acc_p * p_ko_if_hit * first_prob

                # Opponent counter EV (worst expected they can deal to us) and chance they act
                incoming_frac = 0.0
                try:
                    incoming_frac = _opp_best_on_target(state, opp_key, my_key, mi)
                except Exception:
                    incoming_frac = 0.0
                p_opp_acts = (1.0 - first_prob) + first_prob * (1.0 - p_ko_if_hit * acc_p)
                opp_counter_ev = incoming_frac * p_opp_acts

                # Move score with learned weights
                W = self._W
                score = (exp_frac
                         - W["opp_dmg_penalty"] * opp_counter_ev
                         + W["go_first_bonus"] * first_prob
                         + W["survival_bonus"] * 0.0  # placeholder hook
                         )

                eff = float(dmg.get('effectiveness', 1.0) or 1.0)  # trust pipeline's effectiveness

                moves_eval.append({
                    'id': str(mid), 'name': name,
                    'score': float(score),
                    'expected': float(exp_frac), 'exp_dmg': float(exp_frac),
                    'acc_mult': float(acc_p), 'acc': float(acc_p),
                    'effectiveness': float(eff), 'eff': float(eff),
                    'first_prob': float(first_prob),
                    'opp_counter_ev': float(opp_counter_ev),
                })

            moves_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # ---- Evaluate SWITCHES (delta vs staying) ----
        switches_eval: List[Dict[str, Any]] = []
        if legal_switches and opp_key:
            # Baseline if we stay
            stay_out = _our_best_on_target(state, my_key, opp_key, mi, legal_moves) if my_key else 0.0
            stay_threat = _opp_best_on_target(state, opp_key, my_key, mi) if my_key else 0.0
            # Clamp by remaining HP
            opp_now = _hp_frac(state.team.opponent[opp_key]) if opp_key else 1.0
            me_now  = _hp_frac(state.team.ours[my_key]) if my_key else 1.0
            stay_out = min(stay_out, opp_now)
            stay_threat = min(stay_threat, me_now)
            baseline_stay = stay_out - stay_threat

            # Evaluate each bench option
            my_species_active = None
            try:
                if my_key:
                    my_species_active = str(getattr(state.team.ours[my_key], "species", "") or "").lower()
            except Exception:
                pass

            for p in legal_switches:
                species = str(getattr(p, 'species', '') or '')
                if species.lower() == (my_species_active or ""):
                    continue

                # find matching TeamState key (non-active preferred)
                cand_key = None
                for k, ps in state.team.ours.items():
                    if str(getattr(ps, 'species', '')).lower() == species.lower() and not getattr(ps, "is_active", False):
                        cand_key = k; break
                if cand_key is None:
                    for k, ps in state.team.ours.items():
                        if str(getattr(ps, 'species', '')).lower() == species.lower():
                            cand_key = k; break
                if cand_key is None:
                    switches_eval.append({'species': species, 'score': 0.0, 'hp_fraction': float(getattr(p, 'current_hp_fraction', 1.0) or 1.0), 'note': 'no-key'})
                    continue

                # Hazard + incoming
                haz = apply_switch_in_effects(state, cand_key, "ally", mi, mutate=False)
                haz_frac = float(haz.get("fraction_lost") or 0.0)

                cand_hp_now = _hp_frac(state.team.ours[cand_key])
                incoming = _opp_best_on_target(state, opp_key, cand_key, mi)
                inc_on_switch = min(haz_frac + incoming, cand_hp_now)

                # Outgoing next turn from candidate's revealed moves (fallback to STAB if none)
                out_next = 0.0
                cand_ps = state.team.ours[cand_key]
                # revealed
                for mslot in (cand_ps.moves or []):
                    mid = getattr(mslot, "id", None) or getattr(mslot, "move_id", None)
                    if not mid: continue
                    cat = str(getattr(mslot, "category", "") or "").lower()
                    bp  = int(getattr(mslot, "base_power", 0) or getattr(mslot, "basePower", 0) or 0)
                    if cat == "status" or bp <= 0:
                        continue
                    frac, _ = _expected_damage_fraction(state, cand_key, opp_key, str(mid), mi)
                    out_next = max(out_next, frac * _acc_to_prob(getattr(mslot, "accuracy", None)))
                # fallback
                if out_next == 0.0:
                    for t in (cand_ps.types or []):
                        for mid in _COMMON_STAB.get((t or "").lower(), [])[:2]:
                            try:
                                frac, _ = _expected_damage_fraction(state, cand_key, opp_key, mid, mi)
                                raw = mi.raw(mid) or {}
                                out_next = max(out_next, frac * _acc_to_prob(raw.get("accuracy", None)))
                            except Exception:
                                continue

                out_next = min(out_next, opp_now)

                base_score = (out_next - inc_on_switch) - baseline_stay
                switches_eval.append({
                    'species': species,
                    'score': float(base_score),       # pure delta; no type bonus added
                    'base_score': float(base_score),
                    'outgoing_frac': float(out_next),
                    'incoming_on_switch': float(inc_on_switch),
                    'hazards_frac': float(haz_frac),
                    'hp_fraction': float(cand_hp_now),
                })

            switches_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # ---- Decide ----
        snap = snapshot_battle(battle)
        best_move = moves_eval[0] if moves_eval else None
        best_switch = switches_eval[0] if switches_eval else None

        # Forced switch
        if force_switch and best_switch:
            return ChosenAction(kind='switch', switch_species=best_switch['species'],
                                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                                       'picked': {'kind': 'switch', **best_switch}})

        # Hysteresis to avoid thrash
        MARGIN = 0.05
        if best_move and (not best_switch or best_move['score'] >= best_switch['score'] + MARGIN):
            return ChosenAction(kind='move', move_id=str(best_move['id']),
                                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                                       'order': {'p_user_first': float(best_move.get('first_prob', 0.5))},
                                       'picked': {'kind': 'move', **best_move}})

        if best_switch:
            return ChosenAction(kind='switch', switch_species=best_switch['species'],
                                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                                       'picked': {'kind': 'switch', **best_switch}})

        # Fallbacks
        if legal_moves:
            return ChosenAction(kind='move', move_id=str(getattr(legal_moves[0], 'id', '')),
                                debug={'snapshot': snap, 'fallback': True})
        if legal_switches:
            # pick first non-active bench
            cursp = ""
            try:
                cur = getattr(battle, "active_pokemon", None)
                cursp = (getattr(cur, "species", None) or "").lower()
            except Exception:
                pass
            for p in legal_switches:
                sp = (str(getattr(p, 'species', '')).lower())
                if sp != cursp:
                    return ChosenAction(kind='switch', switch_species=str(getattr(p, 'species', '')), debug={'snapshot': snap, 'fallback': True})
            return ChosenAction(kind='switch', switch_species=str(getattr(legal_switches[0], 'species', '')), debug={'snapshot': snap, 'fallback': True})

        return ChosenAction(kind='move', move_id='struggle', debug={'snapshot': snap, 'fallback': True})

# -------------- Poke-env Player wrapper -----------------------------------
try:
    from poke_env.player.player import Player  # type: ignore
except Exception:
    Player = object  # fallback for type checking

class StockfishPokeEnvPlayer(Player):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        # Hook for UI
        self.on_think_hook = kwargs.pop('on_think', None)
        engine_depth = kwargs.pop('engine_depth', None)
        self.engine = kwargs.pop('engine', None) or StockfishModel(kwargs.get('battle_format', 'gen9ou'))
        try:
            self.engine.set_depth(int(engine_depth))
        except Exception:
            pass
        super().__init__(*args, **kwargs)

        try:
            self._psclient = self.ps_client
        except Exception:
            pass

    # Convenience controls used by UI
    async def forfeit_all(self):
        try:
            battles = dict(getattr(self, 'battles', {}) or {})
            client = getattr(self, 'ps_client', None)
            if not client:
                return
            for _id, b in battles.items():
                try:
                    tag = getattr(b, 'battle_tag', getattr(b, 'room_id', None)) or _id
                    await client.send_message('/forfeit', room=str(tag))
                except Exception:
                    continue
        except Exception:
            pass

    async def timer_all(self, on: bool):
        try:
            battles = dict(getattr(self, 'battles', {}) or {})
            client = getattr(self, 'ps_client', None)
            if not client:
                return
            msg = '/timer on' if on else '/timer off'
            for _id, b in battles.items():
                try:
                    tag = getattr(b, 'battle_tag', getattr(b, 'room_id', None)) or _id
                    await client.send_message(msg, room=str(tag))
                except Exception:
                    continue
        except Exception:
            pass

    def choose_move(self, battle):
        decision = self.engine.choose_action(battle)
        try:
            if self.on_think_hook and isinstance(decision.debug, dict):
                dd = dict(decision.debug)
                dd.setdefault('snapshot', snapshot_battle(battle))
                # Add battle_tag + turn for telemetry consumers
                try:
                    dd['battle_tag'] = getattr(battle, 'battle_tag', getattr(battle, 'room_id', None))
                    dd['turn'] = getattr(battle, 'turn', None)
                except Exception:
                    pass
                import json, logging
                logging.getLogger('Think').info('UI_THINK turn=%s payload=%s', getattr(battle, 'turn', None), json.dumps(dd, default=str))
                self.on_think_hook(battle, dd)
        except Exception:
            pass

        try:
            if decision.kind == 'move' and decision.move_id:
                for m in (getattr(battle, 'available_moves', []) or []):
                    if str(getattr(m, 'id', '')) == str(decision.move_id):
                        return self.create_order(m)
                return self.create_order((getattr(battle, 'available_moves', []) or [None])[0]) or self.choose_random_move(battle)
            elif decision.kind == 'switch' and decision.switch_species:
                try:
                    cur = getattr(battle, "active_pokemon", None)
                    cursp = (getattr(cur, "species", None) or "").lower()
                except Exception:
                    cursp = ""
                for p in (getattr(battle, 'available_switches', []) or []):
                    sp = (str(getattr(p, 'species', '')).lower())
                    if sp == cursp:
                        continue
                    if sp == str(decision.switch_species).lower():
                        return self.create_order(p)
                for p in (getattr(battle, 'available_switches', []) or []):
                    if (str(getattr(p, 'species', '')).lower()) != cursp:
                        return self.create_order(p)
                return self.choose_random_move(battle)
            return self.choose_random_move(battle)
        except Exception:
            return self.choose_random_move(battle)
"""
UI_MAIN_NEW = r"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

# Ensure repo root is on sys.path so `Data` and `Models` can be imported
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -- Optional runtime patch to the model (accept on_think, set engine_depth, robust forfeit)
try:
    from UI.ui_think_patch import patch_stockfish_player  # noqa: F401
except Exception:  # patch file might not be present yet
    patch_stockfish_player = None  # type: ignore

from Models.stockfish_model import StockfishPokeEnvPlayer  # type: ignore
from Data.poke_env_battle_environment import snapshot as snapshot_battle  # type: ignore
from poke_env.ps_client.account_configuration import AccountConfiguration  # type: ignore
from poke_env.ps_client.server_configuration import (  # type: ignore
    ShowdownServerConfiguration,
    LocalhostServerConfiguration,
)

# ----------------------- Formats dropdown ---------------------------------
KNOWN_FORMATS = [
    "gen9randombattle",
    "gen9unratedrandombattle",
    "gen9randomdoublesbattle",
    "gen9hackmonscup",
    "gen9ou", "gen9ubers", "gen9uu", "gen9ru", "gen9nu", "gen9pu", "gen9lc", "gen9monotype",
    "gen9doublesou",
    "vgc2025regh",
]

# --------------------------- Small helpers --------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self.q = q
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.q.put_nowait(msg)
        except Exception:
            pass

def pretty_boosts(boosts: Dict[str, int] | None) -> str:
    if not boosts:
        return ""
    ordered = ["atk", "def", "spa", "spd", "spe", "acc", "evasion"]
    return ", ".join(f"{k}+{v}" if v > 0 else f"{k}{v}" for k, v in ((k, boosts.get(k, 0)) for k in ordered) if v != 0)

# --------------------------------- Window ---------------------------------
class StockfishWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, username: str, password: Optional[str], server_mode: str,
                 custom_ws: Optional[str], battle_format: str):
        super().__init__(parent)
        self.title("PokeCHAD — Stockfish Model")
        self.geometry("1180x740")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.username = username
        self.password = password
        self.server_mode = server_mode
        self.custom_ws = custom_ws
        self.battle_format = battle_format

        # Telemetry file
        os.makedirs("logs", exist_ok=True)
        self._telemetry_path = os.path.join("logs", f"telemetry_{os.getpid()}.jsonl")

        # Async loop thread
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Player
        self.player: Optional[StockfishPokeEnvPlayer] = None

        # Logging pane & handler
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # UI State
        self._scheduled_tasks: List[str] = []
        self._latest_think: Dict[str, Any] = {}
        self._latest_snapshot: Dict[str, Any] = {}
        self._last_fallback_turn: Optional[int] = None
        self._last_real_think_turn: Optional[int] = None
        self._root_log_handler_attached = False

        self._build_ui()
        self._pump_logs()

        # Bootstrap: connect immediately
        self._submit(self._async_connect())

    def _submit(self, coro: "asyncio.coroutines.Coroutine[Any, Any, Any]"):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        def handle_result():
            try:
                future.result(timeout=0.1)
            except asyncio.TimeoutError:
                self.after(100, handle_result)
            except Exception as e:
                error_msg = f"Action failed: {e}"
                self._append_log(error_msg)
                messagebox.showerror("Error", error_msg)
        self.after(100, handle_result)

    def _call_on_main(self, fn, delay_ms: int = 0):
        if delay_ms <= 0:
            return self.after(0, fn)
        return self.after(delay_ms, fn)

    # ---------- UI construction ----------
    def _build_ui(self):
        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True)

        # --- Dashboard tab
        dash = ttk.Frame(nb); nb.add(dash, text="Dashboard")
        controls = ttk.LabelFrame(dash, text="Controls"); controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(controls, text=f"User: {self.username}").pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Format:").pack(side=tk.LEFT, padx=(14, 2))
        self.format_var = tk.StringVar(value=self.battle_format)
        self.format_combo = ttk.Combobox(controls, textvariable=self.format_var, values=KNOWN_FORMATS, width=26, state="readonly")
        self.format_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Depth:").pack(side=tk.LEFT, padx=(14, 2))
        self.depth_var = tk.IntVar(value=1)
        self.depth_spin = ttk.Spinbox(controls, from_=1, to=3, textvariable=self.depth_var, width=4, command=self._on_depth_changed)
        self.depth_spin.pack(side=tk.LEFT, padx=4)

        ttk.Button(controls, text="Ladder 1", command=lambda: self._submit(self._ladder(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Challenge…", command=self._challenge_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Accept 1", command=lambda: self._submit(self._accept(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Start Timer", command=lambda: self._submit(self._timer_all(True))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Forfeit", command=lambda: self._submit(self._forfeit_all())).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Train Weights", command=self._train_weights).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reload Weights", command=self._reload_weights).pack(side=tk.LEFT, padx=4)

        # Team panes
        teams = ttk.Frame(dash); teams.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.team_tree = self._make_team_tree(teams, "Your team")
        self.opp_tree = self._make_team_tree(teams, "Opponent team")

        # --- Thinking tab
        think_tab = ttk.Frame(nb); nb.add(think_tab, text="Thinking")
        self.cand_tree = self._make_cand_tree(think_tab, "Move candidates")
        self.switch_tree = self._make_switch_tree(think_tab, "Switch candidates")

        # --- Logs tab
        logs_tab = ttk.Frame(nb); nb.add(logs_tab, text="Logs")
        self.logs_text = tk.Text(logs_tab, height=20, wrap="word")
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _make_team_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("slot", "species", "hp", "status", "boosts")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (60, 160, 60, 80, 220)):
            tree.heading(c, text=c.upper())
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_cand_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("move", "score", "exp_dmg", "acc", "eff", "first", "opp", "note")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        hdrs = ("MOVE", "SCORE", "EXP", "ACC", "EFF", "FIRST", "OPP", "WHY/NOTE")
        widths = (200, 80, 70, 60, 60, 60, 60, 260)
        for c, h, w in zip(cols, hdrs, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_switch_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("species", "score", "base", "out", "in", "haz", "hp")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        headers = ("SPECIES", "SCORE", "BASE", "OUT", "IN", "HAZ", "HP")
        widths = (140, 70, 70, 60, 60, 50, 60)
        for c, h, w in zip(cols, headers, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    # ---------- Connect ----------
    async def _async_connect(self):
        try:
            if patch_stockfish_player:
                patch_stockfish_player()
        except Exception as e:
            self._append_log(f"Model patch failed (non-fatal): {e}")

        account = AccountConfiguration(self.username, self.password)
        if self.server_mode == "Showdown":
            server = ShowdownServerConfiguration
        elif self.server_mode == "Localhost":
            server = LocalhostServerConfiguration
        else:
            server = ShowdownServerConfiguration
            if self.custom_ws:
                try:
                    server = type("CustomServerConf", (tuple,), {})((self.custom_ws, ShowdownServerConfiguration[1]))
                except Exception:
                    server = ShowdownServerConfiguration

        common = dict(
            account_configuration=account,
            server_configuration=server,
            battle_format=self.battle_format,
            log_level=logging.INFO,
        )

        try:
            depth_val = int(self.depth_var.get())
        except Exception:
            depth_val = None

        player = None; last_error = None
        sigs = [
            dict(on_think=self._on_think, engine_depth=depth_val, start_listening=True),
            dict(on_think=self._on_think, engine_depth=depth_val),
            dict(on_think=self._on_think),
            dict(engine_depth=depth_val, start_listening=True),
            dict(engine_depth=depth_val),
            dict(),
        ]
        for extra in sigs:
            try:
                kv = {k: v for k, v in extra.items() if v is not None}
                player = StockfishPokeEnvPlayer(**common, **kv); break
            except TypeError as e:
                last_error = e; continue
        if player is None:
            self._append_log(f"Failed to construct player with extended kwargs; retrying minimal. Last error: {last_error}")
            player = StockfishPokeEnvPlayer(**common)

        self.player = player

        try:
            attached = False
            for attr in ("on_think", "think_callback", "on_think_hook"):
                if hasattr(self.player, attr):
                    setattr(self.player, attr, self._on_think)
                    attached = True; break
            if not attached:
                for meth in ("set_on_think", "set_think_callback", "register_think_callback", "on_think_connect"):
                    fn = getattr(self.player, meth, None)
                    if callable(fn):
                        fn(self._on_think); attached = True; break
            if not attached:
                self._append_log("Note: model exposes no on_think hook; Thinking tab will use fallback per turn.")
        except Exception as e:
            self._append_log(f"Could not attach think callback: {e}")

        try:
            self.player.logger.addHandler(self.log_handler)
            self.player.logger.setLevel(logging.INFO)
        except Exception:
            pass
        try:
            if not getattr(self, "_root_log_handler_attached", False):
                root = logging.getLogger()
                root.addHandler(self.log_handler)
                if root.level > logging.INFO:
                    root.setLevel(logging.INFO)
                self._root_log_handler_attached = True
        except Exception:
            pass

        await self.player.ps_client.wait_for_login()
        self._append_log("Login confirmed. Ready.")
        self._call_on_main(self._poll_battle)

    # ---------- Actions ----------
    async def _ladder(self, n: int):
        if not self.player: return
        if getattr(self.player, "format", None) != self.format_var.get():
            try: self.player.format = self.format_var.get()
            except Exception: pass
        self._append_log(f"Starting ladder: {n} game(s)…")
        await self.player.ladder(n)

    async def _accept(self, n: int):
        if not self.player: return
        self._append_log(f"Accepting {n} challenge(s)…")
        await self.player.accept_challenges(opponent=None, n_challenges=n)

    def _challenge_dialog(self):
        if not self.player: return
        dlg = tk.Toplevel(self); dlg.title("Challenge a user")
        ttk.Label(dlg, text="Opponent username:").pack(side=tk.TOP, padx=8, pady=8)
        name_var = tk.StringVar(); ttk.Entry(dlg, textvariable=name_var, width=28).pack(side=tk.TOP, padx=8, pady=(0, 8))
        def go():
            opp = name_var.get().strip()
            if opp: self._submit(self.player.send_challenges(opp, n_challenges=1))
            dlg.destroy()
        ttk.Button(dlg, text="Challenge", command=go).pack(side=tk.TOP, padx=8, pady=8)

    async def _forfeit_all(self):
        p = self.player
        if not p: return
        try:
            m = getattr(p, "forfeit_all", None)
            if callable(m):
                await m(); self._append_log("Called player.forfeit_all()."); return
        except Exception as e:
            self._append_log(f"player.forfeit_all() failed: {e} — falling back to direct /forfeit.")
        try:
            client = getattr(p, "ps_client", None) or getattr(p, "_client", None)
            if not client: raise RuntimeError("PSClient missing on player")
            battles = getattr(p, "battles", {}) or {}
            rooms: List[str] = []
            for key, battle in list(battles.items()):
                room_id = getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None) or str(key)
                if room_id: rooms.append(room_id)
            if not rooms:
                self._append_log("No active battle rooms found for /forfeit."); return
            sent = 0
            for r in rooms:
                try:
                    await client.send_message("/forfeit", room=r); sent += 1
                except Exception as e2:
                    self._append_log(f"Failed to send /forfeit to {r}: {e2}")
            self._append_log(f"Sent /forfeit to {sent} room(s).")
        except Exception as e:
            self._append_log(f"Forfeit fallback failed: {e}")

    async def _timer_all(self, on: bool):
        if self.player: await self.player.timer_all(on)

    def _on_depth_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_depth"):
                    eng.set_depth(int(self.depth_var.get()))
                elif hasattr(self.player, "set_depth"):
                    self.player.set_depth(int(self.depth_var.get()))
            except Exception:
                pass

    # ---------- Train / Reload ----------
    def _train_weights(self):
        def run():
            try:
                cmd = [sys.executable, "tools/train_launcher.py"]
                self._append_log("[run] " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout: self._append_log(proc.stdout.strip())
                if proc.returncode != 0:
                    if proc.stderr: self._append_log(proc.stderr.strip())
                    messagebox.showerror("Training failed", proc.stderr or "trainer returned non-zero")
                else:
                    self._append_log("[ok] Training finished.")
            except Exception as e:
                self._append_log(f"Training failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _reload_weights(self):
        try:
            eng = getattr(self.player, "engine", None)
            if eng and hasattr(eng, "reload_weights"):
                eng.reload_weights()
                messagebox.showinfo("Weights", "Weights reloaded from Models/weights.json")
            else:
                messagebox.showwarning("Weights", "Engine does not expose reload_weights()")
        except Exception as e:
            messagebox.showerror("Weights", f"Reload failed: {e}")

    # ---------- Think data from model ----------
    def _on_think(self, battle, think: Dict[str, Any]):
        self._latest_think = think or {}
        try: snap = think.get("snapshot")
        except Exception: snap = None
        self._latest_snapshot = snap or snapshot_battle(battle)
        try:
            self._last_real_think_turn = int(self._latest_snapshot.get("turn")) if self._latest_snapshot else None
        except Exception:
            pass
        # write JSONL telemetry if there is a picked decision
        try:
            if think.get("picked"):
                entry = {
                    "battle_tag": think.get("battle_tag") or getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None),
                    "turn": think.get("turn") or self._latest_snapshot.get("turn"),
                    "picked": think.get("picked"),
                    "order": think.get("order"),
                    "snapshot": self._latest_snapshot,
                }
                with open(self._telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self._append_log(f"telemetry write failed: {e}")

        self._call_on_main(self._refresh_thinking)
        self._call_on_main(self._refresh_teams)

    def _refresh_thinking(self):
        if not self.winfo_exists(): return
        # Candidates
        self._reload_tree(self.cand_tree)
        for d in self._latest_think.get("candidates", []):
            try:
                move = d.get("name") or d.get("id") or d.get("move") or d.get("move_id")
                s = d.get("score"); score = f"{float(s):.2f}" if s is not None else ""
                exp = ""; 
                for k in ("exp_dmg", "expected", "exp", "expdmg", "expected_damage"):
                    if d.get(k) is not None:
                        exp = f"{float(d.get(k)):.2f}"; break
                acc = ""; 
                for k in ("acc", "acc_mult", "accuracy", "hit_chance"):
                    if d.get(k) is not None:
                        acc = f"{float(d.get(k)):.2f}"; break
                eff = ""; 
                for k in ("eff", "effectiveness", "type_mult", "type_effectiveness"):
                    if d.get(k) is not None:
                        eff = f"{float(d.get(k)):.2f}"; break
                first = ""; 
                if d.get("first_prob") is not None: first = f"{float(d.get('first_prob')):.2f}"
                opp = ""; 
                if d.get("opp_counter_ev") is not None: opp = f"{float(d.get('opp_counter_ev')):.2f}"
                note = d.get("why_blocked") or d.get("note") or d.get("why") or ""
                self.cand_tree.insert("", tk.END, values=(move, score, exp, acc, eff, first, opp, note))
            except Exception:
                try:
                    self.cand_tree.insert("", tk.END, values=(str(d), "", "", "", "", "", "", ""))
                except Exception:
                    pass

        # Switches
        self._reload_tree(self.switch_tree)
        for d in self._latest_think.get("switches", []):
            try:
                species = d.get("species") or d.get("name") or d.get("id")
                s = d.get("score"); score = f"{float(s):.2f}" if s is not None else ""
                base = d.get("base_score"); base_s = f"{float(base):.2f}" if base is not None else ""
                out = d.get("outgoing_frac"); out_s = f"{float(out):.2f}" if out is not None else ""
                incoming = d.get("incoming_on_switch"); in_s = f"{float(incoming):.2f}" if incoming is not None else ""
                haz = d.get("hazards_frac"); haz_s = f"{float(haz):.2f}" if haz is not None else ""
                hp = d.get("hp_fraction"); hp_s = f"{int(round(float(hp) * 100))}%" if isinstance(hp, (int, float)) else ""
                self.switch_tree.insert("", tk.END, values=(species, score, base_s, out_s, in_s, haz_s, hp_s))
            except Exception:
                try:
                    species = d.get("species") or str(d)
                    score = f"{float(d.get('score', 0)):.2f}" if d.get('score') is not None else "0.00"
                    self.switch_tree.insert("", tk.END, values=(species, score, "", "", "", "", ""))
                except Exception:
                    pass

    def _refresh_teams(self):
        if not self.winfo_exists(): return
        snap = self._latest_snapshot or {}
        self._reload_tree(self.team_tree)
        for sid, p in (snap.get("my_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            self.team_tree.insert("", tk.END, values=(sid, p.get("species"), hp_s, str(p.get("status") or ""), boosts))
        self._reload_tree(self.opp_tree)
        for sid, p in (snap.get("opp_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            self.opp_tree.insert("", tk.END, values=(sid, p.get("species"), hp_s, str(p.get("status") or ""), boosts))

    def _reload_tree(self, tree: ttk.Treeview):
        try:
            for iid in tree.get_children():
                tree.delete(iid)
        except tk.TclError:
            pass

    # ---------- Polling ----------
    def _find_active_battle(self):
        p = getattr(self, "player", None)
        if not p: return None
        for name in ("current_battle", "battle", "active_battle"):
            b = getattr(p, name, None)
            if b is not None: return b
        battles = getattr(p, "battles", None)
        if isinstance(battles, dict) and battles:
            try:
                for b in battles.values():
                    if getattr(b, "active_pokemon", None) is not None or getattr(b, "turn", None):
                        return b
                return list(battles.values())[-1]
            except Exception:
                try: return next(iter(battles.values()))
                except Exception: return None
        return None

    def _poll_battle(self):
        try:
            if not self.winfo_exists(): return
            b = self._find_active_battle()
            if b is not None:
                try: snap = snapshot_battle(b); self._latest_snapshot = snap
                except Exception: snap = None
                try: self._refresh_teams()
                except Exception: pass
                try:
                    turn = int(snap.get("turn")) if snap else None
                except Exception:
                    turn = None
                if turn is not None and turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                    self._emit_fallback_think(b, snap); self._last_fallback_turn = turn
        finally:
            if self.winfo_exists():
                try: h = self.after(500, self._poll_battle); self._scheduled_tasks.append(h)
                except Exception: pass

    def _emit_fallback_think(self, battle, snap: Optional[Dict[str, Any]]):
        try:
            cands = []
            for m in (getattr(battle, "available_moves", None) or []):
                try:
                    name = getattr(m, "name", None) or getattr(m, "id", None) or str(m)
                    bp = getattr(m, "base_power", None) or getattr(m, "basePower", None) or 0
                    acc = getattr(m, "accuracy", None)
                    if acc is True: acc_val = 1.0
                    elif isinstance(acc, (int, float)): acc_val = float(acc) / (100.0 if acc > 1 else 1.0)
                    else: acc_val = 1.0
                    expected = float(bp or 0) * float(acc_val)
                    cands.append({"name": name, "score": expected, "exp_dmg": expected, "acc": acc_val, "eff": "", "note": "synthetic"})
                except Exception: pass

            switches = []
            for pkm in (getattr(battle, "available_switches", None) or []):
                try:
                    species = getattr(pkm, "species", None) or getattr(pkm, "name", None) or str(pkm)
                    hp_frac = getattr(pkm, "hp_fraction", None) or getattr(pkm, "current_hp_fraction", None)
                    switches.append({"species": species, "score": float(hp_frac or 0.0), "hp_fraction": float(hp_frac or 0.0)})
                except Exception: pass

            think = {"candidates": sorted(cands, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "switches": sorted(switches, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "snapshot": snap or snapshot_battle(battle)}
            self._on_think(battle, think)
        except Exception as e:
            self._append_log(f"Fallback think failed: {e}")

    # ---------- Logs ----------
    def _append_log(self, msg: str):
        try:
            self.logs_text.insert(tk.END, msg + "\n")
            self.logs_text.see(tk.END)
        except Exception:
            pass

    def _pump_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        if self.winfo_exists():
            h = self.after(200, self._pump_logs)
            self._scheduled_tasks.append(h)

    # ---------- Shutdown ----------
    def _on_close(self):
        for h in self._scheduled_tasks:
            try: self.after_cancel(h)
            except Exception: pass
        self._scheduled_tasks.clear()
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        self.destroy()

def launch_stockfish_window(root: tk.Tk, username: str, password: Optional[str],
                            server_mode: str, custom_ws: Optional[str], battle_format: str) -> StockfishWindow:
    return StockfishWindow(root, username=username, password=password,
                           server_mode=server_mode, custom_ws=custom_ws, battle_format=battle_format)
"""
UI_PATCH_NEW = r"""# UI/ui_think_patch.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from Data.battle_runtime import get_state, estimate_damage, apply_switch_in_effects
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo
from Data import battle_helper as _BH  # we'll patch its type_effectiveness

# ------------------------- Type chart fix (monkey patch) -------------------------
# Showdown typechart "damageTaken" codes: 0=Neutral, 1=Weak (2x), 2=Resist (0.5x), 3=Immune (0x).
# Some charts include non-type keys (e.g., "prankster", "sandstorm"); ignore them.
_PS_CODE_TO_MULT = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

# Canonical list of offensive/defensive types for Gen 9 (upper-case)
_CANON_TYPES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK",
    "STEEL", "FAIRY",  # Stellar is special/neutral; treat as 1.0 if present
]

def _norm_type_name(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = str(t).replace("-", "").replace("_", "").strip().upper()
    # Map quirky names to canonical forms if needed
    aliases = {
        "PSY": "PSYCHIC",
        "ELECTR": "ELECTRIC",
        "FIRE": "FIRE",
        "WATR": "WATER",
        "STELLAR": "STELLAR",  # neutral handling below
    }
    return aliases.get(t, t)

def _tc_to_attack_matrix(tc: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    \"\"\"
    Accepts either a PS-style chart (def_type -> {damageTaken:{atk_type:code}})
    or an already-built attack matrix (atk -> def -> mult).
    Returns attack -> defense -> multiplier (floats).
    \"\"\"
    # Fast path: looks like attack->defense->mult already
    try:
        some_k = next(iter(tc))
        some_v = tc[some_k]
        if isinstance(some_v, dict) and all(isinstance(v, (int, float)) for v in some_v.values()):
            # Normalize keys to UPPER
            out: Dict[str, Dict[str, float]] = {}
            for atk, row in tc.items():
                A = _norm_type_name(atk)
                if not A:
                    continue
                out[A] = {}
                for dfd, mult in row.items():
                    D = _norm_type_name(dfd)
                    if not D:
                        continue
                    out[A][D] = float(mult)
            return out
    except Exception:
        pass

    # Otherwise interpret as PS-style defense rows
    atk_mat: Dict[str, Dict[str, float]] = {T: {} for T in _CANON_TYPES}
    # Gather all defense types present
    for def_t, row in (tc or {}).items():
        D = _norm_type_name(def_t)
        if not D:
            continue
        if D not in _CANON_TYPES:
            # treat unknown/auxiliary (e.g. PRANKSTER) as non-type; skip
            continue
        # Showdown rows are like {"damageTaken": {"Fire":2,"Water":1,...}, ...}
        if isinstance(row, dict) and "damageTaken" in row and isinstance(row["damageTaken"], dict):
            dmg_row = row["damageTaken"]
        else:
            dmg_row = row if isinstance(row, dict) else {}
        for atk_t, code in (dmg_row or {}).items():
            A = _norm_type_name(atk_t)
            if not A:
                continue
            if A not in _CANON_TYPES:
                # e.g., "prankster","sandstorm" etc.
                continue
            mult = _PS_CODE_TO_MULT.get(int(code), 1.0)
            atk_mat.setdefault(A, {})[D] = mult

    # Fill unspecified entries as neutral (1.0)
    for A in list(atk_mat.keys()) or _CANON_TYPES:
        for D in _CANON_TYPES:
            if D not in atk_mat[A]:
                atk_mat[A][D] = 1.0

    return atk_mat

def _fixed_type_effectiveness(atk_type: str, dfd_types: List[str] | tuple[str, ...] | None, tc: Dict[str, Any]) -> float:
    \"\"\"
    Replacement for Data.battle_helper.type_effectiveness that is robust to
    PS-style charts and returns the correct product across dual types.
    \"\"\"
    if not atk_type:
        return 1.0
    A = _norm_type_name(atk_type)
    if not A:
        return 1.0

    # Build (and cache) an attack matrix on the tc object
    mat = getattr(tc, "_atk_mat", None)
    if mat is None:
        try:
            mat = _tc_to_attack_matrix(tc)
        except Exception:
            mat = {}
        # tack on a neutral row for STELLAR if present anywhere
        if "STELLAR" in (tc.keys() if isinstance(tc, dict) else []):
            mat["STELLAR"] = {D: 1.0 for D in _CANON_TYPES}
        setattr(tc, "_atk_mat", mat)

    if not dfd_types:
        return 1.0

    # Multiply across up to two defender types; any immunity (0) zeroes the product
    eff = 1.0
    for t in dfd_types:
        D = _norm_type_name(t)
        if not D:
            continue
        mult = float(mat.get(A, {}).get(D, 1.0))
        if mult == 0.0:
            return 0.0
        eff *= mult
    return eff

# Apply the monkey patch once
if not getattr(_BH, "_ui_tc_patch_applied", False):
    try:
        _BH.type_effectiveness = _fixed_type_effectiveness  # type: ignore
        _BH._ui_tc_patch_applied = True  # type: ignore
    except Exception:
        # non-fatal: UI parts will still use _fixed_type_effectiveness directly if needed
        pass


# ------------------------------ Local helpers ------------------------------
def _hp_frac(ps) -> float:
    try:
        if ps.max_hp:
            return max(0.0, min(1.0, (ps.current_hp or ps.max_hp) / ps.max_hp))
    except Exception:
        pass
    return 1.0

def _expected_damage_fraction(state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo) -> float:
    try:
        res = estimate_damage(state, atk_key, dfd_key, move_id, mi, is_critical=False)
        rolls = res.get("rolls") or []
        if not rolls:
            return 0.0
        dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
        max_hp = int(getattr(dfd, "max_hp", 0) or dfd.stats.raw.get("hp", 1) or 1)
        return float(sum(rolls)) / (len(rolls) * max_hp)
    except Exception:
        return 0.0

def _acc_to_prob(acc) -> float:
    if acc is True or acc is None:
        return 1.0
    if isinstance(acc, (int, float)):
        return float(acc) / (100.0 if acc > 1 else 1.0)
    return 1.0

def _opp_best_on_target(state, opp_key: str, target_key: str, mi: MovesInfo) -> float:
    \"\"\"Max expected fraction the opponent can deal to target with revealed moves (accuracy-weighted).\"\"\"
    opp = state.team.opponent[opp_key]
    best = 0.0
    for mv in opp.moves or []:
        if not mv or not mv.id or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
            continue
        frac = _expected_damage_fraction(state, opp_key, target_key, mv.id, mi)
        best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
    return best

def _our_best_on_target(state, my_key: str, opp_key: str, mi: MovesInfo) -> float:
    \"\"\"Best expected fraction we can deal to opponent this turn with our revealed moves (accuracy-weighted).\"\"\"
    me = state.team.ours[my_key]
    best = 0.0
    for mv in me.moves or []:
        if not mv or not mv.id or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
            continue
        frac = _expected_damage_fraction(state, my_key, opp_key, mv.id, mi)
        best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
    return best

def _typing_bonus(candidate, opponent, mi: MovesInfo) -> float:
    \"\"\"Small *display-only* bonus for better defensive typing vs opp's revealed attacking types.
    With extra for immunities. Range ~[0, 0.3]. Not added to the score; purely UI info.
    \"\"\"
    try:
        tc = mi.get_type_chart()
    except Exception:
        tc = {}
    cand_types = [(_norm_type_name(t) or "") for t in (candidate.types or []) if t]
    bonus = 0.0
    seen_offense_types = set()
    for mv in (opponent.moves or []):
        if not mv or not mv.type or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
            continue
        t = _norm_type_name(mv.type)
        if not t or t in seen_offense_types:
            continue
        seen_offense_types.add(t)
        eff = _fixed_type_effectiveness(t, cand_types, tc)
        if eff == 0:
            bonus += 0.2  # immunity => big bump
        elif eff < 1:
            bonus += 0.08 # resist => small bump
        elif eff > 1:
            bonus -= 0.05 # weak to => tiny penalty
    return max(0.0, bonus)


# ------------------------------ Main patch ------------------------------
def patch_stockfish_player():
    \"\"\"Monkey-patch choose_move to emit *weight-free* switch scores and fix type chart use.\"\"\"
    from Models.stockfish_model import StockfishPokeEnvPlayer

    if getattr(StockfishPokeEnvPlayer, "_ui_switch_patch_applied", False):
        return  # idempotent

    _orig_choose_move = StockfishPokeEnvPlayer.choose_move

    def patched_choose_move(self, battle):
        try:
            think = {}
            state = get_state(battle, ev_policy="auto")
            mi = MovesInfo(state.format or 9)

            # Identify my active + opponent active keys
            my_key = None
            for k, ps in state.team.ours.items():
                if getattr(ps, "is_active", False):
                    my_key = k; break
            opp_key = None
            for k, ps in state.team.opponent.items():
                if getattr(ps, "is_active", False):
                    opp_key = k; break

            switches_dbg = []

            # --- Baseline if we stay in (same unit, same horizon: net HP fraction this turn) ---
            stay_threat = 0.0   # expected fraction of our HP we lose if we stay
            stay_out    = 0.0   # expected fraction of opp HP we deal this turn (best move)
            opp_hp_now  = 1.0
            my_hp_now   = 1.0

            if my_key and opp_key:
                stay_threat = _opp_best_on_target(state, opp_key, my_key, mi)
                stay_out    = _our_best_on_target(state, my_key, opp_key, mi)
                opp_hp_now  = _hp_frac(state.team.opponent[opp_key])
                my_hp_now   = _hp_frac(state.team.ours[my_key])
                # Clamp to remaining HP so we don't "benefit" from overkill
                stay_threat = min(stay_threat, my_hp_now)
                stay_out    = min(stay_out,    opp_hp_now)

            baseline_stay = stay_out - stay_threat

            # --- Evaluate each legal switch against that baseline ---
            for p in (getattr(battle, "available_switches", None) or []):
                species = getattr(p, "species", None) or getattr(p, "name", None) or str(p)
                # Find candidate key in resolved state
                cand_key = None
                for k, ps in state.team.ours.items():
                    if str(ps.species or "").lower() == str(species or "").lower():
                        cand_key = k; break
                if cand_key is None:
                    continue

                cand = state.team.ours[cand_key]

                # Incoming on the switch: hazards + opponent's best hit on the incoming mon
                hazards = apply_switch_in_effects(state, cand_key, side="ally", mi=mi, mutate=False)
                hz_frac = float(hazards.get("fraction_lost", 0.0) or 0.0)
                inc_on_switch = hz_frac + (_opp_best_on_target(state, opp_key, cand_key, mi) if opp_key else 0.0)
                cand_hp_now   = _hp_frac(cand)
                inc_on_switch = min(inc_on_switch, cand_hp_now)  # can't lose more than current HP

                # Our pressure *next* turn from the candidate's revealed damaging moves
                out_next = 0.0
                if opp_key:
                    for mv in (cand.moves or []):
                        if not mv or not mv.id or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
                            continue
                        ef = _expected_damage_fraction(state, cand_key, opp_key, mv.id, mi)
                        out_next = max(out_next, ef * _acc_to_prob(getattr(mv, "accuracy", None)))

                out_next = min(out_next, opp_hp_now)  # cannot deal more than opponent's remaining HP

                # Defensive typing info (display-only; don't add to score)
                typ_bonus = 0.0
                try:
                    if opp_key:
                        typ_bonus = _typing_bonus(cand, state.team.opponent[opp_key], mi)
                except Exception:
                    pass

                # Weight-free, tempo-aware delta vs. staying:
                base_score = (out_next - inc_on_switch) - baseline_stay
                score = base_score  # keep pure; no extra weights added

                switches_dbg.append({
                    "species": species,
                    "score": float(score),
                    "base_score": float(base_score),
                    "outgoing_frac": float(out_next),
                    "incoming_on_switch": float(inc_on_switch),
                    "hazards_frac": float(hz_frac),
                    "type_bonus": float(typ_bonus),    # display only
                    "hp_fraction": float(cand_hp_now),
                })

            think["switches"] = sorted(switches_dbg, key=lambda d: d["score"], reverse=True)
            think["snapshot"] = snapshot_battle(battle)

            # Emit updated think packet for the UI
            cb = None
            for attr in ("on_think", "think_callback", "on_think_hook"):
                cb = getattr(self, attr, None)
                if callable(cb): break
            if cb:
                cb(battle, think)
        except Exception:
            # Swallow exceptions so we don't interfere with move choice
            pass

        # Always call the original to actually choose an action
        return _orig_choose_move(self, battle)

    # Install monkey patch
    StockfishPokeEnvPlayer.choose_move = patched_choose_move
    StockfishPokeEnvPlayer._ui_switch_patch_applied = True
"""

def find_path(cands):
    for p in cands:
        path = Path(p)
        if path.exists():
            return path
    return Path(cands[0])

def backup(path: Path):
    try:
        if path.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = path.with_suffix(path.suffix + ".bak." + ts)
            bak.parent.mkdir(parents=True, exist_ok=True)
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print("[backup]", str(path), "->", str(bak))
    except Exception as e:
        print("[warn] could not backup", str(path), ":", e)

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print("[ok] wrote", str(path), f"({len(content.splitlines())} lines)")

def main():
    model_path = find_path(MODEL_CANDIDATES)
    ui_main_path = find_path(UI_MAIN_CANDIDATES)
    ui_patch_path = find_path(UI_PATCH_CANDIDATES)

    print("[target] model:", model_path)
    print("[target] ui   :", ui_main_path)
    print("[target] patch:", ui_patch_path)

    backup(model_path); backup(ui_main_path); backup(ui_patch_path)

    write_file(model_path, MODEL_NEW)
    write_file(ui_main_path, UI_MAIN_NEW)
    write_file(ui_patch_path, UI_PATCH_NEW)

    print("\nDone. Notes:")
    print("  - Switching now uses delta vs staying with KO/HP clamps (no more switch->switch thrash).")
    print("  - Move scoring logs FIRST and OPP (opp_counter_ev) and uses learned weights if present.")
    print("  - Type chart: effectiveness comes from the damage pipeline; UI patch also normalizes PS charts.")
    print("  - UI writes logs/telemetry_*.jsonl and adds Train/Reload buttons.")
    print("\nNext: play a few turns, click Train Weights, then Reload Weights.")

if __name__ == "__main__":
    main()
