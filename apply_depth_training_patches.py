#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time
from pathlib import Path

MODEL_CANDIDATES = ['Models/stockfish_model.py', 'models/stockfish_model.py', 'stockfish_model.py', '/mnt/data/Models/stockfish_model.py', '/mnt/data/stockfish_model.py']
UI_CANDIDATES    = ['UI/tk_stockfish_model_ui.py', 'ui/tk_stockfish_model_ui.py', 'tk_stockfish_model_ui.py', '/mnt/data/UI/tk_stockfish_model_ui.py', '/mnt/data/tk_stockfish_model_ui.py']

MODEL_NEW = r"""from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- BEGIN SAFE TYPE CHART HELPER ---
_SAFE_GET_TYPE_CHART = True
def _safe_get_type_chart(mi):
    try:
        return _safe_get_type_chart(mi)  # recursive guard; always fails -> go to fallback
    except Exception:
        pass
    try:
        raw_tc = getattr(mi, "type_chart", None)
        if raw_tc is None:
            data = getattr(mi, "_data", None)
            raw_tc = getattr(data, "type_chart", None) if data is not None else None
    except Exception:
        raw_tc = None
    try:
        from Data.poke_env_moves_info import _normalize_showdown_typechart  # type: ignore
        return _normalize_showdown_typechart(raw_tc or {})
    except Exception:
        return raw_tc if isinstance(raw_tc, dict) else {}
# --- END SAFE TYPE CHART HELPER ---

from typing import Callable

# Import mechanics from the Data package
from Data.battle_runtime import (
    get_state,
    predict_order_for_ids,
    estimate_damage,
    would_fail,
    apply_switch_in_effects,
)
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo

# --- learned weights (optional) --------------------------------------------
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

# ---------------- helpers -----------------
def _acc_to_prob(acc) -> float:
    if acc is True or acc is None:
        return 1.0
    try:
        x = float(acc)
        return x / 100.0 if x > 1.0 else max(0.0, min(1.0, x))
    except Exception:
        return 1.0

def _type_mult(att_type: str, defender_types, chart: Dict[str, Dict[str, float]]) -> float:
    if not chart or not att_type or not defender_types:
        return 1.0
    chart_keys = set(chart.keys())
    att_type_norm = None
    for key in chart_keys:
        if key.upper() == str(att_type).upper():
            att_type_norm = key
            break
    if not att_type_norm:
        return 1.0
    mult = 1.0
    for def_type in defender_types:
        if not def_type:
            continue
        def_type_norm = None
        for target_key in chart[att_type_norm]:
            if target_key.upper() == str(def_type).upper():
                def_type_norm = target_key
                break
        if def_type_norm:
            effectiveness = chart[att_type_norm][def_type_norm]
            mult *= float(effectiveness)
    return mult

def _hp_frac(ps) -> float:
    try:
        mx = float(getattr(ps, "max_hp", 0) or ps.stats.raw.get("hp", 1) or 1)
        cur = float(getattr(ps, "current_hp", None) or mx)
        return max(0.0, min(1.0, cur / mx))
    except Exception:
        return 1.0

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
        self._depth = 3            # default analysis depth (used now!)
        self._gamma = 0.9          # per-ply discount for deeper horizons

    # exposed for UI
    def set_depth(self, d: int):
        self._depth = max(1, int(d))

    # hot-reload weights after training
    def reload_weights(self, path: str = os.path.join("Models", "weights.json")):
        self._W = _load_weights(path)

    def _opp_best_incoming_frac(self, state, opp_key: str, my_key: str, mi: MovesInfo, default_opp_move: str, opp_moves_known: List[str]) -> float:
        my_ps = state.team.ours[my_key]
        my_max = float(getattr(my_ps, 'max_hp', 0) or my_ps.stats.raw.get('hp', 1) or 1)
        my_now = _hp_frac(my_ps)
        worst = 0.0
        test_moves = list(opp_moves_known) if opp_moves_known else [default_opp_move]
        for om in test_moves:
            try:
                d = estimate_damage(state, opp_key, my_key, str(om), mi)
                rr = d.get('rolls', []) or []
                if rr:
                    worst = max(worst, sum(rr)/len(rr))
            except Exception:
                continue
        incoming = worst / my_max if my_max > 0 else 0.0
        return min(incoming, my_now)

    def _our_best_outgoing_frac(self, state, my_key: str, opp_key: str, mi: MovesInfo) -> float:
        me = state.team.ours[my_key]
        opp_ps = state.team.opponent[opp_key]
        opp_max = float(getattr(opp_ps, 'max_hp', 0) or opp_ps.stats.raw.get('hp', 1) or 1)
        best = 0.0
        for mv in (me.moves or []):
            mid = getattr(mv, "id", None) or getattr(mv, "move_id", None)
            if not mid:
                continue
            try:
                d = estimate_damage(state, my_key, opp_key, str(mid), mi)
                rr = d.get("rolls", []) or []
                if rr:
                    avg = sum(rr)/len(rr)
                    acc = _acc_to_prob(getattr(mv, "accuracy", 1.0))
                    best = max(best, (avg / opp_max) * acc)
            except Exception:
                continue
        opp_now = _hp_frac(opp_ps)
        return min(best, opp_now)

    def _evaluate_move_with_depth(
        self,
        state, my_key: str, opp_key: str, mv, mi: MovesInfo,
        type_chart: Dict[str, Dict[str, float]],
        opp_moves_known: List[str], default_opp_move: str, depth: int
    ) -> Dict[str, Any]:
        \"\"\"Return dict with d1/d2/d3 scores + training features for a single move.\"\"\"
        W = self._W
        # ids/names
        mid = getattr(mv, 'id', None) or getattr(mv, 'move_id', None)
        name = getattr(mv, 'name', str(mid))
        if not mid:
            return {
                'id': '', 'name': name, 'score': 0.0, 'expected': 0.0, 'acc': 0.0,
                'effectiveness': 1.0, 'first_prob': 0.5, 'opp_counter_ev': 0.0,
                'score_d1': 0.0, 'score_d2': 0.0, 'score_d3': 0.0,
            }

        # would_fail gate
        try:
            fail, why = would_fail(str(mid), my_key, opp_key, state, mi)
        except Exception:
            fail, why = False, None
        if fail:
            return {
                'id': str(mid), 'name': name,
                'score': 0.0, 'expected': 0.0, 'acc': 0.0, 'effectiveness': 0.0,
                'first_prob': 0.5, 'opp_counter_ev': 0.0,
                'why_blocked': str(why or 'would fail'),
                'score_d1': 0.0, 'score_d2': 0.0, 'score_d3': 0.0,
            }

        # baseline damage & features
        try:
            dmg = estimate_damage(state, my_key, opp_key, str(mid), mi)
            rolls = list(dmg.get('rolls', []) or [])
            avg = sum(rolls) / len(rolls) if rolls else 0.0
            eff = float(dmg.get('effectiveness', 1.0) or 1.0)
        except Exception:
            rolls, avg, eff = [], 0.0, 1.0

        # Recalculate type effectiveness more robustly (UI mapping fixes)
        try:
            move_raw = mi.raw(str(mid))
            if move_raw:
                move_type = move_raw.get("type", "")
                def_types = getattr(state.team.opponent[opp_key], "types", [])
                if move_type and def_types and type_chart:
                    eff = _type_mult(move_type, def_types, type_chart)
        except Exception:
            pass

        acc_p = _acc_to_prob(getattr(mv, 'accuracy', 1.0))

        # Turn order
        try:
            first_prob, _ = predict_order_for_ids(state, my_key, str(mid), opp_key, default_opp_move or "tackle", mi)
        except Exception:
            first_prob = 0.5

        # Current HP fractions (cap our expectations to remaining HP)
        my_ps = state.team.ours[my_key]
        opp_ps = state.team.opponent[opp_key]
        my_now = _hp_frac(my_ps)
        opp_now = _hp_frac(opp_ps)

        opp_max = float(getattr(opp_ps, 'max_hp', 0) or opp_ps.stats.raw.get('hp', 1) or 1)
        my_max  = float(getattr(my_ps, 'max_hp', 0)  or my_ps.stats.raw.get('hp', 1) or 1)

        expected_out = min((avg / opp_max) * acc_p, opp_now) if opp_max > 0 else 0.0

        # KO probability (if we go first)
        try:
            opp_cur_hp = int(getattr(opp_ps, "current_hp", 0) or getattr(opp_ps, "max_hp", 0) or 0)
        except Exception:
            opp_cur_hp = 0
        ko_prob = 0.0
        if rolls and opp_cur_hp > 0:
            ko_prob = (sum(1 for r in rolls if r >= opp_cur_hp) / len(rolls)) * acc_p

        # Incoming this turn (worst revealed; capped by our remaining HP)
        incoming_frac = self._opp_best_incoming_frac(state, opp_key, my_key, mi, default_opp_move, opp_moves_known)

        # If we KO before the opponent can act, their counter doesn't happen
        opp_counter_ev = incoming_frac * (1.0 - first_prob * ko_prob)

        # Depth 1 score
        d1 = (expected_out - W.get("opp_dmg_penalty", 1.0) * opp_counter_ev) \
             + W.get("go_first_bonus", 0.0) * float(first_prob) \
             + W.get("survival_bonus", 0.0)

        # Depth 2 & 3 (cheap rollouts in HP-fraction space; no full state mutation)
        d2 = d1
        d3 = d1
        if depth >= 2:
            gamma = self._gamma
            my_rem  = max(0.0, my_now  - opp_counter_ev)
            opp_rem = max(0.0, opp_now - expected_out)

            out2 = 0.0 if opp_rem <= 0 else min(self._our_best_outgoing_frac(state, my_key, opp_key, mi), opp_rem)
            in2  = 0.0 if my_rem  <= 0 else min(self._opp_best_incoming_frac(state, opp_key, my_key, mi, default_opp_move, opp_moves_known), my_rem)
            d2 = d1 + gamma * (out2 - W.get("opp_dmg_penalty", 1.0) * in2)

            if depth >= 3:
                my_rem2  = max(0.0, my_rem  - in2)
                opp_rem2 = max(0.0, opp_rem - out2)
                out3 = 0.0 if opp_rem2 <= 0 else min(self._our_best_outgoing_frac(state, my_key, opp_key, mi), opp_rem2)
                in3  = 0.0 if my_rem2  <= 0 else min(self._opp_best_incoming_frac(state, opp_key, my_key, mi, default_opp_move, opp_moves_known), my_rem2)
                d3 = d2 + (gamma ** 2) * (out3 - W.get("opp_dmg_penalty", 1.0) * in3)

        depth_score = d1 if depth <= 1 else d2 if depth == 2 else d3

        return {
            'id': str(mid), 'name': name,
            'score': float(depth_score),
            'expected': float(expected_out), 'exp_dmg': float(expected_out),
            'acc_mult': float(acc_p), 'acc': float(acc_p),
            'effectiveness': float(eff), 'eff': float(eff),
            'first_prob': float(first_prob),
            'opp_counter_ev': float(opp_counter_ev),
            'score_d1': float(d1), 'score_d2': float(d2), 'score_d3': float(d3),
        }

    def choose_action(self, battle: Any) -> ChosenAction:
        \"\"\"Return ChosenAction and rich debug info for the UI (now depth-aware).\"\"\"
        state = get_state(battle)
        mi = MovesInfo(state.format or 9)
        type_chart = _safe_get_type_chart(mi)

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

        # Opponent revealed moves (fallback to "tackle")
        opp_moves_known: List[str] = []
        try:
            if opp_key:
                opp_ps = state.team.opponent[opp_key]
                opp_moves_known = [m.id for m in (opp_ps.moves or []) if getattr(m, "id", None)]
        except Exception:
            pass
        default_opp_move = opp_moves_known[0] if opp_moves_known else "tackle"

        # ---- Evaluate MOVES (depth-aware) -----------------------------------
        moves_eval: List[Dict[str, Any]] = []
        if not force_switch and my_key and opp_key:
            depth = max(1, int(self._depth))
            for mv in legal_moves:
                cand = self._evaluate_move_with_depth(
                    state, my_key, opp_key, mv, mi, type_chart,
                    opp_moves_known, default_opp_move, depth
                )
                moves_eval.append(cand)
            moves_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # ---- Evaluate SWITCHES (keep same heuristic; display-rich) ----------
        switches_eval: List[Dict[str, Any]] = []
        if legal_switches and opp_key:
            chart = _safe_get_type_chart(mi)
            # Current active (for risk comparison if we stay)
            stay_incoming = 0.0
            try:
                if my_key and opp_moves_known:
                    my_ps = state.team.ours[my_key]
                    my_max = float(my_ps.max_hp or my_ps.stats.raw.get("hp", 1) or 1)
                    worst = 0.0
                    for om in opp_moves_known:
                        d = estimate_damage(state, opp_key, my_key, str(om), mi)
                        rr = d.get('rolls', []) or []
                        if rr:
                            worst = max(worst, sum(rr)/len(rr))
                    stay_incoming = min(worst / my_max, _hp_frac(my_ps))
            except Exception:
                pass

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

                # Find key for this species
                tmp_state = get_state(battle)
                cand_key = None
                for k, ps in tmp_state.team.ours.items():
                    if str(getattr(ps, 'species', '')).lower() == species.lower() and not getattr(ps, "is_active", False):
                        cand_key = k; break
                if cand_key is None:
                    for k, ps in tmp_state.team.ours.items():
                        if str(getattr(ps, 'species', '')).lower() == species.lower():
                            cand_key = k; break
                if cand_key is None:
                    switches_eval.append({'species': species, 'score': 0.0, 'hp_fraction': float(getattr(p, 'current_hp_fraction', 1.0) or 1.0), 'note': 'no-key'})
                    continue

                # Hazard + Web effects on switch-in (no mutation)
                haz = apply_switch_in_effects(tmp_state, cand_key, "ally", mi, mutate=False)
                haz_frac = float(haz.get("fraction_lost") or 0.0)

                # Incoming this turn on the switch
                cand_ps = tmp_state.team.ours[cand_key]
                cand_max = float(cand_ps.max_hp or cand_ps.stats.raw.get("hp", 1) or 1)
                incoming = 0.0
                try:
                    worst = 0.0
                    test_moves = opp_moves_known or [default_opp_move]
                    for om in test_moves:
                        d = estimate_damage(tmp_state, opp_key, cand_key, str(om), mi)
                        rr = d.get('rolls', []) or []
                        if rr:
                            worst = max(worst, sum(rr)/len(rr))
                    incoming = min(worst / cand_max, _hp_frac(cand_ps))
                except Exception:
                    pass

                # Outgoing potential next turn from candidate's revealed moves
                out_frac = 0.0
                try:
                    best = 0.0
                    opp_ps = tmp_state.team.opponent[opp_key]
                    opp_max = float(opp_ps.max_hp or opp_ps.stats.raw.get("hp", 1) or 1)
                    cand_moves = cand_ps.moves or []
                    for mslot in cand_moves or []:
                        mid = getattr(mslot, "id", None) or getattr(mslot, "move_id", None) or getattr(mslot, "name", None)
                        if not mid:
                            continue
                        d2 = estimate_damage(tmp_state, cand_key, opp_key, str(mid), mi)
                        rr2 = d2.get('rolls', []) or []
                        if rr2:
                            avg_out = sum(rr2)/len(rr2)
                            best = max(best, (avg_out / opp_max))
                    out_frac = min(best, _hp_frac(opp_ps))
                except Exception:
                    pass

                hp_frac = float(getattr(p, 'current_hp_fraction', 1.0) or 1.0)

                base_score = (out_frac * 1.2) - (incoming * 1.1) - (haz_frac * 0.5)
                score = max(0.0, base_score)  # display-only; move vs switch decided directly below

                switches_eval.append({
                    'species': species,
                    'score': float(score),
                    'hp_fraction': hp_frac,
                    'incoming_on_switch': float(incoming),
                    'hazards_frac': float(haz_frac),
                    'stay_incoming': float(stay_incoming),
                    'outgoing_frac': float(out_frac),
                    'base_score': float(base_score),
                })
            switches_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # ---- Decide -----------------------------------------------------------
        snap = snapshot_battle(battle)
        best_move = moves_eval[0] if moves_eval else None
        best_switch = switches_eval[0] if switches_eval else None

        # order block for telemetry (even if we pick a switch)
        order_block = {}
        if best_move:
            try:
                order_block['p_user_first'] = float(best_move.get('first_prob', 0.5))
            except Exception:
                order_block['p_user_first'] = 0.5

        # If forced to switch, do it
        if force_switch and best_switch:
            return ChosenAction(
                kind='switch', switch_species=best_switch['species'],
                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                       'picked': {'kind': 'switch', **best_switch}, 'order': order_block or {'p_user_first': 0.5}}
            )

        # Otherwise compare EVs (depth-aware move vs heuristic switch)
        if best_move and (not best_switch or best_move['score'] >= best_switch['score']):
            return ChosenAction(
                kind='move', move_id=str(best_move['id']),
                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                       'picked': {'kind': 'move', **best_move}, 'order': order_block or {'p_user_first': 0.5}}
            )

        if best_switch:
            return ChosenAction(
                kind='switch', switch_species=best_switch['species'],
                debug={'candidates': moves_eval, 'switches': switches_eval, 'snapshot': snap,
                       'picked': {'kind': 'switch', **best_switch}, 'order': order_block or {'p_user_first': 0.5}}
            )

        # Fallbacks
        if legal_moves:
            return ChosenAction(kind='move', move_id=str(getattr(legal_moves[0], 'id', '')),
                                debug={'snapshot': snap, 'fallback': True, 'order': {'p_user_first': 0.5}})
        if legal_switches:
            for p in legal_switches:
                sp = str(getattr(p, 'species', '') or '')
                if my_key:
                    cursp = str(getattr(state.team.ours[my_key], 'species', '') or '')
                else:
                    cursp = ''
                if sp.lower() != cursp.lower():
                    return ChosenAction(kind='switch', switch_species=sp, debug={'snapshot': snap, 'fallback': True, 'order': {'p_user_first': 0.5}})
            return ChosenAction(kind='switch', switch_species=str(getattr(legal_switches[0], 'species', '')),
                                debug={'snapshot': snap, 'fallback': True, 'order': {'p_user_first': 0.5}})

        return ChosenAction(kind='move', move_id='struggle', debug={'snapshot': snap, 'fallback': True, 'order': {'p_user_first': 0.5}})

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

        # Convenience alias for older UI code
        try:
            self._psclient = self.ps_client
        except Exception:
            pass

    # --------------- Buttons ---------------
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

    # --------------- Core decision ---------------
    def choose_move(self, battle):
        decision = self.engine.choose_action(battle)
        try:
            if self.on_think_hook and isinstance(decision.debug, dict):
                dd = dict(decision.debug)
                dd.setdefault('snapshot', snapshot_battle(battle))
                try:
                    import json, logging
                    logging.getLogger('Think').info(
                        'UI_THINK turn=%s payload=%s',
                        getattr(battle, 'turn', None),
                        json.dumps(dd, default=str)
                    )
                except Exception:
                    pass
                self.on_think_hook(battle, dd)
        except Exception:
            pass

        # Convert ChosenAction to poke-env order
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
UI_NEW    = r"""\"\"\"
Tkinter UI for Stockfish-like model (depth-aware)
- Login / format picker
- Buttons: Ladder, Challenge, Accept, Forfeit, Start Timer, Train Weights, Reload Weights
- Panels: Team (ours / opponent), Thinking (candidates & switches), Logs
- Adjustable analysis depth (1..3) reflected in engine

This UI writes clean JSONL telemetry (one JSON object per line) into logs/telemetry_*.jsonl
so tools/train_weights.py can learn weights, and can train+reload from the UI.
\"\"\"
from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import json
import time
from pathlib import Path
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
    \"\"\"Push player logger records to a queue so UI can consume them safely.\"\"\"
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
        self.geometry("1200x740")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.username = username
        self.password = password
        self.server_mode = server_mode
        self.custom_ws = custom_ws
        self.battle_format = battle_format

        # Async loop thread
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Player (created on connect)
        self.player: Optional[StockfishPokeEnvPlayer] = None

        # Logging pane & handler
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # UI State
        self._scheduled_tasks: List[str] = []
        self._latest_think: Dict[str, Any] = {}
        self._latest_snapshot: Dict[str, Any] = {}

        # Fallback thinking guard
        self._last_fallback_turn: Optional[int] = None
        self._last_real_think_turn: Optional[int] = None

        self._root_log_handler_attached = False

        # Telemetry writer
        self.telemetry_enabled = True
        ts = time.strftime("%Y%m%d-%H%M%S")
        Path("logs").mkdir(parents=True, exist_ok=True)
        self.telemetry_path = os.path.join("logs", f"telemetry_{ts}.jsonl")

        self._build_ui()
        self._pump_logs()

        # Bootstrap: connect immediately
        self._submit(self._async_connect())

    # ---------- Async scheduling ----------
    def _submit(self, coro: "asyncio.coroutines.Coroutine[Any, Any, Any]"):
        \"\"\"Schedule a coroutine on our loop thread.\"\"\"
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
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # --- Dashboard tab: controls + teams
        dash = ttk.Frame(nb)
        nb.add(dash, text="Dashboard")

        controls = ttk.LabelFrame(dash, text="Controls")
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

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

        ttk.Button(controls, text="Train Weights", command=self._train_weights).pack(side=tk.LEFT, padx=(16, 4))
        ttk.Button(controls, text="Reload Weights", command=self._reload_weights).pack(side=tk.LEFT, padx=4)

        # Team panes
        teams = ttk.Frame(dash)
        teams.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self.team_tree = self._make_team_tree(teams, "Your team")
        self.opp_tree = self._make_team_tree(teams, "Opponent team")

        # --- Thinking tab
        think_tab = ttk.Frame(nb)
        nb.add(think_tab, text="Thinking")
        self.cand_tree = self._make_cand_tree(think_tab, "Move candidates")
        self.switch_tree = self._make_switch_tree(think_tab, "Switch candidates")

        # --- Logs tab
        logs_tab = ttk.Frame(nb)
        nb.add(logs_tab, text="Logs")
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
        setattr(self, f"_{title.replace(' ', '_').lower()}_frame", frame)
        return tree

    def _make_cand_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("move", "score", "exp_dmg", "acc", "eff", "first", "opp", "d1", "d2", "d3", "note")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        hdrs = ("MOVE", "SCORE", "EXP", "ACC", "EFF", "FIRST", "OPP", "D1", "D2", "D3", "WHY/NOTE")
        widths = (180, 80, 70, 60, 60, 60, 60, 70, 70, 70, 260)
        for c, h, w in zip(cols, hdrs, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_switch_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("species", "score", "base", "out", "in", "haz", "hp")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        headers = ("SPECIES", "SCORE", "BASE", "OUT", "IN", "HAZ", "HP")
        widths = (140, 70, 70, 60, 60, 60, 60)
        for c, h, w in zip(cols, headers, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    # ---------- Connect ----------
    async def _async_connect(self):
        # Apply runtime model patch if present
        try:
            if patch_stockfish_player:
                patch_stockfish_player()
        except Exception as e:
            self._append_log(f"Model patch failed (non-fatal): {e}")

        # Build account/server configs
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

        # Common kwargs
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

        player = None
        last_error = None

        # Try the richest signature first
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
                player = StockfishPokeEnvPlayer(**common, **kv)
                break
            except TypeError as e:
                last_error = e
                continue
        if player is None:
            self._append_log(f"Failed to construct player with extended kwargs; retrying minimal. Last error: {last_error}")
            player = StockfishPokeEnvPlayer(**common)

        self.player = player

        # Attach think callback post-construction
        try:
            attached = False
            for attr in ("on_think", "think_callback", "on_think_hook"):
                if hasattr(self.player, attr):
                    setattr(self.player, attr, self._on_think)
                    attached = True
                    break
            if not attached:
                for meth in ("set_on_think", "set_think_callback", "register_think_callback", "on_think_connect"):
                    fn = getattr(self.player, meth, None)
                    if callable(fn):
                        fn(self._on_think)
                        attached = True
                        break
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

        try:
            self._call_on_main(self._poll_battle)
        except Exception:
            pass

    # ---------- Actions ----------
    async def _ladder(self, n: int):
        if not self.player:
            return
        if getattr(self.player, "format", None) != self.format_var.get():
            try:
                self.player.format = self.format_var.get()
            except Exception:
                pass
        self._append_log(f"Starting ladder: {n} game(s)…")
        await self.player.ladder(n)

    async def _accept(self, n: int):
        if not self.player:
            return
        self._append_log(f"Accepting {n} challenge(s)…")
        await self.player.accept_challenges(opponent=None, n_challenges=n)

    def _challenge_dialog(self):
        if not self.player:
            return
        dlg = tk.Toplevel(self)
        dlg.title("Challenge a user")
        ttk.Label(dlg, text="Opponent username:").pack(side=tk.TOP, padx=8, pady=8)
        name_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=name_var, width=28).pack(side=tk.TOP, padx=8, pady=(0, 8))

        def go():
            opp = name_var.get().strip()
            if opp:
                self._submit(self.player.send_challenges(opp, n_challenges=1))
            dlg.destroy()

        ttk.Button(dlg, text="Challenge", command=go).pack(side=tk.TOP, padx=8, pady=8)

    async def _forfeit_all(self):
        p = self.player
        if not p:
            return
        try:
            m = getattr(p, "forfeit_all", None)
            if callable(m):
                await m()
                self._append_log("Called player.forfeit_all().")
                return
        except Exception as e:
            self._append_log(f"player.forfeit_all() failed: {e} — falling back to direct /forfeit.")

        try:
            client = getattr(p, "ps_client", None) or getattr(p, "_client", None)
            if not client:
                raise RuntimeError("PSClient missing on player")
            battles = getattr(p, "battles", {}) or {}
            rooms: List[str] = []
            for key, battle in list(battles.items()):
                room_id = getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None) or str(key)
                if room_id:
                    rooms.append(room_id)
            if not rooms:
                self._append_log("No active battle rooms found for /forfeit.")
                return
            sent = 0
            for r in rooms:
                try:
                    await client.send_message("/forfeit", room=r)
                    sent += 1
                except Exception as e2:
                    self._append_log(f"Failed to send /forfeit to {r}: {e2}")
            self._append_log(f"Sent /forfeit to {sent} room(s).")
        except Exception as e:
            self._append_log(f"Forfeit fallback failed: {e}")

    async def _timer_all(self, on: bool):
        if self.player:
            await self.player.timer_all(on)

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

    # ---------- Think data from model ----------
    def _on_think(self, battle, think: Dict[str, Any]):
        # Called from model thread; marshal to main thread
        self._latest_think = think or {}
        try:
            snap = think.get("snapshot")
        except Exception:
            snap = None
        self._latest_snapshot = snap or snapshot_battle(battle)
        try:
            self._last_real_think_turn = int(self._latest_snapshot.get("turn")) if self._latest_snapshot else None
        except Exception:
            pass

        # Emit one JSON object per line for training (clean JSONL)
        try:
            if self.telemetry_enabled and isinstance(think, dict):
                rec = dict(think)
                rec["turn"] = self._latest_snapshot.get("turn") if self._latest_snapshot else None
                rec["battle_tag"] = getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None)
                with open(self.telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

        self._call_on_main(self._refresh_thinking)
        self._call_on_main(self._refresh_teams)

    def _refresh_thinking(self):
        if not self.winfo_exists():
            return
        # Candidates
        self._reload_tree(self.cand_tree)
        for d in self._latest_think.get("candidates", []):
            try:
                move = d.get("name") or d.get("id") or d.get("move") or d.get("move_id")
                s = d.get("score")
                score = f"{float(s):.2f}" if s is not None else ""
                exp = ""
                for k in ("exp_dmg", "expected", "exp", "expdmg", "expected_damage"):
                    if d.get(k) is not None:
                        exp = f"{float(d.get(k)):.2f}"
                        break
                acc = ""
                for k in ("acc", "acc_mult", "accuracy", "hit_chance"):
                    if d.get(k) is not None:
                        acc = f"{float(d.get(k)):.2f}"
                        break
                eff = ""
                for k in ("eff", "effectiveness", "type_mult", "type_effectiveness"):
                    if d.get(k) is not None:
                        eff = f"{float(d.get(k)):.2f}"
                        break
                first = ""
                if d.get("first_prob") is not None:
                    first = f"{float(d['first_prob']):.2f}"
                opp = ""
                if d.get("opp_counter_ev") is not None:
                    opp = f"{float(d['opp_counter_ev']):.2f}"
                d1 = f"{float(d.get('score_d1', '')):.2f}" if d.get("score_d1") is not None else ""
                d2 = f"{float(d.get('score_d2', '')):.2f}" if d.get("score_d2") is not None else ""
                d3 = f"{float(d.get('score_d3', '')):.2f}" if d.get("score_d3") is not None else ""
                note = d.get("why_blocked") or d.get("note") or d.get("why") or ""
                self.cand_tree.insert("", tk.END, values=(move, score, exp, acc, eff, first, opp, d1, d2, d3, note))
            except Exception:
                try:
                    self.cand_tree.insert("", tk.END, values=(str(d), "", "", "", "", "", "", "", "", "", ""))
                except Exception:
                    pass

        # Switches
        self._reload_tree(self.switch_tree)
        for d in self._latest_think.get("switches", []):
            try:
                species = d.get("species") or d.get("name") or d.get("id")
                s = d.get("score")
                score = f"{float(s):.2f}" if s is not None else ""
                base = d.get("base_score")
                base_s = f"{float(base):.2f}" if base is not None else ""
                out = d.get("outgoing_frac")
                out_s = f"{float(out):.2f}" if out is not None else ""
                incoming = d.get("incoming_on_switch")
                in_s = f"{float(incoming):.2f}" if incoming is not None else ""
                haz = d.get("hazards_frac")
                haz_s = f"{float(haz):.2f}" if haz is not None else ""
                hp = d.get("hp_fraction")
                hp_s = f"{int(round(float(hp) * 100))}%" if isinstance(hp, (int, float)) else ""
                self.switch_tree.insert("", tk.END, values=(species, score, base_s, out_s, in_s, haz_s, hp_s))
            except Exception:
                try:
                    species = d.get("species") or str(d)
                    score = f"{float(d.get('score', 0)):.2f}" if d.get('score') is not None else "0.00"
                    self.switch_tree.insert("", tk.END, values=(species, score, "", "", "", "", ""))
                except Exception:
                    pass

    def _refresh_teams(self):
        if not self.winfo_exists():
            return
        snap = self._latest_snapshot or {}
        # Our team
        self._reload_tree(self.team_tree)
        for sid, p in (snap.get("my_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            self.team_tree.insert("", tk.END, values=(sid, p.get("species"), hp_s, str(p.get("status") or ""), boosts))
        # Opp team
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

    # ---------- Polling (keep UI alive + fallback Thinking) ----------
    def _find_active_battle(self):
        p = getattr(self, "player", None)
        if not p:
            return None
        for name in ("current_battle", "battle", "active_battle"):
            b = getattr(p, name, None)
            if b is not None:
                return b
        battles = getattr(p, "battles", None)
        if isinstance(battles, dict) and battles:
            try:
                for b in battles.values():
                    if getattr(b, "active_pokemon", None) is not None or getattr(b, "turn", None):
                        return b
                return list(battles.values())[-1]
            except Exception:
                try:
                    return next(iter(battles.values()))
                except Exception:
                    return None
        return None

    def _poll_battle(self):
        try:
            if not self.winfo_exists():
                return
            b = self._find_active_battle()
            if b is not None:
                try:
                    snap = snapshot_battle(b)
                    self._latest_snapshot = snap
                except Exception:
                    snap = None
                try:
                    self._refresh_teams()
                except Exception:
                    pass
                try:
                    turn = int(snap.get("turn")) if snap else None
                except Exception:
                    turn = None
                if turn is not None and turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                    self._emit_fallback_think(b, snap)
                    self._last_fallback_turn = turn
        finally:
            if self.winfo_exists():
                try:
                    h = self.after(500, self._poll_battle)
                    self._scheduled_tasks.append(h)
                except Exception:
                    pass

    def _emit_fallback_think(self, battle, snap: Optional[Dict[str, Any]]):
        \"\"\"Synthesize a lightweight 'Thinking' view if model doesn't emit one.\"\"\"
        try:
            cands = []
            for m in (getattr(battle, "available_moves", None) or []):
                try:
                    name = getattr(m, "name", None) or getattr(m, "id", None) or str(m)
                    bp = getattr(m, "base_power", None) or getattr(m, "basePower", None) or 0
                    acc = getattr(m, "accuracy", None)
                    if acc is True:
                        acc_val = 1.0
                    elif isinstance(acc, (int, float)):
                        acc_val = float(acc) / (100.0 if acc > 1 else 1.0)
                    else:
                        acc_val = 1.0
                    expected = float(bp or 0) * float(acc_val)
                    cands.append({
                        "name": name,
                        "score": expected,
                        "exp_dmg": expected,
                        "acc": acc_val,
                        "eff": "",
                        "first": "",
                        "opp": "",
                        "d1": "", "d2": "", "d3": "",
                        "note": "synthetic",
                    })
                except Exception:
                    pass

            switches = []
            for pkm in (getattr(battle, "available_switches", None) or []):
                try:
                    species = getattr(pkm, "species", None) or getattr(pkm, "name", None) or str(pkm)
                    hp_frac = getattr(pkm, "hp_fraction", None) or getattr(pkm, "current_hp_fraction", None)
                    switches.append({
                        "species": species,
                        "score": float(hp_frac or 0.0),
                        "hp_fraction": float(hp_frac or 0.0),
                    })
                except Exception:
                    pass

            think = {"candidates": sorted(cands, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "switches": sorted(switches, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "snapshot": snap or snapshot_battle(battle)}
            self._on_think(battle, think)
        except Exception as e:
            self._append_log(f"Fallback think failed: {e}")

    # ---------- Train / Reload ----------
    def _train_weights(self):
        try:
            self._append_log("Training weights…")
            import subprocess
            cmd = [sys.executable, "tools/train_launcher.py"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            self._append_log(out.strip())
            if proc.returncode == 0:
                self._append_log("Training complete. Click 'Reload Weights' to apply.")
            else:
                messagebox.showerror("Training failed", out or "Unknown error")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reload_weights(self):
        try:
            eng = getattr(self.player, "engine", None)
            if eng and hasattr(eng, "reload_weights"):
                eng.reload_weights()
                self._append_log("Reloaded Models/weights.json into engine.")
            else:
                self._append_log("Engine lacks reload_weights(); hot-reload not supported.")
        except Exception as e:
            self._append_log(f"Reload failed: {e}")

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
            try:
                self.after_cancel(h)
            except Exception:
                pass
        self._scheduled_tasks.clear()

        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

        self.destroy()

# -------------- entry point for main menu ----------------
def launch_stockfish_window(root: tk.Tk, username: str, password: Optional[str],
                            server_mode: str, custom_ws: Optional[str], battle_format: str) -> StockfishWindow:
    return StockfishWindow(root, username=username, password=password,
                           server_mode=server_mode, custom_ws=custom_ws, battle_format=battle_format)
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
            bak = path.with_suffix(path.suffix + f".bak.{ts}")
            bak.parent.mkdir(parents=True, exist_ok=True)
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[backup] {path} -> {bak}")
    except Exception as e:
        print(f"[warn] could not backup {path}: {e}")

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[ok] wrote {path} ({len(content.splitlines())} lines)")

def main():
    model_path = find_path(MODEL_CANDIDATES)
    ui_path    = find_path(UI_CANDIDATES)

    print("[target] model:", model_path)
    print("[target] ui   :", ui_path)

    backup(model_path)
    backup(ui_path)

    write_file(model_path, MODEL_NEW)
    write_file(ui_path, UI_NEW)

    print("\nDone. Next steps:")
    print("  1) Run your app and play a few turns to generate logs/telemetry_*.jsonl")
    print("  2) Click 'Train Weights' in the UI (or run: python tools/train_launcher.py)")
    print("  3) Click 'Reload Weights' to hot-load Models/weights.json into the engine")
    print("  4) Use the Depth spinner (1..3) to see D1/D2/D3 in the Thinking tab")

if __name__ == "__main__":
    main()
