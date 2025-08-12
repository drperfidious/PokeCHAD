
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
