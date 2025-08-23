from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

from Data.battle_runtime import (
    get_state,
    predict_order_for_ids,
    estimate_damage,
    would_fail,
    apply_switch_in_effects,  # NEW: hazards on switch-in
    predict_first_prob_speed_only,  # NEW: speed-only fallback for first move probability
    get_effective_speeds,  # NEW: effective speeds for scarf suspicion
)
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo

# ---------------- Weights ----------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "expected_mult": 1.0,
    "go_first_bonus": 0.4,
    "opp_dmg_penalty": 1.0,
    "survival_bonus": 0.0,
    "accuracy_mult": 0.0,
    "effectiveness_mult": 0.0,
    "ko_bonus": 1.0,
    # Switching
    "switch_outgoing_mult": 1.0,
    "switch_incoming_penalty": 1.0,
}

def _load_weights(path: str = os.path.join("Models", "weights.json")) -> Dict[str, float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        w = dict(_DEFAULT_WEIGHTS)
        for k, v in raw.items():
            if k in w:
                try: w[k] = float(v)
                except Exception: pass
        return w
    except Exception:
        return dict(_DEFAULT_WEIGHTS)

# ---------------- Helpers ----------------

def _acc_to_prob(acc) -> float:
    if acc is True or acc is None: return 1.0
    try:
        x = float(acc)
        return x / 100.0 if x > 1.0 else max(0.0, min(1.0, x))
    except Exception:
        return 1.0

def _hp_frac(ps) -> float:
    try:
        if ps and ps.max_hp:
            return max(0.0, min(1.0, (ps.current_hp or ps.max_hp) / ps.max_hp))
    except Exception: pass
    return 1.0

def _expected_damage_fraction(state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo) -> Tuple[float, Dict[str, Any]]:
    dmg = estimate_damage(state, atk_key, dfd_key, move_id, mi, is_critical=False)
    rolls = dmg.get("rolls") or []
    if not rolls:
        return 0.0, dmg
    dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
    max_hp = int(getattr(dfd, "max_hp", 0) or getattr(dfd, "stats", {}).raw.get("hp", 1) or 1)
    return (float(sum(rolls)) / (len(rolls) * max_hp)), dmg

_COMMON_STAB = {
    "normal": ["return","bodyslam"],
    "fire": ["flamethrower","fireblast"],
    "water": ["surf","hydropump"],
    "electric": ["thunderbolt","thunder"],
    "grass": ["energyball","leafstorm"],
    "ice": ["icebeam","blizzard"],
    "fighting": ["closecombat","drainpunch"],
    "poison": ["sludgebomb","gunkshot"],
    "ground": ["earthquake","earthpower"],
    "flying": ["hurricane","bravebird"],
    "psychic": ["psychic","psyshock"],
    "bug": ["bugbuzz","leechlife"],
    "rock": ["stoneedge","rockslide"],
    "ghost": ["shadowball","poltergeist"],
    "dragon": ["dracometeor","dragonpulse"],
    "dark": ["darkpulse","crunch"],
    "steel": ["flashcannon","ironhead"],
    "fairy": ["moonblast","playrough"],
}

def _opp_best_on_target(state, opp_key: str, target_key: str, mi: MovesInfo) -> float:
    opp = state.team.opponent[opp_key]
    best = 0.0
    mv_list = [m for m in (opp.moves or []) if getattr(m, 'id', None)]
    if mv_list:
        for mv in mv_list:
            if (getattr(mv, 'category', '') or '').lower() == 'status' or (getattr(mv, 'base_power', 0) or 0) <= 0:
                continue
            frac, _ = _expected_damage_fraction(state, opp_key, target_key, mv.id, mi)
            best = max(best, frac * _acc_to_prob(getattr(mv, 'accuracy', None)))
        return best
    for t in opp.types or []:
        for mid in _COMMON_STAB.get((t or '').lower(), [])[:2]:
            try:
                frac, _ = _expected_damage_fraction(state, opp_key, target_key, mid, mi)
                raw = mi.raw(mid) or {}
                best = max(best, frac * _acc_to_prob(raw.get('accuracy')))
            except Exception: continue
    for mid in ("icebeam","closecombat","earthquake"):
        try:
            frac, _ = _expected_damage_fraction(state, opp_key, target_key, mid, mi)
            raw = mi.raw(mid) or {}
            best = max(best, frac * _acc_to_prob(raw.get('accuracy')))
        except Exception: continue
    return best

# ---------------- Data container ----------------
@dataclass
class ChosenAction:
    kind: str
    move_id: Optional[str] = None
    switch_species: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

# ---------------- Engine ----------------
class StockfishModel:
    def __init__(self, battle_format: str = 'gen9ou'):
        self.battle_format = battle_format
        self._W = _load_weights()
        self._depth = 1
        self._branching = 3
        self._softmin_temp: float = 0.0  # 0 => hard minimax, >0 => softmin (Boltzmann-weighted average)
        # Verbose think debug printing (terminal). Enable with env POKECHAD_THINK_DEBUG=1
        try:
            import os as _os
            self._verbose = bool(int(_os.environ.get('POKECHAD_THINK_DEBUG', '0')))
        except Exception:
            self._verbose = False

    # New: external injection of a full weight mapping (used by tuner/self-play). Unknown keys ignored, missing keys preserved.
    def set_weights(self, mapping: Dict[str, float]):  # type: ignore[name-defined]
        try:
            if not isinstance(mapping, dict):
                return
            for k, v in mapping.items():
                if k in self._W:
                    try:
                        self._W[k] = float(v)
                    except Exception:
                        pass
        except Exception:
            pass

    def set_depth(self, d: int):
        try: d = int(d)
        except Exception: d = 1
        self._depth = max(1, min(10, d))

    def set_branching(self, k: int):
        try: k = int(k)
        except Exception: k = 1
        self._branching = max(1, min(10, k))

    def set_softmin_temperature(self, t: float):
        """Set softmin temperature (opponent reply aggregation). 0 => pure minimax, >0 softmin (Boltzmann-weighted average)."""
        try: t = float(t)
        except Exception: t = 0.0
        if t < 0: t = 0.0
        self._softmin_temp = t

    def set_verbose(self, flag: bool=True):
        self._verbose = bool(flag)

    def reload_weights(self, path: str = os.path.join('Models','weights.json')):
        self._W = _load_weights(path)

    # ---- core ----
    def choose_action(self, battle: Any) -> ChosenAction:
        state = get_state(battle)
        mi = MovesInfo(state.format or 9)
        # decision instrumentation accumulator
        _dec_log: Dict[str, Any] = { 'battle_tag': getattr(battle,'battle_tag', getattr(battle,'room_id', None)), 'turn': getattr(battle,'turn', None) }
        # Defensive early initialization to avoid NameError in any early-return / exception paths
        best_move = None  # type: ignore
        best_switch = None  # type: ignore
        # Tree trace accumulator (per-depth branch logging)
        tree_trace: List[str] = []
        try:
            import os as _os
            tree_trace_enabled = bool(int(_os.environ.get('POKECHAD_TREE_TRACE','0')))
        except Exception:
            tree_trace_enabled = False

        # Active keys
        def _active(side: Dict[str, Any]) -> Optional[str]:
            for k,p in side.items():
                if getattr(p,'is_active',False) and getattr(p,'current_hp',0) > 0: return k
            for k,p in side.items():
                if (getattr(p,'status','') or '').lower()=='fnt': continue
                if getattr(p,'current_hp',1) > 0: return k
            return None
        my_key = _active(state.team.ours) if getattr(state,'team',None) else None
        opp_key = _active(state.team.opponent) if getattr(state,'team',None) else None

        legal_moves: List[Any] = list(getattr(battle,'available_moves',[]) or [])
        legal_switches: List[Any] = list(getattr(battle,'available_switches',[]) or [])
        # Define force_switch immediately so it's available for synthetic switch logic below
        force_switch = bool(getattr(battle,'force_switch', False))
        # If force switch but engine sees no available_switches, synthesize from team state (bench) to allow evaluation
        if force_switch and not legal_switches and getattr(state,'team',None):
            class _Stub:
                __slots__ = ('species',)
                def __init__(self, s): self.species = s
            try:
                active_species = None
                if my_key and my_key in state.team.ours:
                    active_species = getattr(state.team.ours[my_key],'species',None)
                for k,ps in state.team.ours.items():
                    if getattr(ps,'current_hp',0) <= 0: continue
                    sp = getattr(ps,'species',None)
                    if not sp or sp==active_species: continue
                    legal_switches.append(_Stub(sp))
            except Exception: pass

        # Only trust battle.force_switch flag; do not auto-set from (not legal_moves and legal_switches) because poke-env already handles this
        # (force_switch already defined above)
        # force_switch = bool(getattr(battle,'force_switch', False))

        # Fallback: if active key missing, attempt to map from battle active pokemon species (helps after form changes / detailschange)
        if not my_key:
            try:
                active_species = getattr(getattr(battle,'active_pokemon',None),'species',None)
                if active_species:
                    for k,p in getattr(state.team,'ours',{}).items():
                        if getattr(p,'species',None)==active_species and (getattr(p,'current_hp',0)>0):
                            my_key = k; break
            except Exception: pass
        if not opp_key:
            try:
                opp_active_species = getattr(getattr(battle,'opponent_active_pokemon',None),'species',None) or getattr(getattr(battle,'opponent_active_pokemon',None),'base_species',None)
                if opp_active_species:
                    for k,p in getattr(state.team,'opponent',{}).items():
                        if getattr(p,'species',None)==opp_active_species and (getattr(p,'current_hp',0)>0):
                            opp_key = k; break
            except Exception: pass

        opp_moves_known: List[str] = []
        if opp_key:
            try:
                opp_ps = state.team.opponent.get(opp_key)
                if opp_ps and getattr(opp_ps,'moves',None):
                    opp_moves_known = [m.id for m in opp_ps.moves if getattr(m,'id',None)]
            except Exception: pass

        moves_eval: List[Dict[str, Any]] = []
        switches_eval: List[Dict[str, Any]] = []  # ensure defined even if no evaluation block runs
        skip_reason: Optional[str] = None
        if not force_switch and my_key and opp_key:
            opp_ps = state.team.opponent[opp_key]
            opp_hp_frac = _hp_frac(opp_ps)
            opp_max = int(getattr(opp_ps,'max_hp',0) or getattr(opp_ps,'stats',{}).raw.get('hp',1) or 1)
            # Precompute effective speeds for scarf suspicion
            try:
                my_eff_spe, opp_eff_spe = get_effective_speeds(state, my_key, opp_key)
            except Exception:
                my_eff_spe, opp_eff_spe = (0, 0)
            opp_item_known = bool((getattr(opp_ps, 'item', None) or '').strip())
            trick_room_active = bool(getattr(state.field, 'trick_room', False))
            for mv in legal_moves:
                mid = getattr(mv,'id',None) or getattr(mv,'move_id',None)
                if not mid: continue
                # failure
                try: fail, why = would_fail(str(mid), my_key, opp_key, state, mi)
                except Exception: fail, why = False, None
                if fail:
                    moves_eval.append({'id':mid,'name':getattr(mv,'name',mid),'score':0.0,'expected':0.0,'exp_dmg':0.0,'acc':0.0,'why_blocked':str(why or 'would fail')})
                    continue
                # expected damage
                try: exp_frac, dmg = _expected_damage_fraction(state, my_key, opp_key, str(mid), mi)
                except Exception: exp_frac, dmg = 0.0, {}
                # Retrieve move info to annotate category and potential boosts/debuffs
                try:
                    _minfo = mi.get(str(mid))
                    _cat = (_minfo.category or 'Status').lower()
                    _target = (_minfo.target or '').lower()
                    _boosts = dict(_minfo.boosts or {})
                except Exception:
                    _cat, _target, _boosts = 'status', '', {}
                # Extract self-boosts and opponent debuffs
                _self_boosts = {}
                _target_debuffs = {}
                if _boosts:
                    # If target is self (typical for Swords Dance/Nasty Plot), treat as self boosts
                    if _target in {'self','selfside'} or _cat == 'status':
                        for k,v in list(_boosts.items()):
                            if isinstance(v, int) and v > 0 and k in ('atk','spa','def','spd','spe'):
                                _self_boosts[k] = int(v)
                            # negative boosts when target is self are rare; ignore for simplicity
                    # If targeting opponent (normal/adjacentFoe), negative atk/spa are opponent debuffs
                    if _target not in {'self','selfside'}:
                        for k,v in list(_boosts.items()):
                            if isinstance(v, int) and v < 0 and k in ('atk','spa'):
                                _target_debuffs[k] = int(v)
                acc_p = _acc_to_prob(getattr(mv,'accuracy',1.0))
                eff = float(dmg.get('effectiveness',1.0) or 1.0)
                # order
                try:
                    first_prob, order_details = predict_order_for_ids(
                        state, my_key, str(mid), opp_key,
                        (opp_moves_known[0] if opp_moves_known else 'tackle'), mi
                    )
                except Exception:
                    # Fallback: speed-only compare
                    try:
                        first_prob = predict_first_prob_speed_only(state, my_key, opp_key)
                    except Exception:
                        first_prob = 0.5
                first_prob = float(first_prob)
                # Possible Scarf suspicion: if we only barely outspeed (opp * 1.5 > us), Trick Room off, opponent item unknown, and our move isn't priority
                try:
                    mv_base_pri = int(getattr(mv, 'priority', 0) or 0)
                except Exception:
                    mv_base_pri = 0
                if (not trick_room_active) and (mv_base_pri <= 0) and (not opp_item_known) and (my_eff_spe > opp_eff_spe > 0) and ((opp_eff_spe * 3) > (my_eff_spe * 2)):
                    # Cap our first probability to reflect scarf risk
                    first_prob = min(first_prob, 0.30)
                # KO chance if we hit
                rolls = (dmg.get('rolls') or [])
                thr_abs = int(round(opp_hp_frac * opp_max))
                ko_rolls = sum(1 for r in rolls if int(r) >= max(1,thr_abs))
                p_ko_if_hit = (ko_rolls / max(1,len(rolls))) if rolls else 0.0
                p_ko_first = acc_p * p_ko_if_hit * first_prob
                # incoming expectation (worst move * chance opp acts)
                try: incoming_best = _opp_best_on_target(state, opp_key, my_key, mi)
                except Exception: incoming_best = 0.0
                p_opp_acts = (1-first_prob) + first_prob * (1 - p_ko_if_hit * acc_p)
                opp_counter_ev = incoming_best * p_opp_acts
                effective_exp = exp_frac * acc_p
                W=self._W
                score = (W['expected_mult']*effective_exp
                         - W['opp_dmg_penalty']*opp_counter_ev
                         + W['go_first_bonus']*first_prob
                         + W['effectiveness_mult']*eff
                         + W['accuracy_mult']*acc_p
                         + W['ko_bonus']*p_ko_first)
                # Breakdown for debug printing
                c_expected = W['expected_mult']*effective_exp
                c_opp = - W['opp_dmg_penalty']*opp_counter_ev
                c_first = W['go_first_bonus']*first_prob
                c_eff = W['effectiveness_mult']*eff
                c_acc = W['accuracy_mult']*acc_p
                c_ko = W['ko_bonus']*p_ko_first
                cand_entry = {
                    'id':mid,'name':getattr(mv,'name',mid),'score':float(score),
                    'score_depth':float(score),'future_proj':0.0,'depth_used':1,
                    'expected':float(exp_frac),'exp_dmg':float(exp_frac),
                    'acc':float(acc_p),'effectiveness':float(eff),'first_prob':first_prob,
                    'p_ko_if_hit':float(p_ko_if_hit),'p_ko_first':float(p_ko_first),
                    'opp_counter_ev':float(opp_counter_ev),'incoming_frac':float(incoming_best),
                    'category': _cat,
                    'self_boosts': _self_boosts or {},
                    'target_debuffs': _target_debuffs or {},
                    'score_breakdown': {
                        'expected': c_expected,
                        'opp_dmg_penalty': c_opp,
                        'go_first_bonus': c_first,
                        'effectiveness_mult': c_eff,
                        'accuracy_mult': c_acc,
                        'ko_bonus': c_ko,
                    }
                }
                # Attach order details if available
                try:
                    if 'order_details' not in cand_entry and 'order_details' in locals():
                        od = order_details or {}
                        cand_entry['order_details'] = {
                            'user_effective_speed': od.get('user_effective_speed'),
                            'opp_effective_speed': od.get('opp_effective_speed'),
                            'user_bracket': od.get('user_bracket'),
                            'opp_bracket': od.get('opp_bracket'),
                            'notes': od.get('notes'),
                        }
                except Exception:
                    pass
                moves_eval.append(cand_entry)
            moves_eval.sort(key=lambda x: x.get('score_depth', x.get('score',0.0)), reverse=True)

            # Build switch evaluations EARLY (needed for tree search) if not already built
            # switches_eval already initialized above
            if my_key and opp_key and legal_switches:
                active_species = ''
                try: active_species = str(getattr(state.team.ours.get(my_key),'species','') or '').lower()
                except Exception: pass
                for sw in legal_switches:
                    try:
                        species = str(getattr(sw,'species','') or '')
                        if species.lower() == active_species: continue
                        # locate key
                        cand_key=None
                        for k,ps in state.team.ours.items():
                            if str(getattr(ps,'species','')).lower()==species.lower():
                                cand_key=k; break
                        if not cand_key: continue
                        cand_hp = _hp_frac(state.team.ours[cand_key])
                        # Hazards on switch-in (ally side switches into opponent hazards)
                        haz_frac = 0.0
                        try:
                            haz = apply_switch_in_effects(state, cand_key, 'ally', mi, mutate=False)
                            haz_frac = float(haz.get('fraction_lost', 0.0) or 0.0)
                        except Exception:
                            haz_frac = 0.0
                        try: incoming = _opp_best_on_target(state, opp_key, cand_key, mi)
                        except Exception: incoming=0.0
                        # Total incoming this turn includes entry hazards + opponent counter damage
                        incoming_total = float(haz_frac) + float(incoming)
                        outgoing=0.0
                        try:
                            for mv_obj in getattr(state.team.ours[cand_key],'moves',[]) or []:
                                mid2 = getattr(mv_obj,'id',None)
                                if not mid2: continue
                                bp2 = int(getattr(mv_obj,'base_power',0) or getattr(mv_obj,'basePower',0) or 0)
                                if (getattr(mv_obj,'category','') or '').lower()=='status' or bp2<=0: continue
                                frac2,_ = _expected_damage_fraction(state, cand_key, opp_key, mid2, mi)
                                outgoing = max(outgoing, frac2*_acc_to_prob(getattr(mv_obj,'accuracy',None)))
                        except Exception: pass
                        W=self._W
                        score = W['switch_outgoing_mult']*outgoing - W['switch_incoming_penalty']*incoming_total
                        switches_eval.append({
                            'species':species,'score':float(score),'base_score':float(score),
                            'outgoing_frac':float(outgoing),'incoming_on_switch':float(incoming_total),
                            'hazards_frac':float(haz_frac),'hp_fraction':float(cand_hp),
                            'score_breakdown': {
                                'switch_outgoing_mult': W['switch_outgoing_mult']*outgoing,
                                'switch_incoming_penalty': - W['switch_incoming_penalty']*incoming_total,
                            }
                        })
                    except Exception: continue
                switches_eval.sort(key=lambda x: x.get('score',0.0), reverse=True)

            # depth augmentation (tree search including switches). Depth = number of full future turns (both sides acting) to evaluate (>=1 already evaluated base turn).
            if self._depth > 1 and (moves_eval or switches_eval):
                # Gather opponent damaging moves (limit by branching)
                opp_moves_full = []
                try:
                    if opp_key:
                        opp_ps = state.team.opponent[opp_key]
                        for mv in (getattr(opp_ps,'moves',[]) or []):
                            mid = getattr(mv,'id',None)
                            if not mid: continue
                            if (getattr(mv,'category','') or '').lower()=='status' or (getattr(mv,'base_power',0) or getattr(mv,'basePower',0) or 0) <= 0:
                                continue
                            try: exp_frac_o, dmg_o = _expected_damage_fraction(state, opp_key, my_key, mid, mi)
                            except Exception: exp_frac_o, dmg_o = 0.0, {}
                            # Lookup opponent move category for future defensive scaling
                            try:
                                _ominf = mi.get(str(mid))
                                _ocat = (_ominf.category or 'Physical').lower()
                            except Exception:
                                _ocat = 'physical'
                            opp_moves_full.append({'id': mid, 'name': getattr(mv,'name',mid), 'exp': exp_frac_o, 'acc': _acc_to_prob(getattr(mv,'accuracy',1.0)), 'category': _ocat})
                except Exception:
                    pass
                if not opp_moves_full:
                    try: base_in = _opp_best_on_target(state, opp_key, my_key, mi)
                    except Exception: base_in = 0.0
                    opp_moves_full = [{'id':'_synthetic','name':'(opp_best)','exp': base_in, 'acc':1.0, 'category': 'physical'}]
                opp_moves = opp_moves_full[:self._branching]

                # Baseline HP fractions
                try: my_hp_now = _hp_frac(state.team.ours[my_key])
                except Exception: my_hp_now = 1.0
                try: opp_hp_now = _hp_frac(state.team.opponent[opp_key])
                except Exception: opp_hp_now = 1.0

                W = self._W

                # Build our action list (top branching moves + top branching switches)
                my_actions: List[Dict[str, Any]] = []
                for mv in moves_eval[:self._branching]:
                    my_actions.append({'kind':'move','ref':mv})
                for sw in switches_eval[:self._branching]:
                    my_actions.append({'kind':'switch','ref':sw})

                # --- NEW: Precompute per-target expectations for dynamic tree (our moves vs each opponent candidate, opp moves vs our candidates) ---
                # Collect alive opponent keys (active + bench)
                opp_alive_keys: List[str] = []
                try:
                    for k,ps in state.team.opponent.items():
                        if getattr(ps,'current_hp',0) > 0 and (getattr(ps,'status','') or '').lower()!='fnt':
                            opp_alive_keys.append(k)
                except Exception:
                    opp_alive_keys = [opp_key] if opp_key else []
                if opp_key and opp_key not in opp_alive_keys:
                    opp_alive_keys.append(opp_key)
                # Collect our alive keys
                my_alive_keys: List[str] = []
                try:
                    for k,ps in state.team.ours.items():
                        if getattr(ps,'current_hp',0) > 0 and (getattr(ps,'status','') or '').lower()!='fnt':
                            my_alive_keys.append(k)
                except Exception:
                    my_alive_keys = [my_key] if my_key else []
                if my_key and my_key not in my_alive_keys:
                    my_alive_keys.append(my_key)

                # Precompute mapping for our moves: expected damage vs each opponent possible active
                for act in my_actions:
                    if act['kind']!='move':
                        continue
                    ref = act['ref']
                    mv_id = ref.get('id')
                    exp_map = {}
                    for ok in opp_alive_keys:
                        try:
                            frac,_ = _expected_damage_fraction(state, my_key, ok, str(mv_id), mi)
                        except Exception:
                            frac = 0.0
                        exp_map[ok] = float(frac) * float(ref.get('acc',1.0))  # include accuracy
                    ref['exp_by_target'] = exp_map
                # For switches we record no direct damage, but keep placeholder map
                for act in my_actions:
                    if act['kind']=='switch':
                        act['ref']['exp_by_target'] = {ok:0.0 for ok in opp_alive_keys}

                # Precompute opponent moves vs each of our possible actives (for when we have switched previously in tree)
                for o in opp_moves_full:
                    exp_map_my = {}
                    for mk in my_alive_keys:
                        try:
                            frac,_ = _expected_damage_fraction(state, opp_key, mk, str(o['id']), mi)
                        except Exception:
                            frac = 0.0
                        exp_map_my[mk] = float(frac) * float(o.get('acc',1.0))
                    o['exp_by_target_my'] = exp_map_my

                # Opponent potential switches (bench keys excluding current active)
                opp_switch_candidates: List[Dict[str, Any]] = []
                try:
                    for k in opp_alive_keys:
                        if k == opp_key:
                            continue
                        ps = state.team.opponent.get(k)
                        if not ps: continue
                        opp_switch_candidates.append({'to_key':k, 'hp_frac': _hp_frac(ps)})
                except Exception:
                    pass

                W = self._W
                temp_soft = self._softmin_temp
                branching_cap = self._branching

                # Dynamic recursion with active keys and HP remaining
                from functools import lru_cache

                def _stage_mult(delta: int) -> float:
                    try:
                        n = int(delta)
                    except Exception:
                        return 1.0
                    if n >= 0:
                        return (2 + n) / 2
                    else:
                        # e.g., n=-1 => 2/(2-(-1)) = 2/3
                        return 2 / (2 - n)

                @lru_cache(maxsize=8192)
                def recurse(active_my: str, active_opp: str, my_rem: float, opp_rem: float, turns_left: int,
                            my_phys_scale: float=1.0, my_spec_scale: float=1.0,
                            opp_phys_scale: float=1.0, opp_spec_scale: float=1.0) -> float:
                    if turns_left <= 0 or my_rem <= 0 or opp_rem <= 0:
                        if tree_trace_enabled:
                            tree_trace.append(f"[TREE][BASE] tl={turns_left} my={my_rem:.3f} opp={opp_rem:.3f} act_my={active_my} act_opp={active_opp} -> 0.000")
                        return 0.0
                    best_val = None
                    # Rebuild candidate action lists each depth (filter out switches to already-active)
                    current_actions: List[Dict[str, Any]] = []
                    # Reuse precomputed stats but adjust for active key changes (switching changes active_my reference for opp damage lookups later)
                    move_count = 0
                    switch_count = 0
                    for act in my_actions:
                        if act['kind']=='move':
                            if move_count < branching_cap:
                                current_actions.append(act)
                                move_count += 1
                        else:
                            # skip switching to same active
                            species = act['ref'].get('species')
                            # Identify key of that species
                            if switch_count < branching_cap:
                                current_actions.append(act)
                                switch_count += 1
                    # Opponent replies: damaging moves + switch-ins
                    opp_reply_moves = opp_moves_full[:branching_cap]
                    opp_reply_switches = opp_switch_candidates[:branching_cap]

                    for act in current_actions:
                        kind = act['kind']; ref = act['ref']
                        # Determine our expected damage vs current opponent (if move) using per-target exp map
                        exp_map = ref.get('exp_by_target') or {}
                        # Apply our current offense scaling by move category
                        mv_cat = str(ref.get('category','status')).lower()
                        scale_now = 1.0
                        if mv_cat == 'physical':
                            scale_now = float(my_phys_scale)
                        elif mv_cat == 'special':
                            scale_now = float(my_spec_scale)
                        our_exp_vs_current = (float(exp_map.get(active_opp, 0.0)) * scale_now) if kind=='move' else 0.0
                        first_prob = float(ref.get('first_prob',0.5)) if kind=='move' else 0.0
                        if tree_trace_enabled:
                            ident = ref.get('id') if kind=='move' else ref.get('species')
                            tree_trace.append(f"[TREE][ACT] tl={turns_left} my={my_rem:.3f} opp={opp_rem:.3f} kind={kind} ident={ident} our_exp={our_exp_vs_current:.3f} first={first_prob:.3f} scaleP={my_phys_scale:.2f} scaleS={my_spec_scale:.2f}")
                        reply_vals: List[float] = []
                        # 1. Opponent move replies
                        for o in opp_reply_moves:
                            opp_exp_map = o.get('exp_by_target_my') or {}
                            opp_exp_base = float(opp_exp_map.get(active_my, 0.0))
                            # Adjust by our defensive/debuff scaling based on opponent move category
                            ocat = str(o.get('category','physical')).lower()
                            opp_scale_now = float(opp_phys_scale if ocat=='physical' else opp_spec_scale)
                            opp_exp_scaled = opp_exp_base * opp_scale_now
                            # If we switched this turn, opponent move hits new active later (we take full damage); if we moved, speed determines
                            if kind=='move':
                                # KO check vs current active opponent
                                ko_if_hit = our_exp_vs_current >= opp_rem - 1e-9
                                opp_effective = opp_exp_scaled * ((1-first_prob) + (first_prob * (0 if ko_if_hit else 1)))
                                our_effective = our_exp_vs_current
                                inc = (W['expected_mult']*our_effective - W['opp_dmg_penalty']*opp_effective + W['go_first_bonus']*first_prob)
                                if ko_if_hit:
                                    inc += W['ko_bonus']
                                next_my = max(0.0, my_rem - opp_effective)
                                next_opp = max(0.0, opp_rem - our_effective)
                                # Update scales if this move is a self-boost or debuffing move
                                next_my_phys_scale = my_phys_scale
                                next_my_spec_scale = my_spec_scale
                                next_opp_phys_scale = opp_phys_scale
                                next_opp_spec_scale = opp_spec_scale
                                if mv_cat == 'status':
                                    sb = ref.get('self_boosts') or {}
                                    if 'atk' in sb:
                                        next_my_phys_scale *= _stage_mult(int(sb['atk']))
                                    if 'spa' in sb:
                                        next_my_spec_scale *= _stage_mult(int(sb['spa']))
                                    if 'def' in sb:
                                        # Increasing our Defense reduces incoming physical damage => divide opponent physical scale by this factor
                                        next_opp_phys_scale /= _stage_mult(int(sb['def']))
                                    if 'spd' in sb:
                                        next_opp_spec_scale /= _stage_mult(int(sb['spd']))
                                    # Opponent offensive debuffs from our status move
                                    tdb = ref.get('target_debuffs') or {}
                                    if 'atk' in tdb:
                                        next_opp_phys_scale *= _stage_mult(int(tdb['atk']))
                                    if 'spa' in tdb:
                                        next_opp_spec_scale *= _stage_mult(int(tdb['spa']))
                                future = recurse(active_my, active_opp, next_my, next_opp, turns_left - 1,
                                                 next_my_phys_scale, next_my_spec_scale,
                                                 next_opp_phys_scale, next_opp_spec_scale) if next_my>0 and next_opp>0 else 0.0
                            else:  # we switched => no damage dealt, incoming based on opponent move vs NEW active (we need to know which we switched to)
                                # Approximation: use current active_my still (we are not modeling identity change)
                                opp_effective = opp_exp_scaled
                                inc = - W['opp_dmg_penalty']*opp_effective
                                our_effective = 0.0
                                next_my = max(0.0, my_rem - opp_effective)
                                next_opp = opp_rem
                                # Switching clears our previous boosts/debuffs context
                                future = recurse(active_my, active_opp, next_my, next_opp, turns_left - 1,
                                                 1.0, 1.0, opp_phys_scale, opp_spec_scale) if next_my>0 else 0.0
                            total = inc + future
                            reply_vals.append(total)
                            if tree_trace_enabled:
                                tree_trace.append(f"[TREE][RPLY] tl={turns_left} act_kind={kind} opp_mv={o['id']} opp_exp={opp_exp_base:.3f} opp_eff={(opp_effective if kind=='move' else opp_exp_scaled):.3f} inc={inc:.3f} future={future:.3f} total={total:.3f} next_my={next_my:.3f} next_opp={next_opp:.3f}")
                        # 2. Opponent switch replies
                        for sw in opp_reply_switches:
                            to_key = sw['to_key']; to_hp_full = float(sw['hp_frac'])
                            if kind=='move':
                                # Our move will hit the new Pokemon after switch (switch happens first). Use damage vs that target.
                                our_exp_vs_new = float(exp_map.get(to_key, 0.0))
                                # Apply our offense scaling by this move's category
                                our_exp_vs_new *= scale_now
                                # No damage taken from opponent this turn
                                inc = W['expected_mult']*our_exp_vs_new + W['go_first_bonus']*first_prob  # we still get go-first bonus heuristic
                                ko_if_hit_new = our_exp_vs_new >= to_hp_full - 1e-9
                                if ko_if_hit_new:
                                    inc += W['ko_bonus']
                                next_my = my_rem
                                next_opp = max(0.0, to_hp_full - our_exp_vs_new)
                                # Update scales for future if this was a boosting/debuffing status move
                                next_my_phys_scale = my_phys_scale
                                next_my_spec_scale = my_spec_scale
                                # On opponent switch, reset their debuffs but keep our own Def/SpD boosts as incoming reduction
                                new_opp_phys_scale = 1.0
                                new_opp_spec_scale = 1.0
                                if mv_cat == 'status':
                                    sb = ref.get('self_boosts') or {}
                                    if 'atk' in sb:
                                        next_my_phys_scale *= _stage_mult(int(sb['atk']))
                                    if 'spa' in sb:
                                        next_my_spec_scale *= _stage_mult(int(sb['spa']))
                                    if 'def' in sb:
                                        new_opp_phys_scale /= _stage_mult(int(sb['def']))
                                    if 'spd' in sb:
                                        new_opp_spec_scale /= _stage_mult(int(sb['spd']))
                                    # Opponent offensive debuffs do not persist through their switch -> intentionally ignored
                                else:
                                    # No new boost this turn; keep any existing incoming reduction from our prior Def/SpD scales
                                    # Approximation: propagate current incoming reduction if it came from our Def/SpD boosts
                                    # We cannot disentangle prior debuffs here, so reset to neutral
                                    new_opp_phys_scale = 1.0
                                    new_opp_spec_scale = 1.0
                                future = recurse(active_my, to_key, next_my, next_opp, turns_left - 1,
                                                 next_my_phys_scale, next_my_spec_scale,
                                                 new_opp_phys_scale, new_opp_spec_scale) if next_opp>0 else 0.0
                            else:
                                # Double switch: no damage this turn
                                inc = 0.0
                                next_my = my_rem
                                next_opp = to_hp_full
                                # Our switch resets our own boost context; opponent switching resets theirs as well
                                future = recurse(active_my, to_key, next_my, next_opp, turns_left - 1,
                                                 1.0, 1.0, 1.0, 1.0)
                            total = inc + future
                            reply_vals.append(total)
                            if tree_trace_enabled:
                                tree_trace.append(f"[TREE][RPLY] tl={turns_left} act_kind={kind} opp_switch={to_key} inc={inc:.3f} future={future:.3f} total={total:.3f} next_my={next_my:.3f} next_opp={next_opp:.3f}")
                        if not reply_vals:
                            continue
                        if temp_soft>0:
                            import math
                            weights = [math.exp(-x/temp_soft) for x in reply_vals]
                            Z = sum(weights) or 1.0
                            agg_val = sum(w*x for w,x in zip(weights, reply_vals))/Z
                            agg_type = 'softmin'
                        else:
                            agg_val = min(reply_vals)
                            agg_type = 'min'
                        if best_val is None or agg_val>best_val:
                            best_val = agg_val
                        if tree_trace_enabled:
                            ident = ref.get('id') if kind=='move' else ref.get('species')
                            tree_trace.append(f"[TREE][EVAL] tl={turns_left} kind={kind} ident={ident} agg={agg_type} val={agg_val:.3f} best={best_val:.3f}")
                    final_val = best_val if best_val is not None else 0.0
                    if tree_trace_enabled:
                        tree_trace.append(f"[TREE][RET] tl={turns_left} my={my_rem:.3f} opp={opp_rem:.3f} act_my={active_my} act_opp={active_opp} -> {final_val:.3f}")
                    return final_val

                # Apply recursion to refine score_depth & future_proj (root actives)
                turns_left = self._depth - 1
                for act in my_actions:
                    ref = act['ref']
                    base_score = float(ref.get('score',0.0))
                    val = recurse(my_key, opp_key, my_hp_now, opp_hp_now, turns_left, 1.0, 1.0, 1.0, 1.0)
                    if val is not None:
                        ref['score_depth'] = val
                        ref['future_proj'] = val - base_score
                        ref['depth_used'] = self._depth
                        ref.setdefault('tree_minimax','maximin')
                        if self._softmin_temp > 0:
                            ref.setdefault('tree_opponent_response','softmin')
                            ref.setdefault('softmin_temp', self._softmin_temp)
                        if act['kind']=='move':
                            ref.setdefault('tree_includes_switches', True)
                # Resort after depth
                moves_eval.sort(key=lambda x: x.get('score_depth', x.get('score',0.0)), reverse=True)
                switches_eval.sort(key=lambda x: x.get('score_depth', x.get('score', x.get('base_score',0.0))), reverse=True)

        # NEW: If not a forced switch and we have legal moves but couldn't resolve opponent, build a simple fallback move ranking to avoid random switching
        elif not force_switch and my_key and legal_moves:
            skip_reason = 'opp_key=None'
            try:
                me = state.team.ours.get(my_key)
            except Exception:
                me = None
            my_types = set(getattr(me, 'types', []) or []) if me else set()
            for mv in legal_moves:
                mid = getattr(mv,'id',None) or getattr(mv,'move_id',None)
                if not mid: continue
                raw = mi.get(str(mid))
                cat = (raw.category or 'Status').lower()
                bp = int(raw.base_power or 0)
                acc = _acc_to_prob(raw.accuracy)
                mv_type = (raw.type or '').lower()
                # Normalize my_types for membership safely
                my_types_lc = {str(t).lower() for t in my_types if t}
                stab = 1.5 if (mv_type and (mv_type in my_types_lc) and cat in {'physical','special'}) else 1.0
                # status moves score low; damaging moves scale with power, STAB and accuracy (normalized)
                base = 0.0
                if cat in {'physical','special'} and bp > 0:
                    base = (bp/100.0) * stab * acc
                moves_eval.append({
                    'id': str(mid), 'name': getattr(mv,'name', mid),
                    'score': float(base), 'score_depth': float(base), 'future_proj': 0.0, 'depth_used': 1,
                    'expected': 0.0, 'exp_dmg': 0.0, 'acc': float(acc), 'effectiveness': 1.0,
                    'first_prob': 0.5, 'p_ko_if_hit': 0.0, 'p_ko_first': 0.0,
                    'opp_counter_ev': 0.0, 'incoming_frac': 0.0,
                    'score_breakdown': {
                        'expected': base,
                        'opp_dmg_penalty': 0.0,
                        'go_first_bonus': 0.0,
                        'effectiveness_mult': 0.0,
                        'accuracy_mult': 0.0,
                        'ko_bonus': 0.0,
                    }
                })
            moves_eval.sort(key=lambda x: x.get('score_depth', x.get('score',0.0)), reverse=True)
            # When we don't know the opponent, avoid considering switches (prevents pointless cycling)
            switches_eval = []

        # If my_key is still None but we have legal moves, build an ultra-safe fallback using active_pokemon types
        elif not force_switch and (not my_key) and legal_moves:
            skip_reason = 'my_key=None'
            # Try to infer our types from active_pokemon
            my_types_lc = set()
            try:
                ap = getattr(battle, 'active_pokemon', None)
                if ap is not None:
                    for t in (getattr(ap,'types',[]) or []):
                        if not t: continue
                        try:
                            my_types_lc.add(str(t).lower())
                        except Exception:
                            pass
            except Exception:
                pass
            for mv in legal_moves:
                mid = getattr(mv,'id',None) or getattr(mv,'move_id',None)
                if not mid: continue
                raw = mi.get(str(mid))
                cat = (raw.category or 'Status').lower()
                bp = int(raw.base_power or 0)
                acc = _acc_to_prob(raw.accuracy)
                mv_type = (raw.type or '').lower()
                stab = 1.5 if (mv_type and (mv_type in my_types_lc) and cat in {'physical','special'}) else 1.0
                base = 0.0
                if cat in {'physical','special'} and bp > 0:
                    base = (bp/100.0) * stab * acc
                moves_eval.append({
                    'id': str(mid), 'name': getattr(mv,'name', mid),
                    'score': float(base), 'score_depth': float(base), 'future_proj': 0.0, 'depth_used': 1,
                    'expected': 0.0, 'exp_dmg': 0.0, 'acc': float(acc), 'effectiveness': 1.0,
                    'first_prob': 0.5, 'p_ko_if_hit': 0.0, 'p_ko_first': 0.0,
                    'opp_counter_ev': 0.0, 'incoming_frac': 0.0,
                    'score_breakdown': {
                        'expected': base,
                        'opp_dmg_penalty': 0.0,
                        'go_first_bonus': 0.0,
                        'effectiveness_mult': 0.0,
                        'accuracy_mult': 0.0,
                        'ko_bonus': 0.0,
                    }
                })
            moves_eval.sort(key=lambda x: x.get('score_depth', x.get('score',0.0)), reverse=True)
            switches_eval = []

        # If we are in a force switch situation (or move eval skipped) we may still need switch evaluations
        if not switches_eval and legal_switches:
            try:
                active_species_lower = ''
                if my_key and my_key in state.team.ours and getattr(state.team.ours[my_key],'current_hp',0)>0:
                    active_species_lower = (getattr(state.team.ours[my_key], 'species', '') or '').lower()
                for sw in legal_switches:
                    species = str(getattr(sw,'species','') or '')
                    species_lower = species.lower()
                    if species_lower == active_species_lower:
                        continue  # do not switch to same
                    cand_key=None
                    for k,ps in state.team.ours.items():
                        if str(getattr(ps,'species','')).lower()==species_lower:
                            cand_key=k; break
                    if not cand_key:
                        continue
                    cand_hp = _hp_frac(state.team.ours[cand_key])
                    # Hazards on switch-in
                    haz_frac = 0.0
                    try:
                        haz = apply_switch_in_effects(state, cand_key, 'ally', mi, mutate=False)
                        haz_frac = float(haz.get('fraction_lost', 0.0) or 0.0)
                    except Exception:
                        haz_frac = 0.0
                    # Incoming damage expectation (if opponent active known)
                    try:
                        incoming = _opp_best_on_target(state, opp_key, cand_key, mi) if opp_key else 0.0
                    except Exception:
                        incoming = 0.0
                    incoming_total = float(haz_frac) + float(incoming)
                    outgoing = 0.0
                    if opp_key:
                        try:
                            for mv_obj in getattr(state.team.ours[cand_key], 'moves', []) or []:
                                mid2 = getattr(mv_obj,'id',None)
                                if not mid2: continue
                                bp2 = int(getattr(mv_obj,'base_power',0) or getattr(mv_obj,'basePower',0) or 0)
                                if (getattr(mv_obj,'category','') or '').lower()=='status' or bp2<=0: continue
                                frac2,_ = _expected_damage_fraction(state, cand_key, opp_key, mid2, mi)
                                outgoing = max(outgoing, frac2*_acc_to_prob(getattr(mv_obj,'accuracy',None)))
                        except Exception:
                            pass
                    W=self._W
                    score = W['switch_outgoing_mult']*outgoing - W['switch_incoming_penalty']*incoming_total
                    switches_eval.append({
                        'species':species,'score':float(score),'base_score':float(score),
                        'outgoing_frac':float(outgoing),'incoming_on_switch':float(incoming_total),
                        'hazards_frac':float(haz_frac),'hp_fraction':float(cand_hp),
                        'score_breakdown': {
                            'switch_outgoing_mult': W['switch_outgoing_mult']*outgoing,
                            'switch_incoming_penalty': - W['switch_incoming_penalty']*incoming_total,
                        }
                    })
                switches_eval.sort(key=lambda x: x.get('score',0.0), reverse=True)
            except Exception:
                pass
            if force_switch and not switches_eval:
                skip_reason = (skip_reason or '') + '|force_switch_no_candidates_after_synth'

        # Simplified switch evaluation
        # (Removed duplicate; switches_eval already built earlier for tree search or fallback above)
        # --- Lethal / last-mon heuristics ---
        active_hp_frac=None; predicted_incoming_frac=None; lethal_before_action=False; lethal_risk=0.0
        if my_key and opp_key:
            try: active_hp_frac = _hp_frac(state.team.ours[my_key])
            except Exception: active_hp_frac = None
            try: predicted_incoming_frac = _opp_best_on_target(state, opp_key, my_key, mi)
            except Exception: predicted_incoming_frac = None
            if active_hp_frac is not None and predicted_incoming_frac is not None and predicted_incoming_frac >= active_hp_frac - 1e-6:
                lethal_before_action = True
        our_remaining = 0; opp_remaining = 0
        try: our_remaining = sum(1 for ps in state.team.ours.values() if getattr(ps,'current_hp',0)>0)
        except Exception: pass
        try: opp_remaining = sum(1 for ps in state.team.opponent.values() if getattr(ps,'current_hp',0)>0)
        except Exception: pass
        best_move = moves_eval[0] if moves_eval else None
        best_switch = switches_eval[0] if switches_eval else None
        if lethal_before_action and best_move and active_hp_frac:
            first_prob = float(best_move.get('first_prob',0.5) or 0.5)
            p_ko_if_hit_by_opp = 0.0
            if predicted_incoming_frac is not None and active_hp_frac>0:
                p_ko_if_hit_by_opp = min(1.0, predicted_incoming_frac/active_hp_frac)
            lethal_risk = p_ko_if_hit_by_opp * (1-first_prob)
            can_switch = our_remaining>1 and bool(switches_eval)
            if lethal_risk >= 0.6 and first_prob < 0.5 and can_switch:
                # Penalize non-priority non-KO move
                if best_move.get('p_ko_first',0.0) < 0.9:
                    best_move['score_depth'] = -999.0
                    best_move['score'] = -999.0
                    best_move.setdefault('heuristics',{})['lethal_risk_penalty'] = lethal_risk
                    moves_eval.sort(key=lambda x: x.get('score_depth', x.get('score',0.0)), reverse=True)
                    best_move = moves_eval[0]
            # Conservative extra: low HP and likely to be chunked hard even if not full lethal -> prefer survivable switch when we likely don't go first
            if can_switch and best_switch and (first_prob <= 0.3 or lethal_risk >= 0.5):
                low_hp = (active_hp_frac <= 0.4)
                heavy_chunk = (predicted_incoming_frac or 0.0) >= 0.8 * (active_hp_frac or 1.0)
                if low_hp and (heavy_chunk or lethal_risk >= 0.6):
                    if best_switch.get('incoming_on_switch',1.0) < (active_hp_frac or 1.0):
                        best_switch['score'] = max(best_switch['score'], (best_move.get('score_depth', best_move.get('score',0.0)) or 0.0)+0.01)
                        best_switch.setdefault('heuristics',{})['low_hp_survival_preference']=True
        # last mon aggression
        if our_remaining <= 1 and moves_eval:
            moves_eval.sort(key=lambda x: (x.get('expected',0.0)*x.get('acc',1.0)), reverse=True)
            best_move = moves_eval[0]

        snap = snapshot_battle(battle)

        def _debug_base():
            return {
                'candidates': moves_eval,
                'switches': switches_eval,
                'switch_meta': {},
                'snapshot': snap,
                'active_hp_frac': active_hp_frac,
                'predicted_incoming_frac': predicted_incoming_frac,
                'lethal_before_action': lethal_before_action,
                'lethal_risk': lethal_risk,
                'our_remaining': our_remaining,
                'opp_remaining': opp_remaining,
                'depth': self._depth,
                'branching': self._branching,
                'softmin_temp': self._softmin_temp,
                'tree_trace': tree_trace[:5000],  # cap to avoid runaway size
            }

        # Forced switch
        # Ensure local vars exist even if earlier evaluation failed
        if 'best_move' not in locals():
            best_move = None  # type: ignore
        if 'best_switch' not in locals():
            best_switch = None  # type: ignore
        
        # Phase 1: Switch validation before execution
        def _validate_switch(switch_species: str) -> bool:
            """Validate that a switch is legal before execution."""
            try:
                if not switch_species:
                    return False
                
                # Check if we have any legal switches
                if not legal_switches:
                    return False
                
                # Check if target species is in available switches
                for sw in legal_switches:
                    if str(getattr(sw, 'species', '')).lower() == str(switch_species).lower():
                        # Additional validation: check if Pokemon is alive
                        if my_key and state and hasattr(state, 'team'):
                            for k, ps in state.team.ours.items():
                                if str(getattr(ps, 'species', '')).lower() == str(switch_species).lower():
                                    current_hp = getattr(ps, 'current_hp', 0)
                                    status = (getattr(ps, 'status', '') or '').lower()
                                    if current_hp > 0 and status != 'fnt':
                                        return True
                        else:
                            # Fallback: if we can't check team state, trust legal_switches
                            return True
                
                return False
            except Exception:
                # On validation error, be conservative and reject
                return False
        
        if force_switch and best_switch:
            # Validate switch before execution
            if _validate_switch(best_switch['species']):
                dbg = _debug_base(); dbg['picked']={'kind':'switch', **best_switch}
                decision = ChosenAction(kind='switch', switch_species=best_switch['species'], debug=dbg)
            else:
                # Switch validation failed, fallback to first legal switch
                dbg = _debug_base(); dbg['fallback']=True; dbg['switch_validation_failed']=True
                if legal_switches:
                    fallback_species = str(getattr(legal_switches[0], 'species', ''))
                    decision = ChosenAction(kind='switch', switch_species=fallback_species, debug=dbg)
                else:
                    decision = ChosenAction(kind='move', move_id='struggle', debug=dbg)
        else:
            # Choose move vs switch (margin)
            MARGIN = 0.05
            if best_move and (not best_switch or best_move.get('score_depth', best_move.get('score',0.0)) >= best_switch.get('score',0.0)+MARGIN):
                dbg = _debug_base();
                dbg['picked']={'kind':'move', **best_move};
                # Enrich order + speed meta for UI/telemetry
                order_info = {
                    'p_user_first': float(best_move.get('first_prob', 0.5))
                }
                try:
                    od = best_move.get('order_details') or {}
                    if od:
                        order_info.update({
                            'user_bracket': od.get('user_bracket'),
                            'opp_bracket': od.get('opp_bracket'),
                            'notes': od.get('notes'),
                        })
                except Exception:
                    pass
                dbg['order'] = order_info
                try:
                    dbg['speed_meta'] = {
                        'my_effective_speed': int(my_eff_spe),
                        'opp_effective_speed': int(opp_eff_spe),
                        'trick_room': bool(trick_room_active),
                    }
                except Exception:
                    pass
                decision = ChosenAction(kind='move', move_id=str(best_move['id']), debug=dbg)
            elif best_switch:
                # Validate optional switch before execution
                if _validate_switch(best_switch['species']):
                    dbg = _debug_base(); dbg['picked']={'kind':'switch', **best_switch}
                    decision = ChosenAction(kind='switch', switch_species=best_switch['species'], debug=dbg)
                else:
                    # Switch validation failed, fallback to best move or first legal action
                    if best_move:
                        dbg = _debug_base(); dbg['picked']={'kind':'move', **best_move}; dbg['switch_validation_failed']=True
                        decision = ChosenAction(kind='move', move_id=str(best_move['id']), debug=dbg)
                    elif legal_moves:
                        dbg = _debug_base(); dbg['fallback']=True; dbg['switch_validation_failed']=True
                        decision = ChosenAction(kind='move', move_id=str(getattr(legal_moves[0],'id','')), debug=dbg)
                    else:
                        dbg = _debug_base(); dbg['fallback']=True; dbg['switch_validation_failed']=True
                        decision = ChosenAction(kind='move', move_id='struggle', debug=dbg)
            else:
                # Fallbacks
                if legal_moves:
                    dbg = _debug_base(); dbg['fallback']=True
                    decision = ChosenAction(kind='move', move_id=str(getattr(legal_moves[0],'id','')), debug=dbg)
                elif legal_switches:
                    dbg = _debug_base(); dbg['fallback']=True
                    decision = ChosenAction(kind='switch', switch_species=str(getattr(legal_switches[0],'species','')), debug=dbg)
                else:
                    dbg = _debug_base(); dbg['fallback']=True
                    decision = ChosenAction(kind='move', move_id='struggle', debug=dbg)

        # ---- Instrumentation logging ----
        try:
            top_moves = []
            for m in (moves_eval[:3] if moves_eval else []):
                top_moves.append({
                    'id': m.get('id'), 'score': round(float(m.get('score_depth', m.get('score',0.0))),4),
                    'exp': round(float(m.get('expected',0.0)),4),
                    'first': round(float(m.get('first_prob',0.0)),3),
                    'p_ko_first': round(float(m.get('p_ko_first',0.0)),3),
                    'opp_ev': round(float(m.get('opp_counter_ev',0.0)),4)
                })
            top_switches = []
            for s in (switches_eval[:2] if switches_eval else []):
                top_switches.append({
                    'species': s.get('species'), 'score': round(float(s.get('score',0.0)),4),
                    'out': round(float(s.get('outgoing_frac', s.get('outgoing',0.0))),4),
                    'in': round(float(s.get('incoming_on_switch', s.get('incoming_frac',0.0))),4),
                })
            picked = decision.debug.get('picked') if isinstance(decision.debug, dict) else None
            _dec_log.update({
                'active_hp_frac': decision.debug.get('active_hp_frac') if isinstance(decision.debug, dict) else None,
                'predicted_incoming_frac': decision.debug.get('predicted_incoming_frac') if isinstance(decision.debug, dict) else None,
                'lethal_before_action': decision.debug.get('lethal_before_action') if isinstance(decision.debug, dict) else None,
                'lethal_risk': decision.debug.get('lethal_risk') if isinstance(decision.debug, dict) else None,
                'our_remaining': decision.debug.get('our_remaining') if isinstance(decision.debug, dict) else None,
                'opp_remaining': decision.debug.get('opp_remaining') if isinstance(decision.debug, dict) else None,
                'top_moves': top_moves,
                'top_switches': top_switches,
                'chosen': {'kind': decision.kind, 'move_id': decision.move_id, 'switch': decision.switch_species, 'picked': picked},
            })
            logging.getLogger('Decision').info('DECISION turn=%s payload=%s', _dec_log.get('turn'), json.dumps(_dec_log, default=str))
        except Exception:
            pass
        # Verbose terminal debugging of the full think process
        if getattr(self, '_verbose', False):
            try:
                # Re-check env each call so user can toggle externally mid-run
                import os as _os
                if not self._verbose and _os.environ.get('POKECHAD_THINK_DEBUG') in ('1','true','TRUE','True'):
                    self._verbose = True
                turn = getattr(battle,'turn', None)
                tag = getattr(battle,'battle_tag', getattr(battle,'room_id', ''))
                header = f"[THINK][Turn {turn}][{tag}] Depth={self._depth} Branching={self._branching}"
                lines = ["", header]
                if moves_eval:
                    lines.append(" Moves (sorted):")
                    for mv in moves_eval:
                        bd = mv.get('score_breakdown', {})
                        comp_str = ' '.join(f"{k}={v:+.3f}" for k,v in bd.items())
                        future = mv.get('future_proj',0.0)
                        depth_score = mv.get('score_depth', mv.get('score'))
                        lines.append(("  - {name:<18} id={id:<12} S={s:+.3f} DS={ds:+.3f} Fut={fut:+.3f} Exp={exp:.3f} Acc={acc:.2f} Eff={eff:.2f} First={first:.2f} KOFirst={kof:.3f} OppEV={opp:.3f} -> {comp}" )
                                      .format(name=str(mv.get('name',''))[:18], id=str(mv.get('id',''))[:12], s=float(mv.get('score',0.0)), ds=float(depth_score), fut=float(future), exp=float(mv.get('expected',0.0)), acc=float(mv.get('acc',0.0)), eff=float(mv.get('effectiveness',0.0)), first=float(mv.get('first_prob',0.0)), kof=float(mv.get('p_ko_first',0.0)), opp=float(mv.get('opp_counter_ev',0.0)), comp=comp_str))
                else:
                    # Build a more specific reason string
                    if not skip_reason:
                        if force_switch: skip_reason = f"force_switch={force_switch}"
                        elif not my_key: skip_reason = "my_key=None"
                        elif not opp_key: skip_reason = "opp_key=None"
                        elif legal_moves: skip_reason = "eval_guard_failed"
                        else: skip_reason = "no legal moves"
                    lines.append(f" No move candidates ({skip_reason})")
                    if legal_moves:
                        lines.append(f"  DEBUG flags: force_switch={force_switch} my_key={my_key} opp_key={opp_key} #legal_moves={len(legal_moves)} #legal_switches={len(legal_switches)}")
                if switches_eval:
                    lines.append(" Switch candidates:")
                    for sw in switches_eval:
                        bd = sw.get('score_breakdown', {})
                        comp_str = ' '.join(f"{k}={v:+.3f}" for k,v in bd.items())
                        lines.append(("  - {sp:<18} S={s:+.3f} Out={out:.3f} In={inn:.3f} HP={hp:.2f} Haz={haz:.3f} -> {comp}")
                                      .format(sp=str(sw.get('species',''))[:18], s=float(sw.get('score',0.0)), out=float(sw.get('outgoing_frac',0.0)), inn=float(sw.get('incoming_on_switch',0.0)), hp=float(sw.get('hp_fraction',0.0)), haz=float(sw.get('hazards_frac',0.0)), comp=comp_str))
                else:
                    lines.append(" No switch candidates")
                # Tree trace printing (requires both verbose and env POKECHAD_TREE_TRACE=1)
                try:
                    tree_trace_env = os.environ.get('POKECHAD_TREE_TRACE','0')
                except Exception:
                    tree_trace_env = '0'
                if str(tree_trace_env) in ('1','true','TRUE','True'):
                    trace_list = decision.debug.get('tree_trace') if isinstance(decision.debug, dict) else None
                    if trace_list:
                        try:
                            max_lines = int(os.environ.get('POKECHAD_TREE_TRACE_MAX','300'))
                        except Exception:
                            max_lines = 300
                        lines.append(f" Tree trace (showing up to {max_lines} of {len(trace_list)} lines):")
                        for tl in trace_list[:max_lines]:
                            lines.append("   " + tl)
                        if len(trace_list) > max_lines:
                            lines.append(f"   ... ({len(trace_list)-max_lines} more lines truncated)")
                pk = decision.debug.get('picked') if isinstance(decision.debug, dict) else {}
                if pk:
                    if pk.get('kind')=='move':
                        lines.append(f" -> PICK MOVE {pk.get('name')} (id={pk.get('id')}) score={pk.get('score')} depth_score={pk.get('score_depth', pk.get('score'))}")
                    else:
                        lines.append(f" -> PICK SWITCH {pk.get('species')} score={pk.get('score')}")
                if decision.debug.get('lethal_before_action'):
                    lines.append(f"  Lethal risk detected: risk={decision.debug.get('lethal_risk'):.3f}")
                # Emit via print (flush) and logging so UI log tab captures it
                logger = logging.getLogger('ThinkVerbose')
                for ln in lines:
                    print(ln, flush=True)
                    try: logger.info(ln)
                    except Exception: pass
            except Exception:
                pass
        return decision

# -------------- Poke-env Player wrapper ----------------
try:
    from poke_env.player.player import Player  # type: ignore
except Exception:
    Player = object  # fallback

class StockfishPokeEnvPlayer(Player):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        self.on_think_hook = kwargs.pop('on_think', None)
        engine_depth = kwargs.pop('engine_depth', None)
        self.engine = kwargs.pop('engine', None) or StockfishModel(kwargs.get('battle_format','gen9ou'))
        try: self.engine.set_depth(int(engine_depth))
        except Exception: pass
        self._request_cache = {}
        
        # Add extended websocket timeout configuration to prevent disconnections
        if 'ping_interval' not in kwargs:
            kwargs['ping_interval'] = 60.0  # Send keepalive every 60 seconds (was 20s default)
        if 'ping_timeout' not in kwargs:
            kwargs['ping_timeout'] = 120.0  # Wait up to 2 minutes for ping response (was 20s default)
        if 'open_timeout' not in kwargs:
            kwargs['open_timeout'] = 30.0   # Wait up to 30 seconds for initial connection (was 10s default)
        
        super().__init__(*args, **kwargs)

    def _build_request_signature(self, battle) -> tuple:
        try:
            turn = getattr(battle,'turn',None)
            force_switch = bool(getattr(battle,'force_switch',False))
            moves = [(getattr(m,'id',None), getattr(m,'pp',None)) for m in (getattr(battle,'available_moves',[]) or [])]
            switches = [getattr(p,'species',None) for p in (getattr(battle,'available_switches',[]) or [])]
            active = getattr(getattr(battle,'active_pokemon',None),'species',None)
            return (turn, force_switch, active, tuple(moves), tuple(switches))
        except Exception:
            return (None,)

    def choose_move(self, battle):
        tag = getattr(battle,'battle_tag', getattr(battle,'room_id',None))
        sig = self._build_request_signature(battle)
        cached = self._request_cache.get(tag)
        if cached and cached[0]==sig:
            try: return cached[1]
            except Exception: pass
        decision = self.engine.choose_action(battle)
        # think hook logging
        try:
            if self.on_think_hook and isinstance(decision.debug, dict):
                dd = dict(decision.debug)
                dd.setdefault('snapshot', snapshot_battle(battle))
                dd['battle_tag']=tag; dd['turn']=getattr(battle,'turn',None)
                import logging as _lg, json as _json
                _lg.getLogger('Think').info('UI_THINK turn=%s payload=%s', getattr(battle,'turn',None), _json.dumps(dd, default=str))
                self.on_think_hook(battle, dd)
        except Exception: pass
        try:
            if decision.kind=='move' and decision.move_id:
                for m in (getattr(battle,'available_moves',[]) or []):
                    if str(getattr(m,'id','')) == str(decision.move_id):
                        order = self.create_order(m); self._request_cache[tag]=(sig, order); return order
                order = self.create_order((getattr(battle,'available_moves',[]) or [None])[0]) or self.choose_random_move(battle)
                self._request_cache[tag]=(sig, order); return order
            elif decision.kind=='switch' and decision.switch_species:
                cursp=''
                try: cursp=(getattr(getattr(battle,'active_pokemon',None),'species',None) or '').lower()
                except Exception: pass
                for p in (getattr(battle,'available_switches',[]) or []):
                    sp = (str(getattr(p,'species','')).lower())
                    if sp==cursp: continue
                    if sp == str(decision.switch_species).lower():
                        order = self.create_order(p); self._request_cache[tag]=(sig, order); return order
                for p in (getattr(battle,'available_switches',[]) or []):
                    if (str(getattr(p,'species','')).lower()) != cursp:
                        order = self.create_order(p); self._request_cache[tag]=(sig, order); return order
                order = self.choose_random_move(battle); self._request_cache[tag]=(sig, order); return order
            order = self.choose_random_move(battle); self._request_cache[tag]=(sig, order); return order
        except Exception:
            order = self.choose_random_move(battle); self._request_cache[tag]=(sig, order); return order
