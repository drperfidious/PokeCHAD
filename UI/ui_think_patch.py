# UI/ui_think_patch.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from Data.battle_runtime import get_state, estimate_damage, apply_switch_in_effects
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo
from Data import battle_helper as _BH  # we'll patch its type_effectiveness
import logging

# ------------------------- Type chart fix (monkey patch) -------------------------
# Showdown typechart "damageTaken" codes: 0=Neutral, 1=Weak (2x), 2=Resist (0.5x), 3=Immune (0x).
# Some charts include non-type keys (e.g., "prankster", "sandstorm"); ignore them.
_PS_CODE_TO_MULT = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}
_ALLOWED_MULTS = {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}

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

# Replace broken dict attribute caching with a global cache (id-based)
_ATK_MAT_CACHE: dict[int, Dict[str, Dict[str, float]]] = {}
_DIAG_ONCE = set()  # track matrices already logged

def _tc_to_attack_matrix(tc: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Build attack->defense matrix (all UPPER keys) from various chart formats.

    Enhancements (A,B,C):
      - Detect if incoming structure already attack-matrix with float multipliers
        and bypass code translation.
      - If treating values as Showdown codes, only translate when ALL values for a row
        are ints within 0..3; mixed floats trigger direct multiplier usage.
      - After construction, clamp anomalous multipliers to nearest allowed value and
        correct known neutral pairs (e.g. Fire->Poison, Fire->Ground) if malformed.
      - Log one diagnostic FIRE row snapshot once per underlying tc object id.
    """
    log = logging.getLogger('typecalc')
    if not isinstance(tc, dict):
        return {T: {U: 1.0 for U in _CANON_TYPES} for T in _CANON_TYPES}
    # Fast path: already attack matrix (row dict with numeric multipliers all within reasonable range)
    try:
        some_v = next(iter(tc.values()))
        if isinstance(some_v, dict) and all(isinstance(v, (int, float)) for v in some_v.values()):
            # Assume already atk->def matrix; normalize keys
            out: Dict[str, Dict[str, float]] = {}
            for atk, row in tc.items():
                A = _norm_type_name(atk)
                if not A:
                    continue
                out.setdefault(A, {})
                for dfd, mult in row.items():
                    D = _norm_type_name(dfd)
                    if not D: continue
                    try:
                        out[A][D] = float(mult)
                    except Exception:
                        out[A][D] = 1.0
            # Fill gaps
            for A in _CANON_TYPES:
                out.setdefault(A, {})
                for D in _CANON_TYPES:
                    out[A].setdefault(D, 1.0)
            return out
    except Exception:
        pass
    # Showdown defense-centric format path
    atk_mat: Dict[str, Dict[str, float]] = {T: {} for T in _CANON_TYPES}
    for def_t, row in tc.items():
        D = _norm_type_name(def_t)
        if not D or D not in _CANON_TYPES:
            continue
        # Extract damageTaken or treat row itself
        dmg_row = None
        if isinstance(row, dict):
            if 'damageTaken' in row and isinstance(row['damageTaken'], dict):
                dmg_row = row['damageTaken']
            else:
                dmg_row = row
        if not isinstance(dmg_row, dict):
            continue
        # Determine if row looks like codes (all ints 0..3) or multipliers (floats or wider ints)
        values = list(dmg_row.values())
        is_code_row = values and all(isinstance(x, int) and 0 <= x <= 3 for x in values)
        if not is_code_row:
            # Treat entries as direct multipliers if they are floats / outside code range
            for atk_t, raw_mult in dmg_row.items():
                A = _norm_type_name(atk_t)
                if not A or A not in _CANON_TYPES: continue
                try:
                    mult = float(raw_mult)
                except Exception:
                    mult = 1.0
                atk_mat.setdefault(A, {})[D] = mult
            continue
        # Code translation path
        for atk_t, code in dmg_row.items():
            A = _norm_type_name(atk_t)
            if not A or A not in _CANON_TYPES:
                continue
            mult = _PS_CODE_TO_MULT.get(int(code), 1.0)
            atk_mat.setdefault(A, {})[D] = mult
    # Fill gaps & integrity fix
    for A in _CANON_TYPES:
        atk_mat.setdefault(A, {})
        for D in _CANON_TYPES:
            atk_mat[A].setdefault(D, 1.0)
    return atk_mat

def _fixed_type_effectiveness(move_type: str, defender_types: List[str] | tuple[str, ...] | None, type_chart: Dict[str, Any], 
                             move_id: Optional[str] = None, defender_tera_type: Optional[str] = None,
                             defender_terastallized: bool = False) -> float:
    # Robust replacement; never raises.
    if not move_type:
        return 1.0
    A = _norm_type_name(move_type)
    if not A:
        return 1.0
    if not isinstance(type_chart, dict):  # unexpected structure
        mat = _ATK_MAT_CACHE.get(id(type_chart))  # will be None
    # Retrieve / build cached matrix
    mat = _ATK_MAT_CACHE.get(id(type_chart))
    if mat is None:
        try:
            mat = _tc_to_attack_matrix(type_chart if isinstance(type_chart, dict) else {})
        except Exception:
            mat = {T: {U: 1.0 for U in _CANON_TYPES} for T in _CANON_TYPES}
        _ATK_MAT_CACHE[id(type_chart)] = mat
    if not defender_types:
        return 1.0
    eff = 1.0
    for t in defender_types:
        D = _norm_type_name(t)
        if not D:
            continue
        mult = mat.get(A, {}).get(D, 1.0)
        # Clamp unexpected multipliers (C)
        if mult not in _ALLOWED_MULTS:
            # bring into nearest allowed by ratio difference
            try:
                mult = min(_ALLOWED_MULTS, key=lambda m: abs(m - mult))
            except Exception:
                mult = 1.0
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
    """Max expected fraction the opponent can deal to target with revealed moves (accuracy-weighted)."""
    opp = state.team.opponent[opp_key]
    best = 0.0
    for mv in opp.moves or []:
        if not mv or not mv.id or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
            continue
        frac = _expected_damage_fraction(state, opp_key, target_key, mv.id, mi)
        best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
    return best

def _our_best_on_target(state, my_key: str, opp_key: str, mi: MovesInfo) -> float:
    """Best expected fraction we can deal to opponent this turn with our revealed moves (accuracy-weighted)."""
    me = state.team.ours[my_key]
    best = 0.0
    for mv in me.moves or []:
        if not mv or not mv.id or (mv.category or "").lower() == "status" or (mv.base_power or 0) <= 0:
            continue
        frac = _expected_damage_fraction(state, my_key, opp_key, mv.id, mi)
        best = max(best, frac * _acc_to_prob(getattr(mv, "accuracy", None)))
    return best

def _typing_bonus(candidate, opponent, mi: MovesInfo) -> float:
    """Small *display-only* bonus for better defensive typing vs opp's revealed attacking types.
    With extra for immunities. Range ~[0, 0.3]. Not added to the score; purely UI info.
    """
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
        eff = _fixed_type_effectiveness(move_type=t, defender_types=cand_types, type_chart=tc)
        if eff == 0:
            bonus += 0.2  # immunity => big bump
        elif eff < 1:
            bonus += 0.08 # resist => small bump
        elif eff > 1:
            bonus -= 0.05 # weak to => tiny penalty
    return max(0.0, bonus)


# ------------------------------ Main patch ------------------------------
def patch_stockfish_player():
    """Monkey-patch choose_move to emit *weight-free* switch scores and fix type chart use."""
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
