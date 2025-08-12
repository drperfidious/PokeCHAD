# battle_runtime.py
"""
Runtime glue for live poke-env battles and our mechanics modules.

This file exposes a minimal, clean API your model can call:
- get_state(battle, gen=9) -> CombinedState
- enumerate_actions(battle) -> {"moves": [MoveChoice...], "switches": [species_id...]}
- predict_order_for_ids(state, my_key, my_move_id, opp_key, opp_move_id, moves_info, *, my_tailwind=None, opp_tailwind=None) -> (prob, details)
- estimate_damage(state, attacker_key, defender_key, move_id, moves_info, *, is_crit=False) -> dict
- apply_switch_in_effects(state, switch_in_key, side: "ally"|"opponent", moves_info) -> dict
- would_fail(move_id, user_key, target_key, state, moves_info) -> (bool, reason)

It uses these modules:
  - team_state.TeamState (resolved stats & volatiles)
  - poke_env_battle_environment.to_field_state (weather/terrain/screens, etc.)
  - turn_order (priority + speed)
  - damage_helper (damage pipeline with fixed-point chain)
  - battle_helper (type/screen/weather/terrain helpers)
  - field_effects (hazards & Sticky Web)
  - stat_effects (stage change machinery; grounded inference)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .team_state import TeamState, PokemonState
from .poke_env_battle_environment import to_field_state, FieldState as EnvFieldState
from .poke_env_moves_info import MovesInfo
from .poke_env_pokemon_info import PokemonInfo
from .stat_effects import compute_passive_multipliers, augment_grounded
from .field_effects import apply_entry_hazards_on_switch_in
from .turn_order import (
    SpeedContext as TO_SpeedContext,
    MoveContext as TO_MoveContext,
    Action as TO_Action,
    predict_order as to_predict_order,
    compute_effective_speed as to_compute_speed,
)
from .damage_helper import (
    CombatantState as DMG_Combatant,
    MoveContext as DMG_Move,
    FieldState as DMG_Field,
    calc_damage_range,
)
from .battle_helper import (
    type_effectiveness,
    weather_modifier,
    screen_modifier,
    terrain_modifier,
    stab_multiplier,
)

# --------------------------------- State -----------------------------------------

@dataclass
class CombinedState:
    team: TeamState
    field: EnvFieldState
    turn: Optional[int]
    format: Optional[str]
    my_side: Dict[str, Any]
    opp_side: Dict[str, Any]

def _extract_side_flags(battle) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # reuse the helper from poke_env_battle_environment
    from .poke_env_battle_environment import _extract_side_conditions
    my_side = _extract_side_conditions(getattr(battle, "side_conditions", None) or getattr(battle, "side", None))
    opp_side = _extract_side_conditions(getattr(battle, "opponent_side_conditions", None) or getattr(battle, "opponent_side", None))
    return my_side, opp_side

def get_state(battle, gen: int = 9, ev_policy: str = "auto") -> CombinedState:
    ts = TeamState.from_battle(battle, gen=gen, ev_policy=ev_policy)
    field = to_field_state(battle)
    my_side, opp_side = _extract_side_flags(battle)
    return CombinedState(
        team=ts,
        field=field,
        turn=getattr(battle, "turn", None),
        format=getattr(battle, "format", None),
        my_side=my_side,
        opp_side=opp_side,
    )

# ------------------------------ Actions ------------------------------------------

@dataclass
class MoveChoice:
    id: str
    name: Optional[str]
    priority: int
    type: Optional[str]
    category: Optional[str]

def enumerate_actions(battle) -> Dict[str, List[Any]]:
    moves = []
    for m in (getattr(battle, "available_moves", None) or []):
        mid = getattr(m, "id", None) or getattr(m, "name", None)
        moves.append(MoveChoice(
            id=str(mid),
            name=getattr(m, "name", None),
            priority=int(getattr(m, "priority", 0) or 0),
            type=getattr(m, "type", None),
            category=getattr(m, "category", None),
        ))
    switches = [getattr(p, "species", None) for p in (getattr(battle, "available_switches", None) or [])]
    return {"moves": moves, "switches": switches}

# --------------------------- Priority + Speed ------------------------------------

def _is_sun(weather: Optional[str]) -> bool:
    return str(weather or "").lower() in {"sun","harshsunlight","desolateland"}

def _is_electric(terrain: Optional[str]) -> bool:
    return str(terrain or "").lower() == "electric"

def _highest_stat_is_speed(ps: PokemonState) -> bool:
    r = ps.stats.raw
    return r.get("spe", 0) >= max(r.get("atk", 0), r.get("spa", 0), r.get("def", 0), r.get("spd", 0))

def _speed_context_from_state(ps: PokemonState, *, field: EnvFieldState, tailwind_active: bool) -> TO_SpeedContext:
    status = (ps.status or "").lower()
    ability = (ps.ability or "").lower()
    item = (ps.item or "").lower() if ps.item else None
    nature = (ps.stats.nature or "serious").lower()

    # Paradox speed boost when Speed is highest stat and activation condition met
    has_proto_speed = ability == "protosynthesis" and (_is_sun(field.weather)) and _highest_stat_is_speed(ps)
    has_quark_speed = ability == "quarkdrive" and (_is_electric(field.terrain)) and _highest_stat_is_speed(ps)

    # Unburden heuristic: if ability is unburden and item is None but consumed_item was present at some point
    unburden_active = ability == "unburden" and (ps.consumed_item is not None) and (ps.item is None)

    return TO_SpeedContext(
        base_spe=int(ps.stats.base.get("spe", 0)),
        iv_spe=int(ps.stats.ivs.get("spe", 31)),
        ev_spe=int(ps.stats.evs.get("spe", 0)),
        level=int(ps.stats.level),
        nature_id=nature,
        boost_stage_spe=int(ps.stats.boosts.get("spe", 0)),
        user_is_paralyzed=(status == "par"),
        user_has_quickfeet=(ability == "quickfeet"),
        user_is_statused=bool(status),
        ability_id=ability,
        item_id=item,
        user_unburden_active=unburden_active,
        user_protosynthesis_speed=has_proto_speed,
        user_quarkdrive_speed=has_quark_speed,
        user_slowstart_active=(ability == "slowstart"),
        is_ditto_untransformed=(ps.species.lower() == "ditto" and "transform" not in ps.volatiles),
        weather=field.weather,
        terrain=field.terrain,
        side_tailwind_active=bool(tailwind_active),
        trick_room_active=bool(getattr(field, "trick_room", False)),
        magic_room_active=False,  # if you track Magic Room, set this accordingly
    )

def _to_move_ctx_for_order(mv: Any, user: PokemonState, target: PokemonState, field: EnvFieldState, mi: MovesInfo) -> TO_MoveContext:
    raw = mi.get(mv)
    is_heal = bool(raw.raw.get("heal")) or bool(raw.raw.get("drain"))
    return TO_MoveContext(
        name=raw.id,
        base_priority=int(raw.priority or 0),
        category=str(raw.category or "Status").capitalize(),
        type=str(raw.type or "Unknown"),
        is_healing_or_drain=is_heal,
        user_hp_is_full=bool((user.current_hp or 0) >= (user.max_hp or 0)),
        terrain=field.terrain,
        target_is_grounded=bool(target.grounded if target.grounded is not None else True),
        opponent_has_priority_block_ability=bool(target.has_priority_block),
        is_prankster_applied=False,   # computed inside the order engine
        target_is_dark=("dark" in {t for t in target.types if t}),
    )

def _tailwind_flags(my_side: Dict[str, Any], opp_side: Dict[str, Any]) -> Tuple[bool, bool]:
    return bool(my_side.get("tailwind", False)), bool(opp_side.get("tailwind", False))

def predict_order_for_ids(
    state: CombinedState,
    my_key: str,
    my_move_id: str,
    opp_key: str,
    opp_move_id: str,
    mi: MovesInfo,
    *,
    my_tailwind: Optional[bool] = None,
    opp_tailwind: Optional[bool] = None,
) -> Tuple[float, Dict[str, Any]]:
    me = state.team.ours[my_key]
    opp = state.team.opponent[opp_key]

    # Ensure grounded flags are known for priority blocks/terrain
    if me.grounded is None: augment_grounded(me, {"gravity": state.field.gravity})
    if opp.grounded is None: augment_grounded(opp, {"gravity": state.field.gravity})

    # Tailwind flags
    tw_my, tw_opp = _tailwind_flags(state.my_side, state.opp_side)
    tw_my = tw_my if my_tailwind is None else my_tailwind
    tw_opp = tw_opp if opp_tailwind is None else opp_tailwind

    # Speed
    me_ctx = _speed_context_from_state(me, field=state.field, tailwind_active=tw_my)
    opp_ctx = _speed_context_from_state(opp, field=state.field, tailwind_active=tw_opp)
    me_speed = to_compute_speed(me_ctx)
    opp_speed = to_compute_speed(opp_ctx)

    # Moves
    u_move = _to_move_ctx_for_order(my_move_id, me, opp, state.field, mi)
    o_move = _to_move_ctx_for_order(opp_move_id, opp, me, state.field, mi)


    # Custap Berry precedence: set flags on contexts so turn_order can read them via move_ctx
    try:
        me_ctx.hp = int(me.current_hp or 0); me_ctx.max_hp = int(me.max_hp or 0)
        opp_ctx.hp = int(opp.current_hp or 0); opp_ctx.max_hp = int(opp.max_hp or 0)
    except Exception:
        pass
    me_ctx.custap_active = ((str(me.item or '').lower() == 'custapberry') and not me_ctx.magic_room_active and me_ctx.max_hp and me_ctx.hp * 4 <= me_ctx.max_hp)
    opp_ctx.custap_active = ((str(opp.item or '').lower() == 'custapberry') and not opp_ctx.magic_room_active and opp_ctx.max_hp and opp_ctx.hp * 4 <= opp_ctx.max_hp)

    pred = to_predict_order(
        me_speed, opp_speed,
        TO_Action(kind="move", move=u_move),
        TO_Action(kind="move", move=o_move),
        me_ctx, opp_ctx,
        (me.ability or "").lower(), (me.item or "").lower() if me.item else None,
        (opp.ability or "").lower(), (opp.item or "").lower() if opp.item else None,
    )

    details = {
        "user_effective_speed": me_speed,
        "opp_effective_speed": opp_speed,
        "user_bracket": pred.bracket_user,
        "opp_bracket": pred.bracket_opp,
        "notes": pred.notes,
    }
    return float(pred.user_first_probability), details

# ------------------------------ Damage -------------------------------------------

def _apply_stat_side_modifiers(
    pinfo: PokemonInfo, ps: PokemonState, *, for_attacker: bool, field: Optional[EnvFieldState] = None
) -> Tuple[int,int,int,int,int,int]:
    """Return modified (hp, atk, def, spa, spd, spe) after item+ability passive stat multipliers."""
    base = (ps.stats.raw["hp"], ps.stats.raw["atk"], ps.stats.raw["def"], ps.stats.raw["spa"], ps.stats.raw["spd"], ps.stats.raw["spe"])

    # Item-side mods (Choice/Eviolite/AV/etc)
    from .poke_env_pokemon_info import apply_item_stat_modifiers, PokemonStats
    it_applied = apply_item_stat_modifiers(pinfo, ps.species, ps.item, PokemonStats(*base))
    hp, atk, deff, spa, spd, spe = it_applied.hp, it_applied.atk, it_applied.def_, it_applied.spa, it_applied.spd, it_applied.spe

    # Ability passives (Huge Power/Guts/etc) -> only stat-side ones here
    mults = compute_passive_multipliers(ps, {"weather": field.weather if field else None})
    # The helper above expects a field dict; we pass a conservative context (weather only).
    # Apply multipliers to the relevant stats:
    atk = int(atk * mults.get("atk", 1.0))
    deff = int(deff * mults.get("def", 1.0))
    spa = int(spa * mults.get("spa", 1.0))
    spd = int(spd * mults.get("spd", 1.0))
    spe = int(spe * mults.get("spe", 1.0))

    return hp, atk, deff, spa, spd, spe

def _combatant_from_state(
    ps: PokemonState,
    pinfo: PokemonInfo,
    *,
    apply_stat_mods: bool = True,
    field: Optional[EnvFieldState] = None
) -> DMG_Combatant:
    if apply_stat_mods:
        hp, atk, deff, spa, spd, spe = _apply_stat_side_modifiers(pinfo, ps, for_attacker=True, field=field)
    else:
        hp, atk, deff, spa, spd, spe = (ps.stats.raw["hp"], ps.stats.raw["atk"], ps.stats.raw["def"], ps.stats.raw["spa"], ps.stats.raw["spd"], ps.stats.raw["spe"])

    return DMG_Combatant(
        level=int(ps.stats.level),
        types=[t.capitalize() for t in ps.types if t],
        atk=atk, def_=deff, spa=spa, spd=spd, spe=spe,
        tera_type=ps.tera_type.capitalize() if ps.tera_type else None,
        grounded=bool(ps.grounded if ps.grounded is not None else True),
        is_burned=(ps.status == "brn"),
        ability=(ps.ability or None),
        item=(ps.item or None),
        atk_stage=int(ps.stats.boosts.get("atk", 0)),
        def_stage=int(ps.stats.boosts.get("def", 0)),
        spa_stage=int(ps.stats.boosts.get("spa", 0)),
        spd_stage=int(ps.stats.boosts.get("spd", 0)),
        spe_stage=int(ps.stats.boosts.get("spe", 0)),
    )

def _dm_field_from_env_field(field: EnvFieldState) -> DMG_Field:
    return DMG_Field(
        weather=field.weather, terrain=field.terrain, gravity=field.gravity, trick_room=field.trick_room,
        is_doubles=field.is_doubles, targets_on_target_side=field.targets_on_target_side,
        reflect=field.reflect, light_screen=field.light_screen, aurora_veil=field.aurora_veil
    )

def estimate_damage(
    state: CombinedState,
    attacker_key: str,
    defender_key: str,
    move_id: str,
    mi: MovesInfo,
    *,
    is_critical: bool = False,
) -> Dict[str, Any]:
    pinfo = PokemonInfo(9)
    atk = state.team.ours.get(attacker_key) or state.team.opponent.get(attacker_key)
    dfd = state.team.ours.get(defender_key) or state.team.opponent.get(defender_key)
    if atk is None or dfd is None:
        raise KeyError("Unknown attacker or defender key.")

    # Build combatants
    c_atk = _combatant_from_state(atk, pinfo, apply_stat_mods=True, field=state.field)
    c_dfd = _combatant_from_state(dfd, pinfo, apply_stat_mods=True, field=state.field)

    # Build move
    raw = mi.get(move_id)
    mv = DMG_Move(
        move_id=raw.id,
        name=raw.name,
        type=str(raw.type or "Unknown"),
        category=str(raw.category or "Status").capitalize(),
        base_power=int(raw.base_power or 0),
        is_spread=(raw.target in {"allAdjacent", "allAdjacentFoes", "all"}),
        hits_multiple_targets_on_execution=(state.field.is_doubles and raw.target in {"allAdjacent", "allAdjacentFoes", "all"}),
        makes_contact=bool(raw.flags.get("contact", False)),
        is_sound=bool(raw.flags.get("sound", False)),
        is_punch=bool(raw.flags.get("punch", False)),
        is_biting=bool(raw.flags.get("bite", False)),
        multihit=(raw.multihit if isinstance(raw.multihit, list) else None),
    )

    # Type chart and helpers
    chart_fn = mi.get_type_chart
    dmg_field = _dm_field_from_env_field(state.field)

    # Extra modifiers (Life Orb, Expert Belt, etc.) -> placeholder (you can extend here)
    extra = []
    it = (atk.item or "").lower() if atk.item else ""
    if it == "lifeorb":
        extra.append(1.3)

    res = calc_damage_range(
        c_atk, c_dfd, mv, dmg_field,
        get_type_chart=chart_fn,
        is_critical=is_critical,
        extra_modifiers=extra,
        type_effectiveness_fn=type_effectiveness,
        stab_fn=stab_multiplier,
        weather_fn=weather_modifier,
        terrain_fn=terrain_modifier,
        screen_fn=screen_modifier,
    )
    return {
        "min": res.min_damage,
        "max": res.max_damage,
        "rolls": res.rolls,
        "effectiveness": res.effectiveness,
        "mods": res.applied_modifiers,
    }

# ------------------------- Switch-in effects -------------------------------------

def apply_switch_in_effects(
    state: CombinedState,
    switch_in_key: str,
    side: str,
    mi: MovesInfo,
    *,
    mutate: bool = False,
) -> Dict[str, Any]:
    """Apply hazards + Sticky Web + entry status logic (prediction helper)."""
    type_chart = mi.get_type_chart()
    ps = state.team.ours.get(switch_in_key) if side == "ally" else state.team.opponent.get(switch_in_key)
    if ps is None:
        raise KeyError(f"Unknown key on side {side}: {switch_in_key}")
    field_dict = {"gravity": state.field.gravity, "terrain": state.field.terrain}

    # Pick the *opponent* side hazards relative to the switching-in Pokémon
    side_obj = type("Side", (), {})()
    opp_side_dict = state.opp_side if side == "ally" else state.my_side
    for k in ("stealth_rock","spikes","toxic_spikes","sticky_web"):
        setattr(side_obj, k, opp_side_dict.get(k, 0 if k in ("spikes","toxic_spikes") else False))

    # Apply
    out = apply_entry_hazards_on_switch_in(ps, side_obj, field_dict, type_chart, mutate=mutate)
    return out

# ------------------------- Move failure checks -----------------------------------

_PROTECTIVES = {"protect","kingsshield","spikyshield","banefulbunker","obstruct","silktrap","maxguard"}
_BYPASS_PROTECT_IDS = {"feint","hyperspacefury","hyperspacehole","phantomforce","shadowforce"}

def _effective_priority_for_blockers(raw_move, user_ps: PokemonState, field: EnvFieldState) -> int:
    # Base
    pri = int(raw_move.priority or 0)
    # Grassy Glide +1 in Grassy Terrain
    if raw_move.id == "grassyglide" and str(field.terrain or "") == "grassy":
        pri += 1
    # Prankster to Status +1
    if (user_ps.ability or "").lower() == "prankster" and (raw_move.category or "").lower() == "status":
        pri += 1
    # Triage +3 to healing
    if raw_move.raw.get("heal") or raw_move.raw.get("drain"):
        if (user_ps.ability or "").lower() == "triage":
            pri += 3
    # Gale Wings +1 to Flying moves at full HP
    if (user_ps.ability or "").lower() == "galewings" and (raw_move.type or "").lower() == "flying"        and (user_ps.current_hp or 0) >= (user_ps.max_hp or 0):
        pri += 1
    return pri

def would_fail(
    move_id: str,
    user_key: str,
    target_key: str,
    state: CombinedState,
    mi: MovesInfo
) -> Tuple[bool, str]:
    """Best-effort "would fail" gate for Protect/Quick Guard/Wide Guard/Psychic Terrain/priority blocks.

    NOTE: This does not resolve dynamic outcomes like Sucker Punch failing or niche exceptions like
    Instruct, Immunity gating, etc. It's aimed at the high-impact gating for planning.
    """
    user = state.team.ours.get(user_key) or state.team.opponent.get(user_key)
    target = state.team.ours.get(target_key) or state.team.opponent.get(target_key)
    if user is None or target is None:
        return False, "unknown-actors"

    raw = mi.get(move_id)
    cat = (raw.category or "Status").lower()
    is_damaging = cat in {"physical","special"}

    # 1) Direct Protect family on target
    if _PROTECTIVES & set(target.volatiles):
        if raw.id in _BYPASS_PROTECT_IDS or raw.raw.get("breaksProtect") or raw.raw.get("bypassProtect"):
            return False, "bypasses-protect"
        if raw.flags.get("protect", True):
            return True, "blocked-by-protect"

    # 2) Psychic Terrain (blocks priority from grounded attackers hitting grounded targets)
    if str(state.field.terrain or "") == "psychic":
        # We need groundedness for both
        if user.grounded is None: augment_grounded(user, {"gravity": state.field.gravity})
        if target.grounded is None: augment_grounded(target, {"gravity": state.field.gravity})
        pri = _effective_priority_for_blockers(raw, user, state.field)
        if pri > 0 and user.grounded and target.grounded:
            # Dazzling/Queenly also block priority at target ability
            return True, "blocked-by-psychic-terrain"

    # 3) Priority blockers at target: Dazzling / Queenly Majesty / Armor Tail
    if is_damaging:
        if target.has_priority_block:
            pri = _effective_priority_for_blockers(raw, user, state.field)
            if pri > 0:
                return True, "blocked-by-priority-ability"

    # 4) Quick Guard on target side blocks all +priority moves (damaging or status)
    if state.opp_side.get("quick_guard", False) and user_key in state.team.ours:
        pri = _effective_priority_for_blockers(raw, user, state.field)
        if pri > 0:
            if raw.id != "feint":
                return True, "blocked-by-quick-guard"
    if state.my_side.get("quick_guard", False) and user_key in state.team.opponent:
        pri = _effective_priority_for_blockers(raw, user, state.field)
        if pri > 0:
            if raw.id != "feint":
                return True, "blocked-by-quick-guard"

    # 5) Wide Guard on target side vs spread damaging move
    targets_spread = raw.target in {"allAdjacent","allAdjacentFoes","all"}
    if state.opp_side.get("wide_guard", False) and user_key in state.team.ours:
        if is_damaging and targets_spread and raw.id != "feint":
            return True, "blocked-by-wide-guard"
    if state.my_side.get("wide_guard", False) and user_key in state.team.opponent:
        if is_damaging and targets_spread and raw.id != "feint":
            return True, "blocked-by-wide-guard"

    # 6) Prankster vs Dark: status move boosted by Prankster fails on Dark-type targets
    if (user.ability or "").lower() == "prankster" and cat == "status" and ("dark" in {t for t in target.types if t}):
        return True, "prankster-vs-dark"

    return False, "ok"
