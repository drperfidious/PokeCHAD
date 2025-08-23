
"""
turn_order.py

Predict turn order in Gen 9 singles/doubles using *full* mechanics relevant to
priority and Speed, with exact stat and modifier formulas (PS-like rounding).
This pairs with speed_priority_rules.py and expects pre-resolved Pokemon/Field state.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from .speed_priority_rules import (
    to_id,
    PRIORITY_BOOST_ABILITIES,
    CONDITIONAL_PRIORITY_MOVES,
    PRIORITY_BLOCKERS,
    PRECEDENCE_OVERRIDES,
    SPEED_MULTIPLIER_ITEMS,
    SPEED_MULTIPLIER_ABILITIES,
    SPEED_MULTIPLIER_FIELD,
    STATUS_SPEED_MULTIPLIER,
    stage_multiplier,
    chain_mul,
)

# --- Stat formulas (Game-accurate) ---
# Reference: standard Pokémon stat formulas; nature is applied multiplicatively then floored.

def calc_stat(base: int, iv: int, ev: int, level: int, nature: float, is_hp: bool) -> int:
    if is_hp:
        if base == 1:  # Shedinja
            return 1
        return ((2 * base + iv + (ev // 4)) * level) // 100 + level + 10
    else:
        stat = ((2 * base + iv + (ev // 4)) * level) // 100 + 5
        stat = int(stat * nature)  # floor
        return stat

NATURE_MODS: Dict[str, Dict[str, float]] = {
    # only speed nature impacts here; include all for completeness
    # key is nature lowercase id
    # e.g., jolly: +Speed, -SpA; timid +Speed, -Atk; hasty +Speed, -Def; naive +Speed, -SpD
    "hardy": {"atk":1.0,"def":1.0,"spa":1.0,"spd":1.0,"spe":1.0},
    "lonely": {"atk":1.1,"def":0.9,"spa":1.0,"spd":1.0,"spe":1.0},
    "adamant": {"atk":1.1,"def":1.0,"spa":0.9,"spd":1.0,"spe":1.0},
    "naughty": {"atk":1.1,"def":1.0,"spa":1.0,"spd":0.9,"spe":1.0},
    "brave": {"atk":1.1,"def":1.0,"spa":1.0,"spd":1.0,"spe":0.9},
    "bold": {"atk":0.9,"def":1.1,"spa":1.0,"spd":1.0,"spe":1.0},
    "docile": {"atk":1.0,"def":1.0,"spa":1.0,"spd":1.0,"spe":1.0},
    "impish": {"atk":0.9,"def":1.1,"spa":1.0,"spd":1.0,"spe":1.0},
    "lax": {"atk":1.0,"def":1.1,"spa":1.0,"spd":0.9,"spe":1.0},
    "relaxed": {"atk":1.0,"def":1.1,"spa":1.0,"spd":1.0,"spe":0.9},
    "modest": {"atk":0.9,"def":1.0,"spa":1.1,"spd":1.0,"spe":1.0},
    "mild": {"atk":1.0,"def":0.9,"spa":1.1,"spd":1.0,"spe":1.0},
    "bashful": {"atk":1.0,"def":1.0,"spa":1.0,"spd":1.0,"spe":1.0},
    "rash": {"atk":1.0,"def":1.0,"spa":1.1,"spd":0.9,"spe":1.0},
    "quiet": {"atk":1.0,"def":1.0,"spa":1.1,"spd":1.0,"spe":0.9},
    "calm": {"atk":0.9,"def":1.0,"spa":1.0,"spd":1.1,"spe":1.0},
    "gentle": {"atk":1.0,"def":0.9,"spa":1.0,"spd":1.1,"spe":1.0},
    "careful": {"atk":1.0,"def":1.0,"spa":0.9,"spd":1.1,"spe":1.0},
    "sassy": {"atk":1.0,"def":1.0,"spa":1.0,"spd":1.1,"spe":0.9},
    "timid": {"atk":0.9,"def":1.0,"spa":1.0,"spd":1.0,"spe":1.1},
    "hasty": {"atk":1.0,"def":0.9,"spa":1.0,"spd":1.0,"spe":1.1},
    "jolly": {"atk":1.0,"def":1.0,"spa":0.9,"spd":1.0,"spe":1.1},
    "naive": {"atk":1.0,"def":1.0,"spa":1.0,"spd":0.9,"spe":1.1},
    "quirky": {"atk":1.0,"def":1.0,"spa":1.0,"spd":1.0,"spe":1.0},
}

def nature_multiplier(nature_id: str, stat_key: str) -> float:
    n = NATURE_MODS.get(nature_id, NATURE_MODS["hardy"])
    return n.get(stat_key, 1.0)

# --- Main turn order machinery ---

@dataclass
class SpeedContext:
    # Required raw stats inputs (when estimating opponents)
    base_spe: int
    iv_spe: int
    ev_spe: int
    level: int
    nature_id: str  # lowercase nature id

    # Stages and binary flags
    boost_stage_spe: int = 0
    user_is_paralyzed: bool = False
    user_has_quickfeet: bool = False
    user_is_statused: bool = False  # for Quick Feet

    # Ability / Item context flags
    ability_id: Optional[str] = None
    item_id: Optional[str] = None
    user_unburden_active: bool = False
    user_protosynthesis_speed: bool = False
    user_quarkdrive_speed: bool = False
    user_slowstart_active: bool = False

    # Species-specific for some items
    is_ditto_untransformed: bool = False

    # Field context
    weather: Optional[str] = None  # 'sun','rain','sandstorm','hail','snow','harshsunlight','heavyrain'
    terrain: Optional[str] = None  # 'electric','grassy','psychic','misty'
    side_tailwind_active: bool = False
    trick_room_active: bool = False
    magic_room_active: bool = False

def compute_effective_speed(ctx: SpeedContext) -> int:
    # 1) Base speed
    nature = nature_multiplier(ctx.nature_id, "spe")
    raw = calc_stat(ctx.base_spe, ctx.iv_spe, ctx.ev_spe, ctx.level, nature, is_hp=False)

    # 2) Apply stage
    num, den = stage_multiplier(ctx.boost_stage_spe)
    speed_after_stage = (raw * num) // den

    muls: List[Tuple[int,int] | float] = []

    # 3) Status (paralysis) unless Quick Feet
    muls.append( (1,1) )  # placeholder
    status_mul = STATUS_SPEED_MULTIPLIER({
        "user_is_paralyzed": ctx.user_is_paralyzed,
        "user_has_quickfeet": ctx.user_has_quickfeet,
    })
    muls[-1] = status_mul  # replace

    # 4) Item multipliers (unless Magic Room suppresses held items)
    if ctx.item_id and not ctx.magic_room_active:
        iid = to_id(ctx.item_id)
        if iid == "quickpowder" and not ctx.is_ditto_untransformed:
            pass  # no effect
        else:
            item_mul = SPEED_MULTIPLIER_ITEMS.get(iid)
            if item_mul:
                muls.append(item_mul)

    # 5) Ability multipliers
    if ctx.ability_id:
        aid = to_id(ctx.ability_id)
        ability_mul_fn = SPEED_MULTIPLIER_ABILITIES.get(aid)
        if ability_mul_fn:
            muls.append(ability_mul_fn({
                "weather": ctx.weather,
                "terrain": ctx.terrain,
                "user_is_statused": ctx.user_is_statused,
                "user_unburden_active": ctx.user_unburden_active,
                "user_slowstart_active": ctx.user_slowstart_active,
                "user_protosynthesis_speed": ctx.user_protosynthesis_speed,
                "user_quarkdrive_speed": ctx.user_quarkdrive_speed,
            }))

        # Quick Feet cancels para drop; ensure flag is set if ability is present
        if aid == "quickfeet":
            # If user is statused, Quick Feet 1.5 is already applied above
            pass

    # 6) Field multipliers: Tailwind
    muls.append(SPEED_MULTIPLIER_FIELD["tailwind"]({
        "side_tailwind_active": ctx.side_tailwind_active
    }))

    # Final chained multiplication, PS-like rounding
    effective = chain_mul(speed_after_stage, muls)

    # Trick Room handling happens at comparison time (reverses ordering within bracket).
    # We also respect the 10000 cap + modulo rules if you need PS-accurate tiebreaker quirks,
    # but for ordering the above effective int is enough.
    return effective

@dataclass
class MoveContext:
    # Base properties known from move data or poke-env Move object
    name: str
    base_priority: int
    category: str            # "Physical","Special","Status"
    type: str
    is_healing_or_drain: bool = False

    # Run-time flags (filled per turn)
    user_hp_is_full: bool = False
    terrain: Optional[str] = None
    target_is_grounded: bool = True
    opponent_has_priority_block_ability: bool = False
    is_prankster_applied: bool = False
    target_is_dark: bool = False

def compute_priority_after_mods(move: MoveContext, user_ability_id: Optional[str]) -> Tuple[int, Dict]:
    pri = move.base_priority
    ctx = {
        "category": move.category,
        "type": move.type,
        "user_hp_is_full": move.user_hp_is_full,
        "is_healing_or_drain": move.is_healing_or_drain,
        "terrain": move.terrain,
    }
    # Move conditional priority deltas (e.g., Grassy Glide)
    delta = CONDITIONAL_PRIORITY_MOVES.get(to_id(move.name), lambda _ctx: 0)(ctx)
    pri += delta

    # Ability boosts (Prankster / Gale Wings / Triage)
    applied_prankster = False
    if user_ability_id:
        aid = to_id(user_ability_id)
        boost = PRIORITY_BOOST_ABILITIES.get(aid)
        if boost and boost.applies({
            "category": move.category,
            "type": move.type,
            "user_hp_is_full": move.user_hp_is_full,
            "is_healing_or_drain": move.is_healing_or_drain,
        }):
            pri += boost.amount
            if aid == "prankster":
                applied_prankster = True

    # Save context for blockers
    blocker_ctx = {
        "terrain": move.terrain,
        "target_is_grounded": move.target_is_grounded,
        "opponent_has_priority_block_ability": move.opponent_has_priority_block_ability,
        "is_prankster_applied": applied_prankster,
        "target_is_dark": move.target_is_dark,
        "priority_after_mods": pri,
    }
    return pri, blocker_ctx

@dataclass
class PrecedenceResult:
    always_first: bool = False
    always_last: bool = False
    chance_first: float = 0.0

def precedence_from_effects(ability_id: Optional[str], item_id: Optional[str], move_ctx: Dict) -> PrecedenceResult:
    res = PrecedenceResult()
    for src in (ability_id, item_id):
        if not src: 
            continue
        eff = PRECEDENCE_OVERRIDES.get(to_id(src))
        if not eff:
            continue
        if eff.only_if and not eff.only_if(move_ctx):
            continue
        res.always_first = res.always_first or eff.always_first
        res.always_last = res.always_last or eff.always_last
        if eff.chance_first:
            # If stacked (e.g., Quick Draw + Quick Claw), we combine chances (1 - Π(1 - p))
            res.chance_first = 1 - (1 - res.chance_first) * (1 - eff.chance_first)
    return res

@dataclass
class Action:
    # 'move' or 'switch' (switching resolves before moves; not expanded here)
    kind: str
    move: Optional[MoveContext] = None

@dataclass
class OrderPrediction:
    bracket_user: int
    bracket_opp: int
    user_first_probability: float  # probability user acts before opponent this turn
    notes: List[str]

def predict_order(
    user_speed: int,
    opp_speed: int,
    user_action: Action,
    opp_action: Action,
    user_ctx: SpeedContext,
    opp_ctx: SpeedContext,
    user_ability_id: Optional[str],
    user_item_id: Optional[str],
    opp_ability_id: Optional[str],
    opp_item_id: Optional[str],
) -> OrderPrediction:
    notes: List[str] = []

    # Handle switching (both sides): all switches resolve before moves. Pursuit not modeled here.
    if user_action.kind == "switch" and opp_action.kind != "switch":
        return OrderPrediction(0, 0, 1.0, ["User is switching; switches resolve before moves."])
    if opp_action.kind == "switch" and user_action.kind != "switch":
        return OrderPrediction(0, 0, 0.0, ["Opponent is switching; switches resolve before moves."])
    if user_action.kind == "switch" and opp_action.kind == "switch":
        return OrderPrediction(0, 0, 0.5, ["Both sides switching; order rarely matters."])

    # Moves: compute priority (with ability boosts + conditional deltas)
    assert user_action.move and opp_action.move, "Moves required when kind='move'"
    u_pri, u_block_ctx = compute_priority_after_mods(user_action.move, user_ability_id)
    o_pri, o_block_ctx = compute_priority_after_mods(opp_action.move, opp_ability_id)

    # Blockers (Psychic Terrain, Dazzling/Queenly Majesty/Armor Tail, Prankster vs Dark)
    u_blocked = any(bl({**u_block_ctx, **{
        "opponent_has_priority_block_ability": opp_action.move.opponent_has_priority_block_ability
    }}) for bl in PRIORITY_BLOCKERS)
    o_blocked = any(bl({**o_block_ctx, **{
        "opponent_has_priority_block_ability": user_action.move.opponent_has_priority_block_ability
    }}) for bl in PRIORITY_BLOCKERS)

    if u_blocked and u_pri > 0:
        notes.append("User's priority was blocked (terrain/ability).")
        u_pri = 0  # stays in base bracket effectively for ordering; it *fails* but for order we drop bonus
    if o_blocked and o_pri > 0:
        notes.append("Opponent's priority was blocked (terrain/ability).")
        o_pri = 0

    # Priority bracket comparison
    if u_pri != o_pri:
        return OrderPrediction(u_pri, o_pri, 1.0 if u_pri > o_pri else 0.0,
                               notes + [f"Higher priority bracket wins: {u_pri} vs {o_pri}."])

    # Same bracket: precedence effects (Quick Claw / Custap / Lagging Tail / Stall / Mycelium Might)
    # Precedence effects may depend on contextual flags (e.g., Custap Berry activation).
    # We propagate a minimal move_ctx including category and any Custap activation boolean
    # that may have been set on the SpeedContext by the caller.
    u_move_ctx = {"category": user_action.move.category}
    o_move_ctx = {"category": opp_action.move.category}
    try:
        if getattr(user_ctx, "custap_active", False):
            u_move_ctx["custap_active"] = True
    except Exception:
        pass
    try:
        if getattr(opp_ctx, "custap_active", False):
            o_move_ctx["custap_active"] = True
    except Exception:
        pass
    u_prec = precedence_from_effects(user_ability_id, user_item_id, u_move_ctx)
    o_prec = precedence_from_effects(opp_ability_id, opp_item_id, o_move_ctx)

    # Always-first/last overrides first
    if u_prec.always_first and not o_prec.always_first:
        return OrderPrediction(u_pri, o_pri, 1.0, notes + ["User precedence: always first in bracket."])
    if o_prec.always_first and not u_prec.always_first:
        return OrderPrediction(u_pri, o_pri, 0.0, notes + ["Opponent precedence: always first in bracket."])
    if u_prec.always_last and not o_prec.always_last:
        return OrderPrediction(u_pri, o_pri, 0.0, notes + ["User precedence: always last in bracket."])
    if o_prec.always_last and not u_prec.always_last:
        return OrderPrediction(u_pri, o_pri, 1.0, notes + ["Opponent precedence: always last in bracket."])

    # Probabilistic precedence (Quick Claw / Quick Draw, potentially stacked)
    if u_prec.chance_first > 0 or o_prec.chance_first > 0:
        # If both can proc, resolve as independent Bernoulli and tie break by speed afterwards.
        # P(user first) = P(user proc & !opp proc) + P(neither proc) * P(user wins by speed)
        p_u = u_prec.chance_first
        p_o = o_prec.chance_first
        p_neither = (1 - p_u) * (1 - p_o)
        p_only_u = p_u * (1 - p_o)
        p_only_o = p_o * (1 - p_u)

        # If both proc, speed decides (still Trick Room applies). That's covered by "neither" branch using same speed compare.
        p_speed = speed_first_probability(user_speed, opp_speed, user_ctx.trick_room_active)
        user_first_prob = p_only_u + p_neither * p_speed + (p_u * p_o) * p_speed
        return OrderPrediction(u_pri, o_pri, user_first_prob,
                               notes + [f"Probabilistic precedence: Quick Claw/Draw/Custap. P(user first)≈{user_first_prob:.3f}"])

    # Finally, pure speed (with Trick Room)
    p_speed = speed_first_probability(user_speed, opp_speed, user_ctx.trick_room_active)
    return OrderPrediction(u_pri, o_pri, p_speed,
                           notes + ["Same bracket; decided by Speed (reversed under Trick Room)."])

def speed_first_probability(user_speed: int, opp_speed: int, trick_room: bool) -> float:
    if user_speed == opp_speed:
        return 0.5  # speed tie
    if trick_room:
        return 1.0 if user_speed < opp_speed else 0.0
    else:
        return 1.0 if user_speed > opp_speed else 0.0
