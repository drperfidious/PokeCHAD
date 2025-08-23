from __future__ import annotations

import logging
import os
import random
import time
import math
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Data.battle_runtime import (
    get_state,
    estimate_damage,
    would_fail,
    predict_order_for_ids,
    predict_first_prob_speed_only,
    apply_switch_in_effects,
    get_effective_speeds,
)
from Data.poke_env_battle_environment import snapshot as snapshot_battle
from Data.poke_env_moves_info import MovesInfo
# New: type chart helper for better effectiveness awareness
try:
    from utils.type_effectiveness import get as get_typecalc  # lightweight singleton
except Exception:  # fallback if module not present in some envs
    get_typecalc = lambda: None
    
# Meta-game knowledge integration
try:
    from Data.opponent_prediction import OpponentPredictor
    from Data.meta_knowledge import MetaKnowledge
except Exception:
    OpponentPredictor = None
    MetaKnowledge = None

# Rust engine acceleration (optional)
try:
    import pokechad_engine
    RUST_ENGINE_AVAILABLE = True
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    pokechad_engine = None  # ensure symbol exists even if import fails

# ---------------- Data containers ----------------
@dataclass
class ChosenAction:
    kind: str  # 'move' | 'switch' | 'tera'
    move_id: Optional[str] = None
    switch_species: Optional[str] = None
    is_tera: bool = False  # True if this is a terastallized move
    tera_type: Optional[str] = None  # The tera type to use
    debug: Optional[Dict[str, Any]] = None

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
        if ps and ps.max_hp:
            return max(0.0, min(1.0, (ps.current_hp or ps.max_hp) / ps.max_hp))
    except Exception:
        pass
    return 1.0


def _get_move_category(mv) -> str:
    """Safely extract move category as string."""
    try:
        cat = getattr(mv, 'category', '')
        if hasattr(cat, 'name'):  # Enum object
            return str(cat.name).lower()
        elif hasattr(cat, 'value'):  # Enum with value
            return str(cat.value).lower()
        else:
            return str(cat or '').lower()
    except Exception:
        return ''


def _expected_damage_fraction(state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo) -> Tuple[float, Dict[str, Any]]:
    dmg = estimate_damage(state, atk_key, dfd_key, move_id, mi, is_critical=False)
    rolls = dmg.get("rolls") or []
    if not rolls:
        return 0.0, dmg
    dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
    max_hp = int(getattr(dfd, "max_hp", 0) or getattr(dfd, "stats", {}).raw.get("hp", 1) or 1)
    cur_hp = int(getattr(dfd, "current_hp", None) or max_hp)
    # Expected fraction of HP lost is bounded by current HP
    bounded = [min(int(r), cur_hp) for r in rolls]
    return (float(sum(bounded)) / (len(bounded) * max_hp)), dmg

def _expected_damage_fraction_with_tera(state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo, tera_type: str) -> Tuple[float, Dict[str, Any]]:
    """Calculate damage with terastallization effects.
    
    This function calculates damage when the attacker is terastallized,
    taking into account:
    1. Changed move types (e.g., Tera Blast becomes tera type)
    2. STAB from tera type
    3. Proper type effectiveness with tera type
    """
    try:
        # Get base damage calculation
        base_frac, base_dmg = _expected_damage_fraction(state, atk_key, dfd_key, move_id, mi)
        
        # Get attacker and defender info
        atk = state.team.ours.get(atk_key) or state.team.opponent.get(atk_key)
        dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
        
        if not (atk and dfd):
            return base_frac, base_dmg
        
        # Calculate tera type effectiveness
        try:
            from utils.type_effectiveness import get as get_typecalc
            from Data.battle_helper import stab_multiplier as calc_stab
            tc = get_typecalc()
            if tc:
                # Species and move typing
                defender_species = getattr(dfd, 'species', None)
                attacker_species = getattr(atk, 'species', None)
                mv_info = mi.get(move_id)
                move_type_static = (getattr(mv_info, 'type', None) or '').title()
                # Terablast changes type under Tera
                move_type_tera = tc.move_type(move_id, is_terastallized=True, attacker_tera_type=tera_type) or move_type_static

                # Effectiveness ratio (only differs for Terablast and similar)
                tera_eff, _ = tc.effectiveness(
                    move_id,
                    attacker_species,
                    defender_species,
                    is_terastallized=True,
                    attacker_tera_type=tera_type,
                )
                normal_eff, _ = tc.effectiveness(
                    move_id,
                    attacker_species,
                    defender_species,
                    is_terastallized=False,
                )

                # STAB ratio using the same rules as the calculator
                orig_types = [t for t in (getattr(atk, 'types', []) or []) if t]
                ability = getattr(atk, 'ability', None)
                normal_stab = calc_stab(move_type_static, orig_types, tera_type, ability, terastallized=False)
                tera_stab = calc_stab(move_type_tera, orig_types, tera_type, ability, terastallized=True)

                # Combine ratios conservatively
                eff_ratio = (tera_eff / normal_eff) if normal_eff else 1.0
                stab_ratio = (tera_stab / normal_stab) if normal_stab else 1.0
                tera_frac = base_frac * eff_ratio * stab_ratio

                # Create enhanced damage dict with tera info
                tera_dmg = dict(base_dmg)
                tera_dmg.update({
                    'tera_effectiveness': tera_eff,
                    'normal_effectiveness': normal_eff,
                    'tera_eff_ratio': eff_ratio,
                    'normal_stab': normal_stab,
                    'tera_stab': tera_stab,
                    'stab_ratio': stab_ratio,
                    'tera_type': tera_type,
                    'effectiveness': tera_eff,
                })

                return tera_frac, tera_dmg
                
        except Exception as e:
            logging.debug(f"Tera type effectiveness calculation failed: {e}")
            
        # Fallback: apply basic tera STAB bonus if type matching
        try:
            mv = mi.get(move_id)
            move_type = getattr(mv, 'type', None)
            if move_type and str(move_type).lower() == tera_type.lower():
                # Simple STAB bonus
                tera_frac = base_frac * 1.5
                tera_dmg = dict(base_dmg)
                tera_dmg['tera_stab_bonus'] = 1.5
                return tera_frac, tera_dmg
        except Exception:
            pass
            
        return base_frac, base_dmg
        
    except Exception as e:
        logging.debug(f"Tera damage calculation failed: {e}")
        # Fallback to normal calculation
        return _expected_damage_fraction(state, atk_key, dfd_key, move_id, mi)


# Minimal set of common STAB moves for fallback when opponent moves are unknown
_COMMON_STAB = {
    "normal": ["return", "bodyslam"],
    "fire": ["flamethrower", "fireblast"],
    "water": ["surf", "hydropump"],
    "electric": ["thunderbolt", "thunder"],
    "grass": ["energyball", "leafstorm"],
    "ice": ["icebeam", "blizzard"],
    "fighting": ["closecombat", "drainpunch"],
    "poison": ["sludgebomb", "gunkshot"],
    "ground": ["earthquake", "earthpower"],
    "flying": ["hurricane", "bravebird"],
    "psychic": ["psychic", "psyshock"],
    "bug": ["bugbuzz", "leechlife"],
    "rock": ["stoneedge", "rockslide"],
    "ghost": ["shadowball", "poltergeist"],
    "dragon": ["dracometeor", "dragonpulse"],
    "dark": ["darkpulse", "crunch"],
    "steel": ["flashcannon", "ironhead"],
    "fairy": ["moonblast", "playrough"],
}


def _has_recovery_item(pokemon) -> bool:
    """Check if pokemon has Leftovers or other recovery items."""
    try:
        item = str(getattr(pokemon, 'item', '') or '').lower()
        return item in {'leftovers', 'blacksludge', 'shellbell'}
    except Exception:
        return False


def _opp_best_on_target(state, opp_key: str, target_key: str, mi: MovesInfo, opponent_knowledge=None) -> float:
    opp = state.team.opponent[opp_key]
    best = 0.0
    mv_list = [m for m in (opp.moves or []) if getattr(m, 'id', None)]
    
    # Consider known damaging moves from game state
    if mv_list:
        for mv in mv_list:
            if _get_move_category(mv) == 'status' or (getattr(mv, 'base_power', 0) or 0) <= 0:
                continue
            frac, _ = _expected_damage_fraction(state, opp_key, target_key, mv.id, mi)
            best = max(best, frac * _acc_to_prob(getattr(mv, 'accuracy', None)))
    
    # Use learned moves from opponent knowledge
    if opponent_knowledge:
        species = getattr(opp, 'species', None)
        if species:
            knowledge = opponent_knowledge.get(species, {})
            learned_moves = knowledge.get('moves', set())
            
            for move_id in learned_moves:
                try:
                    # Skip if we already considered this move
                    if any(getattr(mv, 'id', '').lower() == move_id for mv in mv_list):
                        continue
                        
                    raw = mi.raw(move_id) or {}
                    if raw.get('category', '').lower() == 'status' or (raw.get('base_power', 0) or 0) <= 0:
                        continue
                        
                    frac, _ = _expected_damage_fraction(state, opp_key, target_key, move_id, mi)
                    best = max(best, frac * _acc_to_prob(raw.get('accuracy')))
                except Exception:
                    continue
    
    # Always also consider plausible strong STABs (covers partial knowledge)
    try:
        for t in (opp.types or []):
            if not t:
                continue
            for mid in _COMMON_STAB.get(str(t).lower(), [])[:2]:
                try:
                    frac, _ = _expected_damage_fraction(state, opp_key, target_key, mid, mi)
                    raw = mi.raw(mid) or {}
                    best = max(best, frac * _acc_to_prob(raw.get('accuracy')))
                except Exception:
                    continue
    except Exception:
        pass
    return best

# ---------------- Engine ----------------
class MCTSModel:
    """PUCT-based MCTS with short playouts over our existing mechanics helpers.

    Uses priors from single-ply heuristic, samples order and damage from helper models,
    and backprops net HP swing + KO bonuses.
    
    Enhanced with optional ISMCTS, Neural Networks, and Enhanced LLM Integration.
    """

    def __init__(self, battle_format: str = 'gen9ou', simulations: int = 5000, c_puct: float = 1.2,
                 rollout_depth: int = 5, time_limit: float = 45.0, use_enhancements: bool = True):
        self.battle_format = battle_format
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.rollout_depth = int(rollout_depth)
        self.time_limit = float(time_limit)
        self._verbose = False
        self._llm_enabled = False
        
        # Meta-game integration
        self.opponent_predictor = OpponentPredictor() if OpponentPredictor else None
        self.meta_knowledge = MetaKnowledge(battle_format) if MetaKnowledge else None
        
        # LLM integration
        self.llm_integration = None
        self._api_key = None
        try:
            from Models.llm_integration import create_llm_integration
            # Try to load API key from environment or config
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_integration = create_llm_integration("openai", api_key=api_key)
                self._api_key = api_key
            else:
                # Fallback to mock for now (user can set API key later)
                from Models.mock_llm_provider import create_mock_llm_integration
                self.llm_integration = create_mock_llm_integration()
        except Exception:
            pass
        
        # Enhanced MCTS Integration
        self.use_enhancements = use_enhancements
        self.enhanced_model = None
        
        # Training data collection
        self.collect_training_data = True
        self._training_collector = None
        
        if use_enhancements:
            try:
                from Models.enhanced_mcts_model import create_enhanced_mcts_model
                
                enhancement_config = {
                    'use_ismcts': True,
                    'use_neural_networks': True, 
                    'use_enhanced_llm': True,
                    'neural_network_device': 'cpu',
                    'decision_mode': 'adaptive',
                    'training': {'enable_training': False}
                }
                
                # Create enhanced model using self as base
                self.enhanced_model = create_enhanced_mcts_model(self, enhancement_config)
                logging.info("[MCTS] Enhanced MCTS integrated successfully")
                
            except Exception as e:
                logging.warning(f"[MCTS] Enhanced MCTS integration failed: {e}")
                self.enhanced_model = None
        
        # Initialize training data collector - prioritize standalone for compatible models
        logging.info(f"[MCTS] Initializing with collect_training_data: {self.collect_training_data}")
        if self.collect_training_data:
            try:
                # Use standalone collector if enhanced MCTS is available (830 dims, tensor format)
                if hasattr(self, 'enhanced_model') and self.enhanced_model is not None:
                    from Models.standalone_data_collector import create_standalone_collector
                    logging.info("[MCTS] Using standalone training collector (tensor format)")
                    self._training_collector = create_standalone_collector("UI/training_data/standalone")
                else:
                    # Fallback to legacy collector for old models (JSON format)
                    from Models.training_data_collector import get_training_collector
                    logging.info("[MCTS] Using legacy training collector (JSON format)")
                    self._training_collector = get_training_collector()
                
                logging.info(f"[MCTS] Training data collector initialized successfully: {self._training_collector is not None}")
            except Exception as e:
                logging.error(f"[MCTS] Training data collector failed: {e}")
                import traceback
                logging.error(f"[MCTS] Traceback: {traceback.format_exc()}")
                self._training_collector = None
        else:
            logging.info("[MCTS] Training data collection disabled")
            self._training_collector = None

        # Neural integration (policy priors + value bootstrap)
        self._neural = None
        self._neural_policy_blend = 0.5  # Blend NN policy with heuristic priors
        self._neural_value_blend = 0.5    # Blend NN value with heuristic leaf eval
        
        # Skip legacy neural MCTS if enhanced MCTS is available (prevents dimension conflicts)
        if hasattr(self, 'enhanced_model') and self.enhanced_model is not None:
            logging.info("[MCTS] Enhanced MCTS available - skipping legacy neural integration")
        else:
            try:
                from Models.neural_mcts import NeuralMCTSModel
                self._neural = NeuralMCTSModel(self, device='cpu')
                logging.info("[MCTS] Neural integration available (policy/value)")
            except Exception as e:
                logging.info(f"[MCTS] Neural integration unavailable: {e}")

    # Tunables setters for UI
    def set_simulations(self, n: int):
        try:
            self.simulations = max(1, int(n))
        except Exception:
            pass

    def set_c_puct(self, v: float):
        try:
            self.c_puct = float(v)
        except Exception:
            pass

    def set_rollout_depth(self, d: int):
        try:
            self.rollout_depth = max(0, int(d))
        except Exception:
            pass

    def set_time_limit(self, t: float):
        try:
            self.time_limit = max(0.1, float(t))
        except Exception:
            pass

    # Training collector management
    def _ensure_training_collector(self):
        """Create the training collector lazily when collection is enabled.

        This allows UI toggles (collect_training_data) after engine construction
        to take effect without recreating the engine.
        """
        try:
            if getattr(self, 'collect_training_data', False) and getattr(self, '_training_collector', None) is None:
                # Use standalone collector if enhanced MCTS is available (830 dims, tensor format)
                if hasattr(self, 'enhanced_model') and self.enhanced_model is not None:
                    from Models.standalone_data_collector import create_standalone_collector
                    self._training_collector = create_standalone_collector("UI/training_data/standalone")
                else:
                    # Fallback to legacy collector for old models (JSON format)
                    from Models.training_data_collector import get_training_collector
                    self._training_collector = get_training_collector()
        except Exception:
            # Leave collector as None on failure
            self._training_collector = None

    def set_verbose(self, flag: bool = True):
        self._verbose = bool(flag)

    def set_llm_enabled(self, flag: bool = True):
        self._llm_enabled = bool(flag)
        if self._llm_enabled:
            logging.info(f"LLM ENHANCEMENT ENABLED - Integration: {'Available' if self.llm_integration else 'Not Available'}")
        else:
            logging.info("LLM ENHANCEMENT DISABLED")

    def set_openai_api_key(self, api_key: str):
        """Set OpenAI API key and switch to real LLM."""
        try:
            self._api_key = api_key
            if api_key and api_key.strip():
                from Models.llm_integration import create_llm_integration
                self.llm_integration = create_llm_integration("openai", api_key=api_key.strip())
            else:
                # Switch back to mock if no API key
                from Models.mock_llm_provider import create_mock_llm_integration
                self.llm_integration = create_mock_llm_integration()
        except Exception as e:
            # Fallback to mock on error
            try:
                from Models.mock_llm_provider import create_mock_llm_integration
                self.llm_integration = create_mock_llm_integration()
            except Exception:
                pass
    
    def set_enhancements_enabled(self, enabled: bool):
        """Enable or disable enhanced MCTS features."""
        self.use_enhancements = enabled
        if enabled and not self.enhanced_model:
            # Try to initialize enhanced model if not already done
            try:
                from Models.enhanced_mcts_model import create_enhanced_mcts_model
                enhancement_config = {
                    'use_ismcts': True,
                    'use_neural_networks': True, 
                    'use_enhanced_llm': True,
                    'neural_network_device': 'cpu',
                    'decision_mode': 'adaptive',
                    'training': {'enable_training': False}
                }
                self.enhanced_model = create_enhanced_mcts_model(self, enhancement_config)
                logging.info("[MCTS] Enhanced MCTS features enabled")
            except Exception as e:
                logging.error(f"[MCTS] Failed to enable enhancements: {e}")
                self.use_enhancements = False
        else:
            status = "enabled" if enabled else "disabled"
            logging.info(f"[MCTS] Enhanced MCTS features {status}")
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get status of enhancement features."""
        if self.enhanced_model:
            return {
                'enhancements_enabled': self.use_enhancements,
                'enhanced_model_available': True,
                'available_enhancements': self.enhanced_model.get_available_enhancements(),
                'performance_stats': self.enhanced_model.get_performance_report()
            }
        else:
            return {
                'enhancements_enabled': False,
                'enhanced_model_available': False,
                'available_enhancements': {},
                'performance_stats': {}
            }

    # -------------- Internal MCTS machinery --------------
    class _Node:
        __slots__ = ("P", "N", "W", "Q", "children", "expanded", "order")
        def __init__(self, priors: Dict[str, float]):
            self.P: Dict[str, float] = dict(priors)
            self.N: Dict[str, int] = {a: 0 for a in self.P}
            self.W: Dict[str, float] = {a: 0.0 for a in self.P}
            self.Q: Dict[str, float] = {a: 0.0 for a in self.P}
            self.children: Dict[str, Optional[MCTSModel._Node]] = {a: None for a in self.P}
            self.expanded: List[str] = []
            self.order: List[str] = sorted(self.P.keys(), key=lambda k: self.P[k], reverse=True)

    @staticmethod
    def _act_key(kind: str, ident: str) -> str:
        return f"{kind}:{ident}"

    @staticmethod
    def _softmax(xs: List[float], temp: float = 1.0) -> List[float]:
        if not xs:
            return []
        if temp <= 0:
            temp = 1e-6
        m = max(xs)
        exps = [math.exp((x - m) / temp) for x in xs]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    @staticmethod
    def _discretize_hp(frac: float) -> int:
        return max(0, min(20, int(round((frac or 0.0) * 20))))

    def _hash_state(self, state, my_key: str, opp_key: str) -> Tuple:
        def _boosts_tuple(ps) -> Tuple[int, int, int, int, int]:
            try:
                b = ps.stats.boosts
                return (int(b.get('atk', 0)), int(b.get('def', 0)), int(b.get('spa', 0)), 
                       int(b.get('spd', 0)), int(b.get('spe', 0)))
            except Exception:
                return (0, 0, 0, 0, 0)
                
        def _status_tuple(ps) -> str:
            try:
                return str(getattr(ps, 'status', '') or '')
            except Exception:
                return ''
                
        try:
            me = state.team.ours[my_key]
            opp = state.team.opponent[opp_key]
            me_id = (getattr(me, 'species', None) or '').lower()
            opp_id = (getattr(opp, 'species', None) or '').lower()
            me_hp_b = self._discretize_hp(_hp_frac(me))
            opp_hp_b = self._discretize_hp(_hp_frac(opp))
            boosts_me = _boosts_tuple(me)
            boosts_opp = _boosts_tuple(opp)
            status_me = _status_tuple(me)
            status_opp = _status_tuple(opp)
            
            # Field conditions
            field = state.field
            trick = bool(getattr(field, 'trick_room', False))
            terr = str(getattr(field, 'terrain', None) or '')
            weather = str(getattr(field, 'weather', None) or '')
            refl = bool(getattr(field, 'reflect', False))
            lscr = bool(getattr(field, 'light_screen', False))
            avr = bool(getattr(field, 'aurora_veil', False))
            
            # Hazards state
            our_side = getattr(state, 'our_side', {}) or {}
            opp_side = getattr(state, 'opp_side', {}) or {}
            our_hazards = (bool(our_side.get('stealth_rock')), int(our_side.get('spikes', 0)),
                          int(our_side.get('toxic_spikes', 0)), bool(our_side.get('sticky_web')))
            opp_hazards = (bool(opp_side.get('stealth_rock')), int(opp_side.get('spikes', 0)),
                          int(opp_side.get('toxic_spikes', 0)), bool(opp_side.get('sticky_web')))
            
            return (me_id, opp_id, me_hp_b, opp_hp_b, boosts_me, boosts_opp, status_me, status_opp,
                   trick, terr, weather, refl, lscr, avr, our_hazards, opp_hazards)
        except Exception:
            return (None,)

    def _sample_opponent_action(self, state, my_key: str, opp_key: str, mi: MovesInfo, temp: float = 0.5) -> Tuple[str, str]:
        opp = state.team.opponent[opp_key]
        mv_ids: List[str] = []
        mv_scores: List[float] = []
        try:
            for mv in (opp.moves or []):
                mid = getattr(mv, 'id', None)
                if not mid:
                    continue
                cat = _get_move_category(mv)
                bp = int(getattr(mv, 'base_power', 0) or 0)
                if cat == 'status' or bp <= 0:
                    continue
                expf, _ = _expected_damage_fraction(state, opp_key, my_key, mid, mi)
                mv_ids.append(mid)
                mv_scores.append(expf * _acc_to_prob(getattr(mv, 'accuracy', None)))
        except Exception:
            pass
        # Also include plausible STABs to avoid tunnel-vision on partially revealed sets
        try:
            for t in (opp.types or []) or []:
                for mid in _COMMON_STAB.get(str(t).lower(), [])[:2]:
                    if mid in mv_ids:
                        continue
                    expf, _ = _expected_damage_fraction(state, opp_key, my_key, mid, mi)
                    acc = (mi.raw(mid) or {}).get('accuracy', None)
                    mv_ids.append(mid)
                    mv_scores.append(expf * _acc_to_prob(acc))
        except Exception:
            pass
        if not mv_ids:
            try:
                for t in (opp.types or []):
                    for mid in _COMMON_STAB.get(str(t).lower(), [])[:2]:
                        expf, _ = _expected_damage_fraction(state, opp_key, my_key, mid, mi)
                        acc = (mi.raw(mid) or {}).get('accuracy', None)
                        mv_ids.append(mid)
                        mv_scores.append(expf * _acc_to_prob(acc))
            except Exception:
                pass
        # Enhanced switch evaluation with meta predictions
        sw_ids: List[str] = []
        sw_scores: List[float] = []
        
        # Get team context if available
        revealed_team = []
        try:
            for k, ps in state.team.opponent.items():
                species = getattr(ps, 'species', None)
                if species:
                    revealed_team.append(species.lower())
        except Exception:
            pass
        
        # Predict unrevealed Pokemon
        predicted_switches = []
        if self.opponent_predictor and len(revealed_team) < 6:
            try:
                predicted_switches = self.opponent_predictor.predict_unrevealed_pokemon(revealed_team)
            except Exception:
                pass
        # Evaluate known switches with enhanced prediction
        try:
            for k, ps in state.team.opponent.items():
                if k == opp_key:
                    continue
                if getattr(ps, 'current_hp', 0) <= 0:
                    continue
                    
                outgoing = 0.0
                species = getattr(ps, 'species', '').lower()
                
                # Evaluate known moves
                for mv2 in getattr(ps, 'moves', []) or []:
                    mid2 = getattr(mv2, 'id', None)
                    bp2 = int(getattr(mv2, 'base_power', 0) or 0)
                    cat2 = _get_move_category(mv2)
                    if not mid2 or cat2 == 'status' or bp2 <= 0:
                        continue
                    try:
                        frac2, _ = _expected_damage_fraction(state, k, my_key, mid2, mi)
                        outgoing = max(outgoing, frac2 * _acc_to_prob(getattr(mv2, 'accuracy', None)))
                    except Exception:
                        pass
                
                # If we don't know many moves, predict based on species
                if len(getattr(ps, 'moves', []) or []) < 2 and self.opponent_predictor:
                    try:
                        known_moves = {getattr(m, 'id', '').lower() for m in (getattr(ps, 'moves', []) or [])}
                        predicted = self.opponent_predictor.predict_moves(species, known_moves)
                        
                        for pred_move, prob in list(predicted.items())[:2]:  # Top 2 predictions
                            try:
                                frac2, _ = _expected_damage_fraction(state, k, my_key, pred_move, mi)
                                acc = (mi.raw(pred_move) or {}).get('accuracy', 1.0)
                                outgoing = max(outgoing, frac2 * _acc_to_prob(acc) * prob * 0.7)
                            except Exception:
                                continue
                    except Exception:
                        pass
                        
                sw_ids.append(k)
                sw_scores.append(outgoing)
        except Exception:
            pass
        
        # Add predicted unrevealed switches
        for pred_species, pred_prob in predicted_switches[:2]:  # Top 2 predictions
            if pred_prob < 0.2:  # Skip low-probability predictions
                continue
                
            try:
                # Estimate threat level based on meta knowledge
                base_threat = 0.4  # Default moderate threat
                if self.meta_knowledge and pred_species in self.meta_knowledge.meta_threats:
                    threat_data = self.meta_knowledge.meta_threats[pred_species]
                    base_threat = threat_data.danger_level * 0.6  # Scale down for uncertainty
                
                sw_ids.append(f"predicted_{pred_species}")
                sw_scores.append(base_threat * pred_prob * 0.5)  # Penalty for being predicted
            except Exception:
                continue
        actions: List[Tuple[str, str]] = []
        scores: List[float] = []
        for i, mid in enumerate(mv_ids):
            actions.append(("move", mid)); scores.append(mv_scores[i])
        for i, sid in enumerate(sw_ids):
            actions.append(("switch", sid)); scores.append(0.5 * sw_scores[i])
        if not actions:
            return ("move", mv_ids[0]) if mv_ids else ("move", "tackle")
        # Adaptive temperature based on position complexity
        complexity = len(actions) / 10.0  # Scale complexity
        adjusted_temp = max(0.3, min(1.0, temp + 0.2 * complexity))
        probs = self._softmax(scores, temp=adjusted_temp)
        r = random.random(); cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return actions[i]
        return actions[-1]

    def _apply_damage_sample(self, state, atk_key: str, dfd_key: str, move_id: str, mi: MovesInfo) -> float:
        try:
            fail, _ = would_fail(str(move_id), atk_key, dfd_key, state, mi)
            if fail:
                return 0.0
        except Exception:
            pass
        try:
            acc = None
            user = state.team.ours.get(atk_key) or state.team.opponent.get(atk_key)
            for mv in (getattr(user, 'moves', []) or []):
                if getattr(mv, 'id', None) == str(move_id):
                    acc = getattr(mv, 'accuracy', None); break
            if acc is None:
                acc = (mi.raw(str(move_id)) or {}).get('accuracy', None)
            if random.random() > _acc_to_prob(acc):
                return 0.0
        except Exception:
            pass
        try:
            # Try Rust engine for damage calculation if available
            if RUST_ENGINE_AVAILABLE:
                try:
                    atk_ps = state.team.ours.get(atk_key) or state.team.opponent.get(atk_key)
                    dfd_ps = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
                    
                    if atk_ps and dfd_ps:
                        atk_dict = {'level': getattr(atk_ps, 'level', 50)}
                        dfd_dict = {'level': getattr(dfd_ps, 'level', 50)}
                        move_dict = {'base_power': 80}  # Default base power
                        field_dict = {}
                        
                        damage_rolls = pokechad_engine.calculate_damage_fast(
                            atk_dict, dfd_dict, move_dict, field_dict
                        )
                        if damage_rolls:
                            r = random.choice(damage_rolls)
                            max_hp = int(getattr(dfd_ps, 'max_hp', 1) or 1)
                            if max_hp <= 0:
                                max_hp = 1
                            return max(0.0, min(1.0, float(r) / float(max_hp)))
                except Exception:
                    pass  # Fall back to Python implementation
            
            dmg = estimate_damage(state, atk_key, dfd_key, str(move_id), mi, is_critical=False)
            rolls = dmg.get('rolls') or []
            if not rolls:
                return 0.0
            r = random.choice(rolls)
            dfd = state.team.ours.get(dfd_key) or state.team.opponent.get(dfd_key)
            max_hp = int(getattr(dfd, 'max_hp', 1) or 1)
            if max_hp <= 0:
                max_hp = 1
            return max(0.0, min(1.0, float(r) / float(max_hp)))
        except Exception:
            return 0.0

    def _update_hp(self, ps, frac_loss: float):
        try:
            if getattr(ps, 'max_hp', None) is None:
                return
            dmg = int(round(frac_loss * ps.max_hp))
            ps.current_hp = max(0, int((ps.current_hp or ps.max_hp) - dmg))
            ps.hp_fraction = max(0.0, min(1.0, float(ps.current_hp) / float(ps.max_hp)))
            if ps.current_hp <= 0:
                ps.status = 'fnt'
        except Exception:
            pass

    def _apply_switch(self, state, *, side: str, from_key: str, to_key: str, mi: MovesInfo):
        try:
            team = state.team.ours if side == 'ally' else state.team.opponent
            # clear boosts on the mon that leaves
            try:
                if from_key in team:
                    b = team[from_key].stats.boosts
                    for stat in list(b.keys()):
                        b[stat] = 0
            except Exception:
                pass
            for k, ps in team.items():
                if k == to_key:
                    ps.is_active = True
                elif k == from_key:
                    ps.is_active = False
        except Exception:
            pass
        try:
            apply_switch_in_effects(state, to_key, 'ally' if side == 'ally' else 'opponent', mi, mutate=True)
        except Exception:
            pass

    def _apply_my_status_move_effects(self, state, my_key: str, move_id: str, reward_acc: List[float]):
        mid = (move_id or '').lower()
        
        # Healing moves: rough heal
        HEALS = {"recover","roost","softboiled","morningsun","synthesis","slackoff","milkdrink","shoreup","junglehealing","rest"}
        if mid in HEALS:
            try:
                ps = state.team.ours[my_key]
                if getattr(ps, 'max_hp', None):
                    current_frac = _hp_frac(ps)
                    heal_amount = 0.5 if mid != 'rest' else 1.0
                    target_frac = min(1.0, current_frac + heal_amount)
                    healed = max(0.0, target_frac - current_frac)
                    if healed > 0:
                        heal_hp = int(round(healed * ps.max_hp))
                        ps.current_hp = min(ps.max_hp, (ps.current_hp or ps.max_hp) + heal_hp)
                        ps.hp_fraction = max(0.0, min(1.0, float(ps.current_hp) / float(ps.max_hp)))
                        # Healing value scales with missing HP
                        heal_value = healed * (1.0 + (1.0 - current_frac))
                        reward_acc[0] += heal_value
            except Exception:
                pass
                
        # Hazards: set on opponent side
        if mid in {"stealthrock","spikes","toxicspikes","stickyweb"}:
            try:
                sd = state.opp_side
                hazard_value = 0.0
                
                # Enhanced hazard evaluation with strategic considerations
                if mid == 'stealthrock' and not sd.get('stealth_rock'):
                    sd['stealth_rock'] = True
                    base_value = 0.15
                    
                    # Bonus if opponent has Pokemon weak to SR
                    sr_bonus = 0.0
                    try:
                        for opp_pokemon in state.team.opponent.values():
                            if not opp_pokemon:
                                continue
                            # Check for Rock weakness (SR does 25% to 4x weak, 12.5% to 2x weak)
                            types = getattr(opp_pokemon, 'types', [])
                            rock_effectiveness = 1.0
                            for ptype in types:
                                if str(ptype).lower() in ['fire', 'flying', 'bug', 'ice']:  # 2x weak
                                    rock_effectiveness *= 2.0
                            if rock_effectiveness >= 2.0:
                                sr_bonus += 0.1  # Bonus for each weak Pokemon
                            elif rock_effectiveness >= 4.0:
                                sr_bonus += 0.2  # Extra bonus for 4x weak
                    except Exception:
                        pass
                    
                    hazard_value = base_value + min(sr_bonus, 0.3)  # Cap bonus
                    
                elif mid == 'spikes':
                    cur = int(sd.get('spikes', 0) or 0)
                    if cur < 3:
                        sd['spikes'] = cur + 1
                        base_value = 0.08 + 0.04 * cur  # Diminishing returns
                        
                        # Bonus if opponent lacks reliable hazard removal
                        removal_bonus = 0.0
                        try:
                            has_rapid_spin = False
                            has_defog = False
                            for opp_pokemon in state.team.opponent.values():
                                if not opp_pokemon:
                                    continue
                                moves = getattr(opp_pokemon, 'moves', [])
                                for move in moves:
                                    move_id = getattr(move, 'id', '').lower()
                                    if 'rapid' in move_id and 'spin' in move_id:
                                        has_rapid_spin = True
                                    elif 'defog' in move_id:
                                        has_defog = True
                            
                            if not (has_rapid_spin or has_defog):
                                removal_bonus = 0.05  # No known hazard removal
                                
                        except Exception:
                            pass
                            
                        hazard_value = base_value + removal_bonus
                        
                elif mid == 'toxicspikes':
                    cur = int(sd.get('toxic_spikes', 0) or 0)
                    if cur < 2:
                        sd['toxic_spikes'] = cur + 1
                        base_value = 0.06 + 0.04 * cur
                        
                        # Bonus against stall teams (Pokemon with recovery)
                        stall_bonus = 0.0
                        try:
                            recovery_count = 0
                            for opp_pokemon in state.team.opponent.values():
                                if not opp_pokemon:
                                    continue
                                moves = getattr(opp_pokemon, 'moves', [])
                                for move in moves:
                                    move_id = getattr(move, 'id', '').lower()
                                    if any(heal in move_id for heal in ['recover', 'roost', 'synthesis', 'moonlight']):
                                        recovery_count += 1
                                        break
                            
                            if recovery_count >= 2:  # Stall team
                                stall_bonus = 0.08
                        except Exception:
                            pass
                            
                        hazard_value = base_value + stall_bonus
                        
                elif mid == 'stickyweb' and not sd.get('sticky_web'):
                    sd['sticky_web'] = True
                    base_value = 0.10
                    
                    # Bonus against fast offensive teams
                    speed_bonus = 0.0
                    try:
                        fast_count = 0
                        for opp_pokemon in state.team.opponent.values():
                            if not opp_pokemon:
                                continue
                            stats = getattr(opp_pokemon, 'stats', {})
                            if hasattr(stats, 'spe') and stats.spe > 100:  # Fast Pokemon
                                fast_count += 1
                        
                        if fast_count >= 3:  # Fast team
                            speed_bonus = 0.08
                    except Exception:
                        pass
                        
                    hazard_value = base_value + speed_bonus
                    
                reward_acc[0] += hazard_value
            except Exception:
                pass
                
        # Stat boosts
        STAT_BOOSTS = {
            "swordsdance": {"atk": 2}, "nastyplot": {"spa": 2}, "dragondance": {"atk": 1, "spe": 1},
            "quiverdance": {"spa": 1, "spd": 1, "spe": 1}, "shellsmash": {"atk": 2, "spa": 2, "spe": 2},
            "calmmind": {"spa": 1, "spd": 1}, "bulkup": {"atk": 1, "def": 1}, "irondefense": {"def": 2},
            "amnesia": {"spd": 2}, "agility": {"spe": 2}, "rockpolish": {"spe": 2}
        }
        if mid in STAT_BOOSTS:
            try:
                ps = state.team.ours[my_key]
                boosts = STAT_BOOSTS[mid]
                boost_value = 0.0
                for stat, amount in boosts.items():
                    current = getattr(ps.stats.boosts, stat, 0) if hasattr(ps.stats, 'boosts') else 0
                    if current + amount <= 6:  # Cap at +6
                        if hasattr(ps.stats, 'boosts'):
                            setattr(ps.stats.boosts, stat, min(6, current + amount))
                        # Value offensive boosts more highly
                        if stat in {'atk', 'spa'}:
                            boost_value += 0.2 * amount
                        elif stat == 'spe':
                            boost_value += 0.15 * amount
                        else:
                            boost_value += 0.1 * amount
                reward_acc[0] += boost_value
            except Exception:
                pass
                
        # Utility moves
        UTILITY_MOVES = {
            "substitute": 0.12, "protect": 0.08, "detect": 0.08, "willowisp": 0.15,
            "thunderwave": 0.12, "toxic": 0.18, "sleeppowder": 0.20, "spore": 0.22,
            "stunspore": 0.10, "leechseed": 0.15, "reflect": 0.12, "lightscreen": 0.12,
            "auroraveil": 0.18, "stealthrock": 0.15, "defog": 0.10, "rapidspin": 0.10
        }
        if mid in UTILITY_MOVES:
            reward_acc[0] += UTILITY_MOVES[mid]

    def _simulate_turn(self, state, my_key: str, opp_key: str, my_action: Tuple[str, str], mi: MovesInfo) -> Tuple[Any, str, str, float, bool]:
        s = state
        reward = 0.0
        terminal = False
        kind_me, ident_me = my_action
        kind_opp, ident_opp = self._sample_opponent_action(s, my_key, opp_key, mi, temp=0.7)

        # both switch
        if kind_me == 'switch' and kind_opp == 'switch':
            self._apply_switch(s, side='ally', from_key=my_key, to_key=ident_me, mi=mi)
            self._apply_switch(s, side='opponent', from_key=opp_key, to_key=ident_opp, mi=mi)
            return s, ident_me, ident_opp, 0.0, False
        # our switch, their move
        if kind_me == 'switch' and kind_opp == 'move':
            self._apply_switch(s, side='ally', from_key=my_key, to_key=ident_me, mi=mi)
            df = self._apply_damage_sample(s, opp_key, ident_me, ident_opp, mi)
            if df > 0:
                self._update_hp(s.team.ours.get(ident_me), df)
                reward -= df
            if getattr(s.team.ours.get(ident_me), 'current_hp', 1) <= 0:
                reward -= 1.0; terminal = True
            return s, ident_me, opp_key, reward, terminal
        # their switch, our move
        if kind_me == 'move' and kind_opp == 'switch':
            self._apply_switch(s, side='opponent', from_key=opp_key, to_key=ident_opp, mi=mi)
            # apply our status effects if any before damage
            ra = [0.0]; self._apply_my_status_move_effects(s, my_key, ident_me, ra); reward += ra[0]
            df = self._apply_damage_sample(s, my_key, ident_opp, ident_me, mi)
            if df > 0:
                self._update_hp(s.team.opponent.get(ident_opp), df)
                reward += df
            if getattr(s.team.opponent.get(ident_opp), 'current_hp', 1) <= 0:
                reward += 1.0; terminal = True
            return s, my_key, ident_opp, reward, terminal
        # move vs move
        if kind_me == 'move' and kind_opp == 'move':
            try:
                p_first, _ = predict_order_for_ids(s, my_key, ident_me, opp_key, ident_opp, mi)
            except Exception:
                try:
                    p_first = predict_first_prob_speed_only(s, my_key, opp_key)
                except Exception:
                    p_first = 0.5
            me_first = (random.random() < float(p_first or 0.5))
            acted_first_bonus = 0.05
            if me_first:
                reward += acted_first_bonus
                ra = [0.0]; self._apply_my_status_move_effects(s, my_key, ident_me, ra); reward += ra[0]
                df = self._apply_damage_sample(s, my_key, opp_key, ident_me, mi)
                if df > 0:
                    self._update_hp(s.team.opponent[opp_key], df)
                    reward += df
                if getattr(s.team.opponent[opp_key], 'current_hp', 1) <= 0:
                    reward += 1.0; terminal = True
                else:
                    df2 = self._apply_damage_sample(s, opp_key, my_key, ident_opp, mi)
                    if df2 > 0:
                        self._update_hp(s.team.ours[my_key], df2)
                        reward -= df2
                    if getattr(s.team.ours[my_key], 'current_hp', 1) <= 0:
                        reward -= 1.0; terminal = True
            else:
                reward -= acted_first_bonus
                df2 = self._apply_damage_sample(s, opp_key, my_key, ident_opp, mi)
                if df2 > 0:
                    self._update_hp(s.team.ours[my_key], df2)
                    reward -= df2
                if getattr(s.team.ours[my_key], 'current_hp', 1) <= 0:
                    reward -= 1.0; terminal = True
                else:
                    ra = [0.0]; self._apply_my_status_move_effects(s, my_key, ident_me, ra); reward += ra[0]
                    df = self._apply_damage_sample(s, my_key, opp_key, ident_me, mi)
                    if df > 0:
                        self._update_hp(s.team.opponent[opp_key], df)
                        reward += df
                    if getattr(s.team.opponent[opp_key], 'current_hp', 1) <= 0:
                        reward += 1.0; terminal = True
            return s, my_key, opp_key, reward, terminal
        # nothing
        return s, my_key, opp_key, 0.0, False

    def _progressive_widening_cap(self, n_visits: int) -> int:
        return min(8, 3 + int(1.5 * math.sqrt(max(0, n_visits))))

    def _build_priors(self, moves_eval: List[Dict[str, Any]], switches_eval: List[Dict[str, Any]], branching_cap: int = 6) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        candidates: List[Tuple[str, str, float, Dict[str, Any]]] = []
        for mv in (moves_eval or [])[:]:
            # Phase 1: Enhanced illegal move filtering
            if mv.get('why_blocked'):
                continue
            
            # Additional validation for move legality
            move_id = mv.get('id')
            if not move_id:
                continue
                
            # Skip moves with invalid scores (potential corruption indicators)
            score = mv.get('score', 0.0)
            if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
                continue
                
            # Skip moves with impossible accuracy values
            accuracy = mv.get('acc', 1.0)
            if not isinstance(accuracy, (int, float)) or accuracy < 0 or accuracy > 1.0:
                continue
                
            # Skip moves with negative expected damage (unless status moves)
            expected = mv.get('expected', 0.0)
            category = str(mv.get('category', 'status')).lower()
            if category != 'status' and expected < 0:
                continue
                
            candidates.append(('move', str(move_id), float(score), mv))
        for sw in (switches_eval or [])[:]:
            # Phase 1: Enhanced switch validation
            key = str(sw.get('key') or sw.get('species'))
            if not key or key == 'None':
                continue
                
            # Skip switches with invalid scores
            score = sw.get('score', 0.0)
            if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
                continue
                
            # Skip switches with impossible HP fractions
            hp_fraction = sw.get('hp_fraction', 1.0)
            if not isinstance(hp_fraction, (int, float)) or hp_fraction < 0 or hp_fraction > 1.0:
                continue
                
            candidates.append(('switch', key, float(score), sw))
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:max(1, branching_cap)]
        scores = [c[2] for c in candidates]
        # Dynamic temperature based on score spread
        score_spread = max(scores) - min(scores) if len(scores) > 1 else 1.0
        temp = max(0.5, min(2.0, 1.0 + 0.5 * score_spread))
        probs = self._softmax(scores, temp=temp)
        priors: Dict[str, float] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for (kind, ident, _, m), p in zip(candidates, probs):
            priors[self._act_key(kind, ident)] = float(p)
            meta[self._act_key(kind, ident)] = dict(m)
        return priors, meta

    def _heuristic_evals_for_priors(self, state, my_key: str, opp_key: str, mi: MovesInfo) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        moves_eval: List[Dict[str, Any]] = []
        switches_eval: List[Dict[str, Any]] = []
        
        # Get team context for strategic adjustments
        my_team_list = []
        opp_team_list = []
        try:
            for k, ps in state.team.ours.items():
                species = getattr(ps, 'species', None)
                if species:
                    my_team_list.append(species.lower())
            for k, ps in state.team.opponent.items():
                species = getattr(ps, 'species', None)
                if species:
                    opp_team_list.append(species.lower())
        except Exception:
            pass
        
        # Determine team archetypes and strategic context
        my_archetype = "balance"
        opp_archetype = "balance"
        if self.opponent_predictor:
            try:
                my_archetype = self.opponent_predictor.get_team_archetype(my_team_list)
                opp_archetype = self.opponent_predictor.get_team_archetype(opp_team_list)
            except Exception:
                pass
        try:
            my_eff_spe, opp_eff_spe = get_effective_speeds(state, my_key, opp_key)
        except Exception:
            my_eff_spe, opp_eff_spe = (0, 0)
        trick_room_active = bool(getattr(state.field, 'trick_room', False))
        opp_item_known = False
        try:
            opp_item_known = bool((getattr(state.team.opponent[opp_key], 'item', None) or '').strip())
        except Exception:
            pass
        try:
            opp_ps = state.team.opponent[opp_key]
            opp_hp_frac = _hp_frac(opp_ps)
            opp_max = int(getattr(opp_ps, 'max_hp', 0) or getattr(opp_ps, 'stats', {}).raw.get('hp', 1) or 1)
            my_hp_now = _hp_frac(state.team.ours[my_key])
            
            # Check if terastallization is available from battle snapshot
            can_tera = False
            my_tera_type = None
            try:
                # Get tera availability from the snapshot (set by battle environment)
                snapshot = self.snapshot_battle(None)
                can_tera = snapshot.get('can_tera', False)
                
                # Get my Pokemon's tera type
                if my_key in state.team.ours:
                    my_pokemon = state.team.ours[my_key]
                    my_tera_type = getattr(my_pokemon, 'tera_type', None)
            except Exception:
                pass
            
            for mv in getattr(state.team.ours[my_key], 'moves', []) or []:
                mid = getattr(mv, 'id', None)
                if not mid:
                    continue
                cat = _get_move_category(mv)
                bp = int(getattr(mv, 'base_power', 0) or 0)
                if cat == 'status' or bp <= 0:
                    continue
                try:
                    fail, _ = would_fail(str(mid), my_key, opp_key, state, mi)
                except Exception:
                    fail = False
                if fail:
                    continue
                try:
                    exp_frac, dmg = _expected_damage_fraction(state, my_key, opp_key, str(mid), mi)
                except Exception:
                    exp_frac, dmg = 0.0, {}
                acc_p = _acc_to_prob(getattr(mv, 'accuracy', 1.0))
                try:
                    first_prob, _ = predict_order_for_ids(state, my_key, str(mid), opp_key, 'tackle', mi)
                except Exception:
                    try:
                        first_prob = predict_first_prob_speed_only(state, my_key, opp_key)
                    except Exception:
                        first_prob = 0.5
                mv_base_pri = int(getattr(mv, 'priority', 0) or 0) if hasattr(mv, 'priority') else 0
                if (not trick_room_active) and (mv_base_pri <= 0) and (not opp_item_known) and (my_eff_spe > opp_eff_spe > 0) and ((opp_eff_spe * 3) > (my_eff_spe * 2)):
                    first_prob = min(float(first_prob), 0.30)
                rolls = (dmg.get('rolls') or [])
                thr_abs = int(round(opp_hp_frac * opp_max))
                ko_rolls = sum(1 for r in rolls if int(r) >= max(1, thr_abs))
                p_ko_if_hit = (ko_rolls / max(1, len(rolls))) if rolls else 0.0
                try:
                    incoming_best = self._get_opp_best_damage(state, opp_key, my_key, mi)
                    # Adjust for opponent recovery - they can afford to take more damage
                    if _has_recovery_item(state.team.opponent.get(opp_key)):
                        incoming_best *= 0.8  # They heal, so effective threat is lower
                except Exception:
                    incoming_best = 0.0
                p_opp_acts = (1 - float(first_prob)) + float(first_prob) * (1 - p_ko_if_hit * acc_p)
                opp_counter_ev = incoming_best * p_opp_acts
                effective_exp_raw = exp_frac * acc_p
                p_die_if_opp_hits = min(1.0, incoming_best / max(1e-9, my_hp_now)) if my_hp_now > 0 else 1.0
                p_survive_before_act = float(first_prob) + (1 - float(first_prob)) * (1 - p_die_if_opp_hits)
                effective_exp = effective_exp_raw * max(0.0, min(1.0, p_survive_before_act))
                # Stronger penalty for exposure relative to our current HP (normalized)
                incoming_rel = min(1.0, incoming_best / max(1e-9, my_hp_now)) if my_hp_now > 0 else 1.0
                exposure_penalty = 0.0
                if incoming_rel >= 0.7 and (acc_p * p_ko_if_hit) < 0.4:
                    exposure_penalty = 0.2 + 0.3 * incoming_rel
                # Improved scoring with better balance and accuracy consideration
                ko_bonus = 0.8 * (acc_p * p_ko_if_hit)
                speed_bonus = 0.3 * float(first_prob)
                damage_value = 1.2 * effective_exp
                safety_penalty = 0.8 * opp_counter_ev
                
                # Penalty for missing crucial KOs
                accuracy_risk = 0.0
                if p_ko_if_hit > 0.8 and acc_p < 0.9:  # High KO chance but risky accuracy
                    accuracy_risk = 0.3 * (1.0 - acc_p) * p_ko_if_hit
                    
                # Bonus for reliable moves in endgame
                if my_hp_now < 0.3 and acc_p >= 0.95:  # Endgame reliability bonus
                    damage_value *= 1.2
                    
                score = damage_value + ko_bonus + speed_bonus - safety_penalty - exposure_penalty - accuracy_risk
                # Extract and include type effectiveness multiplier for UI/logging
                try:
                    eff_mult = float(dmg.get('effectiveness', 1.0)) if isinstance(dmg, dict) else 1.0
                except Exception:
                    eff_mult = 1.0
                moves_eval.append({'id': mid, 'name': getattr(mv, 'name', mid), 'score': float(score), 'expected': float(exp_frac), 'acc': float(acc_p), 'first_prob': float(first_prob), 'p_ko_if_hit': float(p_ko_if_hit), 'opp_counter_ev': float(opp_counter_ev), 'effectiveness': float(eff_mult)})
            # Add terastallized versions of moves if tera is available
            if can_tera and my_tera_type:
                tera_moves = []
                for move_data in moves_eval[:]:
                    try:
                        mid = move_data.get('id')
                        if not mid or 'tera:' in str(mid):
                            continue
                            
                        # Calculate tera bonus with proper type effectiveness
                        tera_bonus = 0.3  # Base tera strategic value
                        
                        # Try to calculate proper tera effectiveness and STAB
                        try:
                            # Get enhanced tera damage calculation
                            orig_exp_frac = move_data.get('expected', 0.0)
                            tera_exp_frac, tera_dmg = _expected_damage_fraction_with_tera(state, my_key, opp_key, str(mid), mi, my_tera_type)
                            
                            # Calculate damage improvement from tera
                            if orig_exp_frac > 0:
                                damage_improvement = (tera_exp_frac - orig_exp_frac) / orig_exp_frac
                                tera_bonus += min(0.8, max(0.0, damage_improvement))  # Cap at +0.8
                            
                            # Check for STAB bonus from tera type effectiveness calculation
                            if tera_dmg.get('tera_stab_bonus', 1.0) > 1.0:
                                tera_bonus += 0.4  # Confirmed STAB bonus
                                
                            # Update move data with tera calculations
                            move_data = dict(move_data)
                            move_data.update({
                                'expected': tera_exp_frac,
                                'effectiveness': tera_dmg.get('effectiveness', move_data.get('effectiveness', 1.0)),
                                'tera_damage_improvement': damage_improvement if orig_exp_frac > 0 else 0.0
                            })
                            
                        except Exception as e:
                            logging.debug(f"Enhanced tera calculation failed for {mid}: {e}")
                            # Fallback to simple heuristic STAB detection
                            move_name = move_data.get('name', '').lower()
                            tera_type_lower = my_tera_type.lower()
                            
                            # Simple type matching heuristics as fallback
                            type_hints = {
                                'electric': ['electro', 'thunder', 'volt', 'spark'],
                                'fire': ['fire', 'flame', 'burn', 'heat'],
                                'water': ['water', 'surf', 'hydro', 'aqua'],
                                'grass': ['leaf', 'grass', 'solar', 'energy'],
                                'ice': ['ice', 'blizzard', 'freeze', 'hail'],
                                'fighting': ['punch', 'kick', 'combat', 'fight'],
                                'poison': ['poison', 'toxic', 'acid', 'sludge'],
                                'ground': ['earthquake', 'earth', 'ground', 'sand'],
                                'flying': ['air', 'wing', 'gust', 'hurricane'],
                                'psychic': ['psychic', 'psych', 'confusion', 'mental'],
                                'bug': ['bug', 'buzz', 'leech', 'pin'],
                                'rock': ['rock', 'stone', 'slide', 'tomb'],
                                'ghost': ['shadow', 'spirit', 'ghost', 'hex'],
                                'dragon': ['dragon', 'draco', 'outrage', 'meteor'],
                                'dark': ['dark', 'crunch', 'bite', 'knock'],
                                'steel': ['steel', 'iron', 'metal', 'flash'],
                                'fairy': ['fairy', 'moon', 'play', 'charm']
                            }
                            
                            if tera_type_lower in type_hints:
                                for hint in type_hints[tera_type_lower]:
                                    if hint in move_name:
                                        tera_bonus += 0.5  # STAB bonus
                                        break
                        
                        # Strategic timing bonus
                        if move_data.get('p_ko_if_hit', 0) > 0.7:
                            tera_bonus += 0.3
                        if my_hp_now < 0.4:
                            tera_bonus += 0.2
                            
                        tera_score = move_data.get('score', 0.0) + tera_bonus
                        
                        # Create tera move entry
                        tera_move = dict(move_data)
                        tera_move.update({
                            'id': f'tera:{mid}',
                            'name': f'Tera {move_data.get("name", mid)}',
                            'score': float(tera_score),
                            'tera_type': my_tera_type,
                            'is_tera': True
                        })
                        tera_moves.append(tera_move)
                        
                    except Exception as e:
                        logging.debug(f"Failed to create tera move: {e}")
                        continue
                        
                moves_eval.extend(tera_moves)
            
            moves_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        except Exception:
            pass
        try:
            for k, ps in state.team.ours.items():
                if k == my_key:
                    continue
                if getattr(ps, 'current_hp', 0) <= 0:
                    continue
                outgoing = 0.0
                for mv in getattr(ps, 'moves', []) or []:
                    mid = getattr(mv, 'id', None)
                    bp = int(getattr(mv, 'base_power', 0) or 0)
                    cat = _get_move_category(mv)
                    if not mid or cat == 'status' or bp <= 0:
                        continue
                    try:
                        frac2, _ = _expected_damage_fraction(state, k, opp_key, mid, mi)
                        outgoing = max(outgoing, frac2 * _acc_to_prob(getattr(mv, 'accuracy', None)))
                    except Exception:
                        pass
                try:
                    hazards = apply_switch_in_effects(state, k, 'ally', mi, mutate=False)
                    haz = float(hazards.get('fraction_lost', 0.0) or 0.0)
                except Exception:
                    haz = 0.0
                incoming = 0.0
                try:
                    incoming = self._get_opp_best_damage(state, opp_key, k, mi)
                except Exception:
                    pass
                # Improved switch evaluation scoring
                hp_frac = _hp_frac(ps)
                
                # Defensive value - resist opponent's attacks
                resistance_bonus = 0.0
                if incoming < 0.2:  # Resists well
                    resistance_bonus = 0.8
                elif incoming < 0.4:  # Decent resistance
                    resistance_bonus = 0.5
                elif incoming < 0.6:  # Some resistance
                    resistance_bonus = 0.3
                
                # Offensive potential of the switch-in
                offensive_value = min(1.2, outgoing * 1.5)  # Scale up and cap offensive value
                
                # Health bonus - prefer healthy Pokemon
                health_bonus = hp_frac * 0.4
                
                # Hazard penalty (much reduced)
                hazard_penalty = haz * 0.6  # Reduced from implied 1.2
                
                # Emergency switch bonus - if current Pokemon is in danger
                emergency_bonus = 0.0
                try:
                    # Get current Pokemon's HP and threat level
                    current_pokemon = state.team.ours.get(my_key)
                    if current_pokemon:
                        current_hp = _hp_frac(current_pokemon)
                        current_threat = self._get_opp_best_damage(state, opp_key, my_key, mi)
                        
                        # If current Pokemon is in serious danger, boost switch value
                        if current_hp < 0.4 and current_threat > 0.6:
                            emergency_bonus = 0.6
                        elif current_hp < 0.6 and current_threat > 0.8:
                            emergency_bonus = 0.4
                except Exception:
                    pass
                
                # Strategic switch bonus - type advantage
                type_advantage_bonus = 0.0
                if outgoing > 0.8:  # Can threaten opponent significantly
                    type_advantage_bonus = 0.3
                
                # Final score calculation
                score = (offensive_value + resistance_bonus + health_bonus + 
                        emergency_bonus + type_advantage_bonus - hazard_penalty)
                switches_eval.append({'key': k, 'species': getattr(ps, 'species', k), 'score': float(score), 'outgoing_frac': float(outgoing), 'incoming_on_switch': float(haz + incoming), 'hazards_frac': float(haz), 'hp_fraction': float(hp_frac)})
            switches_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        except Exception:
            pass
        return moves_eval, switches_eval

    def _mcts_plan(self, state, my_key: str, opp_key: str, moves_eval: List[Dict[str, Any]], switches_eval: List[Dict[str, Any]], force_switch: bool, mi: MovesInfo) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        branching_cap = 6
        # Adaptive branching based on position complexity
        position_complexity = len(moves_eval) + len(switches_eval)
        adaptive_cap = min(8, max(3, branching_cap + int(position_complexity / 5)))
        
        priors, meta = self._build_priors(moves_eval, switches_eval, branching_cap=(3 if force_switch else adaptive_cap))
        heur_priors_snapshot = dict(priors)
        nn_pol_snapshot = None
        # Blend with neural policy if available
        if self._neural and priors:
            try:
                cand_keys = list(priors.keys())
                nn_pol, _ = self._neural.policy_value_for_candidates(state, cand_keys)
                nn_pol_snapshot = dict(nn_pol)
                # Normalize both and blend
                s_h = sum(priors.values()) or 1.0
                s_n = sum(nn_pol.get(k, 0.0) for k in cand_keys) or 1.0
                blended = {}
                for k in cand_keys:
                    ph = (priors.get(k, 0.0) / s_h)
                    pn = (nn_pol.get(k, 0.0) / s_n)
                    blended[k] = float(self._neural_policy_blend * ph + (1.0 - self._neural_policy_blend) * pn)
                priors = blended
            except Exception:
                pass
        if force_switch:
            priors = {k: v for k, v in priors.items() if k.startswith('switch:')}
            if not priors:
                for sw in switches_eval[:3]:
                    key = str(sw.get('key') or sw.get('species'))
                    priors[self._act_key('switch', key)] = 1.0
        root = self._Node(priors)
        table: Dict[Tuple, MCTSModel._Node] = {}
        table[self._hash_state(state, my_key, opp_key)] = root

        start = time.time()
        sim_budget = max(1, int(self.simulations))
        max_depth = max(1, int(self.rollout_depth))
        iters = 0
        
        # Performance tracking
        last_progress_check = start
        progress_interval = self.time_limit / 10  # Check progress 10 times during search
        
        # (Disabled) Rust engine acceleration placeholder; falling back to Python implementation
        # The previous Rust block referenced out-of-scope symbols; keep Python path for correctness.
        # if RUST_ENGINE_AVAILABLE and iters == 0:
        #     pass

        while iters < sim_budget and (time.time() - start) < self.time_limit:
            iters += 1
            
            # Early termination if time is running low
            current_time = time.time()
            if current_time - start > self.time_limit * 0.9:
                break
                
            # Progress monitoring
            if current_time - last_progress_check > progress_interval:
                if iters > 0:
                    # Estimate if we can complete another meaningful batch
                    time_per_iter = (current_time - start) / iters
                    remaining_time = self.time_limit - (current_time - start)
                    if remaining_time < time_per_iter * 5:  # Need at least 5 more iterations
                        break
                last_progress_check = current_time
            
            # Use more efficient state copying for simulation
            try:
                s = copy.deepcopy(state)
            except Exception:
                # Fallback to shallow copy if deepcopy fails
                s = copy.copy(state)
            cur_my = my_key
            cur_opp = opp_key
            path: List[Tuple[MCTSModel._Node, str]] = []
            node = root
            depth = 0
            total_reward = 0.0
            terminal = False
            while True:
                # widening
                if len(node.expanded) < min(len(node.order), self._progressive_widening_cap(sum(node.N.values()))):
                    next_act_key = None
                    for ak in node.order:
                        if ak not in node.expanded:
                            next_act_key = ak; break
                    if next_act_key is not None:
                        node.expanded.append(next_act_key)
                        a_kind, a_ident = next_act_key.split(':', 1)
                        try:
                            s2 = copy.deepcopy(s)
                        except Exception:
                            s2 = copy.copy(s)
                        s2, cur_my2, cur_opp2, r, term = self._simulate_turn(s2, cur_my, cur_opp, (a_kind, a_ident), mi)
                        total_reward += r
                        path.append((node, next_act_key))
                        new_hash = self._hash_state(s2, cur_my2, cur_opp2)
                        child = table.get(new_hash)
                        if child is None:
                            child_moves, child_switches = self._heuristic_evals_for_priors(s2, cur_my2, cur_opp2, mi)
                            child_priors, _ = self._build_priors(child_moves, child_switches, branching_cap=branching_cap)
                            if self._neural and child_priors:
                                try:
                                    cand = list(child_priors.keys())
                                    nn_pol, _ = self._neural.policy_value_for_candidates(s2, cand)
                                    s_h = sum(child_priors.values()) or 1.0
                                    s_n = sum(nn_pol.get(k, 0.0) for k in cand) or 1.0
                                    for k in cand:
                                        ph = child_priors.get(k, 0.0) / s_h
                                        pn = nn_pol.get(k, 0.0) / s_n
                                        child_priors[k] = float(self._neural_policy_blend * ph + (1.0 - self._neural_policy_blend) * pn)
                                except Exception:
                                    pass
                            child = self._Node(child_priors)
                            table[new_hash] = child
                        node.children[next_act_key] = child
                        node = child; s = s2; cur_my, cur_opp = cur_my2, cur_opp2
                        terminal = term; depth += 1
                        if terminal or depth >= max_depth:
                            break
                        continue
                if not node.expanded:
                    break
                total_N = 1 + sum(node.N[a] for a in node.expanded)
                best_ak = None; best_score = -1e9
                for ak in node.expanded:
                    q = node.Q.get(ak, 0.0)
                    n = node.N.get(ak, 0)
                    p = node.P.get(ak, 0.0)
                    # Enhanced UCB with improved exploration
                    
                    # Base exploration term
                    exploration = self.c_puct * p * math.sqrt(total_N) / (1 + n)
                    
                    # Adaptive exploration boost to prevent over-exploitation
                    max_visits = max(node.N.values()) if node.N else 1
                    if max_visits > 1000:  # If we're heavily exploiting one move
                        # Boost exploration for less-visited moves
                        visit_ratio = n / max_visits
                        if visit_ratio < 0.1:  # Less than 10% of max visits
                            exploration *= 2.0  # Double exploration
                        elif visit_ratio < 0.3:  # Less than 30% of max visits
                            exploration *= 1.5  # 1.5x exploration
                    
                    # Progressive exploration bonus for diversity
                    diversity_bonus = 0.0
                    if len(node.expanded) > 2:  # Multiple options available
                        min_visits = min(node.N.values()) if node.N else 0
                        if min_visits < 50:  # Some moves barely explored
                            diversity_bonus = 0.1 * (1.0 - n / max(total_N, 1))
                    
                    # Add small noise for tie-breaking
                    noise = 0.01 * (random.random() - 0.5)
                    
                    sc = q + exploration + diversity_bonus + noise
                    if sc > best_score:
                        best_score = sc; best_ak = ak
                if best_ak is None:
                    break
                path.append((node, best_ak))
                a_kind, a_ident = best_ak.split(':', 1)
                try:
                    s2 = copy.deepcopy(s)
                except Exception:
                    s2 = copy.copy(s)
                s2, cur_my2, cur_opp2, r, term = self._simulate_turn(s2, cur_my, cur_opp, (a_kind, a_ident), mi)
                total_reward += r
                child = node.children.get(best_ak)
                if child is None:
                    new_hash = self._hash_state(s2, cur_my2, cur_opp2)
                    child = table.get(new_hash)
                    if child is None:
                        child_moves, child_switches = self._heuristic_evals_for_priors(s2, cur_my2, cur_opp2, mi)
                        child_priors, _ = self._build_priors(child_moves, child_switches, branching_cap=branching_cap)
                        if self._neural and child_priors:
                            try:
                                cand = list(child_priors.keys())
                                nn_pol, _ = self._neural.policy_value_for_candidates(s2, cand)
                                s_h = sum(child_priors.values()) or 1.0
                                s_n = sum(nn_pol.get(k, 0.0) for k in cand) or 1.0
                                for k in cand:
                                    ph = child_priors.get(k, 0.0) / s_h
                                    pn = nn_pol.get(k, 0.0) / s_n
                                    child_priors[k] = float(self._neural_policy_blend * ph + (1.0 - self._neural_policy_blend) * pn)
                            except Exception:
                                pass
                        child = self._Node(child_priors)
                        table[new_hash] = child
                    node.children[best_ak] = child
                node = child; s = s2; cur_my, cur_opp = cur_my2, cur_opp2
                terminal = term; depth += 1
                if terminal or depth >= max_depth:
                    break
            if not terminal and depth < max_depth:
                # Blend heuristic leaf eval with neural value if available
                leaf_value = self._leaf_eval(s, cur_my, cur_opp, mi)
                nn_leaf = None
                if self._neural:
                    try:
                        nn_leaf = self._neural.value_only(s)  # [-1,1]
                        leaf_value = float(self._neural_value_blend * leaf_value + (1.0 - self._neural_value_blend) * nn_leaf)
                    except Exception:
                        nn_leaf = None
                # Decay leaf evaluation based on depth for more accurate estimates
                depth_discount = 0.95 ** depth
                total_reward += leaf_value * depth_discount
            for nd, ak in path:
                nd.N[ak] = nd.N.get(ak, 0) + 1
                nd.W[ak] = nd.W.get(ak, 0.0) + total_reward
                nd.Q[ak] = nd.W[ak] / max(1, nd.N[ak])

        if not root.expanded:
            if priors:
                ak = max(priors.items(), key=lambda kv: kv[1])[0]
                kind, ident = ak.split(':', 1)
                dbg = {"root": {"priors": priors, "meta": meta, "expanded": root.expanded, "iters": iters}}
                return kind, (ident if kind == 'move' else None), (ident if kind == 'switch' else None), dbg
            return 'move', None, None, {"root": {"priors": {}, "iters": iters}}
            
        # Select best action based on visit count and value
        # Use robust selection: pick most visited if enough simulations, otherwise best value
        if iters >= sim_budget * 0.7:  # If we completed most of our budget
            best = max(root.expanded, key=lambda ak: root.N.get(ak, 0))
        else:
            best = max(root.expanded, key=lambda ak: (root.N.get(ak, 0), root.Q.get(ak, 0.0)))
        k, ident = best.split(':', 1)
        dbg = {
            'root': {
                'expanded': list(root.expanded),
                'N': {ak: int(root.N.get(ak, 0)) for ak in root.expanded},
                'Q': {ak: float(root.Q.get(ak, 0.0)) for ak in root.expanded},
                'P': {ak: float(root.P.get(ak, 0.0)) for ak in root.expanded},
                'iters': iters,
                # Diagnostics: show heuristic vs neural vs blended priors at root if available
                'P_heur': heur_priors_snapshot,
                'P_nn': nn_pol_snapshot or {},
                'blend': {
                    'policy': float(self._neural_policy_blend),
                    'value': float(self._neural_value_blend),
                }
            }
        }
        return k, (ident if k == 'move' else None), (ident if k == 'switch' else None), dbg

    def _leaf_eval(self, state, my_key: str, opp_key: str, mi: MovesInfo) -> float:
        try:
            me = state.team.ours[my_key]
            our_best = 0.0
            our_coverage = 0.0  # Track type coverage
            move_count = 0
            
            for mv in getattr(me, 'moves', []) or []:
                mid = getattr(mv, 'id', None)
                bp = int(getattr(mv, 'base_power', 0) or 0)
                cat = _get_move_category(mv)
                if mid and bp > 0 and cat in {'physical','special'}:
                    ef, dmg_info = _expected_damage_fraction(state, my_key, opp_key, mid, mi)
                    acc_prob = _acc_to_prob(getattr(mv, 'accuracy', None))
                    move_value = ef * acc_prob
                    our_best = max(our_best, move_value)
                    # Add type effectiveness bonus
                    eff_mult = dmg_info.get('effectiveness', 1.0) if isinstance(dmg_info, dict) else 1.0
                    if eff_mult > 1.0:
                        our_coverage += 0.1 * (eff_mult - 1.0)
                    move_count += 1
            
            incoming = self._get_opp_best_damage(state, opp_key, my_key, mi)
            my_hp = _hp_frac(me)
            
            # More nuanced survival calculation
            if my_hp > 0:
                survival_factor = min(1.0, my_hp / max(0.1, incoming)) if incoming > 0 else 1.0
                p_die_if_hit = 1.0 - survival_factor
            else:
                p_die_if_hit = 1.0
                
            # Base offensive/defensive balance
            offensive_value = our_best + our_coverage
            defensive_penalty = incoming * p_die_if_hit * 0.8
            base = offensive_value - defensive_penalty
            
            # Enhanced board position evaluation
            try:
                alive_self = sum(1 for ps in state.team.ours.values() if getattr(ps, 'current_hp', 0) > 0)
                alive_opp = sum(1 for ps in state.team.opponent.values() if getattr(ps, 'current_hp', 0) > 0)
                hp_self = sum(_hp_frac(ps) for ps in state.team.ours.values())
                hp_opp = sum(_hp_frac(ps) for ps in state.team.opponent.values())
                
                # Team composition advantage
                team_advantage = 0.15 * (alive_self - alive_opp) + 0.12 * (hp_self - hp_opp)
                
                # Hazard control
                side_self = getattr(state, 'our_side', {}) or {}
                side_opp = getattr(state, 'opp_side', {}) or {}
                def _haz_score(sd: Dict[str, Any]) -> float:
                    sc = 0.0
                    sc += 0.12 if sd.get('stealth_rock') else 0.0
                    sc += 0.06 * float(sd.get('spikes', 0) or 0)
                    sc += 0.06 * float(sd.get('toxic_spikes', 0) or 0)
                    sc += 0.08 if sd.get('sticky_web') else 0.0
                    return sc
                hazard_advantage = 0.6 * (_haz_score(side_opp) - _haz_score(side_self))
                
                # Field control (trick room, weather, terrain)
                field_bonus = 0.0
                try:
                    field = state.field
                    if getattr(field, 'trick_room', False):
                        # Evaluate if trick room benefits us
                        my_speed = getattr(me, 'stats', {}).get('spe', 0) or 0
                        opp = state.team.opponent.get(opp_key)
                        opp_speed = getattr(opp, 'stats', {}).get('spe', 0) if opp else 0
                        if my_speed < opp_speed:
                            field_bonus += 0.1
                        else:
                            field_bonus -= 0.1
                except Exception:
                    pass
                    
                position = team_advantage + hazard_advantage + field_bonus
            except Exception:
                position = 0.0
                
            return base + position
        except Exception:
            return 0.0

    def _evaluate_status_move(self, move_id: str, state, my_key: str, opp_key: str, mi: MovesInfo) -> float:
        """Evaluate the strategic value of status moves."""
        mid = (move_id or '').lower()
        my_hp = _hp_frac(state.team.ours[my_key])
        opp_hp = _hp_frac(state.team.opponent[opp_key])
        
        # Healing moves - value scales with missing HP and opponent pressure
        HEALS = {"recover": 0.6, "roost": 0.6, "softboiled": 0.6, "rest": 0.8,
                "synthesis": 0.5, "morningsun": 0.5, "slackoff": 0.6}
        if mid in HEALS:
            missing_hp = 1.0 - my_hp
            # Higher value when opponent can't immediately punish
            incoming_threat = self._get_opp_best_damage(state, opp_key, my_key, mi)
            safety_factor = 1.0 if incoming_threat < 0.4 else 0.6
            return HEALS[mid] * missing_hp * 1.8 * safety_factor
            
        # Setup moves - value depends on survivability and offensive potential
        SETUP_MOVES = {
            "swordsdance": 0.8, "nastyplot": 0.8, "dragondance": 0.9, "quiverdance": 0.9,
            "shellsmash": 1.0, "calmmind": 0.7, "bulkup": 0.6, "irondefense": 0.4,
            "amnesia": 0.4, "agility": 0.5, "rockpolish": 0.5
        }
        if mid in SETUP_MOVES:
            # Higher value if we can survive to use the boosts
            incoming = self._get_opp_best_damage(state, opp_key, my_key, mi)
            survival_factor = max(0.1, 1.0 - incoming / max(0.1, my_hp))
            return SETUP_MOVES[mid] * survival_factor
            
        # Hazards - strategic value
        HAZARDS = {"stealthrock": 0.8, "spikes": 0.6, "toxicspikes": 0.5, "stickyweb": 0.6}
        if mid in HAZARDS:
            side = getattr(state, 'opp_side', {}) or {}
            if mid == "stealthrock" and not side.get('stealth_rock'):
                return 0.8
            elif mid == "spikes" and side.get('spikes', 0) < 3:
                return 0.6 - 0.15 * side.get('spikes', 0)
            elif mid == "toxicspikes" and side.get('toxic_spikes', 0) < 2:
                return 0.5 - 0.2 * side.get('toxic_spikes', 0)
            elif mid == "stickyweb" and not side.get('sticky_web'):
                return 0.6
            return 0.0  # Already set up
            
        # Status ailments - value depends on opponent's HP and type
        STATUS_MOVES = {
            "willowisp": 0.7, "thunderwave": 0.6, "toxic": 0.8, "sleeppowder": 0.9,
            "spore": 1.0, "stunspore": 0.5, "hypnosis": 0.7
        }
        if mid in STATUS_MOVES:
            opp_status = getattr(state.team.opponent[opp_key], 'status', '') or ''
            if opp_status:  # Already statused
                return 0.0
            # Value scales with opponent's remaining HP and our ability to capitalize
            value = STATUS_MOVES[mid] * opp_hp
            # Bonus if opponent has recovery items/abilities
            if mid in {"toxic", "willowisp"}:  # Residual damage is great vs healing
                value *= 1.3
            return value
            
        # Utility moves
        UTILITY_MOVES = {
            "substitute": 0.5, "protect": 0.3, "detect": 0.3, "roar": 0.4,
            "whirlwind": 0.4, "haze": 0.3, "reflect": 0.5, "lightscreen": 0.5,
            "auroraveil": 0.7, "defog": 0.4, "rapidspin": 0.4, "taunt": 0.6
        }
        if mid in UTILITY_MOVES:
            return UTILITY_MOVES[mid]
            
        # Default for unknown status moves
        return 0.1

    # Core API
    def choose_action(self, battle: Any) -> ChosenAction:
        # Ensure collector availability if enabled
        self._ensure_training_collector()
        
        # Debug enhanced model status
        battle_id = getattr(battle, 'battle_tag', getattr(battle, 'room_id', 'unknown'))
        logging.info(f"[MCTS DEBUG] choose_action called for {battle_id}")
        logging.info(f"[MCTS DEBUG] enhanced_model present: {self.enhanced_model is not None}")
        logging.info(f"[MCTS DEBUG] use_enhancements: {self.use_enhancements}")
        logging.info(f"[MCTS DEBUG] in_enhanced_call: {getattr(self, '_in_enhanced_call', False)}")
        
        # Use enhanced model if available and enabled
        if self.enhanced_model and self.use_enhancements:
            # Use recursion protection that doesn't interfere with enhanced model operation
            if not hasattr(self, '_in_enhanced_call') or not self._in_enhanced_call:
                try:
                    self._in_enhanced_call = True
                    enhanced_decision = self.enhanced_model.choose_action(battle)
                    self._in_enhanced_call = False
                    logging.debug(f"[MCTS] Enhanced model provided decision: {type(enhanced_decision).__name__}")
                    return enhanced_decision
                except Exception as e:
                    self._in_enhanced_call = False
                    logging.warning(f"[MCTS] Enhanced model failed, falling back to base: {e}")
                    import traceback
                    logging.debug(f"[MCTS] Enhanced model traceback: {traceback.format_exc()}")
            else:
                logging.debug(f"[MCTS] Recursion detected, using base MCTS for enhanced model internal call")
        
        # Original MCTS implementation
        state = get_state(battle)
        mi = MovesInfo(state.format or 9)
        dec_log: Dict[str, Any] = {
            'battle_tag': getattr(battle, 'battle_tag', getattr(battle, 'room_id', None)),
            'turn': getattr(battle, 'turn', None),
            'candidates': [],
            'switches': [],
        }

        # Resolve active keys
        def _active(side: Dict[str, Any]) -> Optional[str]:
            for k, p in side.items():
                if getattr(p, 'is_active', False) and getattr(p, 'current_hp', 0) > 0:
                    return k
            for k, p in side.items():
                if (getattr(p, 'status', '') or '').lower() == 'fnt':
                    continue
                if getattr(p, 'current_hp', 1) > 0:
                    return k
            return None

        my_key = _active(state.team.ours) if getattr(state, 'team', None) else None
        opp_key = _active(state.team.opponent) if getattr(state, 'team', None) else None

        legal_moves: List[Any] = list(getattr(battle, 'available_moves', []) or [])
        legal_switches: List[Any] = list(getattr(battle, 'available_switches', []) or [])
        force_switch = bool(getattr(battle, 'force_switch', False))

        # Synthesize switches if forced but none provided
        if force_switch and not legal_switches and getattr(state, 'team', None):
            class _Stub:
                __slots__ = ('species',)
                def __init__(self, s): self.species = s
            try:
                active_species = None
                if my_key and my_key in state.team.ours:
                    active_species = getattr(state.team.ours[my_key], 'species', None)
                for k, ps in state.team.ours.items():
                    if getattr(ps, 'current_hp', 0) <= 0:
                        continue
                    sp = getattr(ps, 'species', None)
                    if not sp or sp == active_species:
                        continue
                    legal_switches.append(_Stub(sp))
            except Exception:
                pass

        # Fallback map from active species if needed
        if not my_key:
            try:
                active_species = getattr(getattr(battle, 'active_pokemon', None), 'species', None)
                if active_species:
                    for k, p in getattr(state.team, 'ours', {}).items():
                        if getattr(p, 'species', None) == active_species and (getattr(p, 'current_hp', 0) > 0):
                            my_key = k; break
            except Exception:
                pass
        if not opp_key:
            try:
                opp_active_species = getattr(getattr(battle, 'opponent_active_pokemon', None), 'species', None) or getattr(getattr(battle, 'opponent_active_pokemon', None), 'base_species', None)
                if opp_active_species:
                    for k, p in getattr(state.team, 'opponent', {}).items():
                        if getattr(p, 'species', None) == opp_active_species and (getattr(p, 'current_hp', 0) > 0):
                            opp_key = k; break
            except Exception:
                pass

        if not (my_key and opp_key):
            mv = (legal_moves or [None])[0]
            if mv and getattr(mv, 'id', None):
                return ChosenAction(kind='move', move_id=getattr(mv, 'id'), debug=dec_log)
            return ChosenAction(kind='move', move_id=None, debug=dec_log)

        # Build priors via heuristic
        moves_eval: List[Dict[str, Any]] = []
        try:
            my_eff_spe, opp_eff_spe = get_effective_speeds(state, my_key, opp_key)
        except Exception:
            my_eff_spe, opp_eff_spe = (0, 0)
        trick_room_active = bool(getattr(state.field, 'trick_room', False))
        opp_item_known = False
        try:
            opp_item_known = bool((getattr(state.team.opponent[opp_key], 'item', None) or '').strip())
        except Exception:
            pass

        if not force_switch:
            opp_ps = state.team.opponent[opp_key]
            opp_hp_frac = _hp_frac(opp_ps)
            opp_max = int(getattr(opp_ps, 'max_hp', 0) or getattr(opp_ps, 'stats', {}).raw.get('hp', 1) or 1)
            try:
                my_hp_now = _hp_frac(state.team.ours[my_key])
            except Exception:
                my_hp_now = 1.0
                
            # Evaluate all moves including status moves
            for mv in legal_moves:
                mid = getattr(mv, 'id', None) or getattr(mv, 'move_id', None)
                if not mid:
                    continue
                try:
                    fail, why = would_fail(str(mid), my_key, opp_key, state, mi)
                except Exception:
                    fail, why = False, None
                if fail:
                    moves_eval.append({'id': mid, 'name': getattr(mv, 'name', mid), 'score': 0.0, 'why_blocked': str(why or 'would fail')})
                    continue
                    
                # Check if it's a status move
                cat = _get_move_category(mv)
                bp = int(getattr(mv, 'base_power', 0) or 0)
                
                if cat == 'status' or bp <= 0:
                    # Evaluate status moves
                    status_score = self._evaluate_status_move(mid, state, my_key, opp_key, mi)
                    moves_eval.append({
                        'id': mid, 'name': getattr(mv, 'name', mid), 'score': status_score,
                        'expected': 0.0, 'acc': _acc_to_prob(getattr(mv, 'accuracy', 1.0)),
                        'first_prob': 0.5, 'p_ko_if_hit': 0.0, 'opp_counter_ev': 0.0,
                        'effectiveness': 1.0, 'note': 'status'
                    })
                    continue
                    
                # Evaluate damaging moves
                try:
                    exp_frac, dmg = _expected_damage_fraction(state, my_key, opp_key, str(mid), mi)
                except Exception:
                    exp_frac, dmg = 0.0, {}
                acc_p = _acc_to_prob(getattr(mv, 'accuracy', 1.0))
                try:
                    first_prob, _ = predict_order_for_ids(state, my_key, str(mid), opp_key, 'tackle', mi)
                except Exception:
                    try:
                        first_prob = predict_first_prob_speed_only(state, my_key, opp_key)
                    except Exception:
                        first_prob = 0.5
                first_prob = float(first_prob)
                try:
                    mv_base_pri = int(getattr(mv, 'priority', 0) or 0)
                except Exception:
                    mv_base_pri = 0
                if (not trick_room_active) and (mv_base_pri <= 0) and (not opp_item_known) and (my_eff_spe > opp_eff_spe > 0) and ((opp_eff_spe * 3) > (my_eff_spe * 2)):
                    first_prob = min(first_prob, 0.30)
                rolls = (dmg.get('rolls') or [])
                thr_abs = int(round(opp_hp_frac * opp_max))
                ko_rolls = sum(1 for r in rolls if int(r) >= max(1, thr_abs))
                p_ko_if_hit = (ko_rolls / max(1, len(rolls))) if rolls else 0.0
                try:
                    incoming_best = self._get_opp_best_damage(state, opp_key, my_key, mi)
                    # Adjust for opponent recovery - they can afford to take more damage
                    if _has_recovery_item(state.team.opponent.get(opp_key)):
                        incoming_best *= 0.8  # They heal, so effective threat is lower
                except Exception:
                    incoming_best = 0.0
                p_opp_acts = (1 - first_prob) + first_prob * (1 - p_ko_if_hit * acc_p)
                opp_counter_ev = incoming_best * p_opp_acts
                effective_exp_raw = exp_frac * acc_p
                p_die_if_opp_hits = 0.0
                try:
                    if my_hp_now > 0:
                        p_die_if_opp_hits = min(1.0, float(incoming_best) / float(my_hp_now))
                except Exception:
                    p_die_if_opp_hits = 0.0
                p_survive_before_act = first_prob + (1 - first_prob) * (1 - p_die_if_opp_hits)
                effective_exp = effective_exp_raw * max(0.0, min(1.0, p_survive_before_act))
                exposure_penalty = 0.0
                if incoming_best >= 0.7 * max(0.05, my_hp_now) and (acc_p * p_ko_if_hit) < 0.4:
                    exposure_penalty = 0.2 + 0.3 * (incoming_best)
                # Improved scoring with better balance and accuracy consideration  
                ko_bonus = 0.8 * (acc_p * p_ko_if_hit)
                speed_bonus = 0.3 * first_prob
                damage_value = 1.2 * effective_exp
                safety_penalty = 0.8 * opp_counter_ev
                
                # Penalty for missing crucial KOs
                accuracy_risk = 0.0
                if p_ko_if_hit > 0.8 and acc_p < 0.9:  # High KO chance but risky accuracy
                    accuracy_risk = 0.3 * (1.0 - acc_p) * p_ko_if_hit
                    
                # Bonus for reliable moves in endgame
                if my_hp_now < 0.3 and acc_p >= 0.95:  # Endgame reliability bonus
                    damage_value *= 1.2
                    
                score = damage_value + ko_bonus + speed_bonus - safety_penalty - exposure_penalty - accuracy_risk
                # Extract and include type effectiveness multiplier for UI/logging
                try:
                    eff_mult = float(dmg.get('effectiveness', 1.0)) if isinstance(dmg, dict) else 1.0
                except Exception:
                    eff_mult = 1.0
                moves_eval.append({'id': mid, 'name': getattr(mv, 'name', mid), 'score': float(score), 'expected': float(exp_frac), 'acc': float(acc_p), 'first_prob': float(first_prob), 'p_ko_if_hit': float(p_ko_if_hit), 'opp_counter_ev': float(opp_counter_ev), 'effectiveness': float(eff_mult)})
            moves_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        switches_eval: List[Dict[str, Any]] = []
        if my_key and opp_key:
            active_species = ''
            try:
                active_species = str(getattr(state.team.ours.get(my_key), 'species', '') or '').lower()
            except Exception:
                pass
            for sw in legal_switches:
                try:
                    species = str(getattr(sw, 'species', '') or '')
                    if species.lower() == active_species:
                        continue
                    cand_key = None
                    for k, ps in state.team.ours.items():
                        if str(getattr(ps, 'species', '')).lower() == species.lower():
                            cand_key = k; break
                    if not cand_key:
                        continue
                    cand_hp = _hp_frac(state.team.ours[cand_key])
                    haz_frac = 0.0
                    try:
                        haz = apply_switch_in_effects(state, cand_key, 'ally', mi, mutate=False)
                        haz_frac = float(haz.get('fraction_lost', 0.0) or 0.0)
                    except Exception:
                        haz_frac = 0.0
                    try:
                        incoming = self._get_opp_best_damage(state, opp_key, cand_key, mi)
                    except Exception:
                        incoming = 0.0
                    incoming_total = float(haz_frac) + float(incoming)
                    outgoing = 0.0
                    try:
                        for mv_obj in getattr(state.team.ours[cand_key], 'moves', []) or []:
                            mid2 = getattr(mv_obj, 'id', None)
                            if not mid2:
                                continue
                            bp2 = int(getattr(mv_obj, 'base_power', 0) or getattr(mv_obj, 'basePower', 0) or 0)
                            if _get_move_category(mv_obj) == 'status' or bp2 <= 0:
                                continue
                            frac2, _ = _expected_damage_fraction(state, cand_key, opp_key, mid2, mi)
                            outgoing = max(outgoing, frac2 * _acc_to_prob(getattr(mv_obj, 'accuracy', None)))
                        if outgoing <= 0:
                            try:
                                cand_types = [t for t in (getattr(state.team.ours[cand_key], 'types', []) or []) if t]
                                for t in cand_types:
                                    for mid_guess in _COMMON_STAB.get(str(t).lower(), [])[:2]:
                                        try:
                                            frac2, _ = _expected_damage_fraction(state, cand_key, opp_key, mid_guess, mi)
                                            raw = mi.raw(mid_guess) or {}
                                            outgoing = max(outgoing, frac2 * _acc_to_prob(raw.get('accuracy')))
                                        except Exception:
                                            continue
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Apply improved switch evaluation (same as heuristic version)
                    # Defensive value - resist opponent's attacks
                    resistance_bonus = 0.0
                    if incoming < 0.2:  # Resists well
                        resistance_bonus = 0.8
                    elif incoming < 0.4:  # Decent resistance
                        resistance_bonus = 0.5
                    elif incoming < 0.6:  # Some resistance
                        resistance_bonus = 0.3
                    
                    # Offensive potential of the switch-in
                    offensive_value = min(1.2, outgoing * 1.5)  # Scale up and cap offensive value
                    
                    # Health bonus - prefer healthy Pokemon
                    health_bonus = cand_hp * 0.4
                    
                    # Hazard penalty (much reduced)
                    hazard_penalty = haz_frac * 0.6  # Reduced from 1.2
                    
                    # Emergency switch bonus - if current Pokemon is in danger
                    emergency_bonus = 0.0
                    try:
                        # Get current Pokemon's HP and threat level
                        current_pokemon = state.team.ours.get(my_key)
                        if current_pokemon:
                            current_hp = _hp_frac(current_pokemon)
                            current_threat = self._get_opp_best_damage(state, opp_key, my_key, mi)
                            
                            # If current Pokemon is in serious danger, boost switch value
                            if current_hp < 0.4 and current_threat > 0.6:
                                emergency_bonus = 0.6
                            elif current_hp < 0.6 and current_threat > 0.8:
                                emergency_bonus = 0.4
                    except Exception:
                        pass
                    
                    # Strategic switch bonus - type advantage
                    type_advantage_bonus = 0.0
                    if outgoing > 0.8:  # Can threaten opponent significantly
                        type_advantage_bonus = 0.3
                    
                    # Final score calculation
                    score = (offensive_value + resistance_bonus + health_bonus + 
                            emergency_bonus + type_advantage_bonus - hazard_penalty)
                    switches_eval.append({'key': cand_key, 'species': species, 'score': float(score), 'outgoing_frac': float(outgoing), 'incoming_on_switch': float(incoming_total), 'hazards_frac': float(haz_frac), 'hp_fraction': float(cand_hp)})
                except Exception:
                    continue
            switches_eval.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        dec_log['candidates'] = moves_eval
        dec_log['switches'] = switches_eval

        # Threat-aware immediate bailout: if we're likely to get chunked and can't threaten back, prefer best switch
        if not force_switch:
            try:
                my_hp_now = _hp_frac(state.team.ours[my_key])
                incoming_now = self._get_opp_best_damage(state, opp_key, my_key, mi)
                incoming_rel_now = min(1.0, incoming_now / max(1e-9, my_hp_now)) if my_hp_now > 0 else 1.0
                can_threaten = False
                for m in moves_eval:
                    if (m.get('acc', 0.0) or 0.0) * (m.get('p_ko_if_hit', 0.0) or 0.0) >= 0.5:
                        can_threaten = True; break
                if incoming_rel_now >= 0.6 and not can_threaten and switches_eval:
                    best_sw = switches_eval[0]
                    if best_sw.get('score', -1.0) > -0.5:  # avoid catastrophically bad switches
                        dec_log['picked'] = {'kind': 'switch', **best_sw, 'why': 'threat-bailout'}
                        return ChosenAction(kind='switch', switch_species=best_sw.get('species'), debug=dec_log)
            except Exception:
                pass

        # Run MCTS
        picked_kind, picked_move, picked_switch_key, mcts_dbg = self._mcts_plan(state, my_key, opp_key, moves_eval, switches_eval, force_switch, mi)
        dec_log['mcts'] = mcts_dbg

        # LLM Enhancement
        if self._llm_enabled and self.llm_integration:
            try:
                # Create battle context for LLM
                context = self._create_battle_context(battle, state, my_key, opp_key)
                
                # Check if this is a critical situation where LLM should override
                is_critical = self._should_llm_override(context, battle)
                
                # Log LLM decision process
                logging.getLogger('ThinkVerbose').info(f"[LLM] Turn {context.turn} - LLM Enhancement ACTIVE | Critical: {is_critical}")
                try:
                    logging.getLogger('ThinkVerbose').info(
                        f"LLM CHECK - Turn {context.turn} | Critical: {is_critical} | LLM Integration: Available")
                except Exception:
                    pass

                # Always use LLM when enabled (not just in critical situations)
                if True:  # Changed from: if is_critical or getattr(self, '_verbose', False):
                    # Get MCTS move scores
                    mcts_scores = self._extract_mcts_scores(moves_eval, switches_eval, picked_kind, picked_move, picked_switch_key)
                    
                    # Get LLM enhanced decision with verbose details
                    enhanced_scores, decision_details = self.llm_integration.enhance_move_evaluation(mcts_scores, context)
                    strategic_context = self.llm_integration.get_strategic_context(context)
                    
                    # Log LLM decision
                    logging.getLogger('ThinkVerbose').info(f"[LLM] Strategic Analysis: {decision_details.get('strategic_advice', 'No advice')}")
                    
                    # Log verbose LLM reasoning if available
                    if getattr(self, '_verbose', False):
                        self._log_llm_decision_details(decision_details, battle.battle_tag if hasattr(battle, 'battle_tag') else 'unknown')
                    
                    # Find best LLM-enhanced choice
                    if enhanced_scores:
                        best_choice = max(enhanced_scores.items(), key=lambda x: x[1])
                        best_move_id, best_score = best_choice
                        
                        # Log the LLM's recommendation
                        current_key = picked_move or f"switch_{picked_switch_key}"
                        current_score = mcts_scores.get(current_key, 0.0)
                        logging.getLogger('ThinkVerbose').info(f"[LLM] MCTS chose: {current_key} (score: {current_score:.3f})")
                        logging.getLogger('ThinkVerbose').info(f"[LLM] LLM recommends: {best_move_id} (score: {best_score:.3f})")
                        
                        # Always consider LLM override when enabled (not just in critical situations)
                        override_threshold = 0.1 if is_critical else 0.15  # Lower threshold for critical situations
                        if abs(best_score - current_score) > override_threshold:
                            logging.getLogger('ThinkVerbose').info(f"[LLM] -> OVERRIDE! Score diff: {abs(best_score - current_score):.3f} > {override_threshold}")
                            old_decision = f"{picked_kind}:{picked_move or picked_switch_key}"
                            
                            if best_move_id.startswith('switch_'):
                                picked_kind = 'switch'
                                picked_switch_key = best_move_id.replace('switch_', '')
                                picked_move = None
                                logging.getLogger('ThinkVerbose').info(f"[LLM] -> PICK SWITCH {picked_switch_key}")
                                dec_log['llm_override'] = {
                                    'from': old_decision,
                                    'to': f"switch:{picked_switch_key}",
                                    'reason': decision_details.get('strategic_advice', 'LLM strategic override'),
                                    'decision_details': decision_details
                                }
                            else:
                                picked_kind = 'move'
                                picked_move = best_move_id
                                picked_switch_key = None
                                logging.getLogger('ThinkVerbose').info(f"[LLM] -> PICK MOVE {picked_move}")
                                dec_log['llm_override'] = {
                                    'from': old_decision,
                                    'to': f"move:{picked_move}",
                                    'reason': decision_details.get('strategic_advice', 'LLM strategic override'),
                                    'decision_details': decision_details
                                }
                        else:
                            logging.getLogger('ThinkVerbose').info(f"[LLM] -> NO OVERRIDE (score diff: {abs(best_score - current_score):.3f} <= {override_threshold})")

                    dec_log['llm_context'] = strategic_context

            except Exception as e:
                logging.getLogger('ThinkVerbose').error(f"[LLM] ERROR during LLM processing: {e}")
                dec_log['llm_error'] = str(e)
        else:
            # Debug: Why LLM is not being used
            if not self._llm_enabled:
                logging.getLogger('ThinkVerbose').info("[LLM] NOT USED: LLM Enhancement disabled in UI")
            elif not self.llm_integration:
                logging.getLogger('ThinkVerbose').warning("[LLM] NOT USED: No LLM integration available (missing API key?)")

        if force_switch and picked_kind == 'switch' and picked_switch_key:
            key = picked_switch_key
            sp = None
            try:
                sp = getattr(state.team.ours.get(key), 'species', None) or key
            except Exception:
                sp = key
            dec_log['picked'] = {'kind': 'switch', 'species': sp, 'key': key}
            decision = ChosenAction(kind='switch', switch_species=sp, debug=dec_log)
        elif picked_kind == 'move' and picked_move:
            mid = picked_move
            move_data = next((m for m in moves_eval if m.get('id') == mid), {})
            dec_log['picked'] = {'kind': 'move', 'id': mid, **move_data}
            
            # Handle tera moves from MCTS
            if mid and str(mid).startswith('tera:'):
                actual_move_id = str(mid)[5:]  # Remove 'tera:' prefix
                decision = ChosenAction(
                    kind='move', 
                    move_id=actual_move_id, 
                    is_tera=True,
                    tera_type=move_data.get('tera_type'),
                    debug=dec_log
                )
            else:
                decision = ChosenAction(kind='move', move_id=mid, debug=dec_log)
        elif picked_kind == 'switch' and picked_switch_key:
            key = picked_switch_key
            sp = None
            try:
                sp = getattr(state.team.ours.get(key), 'species', None) or key
            except Exception:
                sp = key
            dec_log['picked'] = {'kind': 'switch', 'species': sp, 'key': key}
            decision = ChosenAction(kind='switch', switch_species=sp, debug=dec_log)
        else:
            # Fallback to heuristic best
            best_move = moves_eval[0] if moves_eval else None
            best_switch = switches_eval[0] if switches_eval else None
            if best_move and not force_switch:
                dec_log['picked'] = {'kind': 'move', **best_move}
                
                # Handle tera moves
                move_id = best_move.get('id')
                if move_id and str(move_id).startswith('tera:'):
                    # Extract the actual move ID from tera:moveid format
                    actual_move_id = str(move_id)[5:]  # Remove 'tera:' prefix
                    decision = ChosenAction(
                        kind='move', 
                        move_id=actual_move_id, 
                        is_tera=True,
                        tera_type=best_move.get('tera_type'),
                        debug=dec_log
                    )
                else:
                    decision = ChosenAction(kind='move', move_id=move_id, debug=dec_log)
            elif best_switch:
                dec_log['picked'] = {'kind': 'switch', **best_switch}
                decision = ChosenAction(kind='switch', switch_species=best_switch.get('species'), debug=dec_log)
            else:
                mv = (legal_moves or [None])[0]
                if mv and getattr(mv, 'id', None):
                    decision = ChosenAction(kind='move', move_id=getattr(mv, 'id'), debug=dec_log)
                else:
                    decision = ChosenAction(kind='move', move_id=None, debug=dec_log)

        if getattr(self, '_verbose', False):
            try:
                turn = getattr(battle, 'turn', None)
                tag = getattr(battle, 'battle_tag', getattr(battle, 'room_id', ''))
                header = f"[MCTS THINK][Turn {turn}][{tag}] Sims={self.simulations} c_puct={self.c_puct} Rollout={self.rollout_depth} T={self.time_limit}"
                lines = ["", header]
                # Quick state/threat summary
                try:
                    me_ps = state.team.ours.get(my_key)
                    opp_ps = state.team.opponent.get(opp_key)
                    my_hp_now = _hp_frac(me_ps)
                    opp_hp_now = _hp_frac(opp_ps)
                    incoming_now = self._get_opp_best_damage(state, opp_key, my_key, mi)
                    incoming_rel_now = min(1.0, incoming_now / max(1e-9, my_hp_now)) if my_hp_now > 0 else 1.0
                    lines.append(" Summary: MyHP={:.2f} OppHP={:.2f} Incoming~={:.3f} IncomingRel~={:.2f} Switches={} ForceSwitch={}".format(
                        float(my_hp_now), float(opp_hp_now), float(incoming_now), float(incoming_rel_now),
                        len(legal_switches or []), bool(force_switch)))
                except Exception:
                    pass
                if moves_eval:
                    lines.append(" Moves (priors sorted):")
                    for mv in moves_eval[:8]:
                        lines.append((
                            "  - {name:<18} id={id:<12} S={s:+.3f} Exp={exp:.3f} Acc={acc:.2f} Eff={eff:.2f} First={first:.2f} KOIfHit={koi:.3f} OppEV={opp:.3f}"
                        ).format(
                            name=str(mv.get('name', ''))[:18], id=str(mv.get('id', ''))[:12],
                            s=float(mv.get('score', 0.0)), exp=float(mv.get('expected', 0.0)),
                            acc=float(mv.get('acc', 0.0)), eff=float(mv.get('effectiveness', 1.0)), first=float(mv.get('first_prob', 0.0)),
                            koi=float(mv.get('p_ko_if_hit', 0.0)), opp=float(mv.get('opp_counter_ev', 0.0))
                        ))
                else:
                    lines.append(" No move candidates")
                if switches_eval and len(switches_eval) > 0:
                    lines.append(" Switch candidates:")
                    for sw in switches_eval[:6]:
                        lines.append((
                            "  - {sp:<18} S={s:+.3f} Out={out:.3f} In={inn:.3f} HP={hp:.2f} Haz={haz:.3f}"
                        ).format(
                            sp=str(sw.get('species', ''))[:18], s=float(sw.get('score', 0.0)),
                            out=float(sw.get('outgoing_frac', 0.0)), inn=float(sw.get('incoming_on_switch', 0.0)),
                            hp=float(sw.get('hp_fraction', 0.0)), haz=float(sw.get('hazards_frac', 0.0))
                        ))
                else:
                    # Clarify whether none were legal vs just low-scoring
                    if not (legal_switches or []):
                        lines.append(" No switch candidates (no legal switches)")
                    else:
                        lines.append(" No switch candidates")
                rt = dec_log.get('mcts', {}).get('root', {}) if isinstance(dec_log.get('mcts'), dict) else {}
                if rt:
                    lines.append(" MCTS root stats:")
                    ex = rt.get('expanded', [])
                    Ns = rt.get('N', {})
                    Qs = rt.get('Q', {})
                    Ps = rt.get('P', {})
                    for ak in ex:
                        lines.append(f"  - {ak} N={Ns.get(ak)} Q={Qs.get(ak):+.3f} P={Ps.get(ak):.3f}")
                pk = decision.debug.get('picked') if isinstance(decision.debug, dict) else {}
                if pk:
                    if pk.get('kind') == 'move':
                        lines.append(f" -> PICK MOVE {pk.get('id')}")
                    else:
                        lines.append(f" -> PICK SWITCH {pk.get('species')}")
                logger = logging.getLogger('ThinkVerbose')
                for ln in lines:
                    print(ln, flush=True)
                    try:
                        logger.info(ln)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Collect training data
        logging.info(f"[MCTS] Training collector available: {self._training_collector is not None}")
        if self._training_collector:
            try:
                # Extract action scores for training
                action_scores = {}
                logging.info(f"[MCTS] Debug - dec_log keys: {list(dec_log.keys())}")
                if 'mcts' in dec_log:
                    mcts_info = dec_log['mcts']
                    logging.info(f"[MCTS] Debug - mcts_info keys: {list(mcts_info.keys())}")
                    # Extract Q-values from MCTS tree search results
                    if 'root' in mcts_info and 'Q' in mcts_info['root']:
                        action_scores.update(mcts_info['root']['Q'])
                        logging.info(f"[MCTS] Debug - extracted {len(action_scores)} Q-values")
                    else:
                        logging.warning(f"[MCTS] Debug - missing root/Q in mcts_info: root={('root' in mcts_info)}, Q={('Q' in mcts_info.get('root', {}))}")
                else:
                    logging.warning(f"[MCTS] Debug - no 'mcts' key in dec_log")
                
                # Extract LLM scores if available
                llm_scores = {}
                if 'llm_context' in dec_log or 'llm_override' in dec_log:
                    # LLM was used - extract enhanced scores
                    llm_scores = action_scores.copy()  # Simplified for now
                
                # Determine chosen action
                chosen_action = "unknown"
                if decision and hasattr(decision, 'kind'):
                    if decision.kind == 'move':
                        if decision.is_tera and decision.tera_type:
                            chosen_action = f"tera:{decision.move_id}:{decision.tera_type}"
                        else:
                            chosen_action = f"move:{decision.move_id}"
                    elif decision.kind == 'switch':
                        chosen_action = f"switch:{decision.switch_species}"
                
                # Check if this was a critical position
                is_critical = 'llm_override' in dec_log
                
                # Collect the position
                logging.info(f"[MCTS] Calling collect_position with {len(action_scores)} action scores")
                self._training_collector.collect_position(
                    battle=battle,
                    mcts_scores=action_scores,
                    chosen_action=chosen_action,
                    llm_scores=llm_scores if llm_scores else None,
                    neural_policy_scores=None,  # Base MCTS doesn't have neural scores
                    neural_value_prediction=None,  # Base MCTS doesn't have neural values
                    is_critical=is_critical
                )
                
            except Exception as e:
                logging.debug(f"[MCTS] Training data collection failed: {e}")
        
        return decision

    def _create_battle_context(self, battle, state, my_key, opp_key):
        """Create battle context for LLM analysis."""
        from Models.llm_integration import BattleContext
        
        # Extract active Pokemon info
        my_active = {}
        if my_key and state.team.ours.get(my_key):
            ps = state.team.ours[my_key]
            my_active = {
                "species": getattr(ps, 'species', ''),
                "hp_percent": _hp_frac(ps) * 100,
                "status": str(getattr(ps, 'status', '') or ''),
                "types": getattr(ps, 'types', []),
            }

        opp_active = {}
        if opp_key and state.team.opponent.get(opp_key):
            ps = state.team.opponent[opp_key]
            opp_active = {
                "species": getattr(ps, 'species', ''),
                "hp_percent": _hp_frac(ps) * 100,
                "status": str(getattr(ps, 'status', '') or ''),
                "types": getattr(ps, 'types', []),
            }

        # Extract team info
        my_team = []
        for pokemon in state.team.ours.values():
            if _hp_frac(pokemon) > 0:
                my_team.append({
                    "species": getattr(pokemon, 'species', ''),
                    "hp_percent": _hp_frac(pokemon) * 100,
                    "status": str(getattr(pokemon, 'status', '') or ''),
                    "types": getattr(pokemon, 'types', [])
                })

        opp_team = []
        for pokemon in state.team.opponent.values():
            if _hp_frac(pokemon) > 0:
                opp_team.append({
                    "species": getattr(pokemon, 'species', ''),
                    "hp_percent": _hp_frac(pokemon) * 100,
                    "status": str(getattr(pokemon, 'status', '') or ''),
                    "types": getattr(pokemon, 'types', [])
                })

        # Extract available moves
        available_moves = []
        for move in getattr(battle, 'available_moves', []):
            available_moves.append({
                "name": getattr(move, 'id', ''),
                "type": str(getattr(move, 'type', '')),
                "category": str(getattr(move, 'category', '')),
                "power": getattr(move, 'base_power', 0),
            })

        # Extract available switches  
        available_switches = []
        for pokemon in getattr(battle, 'available_switches', []):
            available_switches.append({
                "species": getattr(pokemon, 'species', ''),
                "hp_percent": getattr(pokemon, 'current_hp_fraction', 1.0) * 100,
                "types": [str(t) for t in getattr(pokemon, 'types', [])]
            })

        return BattleContext(
            turn=getattr(battle, 'turn', 1),
            my_active=my_active,
            opp_active=opp_active,
            my_team=my_team,
            opp_team=opp_team,
            field_conditions={},  # Could be expanded
            available_moves=available_moves,
            available_switches=available_switches,
            battle_history=[]  # Could be expanded
        )

    def _should_llm_override(self, context, battle):
        """Determine if this is a critical situation requiring LLM override."""
        
        # Get battle history to detect setup threats
        battle_history = getattr(battle, 'battle_history', [])
        recent_actions = ' '.join(battle_history[-5:]) if battle_history else ''
        
        # Critical situations
        critical_checks = [
            # Early game (turns 1-5) - LLM should be more aggressive
            context.turn <= 5,
            
            # Setup threat detection from battle history
            any(setup in recent_actions.lower() for setup in ["bulk up", "dragon dance", "shell smash", "calm mind", "sword dance", "quiver dance"]),
            
            # Direct setup threat detection
            "Shell Smash" in str(getattr(battle, 'opponent_active_pokemon', '')),
            "Dragon Dance" in str(getattr(battle, 'opponent_active_pokemon', '')),
            "Bulk Up" in str(getattr(battle, 'opponent_active_pokemon', '')),
            
            # Opponent stat boosts detected
            hasattr(battle, 'opponent_active_pokemon') and 
            getattr(battle.opponent_active_pokemon, 'boosts', {}) and
            any(boost > 0 for boost in getattr(battle.opponent_active_pokemon, 'boosts', {}).values()),
            
            # Low HP situations
            context.my_active.get("hp_percent", 100) < 30,
            
            # Endgame scenarios
            len([p for p in context.my_team if p.get("hp_percent", 0) > 0]) <= 2,
            
            # Late game
            context.turn >= 15,
            
            # Explosion available against threats
            any("explosion" in move.get("name", "").lower() for move in context.available_moves) and context.turn <= 5
        ]
        
        return any(critical_checks)
    
    def _log_llm_decision_details(self, decision_details: Dict[str, Any], battle_tag: str):
        """Log detailed LLM decision reasoning for analysis."""
        
        if not decision_details or decision_details.get("error"):
            return
            
        # Format the verbose log
        log_lines = [
            f"\n{'='*60}",
            f"LLM DECISION ANALYSIS - Battle: {battle_tag}",
            f"Turn {decision_details.get('turn', '?')} | Critical: {decision_details.get('is_critical', False)}",
            f"{'='*60}",
            f"STRATEGIC ADVICE:",
            f"  {decision_details.get('strategic_advice', 'No advice provided')}",
            f"",
            f"WEIGHTING: {decision_details.get('weight_reason', 'Unknown')}",
            f"  MCTS Weight: {decision_details.get('weights', {}).get('mcts', 0):.1%}",
            f"  LLM Weight: {decision_details.get('weights', {}).get('llm', 0):.1%}",
            f"",
            f"DECISION COMPARISON:",
            f"  MCTS Choice: {decision_details.get('best_mcts_choice', {}).get('move', 'none')} ({decision_details.get('best_mcts_choice', {}).get('score', 0):.3f})",
            f"  LLM Choice: {decision_details.get('best_llm_choice', {}).get('move', 'none')} ({decision_details.get('best_llm_choice', {}).get('score', 0):.3f})",
            f"  Final Choice: {decision_details.get('best_combined_choice', {}).get('move', 'none')} ({decision_details.get('best_combined_choice', {}).get('score', 0):.3f})",
            f"  Decision Changed: {'YES' if decision_details.get('decision_changed', False) else 'NO'}",
            f"",
            f"SCORE BREAKDOWN:"
        ]
        
        # Add detailed score breakdown
        for move, breakdown in decision_details.get('score_breakdown', {}).items():
            mcts_score = breakdown.get('mcts_score', 0)
            llm_score = breakdown.get('llm_score', 0)
            final_score = breakdown.get('weighted_score', 0)
            override_applied = breakdown.get('override_applied', False)
            
            log_lines.append(f"  {move:12} | MCTS: {mcts_score:.3f} | LLM: {llm_score:.3f} | Final: {final_score:.3f}")
            if override_applied:
                log_lines.append(f"               -> OVERRIDE: {breakdown.get('override_reason', 'Unknown override')}")
        
        # Add context information
        log_lines.extend([
            f"",
            f"BATTLE CONTEXT:",
            f"  Opponent HP: {100 - decision_details.get('my_hp', 100):.0f}% | My HP: {decision_details.get('my_hp', 100):.0f}%",
            f"  Opponent Boosts: {decision_details.get('opponent_boosts', {})}",
            f"{'='*60}"
        ])
        
        # Log to UI's ThinkVerbose logger and also write to file
        verbose_log = '\n'.join(log_lines)
        try:
            logging.getLogger('ThinkVerbose').info(verbose_log)
        except Exception:
            logging.info(verbose_log)

        try:
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'llm_decisions.log'), 'a', encoding='utf-8') as f:
                f.write(verbose_log + '\n\n')
        except Exception as e:
            logging.warning(f"Could not write to LLM decisions log: {e}")

    def _extract_mcts_scores(self, moves_eval, switches_eval, picked_kind, picked_move, picked_switch_key):
        """Extract MCTS scores for LLM comparison."""
        
        scores = {}
        
        # Add move scores
        for move_data in moves_eval:
            move_id = move_data.get('id', '')
            if move_id:
                # Higher score for picked move
                base_score = 0.5
                if picked_kind == 'move' and picked_move == move_id:
                    base_score = 0.8
                scores[move_id] = base_score

        # Add switch scores
        for switch_data in switches_eval:
            species = switch_data.get('species', '')
            if species:
                switch_key = f"switch_{species.lower()}"
                base_score = 0.3
                if picked_kind == 'switch' and picked_switch_key and picked_switch_key in species.lower():
                    base_score = 0.8
                scores[switch_key] = base_score
        
        return scores

    def get_opponent_knowledge(self, species: str) -> Dict:
        """Get accumulated knowledge about opponent species."""
        if not hasattr(self, '_opponent_knowledge'):
            return {'moves': set(), 'abilities': set(), 'items': set()}
            
        return self._opponent_knowledge.get(species, {'moves': set(), 'abilities': set(), 'items': set()})
    
    def _get_opp_best_damage(self, state, opp_key: str, target_key: str, mi: MovesInfo) -> float:
        """Get opponent's best damage against target, using learned knowledge."""
        opponent_knowledge = getattr(self, '_opponent_knowledge', {})
        return _opp_best_on_target(state, opp_key, target_key, mi, opponent_knowledge)


# -------------- Poke-env Player wrapper ----------------
try:
    from poke_env.player.player import Player  # type: ignore
except Exception:
    Player = object  # fallback


class MCTSPokeEnvPlayer(Player):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        self.on_think_hook = kwargs.pop('on_think', None)
        engine = kwargs.pop('engine', None)
        battle_format = kwargs.get('battle_format', 'gen9ou')
        self.engine: MCTSModel = engine or MCTSModel(battle_format=battle_format)
        
        # Add extended websocket timeout configuration to prevent disconnections
        if 'ping_interval' not in kwargs:
            kwargs['ping_interval'] = 60.0  # Send keepalive every 60 seconds (was 20s default)
        if 'ping_timeout' not in kwargs:
            kwargs['ping_timeout'] = 120.0  # Wait up to 2 minutes for ping response (was 20s default)
        if 'open_timeout' not in kwargs:
            kwargs['open_timeout'] = 30.0   # Wait up to 30 seconds for initial connection (was 10s default)
        
        # Track active battles for training data collection
        self._active_battles = set()
        # Cache battle states to handle race conditions with quick deinit
        self._battle_state_cache = {}
        try:
            sims = kwargs.pop('simulations', None) or kwargs.pop('engine_depth', None)
            if sims is not None:
                self.engine.set_simulations(int(sims))
        except Exception:
            pass
        try:
            cpuct = kwargs.pop('c_puct', None)
            if cpuct is not None:
                self.engine.set_c_puct(float(cpuct))
        except Exception:
            pass
        self._request_cache = {}
        super().__init__(*args, **kwargs)

    def _build_request_signature(self, battle) -> tuple:
        try:
            turn = getattr(battle, 'turn', None)
            force_switch = bool(getattr(battle, 'force_switch', False))
            moves = [(getattr(m, 'id', None), getattr(m, 'pp', None)) for m in (getattr(battle, 'available_moves', []) or [])]
            switches = [getattr(p, 'species', None) for p in (getattr(battle, 'available_switches', []) or [])]
            active = getattr(getattr(battle, 'active_pokemon', None), 'species', None)
            return (turn, force_switch, active, tuple(moves), tuple(switches))
        except Exception:
            return (None,)
    
    def _handle_battle_end(self, battle, battle_tag: str):
        """Handle battle end for training data collection"""
        try:
            if self.engine._training_collector and battle_tag in self._active_battles:
                # Use cached state if battle object is corrupted due to race condition
                cached_state = self._battle_state_cache.get(battle_tag, {})
                # Determine game result with multiple detection methods
                game_result = 0.0  # Default to draw
                final_score = None
                
                # Method 1: Check battle.won attribute (with cached fallback)
                battle_won = getattr(battle, 'won', None)
                if battle_won is None and cached_state:
                    battle_won = cached_state.get('won')
                
                if battle_won is not None:
                    if battle_won is True:
                        game_result = 1.0  # Win
                    elif battle_won is False:
                        game_result = -1.0  # Loss
                    logging.info(f"[Training] Battle result from won attribute: {game_result} (cached: {battle_won != getattr(battle, 'won', None)})")
                
                # Method 2: FNT-based scoring (count fainted pokemon for accuracy)
                elif hasattr(battle, 'team') and hasattr(battle, 'opponent_team') or cached_state:
                    try:
                        # Count fainted pokemon (use cached data if battle corrupted)
                        if hasattr(battle, 'team') and hasattr(battle, 'opponent_team'):
                            my_fainted = sum(1 for p in battle.team.values() if p.fainted)
                            my_total = len(battle.team)
                            opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
                            opp_total_discovered = len(battle.opponent_team)
                        else:
                            # Use cached data due to race condition
                            my_fainted = cached_state.get('my_fainted', 0)
                            my_total = cached_state.get('my_total', 6)
                            opp_fainted = cached_state.get('opp_fainted', 0)
                            opp_total_discovered = cached_state.get('opp_total', 0)
                            logging.warning(f"[Training] Using cached battle state due to corrupted battle object")
                        
                        my_alive = my_total - my_fainted
                        opp_alive_discovered = opp_total_discovered - opp_fainted
                        
                        # Determine game outcome and correct alive counts
                        if hasattr(battle, 'won') and battle.won is not None:
                            # Use battle.won as ground truth, adjust counts accordingly
                            if battle.won is True:
                                game_result = 1.0
                                # We won - opponent has 0 left, we have at least 1
                                opp_alive = 0
                                my_alive = max(1, my_alive)  # Ensure we show at least 1 survivor
                            elif battle.won is False:
                                game_result = -1.0  
                                # We lost - we have 0 left, opponent has at least 1
                                my_alive = 0
                                opp_alive = max(1, opp_alive_discovered)
                            else:
                                game_result = 0.0  # Draw
                                opp_alive = opp_alive_discovered
                        else:
                            # No battle.won attribute - infer from FNT counts
                            if my_fainted >= 6:  # All our pokemon fainted
                                game_result = -1.0
                                my_alive = 0
                                opp_alive = max(1, opp_alive_discovered)
                            elif opp_fainted >= 6 or (opp_alive_discovered == 0 and opp_total_discovered >= 6):
                                game_result = 1.0  # We won
                                opp_alive = 0
                                my_alive = max(1, my_alive)
                            else:
                                game_result = 0.0  # Ongoing or unclear
                                opp_alive = opp_alive_discovered
                        
                        final_score = f"{my_alive}-{opp_alive}"
                        logging.info(f"[Training] Battle result from FNT analysis: {game_result}, Score: {final_score}")
                        logging.debug(f"[Training] FNT counts: my_fainted={my_fainted}/6, opp_fainted={opp_fainted}/{opp_total_discovered}")
                        
                    except Exception as e:
                        logging.debug(f"[Training] FNT analysis failed: {e}")
                
                # Method 3: Check if battle has rating or score information
                if final_score is None:
                    if hasattr(battle, 'rating'):
                        final_score = str(battle.rating)
                    elif hasattr(battle, 'score'):
                        final_score = str(battle.score)
                
                # Method 4: Try to infer from battle state
                if game_result == 0.0 and hasattr(battle, 'finished') and battle.finished:
                    # Battle finished but no clear result - try inference
                    try:
                        if hasattr(battle, 'force_switch') and not battle.force_switch:
                            # If not force switching and battle finished, likely a win/loss scenario
                            logging.debug(f"[Training] Battle finished with unclear result, using draw")
                    except:
                        pass
                
                # End game collection
                self.engine._training_collector.end_game_collection(game_result, final_score)
                self._active_battles.discard(battle_tag)
                
                # Clean up battle state cache
                self._battle_state_cache.pop(battle_tag, None)
                
                logging.info(f"[Training] Battle ended: {battle_tag}, Result: {game_result}, Score: {final_score}")
                
        except Exception as e:
            logging.error(f"[Training] Battle end handling failed: {e}")
            import traceback
            logging.error(f"[Training] Traceback: {traceback.format_exc()}")

    def choose_move(self, battle):
        tag = getattr(battle, 'battle_tag', getattr(battle, 'room_id', None))
        
        # Track battle start for training data collection
        if tag and tag not in self._active_battles and self.engine._training_collector:
            self._active_battles.add(tag)
            self.engine._training_collector.start_game_collection(tag)
        
        # Cache current battle state to handle race conditions
        if tag and tag in self._active_battles:
            try:
                self._battle_state_cache[tag] = {
                    'my_fainted': sum(1 for p in battle.team.values() if p.fainted) if hasattr(battle, 'team') else 0,
                    'my_total': len(battle.team) if hasattr(battle, 'team') else 6,
                    'opp_fainted': sum(1 for p in battle.opponent_team.values() if p.fainted) if hasattr(battle, 'opponent_team') else 0,
                    'opp_total': len(battle.opponent_team) if hasattr(battle, 'opponent_team') else 0,
                    'won': getattr(battle, 'won', None),
                    'finished': getattr(battle, 'finished', False),
                    'turn': getattr(battle, 'turn', 1)
                }
            except Exception as e:
                logging.debug(f"[Training] Failed to cache battle state: {e}")
            
        # Check for battle end (multiple conditions with better detection)
        battle_ended = False
        
        # Primary indicators of battle end
        if hasattr(battle, 'finished') and battle.finished:
            battle_ended = True
            logging.debug(f"[Training] Battle {tag} ended - finished=True")
        elif hasattr(battle, 'won') and battle.won is not None:
            battle_ended = True
            logging.debug(f"[Training] Battle {tag} ended - won={battle.won}")
        
        # Secondary indicators - team status check
        elif hasattr(battle, 'team') and hasattr(battle, 'opponent_team'):
            try:
                my_alive = sum(1 for p in battle.team.values() if not p.fainted)
                opp_discovered = sum(1 for p in battle.opponent_team.values() if not p.fainted)
                opp_total_discovered = len(battle.opponent_team)
                
                logging.debug(f"[Training] Battle status check - My alive: {my_alive}, Opp discovered alive: {opp_discovered}/{opp_total_discovered}")
                
                # Battle ends when:
                # 1. We have no pokemon left (loss)
                # 2. All discovered opponent pokemon are fainted AND we've seen ALL their team (win)
                # Note: Only end on opponent faint if we've seen their full team (6 pokemon)
                
                if my_alive == 0:
                    battle_ended = True
                    logging.debug(f"[Training] Battle {tag} ended - we have no pokemon left")
                elif opp_discovered == 0 and opp_total_discovered >= 6:
                    # Only declare victory if ALL opponent pokemon are fainted AND we've seen their full team
                    battle_ended = True
                    logging.debug(f"[Training] Battle {tag} ended - opponent full team fainted (seen {opp_total_discovered})")
                    
            except Exception as e:
                logging.debug(f"[Training] Team status check failed: {e}")
        
        # Fallback indicators
        elif not (getattr(battle, 'available_moves', []) or getattr(battle, 'available_switches', [])):
            # No actions available might indicate battle end (but less reliable)
            battle_ended = True
            logging.debug(f"[Training] Battle {tag} ended - no available actions")
        
        # Additional check for turn-based termination (battles shouldn't go on forever)
        elif hasattr(battle, 'turn') and battle.turn > 1000:
            battle_ended = True
            logging.warning(f"[Training] Battle {tag} ended due to turn limit (turn {battle.turn})")
            
        if battle_ended and tag in self._active_battles:
            self._handle_battle_end(battle, tag)
        
        sig = self._build_request_signature(battle)
        cached = self._request_cache.get(tag)
        if cached and cached[0] == sig:
            try:
                return cached[1]
            except Exception:
                pass
        # Learn from opponent before making decision
        self._learn_from_battle_state(battle)
        
        decision = self.engine.choose_action(battle)
        # think hook
        try:
            if self.on_think_hook and isinstance(decision.debug, dict):
                dd = dict(decision.debug)
                dd.setdefault('snapshot', snapshot_battle(battle))
                dd['battle_tag'] = tag; dd['turn'] = getattr(battle, 'turn', None)
                import logging as _lg, json as _json
                _lg.getLogger('Think').info('UI_THINK turn=%s payload=%s', getattr(battle, 'turn', None), _json.dumps(dd, default=str))
                self.on_think_hook(battle, dd)
        except Exception:
            pass
        # Map decision to poke-env order
        try:
            # Handle direct poke-env objects (from standalone neural MCTS)
            if hasattr(decision, 'id') and hasattr(decision, 'type'):
                # Decision is a Move object, return directly
                order = self.create_order(decision)
                self._request_cache[tag] = (sig, order)
                return order
            elif hasattr(decision, 'species'):
                # Decision is a Pokemon object (switch), return directly
                order = self.create_order(decision)
                self._request_cache[tag] = (sig, order)
                return order
            elif str(type(decision).__name__) == 'TeraAction':
                # Decision is a TeraAction, create tera order
                # Find first available move for tera command
                first_move = (getattr(battle, 'available_moves', []) or [None])[0]
                if first_move:
                    order = self._create_tera_order(first_move)
                    self._request_cache[tag] = (sig, order)
                    return order
                else:
                    # Fallback if no moves available
                    order = self.choose_random_move(battle)
                    self._request_cache[tag] = (sig, order)
                    return order
            elif decision.kind == 'move' and decision.move_id:
                for m in (getattr(battle, 'available_moves', []) or []):
                    if str(getattr(m, 'id', '')) == str(decision.move_id):
                        if decision.is_tera:
                            # Create custom tera command
                            order = self._create_tera_order(m, decision.tera_type)
                        else:
                            order = self.create_order(m)
                        self._request_cache[tag] = (sig, order); return order
                # Fallback to first available move
                first_move = (getattr(battle, 'available_moves', []) or [None])[0]
                if first_move:
                    if decision.is_tera:
                        order = self._create_tera_order(first_move, decision.tera_type)
                    else:
                        order = self.create_order(first_move)
                else:
                    order = self.choose_random_move(battle)
                self._request_cache[tag] = (sig, order); return order
            elif decision.kind == 'switch' and decision.switch_species:
                cursp = ''
                try:
                    cursp = (getattr(getattr(battle, 'active_pokemon', None), 'species', None) or '').lower()
                except Exception:
                    pass
                for p in (getattr(battle, 'available_switches', []) or []):
                    sp = (str(getattr(p, 'species', '')).lower())
                    if sp == cursp:
                        continue
                    if sp == str(decision.switch_species).lower():
                        order = self.create_order(p); self._request_cache[tag] = (sig, order); return order
                for p in (getattr(battle, 'available_switches', []) or []):
                    if (str(getattr(p, 'species', '')).lower()) != cursp:
                        order = self.create_order(p); self._request_cache[tag] = (sig, order); return order
                order = self.choose_random_move(battle); self._request_cache[tag] = (sig, order); return order
            order = self.choose_random_move(battle); self._request_cache[tag] = (sig, order); return order
        except Exception:
            order = self.choose_random_move(battle); self._request_cache[tag] = (sig, order); return order
    
    def _create_tera_order(self, move, tera_type=None):
        """Create a terastallization order for Showdown protocol.
        
        According to Showdown protocol, tera commands are:
        move [MOVE] tera
        """
        try:
            if not move:
                return self.choose_random_move(None)
            
            # Use poke-env's terastallization method
            try:
                # Try using poke-env's built-in terastallization
                return self.create_order(move, terastallize=True)
            except TypeError:
                # Fallback: create order with manual tera construction
                move_id = getattr(move, 'id', None)
                if move_id:
                    # Use the regular create_order but modify the message
                    regular_order = self.create_order(move)
                    # Modify the message to add tera
                    if hasattr(regular_order, 'message'):
                        tera_message = regular_order.message.replace(f"move {move_id}", f"move {move_id} tera")
                        # Create new order with tera message
                        from poke_env.player.player import BattleOrder
                        return BattleOrder(tera_message)
                    else:
                        return regular_order
                else:
                    return self.create_order(move)
        except Exception as e:
            # Fallback to regular move if tera command fails
            logging.warning(f"[Tera Order] Failed to create tera order: {e}, using regular move")
            return self.create_order(move) if move else self.choose_random_move(None)
    
    def _learn_from_battle_state(self, battle):
        """Learn opponent information from current battle state."""
        try:
            from Data.battle_runtime import get_state
            state = get_state(battle)
            
            if not state or not getattr(state, 'team', None):
                return
                
            # Learn from opponent Pokemon
            for opp_key, opp_pokemon in state.team.opponent.items():
                if not opp_pokemon:
                    continue
                    
                species = getattr(opp_pokemon, 'species', None)
                if not species:
                    continue
                
                # Learn revealed moves
                revealed_moves = getattr(opp_pokemon, 'moves', [])
                if revealed_moves:
                    self._update_opponent_moves(species, revealed_moves)
                
                # Learn ability if revealed
                ability = getattr(opp_pokemon, 'ability', None)
                if ability and ability != 'unknown':
                    self._update_opponent_ability(species, ability)
                
                # Infer item from effects
                item = getattr(opp_pokemon, 'item', None)
                if item and item not in ['unknown', 'unknown_item', None]:
                    self._update_opponent_item(species, item)
                else:
                    # Try to infer item from battle effects
                    inferred_item = self._infer_opponent_item(opp_pokemon, battle)
                    if inferred_item:
                        self._update_opponent_item(species, inferred_item)
                        
        except Exception as e:
            logging.debug(f"Opponent learning failed: {e}")
    
    def _update_opponent_moves(self, species: str, moves):
        """Update known moves for opponent species."""
        if not hasattr(self, '_opponent_knowledge'):
            self._opponent_knowledge = {}
            
        if species not in self._opponent_knowledge:
            self._opponent_knowledge[species] = {'moves': set(), 'abilities': set(), 'items': set()}
            
        for move in moves:
            move_id = getattr(move, 'id', None) or str(move)
            if move_id:
                self._opponent_knowledge[species]['moves'].add(move_id.lower())
                logging.debug(f"Learned move {move_id} for {species}")
    
    def _update_opponent_ability(self, species: str, ability: str):
        """Update known ability for opponent species."""
        if not hasattr(self, '_opponent_knowledge'):
            self._opponent_knowledge = {}
            
        if species not in self._opponent_knowledge:
            self._opponent_knowledge[species] = {'moves': set(), 'abilities': set(), 'items': set()}
            
        self._opponent_knowledge[species]['abilities'].add(ability.lower())
        logging.debug(f"Learned ability {ability} for {species}")
    
    def _update_opponent_item(self, species: str, item: str):
        """Update known item for opponent species."""
        if not hasattr(self, '_opponent_knowledge'):
            self._opponent_knowledge = {}
            
        if species not in self._opponent_knowledge:
            self._opponent_knowledge[species] = {'moves': set(), 'abilities': set(), 'items': set()}
            
        self._opponent_knowledge[species]['items'].add(item.lower())
        logging.debug(f"Learned item {item} for {species}")
    
    def _infer_opponent_item(self, opp_pokemon, battle) -> Optional[str]:
        """Infer opponent item from battle effects and HP changes."""
        try:
            # Check for Leftovers - healing at end of turn
            current_hp = getattr(opp_pokemon, 'current_hp', 0)
            max_hp = getattr(opp_pokemon, 'max_hp', 1)
            hp_fraction = current_hp / max_hp if max_hp > 0 else 0
            
            # Simple heuristics for common items
            # Note: This would need to track HP changes over time in a real implementation
            if hasattr(opp_pokemon, '_previous_hp'):
                prev_hp = opp_pokemon._previous_hp
                hp_change = current_hp - prev_hp
                
                # Leftovers healing (6.25% per turn)
                if hp_change > 0 and abs(hp_change - max_hp * 0.0625) < max_hp * 0.01:
                    return 'leftovers'
                
                # Black Sludge for Poison types
                if hp_change > 0:
                    types = getattr(opp_pokemon, 'types', [])
                    if any('poison' in str(t).lower() for t in types):
                        return 'blacksludge'
            
            # Store current HP for next comparison
            opp_pokemon._previous_hp = current_hp
            
            # Heavy-Duty Boots inference - no hazard damage
            if hasattr(battle, 'side_conditions'):
                hazards = getattr(battle, 'side_conditions', {})
                if any(hazards.get(h, 0) > 0 for h in ['stealth_rock', 'spikes', 'toxic_spikes']):
                    # If opponent switched in without taking hazard damage, might have boots
                    # This would require tracking switch-ins
                    pass
                    
            return None
            
        except Exception:
            return None
