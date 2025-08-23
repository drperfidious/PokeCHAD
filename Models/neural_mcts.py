# Neural Network enhanced MCTS for Pokemon
# Implements value networks and policy networks to improve MCTS performance

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from pathlib import Path

class PokemonStateEncoder:
    """Encodes Pokemon battle states for neural network input"""
    
    def __init__(self):
        # Pokemon species encoding (you'd populate this from Pokedex)
        self.species_to_id = {}
        self.move_to_id = {}
        self.ability_to_id = {}
        self.item_to_id = {}
        self.type_to_id = {}
        
        # Dimensions
        self.max_species = 1000
        self.max_moves = 1000
        self.max_abilities = 300
        self.max_items = 500
        self.max_types = 18
        
        self._load_encodings()
        self._load_move_database()
    
    def _load_encodings(self):
        """Load encoding dictionaries from data files"""
        try:
            # Try to load from existing data files
            data_dir = Path("Data")
            
            if (data_dir / "species_encoding.json").exists():
                with open(data_dir / "species_encoding.json", 'r') as f:
                    self.species_to_id = json.load(f)
            
            if (data_dir / "move_encoding.json").exists():
                with open(data_dir / "move_encoding.json", 'r') as f:
                    self.move_to_id = json.load(f)
                    
            # Create basic type encoding if none exists
            if not self.type_to_id:
                types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", 
                        "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", 
                        "Dragon", "Dark", "Steel", "Fairy"]
                self.type_to_id = {t: i for i, t in enumerate(types)}
                
        except Exception as e:
            logging.warning(f"Failed to load encodings: {e}")

    def _load_move_database(self):
        """Load move properties database for enhanced encoding"""
        try:
            import json
            from pathlib import Path
            import os
            
            # Get the absolute path to the project root
            current_dir = Path(__file__).parent.parent  # Go up from Models/ to project root
            moves_path = current_dir / "tools/Data/showdown/moves.json"
            
            if moves_path.exists():
                with open(moves_path, 'r') as f:
                    moves_data = json.load(f)
                
                self.move_properties = {}
                for move_id, move_info in moves_data.items():
                    # Extract key properties for neural encoding
                    self.move_properties[move_id] = {
                        'base_power': move_info.get('basePower', 0),
                        'accuracy': move_info.get('accuracy', 100) if isinstance(move_info.get('accuracy'), int) else 100,
                        'priority': move_info.get('priority', 0),
                        'type': move_info.get('type', 'Normal'),
                        'category': move_info.get('category', 'Status')  # Physical/Special/Status
                    }
                
                logging.info(f"[Neural Encoder] Loaded properties for {len(self.move_properties)} moves")
            else:
                logging.warning(f"[Neural Encoder] Move database not found at {moves_path}")
                self.move_properties = {}
                
        except Exception as e:
            logging.warning(f"[Neural Encoder] Failed to load move database: {e}")
            self.move_properties = {}

    def _type_title(self, ptype: Any) -> str:
        """Best-effort normalization of type names from mixed sources.

        Accepts enum-like objects with .name, plain strings (e.g., 'water'), and
        noisy strings such as 'WATER (pokemon type) object'. Returns a canonical
        capitalized name like 'Water', or '' if unrecognized.
        """
        try:
            name = getattr(ptype, 'name', None)
            if name:
                return str(name).strip().capitalize()
            s = str(ptype or '').strip()
            if not s:
                return ''
            # Strip noise like '(pokemon type) object' and take first token
            for sep in ['(', ' ']:
                if sep in s:
                    s = s.split(sep, 1)[0]
                    break
            return s[:1].upper() + s[1:].lower()
        except Exception:
            return ''
    
    def encode_battle_state(self, battle: Any) -> torch.Tensor:
        """Encode a battle state into a tensor suitable for neural networks"""
        
        try:
            # Use the proven snapshot method to extract battle data
            from Data.poke_env_battle_environment import snapshot
            battle_snapshot = snapshot(battle)
            
            # Feature dimensions (Phase 4 Complete)  
            team_features = 6 * 25  # 6 Pokemon * 25 features each
            field_features = 183    # Field + All Phase enhancements (50->183)
            game_features = 23      # Turn, timer, etc. (adjusted for 830 total)
            
            total_features = team_features * 2 + field_features + game_features  # = 830
            state_vector = torch.zeros(total_features)
            
            offset = 0
            
            # Encode my team using snapshot data
            offset = self._encode_team_from_snapshot(battle_snapshot["my_team"], state_vector, offset)
            
            # Encode opponent team using snapshot data
            offset = self._encode_team_from_snapshot(battle_snapshot["opp_team"], state_vector, offset)
            
            # Encode field conditions using snapshot data
            offset = self._encode_field_from_snapshot(battle_snapshot, state_vector, offset)
            
            # Encode game state using snapshot data
            self._encode_game_state_from_snapshot(battle_snapshot, state_vector, offset)
            
            # Count truly meaningful features (exclude normalized neutral values like 0.5 for boosts)
            meaningful_features = torch.sum((state_vector != 0.0) & (state_vector != 0.5)).item()
            total_nonzero = torch.count_nonzero(state_vector).item()
            logging.debug(f"[Neural Encoder] Successfully encoded battle state with {meaningful_features} meaningful features ({total_nonzero} total non-zero)")
            return state_vector
            
        except Exception as e:
            logging.error(f"[Neural Encoder] Failed to encode battle state: {e}")
            raise e
    
    def encode_battle_state_from_snapshot(self, battle_snapshot: Dict[str, Any]) -> torch.Tensor:
        """Encode a battle state directly from a snapshot dict"""
        
        try:
            # Feature dimensions (Phase 4 Complete)  
            team_features = 6 * 25  # 6 Pokemon * 25 features each
            field_features = 183    # Field + All Phase enhancements (50->183)
            game_features = 23      # Turn, timer, etc. (adjusted for 830 total)
            
            total_features = team_features * 2 + field_features + game_features  # = 830
            state_vector = torch.zeros(total_features)
            
            offset = 0
            
            # Encode my team using snapshot data
            offset = self._encode_team_from_snapshot(battle_snapshot["my_team"], state_vector, offset)
            
            # Encode opponent team using snapshot data
            offset = self._encode_team_from_snapshot(battle_snapshot["opp_team"], state_vector, offset)
            
            # Encode field conditions using snapshot data
            offset = self._encode_field_from_snapshot(battle_snapshot, state_vector, offset)
            
            # Encode game state using snapshot data
            self._encode_game_state_from_snapshot(battle_snapshot, state_vector, offset)
            
            # Count truly meaningful features (exclude normalized neutral values like 0.5 for boosts)
            meaningful_features = torch.sum((state_vector != 0.0) & (state_vector != 0.5)).item()
            total_nonzero = torch.count_nonzero(state_vector).item()
            logging.debug(f"[Neural Encoder] Successfully encoded snapshot with {meaningful_features} meaningful features ({total_nonzero} total non-zero)")
            return state_vector
            
        except Exception as e:
            logging.error(f"[Neural Encoder] Failed to encode snapshot: {e}")
            raise e
    
    def _encode_team_from_snapshot(self, team_data: Dict[str, Any], state_vector: torch.Tensor, offset: int) -> int:
        """Encode team information from snapshot data"""
        
        pokemon_list = list(team_data.items())[:6]  # Get up to 6 Pokemon
        
        for i in range(6):  # Always encode 6 slots
            base_offset = offset + i * 25
            
            if i < len(pokemon_list):
                pokemon_key, pokemon_data = pokemon_list[i]
                
                # Species (encoded ID)
                species = pokemon_data.get('species', '')
                if species:
                    # Handle both string and enum species
                    species_name = getattr(species, 'name', str(species)).lower()
                    if species_name in self.species_to_id:
                        state_vector[base_offset] = self.species_to_id[species_name] / self.max_species
                
                # HP fraction
                hp_frac = pokemon_data.get('hp_fraction', 1.0)
                if hp_frac is not None:
                    state_vector[base_offset + 1] = max(0.0, min(1.0, hp_frac))
                
                # Status condition
                status = pokemon_data.get('status')
                if status:
                    status_encoding = {
                        'par': 1, 'slp': 2, 'frz': 3, 'brn': 4, 'psn': 5, 'tox': 6, 'fnt': 7
                    }
                    state_vector[base_offset + 2] = status_encoding.get(status.lower(), 0) / 7.0
                
                # Stat boosts (6 stats)
                boosts = pokemon_data.get('boosts', {})
                if boosts:
                    for j, stat in enumerate(['atk', 'def', 'spa', 'spd', 'spe', 'evasion']):
                        boost_val = boosts.get(stat, 0)
                        # Normalize -6 to +6 into 0.0 to 1.0
                        state_vector[base_offset + 3 + j] = (boost_val + 6) / 12.0
                
                # Types (2 types max)
                types = pokemon_data.get('types', [])
                if types:
                    for j, ptype in enumerate(types[:2]):
                        if ptype:
                            type_name = self._type_title(ptype)
                            if type_name in self.type_to_id:
                                state_vector[base_offset + 9 + j] = self.type_to_id[type_name] / self.max_types
                
                # Active flag
                is_active = pokemon_data.get('is_active', False)
                state_vector[base_offset + 11] = 1.0 if is_active else 0.0
                
                # Ability (if known)
                ability = pokemon_data.get('ability')
                if ability:
                    # Handle both string and enum abilities
                    ability_name = getattr(ability, 'name', str(ability)).lower()
                    if ability_name in self.ability_to_id:
                        state_vector[base_offset + 12] = self.ability_to_id[ability_name] / self.max_abilities
                
                # Item (if known)
                item = pokemon_data.get('item')
                if item:
                    # Handle both string and enum items
                    item_name = getattr(item, 'name', str(item)).lower()
                    if item_name in self.item_to_id:
                        state_vector[base_offset + 13] = self.item_to_id[item_name] / self.max_items
                
                # Level (normalized)
                level = pokemon_data.get('level', 50)
                if level:
                    state_vector[base_offset + 14] = level / 100.0
                
                # Revealed moves count
                revealed_moves = pokemon_data.get('revealed_moves', [])
                if revealed_moves:
                    state_vector[base_offset + 15] = len(revealed_moves) / 4.0  # Max 4 moves
                    
                    # Encode up to 4 moves
                    for j, move_id in enumerate(revealed_moves[:4]):
                        if move_id and move_id in self.move_to_id:
                            state_vector[base_offset + 16 + j] = self.move_to_id[move_id] / self.max_moves
                
                # Tera information (slots 20-22)
                tera_type = pokemon_data.get('tera_type')
                if tera_type:
                    # Handle both string and enum tera types
                    tera_type_name = getattr(tera_type, 'name', str(tera_type)).capitalize()
                    if tera_type_name in self.type_to_id:
                        state_vector[base_offset + 20] = self.type_to_id[tera_type_name] / self.max_types
                
                # Terastallized status
                terastallized = pokemon_data.get('terastallized', False)
                state_vector[base_offset + 21] = 1.0 if terastallized else 0.0
                
                # Effective defending types (for terastallized Pokemon, this would be just the tera type)
                # This helps the network understand the current type matchups
                if terastallized and tera_type:
                    # When terastallized, only tera type matters for defense
                    state_vector[base_offset + 22] = 1.0  # Tera defense active
                else:
                    # Normal dual typing defense
                    state_vector[base_offset + 22] = 0.0
                
                # Additional features could go in slots 23-24
                # Reserved for future expansion (held item effects, etc.)
        
        return offset + 6 * 25
    
    def _encode_field_from_snapshot(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int) -> int:
        """Encode field conditions from snapshot data"""
        
        # Weather
        weather = snapshot.get('weather')
        if weather:
            weather_encoding = {
                'sunnyday': 1, 'raindance': 2, 'sandstorm': 3, 'hail': 4, 'snow': 5,
                'sun': 1, 'rain': 2, 'sand': 3  # Alternative names
            }
            state_vector[offset] = weather_encoding.get(weather.lower(), 0) / 5.0
        
        # Terrain
        terrain = snapshot.get('terrain')
        if terrain:
            terrain_encoding = {
                'electricterrain': 1, 'grassyterrain': 2, 'mistyterrain': 3, 'psychicterrain': 4,
                'electric': 1, 'grassy': 2, 'misty': 3, 'psychic': 4  # Alternative names
            }
            state_vector[offset + 1] = terrain_encoding.get(terrain.lower(), 0) / 4.0
        
        # Trick Room
        if snapshot.get('trick_room', False):
            state_vector[offset + 2] = 1.0
        
        # Gravity
        if snapshot.get('gravity', False):
            state_vector[offset + 3] = 1.0
        
        # My side conditions (hazards and screens)
        my_side = snapshot.get('side_conditions', {})
        if my_side:
            # Hazards
            state_vector[offset + 4] = 1.0 if my_side.get('stealth_rock', False) else 0.0
            state_vector[offset + 5] = my_side.get('spikes', 0) / 3.0  # 0-3 layers
            state_vector[offset + 6] = my_side.get('toxic_spikes', 0) / 2.0  # 0-2 layers
            state_vector[offset + 7] = 1.0 if my_side.get('sticky_web', False) else 0.0
            
            # Screens
            state_vector[offset + 8] = 1.0 if my_side.get('reflect', False) else 0.0
            state_vector[offset + 9] = 1.0 if my_side.get('light_screen', False) else 0.0
            state_vector[offset + 10] = 1.0 if my_side.get('aurora_veil', False) else 0.0
            
            # Other conditions
            state_vector[offset + 11] = 1.0 if my_side.get('tailwind', False) else 0.0
            state_vector[offset + 12] = 1.0 if my_side.get('safeguard', False) else 0.0
            state_vector[offset + 13] = 1.0 if my_side.get('mist', False) else 0.0
        
        # Opponent side conditions
        opp_side = snapshot.get('opp_side_conditions', {})
        if opp_side:
            # Hazards
            state_vector[offset + 14] = 1.0 if opp_side.get('stealth_rock', False) else 0.0
            state_vector[offset + 15] = opp_side.get('spikes', 0) / 3.0
            state_vector[offset + 16] = opp_side.get('toxic_spikes', 0) / 2.0
            state_vector[offset + 17] = 1.0 if opp_side.get('sticky_web', False) else 0.0
            
            # Screens
            state_vector[offset + 18] = 1.0 if opp_side.get('reflect', False) else 0.0
            state_vector[offset + 19] = 1.0 if opp_side.get('light_screen', False) else 0.0
            state_vector[offset + 20] = 1.0 if opp_side.get('aurora_veil', False) else 0.0
            
            # Other conditions
            state_vector[offset + 21] = 1.0 if opp_side.get('tailwind', False) else 0.0
            state_vector[offset + 22] = 1.0 if opp_side.get('safeguard', False) else 0.0
            state_vector[offset + 23] = 1.0 if opp_side.get('mist', False) else 0.0
        
        # Enhanced move encoding with properties (Phase 1)
        active_moves = snapshot.get('active_moves_ids', [])
        if active_moves:
            state_vector[offset + 24] = len(active_moves) / 4.0  # Normalize by max moves
            
            # Encode up to 4 available moves with enhanced properties
            for i, move_id in enumerate(active_moves[:4]):
                base_idx = offset + 25 + (i * 5)  # 5 features per move
                
                if move_id and move_id in self.move_to_id:
                    # Original move ID encoding
                    state_vector[base_idx] = self.move_to_id[move_id] / self.max_moves
                    
                    # Enhanced move properties
                    if move_id in self.move_properties:
                        props = self.move_properties[move_id]
                        
                        # Base power (normalized to 0-1, max 200 power)
                        state_vector[base_idx + 1] = min(props['base_power'] / 200.0, 1.0)
                        
                        # Accuracy (normalized to 0-1)
                        state_vector[base_idx + 2] = props['accuracy'] / 100.0
                        
                        # Priority (normalized from -7 to +5 range)
                        state_vector[base_idx + 3] = (props['priority'] + 7) / 12.0
                        
                        # Move type (using existing type encoding)
                        move_type = props['type']
                        if move_type in self.type_to_id:
                            state_vector[base_idx + 4] = self.type_to_id[move_type] / self.max_types
        
        # Available switches (updated offset: was 29, now 45 due to move expansion)
        active_switches = snapshot.get('active_switch_ids', [])
        if active_switches:
            state_vector[offset + 45] = len(active_switches) / 5.0  # Max 5 switches
        
        # Force switch flag
        if snapshot.get('force_switch', False):
            state_vector[offset + 46] = 1.0
        
        # Can Tera
        if snapshot.get('can_tera', False):
            state_vector[offset + 47] = 1.0
        
        # Doubles battle flag
        if snapshot.get('is_doubles', False):
            state_vector[offset + 48] = 1.0
        
        # Phase 1 Enhancements: Boost Context (+12 dims)
        self._encode_boost_context(snapshot, state_vector, offset + 49)
        
        # Phase 1 Enhancements: HP Strategic Context (+6 dims)  
        self._encode_hp_strategic_context(snapshot, state_vector, offset + 61)
        
        # Phase 2 Enhancements: Speed Tier Analysis (+8 dims)
        self._encode_speed_tier_analysis(snapshot, state_vector, offset + 67)
        
        # Phase 2 Enhancements: Priority Move Context (+6 dims)
        self._encode_priority_move_context(snapshot, state_vector, offset + 75)
        
        # Phase 2 Enhancements: Choice Item Detection (+4 dims)
        self._encode_choice_item_detection(snapshot, state_vector, offset + 81)
        
        # Phase 2 Enhancements: Weather/Terrain Speed (+2 dims)
        self._encode_weather_terrain_speed(snapshot, state_vector, offset + 85)
        
        # Phase 3 Enhancements: Item/Ability Inference (+12 dims)
        self._encode_item_ability_inference(snapshot, state_vector, offset + 87)
        
        # Phase 3 Enhancements: Opponent Modeling (+8 dims)
        self._encode_opponent_modeling(snapshot, state_vector, offset + 99)
        
        # Phase 3 Enhancements: Strategic Pattern Recognition (+8 dims)
        self._encode_strategic_patterns(snapshot, state_vector, offset + 107)
        
        # Phase 3 Enhancements: Win Condition Tracking (+4 dims)
        self._encode_win_conditions(snapshot, state_vector, offset + 115)
        
        # Phase 4 Enhancements: Momentum & Turn History (+32 dims)
        self._encode_momentum_turn_history(snapshot, state_vector, offset + 119)
        
        # Phase 4 Enhancements: Battle Phase Analysis (+16 dims)
        self._encode_battle_phase_analysis(snapshot, state_vector, offset + 151)
        
        # Phase 4 Enhancements: Advanced Win Conditions (+16 dims)
        self._encode_advanced_win_conditions(snapshot, state_vector, offset + 167)
        
        # Future expansion slots available at offset + 183+
        
        return offset + 183  # Expanded to complete 830-dimension system
    
    def _encode_boost_context(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode boost strategic context for setup sweeper recognition"""
        
        # Extract team data
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team boost analysis (offset + 0 to offset + 5)
        my_stats = self._analyze_team_boosts(my_team)
        state_vector[offset + 0] = my_stats['setup_stage']  # 0-1 normalized setup progress
        state_vector[offset + 1] = my_stats['sweep_threat']  # 0-1 calculated sweep potential
        state_vector[offset + 2] = my_stats['debuff_count']  # 0-1 negative boosts
        state_vector[offset + 3] = my_stats['recovery_potential']  # 0-1 healing availability
        state_vector[offset + 4] = my_stats['offensive_boosts']  # 0-1 attack boosts
        state_vector[offset + 5] = my_stats['defensive_boosts']  # 0-1 defense boosts
        
        # Opponent team boost analysis (offset + 6 to offset + 11)
        opp_stats = self._analyze_team_boosts(opp_team)
        state_vector[offset + 6] = opp_stats['setup_stage']
        state_vector[offset + 7] = opp_stats['sweep_threat']
        state_vector[offset + 8] = opp_stats['debuff_count']
        state_vector[offset + 9] = opp_stats['recovery_potential']
        state_vector[offset + 10] = opp_stats['offensive_boosts']
        state_vector[offset + 11] = opp_stats['defensive_boosts']
        
    def _encode_hp_strategic_context(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode HP strategic context for tactical decisions"""
        
        # Extract team data
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team HP analysis (offset + 0 to offset + 2)
        my_hp_stats = self._analyze_team_hp(my_team)
        state_vector[offset + 0] = my_hp_stats['low_hp_count']  # Pokemon below 30%
        state_vector[offset + 1] = my_hp_stats['critical_hp_count']  # Pokemon below 10%
        state_vector[offset + 2] = my_hp_stats['average_team_hp']  # Overall team health
        
        # Opponent team HP analysis (offset + 3 to offset + 5)
        opp_hp_stats = self._analyze_team_hp(opp_team)
        state_vector[offset + 3] = opp_hp_stats['low_hp_count']
        state_vector[offset + 4] = opp_hp_stats['critical_hp_count']
        state_vector[offset + 5] = opp_hp_stats['average_team_hp']
    
    def _analyze_team_boosts(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze team for boost context"""
        
        total_positive_boosts = 0
        total_negative_boosts = 0
        offensive_boosts = 0
        defensive_boosts = 0
        max_sweep_threat = 0.0
        
        try:
            for pokemon_data in team_data.values():
                boosts = pokemon_data.get('boosts', {})
                hp_fraction = pokemon_data.get('hp_fraction', 1.0)
                
                # Handle None hp_fraction
                if hp_fraction is None:
                    hp_fraction = 1.0
                
                # Count positive and negative boosts
                for stat, boost_value in boosts.items():
                    if boost_value > 0:
                        total_positive_boosts += boost_value
                        if stat in ['atk', 'spa']:
                            offensive_boosts += boost_value
                        elif stat in ['def', 'spd']:
                            defensive_boosts += boost_value
                    elif boost_value < 0:
                        total_negative_boosts += abs(boost_value)
                
                # Calculate individual sweep threat
                atk_boost = boosts.get('atk', 0) + boosts.get('spa', 0)  # Combined offensive
                spe_boost = boosts.get('spe', 0)
                individual_threat = min((atk_boost + spe_boost) * hp_fraction / 6.0, 1.0)
                max_sweep_threat = max(max_sweep_threat, individual_threat)
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Boost analysis error: {e}")
        
        return {
            'setup_stage': min(total_positive_boosts / 12.0, 1.0),  # Normalize to 0-1
            'sweep_threat': max_sweep_threat,
            'debuff_count': min(total_negative_boosts / 12.0, 1.0),
            'recovery_potential': 0.5,  # Placeholder - would check for recovery moves
            'offensive_boosts': min(offensive_boosts / 6.0, 1.0),
            'defensive_boosts': min(defensive_boosts / 6.0, 1.0)
        }
    
    def _analyze_team_hp(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze team HP for strategic context"""
        
        total_pokemon = 0
        low_hp_count = 0  # Below 30%
        critical_hp_count = 0  # Below 10%
        total_hp = 0.0
        
        try:
            for pokemon_data in team_data.values():
                hp_fraction = pokemon_data.get('hp_fraction', 1.0)
                fainted = pokemon_data.get('fainted', False)
                
                # Handle None hp_fraction
                if hp_fraction is None:
                    hp_fraction = 1.0
                
                if not fainted:
                    total_pokemon += 1
                    total_hp += hp_fraction
                    
                    if hp_fraction < 0.3:
                        low_hp_count += 1
                    if hp_fraction < 0.1:
                        critical_hp_count += 1
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] HP analysis error: {e}")
        
        if total_pokemon == 0:
            return {'low_hp_count': 0.0, 'critical_hp_count': 0.0, 'average_team_hp': 0.0}
        
        return {
            'low_hp_count': low_hp_count / 6.0,  # Normalize to max 6 Pokemon
            'critical_hp_count': critical_hp_count / 6.0,
            'average_team_hp': total_hp / total_pokemon
        }
    
    # ============================================================================
    # Phase 2 Enhancements: Speed Calculations & Turn Order (+20 dims)
    # ============================================================================
    
    def _encode_speed_tier_analysis(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode speed tier analysis for turn order prediction (+8 dims)"""
        
        # Extract team data
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team speed analysis (offset + 0 to offset + 3)
        my_speed_stats = self._analyze_team_speeds(my_team)
        state_vector[offset + 0] = my_speed_stats['fastest_relative']      # Fastest Pokemon speed tier (0-1)
        state_vector[offset + 1] = my_speed_stats['slowest_relative']      # Slowest Pokemon speed tier (0-1)
        state_vector[offset + 2] = my_speed_stats['average_speed']         # Team average speed (0-1)
        state_vector[offset + 3] = my_speed_stats['speed_boost_advantage'] # Net speed boosts (0-1)
        
        # Opponent team speed analysis (offset + 4 to offset + 7)
        opp_speed_stats = self._analyze_team_speeds(opp_team)
        state_vector[offset + 4] = opp_speed_stats['fastest_relative']
        state_vector[offset + 5] = opp_speed_stats['slowest_relative']
        state_vector[offset + 6] = opp_speed_stats['average_speed']
        state_vector[offset + 7] = opp_speed_stats['speed_boost_advantage']
    
    def _analyze_team_speeds(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze team for speed tier context"""
        
        speeds = []
        total_speed_boosts = 0
        fastest_speed = 0
        slowest_speed = 9999
        
        try:
            for pokemon_data in team_data.values():
                # Get base speed stats if available
                stats = pokemon_data.get('stats', {})
                base_speed = stats.get('spe', 100) if stats else 100  # Default to reasonable speed
                
                # Handle None values from Phase 1 lessons
                if base_speed is None:
                    base_speed = 100
                
                # Apply speed boosts (1.5x per stage)
                boosts = pokemon_data.get('boosts', {})
                speed_boost = boosts.get('spe', 0) if boosts else 0
                if speed_boost is None:
                    speed_boost = 0
                
                # Calculate effective speed (simplified)
                boost_multiplier = 1.0 + (speed_boost * 0.5)  # Each +1 = 50% increase
                effective_speed = base_speed * boost_multiplier
                
                speeds.append(effective_speed)
                total_speed_boosts += speed_boost
                fastest_speed = max(fastest_speed, effective_speed)
                if effective_speed > 0:  # Avoid 0 speed
                    slowest_speed = min(slowest_speed, effective_speed)
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Speed analysis error: {e}")
            # Fallback values
            fastest_speed = 100
            slowest_speed = 100
            speeds = [100]
        
        # Ensure we have valid speeds
        if not speeds:
            speeds = [100]
        if slowest_speed == 9999:
            slowest_speed = 100
        
        # Normalize speeds to 0-1 range (typical Pokemon speeds: 20-300 with boosts)
        max_possible_speed = 400  # Accounting for boosts and fast Pokemon
        
        return {
            'fastest_relative': min(fastest_speed / max_possible_speed, 1.0),
            'slowest_relative': min(slowest_speed / max_possible_speed, 1.0),
            'average_speed': min(sum(speeds) / len(speeds) / max_possible_speed, 1.0),
            'speed_boost_advantage': max(-1.0, min(total_speed_boosts / 6.0, 1.0))  # -6 to +6 boosts
        }
    
    def _encode_priority_move_context(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode priority move context for turn order guarantees (+6 dims)"""
        
        # Extract team data and available moves
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team priority analysis (offset + 0 to offset + 2)
        my_priority_stats = self._analyze_priority_moves(my_team)
        state_vector[offset + 0] = my_priority_stats['has_priority_moves']    # 0-1: priority move availability
        state_vector[offset + 1] = my_priority_stats['priority_level']        # 0-1: highest priority level
        state_vector[offset + 2] = my_priority_stats['guaranteed_first']      # 0-1: can guarantee first move
        
        # Opponent priority analysis (offset + 3 to offset + 5)
        opp_priority_stats = self._analyze_priority_moves(opp_team)
        state_vector[offset + 3] = opp_priority_stats['has_priority_moves']
        state_vector[offset + 4] = opp_priority_stats['priority_level']
        state_vector[offset + 5] = opp_priority_stats['guaranteed_first']
    
    def _analyze_priority_moves(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze team for priority move context"""
        
        priority_moves_found = 0
        max_priority = 0
        can_guarantee_first = 0
        
        # Common priority moves and their levels
        priority_moves = {
            'extremespeed': 2, 'quickattack': 1, 'aquajet': 1, 'machpunch': 1,
            'bulletpunch': 1, 'shadowsneak': 1, 'suckerpunch': 1, 'fakeout': 3,
            'iceshards': 1, 'vacuumwave': 1, 'grassyglide': 0  # Priority in grassy terrain
        }
        
        try:
            for pokemon_data in team_data.values():
                # Check if Pokemon has priority moves
                # Note: In actual implementation, would need move data
                # For now, use heuristic based on species
                species = pokemon_data.get('species', '')
                if species:
                    species_name = str(species).lower()
                    
                    # Check for common priority move users
                    if any(name in species_name for name in ['lucario', 'scizor', 'dragonite', 'talonflame']):
                        priority_moves_found += 1
                        max_priority = max(max_priority, 1)  # Assume +1 priority
                        can_guarantee_first = 1  # Can outspeed most opponents
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Priority analysis error: {e}")
        
        return {
            'has_priority_moves': min(priority_moves_found / 6.0, 1.0),  # Normalize to team size
            'priority_level': min(max_priority / 3.0, 1.0),  # Normalize to max +3 priority
            'guaranteed_first': can_guarantee_first  # Binary: can guarantee first move
        }
    
    def _encode_choice_item_detection(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode choice item detection for locked move prediction (+4 dims)"""
        
        # Extract team data
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team choice item analysis (offset + 0 to offset + 1)
        my_choice_stats = self._analyze_choice_items(my_team)
        state_vector[offset + 0] = my_choice_stats['choice_item_count']    # 0-1: number of choice items
        state_vector[offset + 1] = my_choice_stats['locked_move_risk']     # 0-1: risk of being locked
        
        # Opponent choice item analysis (offset + 2 to offset + 3)
        opp_choice_stats = self._analyze_choice_items(opp_team)
        state_vector[offset + 2] = opp_choice_stats['choice_item_count']
        state_vector[offset + 3] = opp_choice_stats['locked_move_risk']
    
    def _analyze_choice_items(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze team for choice item context"""
        
        choice_items = 0
        locked_risk = 0.0
        
        try:
            for pokemon_data in team_data.values():
                # Check for choice items (would need item data in real implementation)
                item = pokemon_data.get('item', '')
                if item and any(choice in str(item).lower() for choice in ['choice', 'band', 'specs', 'scarf']):
                    choice_items += 1
                    
                    # If active and healthy, higher lock risk
                    is_active = pokemon_data.get('is_active', False)
                    hp_fraction = pokemon_data.get('hp_fraction', 1.0)
                    if hp_fraction is None:
                        hp_fraction = 1.0
                    
                    if is_active and hp_fraction > 0.5:
                        locked_risk = 0.8  # High risk of being locked into suboptimal move
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Choice item analysis error: {e}")
        
        return {
            'choice_item_count': min(choice_items / 6.0, 1.0),  # Normalize to team size
            'locked_move_risk': locked_risk  # Risk of bad choice lock
        }
    
    def _encode_weather_terrain_speed(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode weather/terrain speed modifiers (+2 dims)"""
        
        # Extract field conditions
        field = snapshot.get('field', {})
        weather = field.get('weather', '')
        terrain = field.get('terrain', '')
        
        # My team and opp team for ability analysis
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team speed modifiers (offset + 0)
        my_speed_bonus = self._calculate_speed_modifiers(my_team, weather, terrain)
        state_vector[offset + 0] = my_speed_bonus
        
        # Opponent team speed modifiers (offset + 1)
        opp_speed_bonus = self._calculate_speed_modifiers(opp_team, weather, terrain)
        state_vector[offset + 1] = opp_speed_bonus
    
    def _calculate_speed_modifiers(self, team_data: Dict[str, Any], weather: str, terrain: str) -> float:
        """Calculate speed modifiers from weather/terrain"""
        
        max_bonus = 0.0
        
        try:
            for pokemon_data in team_data.values():
                bonus = 0.0
                species = pokemon_data.get('species', '').lower()
                
                # Weather-based speed boosts
                if weather:
                    weather_lower = str(weather).lower()
                    if 'rain' in weather_lower and any(water in species for water in ['kingdra', 'swift swim']):
                        bonus = 1.0  # Swift Swim doubles speed
                    elif 'sun' in weather_lower and any(chloro in species for chloro in ['venusaur', 'chlorophyll']):
                        bonus = 1.0  # Chlorophyll doubles speed
                    elif 'sand' in weather_lower and any(sand in species for sand in ['excadrill', 'sand rush']):
                        bonus = 1.0  # Sand Rush doubles speed
                
                # Terrain-based speed boosts
                if terrain:
                    terrain_lower = str(terrain).lower()
                    if 'electric' in terrain_lower and any(elec in species for elec in ['tapu koko', 'surge']):
                        bonus = max(bonus, 0.5)  # Electric Surge boosts in Electric Terrain
                
                max_bonus = max(max_bonus, bonus)
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Speed modifier analysis error: {e}")
        
        return min(max_bonus, 1.0)  # Cap at 1.0
    
    # ============================================================================
    # Phase 3 Enhancements: Meta-game & Prediction (+32 dims)
    # ============================================================================
    
    def _encode_item_ability_inference(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode item/ability inference from battle patterns (+12 dims)"""
        
        # Extract team data
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # My team item/ability analysis (offset + 0 to offset + 5)
        my_inference = self._analyze_item_ability_patterns(my_team, True)
        state_vector[offset + 0] = my_inference['hidden_item_probability']    # 0-1: likely hidden items
        state_vector[offset + 1] = my_inference['mega_potential']             # 0-1: mega evolution capability
        state_vector[offset + 2] = my_inference['z_move_available']           # 0-1: Z-move availability
        state_vector[offset + 3] = my_inference['ability_synergy']            # 0-1: team ability synergy
        state_vector[offset + 4] = my_inference['item_advantage']             # 0-1: item-based advantage
        state_vector[offset + 5] = my_inference['setup_item_risk']            # 0-1: setup item vulnerability
        
        # Opponent team inference (offset + 6 to offset + 11)
        opp_inference = self._analyze_item_ability_patterns(opp_team, False)
        state_vector[offset + 6] = opp_inference['hidden_item_probability']
        state_vector[offset + 7] = opp_inference['mega_potential']
        state_vector[offset + 8] = opp_inference['z_move_available']
        state_vector[offset + 9] = opp_inference['ability_synergy']
        state_vector[offset + 10] = opp_inference['item_advantage']
        state_vector[offset + 11] = opp_inference['setup_item_risk']
    
    def _analyze_item_ability_patterns(self, team_data: Dict[str, Any], is_my_team: bool) -> Dict[str, float]:
        """Analyze item/ability patterns for strategic inference"""
        
        hidden_items = 0.0
        mega_stones = 0.0
        z_crystals = 0.0
        ability_synergy = 0.0
        powerful_items = 0.0
        setup_vulnerability = 0.0
        
        try:
            for pokemon_data in team_data.values():
                species = str(pokemon_data.get('species', '')).lower()
                item = str(pokemon_data.get('item', '')).lower()
                ability = str(pokemon_data.get('ability', '')).lower()
                
                # Hidden item inference (items that aren't immediately obvious)
                if not item or 'unknown' in item:
                    # Common hidden items based on species
                    if any(sp in species for sp in ['garchomp', 'tyranitar', 'metagross']):
                        hidden_items += 0.7  # Likely has Choice item or Life Orb
                    elif any(sp in species for sp in ['toxapex', 'ferrothorn', 'skarmory']):
                        hidden_items += 0.5  # Likely has Leftovers or Rocky Helmet
                
                # Mega evolution potential
                mega_species = ['charizard', 'venusaur', 'blastoise', 'lucario', 'garchomp', 'tyranitar', 'metagross']
                if any(mega in species for mega in mega_species) and ('mega' in item or not item):
                    mega_stones += 1.0
                
                # Z-move availability (Gen 7 mechanic)
                if 'crystal' in item or 'ium' in item:
                    z_crystals += 1.0
                
                # Ability synergy detection
                weather_abilities = ['drought', 'drizzle', 'sand stream', 'snow warning']
                terrain_abilities = ['electric surge', 'grassy surge', 'psychic surge', 'misty surge']
                if any(weather in ability for weather in weather_abilities + terrain_abilities):
                    ability_synergy += 0.8
                
                # Powerful item detection
                power_items = ['choice', 'life orb', 'assault vest', 'focus sash']
                if any(power in item for power in power_items):
                    powerful_items += 1.0
                
                # Setup vulnerability (frail setup sweepers)
                setup_pokemon = ['alakazam', 'gengar', 'dragonite', 'salamence']
                if any(setup in species for setup in setup_pokemon):
                    hp_fraction = pokemon_data.get('hp_fraction', 1.0)
                    if hp_fraction is None:
                        hp_fraction = 1.0
                    if hp_fraction < 0.6:  # Low HP setup sweeper = vulnerable
                        setup_vulnerability += 0.8
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Item/ability inference error: {e}")
        
        team_size = max(len(team_data), 1)
        return {
            'hidden_item_probability': min(hidden_items / team_size, 1.0),
            'mega_potential': min(mega_stones / team_size, 1.0),
            'z_move_available': min(z_crystals / team_size, 1.0),
            'ability_synergy': min(ability_synergy / team_size, 1.0),
            'item_advantage': min(powerful_items / team_size, 1.0),
            'setup_item_risk': min(setup_vulnerability / team_size, 1.0)
        }
    
    def _encode_opponent_modeling(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode opponent modeling and play style analysis (+8 dims)"""
        
        # Extract battle data for opponent analysis
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        opp_team = snapshot.get('opp_team', {})
        
        # Opponent modeling analysis
        opp_model = self._analyze_opponent_patterns(opp_team, turn)
        
        state_vector[offset + 0] = opp_model['aggression_level']         # 0-1: aggressive vs defensive
        state_vector[offset + 1] = opp_model['setup_tendency']          # 0-1: prefers setup vs immediate offense
        state_vector[offset + 2] = opp_model['prediction_accuracy']     # 0-1: how predictable opponent is
        state_vector[offset + 3] = opp_model['risk_taking']             # 0-1: conservative vs high-risk plays
        state_vector[offset + 4] = opp_model['switching_frequency']     # 0-1: switches often vs stays in
        state_vector[offset + 5] = opp_model['endgame_preparation']     # 0-1: plans for endgame vs reactive
        state_vector[offset + 6] = opp_model['item_usage_skill']        # 0-1: optimal item timing
        state_vector[offset + 7] = opp_model['overall_skill_estimate']  # 0-1: estimated skill level
    
    def _analyze_opponent_patterns(self, opp_team: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze opponent play patterns for modeling"""
        
        # Simplified opponent analysis (in real implementation, would track history)
        aggression = 0.5  # Default neutral
        setup_preference = 0.3  # Slightly offensive
        predictability = 0.6  # Moderately predictable
        risk_level = 0.4  # Conservative default
        switch_rate = 0.5  # Moderate switching
        endgame_prep = 0.5  # Average planning
        item_skill = 0.5  # Average item usage
        skill_estimate = 0.5  # Default skill level
        
        try:
            # Early game aggression indicators
            if turn <= 3:
                # Look for immediate offensive pressure
                active_pokemon = None
                for pokemon_data in opp_team.values():
                    if pokemon_data.get('is_active', False):
                        active_pokemon = pokemon_data
                        break
                
                if active_pokemon:
                    species = str(active_pokemon.get('species', '')).lower()
                    # Aggressive species indicate offensive play style
                    if any(aggro in species for aggro in ['garchomp', 'salamence', 'dragonite', 'tyranitar']):
                        aggression = 0.8
                        risk_level = 0.7
                    # Setup species indicate setup preference
                    elif any(setup in species for setup in ['alakazam', 'gengar', 'clefable']):
                        setup_preference = 0.9
                        endgame_prep = 0.8
            
            # Team composition analysis for play style
            offensive_count = 0
            defensive_count = 0
            
            for pokemon_data in opp_team.values():
                species = str(pokemon_data.get('species', '')).lower()
                
                # Categorize by common roles
                if any(off in species for off in ['garchomp', 'salamence', 'metagross', 'tyranitar']):
                    offensive_count += 1
                elif any(def_ in species for def_ in ['toxapex', 'ferrothorn', 'skarmory', 'chansey']):
                    defensive_count += 1
            
            # Adjust play style based on team composition
            if offensive_count > defensive_count:
                aggression = min(aggression + 0.3, 1.0)
                risk_level = min(risk_level + 0.2, 1.0)
            elif defensive_count > offensive_count:
                setup_preference = min(setup_preference + 0.3, 1.0)
                endgame_prep = min(endgame_prep + 0.3, 1.0)
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Opponent modeling error: {e}")
        
        return {
            'aggression_level': aggression,
            'setup_tendency': setup_preference,
            'prediction_accuracy': predictability,
            'risk_taking': risk_level,
            'switching_frequency': switch_rate,
            'endgame_preparation': endgame_prep,
            'item_usage_skill': item_skill,
            'overall_skill_estimate': skill_estimate
        }
    
    def _encode_strategic_patterns(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode strategic pattern recognition (+8 dims)"""
        
        # Extract battle context
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        field = snapshot.get('field', {})
        
        # Strategic pattern analysis
        patterns = self._analyze_strategic_patterns(my_team, opp_team, field, turn)
        
        state_vector[offset + 0] = patterns['opening_pattern_strength']   # 0-1: recognizable opening
        state_vector[offset + 1] = patterns['momentum_direction']         # 0-1: who has momentum
        state_vector[offset + 2] = patterns['endgame_proximity']          # 0-1: approaching endgame
        state_vector[offset + 3] = patterns['strategy_commitment']        # 0-1: committed to strategy
        state_vector[offset + 4] = patterns['counter_preparation']        # 0-1: prepared for counters
        state_vector[offset + 5] = patterns['win_condition_clarity']      # 0-1: clear path to victory
        state_vector[offset + 6] = patterns['adaptation_needed']          # 0-1: need to adapt strategy
        state_vector[offset + 7] = patterns['pattern_confidence']         # 0-1: confidence in pattern read
    
    def _analyze_strategic_patterns(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], 
                                   field: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze strategic patterns in the battle"""
        
        # Initialize pattern scores
        opening_strength = 0.0
        momentum = 0.5  # Neutral momentum
        endgame_proximity = 0.0
        strategy_commitment = 0.5
        counter_prep = 0.5
        win_clarity = 0.3
        adaptation_need = 0.3
        confidence = 0.6
        
        try:
            # Opening pattern recognition (turns 1-5)
            if turn <= 5:
                # Weather/terrain setup patterns
                weather = field.get('weather', '')
                terrain = field.get('terrain', '')
                if weather or terrain:
                    opening_strength = 0.8  # Clear weather/terrain setup
                    strategy_commitment = 0.9
                
                # Lead Pokemon analysis
                my_active = None
                opp_active = None
                for pokemon_data in my_team.values():
                    if pokemon_data.get('is_active', False):
                        my_active = pokemon_data
                for pokemon_data in opp_team.values():
                    if pokemon_data.get('is_active', False):
                        opp_active = pokemon_data
                
                if my_active and opp_active:
                    my_species = str(my_active.get('species', '')).lower()
                    opp_species = str(opp_active.get('species', '')).lower()
                    
                    # Stealth Rock leads
                    if any(sr in my_species for sr in ['garchomp', 'landorus', 'tyranitar']):
                        opening_strength = max(opening_strength, 0.7)
                    
                    # Setup leads vs offensive leads
                    setup_species = ['clefable', 'toxapex', 'ferrothorn']
                    if any(setup in opp_species for setup in setup_species):
                        counter_prep = 0.8  # Need offensive pressure
            
            # Momentum calculation based on HP
            my_total_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_total_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            
            if my_total_hp > 0 and opp_total_hp > 0:
                hp_ratio = my_total_hp / (my_total_hp + opp_total_hp)
                momentum = hp_ratio  # Higher HP = better momentum
            
            # Endgame proximity (based on total Pokemon remaining)
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            
            total_alive = my_alive + opp_alive
            if total_alive <= 4:  # 2v2 or less
                endgame_proximity = 1.0
            elif total_alive <= 6:  # 3v3
                endgame_proximity = 0.7
            elif total_alive <= 8:  # 4v4
                endgame_proximity = 0.4
            
            # Win condition clarity
            if endgame_proximity > 0.7:
                if my_alive > opp_alive:
                    win_clarity = 0.9  # Clear numbers advantage
                elif momentum > 0.7:
                    win_clarity = 0.8  # Clear momentum advantage
            
            # Adaptation needed (high turn count with low momentum)
            if turn > 15 and abs(momentum - 0.5) < 0.2:
                adaptation_need = 0.8  # Stalemate, need new approach
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Strategic pattern error: {e}")
        
        return {
            'opening_pattern_strength': min(opening_strength, 1.0),
            'momentum_direction': min(momentum, 1.0),
            'endgame_proximity': min(endgame_proximity, 1.0),
            'strategy_commitment': min(strategy_commitment, 1.0),
            'counter_preparation': min(counter_prep, 1.0),
            'win_condition_clarity': min(win_clarity, 1.0),
            'adaptation_needed': min(adaptation_need, 1.0),
            'pattern_confidence': min(confidence, 1.0)
        }
    
    def _encode_win_conditions(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode win condition tracking (+4 dims)"""
        
        # Extract battle context
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        
        # Win condition analysis
        win_conditions = self._analyze_win_conditions(my_team, opp_team, turn)
        
        state_vector[offset + 0] = win_conditions['primary_win_condition']    # 0-1: strength of main win con
        state_vector[offset + 1] = win_conditions['backup_strategy_viability'] # 0-1: backup plan strength
        state_vector[offset + 2] = win_conditions['win_probability_estimate']  # 0-1: estimated win chance
        state_vector[offset + 3] = win_conditions['threat_priority_level']     # 0-1: immediate threat level
    
    def _analyze_win_conditions(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze win conditions and strategic objectives"""
        
        primary_strength = 0.5
        backup_viability = 0.5
        win_probability = 0.5
        threat_level = 0.3
        
        try:
            # Calculate team strengths
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            
            my_total_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_total_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            
            # Primary win condition assessment
            if my_alive > opp_alive + 1:
                primary_strength = 0.9  # Strong numbers advantage
                win_probability = 0.8
            elif my_total_hp > opp_total_hp * 1.5:
                primary_strength = 0.8  # Strong HP advantage
                win_probability = 0.7
            elif my_alive == opp_alive and my_total_hp > opp_total_hp:
                primary_strength = 0.6  # Moderate advantage
                win_probability = 0.6
            
            # Backup strategy viability
            if my_alive >= 3:
                backup_viability = 0.8  # Multiple options available
            elif my_alive == 2:
                backup_viability = 0.5  # Limited options
            else:
                backup_viability = 0.2  # Must execute primary plan
            
            # Threat assessment
            opp_active = None
            for pokemon_data in opp_team.values():
                if pokemon_data.get('is_active', False):
                    opp_active = pokemon_data
                    break
            
            if opp_active:
                opp_hp = opp_active.get('hp_fraction', 1.0)
                if opp_hp is None:
                    opp_hp = 1.0
                
                # High HP opponent with setup potential = high threat
                species = str(opp_active.get('species', '')).lower()
                setup_threats = ['clefable', 'toxapex', 'dragonite', 'salamence']
                if any(threat in species for threat in setup_threats) and opp_hp > 0.8:
                    threat_level = 0.9
                elif opp_hp > 0.9:
                    threat_level = 0.6
                elif opp_hp < 0.3:
                    threat_level = 0.2  # Low threat
            
            # Endgame adjustments
            if my_alive + opp_alive <= 4:
                # Endgame: more decisive
                if win_probability > 0.5:
                    win_probability = min(win_probability + 0.2, 1.0)
                else:
                    win_probability = max(win_probability - 0.2, 0.0)
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Win condition analysis error: {e}")
        
        return {
            'primary_win_condition': min(primary_strength, 1.0),
            'backup_strategy_viability': min(backup_viability, 1.0),
            'win_probability_estimate': min(win_probability, 1.0),
            'threat_priority_level': min(threat_level, 1.0)
        }
    
    # ============================================================================
    # Phase 4 Enhancements: Game State & Momentum (+64 dims)
    # ============================================================================
    
    def _encode_momentum_turn_history(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode momentum and turn history analysis (+32 dims)"""
        
        # Extract battle context with null safety
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        # Calculate comprehensive momentum
        momentum_data = self._analyze_comprehensive_momentum(my_team, opp_team, turn)
        
        # Current momentum state (offset + 0 to offset + 7)
        state_vector[offset + 0] = momentum_data['hp_momentum']           # 0-1: HP-based momentum
        state_vector[offset + 1] = momentum_data['position_momentum']     # 0-1: positional advantage
        state_vector[offset + 2] = momentum_data['tempo_control']         # 0-1: tempo control
        state_vector[offset + 3] = momentum_data['pressure_level']        # 0-1: offensive pressure
        state_vector[offset + 4] = momentum_data['momentum_trend']        # 0-1: momentum direction
        state_vector[offset + 5] = momentum_data['initiative']            # 0-1: who has initiative
        state_vector[offset + 6] = momentum_data['resource_advantage']    # 0-1: resource (items/abilities) advantage
        state_vector[offset + 7] = momentum_data['momentum_stability']    # 0-1: how stable current momentum is
        
        # Turn progression analysis (offset + 8 to offset + 15)
        turn_data = self._analyze_turn_progression(turn, my_team, opp_team)
        state_vector[offset + 8] = turn_data['game_phase_progress']       # 0-1: progression through game phases
        state_vector[offset + 9] = turn_data['turn_efficiency']           # 0-1: efficient use of turns
        state_vector[offset + 10] = turn_data['time_pressure']            # 0-1: time pressure factor
        state_vector[offset + 11] = turn_data['setup_window']             # 0-1: remaining setup opportunities
        state_vector[offset + 12] = turn_data['endgame_countdown']         # 0-1: proximity to forced endgame
        state_vector[offset + 13] = turn_data['critical_turn_proximity']  # 0-1: approaching critical decisions
        state_vector[offset + 14] = turn_data['turn_advantage']           # 0-1: turn count advantage
        state_vector[offset + 15] = turn_data['phase_transition']         # 0-1: transitioning between phases
        
        # Historical momentum tracking (offset + 16 to offset + 23)
        history_data = self._analyze_momentum_history(turn)
        state_vector[offset + 16] = history_data['recent_momentum_shift']  # 0-1: recent momentum changes
        state_vector[offset + 17] = history_data['momentum_volatility']    # 0-1: momentum stability
        state_vector[offset + 18] = history_data['comeback_potential']     # 0-1: ability to comeback
        state_vector[offset + 19] = history_data['snowball_risk']          # 0-1: risk of momentum snowball
        state_vector[offset + 20] = history_data['turning_point_proximity'] # 0-1: near major turning point
        state_vector[offset + 21] = history_data['momentum_consistency']   # 0-1: consistent momentum direction
        state_vector[offset + 22] = history_data['pressure_accumulation']  # 0-1: building pressure over time
        state_vector[offset + 23] = history_data['strategic_patience']     # 0-1: patience vs urgency balance
        
        # Advanced momentum factors (offset + 24 to offset + 31)
        advanced_data = self._analyze_advanced_momentum(my_team, opp_team, turn)
        state_vector[offset + 24] = advanced_data['psychological_advantage'] # 0-1: mental/psychological edge
        state_vector[offset + 25] = advanced_data['adaptation_momentum']    # 0-1: adapting successfully
        state_vector[offset + 26] = advanced_data['information_advantage']  # 0-1: information asymmetry
        state_vector[offset + 27] = advanced_data['execution_quality']      # 0-1: quality of play execution
        state_vector[offset + 28] = advanced_data['option_density']         # 0-1: number of viable options
        state_vector[offset + 29] = advanced_data['threat_density']         # 0-1: multiple threats active
        state_vector[offset + 30] = advanced_data['control_leverage']       # 0-1: ability to control game flow
        state_vector[offset + 31] = advanced_data['momentum_amplification'] # 0-1: momentum amplification factors
    
    def _analyze_comprehensive_momentum(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze comprehensive momentum across multiple dimensions"""
        
        try:
            # Calculate HP momentum
            my_total_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_total_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            hp_momentum = my_total_hp / max(my_total_hp + opp_total_hp, 0.1)
            
            # Position momentum (based on team positioning)
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            position_momentum = my_alive / max(my_alive + opp_alive, 1)
            
            # Tempo control (early game aggression vs late game control)
            if turn <= 10:
                tempo_control = 0.7 if hp_momentum > 0.6 else 0.3  # Early aggression
            else:
                tempo_control = 0.8 if position_momentum > 0.5 else 0.2  # Late control
            
            # Pressure level calculation
            opp_active = None
            my_active = None
            for p in opp_team.values():
                if p.get('is_active', False):
                    opp_active = p
            for p in my_team.values():
                if p.get('is_active', False):
                    my_active = p
            
            pressure_level = 0.5
            if opp_active:
                opp_hp = opp_active.get('hp_fraction') or 1.0
                if opp_hp < 0.3:
                    pressure_level = 0.9  # High pressure on low HP opponent
                elif opp_hp > 0.9:
                    pressure_level = 0.2  # Low pressure on healthy opponent
            
            # Momentum trend (increasing vs decreasing)
            momentum_trend = hp_momentum  # Simplified - would track over time
            
            # Initiative (who's driving the action)
            initiative = 0.7 if my_alive >= opp_alive else 0.3
            
            # Resource advantage
            my_items = sum(1 for p in my_team.values() if p.get('item', '') not in ['', 'unknown'])
            opp_items = sum(1 for p in opp_team.values() if p.get('item', '') not in ['', 'unknown'])
            resource_advantage = 0.5 + (my_items - opp_items) * 0.1
            
            # Momentum stability
            stability = 0.8 if abs(hp_momentum - 0.5) > 0.3 else 0.3  # Stable if clear advantage
            
            return {
                'hp_momentum': min(hp_momentum, 1.0),
                'position_momentum': min(position_momentum, 1.0),
                'tempo_control': min(tempo_control, 1.0),
                'pressure_level': min(pressure_level, 1.0),
                'momentum_trend': min(momentum_trend, 1.0),
                'initiative': min(initiative, 1.0),
                'resource_advantage': max(0.0, min(resource_advantage, 1.0)),
                'momentum_stability': min(stability, 1.0)
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Momentum analysis error: {e}")
            return {k: 0.5 for k in ['hp_momentum', 'position_momentum', 'tempo_control', 'pressure_level', 
                                   'momentum_trend', 'initiative', 'resource_advantage', 'momentum_stability']}
    
    def _analyze_turn_progression(self, turn: int, my_team: Dict[str, Any], opp_team: Dict[str, Any]) -> Dict[str, float]:
        """Analyze turn progression and phase timing"""
        
        try:
            # Game phase progress (opening -> midgame -> endgame)
            if turn <= 5:
                phase_progress = 0.1  # Opening
            elif turn <= 15:
                phase_progress = 0.4  # Early midgame
            elif turn <= 30:
                phase_progress = 0.7  # Late midgame
            else:
                phase_progress = 1.0  # Endgame
            
            # Turn efficiency (productive turns vs stalling)
            turn_efficiency = max(0.3, 1.0 - turn / 50.0)  # Efficiency decreases with length
            
            # Time pressure
            time_pressure = min(turn / 40.0, 1.0)  # Increases with turn count
            
            # Setup window (remaining opportunities for setup)
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            setup_window = max(0.0, (my_alive - 2) / 4.0)  # More Pokemon = more setup options
            
            # Endgame countdown
            total_alive = my_alive + sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            endgame_countdown = max(0.0, 1.0 - total_alive / 8.0)
            
            # Critical turn proximity
            critical_proximity = 0.8 if total_alive <= 4 else 0.3
            
            # Turn advantage (based on available options)
            turn_advantage = min(my_alive / 6.0, 1.0)
            
            # Phase transition
            phase_transition = 0.8 if turn in [5, 15, 30] else 0.2
            
            return {
                'game_phase_progress': phase_progress,
                'turn_efficiency': turn_efficiency,
                'time_pressure': time_pressure,
                'setup_window': setup_window,
                'endgame_countdown': endgame_countdown,
                'critical_turn_proximity': critical_proximity,
                'turn_advantage': turn_advantage,
                'phase_transition': phase_transition
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Turn progression error: {e}")
            return {k: 0.5 for k in ['game_phase_progress', 'turn_efficiency', 'time_pressure', 'setup_window',
                                   'endgame_countdown', 'critical_turn_proximity', 'turn_advantage', 'phase_transition']}
    
    def _analyze_momentum_history(self, turn: int) -> Dict[str, float]:
        """Analyze momentum history patterns"""
        
        # Simplified momentum history (would track actual history in full implementation)
        try:
            recent_shift = 0.6 if turn % 5 == 0 else 0.3  # Momentum shifts every 5 turns
            volatility = min(turn / 20.0, 0.8)  # More volatile as game progresses
            comeback_potential = max(0.2, 1.0 - turn / 40.0)  # Decreases over time
            snowball_risk = min(turn / 30.0, 0.9)  # Increases over time
            turning_point = 0.8 if turn in [10, 20, 35] else 0.2
            consistency = max(0.3, 1.0 - volatility)
            pressure_accumulation = min(turn / 25.0, 1.0)
            strategic_patience = max(0.1, 1.0 - turn / 45.0)
            
            return {
                'recent_momentum_shift': recent_shift,
                'momentum_volatility': volatility,
                'comeback_potential': comeback_potential,
                'snowball_risk': snowball_risk,
                'turning_point_proximity': turning_point,
                'momentum_consistency': consistency,
                'pressure_accumulation': pressure_accumulation,
                'strategic_patience': strategic_patience
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Momentum history error: {e}")
            return {k: 0.5 for k in ['recent_momentum_shift', 'momentum_volatility', 'comeback_potential', 'snowball_risk',
                                   'turning_point_proximity', 'momentum_consistency', 'pressure_accumulation', 'strategic_patience']}
    
    def _analyze_advanced_momentum(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze advanced momentum factors"""
        
        try:
            # Psychological advantage (based on position)
            my_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            psychological = min(my_hp / max(opp_hp, 0.1), 2.0) / 2.0
            
            # Adaptation momentum (successfully adapting)
            adaptation = 0.7 if turn > 10 else 0.4  # Better adaptation later
            
            # Information advantage
            my_revealed = sum(1 for p in my_team.values() if p.get('item', '') != 'unknown')
            opp_revealed = sum(1 for p in opp_team.values() if p.get('item', '') != 'unknown')
            info_advantage = 0.5 + (opp_revealed - my_revealed) * 0.1  # Want to know opponent's items
            
            # Execution quality
            execution = 0.8 if psychological > 0.6 else 0.4
            
            # Option density
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            option_density = min(my_alive / 6.0, 1.0)
            
            # Threat density
            threat_density = min((my_alive - 1) / 5.0, 1.0) if my_alive > 1 else 0.0
            
            # Control leverage
            control_leverage = psychological * option_density
            
            # Momentum amplification
            amplification = min(psychological + (turn / 50.0), 1.0)
            
            return {
                'psychological_advantage': min(psychological, 1.0),
                'adaptation_momentum': adaptation,
                'information_advantage': max(0.0, min(info_advantage, 1.0)),
                'execution_quality': execution,
                'option_density': option_density,
                'threat_density': threat_density,
                'control_leverage': min(control_leverage, 1.0),
                'momentum_amplification': amplification
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Advanced momentum error: {e}")
            return {k: 0.5 for k in ['psychological_advantage', 'adaptation_momentum', 'information_advantage', 'execution_quality',
                                   'option_density', 'threat_density', 'control_leverage', 'momentum_amplification']}
    
    def _encode_battle_phase_analysis(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode battle phase analysis (+16 dims)"""
        
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        field = snapshot.get('field', {})
        
        phase_data = self._analyze_battle_phases(my_team, opp_team, field, turn)
        
        # Phase identification (offset + 0 to offset + 3)
        state_vector[offset + 0] = phase_data['opening_phase_strength']    # 0-1: still in opening
        state_vector[offset + 1] = phase_data['midgame_phase_strength']    # 0-1: midgame dominance
        state_vector[offset + 2] = phase_data['endgame_phase_strength']    # 0-1: endgame positioning
        state_vector[offset + 3] = phase_data['transition_phase']          # 0-1: between phases
        
        # Phase-specific strategies (offset + 4 to offset + 7)
        state_vector[offset + 4] = phase_data['setup_phase_value']         # 0-1: value of setup strategies
        state_vector[offset + 5] = phase_data['aggression_phase_value']    # 0-1: value of aggressive plays
        state_vector[offset + 6] = phase_data['control_phase_value']       # 0-1: value of control strategies
        state_vector[offset + 7] = phase_data['preservation_phase_value']  # 0-1: value of preservation
        
        # Phase advantages (offset + 8 to offset + 11)
        state_vector[offset + 8] = phase_data['early_game_advantage']      # 0-1: early game strength
        state_vector[offset + 9] = phase_data['mid_game_advantage']        # 0-1: mid game strength
        state_vector[offset + 10] = phase_data['late_game_advantage']       # 0-1: late game strength
        state_vector[offset + 11] = phase_data['clutch_game_advantage']     # 0-1: clutch situation strength
        
        # Phase dynamics (offset + 12 to offset + 15)
        state_vector[offset + 12] = phase_data['phase_momentum']            # 0-1: momentum in current phase
        state_vector[offset + 13] = phase_data['phase_preparation']         # 0-1: preparation for next phase
        state_vector[offset + 14] = phase_data['phase_adaptability']        # 0-1: ability to adapt to phase changes
        state_vector[offset + 15] = phase_data['phase_timing_control']      # 0-1: control over phase timing
    
    def _analyze_battle_phases(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], 
                              field: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze battle phase characteristics"""
        
        try:
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            total_alive = my_alive + opp_alive
            
            # Phase strength calculation
            if turn <= 5:
                opening_strength = 1.0
                midgame_strength = 0.0
                endgame_strength = 0.0
            elif turn <= 15:
                opening_strength = max(0.0, 1.0 - (turn - 5) / 10.0)
                midgame_strength = min(1.0, (turn - 5) / 10.0)
                endgame_strength = 0.0
            elif total_alive > 6:
                opening_strength = 0.0
                midgame_strength = 1.0
                endgame_strength = 0.0
            else:
                opening_strength = 0.0
                midgame_strength = max(0.0, 1.0 - (8 - total_alive) / 4.0)
                endgame_strength = min(1.0, (8 - total_alive) / 4.0)
            
            transition_phase = 1.0 if turn in [5, 6, 15, 16] or total_alive in [6, 4] else 0.0
            
            # Phase-specific strategy values
            setup_value = max(0.2, 1.0 - turn / 20.0)  # Setup better early
            aggression_value = 0.8 if midgame_strength > 0.5 else 0.4
            control_value = min(1.0, turn / 15.0)  # Control better later
            preservation_value = endgame_strength
            
            # Phase advantages
            early_adv = 0.8 if my_alive >= opp_alive and turn <= 10 else 0.3
            mid_adv = 0.8 if my_alive > opp_alive and 10 < turn <= 25 else 0.4
            late_adv = 0.9 if my_alive > opp_alive and total_alive <= 4 else 0.2
            clutch_adv = 0.9 if my_alive >= opp_alive and total_alive <= 2 else 0.3
            
            # Phase dynamics
            my_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            phase_momentum = my_hp / max(my_hp + opp_hp, 0.1)
            
            phase_preparation = 0.7 if my_alive >= 3 else 0.3
            phase_adaptability = min(my_alive / 4.0, 1.0)
            timing_control = phase_momentum * (my_alive / 6.0)
            
            return {
                'opening_phase_strength': opening_strength,
                'midgame_phase_strength': midgame_strength,
                'endgame_phase_strength': endgame_strength,
                'transition_phase': transition_phase,
                'setup_phase_value': setup_value,
                'aggression_phase_value': aggression_value,
                'control_phase_value': control_value,
                'preservation_phase_value': preservation_value,
                'early_game_advantage': early_adv,
                'mid_game_advantage': mid_adv,
                'late_game_advantage': late_adv,
                'clutch_game_advantage': clutch_adv,
                'phase_momentum': phase_momentum,
                'phase_preparation': phase_preparation,
                'phase_adaptability': phase_adaptability,
                'phase_timing_control': min(timing_control, 1.0)
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Battle phase error: {e}")
            return {k: 0.5 for k in ['opening_phase_strength', 'midgame_phase_strength', 'endgame_phase_strength', 'transition_phase',
                                   'setup_phase_value', 'aggression_phase_value', 'control_phase_value', 'preservation_phase_value',
                                   'early_game_advantage', 'mid_game_advantage', 'late_game_advantage', 'clutch_game_advantage',
                                   'phase_momentum', 'phase_preparation', 'phase_adaptability', 'phase_timing_control']}
    
    def _encode_advanced_win_conditions(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode advanced win condition analysis (+16 dims)"""
        
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        field = snapshot.get('field', {})
        
        win_data = self._analyze_advanced_win_conditions(my_team, opp_team, field, turn)
        
        # Win condition types (offset + 0 to offset + 3)
        state_vector[offset + 0] = win_data['sweep_win_condition']         # 0-1: setup sweeper path
        state_vector[offset + 1] = win_data['attrition_win_condition']     # 0-1: attrition warfare path
        state_vector[offset + 2] = win_data['tempo_win_condition']         # 0-1: tempo/pressure path
        state_vector[offset + 3] = win_data['control_win_condition']       # 0-1: board control path
        
        # Win condition execution (offset + 4 to offset + 7)
        state_vector[offset + 4] = win_data['win_condition_progress']      # 0-1: progress toward win
        state_vector[offset + 5] = win_data['win_condition_clarity']       # 0-1: clarity of path
        state_vector[offset + 6] = win_data['win_condition_urgency']       # 0-1: urgency to execute
        state_vector[offset + 7] = win_data['win_condition_viability']     # 0-1: realistic achievement
        
        # Counter-conditions (offset + 8 to offset + 11)
        state_vector[offset + 8] = win_data['opponent_win_condition']      # 0-1: opponent's win path strength
        state_vector[offset + 9] = win_data['counter_strategy_strength']   # 0-1: our counter-strategy
        state_vector[offset + 10] = win_data['disruption_potential']       # 0-1: ability to disrupt opponent
        state_vector[offset + 11] = win_data['defensive_stability']        # 0-1: defensive stability
        
        # Advanced factors (offset + 12 to offset + 15)
        state_vector[offset + 12] = win_data['win_condition_redundancy']   # 0-1: multiple paths available
        state_vector[offset + 13] = win_data['execution_risk']             # 0-1: risk in executing plan
        state_vector[offset + 14] = win_data['time_factor']                # 0-1: time pressure on execution
        state_vector[offset + 15] = win_data['win_condition_synergy']      # 0-1: synergy between win paths
    
    def _analyze_advanced_win_conditions(self, my_team: Dict[str, Any], opp_team: Dict[str, Any], 
                                        field: Dict[str, Any], turn: int) -> Dict[str, float]:
        """Analyze advanced win condition dynamics"""
        
        try:
            my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
            opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
            
            my_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
            opp_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
            
            # Win condition types
            sweep_condition = 0.8 if my_alive >= 3 and turn <= 15 else 0.3
            attrition_condition = min(my_hp / max(opp_hp, 0.1), 2.0) / 2.0
            tempo_condition = 0.8 if my_alive > opp_alive else 0.3
            control_condition = 0.7 if turn > 20 and my_alive >= opp_alive else 0.4
            
            # Execution metrics
            progress = (my_hp - opp_hp + 1.0) / 2.0  # Normalize to 0-1
            clarity = 0.9 if abs(my_alive - opp_alive) >= 2 else 0.4
            urgency = min(turn / 30.0, 1.0)
            viability = min(my_alive / max(opp_alive, 1), 2.0) / 2.0
            
            # Counter-conditions
            opp_win_condition = opp_hp / max(my_hp, 0.1) if opp_hp > my_hp else 0.3
            counter_strength = max(0.2, 1.0 - opp_win_condition)
            disruption = min(my_alive / 4.0, 1.0)
            defensive_stability = min(my_hp / 3.0, 1.0)
            
            # Advanced factors
            redundancy = min(my_alive / 3.0, 1.0)
            execution_risk = max(0.1, 1.0 - viability)
            time_factor = min(turn / 40.0, 1.0)
            synergy = (sweep_condition + attrition_condition + tempo_condition) / 3.0
            
            return {
                'sweep_win_condition': min(sweep_condition, 1.0),
                'attrition_win_condition': min(attrition_condition, 1.0),
                'tempo_win_condition': min(tempo_condition, 1.0),
                'control_win_condition': min(control_condition, 1.0),
                'win_condition_progress': max(0.0, min(progress, 1.0)),
                'win_condition_clarity': clarity,
                'win_condition_urgency': urgency,
                'win_condition_viability': min(viability, 1.0),
                'opponent_win_condition': min(opp_win_condition, 1.0),
                'counter_strategy_strength': counter_strength,
                'disruption_potential': disruption,
                'defensive_stability': defensive_stability,
                'win_condition_redundancy': redundancy,
                'execution_risk': execution_risk,
                'time_factor': time_factor,
                'win_condition_synergy': min(synergy, 1.0)
            }
        
        except Exception as e:
            logging.debug(f"[Neural Encoder] Advanced win conditions error: {e}")
            return {k: 0.5 for k in ['sweep_win_condition', 'attrition_win_condition', 'tempo_win_condition', 'control_win_condition',
                                   'win_condition_progress', 'win_condition_clarity', 'win_condition_urgency', 'win_condition_viability',
                                   'opponent_win_condition', 'counter_strategy_strength', 'disruption_potential', 'defensive_stability',
                                   'win_condition_redundancy', 'execution_risk', 'time_factor', 'win_condition_synergy']}
    
    def _encode_game_state_from_snapshot(self, snapshot: Dict[str, Any], state_vector: torch.Tensor, offset: int):
        """Encode general game state from snapshot data"""
        
        # Turn number (normalized)
        turn = snapshot.get('turn', 1)
        if turn is None:
            turn = 1
        state_vector[offset] = min(turn / 50.0, 1.0)  # Normalize to 50 turns max
        
        # Battle format encoding
        battle_format = snapshot.get('format', '')
        if battle_format:
            format_encoding = {
                'gen9randombattle': 1, 'gen9ou': 2, 'gen9ubers': 3, 'gen9uu': 4,
                'gen9ru': 5, 'gen9nu': 6, 'gen9pu': 7, 'gen9lc': 8, 'gen9monotype': 9,
                'gen9doublesou': 10, 'gen9randomdoublesbattle': 11
            }
            if battle_format in format_encoding:
                state_vector[offset + 1] = format_encoding[battle_format] / 11.0
        
        # Game phase based on turn
        if turn:
            if turn <= 5:
                state_vector[offset + 2] = 1.0  # Opening
            elif turn <= 15:
                state_vector[offset + 3] = 1.0  # Early game
            elif turn <= 30:
                state_vector[offset + 4] = 1.0  # Mid game
            else:
                state_vector[offset + 5] = 1.0  # Late game
        
        # Active Pokemon count (health-based urgency)
        my_team = snapshot.get('my_team', {})
        opp_team = snapshot.get('opp_team', {})
        
        my_alive = sum(1 for p in my_team.values() if (p.get('hp_fraction') or 0) > 0)
        opp_alive = sum(1 for p in opp_team.values() if (p.get('hp_fraction') or 0) > 0)
        
        state_vector[offset + 6] = my_alive / 6.0  # My team health ratio
        state_vector[offset + 7] = opp_alive / 6.0  # Opponent team health ratio
        
        # Team health summary
        my_total_hp = sum((p.get('hp_fraction') or 0) for p in my_team.values())
        opp_total_hp = sum((p.get('hp_fraction') or 0) for p in opp_team.values())
        
        state_vector[offset + 8] = my_total_hp / 6.0  # Total HP ratio
        state_vector[offset + 9] = opp_total_hp / 6.0
        
        # HP advantage
        hp_advantage = (my_total_hp - opp_total_hp) / 6.0
        state_vector[offset + 10] = (hp_advantage + 1.0) / 2.0  # Normalize to 0-1
        
        # Active Pokemon health
        my_active = next((p for p in my_team.values() if p.get('is_active')), None)
        opp_active = next((p for p in opp_team.values() if p.get('is_active')), None)
        
        if my_active:
            state_vector[offset + 11] = my_active.get('hp_fraction') or 0
        if opp_active:
            state_vector[offset + 12] = opp_active.get('hp_fraction') or 0
        
        # Battle tag encoding (for unique battle identification)
        battle_tag = snapshot.get('battle_tag', '')
        if battle_tag:
            # Simple hash-based encoding for battle uniqueness
            import hashlib
            tag_hash = int(hashlib.md5(battle_tag.encode()).hexdigest()[:8], 16)
            state_vector[offset + 13] = (tag_hash % 10000) / 10000.0
        
        # Remaining slots for additional game state features
        # offset + 14 to offset + 19 available for future expansion

class ValueNetwork(nn.Module):
    """Neural network for position evaluation"""
    
    def __init__(self, input_size: int = 390, hidden_size: int = 512):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Main value network
        self.value_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_net(state)

class PolicyNetwork(nn.Module):
    """Neural network for move prior prediction"""
    
    def __init__(self, input_size: int = 390, max_actions: int = 16, hidden_size: int = 512):
        super().__init__()
        
        self.input_size = input_size
        self.max_actions = max_actions
        self.hidden_size = hidden_size
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_actions)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.policy_net(state)
        
        if action_mask is not None:
            # Mask invalid actions
            logits = logits.masked_fill(~action_mask, -float('inf'))
        
        return F.softmax(logits, dim=-1)

class NeuralMCTSModel:
    """Enhanced MCTS with neural network value and policy functions"""
    
    def __init__(self, base_mcts_model, device: str = 'cpu'):
        self.base_mcts = base_mcts_model
        self.device = device
        
        # Neural network components
        self.encoder = PokemonStateEncoder()
        self.value_net = ValueNetwork(input_size=830).to(device)  # Use 830 dimensions (Phase 5)
        self.policy_net = PolicyNetwork(input_size=830).to(device)  # Use 830 dimensions (Phase 5)
        
        # Training components
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Performance tracking
        self.neural_evaluations = 0
        self.fallback_evaluations = 0
        
        # Try to load pre-trained weights
        self._load_networks()

    def policy_value_for_candidates(self, combined_state: Any, candidate_keys: List[str]) -> Tuple[Dict[str, float], float]:
        """Return (policy over candidates, value in [-1,1]) for a CombinedState.

        The policy head has a fixed size; we map the first N candidates into it and
        apply a boolean mask so logits for unused slots are ignored.
        """
        try:
            x = self.encoder.encode_from_combined_state(combined_state).to(self.device)
            x = x.float().unsqueeze(0)
            max_actions = self.policy_net.max_actions
            n = min(len(candidate_keys), max_actions)
            mask = torch.zeros((1, max_actions), dtype=torch.bool, device=self.device)
            if n > 0:
                mask[0, :n] = True
            with torch.no_grad():
                probs_full = self.policy_net(x, action_mask=mask)[0].cpu().numpy().tolist()
                value = float(self.value_net(x)[0].item())
            # Extract and renormalize first n probabilities
            raw = probs_full[:n]
            s = sum(raw) or 1.0
            raw = [p / s for p in raw]
            pol = {candidate_keys[i]: float(raw[i]) for i in range(n)}
            self.neural_evaluations += 1
            return pol, value
        except Exception as e:
            logging.debug(f"[Neural MCTS] policy_value_for_candidates failed: {e}")
            self.fallback_evaluations += 1
            n = len(candidate_keys)
            if n:
                u = 1.0 / n
                return {k: u for k in candidate_keys}, 0.0
            return {}, 0.0

    def value_only(self, combined_state: Any) -> float:
        """Return value estimate in [-1,1] for a CombinedState."""
        try:
            x = self.encoder.encode_from_combined_state(combined_state).to(self.device)
            x = x.float().unsqueeze(0)
            with torch.no_grad():
                return float(self.value_net(x)[0].item())
        except Exception as e:
            logging.debug(f"[Neural MCTS] value_only failed: {e}")
            return 0.0
    
    def choose_action(self, battle: Any) -> Any:
        """Enhanced action selection using neural-guided MCTS"""
        
        import time
        start_time = time.time()
        
        try:
            # Ensure networks are in evaluation mode
            self.value_net.eval()
            self.policy_net.eval()
            
            # Encode state for neural networks
            state_tensor = self.encoder.encode_battle_state(battle).to(self.device)
            
            # Get neural network predictions
            with torch.no_grad():
                value_pred = self.value_net(state_tensor.unsqueeze(0))
                policy_pred = self.policy_net(state_tensor.unsqueeze(0))
            
            # Enhanced MCTS with neural priors
            result = self._neural_mcts_search(battle, state_tensor, value_pred, policy_pred)
            
            decision_time = time.time() - start_time
            logging.info(f"[Neural MCTS] Decision completed in {decision_time:.3f}s")
            
            return result
            
        except Exception as e:
            decision_time = time.time() - start_time
            logging.warning(f"[Neural MCTS] Error: {e}, falling back to base MCTS (time: {decision_time:.3f}s)")
            self.fallback_evaluations += 1
            return self.base_mcts.choose_action(battle)
    
    def _neural_mcts_search(self, battle: Any, state_tensor: torch.Tensor, 
                           value_pred: torch.Tensor, policy_pred: torch.Tensor) -> Any:
        """Run MCTS with neural network guidance"""
        
        # Get available actions
        actions = self._get_available_actions(battle)
        
        if not actions:
            return self.base_mcts.choose_action(battle)
        
        # Create action mask for policy network
        action_mask = torch.zeros(self.policy_net.max_actions, dtype=torch.bool)
        action_indices = []
        
        for i, action in enumerate(actions[:self.policy_net.max_actions]):
            action_mask[i] = True
            action_indices.append(i)
        
        # Get policy priors for available actions
        with torch.no_grad():
            policy_masked = self.policy_net(state_tensor.unsqueeze(0), action_mask.unsqueeze(0))
            action_priors = policy_masked[0][:len(actions)].cpu().numpy()
        
        # Run enhanced MCTS with neural priors
        action_values = {}
        action_visits = {}
        
        for i, action in enumerate(actions):
            prior = action_priors[i] if i < len(action_priors) else 1.0 / len(actions)
            action_values[action] = prior
            action_visits[action] = 0
        
        # MCTS simulations with neural guidance
        # Use intelligent simulation count: minimum 100, maximum 1000, or 1/3 of base MCTS
        neural_simulations = max(100, min(1000, self.base_mcts.simulations // 3))
        logging.debug(f"[Neural MCTS] Running {neural_simulations} simulations (base: {self.base_mcts.simulations})")
        
        for _ in range(neural_simulations):
            selected_action = self._select_action_neural(actions, action_values, action_visits, action_priors)
            
            # Simulate action outcome
            value = self._simulate_action_outcome(battle, selected_action, state_tensor)
            
            # Update statistics
            action_visits[selected_action] += 1
            action_values[selected_action] += (value - action_values[selected_action]) / action_visits[selected_action]
        
        # Select best action
        best_action = max(actions, key=lambda a: action_visits.get(a, 0))
        
        logging.info(f"[Neural MCTS] Selected: {best_action}")
        logging.info(f"[Neural MCTS] Values: {dict(action_values)}")
        logging.info(f"[Neural MCTS] Prior position value: {value_pred.detach().item():.3f}")
        
        # Collect training data if available
        try:
            if hasattr(self.base_mcts, '_training_collector') and self.base_mcts._training_collector:
                # Convert neural policy predictions to action scores
                neural_policy_scores = {}
                for i, action in enumerate(actions):
                    if i < len(action_priors):
                        neural_policy_scores[action] = float(action_priors[i])
                
                # Clean and validate data before passing to collector
                cleaned_mcts_scores = {}
                for action, value in action_values.items():
                    try:
                        cleaned_mcts_scores[str(action)] = float(value)
                    except (ValueError, TypeError):
                        cleaned_mcts_scores[str(action)] = 0.0
                
                # Collect the position for training
                self.base_mcts._training_collector.collect_position(
                    battle=battle,
                    mcts_scores=cleaned_mcts_scores,  # Cleaned MCTS values after search
                    chosen_action=str(best_action),
                    neural_policy_scores=neural_policy_scores,  # Neural priors
                    neural_value_prediction=float(value_pred.detach().item()),  # Neural value
                    is_critical=False  # Could enhance this logic later
                )
                logging.debug(f"[Neural MCTS] Training data collected for position")
        except Exception as e:
            logging.debug(f"[Neural MCTS] Training data collection failed: {e}")
        
        return self._convert_action_to_move(best_action, battle)
    
    def _select_action_neural(self, actions: List[str], values: Dict[str, float], 
                             visits: Dict[str, int], priors: np.ndarray) -> str:
        """UCB selection with neural priors"""
        
        total_visits = sum(visits.values()) + 1
        sqrt_total = np.sqrt(total_visits)
        
        best_score = -float('inf')
        best_action = actions[0]
        
        for i, action in enumerate(actions):
            if visits[action] == 0:
                return action  # Unvisited nodes
            
            prior = priors[i] if i < len(priors) else 1.0 / len(actions)
            
            # UCB with neural prior
            q_value = values[action]
            u_value = self.base_mcts.c_puct * prior * sqrt_total / (1 + visits[action])
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _simulate_action_outcome(self, battle: Any, action: str, state_tensor: torch.Tensor) -> float:
        """Simulate outcome of an action using neural network evaluation"""
        
        try:
            # Quick simulation or neural evaluation
            # For now, use neural value function with some noise for exploration
            with torch.no_grad():
                base_value = self.value_net(state_tensor.unsqueeze(0)).item()
            
            # Add action-specific adjustment (simplified)
            action_bonus = 0.1 * np.random.normal()  # Exploration noise
            
            self.neural_evaluations += 1
            result = base_value + action_bonus
            
            # Validate result
            import math
            if math.isnan(result) or math.isinf(result):
                logging.warning(f"[Neural MCTS] Invalid simulation result: {result}, using 0.0")
                return 0.0
            
            return result
            
        except Exception as e:
            logging.warning(f"[Neural MCTS] Simulation error: {e}")
            return 0.0
    
    def _get_available_actions(self, battle: Any) -> List[str]:
        """Get list of available actions"""
        actions = []
        
        # Moves
        if hasattr(battle, 'available_moves'):
            for move in battle.available_moves:
                move_id = getattr(move, 'id', str(move))
                actions.append(f"move:{move_id}")
        
        # Switches
        if hasattr(battle, 'available_switches'):
            for pokemon in battle.available_switches:
                species = getattr(pokemon, 'species', str(pokemon))
                actions.append(f"switch:{species}")
        
        return actions
    
    def _convert_action_to_move(self, action: str, battle: Any) -> Any:
        """Convert action string back to move object"""
        if action.startswith("move:"):
            move_id = action.replace("move:", "")
            if hasattr(battle, 'available_moves'):
                for move in battle.available_moves:
                    if getattr(move, 'id', str(move)) == move_id:
                        return move
        elif action.startswith("switch:"):
            species = action.replace("switch:", "")
            if hasattr(battle, 'available_switches'):
                for pokemon in battle.available_switches:
                    if getattr(pokemon, 'species', str(pokemon)) == species:
                        return pokemon
        
        # Fallback
        return self.base_mcts.choose_action(battle)
    
    def _load_networks(self):
        """Load pre-trained network weights if available"""
        models_dir = Path("Models")
        
        try:
            # Prefer best checkpoints if available
            value_best = models_dir / "value_network_best.pth"
            value_path = models_dir / "value_network.pth"
            load_value = value_best if value_best.exists() else value_path
            if load_value.exists():
                try:
                    self.value_net.load_state_dict(torch.load(load_value, map_location=self.device), strict=False)
                except Exception as e:
                    logging.warning(f"[Neural MCTS] Value load mismatch: {e}; continuing with random init")
                logging.info(f"[Neural MCTS] Loaded value network weights from {load_value.name}")
            
            policy_best = models_dir / "policy_network_best.pth"
            policy_path = models_dir / "policy_network.pth"
            load_policy = policy_best if policy_best.exists() else policy_path
            if load_policy.exists():
                try:
                    self.policy_net.load_state_dict(torch.load(load_policy, map_location=self.device), strict=False)
                except Exception as e:
                    logging.warning(f"[Neural MCTS] Policy load mismatch: {e}; continuing with random init")
                logging.info(f"[Neural MCTS] Loaded policy network weights from {load_policy.name}")
                
        except Exception as e:
            logging.warning(f"[Neural MCTS] Failed to load networks: {e}")
    
    def save_networks(self):
        """Save network weights"""
        models_dir = Path("Models")
        models_dir.mkdir(exist_ok=True)
        
        try:
            torch.save(self.value_net.state_dict(), models_dir / "value_network.pth")
            torch.save(self.policy_net.state_dict(), models_dir / "policy_network.pth")
            logging.info("[Neural MCTS] Saved network weights")
        except Exception as e:
            logging.error(f"[Neural MCTS] Failed to save networks: {e}")
    
    def reload_networks(self):
        """Reload network weights from disk (for live updates)"""
        logging.info("[Neural MCTS] Reloading networks...")
        old_neural_evals = self.neural_evaluations
        old_fallback_evals = self.fallback_evaluations
        
        # Clear performance counters for fresh start
        self.neural_evaluations = 0
        self.fallback_evaluations = 0
        
        # Reload networks
        self._load_networks()
        
        logging.info(f"[Neural MCTS] Network reload complete. Previous performance: {old_neural_evals} neural, {old_fallback_evals} fallback")
        return True
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about loaded networks"""
        models_dir = Path("Models")
        
        value_path = models_dir / "value_network.pth"
        policy_path = models_dir / "policy_network.pth"
        
        info = {
            'value_network_exists': value_path.exists(),
            'policy_network_exists': policy_path.exists(),
            'value_network_size': value_path.stat().st_size if value_path.exists() else 0,
            'policy_network_size': policy_path.stat().st_size if policy_path.exists() else 0,
            'neural_evaluations': self.neural_evaluations,
            'fallback_evaluations': self.fallback_evaluations
        }
        
        if value_path.exists():
            import os
            info['value_network_modified'] = os.path.getmtime(value_path)
        if policy_path.exists():
            import os
            info['policy_network_modified'] = os.path.getmtime(policy_path)
            
        return info
    
    def train_step(self, state_batch: torch.Tensor, value_targets: torch.Tensor, 
                   policy_targets: torch.Tensor, action_masks: torch.Tensor):
        """Single training step for both networks"""
        
        # Value network training
        self.optimizer_value.zero_grad()
        value_pred = self.value_net(state_batch)
        value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
        value_loss.backward()
        self.optimizer_value.step()
        
        # Policy network training
        self.optimizer_policy.zero_grad()
        policy_pred = self.policy_net(state_batch, action_masks)
        policy_loss = F.cross_entropy(policy_pred, policy_targets)
        policy_loss.backward()
        self.optimizer_policy.step()
        
        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
