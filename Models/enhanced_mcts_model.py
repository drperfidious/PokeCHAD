# Enhanced MCTS Model Integration
# Combines ISMCTS, Neural Networks, Enhanced LLM, and Training Infrastructure

import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
from pathlib import Path

# Import only standalone components
try:
    from .standalone_integration import create_optimized_config
    from .standalone_neural_mcts import StandaloneNeuralMCTS
except ImportError:
    # Fallback for direct execution
    from standalone_integration import create_optimized_config
    from standalone_neural_mcts import StandaloneNeuralMCTS

class EnhancedMCTSModel:
    """
    Standalone Neural MCTS model:
    - Clean standalone Neural MCTS implementation
    - No legacy dependencies or confusion
    - Simplified decision making
    """
    
    def __init__(self, base_mcts_model, config: Optional[Dict[str, Any]] = None):
        self.base_mcts = base_mcts_model
        self.config = config or self._get_default_config()
        
        # Initialize standalone components
        self._initialize_standalone()
        
        # Performance tracking (simplified)
        self.performance_stats = {
            'total_decisions': 0,
            'neural_usage': 0,
            'average_decision_time': 0.0
        }
        
        # Always neural mode for standalone
        self.decision_mode = 'neural'
        
        logging.info("[Enhanced MCTS] Initialized with standalone Neural MCTS")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'use_standalone_mcts': True,
            'neural_network_device': 'cpu',
            'decision_mode': 'neural',
            'training': {
                'enable_training': False,
                'save_training_data': True,
                'training_frequency': 10
            }
        }
    
    def _initialize_standalone(self):
        """Initialize standalone Neural MCTS only"""
        
        # Standalone Neural MCTS - the only component we need
        try:
            device = self.config.get('neural_network_device', 'cpu')
            standalone_config = create_optimized_config(device=device, fast_mode=False)
            self.standalone_mcts = StandaloneNeuralMCTS(standalone_config)
            logging.info(f"[Enhanced MCTS] Standalone Neural MCTS initialized on {device}")
        except Exception as e:
            logging.error(f"[Enhanced MCTS] Standalone MCTS initialization failed: {e}")
            self.standalone_mcts = None
            raise RuntimeError(f"Cannot initialize standalone MCTS: {e}")
        
        # No legacy components - clean and simple
    
    def choose_action(self, battle: Any) -> Any:
        """Standalone Neural MCTS action selection"""
        
        import time
        start_time = time.time()
        battle_id = getattr(battle, 'battle_tag', 'unknown')
        
        logging.info(f"[Standalone MCTS] choose_action called for battle {battle_id}")
        
        self.performance_stats['total_decisions'] += 1
        self.performance_stats['neural_usage'] += 1
        
        try:
            # Use standalone Neural MCTS - simple and clean
            if not self.standalone_mcts:
                raise RuntimeError("Standalone MCTS not initialized")
                
            decision = self.standalone_mcts.choose_action(battle)
            
            # Update performance statistics
            decision_time = time.time() - start_time
            self._update_performance_stats(decision_time)
            
            logging.info(f"[Standalone MCTS] Decision made in {decision_time:.3f}s - Action: {type(decision).__name__}")
            if hasattr(decision, 'id'):
                logging.info(f"[Standalone MCTS] Action ID: {decision.id}")
            elif hasattr(decision, 'species'):
                logging.info(f"[Standalone MCTS] Switch to: {decision.species}")
            
            return decision
            
        except Exception as e:
            import traceback
            logging.error(f"[Standalone MCTS] Error in decision: {e}")
            logging.error(f"[Standalone MCTS] Traceback: {traceback.format_exc()}")
            # Fallback to base MCTS
            fallback_decision = self.base_mcts.choose_action(battle)
            logging.warning(f"[Standalone MCTS] Using base MCTS fallback: {type(fallback_decision).__name__}")
            return fallback_decision
    
    def _update_performance_stats(self, decision_time: float):
        """Update performance statistics"""
        
        total_decisions = self.performance_stats['total_decisions']
        current_avg_time = self.performance_stats['average_decision_time']
        
        # Update average decision time
        self.performance_stats['average_decision_time'] = (
            (current_avg_time * (total_decisions - 1) + decision_time) / total_decisions
        )
    
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for standalone system"""
        
        stats = self.performance_stats.copy()
        
        # Calculate usage percentages
        total = stats['total_decisions']
        if total > 0:
            stats['neural_usage_percent'] = stats['neural_usage'] / total
        
        # Add standalone MCTS stats
        if self.standalone_mcts:
            stats['standalone_stats'] = self.standalone_mcts.get_performance_stats()
        
        return stats
    
    def save_performance_data(self, filepath: str = "logs/standalone_mcts_performance.json"):
        """Save performance data to file"""
        
        try:
            import json
            
            performance_data = {
                'performance_stats': self.get_performance_report(),
                'config': self.config,
                'component_status': {
                    'standalone_mcts_available': self.standalone_mcts is not None
                }
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logging.info(f"[Standalone MCTS] Saved performance data to {filepath}")
            
        except Exception as e:
            logging.error(f"[Standalone MCTS] Failed to save performance data: {e}")
    
    def set_decision_mode(self, mode: str):
        """Set decision mode (always neural for standalone)"""
        self.decision_mode = 'neural'
        logging.info(f"[Standalone MCTS] Decision mode is always neural")
    
    def get_available_enhancements(self) -> Dict[str, bool]:
        """Get status of available enhancements"""
        
        return {
            'standalone_mcts': self.standalone_mcts is not None,
            'neural_networks': True,
            'gumbel_mcts': True,
            'value_prefix_learning': True,
            'self_supervised_learning': True
        }


# Factory function for easy integration
def create_enhanced_mcts_model(base_mcts_model, config: Optional[Dict[str, Any]] = None) -> EnhancedMCTSModel:
    """
    Factory function to create a standalone Neural MCTS model
    
    Args:
        base_mcts_model: Your existing MCTS model (used only for fallback)
        config: Optional configuration dictionary
        
    Returns:
        EnhancedMCTSModel with standalone Neural MCTS
    """
    
    return EnhancedMCTSModel(base_mcts_model, config)