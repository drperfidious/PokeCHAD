# Neural Network Training System
# Trains value and policy networks using collected game data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from Models.training_data_collector import TrainingGame, TrainingPosition, get_training_collector
from Models.neural_mcts import ValueNetwork, PolicyNetwork, PokemonStateEncoder

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    validation_split: float = 0.2
    device: str = 'cpu'
    save_checkpoints: bool = True
    checkpoint_interval: int = 5
    early_stopping_patience: int = 10
    min_games_for_training: int = 10

class PokemonTrainingDataset(Dataset):
    """PyTorch dataset for Pokemon battle training data"""
    
    def __init__(self, training_games: List[TrainingGame], 
                 state_encoder: PokemonStateEncoder, device: str = 'cpu'):
        self.games = training_games
        self.encoder = state_encoder
        self.device = device
        
        # Extract all positions from all games
        self.positions = []
        for game in training_games:
            if game.positions:
                self.positions.extend(game.positions)
        
        logging.info(f"[Training Dataset] Created dataset with {len(self.positions)} positions from {len(training_games)} games")
        
        # Log training data format information
        self._log_data_format_info()
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        position = self.positions[idx]
        
        # State encoding
        state = torch.tensor(position.position_encoding, dtype=torch.float32)
        
        # Value target (game outcome from this position)
        value_target = float(position.position_value if position.position_value is not None else 0.0)
        
        # Policy target (convert MCTS scores to probabilities)
        policy_target, action_mask = self._create_policy_target(position)
        
        return state, value_target, policy_target, action_mask
    
    def _create_policy_target(self, position: TrainingPosition) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create policy target and action mask from MCTS scores"""
        
        max_actions = 10  # Match policy network output size
        policy_target = torch.zeros(max_actions, dtype=torch.float32)
        action_mask = torch.zeros(max_actions, dtype=torch.bool)
        
        # Use best available scores: Neural > LLM > MCTS
        if hasattr(position, 'neural_policy_scores') and position.neural_policy_scores:
            scores = position.neural_policy_scores
            data_source = "neural"
        elif position.llm_action_scores:
            scores = position.llm_action_scores
            data_source = "llm"
        else:
            scores = position.mcts_action_scores
            data_source = "mcts"
        
        # Log data source for debugging (only occasionally to avoid spam)
        if hasattr(position, 'turn') and position.turn % 10 == 1:
            logging.debug(f"[Training] Using {data_source} scores for turn {position.turn}")
        
        # Use available_actions if present (enhanced data format)
        if hasattr(position, 'available_actions') and position.available_actions:
            # Filter scores to only include actually available actions
            available_set = set(position.available_actions)
            filtered_scores = {action: score for action, score in scores.items() 
                             if action in available_set}
            if filtered_scores:
                scores = filtered_scores
        
        if scores:
            # Convert scores to probabilities using softmax
            actions = list(scores.keys())[:max_actions]  # Take first max_actions
            score_values = [scores[action] for action in actions]
            
            if score_values:
                # Apply softmax to convert scores to probabilities
                score_tensor = torch.tensor(score_values, dtype=torch.float32)
                probabilities = torch.softmax(score_tensor, dim=0)
                
                # Fill policy target
                for i, prob in enumerate(probabilities):
                    if i < max_actions:
                        policy_target[i] = prob
                        action_mask[i] = True
        
        # Ensure at least one action is valid
        if not action_mask.any():
            action_mask[0] = True
            policy_target[0] = 1.0
        
        return policy_target, action_mask
    
    def _log_data_format_info(self):
        """Log information about the training data format"""
        if not self.positions:
            return
            
        # Check first position to see what format we have
        first_pos = self.positions[0]
        
        # Check for enhanced format features
        has_snapshot = hasattr(first_pos, 'battle_snapshot')
        has_available_actions = hasattr(first_pos, 'available_actions')
        
        format_type = "Enhanced" if (has_snapshot and has_available_actions) else "Basic"
        
        logging.info(f"[Training Dataset] Data format: {format_type}")
        logging.info(f"[Training Dataset] Features: snapshot={has_snapshot}, available_actions={has_available_actions}")
        
        # Log encoding statistics
        encoding_lengths = [len(pos.position_encoding) for pos in self.positions[:10]]
        if encoding_lengths:
            avg_length = sum(encoding_lengths) / len(encoding_lengths)
            logging.info(f"[Training Dataset] Position encoding length: {avg_length:.0f} features")
            
            # Check feature richness
            non_zero_counts = [sum(1 for x in pos.position_encoding if x != 0.0) for pos in self.positions[:10]]
            if non_zero_counts:
                avg_non_zero = sum(non_zero_counts) / len(non_zero_counts)
                logging.info(f"[Training Dataset] Average non-zero features: {avg_non_zero:.0f}")

class NeuralNetworkTrainer:
    """Trains neural networks using collected battle data"""
    
    def __init__(self, config: Optional[TrainingConfig] = None, trial_id: Optional[str] = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.trial_id = trial_id
        
        # Initialize networks with correct dimensions
        self.value_network = ValueNetwork(input_size=830).to(self.device)  # Use Phase 5 dimensions
        self.policy_network = PolicyNetwork(input_size=830).to(self.device)  # Use Phase 5 dimensions
        self.state_encoder = PokemonStateEncoder()
        
        # Initialize optimizers
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.learning_rate)
        
        # Training state
        self.training_history = {
            'value_losses': [],
            'policy_losses': [],
            'validation_losses': [],
            'epochs_completed': 0
        }
        
        # Paths
        self.checkpoint_dir = Path("training_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logging.info(f"[Neural Trainer] Initialized on device: {self.device}")
    
    def train(self, max_games: Optional[int] = None) -> Dict[str, Any]:
        """Train neural networks on collected data"""
        
        # Load training data
        collector = get_training_collector()
        training_games = collector.load_training_data(max_games)
        
        if len(training_games) < self.config.min_games_for_training:
            error_msg = f"Need at least {self.config.min_games_for_training} games for training, got {len(training_games)}"
            logging.error(f"[Neural Trainer] {error_msg}")
            return {'error': error_msg}
        
        # Create dataset and dataloader
        dataset = PokemonTrainingDataset(training_games, self.state_encoder, str(self.device))
        
        # Split into train/validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.config.validation_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        logging.info(f"[Neural Trainer] Starting training: {train_size} train, {val_size} validation")
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_value_loss, train_policy_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_value_loss, val_policy_loss = self._validate_epoch(val_loader)
            val_total_loss = val_value_loss + val_policy_loss
            
            # Record history
            self.training_history['value_losses'].append(train_value_loss)
            self.training_history['policy_losses'].append(train_policy_loss)
            self.training_history['validation_losses'].append(val_total_loss)
            self.training_history['epochs_completed'] += 1
            
            epoch_time = time.time() - epoch_start
            
            logging.info(f"[Neural Trainer] Epoch {epoch + 1}/{self.config.epochs} ({epoch_time:.1f}s)")
            logging.info(f"  Train - Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}")
            logging.info(f"  Val   - Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}")
            
            # Save live progress for UI
            self._save_live_progress(epoch + 1, train_value_loss, train_policy_loss, val_value_loss, val_policy_loss)
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"[Neural Trainer] Early stopping after {epoch + 1} epochs")
                    break
        
        total_time = time.time() - start_time
        
        # Save final model
        self._save_final_model()
        
        # Training summary
        summary = {
            'epochs_completed': self.training_history['epochs_completed'],
            'final_value_loss': self.training_history['value_losses'][-1] if self.training_history['value_losses'] else 0,
            'final_policy_loss': self.training_history['policy_losses'][-1] if self.training_history['policy_losses'] else 0,
            'best_validation_loss': best_val_loss,
            'training_time_seconds': total_time,
            'games_used': len(training_games),
            'positions_used': len(dataset),
            'device_used': str(self.device)
        }
        
        logging.info(f"[Neural Trainer] Training completed: {summary}")
        return summary
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.value_network.train()
        self.policy_network.train()
        
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        total_batches = len(train_loader)
        for batch_idx, (states, value_targets, policy_targets, action_masks) in enumerate(train_loader):
            states = states.to(self.device)
            value_targets = value_targets.float().to(self.device)
            policy_targets = policy_targets.to(self.device)
            action_masks = action_masks.to(self.device)
            
            # Value network training
            self.value_optimizer.zero_grad()
            value_predictions = self.value_network(states).squeeze()
            value_loss = nn.MSELoss()(value_predictions, value_targets)
            value_loss.backward()
            self.value_optimizer.step()
            
            # Policy network training
            self.policy_optimizer.zero_grad()
            policy_predictions = self.policy_network(states, action_masks)
            policy_loss = nn.CrossEntropyLoss()(policy_predictions, policy_targets)
            policy_loss.backward()
            self.policy_optimizer.step()
            
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
            
            # Progress logging every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                current_val_loss = total_value_loss / num_batches
                current_pol_loss = total_policy_loss / num_batches
                logging.info(f"[Neural Trainer] Batch {batch_idx + 1}/{total_batches} - Value: {current_val_loss:.4f}, Policy: {current_pol_loss:.4f}")
        
        return total_value_loss / num_batches, total_policy_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        
        self.value_network.eval()
        self.policy_network.eval()
        
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, value_targets, policy_targets, action_masks in val_loader:
                states = states.to(self.device)
                value_targets = value_targets.float().to(self.device)
                policy_targets = policy_targets.to(self.device)
                action_masks = action_masks.to(self.device)
                
                # Value network validation
                value_predictions = self.value_network(states).squeeze()
                value_loss = nn.MSELoss()(value_predictions, value_targets)
                
                # Policy network validation
                policy_predictions = self.policy_network(states, action_masks)
                policy_loss = nn.CrossEntropyLoss()(policy_predictions, policy_targets)
                
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1
        
        return total_value_loss / num_batches, total_policy_loss / num_batches
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'value_network_state': self.value_network.state_dict(),
            'policy_network_state': self.policy_network.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"[Neural Trainer] Saved checkpoint: {checkpoint_path}")
    
    def _save_best_model(self):
        """Save best model so far"""
        models_dir = Path("Models")
        models_dir.mkdir(exist_ok=True)
        
        # Use trial_id suffix if provided (for hyperparameter search)
        suffix = f"_{self.trial_id}" if self.trial_id else "_best"
        torch.save(self.value_network.state_dict(), models_dir / f"value_network{suffix}.pth")
        torch.save(self.policy_network.state_dict(), models_dir / f"policy_network{suffix}.pth")
    
    def _save_final_model(self):
        """Save final trained model"""
        models_dir = Path("Models")
        models_dir.mkdir(exist_ok=True)
        
        # Use trial_id suffix if provided (for hyperparameter search)
        suffix = f"_{self.trial_id}" if self.trial_id else ""
        torch.save(self.value_network.state_dict(), models_dir / f"value_network{suffix}.pth")
        torch.save(self.policy_network.state_dict(), models_dir / f"policy_network{suffix}.pth")
        
        # Save training history
        history_path = models_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logging.info("[Neural Trainer] Saved final trained models")
    
    def _save_live_progress(self, epoch: int, train_val_loss: float, train_pol_loss: float, 
                           val_val_loss: float, val_pol_loss: float):
        """Save live training progress for UI monitoring"""
        try:
            progress_data = {
                'training_active': True,
                'current_epoch': epoch,
                'total_epochs': self.config.epochs,
                'train_value_loss': train_val_loss,
                'train_policy_loss': train_pol_loss,
                'val_value_loss': val_val_loss,
                'val_policy_loss': val_pol_loss,
                'epochs_completed': len(self.training_history['value_losses']),
                'timestamp': time.time()
            }
            
            progress_path = Path("Models/training_progress.json")
            progress_path.parent.mkdir(exist_ok=True)
            
            with open(progress_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"[Neural Trainer] Failed to save live progress: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.value_network.load_state_dict(checkpoint['value_network_state'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.training_history = checkpoint['training_history']
        
        logging.info(f"[Neural Trainer] Loaded checkpoint: {checkpoint_path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'epochs_completed': self.training_history['epochs_completed'],
            'training_history': self.training_history,
            'device': str(self.device),
            'networks_available': {
                'value_network': True,
                'policy_network': True
            }
        }