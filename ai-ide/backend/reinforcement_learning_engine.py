"""
Reinforcement Learning Engine for AI IDE

This module implements the reinforcement learning system that learns from user feedback
to improve AI assistance quality and adapt to user preferences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import random
import sqlite3
import threading
from pathlib import Path

from feedback_collection_system import (
    FeedbackCollector, UserInteraction, FeedbackSignal, 
    InteractionType, FeedbackType
)


@dataclass
class Experience:
    """Represents a single experience for reinforcement learning."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    user_id: str
    interaction_type: str
    timestamp: datetime


@dataclass
class PolicyAction:
    """Represents an action taken by the policy."""
    action_id: int
    action_type: str
    parameters: Dict[str, Any]
    confidence: float


@dataclass
class TrainingMetrics:
    """Metrics for tracking training progress."""
    episode: int
    total_reward: float
    average_reward: float
    loss: float
    epsilon: float
    learning_rate: float
    timestamp: datetime


class StateEncoder:
    """Encodes interaction context into state vectors for RL."""
    
    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        self.feature_extractors = {
            'interaction_type': self._encode_interaction_type,
            'file_extension': self._encode_file_extension,
            'time_features': self._encode_time_features,
            'context_features': self._encode_context_features,
            'user_history': self._encode_user_history
        }
        
        # Vocabulary for categorical features
        self.interaction_type_vocab = {t.value: i for i, t in enumerate(InteractionType)}
        self.file_ext_vocab = {}
        self.vocab_size = 0
    
    def encode_state(self, interaction: UserInteraction, 
                    user_history: List[UserInteraction]) -> np.ndarray:
        """Encode interaction and context into state vector."""
        features = []
        
        # Encode different aspects of the state
        for feature_name, extractor in self.feature_extractors.items():
            if feature_name == 'user_history':
                feature_vector = extractor(user_history)
            else:
                feature_vector = extractor(interaction)
            features.extend(feature_vector)
        
        # Pad or truncate to fixed size
        state_vector = np.array(features[:self.state_dim])
        if len(state_vector) < self.state_dim:
            padding = np.zeros(self.state_dim - len(state_vector))
            state_vector = np.concatenate([state_vector, padding])
        
        return state_vector.astype(np.float32)
    
    def _encode_interaction_type(self, interaction: UserInteraction) -> List[float]:
        """Encode interaction type as one-hot vector."""
        vector = [0.0] * len(self.interaction_type_vocab)
        if interaction.interaction_type.value in self.interaction_type_vocab:
            idx = self.interaction_type_vocab[interaction.interaction_type.value]
            vector[idx] = 1.0
        return vector
    
    def _encode_file_extension(self, interaction: UserInteraction) -> List[float]:
        """Encode file extension."""
        if not interaction.file_path:
            return [0.0] * 10  # Default size for file extension encoding
        
        ext = interaction.file_path.split('.')[-1] if '.' in interaction.file_path else 'unknown'
        
        # Build vocabulary dynamically
        if ext not in self.file_ext_vocab:
            self.file_ext_vocab[ext] = len(self.file_ext_vocab)
        
        vector = [0.0] * 10  # Fixed size for file extensions
        if self.file_ext_vocab[ext] < 10:
            vector[self.file_ext_vocab[ext]] = 1.0
        
        return vector
    
    def _encode_time_features(self, interaction: UserInteraction) -> List[float]:
        """Encode time-based features."""
        hour = interaction.timestamp.hour / 24.0  # Normalize to [0, 1]
        day_of_week = interaction.timestamp.weekday() / 7.0  # Normalize to [0, 1]
        
        # Cyclical encoding for hour
        hour_sin = np.sin(2 * np.pi * hour)
        hour_cos = np.cos(2 * np.pi * hour)
        
        # Cyclical encoding for day of week
        day_sin = np.sin(2 * np.pi * day_of_week)
        day_cos = np.cos(2 * np.pi * day_of_week)
        
        return [hour_sin, hour_cos, day_sin, day_cos]
    
    def _encode_context_features(self, interaction: UserInteraction) -> List[float]:
        """Encode context-specific features."""
        features = []
        
        if interaction.context:
            # Cursor position (normalized)
            cursor_pos = interaction.context.get('cursor_position', 0)
            features.append(min(cursor_pos / 10000.0, 1.0))  # Normalize large positions
            
            # Has selection
            features.append(1.0 if 'selected_text' in interaction.context else 0.0)
            
            # Context length
            context_len = len(str(interaction.context))
            features.append(min(context_len / 5000.0, 1.0))  # Normalize
        else:
            features = [0.0, 0.0, 0.0]
        
        # Line number (normalized)
        line_num = interaction.line_number or 0
        features.append(min(line_num / 1000.0, 1.0))
        
        return features
    
    def _encode_user_history(self, user_history: List[UserInteraction]) -> List[float]:
        """Encode user interaction history."""
        if not user_history:
            return [0.0] * 20  # Default history encoding size
        
        # Recent interaction types distribution
        recent_interactions = user_history[-10:]  # Last 10 interactions
        type_counts = defaultdict(int)
        for interaction in recent_interactions:
            type_counts[interaction.interaction_type.value] += 1
        
        # Normalize counts
        total = len(recent_interactions)
        features = []
        for interaction_type in self.interaction_type_vocab.keys():
            features.append(type_counts[interaction_type] / total)
        
        # Average session length (approximate)
        session_lengths = []
        current_session = []
        current_session_id = None
        
        for interaction in recent_interactions:
            if interaction.session_id != current_session_id:
                if current_session:
                    duration = (current_session[-1].timestamp - current_session[0].timestamp).total_seconds()
                    session_lengths.append(duration)
                current_session = [interaction]
                current_session_id = interaction.session_id
            else:
                current_session.append(interaction)
        
        avg_session_length = np.mean(session_lengths) if session_lengths else 0.0
        features.append(min(avg_session_length / 3600.0, 1.0))  # Normalize to hours
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]


class PolicyNetwork(nn.Module):
    """Neural network for the RL policy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Use softmax for action probabilities
        return F.softmax(x, dim=-1)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> Tuple[int, float]:
        """Get action from policy with optional epsilon-greedy exploration."""
        with torch.no_grad():
            action_probs = self.forward(state)
            
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
                confidence = 1.0 / self.action_dim  # Uniform confidence for random actions
            else:
                action = torch.argmax(action_probs, dim=-1).item()
                confidence = action_probs[0, action].item()  # Handle batch dimension
            
            return action, confidence


class RewardFunction:
    """Calculates rewards based on user satisfaction metrics."""
    
    def __init__(self):
        self.reward_weights = {
            'explicit_positive': 1.0,
            'explicit_negative': -1.0,
            'implicit_acceptance': 0.8,
            'implicit_rejection': -0.6,
            'implicit_modification': 0.4,
            'implicit_deletion': -0.8,
            'implicit_timeout': -0.2
        }
        
        # Bonus rewards for consistency and improvement
        self.consistency_bonus = 0.2
        self.improvement_bonus = 0.3
        
    def calculate_reward(self, feedback_signals: List[FeedbackSignal], 
                        interaction: UserInteraction,
                        user_history: List[UserInteraction]) -> float:
        """Calculate reward based on feedback signals and context."""
        if not feedback_signals:
            return 0.0
        
        total_reward = 0.0
        
        # Base reward from feedback signals
        for signal in feedback_signals:
            base_reward = self.reward_weights.get(signal.feedback_type.value, 0.0)
            
            # Weight by signal strength
            weighted_reward = base_reward * signal.strength
            
            # Apply time decay (more recent feedback is more important)
            time_diff = (datetime.now() - signal.timestamp).total_seconds()
            time_decay = np.exp(-time_diff / 3600.0)  # Decay over 1 hour
            
            total_reward += weighted_reward * time_decay
        
        # Consistency bonus: reward consistent positive feedback
        if len(feedback_signals) > 1:
            positive_signals = [s for s in feedback_signals if s.strength > 0.5]
            if len(positive_signals) == len(feedback_signals):
                total_reward += self.consistency_bonus
        
        # Context-based adjustments
        total_reward += self._calculate_context_bonus(interaction, user_history)
        
        # Normalize reward to [-2, 2] range
        return np.clip(total_reward, -2.0, 2.0)
    
    def _calculate_context_bonus(self, interaction: UserInteraction, 
                               user_history: List[UserInteraction]) -> float:
        """Calculate context-based reward bonuses."""
        bonus = 0.0
        
        # Bonus for helping during productive hours
        if 9 <= interaction.timestamp.hour <= 17:  # Business hours
            bonus += 0.1
        
        # Bonus for consistency with user's preferred file types
        if user_history:
            recent_files = [i.file_path for i in user_history[-10:] if i.file_path]
            if recent_files and interaction.file_path:
                current_ext = interaction.file_path.split('.')[-1] if '.' in interaction.file_path else ''
                ext_counts = defaultdict(int)
                for file_path in recent_files:
                    ext = file_path.split('.')[-1] if '.' in file_path else ''
                    ext_counts[ext] += 1
                
                # Bonus if working with frequently used file type
                if current_ext and ext_counts[current_ext] >= 3:
                    bonus += 0.15
        
        return bonus


class ExperienceReplayBuffer:
    """Buffer for storing and sampling experiences for training."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_user_experiences(self, user_id: str, limit: int = 100) -> List[Experience]:
        """Get experiences for specific user."""
        user_experiences = [exp for exp in self.buffer if exp.user_id == user_id]
        return user_experiences[-limit:]  # Most recent experiences


class ReinforcementLearningEngine:
    """Main reinforcement learning engine."""
    
    def __init__(self, feedback_collector: FeedbackCollector, 
                 state_dim: int = 128, action_dim: int = 10,
                 learning_rate: float = 0.001, model_path: str = "rl_model.pth"):
        
        self.feedback_collector = feedback_collector
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = model_path
        
        # Initialize components
        self.state_encoder = StateEncoder(state_dim)
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.target_network = PolicyNetwork(state_dim, action_dim)
        self.reward_function = RewardFunction()
        self.experience_buffer = ExperienceReplayBuffer()
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        self.target_update_freq = 100  # Update target network every N steps
        
        # Training state
        self.training_step = 0
        self.episode = 0
        self.training_metrics = []
        
        # User-specific models
        self.user_models = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing model if available
        self._load_model()
    
    def get_action(self, interaction: UserInteraction, 
                  user_history: List[UserInteraction]) -> PolicyAction:
        """Get action recommendation from policy."""
        state = self.state_encoder.encode_state(interaction, user_history)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get user-specific model if available
        user_model = self.user_models.get(interaction.user_id, self.policy_network)
        
        action_id, confidence = user_model.get_action(state_tensor, self.epsilon)
        
        # Map action ID to specific action
        action = self._map_action_id_to_action(action_id, interaction)
        
        return PolicyAction(
            action_id=action_id,
            action_type=action['type'],
            parameters=action['parameters'],
            confidence=confidence
        )
    
    def _map_action_id_to_action(self, action_id: int, 
                               interaction: UserInteraction) -> Dict[str, Any]:
        """Map action ID to specific action parameters."""
        action_mapping = {
            0: {'type': 'provide_completion', 'parameters': {'style': 'concise'}},
            1: {'type': 'provide_completion', 'parameters': {'style': 'detailed'}},
            2: {'type': 'provide_explanation', 'parameters': {'depth': 'shallow'}},
            3: {'type': 'provide_explanation', 'parameters': {'depth': 'deep'}},
            4: {'type': 'suggest_refactor', 'parameters': {'scope': 'local'}},
            5: {'type': 'suggest_refactor', 'parameters': {'scope': 'global'}},
            6: {'type': 'provide_examples', 'parameters': {'count': 1}},
            7: {'type': 'provide_examples', 'parameters': {'count': 3}},
            8: {'type': 'ask_clarification', 'parameters': {}},
            9: {'type': 'no_action', 'parameters': {}}
        }
        
        return action_mapping.get(action_id, action_mapping[9])  # Default to no_action
    
    def learn_from_feedback(self, interaction: UserInteraction, 
                          action: PolicyAction,
                          feedback_signals: List[FeedbackSignal],
                          user_history: List[UserInteraction]) -> None:
        """Learn from user feedback."""
        # Encode current state
        state = self.state_encoder.encode_state(interaction, user_history)
        
        # Calculate reward
        reward = self.reward_function.calculate_reward(
            feedback_signals, interaction, user_history
        )
        
        # Create next state (simplified - could be more sophisticated)
        next_state = state.copy()  # For now, assume state doesn't change much
        
        # Create experience
        experience = Experience(
            state=state,
            action=action.action_id,
            reward=reward,
            next_state=next_state,
            done=True,  # Each interaction is treated as terminal for simplicity
            user_id=interaction.user_id,
            interaction_type=interaction.interaction_type.value,
            timestamp=datetime.now()
        )
        
        # Add to experience buffer
        self.experience_buffer.push(experience)
        
        # Train if we have enough experiences
        if len(self.experience_buffer) >= 32:  # Minimum batch size
            self._train_step()
        
        self.logger.info(f"Learned from feedback: reward={reward:.3f}, action={action.action_type}")
    
    def _train_step(self) -> None:
        """Perform one training step."""
        batch_size = 32
        experiences = self.experience_buffer.sample(batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor(np.array([exp.state for exp in experiences]))
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in experiences]))
        dones = torch.BoolTensor([exp.done for exp in experiences])
        
        # Current Q values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record metrics
        metrics = TrainingMetrics(
            episode=self.episode,
            total_reward=sum(exp.reward for exp in experiences),
            average_reward=np.mean([exp.reward for exp in experiences]),
            loss=loss.item(),
            epsilon=self.epsilon,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            timestamp=datetime.now()
        )
        self.training_metrics.append(metrics)
        
        # Save model periodically
        if self.training_step % 100 == 0:
            self._save_model()
        
        self.logger.debug(f"Training step {self.training_step}: loss={loss.item():.4f}, epsilon={self.epsilon:.4f}")
    
    def adapt_to_user(self, user_id: str) -> None:
        """Create or update user-specific model."""
        user_experiences = self.experience_buffer.get_user_experiences(user_id)
        
        if len(user_experiences) < 10:  # Need minimum experiences
            return
        
        # Create user-specific model if not exists
        if user_id not in self.user_models:
            self.user_models[user_id] = PolicyNetwork(self.state_dim, self.action_dim)
            # Initialize with main model weights
            self.user_models[user_id].load_state_dict(self.policy_network.state_dict())
        
        # Fine-tune on user-specific experiences
        user_model = self.user_models[user_id]
        user_optimizer = optim.Adam(user_model.parameters(), lr=0.0001)  # Lower learning rate
        
        # Train on user experiences
        for _ in range(5):  # Few epochs of fine-tuning
            random.shuffle(user_experiences)
            batch_size = min(16, len(user_experiences))
            
            for i in range(0, len(user_experiences), batch_size):
                batch = user_experiences[i:i + batch_size]
                
                states = torch.FloatTensor(np.array([exp.state for exp in batch]))
                actions = torch.LongTensor([exp.action for exp in batch])
                rewards = torch.FloatTensor([exp.reward for exp in batch])
                
                # Simple policy gradient update
                action_probs = user_model(states)
                selected_probs = action_probs.gather(1, actions.unsqueeze(1))
                
                # Policy gradient loss
                loss = -torch.mean(torch.log(selected_probs.squeeze()) * rewards)
                
                user_optimizer.zero_grad()
                loss.backward()
                user_optimizer.step()
        
        self.logger.info(f"Adapted model for user {user_id} with {len(user_experiences)} experiences")
    
    def get_training_metrics(self, last_n: int = 100) -> List[TrainingMetrics]:
        """Get recent training metrics."""
        return self.training_metrics[-last_n:]
    
    def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """Get performance metrics for specific user."""
        user_experiences = self.experience_buffer.get_user_experiences(user_id)
        
        if not user_experiences:
            return {"error": "No experiences found for user"}
        
        rewards = [exp.reward for exp in user_experiences]
        
        return {
            "total_experiences": len(user_experiences),
            "average_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "positive_feedback_rate": len([r for r in rewards if r > 0]) / len(rewards),
            "recent_trend": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            "has_personalized_model": user_id in self.user_models
        }
    
    def _save_model(self) -> None:
        """Save model and training state."""
        checkpoint = {
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode': self.episode,
            'epsilon': self.epsilon,
            'user_models': {uid: model.state_dict() for uid, model in self.user_models.items()}
        }
        
        torch.save(checkpoint, self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")
    
    def _load_model(self) -> None:
        """Load model and training state."""
        if not Path(self.model_path).exists():
            self.logger.info("No existing model found, starting fresh")
            return
        
        try:
            checkpoint = torch.load(self.model_path)
            
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint.get('training_step', 0)
            self.episode = checkpoint.get('episode', 0)
            self.epsilon = checkpoint.get('epsilon', 0.1)
            
            # Load user models
            user_models_state = checkpoint.get('user_models', {})
            for user_id, state_dict in user_models_state.items():
                self.user_models[user_id] = PolicyNetwork(self.state_dim, self.action_dim)
                self.user_models[user_id].load_state_dict(state_dict)
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def reset_training(self) -> None:
        """Reset training state (keep model weights)."""
        self.training_step = 0
        self.episode = 0
        self.epsilon = 0.1
        self.training_metrics = []
        self.experience_buffer = ExperienceReplayBuffer()
        self.logger.info("Training state reset")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would normally be initialized with real feedback collector
    from unittest.mock import Mock
    mock_collector = Mock()
    
    # Initialize RL engine
    rl_engine = ReinforcementLearningEngine(mock_collector)
    
    # Example interaction
    interaction = UserInteraction(
        id="test_001",
        user_id="user_123",
        interaction_type=InteractionType.CODE_COMPLETION,
        timestamp=datetime.now(),
        context={"cursor_position": 100, "file_type": "python"},
        ai_response={"suggestion": "def hello_world():"},
        user_action=None,
        session_id="session_001",
        file_path="test.py",
        line_number=10
    )
    
    # Get action recommendation
    action = rl_engine.get_action(interaction, [])
    print(f"Recommended action: {action.action_type} (confidence: {action.confidence:.3f})")
    
    # Simulate learning from feedback
    from feedback_collection_system import FeedbackSignal, FeedbackType
    feedback = [FeedbackSignal(
        interaction_id=interaction.id,
        feedback_type=FeedbackType.EXPLICIT_POSITIVE,
        strength=0.8,
        timestamp=datetime.now(),
        context_features={},
        explicit_rating=4,
        implicit_signals={}
    )]
    
    rl_engine.learn_from_feedback(interaction, action, feedback, [])
    
    # Get performance metrics
    performance = rl_engine.get_user_performance("user_123")
    print(f"User performance: {performance}")