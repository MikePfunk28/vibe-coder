"""
Tests for the Reinforcement Learning Engine
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path

from reinforcement_learning_engine import (
    ReinforcementLearningEngine, StateEncoder, PolicyNetwork, RewardFunction,
    ExperienceReplayBuffer, Experience, PolicyAction, TrainingMetrics
)
from feedback_collection_system import (
    UserInteraction, FeedbackSignal, InteractionType, FeedbackType
)


class TestStateEncoder:
    """Test cases for StateEncoder."""
    
    def setup_method(self):
        """Set up test environment."""
        self.encoder = StateEncoder(state_dim=64)  # Smaller for testing
    
    def test_encode_state_basic(self):
        """Test basic state encoding."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100},
            ai_response={"suggestion": "def test():"},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        state = self.encoder.encode_state(interaction, [])
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (64,)
        assert state.dtype == np.float32
    
    def test_encode_interaction_type(self):
        """Test interaction type encoding."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        encoded = self.encoder._encode_interaction_type(interaction)
        
        # Should be one-hot encoded
        assert sum(encoded) == 1.0
        assert len(encoded) == len(self.encoder.interaction_type_vocab)
    
    def test_encode_file_extension(self):
        """Test file extension encoding."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        encoded = self.encoder._encode_file_extension(interaction)
        
        assert len(encoded) == 10  # Fixed size
        assert sum(encoded) <= 1.0  # At most one hot
    
    def test_encode_time_features(self):
        """Test time feature encoding."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime(2023, 1, 1, 14, 30),  # Fixed time for testing
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        encoded = self.encoder._encode_time_features(interaction)
        
        assert len(encoded) == 4  # hour_sin, hour_cos, day_sin, day_cos
        # Check that values are in reasonable range for cyclical encoding
        for value in encoded:
            assert -1.0 <= value <= 1.0
    
    def test_encode_context_features(self):
        """Test context feature encoding."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100, "selected_text": "some text"},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        encoded = self.encoder._encode_context_features(interaction)
        
        assert len(encoded) == 4
        assert encoded[1] == 1.0  # Has selection
        assert 0.0 <= encoded[0] <= 1.0  # Normalized cursor position
    
    def test_encode_user_history(self):
        """Test user history encoding."""
        history = []
        for i in range(5):
            interaction = UserInteraction(
                id=f"test_{i:03d}",
                user_id="user_123",
                interaction_type=InteractionType.CODE_COMPLETION,
                timestamp=datetime.now() - timedelta(minutes=i),
                context={},
                ai_response={},
                user_action=None,
                session_id="session_001",
                file_path="test.py",
                line_number=10
            )
            history.append(interaction)
        
        encoded = self.encoder._encode_user_history(history)
        
        assert len(encoded) == 20  # Fixed size
        # Should have some non-zero values for interaction type distribution
        assert sum(encoded) > 0


class TestPolicyNetwork:
    """Test cases for PolicyNetwork."""
    
    def setup_method(self):
        """Set up test environment."""
        self.state_dim = 32
        self.action_dim = 5
        self.network = PolicyNetwork(self.state_dim, self.action_dim)
    
    def test_network_initialization(self):
        """Test network initialization."""
        assert self.network.state_dim == self.state_dim
        assert self.network.action_dim == self.action_dim
        
        # Check that layers have correct dimensions
        assert self.network.fc1.in_features == self.state_dim
        assert self.network.fc4.out_features == self.action_dim
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        batch_size = 4
        state = torch.randn(batch_size, self.state_dim)
        
        output = self.network(state)
        
        assert output.shape == (batch_size, self.action_dim)
        
        # Check that output is a valid probability distribution
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_get_action(self):
        """Test action selection."""
        state = torch.randn(1, self.state_dim)
        
        # Test without exploration
        action, confidence = self.network.get_action(state, epsilon=0.0)
        
        assert 0 <= action < self.action_dim
        assert 0.0 <= confidence <= 1.0
        
        # Test with exploration
        action_explore, confidence_explore = self.network.get_action(state, epsilon=1.0)
        
        assert 0 <= action_explore < self.action_dim
        assert 0.0 <= confidence_explore <= 1.0
    
    def test_deterministic_behavior(self):
        """Test that network is deterministic for same input."""
        state = torch.randn(1, self.state_dim)
        
        # Set to eval mode to disable dropout
        self.network.eval()
        
        output1 = self.network(state)
        output2 = self.network(state)
        
        assert torch.allclose(output1, output2)


class TestRewardFunction:
    """Test cases for RewardFunction."""
    
    def setup_method(self):
        """Set up test environment."""
        self.reward_function = RewardFunction()
    
    def test_calculate_reward_positive_feedback(self):
        """Test reward calculation for positive feedback."""
        feedback_signals = [
            FeedbackSignal(
                interaction_id="test_001",
                feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                strength=0.8,
                timestamp=datetime.now(),
                context_features={},
                explicit_rating=4,
                implicit_signals={}
            )
        ]
        
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        reward = self.reward_function.calculate_reward(feedback_signals, interaction, [])
        
        assert reward > 0  # Should be positive for positive feedback
        assert -2.0 <= reward <= 2.0  # Within expected range
    
    def test_calculate_reward_negative_feedback(self):
        """Test reward calculation for negative feedback."""
        feedback_signals = [
            FeedbackSignal(
                interaction_id="test_001",
                feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
                strength=0.9,
                timestamp=datetime.now(),
                context_features={},
                explicit_rating=1,
                implicit_signals={}
            )
        ]
        
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        reward = self.reward_function.calculate_reward(feedback_signals, interaction, [])
        
        assert reward < 0  # Should be negative for negative feedback
        assert -2.0 <= reward <= 2.0  # Within expected range
    
    def test_calculate_reward_mixed_feedback(self):
        """Test reward calculation for mixed feedback."""
        feedback_signals = [
            FeedbackSignal(
                interaction_id="test_001",
                feedback_type=FeedbackType.IMPLICIT_ACCEPTANCE,
                strength=0.7,
                timestamp=datetime.now(),
                context_features={},
                explicit_rating=None,
                implicit_signals={}
            ),
            FeedbackSignal(
                interaction_id="test_001",
                feedback_type=FeedbackType.IMPLICIT_MODIFICATION,
                strength=0.5,
                timestamp=datetime.now(),
                context_features={},
                explicit_rating=None,
                implicit_signals={}
            )
        ]
        
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        reward = self.reward_function.calculate_reward(feedback_signals, interaction, [])
        
        assert -2.0 <= reward <= 2.0  # Within expected range
    
    def test_calculate_reward_no_feedback(self):
        """Test reward calculation with no feedback."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        reward = self.reward_function.calculate_reward([], interaction, [])
        
        assert reward == 0.0
    
    def test_context_bonus(self):
        """Test context-based reward bonuses."""
        # Test business hours bonus
        business_hour_interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now().replace(hour=14),  # 2 PM
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        bonus = self.reward_function._calculate_context_bonus(business_hour_interaction, [])
        
        assert bonus >= 0.1  # Should get business hours bonus


class TestExperienceReplayBuffer:
    """Test cases for ExperienceReplayBuffer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.buffer = ExperienceReplayBuffer(capacity=100)
    
    def test_push_and_sample(self):
        """Test adding and sampling experiences."""
        # Add some experiences
        for i in range(10):
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=np.random.randn(),
                next_state=np.random.randn(32),
                done=True,
                user_id=f"user_{i % 3}",
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            self.buffer.push(experience)
        
        assert len(self.buffer) == 10
        
        # Sample experiences
        sample = self.buffer.sample(5)
        assert len(sample) == 5
        
        # Sample more than available
        large_sample = self.buffer.sample(20)
        assert len(large_sample) == 10  # Should return all available
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        small_buffer = ExperienceReplayBuffer(capacity=5)
        
        # Add more experiences than capacity
        for i in range(10):
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=np.random.randn(),
                next_state=np.random.randn(32),
                done=True,
                user_id="user_123",
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            small_buffer.push(experience)
        
        assert len(small_buffer) == 5  # Should not exceed capacity
    
    def test_get_user_experiences(self):
        """Test getting experiences for specific user."""
        # Add experiences for different users
        for i in range(10):
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=np.random.randn(),
                next_state=np.random.randn(32),
                done=True,
                user_id=f"user_{i % 3}",
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            self.buffer.push(experience)
        
        user_0_experiences = self.buffer.get_user_experiences("user_0")
        
        # Should only contain experiences for user_0
        for exp in user_0_experiences:
            assert exp.user_id == "user_0"
        
        # Should have correct number of experiences
        expected_count = len([i for i in range(10) if i % 3 == 0])
        assert len(user_0_experiences) == expected_count


class TestReinforcementLearningEngine:
    """Test cases for ReinforcementLearningEngine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_collector = Mock()
        self.temp_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        self.temp_model.close()
        
        self.rl_engine = ReinforcementLearningEngine(
            feedback_collector=self.mock_collector,
            state_dim=32,  # Smaller for testing
            action_dim=5,
            model_path=self.temp_model.name
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_model.name)
        except (PermissionError, FileNotFoundError):
            pass
    
    def test_initialization(self):
        """Test RL engine initialization."""
        assert self.rl_engine.state_dim == 32
        assert self.rl_engine.action_dim == 5
        assert isinstance(self.rl_engine.policy_network, PolicyNetwork)
        assert isinstance(self.rl_engine.target_network, PolicyNetwork)
        assert isinstance(self.rl_engine.state_encoder, StateEncoder)
        assert isinstance(self.rl_engine.reward_function, RewardFunction)
        assert isinstance(self.rl_engine.experience_buffer, ExperienceReplayBuffer)
    
    def test_get_action(self):
        """Test getting action from policy."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        action = self.rl_engine.get_action(interaction, [])
        
        assert isinstance(action, PolicyAction)
        assert 0 <= action.action_id < self.rl_engine.action_dim
        assert action.action_type in ['provide_completion', 'provide_explanation', 
                                    'suggest_refactor', 'provide_examples', 
                                    'ask_clarification', 'no_action']
        assert 0.0 <= action.confidence <= 1.0
    
    def test_learn_from_feedback(self):
        """Test learning from user feedback."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        action = PolicyAction(
            action_id=0,
            action_type="provide_completion",
            parameters={"style": "concise"},
            confidence=0.8
        )
        
        feedback_signals = [
            FeedbackSignal(
                interaction_id="test_001",
                feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                strength=0.8,
                timestamp=datetime.now(),
                context_features={},
                explicit_rating=4,
                implicit_signals={}
            )
        ]
        
        initial_buffer_size = len(self.rl_engine.experience_buffer)
        
        self.rl_engine.learn_from_feedback(interaction, action, feedback_signals, [])
        
        # Should have added experience to buffer
        assert len(self.rl_engine.experience_buffer) == initial_buffer_size + 1
    
    def test_map_action_id_to_action(self):
        """Test action ID mapping."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        # Test valid action IDs
        for action_id in range(self.rl_engine.action_dim):
            action = self.rl_engine._map_action_id_to_action(action_id, interaction)
            assert 'type' in action
            assert 'parameters' in action
        
        # Test invalid action ID (should default to no_action)
        invalid_action = self.rl_engine._map_action_id_to_action(999, interaction)
        assert invalid_action['type'] == 'no_action'
    
    def test_adapt_to_user(self):
        """Test user-specific model adaptation."""
        user_id = "user_123"
        
        # Add some experiences for the user
        for i in range(15):  # Need at least 10 for adaptation
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=np.random.randn(),
                next_state=np.random.randn(32),
                done=True,
                user_id=user_id,
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            self.rl_engine.experience_buffer.push(experience)
        
        # Should not have user model initially
        assert user_id not in self.rl_engine.user_models
        
        # Adapt to user
        self.rl_engine.adapt_to_user(user_id)
        
        # Should now have user-specific model
        assert user_id in self.rl_engine.user_models
        assert isinstance(self.rl_engine.user_models[user_id], PolicyNetwork)
    
    def test_get_user_performance(self):
        """Test getting user performance metrics."""
        user_id = "user_123"
        
        # Add some experiences with varying rewards
        rewards = [0.5, -0.2, 0.8, 0.1, -0.1]
        for i, reward in enumerate(rewards):
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=reward,
                next_state=np.random.randn(32),
                done=True,
                user_id=user_id,
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            self.rl_engine.experience_buffer.push(experience)
        
        performance = self.rl_engine.get_user_performance(user_id)
        
        assert "total_experiences" in performance
        assert "average_reward" in performance
        assert "reward_std" in performance
        assert "positive_feedback_rate" in performance
        assert "recent_trend" in performance
        assert "has_personalized_model" in performance
        
        assert performance["total_experiences"] == len(rewards)
        assert abs(performance["average_reward"] - np.mean(rewards)) < 1e-6
    
    def test_get_user_performance_no_data(self):
        """Test getting performance for user with no data."""
        performance = self.rl_engine.get_user_performance("nonexistent_user")
        
        assert "error" in performance
    
    def test_training_metrics(self):
        """Test training metrics collection."""
        initial_metrics_count = len(self.rl_engine.get_training_metrics())
        
        # Add enough experiences to trigger training
        for i in range(35):  # More than minimum batch size
            experience = Experience(
                state=np.random.randn(32),
                action=i % 5,
                reward=np.random.randn(),
                next_state=np.random.randn(32),
                done=True,
                user_id="user_123",
                interaction_type="code_completion",
                timestamp=datetime.now()
            )
            self.rl_engine.experience_buffer.push(experience)
        
        # Trigger training
        self.rl_engine._train_step()
        
        # Should have recorded training metrics
        metrics = self.rl_engine.get_training_metrics()
        assert len(metrics) > initial_metrics_count
        
        if metrics:
            latest_metric = metrics[-1]
            assert isinstance(latest_metric, TrainingMetrics)
            assert hasattr(latest_metric, 'loss')
            assert hasattr(latest_metric, 'average_reward')
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Save current model
        self.rl_engine._save_model()
        
        # Verify file was created
        assert Path(self.temp_model.name).exists()
        
        # Create new engine and load model
        new_engine = ReinforcementLearningEngine(
            feedback_collector=self.mock_collector,
            state_dim=32,
            action_dim=5,
            model_path=self.temp_model.name
        )
        
        # Should have loaded the saved state
        assert new_engine.training_step == self.rl_engine.training_step
        assert new_engine.episode == self.rl_engine.episode
        assert abs(new_engine.epsilon - self.rl_engine.epsilon) < 1e-6
    
    def test_reset_training(self):
        """Test training state reset."""
        # Set some training state
        self.rl_engine.training_step = 100
        self.rl_engine.episode = 10
        self.rl_engine.epsilon = 0.05
        
        # Add some experiences and metrics
        experience = Experience(
            state=np.random.randn(32),
            action=0,
            reward=0.5,
            next_state=np.random.randn(32),
            done=True,
            user_id="user_123",
            interaction_type="code_completion",
            timestamp=datetime.now()
        )
        self.rl_engine.experience_buffer.push(experience)
        
        # Reset training
        self.rl_engine.reset_training()
        
        # Should have reset state
        assert self.rl_engine.training_step == 0
        assert self.rl_engine.episode == 0
        assert self.rl_engine.epsilon == 0.1
        assert len(self.rl_engine.training_metrics) == 0
        assert len(self.rl_engine.experience_buffer) == 0


class TestIntegration:
    """Integration tests for the complete RL system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_collector = Mock()
        self.temp_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        self.temp_model.close()
        
        self.rl_engine = ReinforcementLearningEngine(
            feedback_collector=self.mock_collector,
            state_dim=32,
            action_dim=5,
            model_path=self.temp_model.name
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_model.name)
        except (PermissionError, FileNotFoundError):
            pass
    
    def test_complete_learning_cycle(self):
        """Test complete learning cycle from interaction to adaptation."""
        user_id = "user_123"
        
        # Simulate multiple interactions and feedback
        for i in range(20):
            # Create interaction
            interaction = UserInteraction(
                id=f"test_{i:03d}",
                user_id=user_id,
                interaction_type=InteractionType.CODE_COMPLETION,
                timestamp=datetime.now() - timedelta(minutes=i),
                context={"cursor_position": 100 + i},
                ai_response={"suggestion": f"def test_{i}():"},
                user_action=None,
                session_id="session_001",
                file_path="test.py",
                line_number=10 + i
            )
            
            # Get action
            action = self.rl_engine.get_action(interaction, [])
            
            # Create feedback (mostly positive for this test)
            feedback_type = FeedbackType.EXPLICIT_POSITIVE if i % 4 != 0 else FeedbackType.EXPLICIT_NEGATIVE
            rating = 4 if feedback_type == FeedbackType.EXPLICIT_POSITIVE else 2
            
            feedback_signals = [
                FeedbackSignal(
                    interaction_id=interaction.id,
                    feedback_type=feedback_type,
                    strength=0.8 if feedback_type == FeedbackType.EXPLICIT_POSITIVE else 0.3,
                    timestamp=datetime.now(),
                    context_features={},
                    explicit_rating=rating,
                    implicit_signals={}
                )
            ]
            
            # Learn from feedback
            self.rl_engine.learn_from_feedback(interaction, action, feedback_signals, [])
        
        # Should have experiences in buffer
        assert len(self.rl_engine.experience_buffer) == 20
        
        # Adapt to user
        self.rl_engine.adapt_to_user(user_id)
        
        # Should have user-specific model
        assert user_id in self.rl_engine.user_models
        
        # Get performance metrics
        performance = self.rl_engine.get_user_performance(user_id)
        
        assert performance["total_experiences"] == 20
        assert performance["has_personalized_model"] == True
        assert performance["positive_feedback_rate"] > 0.5  # Mostly positive feedback
    
    def test_multi_user_learning(self):
        """Test learning with multiple users."""
        users = ["user_1", "user_2", "user_3"]
        
        # Create interactions for multiple users
        for user_id in users:
            for i in range(10):
                interaction = UserInteraction(
                    id=f"{user_id}_test_{i:03d}",
                    user_id=user_id,
                    interaction_type=InteractionType.CODE_COMPLETION,
                    timestamp=datetime.now() - timedelta(minutes=i),
                    context={"cursor_position": 100 + i},
                    ai_response={"suggestion": f"def test_{i}():"},
                    user_action=None,
                    session_id=f"{user_id}_session",
                    file_path="test.py",
                    line_number=10 + i
                )
                
                action = self.rl_engine.get_action(interaction, [])
                
                # Different users have different preferences
                if user_id == "user_1":
                    feedback_type = FeedbackType.EXPLICIT_POSITIVE
                    rating = 4
                elif user_id == "user_2":
                    feedback_type = FeedbackType.EXPLICIT_NEGATIVE
                    rating = 2
                else:
                    feedback_type = FeedbackType.IMPLICIT_ACCEPTANCE
                    rating = None
                
                feedback_signals = [
                    FeedbackSignal(
                        interaction_id=interaction.id,
                        feedback_type=feedback_type,
                        strength=0.8 if "POSITIVE" in feedback_type.value or "ACCEPTANCE" in feedback_type.value else 0.3,
                        timestamp=datetime.now(),
                        context_features={},
                        explicit_rating=rating,
                        implicit_signals={}
                    )
                ]
                
                self.rl_engine.learn_from_feedback(interaction, action, feedback_signals, [])
        
        # Adapt to each user
        for user_id in users:
            self.rl_engine.adapt_to_user(user_id)
        
        # Each user should have their own model
        for user_id in users:
            assert user_id in self.rl_engine.user_models
            
            performance = self.rl_engine.get_user_performance(user_id)
            assert performance["total_experiences"] == 10
            assert performance["has_personalized_model"] == True
            
            # User 1 should have higher average reward than User 2
            if user_id == "user_1":
                user_1_performance = performance["average_reward"]
            elif user_id == "user_2":
                user_2_performance = performance["average_reward"]
        
        # User 1 (positive feedback) should have better performance than User 2 (negative feedback)
        assert user_1_performance > user_2_performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])