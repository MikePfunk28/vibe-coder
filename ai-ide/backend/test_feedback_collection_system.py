"""
Tests for the Feedback Collection and Processing System
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from feedback_collection_system import (
    UserInteractionTracker, FeedbackCollector, UserPreferenceExtractor,
    FeedbackAggregator, UserInteraction, FeedbackSignal, UserPreference,
    InteractionType, FeedbackType
)


class TestUserInteractionTracker:
    """Test cases for UserInteractionTracker."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UserInteractionTracker(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except (PermissionError, FileNotFoundError):
            pass  # Ignore cleanup errors on Windows
    
    def test_database_initialization(self):
        """Test that database is properly initialized."""
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('interactions', 'feedback_signals')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        assert 'interactions' in tables
        assert 'feedback_signals' in tables
    
    def test_track_interaction(self):
        """Test tracking user interactions."""
        interaction = UserInteraction(
            id="test_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100},
            ai_response={"suggestion": "def test():"},
            user_action="accept",
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        self.tracker.track_interaction(interaction)
        
        # Verify interaction was stored
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute("SELECT * FROM interactions WHERE id = ?", (interaction.id,))
            row = cursor.fetchone()
            
        assert row is not None
        assert row[1] == "user_123"  # user_id
        assert row[2] == "code_completion"  # interaction_type
    
    def test_extract_implicit_feedback_acceptance(self):
        """Test extraction of implicit acceptance feedback."""
        # Create initial interaction
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
        
        # Create subsequent acceptance action
        acceptance_action = UserInteraction(
            id="test_002",
            user_id="user_123",
            interaction_type=InteractionType.SUGGESTION_ACCEPTANCE,
            timestamp=interaction.timestamp + timedelta(seconds=5),
            context={},
            ai_response=None,
            user_action="accept",
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        feedback_signals = self.tracker.extract_implicit_feedback(
            interaction, [acceptance_action]
        )
        
        assert len(feedback_signals) == 1
        assert feedback_signals[0].feedback_type == FeedbackType.IMPLICIT_ACCEPTANCE
        assert feedback_signals[0].strength > 0.8
    
    def test_extract_implicit_feedback_timeout(self):
        """Test extraction of timeout feedback."""
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
        
        # No subsequent actions (timeout)
        feedback_signals = self.tracker.extract_implicit_feedback(interaction, [])
        
        assert len(feedback_signals) == 1
        assert feedback_signals[0].feedback_type == FeedbackType.IMPLICIT_TIMEOUT
        assert feedback_signals[0].strength < 0.5
    
    def test_get_recent_interactions(self):
        """Test retrieving recent interactions."""
        # Create interactions at different times
        now = datetime.now()
        old_interaction = UserInteraction(
            id="old_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=now - timedelta(hours=25),  # Older than 24 hours
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        recent_interaction = UserInteraction(
            id="recent_001",
            user_id="user_123",
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=now - timedelta(hours=1),  # Within 24 hours
            context={},
            ai_response={},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        self.tracker.track_interaction(old_interaction)
        self.tracker.track_interaction(recent_interaction)
        
        recent_interactions = self.tracker.get_recent_interactions("user_123", hours=24)
        
        assert len(recent_interactions) == 1
        assert recent_interactions[0].id == "recent_001"


class TestFeedbackCollector:
    """Test cases for FeedbackCollector."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UserInteractionTracker(self.temp_db.name)
        self.collector = FeedbackCollector(self.tracker)
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except (PermissionError, FileNotFoundError):
            pass  # Ignore cleanup errors on Windows
    
    def test_collect_explicit_feedback(self):
        """Test collecting explicit user feedback."""
        interaction_id = "test_001"
        rating = 4
        comment = "Good suggestion!"
        
        self.collector.collect_explicit_feedback(interaction_id, rating, comment)
        
        # Verify feedback was stored
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute("""
                SELECT * FROM feedback_signals WHERE interaction_id = ?
            """, (interaction_id,))
            row = cursor.fetchone()
        
        assert row is not None
        assert row[3] == 0.8  # strength (4/5)
        assert row[6] == 4  # explicit_rating
    
    def test_collect_explicit_feedback_invalid_rating(self):
        """Test that invalid ratings are rejected."""
        with pytest.raises(ValueError):
            self.collector.collect_explicit_feedback("test_001", 6)  # Invalid rating
        
        with pytest.raises(ValueError):
            self.collector.collect_explicit_feedback("test_001", 0)  # Invalid rating
    
    def test_collect_implicit_feedback(self):
        """Test collecting implicit feedback."""
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
        
        subsequent_actions = [
            UserInteraction(
                id="test_002",
                user_id="user_123",
                interaction_type=InteractionType.SUGGESTION_ACCEPTANCE,
                timestamp=interaction.timestamp + timedelta(seconds=5),
                context={},
                ai_response=None,
                user_action="accept",
                session_id="session_001",
                file_path="test.py",
                line_number=10
            )
        ]
        
        self.collector.collect_implicit_feedback(interaction, subsequent_actions)
        
        # Verify feedback was stored
        feedback_signals = self.collector.get_feedback_for_interaction("test_001")
        assert len(feedback_signals) >= 1
        assert any(signal.feedback_type == FeedbackType.IMPLICIT_ACCEPTANCE 
                  for signal in feedback_signals)
    
    def test_anonymization(self):
        """Test that sensitive information is anonymized."""
        signal = FeedbackSignal(
            interaction_id="test_001",
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            strength=0.8,
            timestamp=datetime.now(),
            context_features={"file_path": "/sensitive/path/file.py", "user_id": "user_123"},
            explicit_rating=4,
            implicit_signals={}
        )
        
        anonymized = self.collector._anonymize_feedback(signal)
        
        # Check that sensitive fields are hashed
        assert anonymized.context_features["file_path"] != "/sensitive/path/file.py"
        assert anonymized.context_features["user_id"] != "user_123"
        assert len(anonymized.context_features["file_path"]) == 16  # Hash length
        assert len(anonymized.context_features["user_id"]) == 16  # Hash length


class TestUserPreferenceExtractor:
    """Test cases for UserPreferenceExtractor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UserInteractionTracker(self.temp_db.name)
        self.collector = FeedbackCollector(self.tracker)
        self.extractor = UserPreferenceExtractor(self.collector)
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except (PermissionError, FileNotFoundError):
            pass  # Ignore cleanup errors on Windows
    
    def test_extract_code_style_preferences(self):
        """Test extraction of code style preferences."""
        user_id = "user_123"
        
        # Create multiple interactions with Python files
        for i in range(10):
            interaction = UserInteraction(
                id=f"test_{i:03d}",
                user_id=user_id,
                interaction_type=InteractionType.CODE_COMPLETION,
                timestamp=datetime.now() - timedelta(hours=i),
                context={"cursor_position": 100},
                ai_response={"suggestion": f"def test_{i}():"},
                user_action=None,
                session_id="session_001",
                file_path=f"test_{i}.py",
                line_number=10
            )
            
            self.tracker.track_interaction(interaction)
            
            # Add positive feedback for most interactions
            if i < 8:
                self.collector.collect_explicit_feedback(interaction.id, 4)
            else:
                self.collector.collect_explicit_feedback(interaction.id, 2)
        
        preferences = self.extractor.extract_preferences(user_id)
        
        # Should have extracted Python code style preference
        python_prefs = [p for p in preferences if p.preference_type == "code_style_py"]
        assert len(python_prefs) == 1
        assert python_prefs[0].preference_value > 0.5  # Generally positive
    
    def test_extract_timing_preferences(self):
        """Test extraction of timing preferences."""
        user_id = "user_123"
        
        # Create interactions at specific hours with good feedback
        good_hours = [9, 10, 14, 15]  # Morning and afternoon
        
        for hour in good_hours:
            for i in range(5):  # Multiple interactions per hour
                interaction = UserInteraction(
                    id=f"test_{hour}_{i:03d}",
                    user_id=user_id,
                    interaction_type=InteractionType.CODE_COMPLETION,
                    timestamp=datetime.now().replace(hour=hour, minute=i*10),
                    context={"cursor_position": 100},
                    ai_response={"suggestion": f"def test_{hour}_{i}():"},
                    user_action=None,
                    session_id="session_001",
                    file_path="test.py",
                    line_number=10
                )
                
                self.tracker.track_interaction(interaction)
                self.collector.collect_explicit_feedback(interaction.id, 4)  # Good feedback
        
        preferences = self.extractor.extract_preferences(user_id)
        
        # Should have extracted preferred hours
        hour_prefs = [p for p in preferences if p.preference_type == "preferred_hours"]
        assert len(hour_prefs) == 1
        assert set(hour_prefs[0].preference_value) == set(good_hours)


class TestFeedbackAggregator:
    """Test cases for FeedbackAggregator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UserInteractionTracker(self.temp_db.name)
        self.collector = FeedbackCollector(self.tracker)
        self.extractor = UserPreferenceExtractor(self.collector)
        self.aggregator = FeedbackAggregator(self.collector, self.extractor)
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except (PermissionError, FileNotFoundError):
            pass  # Ignore cleanup errors on Windows
    
    def test_aggregate_user_feedback(self):
        """Test aggregating user feedback."""
        user_id = "user_123"
        
        # Create sample interactions with feedback
        for i in range(5):
            interaction = UserInteraction(
                id=f"test_{i:03d}",
                user_id=user_id,
                interaction_type=InteractionType.CODE_COMPLETION,
                timestamp=datetime.now() - timedelta(hours=i),
                context={"cursor_position": 100},
                ai_response={"suggestion": f"def test_{i}():"},
                user_action=None,
                session_id="session_001",
                file_path="test.py",
                line_number=10
            )
            
            self.tracker.track_interaction(interaction)
            self.collector.collect_explicit_feedback(interaction.id, 4)
        
        aggregation = self.aggregator.aggregate_user_feedback(user_id, days=1)
        
        assert aggregation["total_interactions"] == 5
        assert "feedback_by_type" in aggregation
        assert "satisfaction_trend" in aggregation
        assert "preferences" in aggregation
        assert "patterns" in aggregation
    
    def test_identify_patterns(self):
        """Test pattern identification."""
        user_id = "user_123"
        session_id = "session_001"
        
        # Create interactions with patterns
        interactions = []
        for i in range(10):
            interaction = UserInteraction(
                id=f"test_{i:03d}",
                user_id=user_id,
                interaction_type=InteractionType.CODE_COMPLETION,
                timestamp=datetime.now().replace(hour=14) - timedelta(minutes=i*5),  # All at 2 PM
                context={"cursor_position": 100},
                ai_response={"suggestion": f"def test_{i}():"},
                user_action=None,
                session_id=session_id,
                file_path="test.py",
                line_number=10
            )
            interactions.append(interaction)
        
        patterns = self.aggregator._identify_patterns(interactions)
        
        # The hour might be off by one due to timezone/DST issues
        assert patterns["most_active_hour"] in [13, 14]  # Around 2 PM
        assert patterns["most_used_interaction"] == "code_completion"
        assert "avg_session_length_minutes" in patterns
    
    def test_global_feedback_stats(self):
        """Test global feedback statistics."""
        # Create sample data across multiple users
        for user_i in range(3):
            user_id = f"user_{user_i}"
            for i in range(5):
                interaction = UserInteraction(
                    id=f"test_{user_i}_{i:03d}",
                    user_id=user_id,
                    interaction_type=InteractionType.CODE_COMPLETION,
                    timestamp=datetime.now() - timedelta(hours=i),
                    context={"cursor_position": 100},
                    ai_response={"suggestion": f"def test_{i}():"},
                    user_action=None,
                    session_id=f"session_{user_i}",
                    file_path="test.py",
                    line_number=10
                )
                
                self.tracker.track_interaction(interaction)
                self.collector.collect_explicit_feedback(interaction.id, 4)
        
        stats = self.aggregator.get_global_feedback_stats()
        
        assert stats["total_interactions"] == 15  # 3 users * 5 interactions
        assert stats["total_feedback_signals"] > 0
        assert "satisfaction_by_type" in stats
        assert "interaction_type_distribution" in stats
        assert stats["feedback_coverage"] > 0


class TestIntegration:
    """Integration tests for the complete feedback system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UserInteractionTracker(self.temp_db.name)
        self.collector = FeedbackCollector(self.tracker)
        self.extractor = UserPreferenceExtractor(self.collector)
        self.aggregator = FeedbackAggregator(self.collector, self.extractor)
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_db.name)
        except (PermissionError, FileNotFoundError):
            pass  # Ignore cleanup errors on Windows
    
    def test_complete_feedback_workflow(self):
        """Test the complete feedback collection and processing workflow."""
        user_id = "user_123"
        
        # 1. Track user interaction
        interaction = UserInteraction(
            id="test_001",
            user_id=user_id,
            interaction_type=InteractionType.CODE_COMPLETION,
            timestamp=datetime.now(),
            context={"cursor_position": 100, "file_type": "python"},
            ai_response={"suggestion": "def hello_world():"},
            user_action=None,
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        self.tracker.track_interaction(interaction)
        
        # 2. Collect explicit feedback
        self.collector.collect_explicit_feedback(interaction.id, 4, "Good suggestion!")
        
        # 3. Simulate subsequent action for implicit feedback
        acceptance_action = UserInteraction(
            id="test_002",
            user_id=user_id,
            interaction_type=InteractionType.SUGGESTION_ACCEPTANCE,
            timestamp=interaction.timestamp + timedelta(seconds=5),
            context={},
            ai_response=None,
            user_action="accept",
            session_id="session_001",
            file_path="test.py",
            line_number=10
        )
        
        self.tracker.track_interaction(acceptance_action)
        self.collector.collect_implicit_feedback(interaction, [acceptance_action])
        
        # 4. Extract preferences
        preferences = self.extractor.extract_preferences(user_id)
        
        # 5. Aggregate feedback
        aggregation = self.aggregator.aggregate_user_feedback(user_id)
        
        # Verify the complete workflow
        assert aggregation["total_interactions"] >= 1
        assert len(aggregation["feedback_by_type"]) > 0
        
        # Verify feedback signals were created
        feedback_signals = self.collector.get_feedback_for_interaction(interaction.id)
        assert len(feedback_signals) >= 1
        
        # Verify both explicit and implicit feedback
        feedback_types = [signal.feedback_type for signal in feedback_signals]
        assert FeedbackType.EXPLICIT_POSITIVE in feedback_types
        assert FeedbackType.IMPLICIT_ACCEPTANCE in feedback_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])