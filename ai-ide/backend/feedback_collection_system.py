"""
Feedback Collection and Processing System for AI IDE

This module implements the feedback collection system that tracks user interactions,
collects explicit and implicit feedback, and processes it for reinforcement learning.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from collections import defaultdict, deque
import numpy as np


class InteractionType(Enum):
    """Types of user interactions with the AI system."""
    CODE_COMPLETION = "code_completion"
    CODE_GENERATION = "code_generation"
    SEMANTIC_SEARCH = "semantic_search"
    SUGGESTION_ACCEPTANCE = "suggestion_acceptance"
    SUGGESTION_REJECTION = "suggestion_rejection"
    EXPLICIT_FEEDBACK = "explicit_feedback"
    CONTEXT_SWITCH = "context_switch"
    ERROR_CORRECTION = "error_correction"


class FeedbackType(Enum):
    """Types of feedback signals."""
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    IMPLICIT_ACCEPTANCE = "implicit_acceptance"
    IMPLICIT_REJECTION = "implicit_rejection"
    IMPLICIT_MODIFICATION = "implicit_modification"
    IMPLICIT_DELETION = "implicit_deletion"
    IMPLICIT_TIMEOUT = "implicit_timeout"


@dataclass
class UserInteraction:
    """Represents a single user interaction with the AI system."""
    id: str
    user_id: str
    interaction_type: InteractionType
    timestamp: datetime
    context: Dict[str, Any]
    ai_response: Optional[Dict[str, Any]]
    user_action: Optional[str]
    session_id: str
    file_path: Optional[str]
    line_number: Optional[int]


@dataclass
class FeedbackSignal:
    """Represents a feedback signal from user interaction."""
    interaction_id: str
    feedback_type: FeedbackType
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    context_features: Dict[str, Any]
    explicit_rating: Optional[int]  # 1-5 scale for explicit feedback
    implicit_signals: Dict[str, float]


@dataclass
class UserPreference:
    """Represents learned user preferences."""
    user_id: str
    preference_type: str
    preference_value: Any
    confidence: float
    last_updated: datetime
    evidence_count: int


class UserInteractionTracker:
    """Tracks user interactions and extracts implicit feedback signals."""
    
    def __init__(self, db_path: str = "ai_ide_interactions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._interaction_buffer = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def _init_database(self):
        """Initialize the SQLite database for storing interactions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    ai_response TEXT,
                    user_action TEXT,
                    session_id TEXT,
                    file_path TEXT,
                    line_number INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_signals (
                    id TEXT PRIMARY KEY,
                    interaction_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_features TEXT,
                    explicit_rating INTEGER,
                    implicit_signals TEXT,
                    FOREIGN KEY (interaction_id) REFERENCES interactions (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_user_time 
                ON interactions (user_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_interaction 
                ON feedback_signals (interaction_id)
            """)
    
    def track_interaction(self, interaction: UserInteraction) -> None:
        """Track a user interaction."""
        with self._lock:
            self._interaction_buffer.append(interaction)
            
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO interactions 
                (id, user_id, interaction_type, timestamp, context, ai_response, 
                 user_action, session_id, file_path, line_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.id,
                interaction.user_id,
                interaction.interaction_type.value,
                interaction.timestamp.isoformat(),
                json.dumps(interaction.context) if interaction.context else None,
                json.dumps(interaction.ai_response) if interaction.ai_response else None,
                interaction.user_action,
                interaction.session_id,
                interaction.file_path,
                interaction.line_number
            ))
        
        self.logger.debug(f"Tracked interaction: {interaction.id}")
    
    def extract_implicit_feedback(self, interaction: UserInteraction, 
                                subsequent_actions: List[UserInteraction]) -> List[FeedbackSignal]:
        """Extract implicit feedback signals from user behavior."""
        feedback_signals = []
        
        # Analyze subsequent actions to infer feedback
        if interaction.interaction_type in [InteractionType.CODE_COMPLETION, 
                                          InteractionType.CODE_GENERATION]:
            feedback_signals.extend(
                self._analyze_code_suggestion_feedback(interaction, subsequent_actions)
            )
        elif interaction.interaction_type == InteractionType.SEMANTIC_SEARCH:
            feedback_signals.extend(
                self._analyze_search_feedback(interaction, subsequent_actions)
            )
        
        return feedback_signals
    
    def _analyze_code_suggestion_feedback(self, interaction: UserInteraction, 
                                        subsequent_actions: List[UserInteraction]) -> List[FeedbackSignal]:
        """Analyze feedback for code suggestions."""
        feedback_signals = []
        
        # Look for acceptance/rejection patterns
        for action in subsequent_actions[:5]:  # Look at next 5 actions
            time_diff = (action.timestamp - interaction.timestamp).total_seconds()
            
            if time_diff > 300:  # Stop looking after 5 minutes
                break
                
            # Quick acceptance (within 10 seconds)
            if (action.interaction_type == InteractionType.SUGGESTION_ACCEPTANCE and 
                time_diff < 10):
                feedback_signals.append(FeedbackSignal(
                    interaction_id=interaction.id,
                    feedback_type=FeedbackType.IMPLICIT_ACCEPTANCE,
                    strength=0.9,
                    timestamp=action.timestamp,
                    context_features=self._extract_context_features(interaction),
                    explicit_rating=None,
                    implicit_signals={"acceptance_speed": 10 - time_diff}
                ))
            
            # Modification after suggestion (partial acceptance)
            elif (action.user_action and "modify" in action.user_action.lower() and 
                  time_diff < 60):
                feedback_signals.append(FeedbackSignal(
                    interaction_id=interaction.id,
                    feedback_type=FeedbackType.IMPLICIT_MODIFICATION,
                    strength=0.6,
                    timestamp=action.timestamp,
                    context_features=self._extract_context_features(interaction),
                    explicit_rating=None,
                    implicit_signals={"modification_delay": time_diff}
                ))
            
            # Deletion/rejection
            elif (action.user_action and "delete" in action.user_action.lower() and 
                  time_diff < 30):
                feedback_signals.append(FeedbackSignal(
                    interaction_id=interaction.id,
                    feedback_type=FeedbackType.IMPLICIT_DELETION,
                    strength=0.8,
                    timestamp=action.timestamp,
                    context_features=self._extract_context_features(interaction),
                    explicit_rating=None,
                    implicit_signals={"deletion_speed": time_diff}
                ))
        
        # Timeout (no action taken)
        if not feedback_signals:
            feedback_signals.append(FeedbackSignal(
                interaction_id=interaction.id,
                feedback_type=FeedbackType.IMPLICIT_TIMEOUT,
                strength=0.3,
                timestamp=datetime.now(),
                context_features=self._extract_context_features(interaction),
                explicit_rating=None,
                implicit_signals={"timeout": True}
            ))
        
        return feedback_signals
    
    def _analyze_search_feedback(self, interaction: UserInteraction, 
                               subsequent_actions: List[UserInteraction]) -> List[FeedbackSignal]:
        """Analyze feedback for search results."""
        feedback_signals = []
        
        # Look for result selection patterns
        for action in subsequent_actions[:3]:  # Look at next 3 actions
            time_diff = (action.timestamp - interaction.timestamp).total_seconds()
            
            if time_diff > 120:  # Stop looking after 2 minutes
                break
            
            # Quick result selection indicates good search
            if (action.user_action and "select" in action.user_action.lower() and 
                time_diff < 30):
                feedback_signals.append(FeedbackSignal(
                    interaction_id=interaction.id,
                    feedback_type=FeedbackType.IMPLICIT_ACCEPTANCE,
                    strength=0.8,
                    timestamp=action.timestamp,
                    context_features=self._extract_context_features(interaction),
                    explicit_rating=None,
                    implicit_signals={"selection_speed": time_diff}
                ))
        
        return feedback_signals
    
    def _extract_context_features(self, interaction: UserInteraction) -> Dict[str, Any]:
        """Extract relevant context features for learning."""
        features = {
            "interaction_type": interaction.interaction_type.value,
            "file_extension": self._get_file_extension(interaction.file_path),
            "time_of_day": interaction.timestamp.hour,
            "day_of_week": interaction.timestamp.weekday(),
        }
        
        if interaction.context:
            features.update({
                "context_length": len(str(interaction.context)),
                "has_selection": "selected_text" in interaction.context,
                "cursor_position": interaction.context.get("cursor_position", 0)
            })
        
        return features
    
    def _get_file_extension(self, file_path: Optional[str]) -> Optional[str]:
        """Extract file extension from path."""
        if not file_path:
            return None
        return file_path.split('.')[-1] if '.' in file_path else None
    
    def get_recent_interactions(self, user_id: str, hours: int = 24) -> List[UserInteraction]:
        """Get recent interactions for a user."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM interactions 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, cutoff_time.isoformat()))
            
            interactions = []
            for row in cursor.fetchall():
                interactions.append(UserInteraction(
                    id=row[0],
                    user_id=row[1],
                    interaction_type=InteractionType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    context=json.loads(row[4]) if row[4] else None,
                    ai_response=json.loads(row[5]) if row[5] else None,
                    user_action=row[6],
                    session_id=row[7],
                    file_path=row[8],
                    line_number=row[9]
                ))
            
            return interactions


class FeedbackCollector:
    """Collects and processes feedback with privacy preservation."""
    
    def __init__(self, interaction_tracker: UserInteractionTracker):
        self.interaction_tracker = interaction_tracker
        self.logger = logging.getLogger(__name__)
        self._feedback_buffer = deque(maxlen=500)
        self._lock = threading.Lock()
        
    def collect_explicit_feedback(self, interaction_id: str, rating: int, 
                                comment: Optional[str] = None) -> None:
        """Collect explicit user feedback."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback_signal = FeedbackSignal(
            interaction_id=interaction_id,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE if rating >= 4 else FeedbackType.EXPLICIT_NEGATIVE,
            strength=rating / 5.0,
            timestamp=datetime.now(),
            context_features={},
            explicit_rating=rating,
            implicit_signals={"comment": comment} if comment else {}
        )
        
        self._store_feedback_signal(feedback_signal)
        self.logger.info(f"Collected explicit feedback: {rating}/5 for interaction {interaction_id}")
    
    def collect_implicit_feedback(self, interaction: UserInteraction, 
                                subsequent_actions: List[UserInteraction]) -> None:
        """Collect implicit feedback from user behavior."""
        feedback_signals = self.interaction_tracker.extract_implicit_feedback(
            interaction, subsequent_actions
        )
        
        for signal in feedback_signals:
            self._store_feedback_signal(signal)
        
        self.logger.debug(f"Collected {len(feedback_signals)} implicit feedback signals")
    
    def _store_feedback_signal(self, signal: FeedbackSignal) -> None:
        """Store feedback signal with privacy preservation."""
        # Hash sensitive information
        anonymized_signal = self._anonymize_feedback(signal)
        
        with self._lock:
            self._feedback_buffer.append(anonymized_signal)
        
        # Store in database
        with sqlite3.connect(self.interaction_tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback_signals 
                (id, interaction_id, feedback_type, strength, timestamp, 
                 context_features, explicit_rating, implicit_signals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._generate_feedback_id(signal),
                signal.interaction_id,
                signal.feedback_type.value,
                signal.strength,
                signal.timestamp.isoformat(),
                json.dumps(signal.context_features),
                signal.explicit_rating,
                json.dumps(signal.implicit_signals)
            ))
    
    def _anonymize_feedback(self, signal: FeedbackSignal) -> FeedbackSignal:
        """Anonymize sensitive information in feedback."""
        # Remove or hash sensitive context features
        anonymized_features = {}
        for key, value in signal.context_features.items():
            if key in ["file_path", "user_id"]:
                # Hash sensitive values
                anonymized_features[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            else:
                anonymized_features[key] = value
        
        return FeedbackSignal(
            interaction_id=signal.interaction_id,
            feedback_type=signal.feedback_type,
            strength=signal.strength,
            timestamp=signal.timestamp,
            context_features=anonymized_features,
            explicit_rating=signal.explicit_rating,
            implicit_signals=signal.implicit_signals
        )
    
    def _generate_feedback_id(self, signal: FeedbackSignal) -> str:
        """Generate unique ID for feedback signal."""
        content = f"{signal.interaction_id}_{signal.feedback_type.value}_{signal.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_feedback_for_interaction(self, interaction_id: str) -> List[FeedbackSignal]:
        """Get all feedback signals for a specific interaction."""
        with sqlite3.connect(self.interaction_tracker.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM feedback_signals WHERE interaction_id = ?
            """, (interaction_id,))
            
            signals = []
            for row in cursor.fetchall():
                signals.append(FeedbackSignal(
                    interaction_id=row[1],
                    feedback_type=FeedbackType(row[2]),
                    strength=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    context_features=json.loads(row[5]),
                    explicit_rating=row[6],
                    implicit_signals=json.loads(row[7])
                ))
            
            return signals


class UserPreferenceExtractor:
    """Extracts user preferences from feedback patterns."""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.logger = logging.getLogger(__name__)
    
    def extract_preferences(self, user_id: str) -> List[UserPreference]:
        """Extract user preferences from interaction history."""
        interactions = self.feedback_collector.interaction_tracker.get_recent_interactions(
            user_id, hours=168  # Last week
        )
        
        preferences = []
        
        # Extract code style preferences
        preferences.extend(self._extract_code_style_preferences(user_id, interactions))
        
        # Extract timing preferences
        preferences.extend(self._extract_timing_preferences(user_id, interactions))
        
        # Extract context preferences
        preferences.extend(self._extract_context_preferences(user_id, interactions))
        
        return preferences
    
    def _extract_code_style_preferences(self, user_id: str, 
                                      interactions: List[UserInteraction]) -> List[UserPreference]:
        """Extract code style preferences."""
        preferences = []
        
        # Analyze accepted vs rejected suggestions by file type
        file_type_feedback = defaultdict(list)
        
        for interaction in interactions:
            if interaction.interaction_type in [InteractionType.CODE_COMPLETION, 
                                              InteractionType.CODE_GENERATION]:
                file_ext = self._get_file_extension(interaction.file_path)
                if file_ext:
                    feedback_signals = self.feedback_collector.get_feedback_for_interaction(interaction.id)
                    for signal in feedback_signals:
                        file_type_feedback[file_ext].append(signal.strength)
        
        # Calculate preferences by file type
        for file_ext, strengths in file_type_feedback.items():
            if len(strengths) >= 5:  # Need minimum samples
                avg_satisfaction = np.mean(strengths)
                confidence = min(len(strengths) / 20.0, 1.0)  # Max confidence at 20 samples
                
                preferences.append(UserPreference(
                    user_id=user_id,
                    preference_type=f"code_style_{file_ext}",
                    preference_value=avg_satisfaction,
                    confidence=confidence,
                    last_updated=datetime.now(),
                    evidence_count=len(strengths)
                ))
        
        return preferences
    
    def _extract_timing_preferences(self, user_id: str, 
                                  interactions: List[UserInteraction]) -> List[UserPreference]:
        """Extract timing-based preferences."""
        preferences = []
        
        # Analyze satisfaction by time of day
        hourly_feedback = defaultdict(list)
        
        for interaction in interactions:
            hour = interaction.timestamp.hour
            feedback_signals = self.feedback_collector.get_feedback_for_interaction(interaction.id)
            for signal in feedback_signals:
                hourly_feedback[hour].append(signal.strength)
        
        # Find preferred working hours
        best_hours = []
        for hour, strengths in hourly_feedback.items():
            if len(strengths) >= 3:
                avg_satisfaction = np.mean(strengths)
                if avg_satisfaction > 0.7:
                    best_hours.append(hour)
        
        if best_hours:
            preferences.append(UserPreference(
                user_id=user_id,
                preference_type="preferred_hours",
                preference_value=best_hours,
                confidence=0.8,
                last_updated=datetime.now(),
                evidence_count=len(best_hours)
            ))
        
        return preferences
    
    def _extract_context_preferences(self, user_id: str, 
                                   interactions: List[UserInteraction]) -> List[UserPreference]:
        """Extract context-based preferences."""
        preferences = []
        
        # Analyze preferences for different context lengths
        context_feedback = defaultdict(list)
        
        for interaction in interactions:
            if interaction.context:
                context_length = len(str(interaction.context))
                context_bucket = "short" if context_length < 500 else "medium" if context_length < 2000 else "long"
                
                feedback_signals = self.feedback_collector.get_feedback_for_interaction(interaction.id)
                for signal in feedback_signals:
                    context_feedback[context_bucket].append(signal.strength)
        
        # Determine preferred context length
        best_context = None
        best_score = 0
        
        for context_type, strengths in context_feedback.items():
            if len(strengths) >= 5:
                avg_satisfaction = np.mean(strengths)
                if avg_satisfaction > best_score:
                    best_score = avg_satisfaction
                    best_context = context_type
        
        if best_context:
            preferences.append(UserPreference(
                user_id=user_id,
                preference_type="preferred_context_length",
                preference_value=best_context,
                confidence=0.7,
                last_updated=datetime.now(),
                evidence_count=len(context_feedback[best_context])
            ))
        
        return preferences
    
    def _get_file_extension(self, file_path: Optional[str]) -> Optional[str]:
        """Extract file extension from path."""
        if not file_path:
            return None
        return file_path.split('.')[-1] if '.' in file_path else None


class FeedbackAggregator:
    """Aggregates and analyzes feedback patterns."""
    
    def __init__(self, feedback_collector: FeedbackCollector, 
                 preference_extractor: UserPreferenceExtractor):
        self.feedback_collector = feedback_collector
        self.preference_extractor = preference_extractor
        self.logger = logging.getLogger(__name__)
    
    def aggregate_user_feedback(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Aggregate feedback for a user over specified days."""
        interactions = self.feedback_collector.interaction_tracker.get_recent_interactions(
            user_id, hours=days * 24
        )
        
        aggregation = {
            "total_interactions": len(interactions),
            "feedback_by_type": defaultdict(list),
            "satisfaction_trend": [],
            "preferences": self.preference_extractor.extract_preferences(user_id),
            "patterns": {}
        }
        
        # Aggregate feedback by type
        for interaction in interactions:
            feedback_signals = self.feedback_collector.get_feedback_for_interaction(interaction.id)
            for signal in feedback_signals:
                aggregation["feedback_by_type"][signal.feedback_type.value].append(signal.strength)
        
        # Calculate satisfaction trend (daily averages)
        daily_satisfaction = defaultdict(list)
        for interaction in interactions:
            day = interaction.timestamp.date()
            feedback_signals = self.feedback_collector.get_feedback_for_interaction(interaction.id)
            if feedback_signals:
                avg_strength = np.mean([s.strength for s in feedback_signals])
                daily_satisfaction[day].append(avg_strength)
        
        for day, strengths in daily_satisfaction.items():
            aggregation["satisfaction_trend"].append({
                "date": day.isoformat(),
                "satisfaction": np.mean(strengths)
            })
        
        # Identify patterns
        aggregation["patterns"] = self._identify_patterns(interactions)
        
        return aggregation
    
    def _identify_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Identify patterns in user behavior."""
        patterns = {}
        
        # Most active hours
        hourly_activity = defaultdict(int)
        for interaction in interactions:
            hourly_activity[interaction.timestamp.hour] += 1
        
        if hourly_activity:
            most_active_hour = max(hourly_activity.items(), key=lambda x: x[1])
            patterns["most_active_hour"] = most_active_hour[0]
        
        # Most used interaction types
        type_counts = defaultdict(int)
        for interaction in interactions:
            type_counts[interaction.interaction_type.value] += 1
        
        if type_counts:
            most_used_type = max(type_counts.items(), key=lambda x: x[1])
            patterns["most_used_interaction"] = most_used_type[0]
        
        # Average session length (approximate)
        session_lengths = defaultdict(list)
        for interaction in interactions:
            session_lengths[interaction.session_id].append(interaction.timestamp)
        
        avg_session_length = 0
        if session_lengths:
            session_durations = []
            for session_times in session_lengths.values():
                if len(session_times) > 1:
                    duration = (max(session_times) - min(session_times)).total_seconds() / 60
                    session_durations.append(duration)
            
            if session_durations:
                avg_session_length = np.mean(session_durations)
        
        patterns["avg_session_length_minutes"] = avg_session_length
        
        return patterns
    
    def get_global_feedback_stats(self) -> Dict[str, Any]:
        """Get global feedback statistics across all users."""
        with sqlite3.connect(self.feedback_collector.interaction_tracker.db_path) as conn:
            # Total interactions
            cursor = conn.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Total feedback signals
            cursor = conn.execute("SELECT COUNT(*) FROM feedback_signals")
            total_feedback = cursor.fetchone()[0]
            
            # Average satisfaction by feedback type
            cursor = conn.execute("""
                SELECT feedback_type, AVG(strength) 
                FROM feedback_signals 
                GROUP BY feedback_type
            """)
            satisfaction_by_type = dict(cursor.fetchall())
            
            # Most common interaction types
            cursor = conn.execute("""
                SELECT interaction_type, COUNT(*) 
                FROM interactions 
                GROUP BY interaction_type 
                ORDER BY COUNT(*) DESC
            """)
            interaction_types = dict(cursor.fetchall())
        
        return {
            "total_interactions": total_interactions,
            "total_feedback_signals": total_feedback,
            "satisfaction_by_type": satisfaction_by_type,
            "interaction_type_distribution": interaction_types,
            "feedback_coverage": total_feedback / max(total_interactions, 1)
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    tracker = UserInteractionTracker()
    collector = FeedbackCollector(tracker)
    extractor = UserPreferenceExtractor(collector)
    aggregator = FeedbackAggregator(collector, extractor)
    
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
    
    tracker.track_interaction(interaction)
    collector.collect_explicit_feedback(interaction.id, 4, "Good suggestion!")
    
    # Get aggregated feedback
    stats = aggregator.aggregate_user_feedback("user_123")
    print(f"User feedback stats: {stats}")