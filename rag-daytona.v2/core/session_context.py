"""
Session Context Management for Context-Aware Conversations

Provides conversation state management, user tracking, and context persistence
for enhanced RAG responses with memory and personalization.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class UserIntent(Enum):
    """User intent types for context-aware responses"""
    APPOINTMENT = "appointment"
    QUESTION = "question"
    COMPLAINT = "complaint"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    SMALL_TALK = "small_talk"
    ACKNOWLEDGMENT = "acknowledgment"


class UserTone(Enum):
    """User emotional tone for response adaptation"""
    FRUSTRATED = "frustrated"
    HAPPY = "happy"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    IMPATIENT = "impatient"
    SATISFIED = "satisfied"


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    timestamp: datetime
    role: str  # "user" or "agent"
    text: str
    intent: Optional[UserIntent] = None
    tone: Optional[UserTone] = None
    entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "role": self.role,
            "text": self.text,
            "intent": self.intent.value if self.intent else None,
            "tone": self.tone.value if self.tone else None,
            "entities": self.entities
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            role=data["role"],
            text=data["text"],
            intent=UserIntent(data["intent"]) if data.get("intent") else None,
            tone=UserTone(data["tone"]) if data.get("tone") else None,
            entities=data.get("entities", {})
        )


@dataclass
class SessionContext:
    """
    Conversation session state for context-aware responses.

    Maintains user information, conversation history, and extracted data
    to provide personalized, memory-aware responses.
    """

    session_id: str
    user_name: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)

    # Extracted entities
    form_data: Dict[str, Any] = field(default_factory=dict)  # {"name": "John", "email": "..."}
    user_entities: Dict[str, Any] = field(default_factory=dict)  # Persistent user info

    # Conversation state
    last_user_intent: Optional[UserIntent] = None
    last_user_tone: Optional[UserTone] = None
    conversation_topic: Optional[str] = None
    interaction_count: int = 0
    is_escalated: bool = False

    # Context awareness features
    context_cache: Dict[str, Any] = field(default_factory=dict)  # Predefined context cache

    def add_turn(self, role: str, text: str, intent: Optional[UserIntent] = None,
                 tone: Optional[UserTone] = None, entities: Optional[Dict[str, Any]] = None) -> None:
        """Add conversation turn and update session state"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            role=role,
            text=text,
            intent=intent,
            tone=tone,
            entities=entities or {}
        )
        self.turns.append(turn)

        if role == "user":
            self.interaction_count += 1
            self.last_user_intent = intent
            self.last_user_tone = tone

            # Update user entities
            if entities:
                self.user_entities.update(entities)

            # Extract and store user name if mentioned
            if not self.user_name and entities.get("possible_name"):
                self.user_name = entities["possible_name"]

    def get_conversation_summary(self, last_n: int = 5) -> str:
        """Get last N turns as context string"""
        recent_turns = self.turns[-last_n:]
        summary_lines = []

        for turn in recent_turns:
            prefix = "User:" if turn.role == "user" else "Agent:"
            summary_lines.append(f"{prefix} {turn.text}")

        return "\n".join(summary_lines)

    def get_context_for_prompt(self) -> str:
        """Get full context to inject into RAG prompt"""
        context_parts = []

        # User name
        if self.user_name:
            context_parts.append(f"User's name: {self.user_name}")

        # Conversation history (last 5 turns)
        if self.turns:
            context_parts.append("\nRecent conversation:")
            context_parts.append(self.get_conversation_summary(last_n=5))

        # User intent
        if self.last_user_intent:
            context_parts.append(f"\nUser's likely intent: {self.last_user_intent.value}")

        # User tone
        if self.last_user_tone:
            context_parts.append(f"User's tone: {self.last_user_tone.value}")

        # Form data collected
        if self.form_data:
            context_parts.append(f"\nForm data collected: {self.form_data}")

        # Context cache (predefined scenarios)
        if self.context_cache:
            context_parts.append(f"\nCached context: {json.dumps(self.context_cache, ensure_ascii=False)}")

        return "\n".join(context_parts)

    def should_escalate(self) -> bool:
        """Determine if conversation should escalate to human"""
        # Escalate if user is frustrated AND interaction count > 3
        if self.last_user_tone == UserTone.FRUSTRATED and self.interaction_count > 3:
            return True

        # Escalate if too many turns without resolution
        if self.interaction_count > 12:
            return True

        # Escalate for complex complaints
        if self.last_user_intent == UserIntent.COMPLAINT and self.interaction_count > 2:
            return True

        return False

    def add_to_cache(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Add item to context cache with TTL"""
        self.context_cache[key] = {
            "value": value,
            "expires_at": datetime.now().timestamp() + ttl_seconds
        }

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from context cache if not expired"""
        if key not in self.context_cache:
            return None

        cache_item = self.context_cache[key]
        if datetime.now().timestamp() > cache_item["expires_at"]:
            # Expired, remove it
            del self.context_cache[key]
            return None

        return cache_item["value"]

    def clear_expired_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = datetime.now().timestamp()
        expired_keys = [
            key for key, item in self.context_cache.items()
            if current_time > item["expires_at"]
        ]
        for key in expired_keys:
            del self.context_cache[key]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session context for storage"""
        return {
            "session_id": self.session_id,
            "user_name": self.user_name,
            "turns": [turn.to_dict() for turn in self.turns],
            "form_data": self.form_data,
            "user_entities": self.user_entities,
            "last_user_intent": self.last_user_intent.value if self.last_user_intent else None,
            "last_user_tone": self.last_user_tone.value if self.last_user_tone else None,
            "conversation_topic": self.conversation_topic,
            "interaction_count": self.interaction_count,
            "is_escalated": self.is_escalated,
            "context_cache": self.context_cache
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        """Deserialize session context from storage"""
        session = cls(session_id=data["session_id"])
        session.user_name = data.get("user_name")
        session.turns = [ConversationTurn.from_dict(turn_data) for turn_data in data.get("turns", [])]
        session.form_data = data.get("form_data", {})
        session.user_entities = data.get("user_entities", {})
        session.last_user_intent = UserIntent(data["last_user_intent"]) if data.get("last_user_intent") else None
        session.last_user_tone = UserTone(data["last_user_tone"]) if data.get("last_user_tone") else None
        session.conversation_topic = data.get("conversation_topic")
        session.interaction_count = data.get("interaction_count", 0)
        session.is_escalated = data.get("is_escalated", False)
        session.context_cache = data.get("context_cache", {})

        # Clean expired cache on load
        session.clear_expired_cache()

        return session