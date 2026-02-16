# Context-Rich & Humanized LLM Responses: Complete Strategy

## Executive Summary

Your RAG responses are **context-poor** because they lack 3 critical layers:

1. **Conversation Memory** - Agent forgets what user said last turn
2. **Intent & Tone Detection** - Agent doesn't know if user is frustrated, confused, or satisfied
3. **Graceful Fallbacks** - Agent has no recovery strategies for edge cases

This doc provides **production-ready implementations** for all three.

---

## Part 1: Session Context Layer (Memory)

### What's Missing

Currently:
```python
# rag_engine.py: process_query(query, context, ...)
# You pass 'context' but it's UNUSED in response generation!
```

**Fix:** Build conversation history into prompts

### Implementation: `session_context.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
from enum import Enum

class UserIntent(Enum):
    """User intent types"""
    APPOINTMENT = "appointment"
    QUESTION = "question"
    COMPLAINT = "complaint"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    SMALL_TALK = "small_talk"
    ACKNOWLEDGMENT = "acknowledgment"

class UserTone(Enum):
    """User emotional tone"""
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
    intent: UserIntent = None
    tone: UserTone = None
    entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "role": self.role,
            "text": self.text,
            "intent": self.intent.value if self.intent else None,
            "tone": self.tone.value if self.tone else None,
            "entities": self.entities
        }

@dataclass
class SessionContext:
    """Conversation session state"""
    session_id: str
    user_name: str = None
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Extracted entities
    form_data: Dict[str, Any] = field(default_factory=dict)  # {"name": "John", "email": "..."}
    
    # Conversation state
    last_user_intent: UserIntent = None
    last_user_tone: UserTone = None
    conversation_topic: str = None
    interaction_count: int = 0
    is_escalated: bool = False
    
    def add_turn(self, role: str, text: str, intent=None, tone=None, entities=None):
        """Add conversation turn"""
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
    
    def get_conversation_summary(self, last_n: int = 3) -> str:
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
        
        # Conversation topic
        if self.conversation_topic:
            context_parts.append(f"Topic: {self.conversation_topic}")
        
        return "\n".join(context_parts)
    
    def should_escalate(self) -> bool:
        """Determine if conversation should escalate"""
        # Escalate if user is frustrated AND interaction count > 2
        if self.last_user_tone == UserTone.FRUSTRATED and self.interaction_count > 2:
            return True
        
        # Escalate if too many turns without resolution
        if self.interaction_count > 10:
            return True
        
        return False
```

---

## Part 2: Intent & Tone Detection Layer

### Implementation: `intent_tone_detector.py`

```python
import re
from typing import Tuple, Dict, Any
from session_context import UserIntent, UserTone

class IntentToneDetector:
    """Detect user intent and emotional tone"""
    
    # Frustration keywords (English + German)
    FRUSTRATION_PATTERNS = {
        "en": [
            r"this is ridiculous",
            r"i don't understand",
            r"why is this so hard",
            r"can't you just",
            r"i'm frustrated",
            r"this doesn't work",
            r"stupid",
            r"waste of time",
            r"useless",
            r"unhelpful"
        ],
        "de": [
            r"das ist lächerlich",
            r"ich verstehe nicht",
            r"warum ist das so schwer",
            r"kannst du nicht einfach",
            r"ich bin frustriert",
            r"das funktioniert nicht",
            r"dumm",
            r"zeitverschwendung",
            r"nutzlos",
            r"unhilfreich"
        ]
    }
    
    # Satisfaction keywords
    SATISFACTION_PATTERNS = {
        "en": [
            r"thanks? you",
            r"thank you so much",
            r"that helps",
            r"that's great",
            r"perfect",
            r"exactly what i needed",
            r"brilliant",
            r"love this"
        ],
        "de": [
            r"danke.*dir",
            r"vielen dank",
            r"das hilft",
            r"das ist toll",
            r"perfekt",
            r"genau das",
            r"brillant",
            r"liebes"
        ]
    }
    
    # Confusion keywords
    CONFUSION_PATTERNS = {
        "en": [
            r"confused",
            r"not sure",
            r"what do you mean",
            r"don't understand",
            r"unclear",
            r"huh",
            r"what",
            r"can you explain"
        ],
        "de": [
            r"verwirrt",
            r"nicht sicher",
            r"was meinst du",
            r"verstehe nicht",
            r"unklar",
            r"wie bitte",
            r"was",
            r"kannst du erklären"
        ]
    }
    
    # Intent detection keywords
    INTENT_PATTERNS = {
        "appointment": {
            "keywords": [
                "appointment", "schedule", "book", "meeting", "reserve", 
                "slot", "time", "date", "when can i",
                "termin", "zeitpunkt", "wann kann ich"
            ],
            "pattern": r"(appointment|schedule|book|meeting|reserve|slot|termin)"
        },
        "complaint": {
            "keywords": [
                "complaint", "problem", "issue", "doesn't work", "broken",
                "wrong", "error", "problem", "beschwerde", "problem"
            ],
            "pattern": r"(problem|issue|broken|doesn't work|complaint|beschwerde)"
        },
        "clarification": {
            "keywords": [
                "what", "how", "explain", "understand", "mean", "mean by",
                "was", "wie", "erklären", "verstehen"
            ],
            "pattern": r"(what do you mean|explain|how does|was bedeutet|wie)"
        },
        "follow_up": {
            "keywords": [
                "again", "also", "still", "more", "additionally", "what about",
                "nochmal", "auch", "immer noch", "mehr", "was ist mit"
            ],
            "pattern": r"(what about|also|additionally|still|and what|nochmal)"
        }
    }
    
    @classmethod
    def detect_tone(cls, text: str, language: str = "en") -> UserTone:
        """Detect user emotional tone"""
        text_lower = text.lower()
        
        # Check frustration
        for pattern in cls.FRUSTRATION_PATTERNS.get(language, []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                return UserTone.FRUSTRATED
        
        # Check satisfaction
        for pattern in cls.SATISFACTION_PATTERNS.get(language, []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                return UserTone.SATISFIED
        
        # Check confusion
        for pattern in cls.CONFUSION_PATTERNS.get(language, []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                return UserTone.CONFUSED
        
        # Check impatience
        if re.search(r"(urgent|asap|right now|immediately|jetzt|sofort)", text_lower):
            return UserTone.IMPATIENT
        
        # Default
        return UserTone.NEUTRAL
    
    @classmethod
    def detect_intent(cls, text: str, language: str = "en") -> UserIntent:
        """Detect user intent"""
        text_lower = text.lower()
        
        # Check each intent type
        for intent_name, intent_config in cls.INTENT_PATTERNS.items():
            if re.search(intent_config["pattern"], text_lower, re.IGNORECASE):
                return UserIntent[intent_name.upper()]
        
        # Check for follow-up patterns
        if re.search(r"^(yes|yeah|ok|okay|fine|sure|ok fine)", text_lower):
            return UserIntent.ACKNOWLEDGMENT
        
        # Check for small talk
        if re.search(r"(hello|hi|bye|thanks|how are you|guten tag|hallo)", text_lower):
            return UserIntent.SMALL_TALK
        
        # Default to question
        return UserIntent.QUESTION
    
    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, Any]:
        """Extract useful entities from user text"""
        entities = {}
        
        # Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            entities["email"] = email_match.group(0)
        
        # Phone (German format: +49 XXX or 0XXX)
        phone_match = re.search(r'(\+49|0)\s*\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{4}', text)
        if phone_match:
            entities["phone"] = phone_match.group(0)
        
        # Date patterns
        date_match = re.search(r'(\d{1,2}[./]\d{1,2}[./]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})', text)
        if date_match:
            entities["date"] = date_match.group(0)
        
        # Time patterns
        time_match = re.search(r'(\d{1,2}:\d{2}|\d{1,2}:\d{2}:\d{2})', text)
        if time_match:
            entities["time"] = time_match.group(0)
        
        # Name patterns (capitalized words)
        name_matches = re.findall(r'\b[A-Z][a-z]+\b', text)
        if name_matches and len(name_matches[0]) > 2:
            entities["possible_name"] = name_matches[0]
        
        return entities
```

---

## Part 3: Enhanced Prompt Engineering with Context

### Implementation: Modify `rag_engine.py`'s `process_query()`

**Before (BROKEN):**
```python
def process_query(self, query: str, context: Dict = None, ...):
    # Ignores context!
    prompt = f"""Answer this question: {query}
    
    Context: {context_text}
    Response:"""
```

**After (CONTEXT-AWARE):**

```python
async def process_query(
    self, 
    query: str, 
    context: Optional[Dict] = None,
    session_context: Optional[SessionContext] = None,
    streaming_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process query with FULL conversation context.
    
    Enhanced to use:
    - Session history
    - User intent & tone
    - Form data
    - Graceful fallbacks
    """
    
    start_time = time.time()
    
    try:
        # ===== STEP 1: Detect intent and tone =====
        detector = IntentToneDetector()
        user_intent = detector.detect_intent(query)
        user_tone = detector.detect_tone(query)
        entities = detector.extract_entities(query)
        
        logger.info(f"🎯 Intent: {user_intent.value}, Tone: {user_tone.value}")
        
        # ===== STEP 2: Update session context =====
        if session_context:
            session_context.add_turn(
                role="user",
                text=query,
                intent=user_intent,
                tone=user_tone,
                entities=entities
            )
        
        # ===== STEP 3: Pattern detection for quick responses =====
        pattern = self._detect_query_pattern(query)
        if pattern and user_intent != UserIntent.CLARIFICATION:
            # Use quick-answer pattern
            logger.info(f"⚡ Quick-answer pattern matched: {pattern['name']}")
            return self._handle_pattern_response(query, pattern, session_context)
        
        # ===== STEP 4: Enhanced retrieval with context =====
        # Boost retrieval based on detected intent
        boost_categories = self._get_boost_categories(user_intent)
        max_context = self._get_max_context_chars(user_intent)
        
        relevant_docs, retrieval_timing = self._retrieve_with_boosting(
            query=query,
            context=context,
            boost_categories=boost_categories,
            max_context_chars=max_context
        )
        
        # ===== STEP 5: Build context-rich prompt =====
        prompt = self._build_context_aware_prompt(
            query=query,
            relevant_docs=relevant_docs,
            session_context=session_context,
            user_intent=user_intent,
            user_tone=user_tone,
            detected_pattern=pattern
        )
        
        # ===== STEP 6: Generate response with Gemini =====
        raw_response = await self._generate_response(prompt, streaming_callback)
        
        # ===== STEP 7: Humanize response =====
        humanized_response = self.humanize_response(
            response=raw_response,
            query=query,
            session_context=session_context,
            user_intent=user_intent,
            user_tone=user_tone,
            is_first_turn=(session_context and session_context.interaction_count == 1)
        )
        
        # ===== STEP 8: Validate quality =====
        quality_check = self.validate_response_quality(humanized_response)
        if quality_check['retry'] and not pattern:
            # Retry with more context
            logger.warning(f"⚠️ Quality check failed: {quality_check['issues']}")
            return await self._retry_with_fallback(query, session_context)
        
        # ===== Build result =====
        sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
        
        result = {
            'answer': humanized_response,
            'sources': sources,
            'confidence': 0.8,  # Calculate based on retrieval quality
            'timing_breakdown': {
                **retrieval_timing,
                'total_ms': (time.time() - start_time) * 1000
            },
            'metadata': {
                'intent': user_intent.value,
                'tone': user_tone.value,
                'entities': entities,
                'pattern_used': pattern['name'] if pattern else None,
                'session_id': session_context.session_id if session_context else None
            }
        }
        
        # Store in session if provided
        if session_context:
            session_context.add_turn(
                role="agent",
                text=humanized_response
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return self._handle_error_response(query, session_context, str(e))
```

---

## Part 4: Pattern-Based Fallback Responses

### Implementation: Graceful handling of edge cases

```python
def _handle_pattern_response(
    self, 
    query: str, 
    pattern: Dict,
    session_context: Optional[SessionContext] = None
) -> Dict[str, Any]:
    """Generate response from detected pattern"""
    
    try:
        # Retrieve supporting docs for pattern
        boost_categories = pattern.get('faiss_boost', [])
        max_context_chars = pattern.get('max_context_chars', 2000)
        
        relevant_docs, timing = self._retrieve_with_boosting(
            query=query,
            context=None,
            boost_categories=boost_categories,
            max_context_chars=max_context_chars
        )
        
        # Extract structured fields
        extracted_fields = self._extract_template_fields(relevant_docs, pattern)
        
        # Fill template
        template = pattern.get('response_template', "{query}")
        
        if isinstance(template, dict):
            # Multi-language template
            language = self._detect_language(query)
            template = template.get(language, template.get('en', str(template)))
        
        response_text = template.format(**extracted_fields)
        
        # Humanize
        response_text = self.humanize_response(
            response_text,
            query,
            session_context=session_context,
            user_intent=pattern.get('intent'),
            is_first_turn=True
        )
        
        return {
            'answer': response_text,
            'sources': [doc['metadata'].get('source', '') for doc in relevant_docs],
            'confidence': 0.9,  # Patterns are high confidence
            'timing_breakdown': timing,
            'metadata': {
                'pattern': pattern['name'],
                'template_used': True
            }
        }
    
    except Exception as e:
        logger.error(f"Pattern response error: {e}")
        raise

async def _retry_with_fallback(
    self, 
    query: str, 
    session_context: Optional[SessionContext] = None
) -> Dict[str, Any]:
    """Retry with fallback strategy when first attempt fails"""
    
    logger.info("🔄 Retrying with fallback strategy...")
    
    # Strategy 1: Expand context
    expanded_query = f"{session_context.get_conversation_summary() if session_context else ''}\n\n{query}"
    
    # Strategy 2: Use keyword extraction only
    keywords = self._extract_keywords(query)
    keyword_query = " ".join(keywords)
    
    # Strategy 3: Direct RAG without pattern matching
    relevant_docs, timing = self._retrieve_with_boosting(
        query=keyword_query,
        context=None,
        boost_categories=[],
        max_context_chars=3000  # More context
    )
    
    if not relevant_docs:
        # Strategy 4: Acknowledgment + offer help
        return self._handle_no_match_response(query, session_context)
    
    # Generate with expanded context
    prompt = f"""You are Lexi, a helpful university assistant.

Previous conversation context:
{session_context.get_conversation_summary() if session_context else ''}

Current question: {query}

Here's relevant information:
{chr(10).join([doc['text'] for doc in relevant_docs])}

Provide a helpful response. If information is limited, acknowledge this and offer alternatives."""
    
    response_text = self.gemini_model.generate_content(prompt).text.strip()
    
    response_text = self.humanize_response(
        response_text,
        query,
        session_context=session_context,
        is_first_turn=False
    )
    
    return {
        'answer': response_text,
        'sources': [doc['metadata'].get('source', '') for doc in relevant_docs],
        'confidence': 0.6,
        'timing_breakdown': timing,
        'metadata': {'fallback_strategy': 'retry_with_context'}
    }

def _handle_no_match_response(
    self, 
    query: str, 
    session_context: Optional[SessionContext] = None
) -> Dict[str, Any]:
    """Handle when no matching documents found"""
    
    # Tone-aware response
    tone_response_map = {
        "frustrated": "I understand this is frustrating. Let me help you find the right information.",
        "confused": "No problem, let me clarify that for you.",
        "impatient": "Let me quickly help you with that.",
        "neutral": "I don't have that specific information, but I can help you find it.",
        "happy": "Great question! Here's what I can help with.",
    }
    
    tone = session_context.last_user_tone.value if session_context else "neutral"
    starter = tone_response_map.get(tone, tone_response_map["neutral"])
    
    # Escalation suggestions based on intent
    escalation_map = {
        "appointment": "Would you like me to connect you with our scheduling office?",
        "complaint": "I'd like to escalate this to our support team.",
        "question": "Would you like me to search our knowledge base differently?",
    }
    
    intent = session_context.last_user_intent.value if session_context else "question"
    escalation = escalation_map.get(intent, "")
    
    response_text = f"{starter} {escalation}".strip()
    
    return {
        'answer': response_text,
        'sources': [],
        'confidence': 0.0,
        'timing_breakdown': {'total_ms': 0},
        'metadata': {
            'type': 'no_match',
            'should_escalate': True,
            'intent': intent,
            'tone': tone
        }
    }
```

---

## Part 5: Context-Aware Humanization

### Key Principles

```python
def humanize_response(
    self, 
    response: str, 
    query: str,
    session_context: Optional[SessionContext] = None,
    user_intent: UserIntent = None,
    user_tone: UserTone = None,
    is_first_turn: bool = True
) -> str:
    """
    Make responses feel conversational based on context.
    
    Rules:
    1. Match user's tone (frustrated → empathetic, confused → clear)
    2. Use conversation history (reference previous turns)
    3. Personalize (use user's name if known)
    4. Add natural pauses/thinking delays
    5. Escalate gracefully if needed
    """
    
    # ===== RULE 1: Remove formality =====
    formal_starters = [
        "According to the context",
        "The knowledge base indicates",
        "As per the documentation",
        "It is stated that"
    ]
    
    for starter in formal_starters:
        if response.startswith(starter):
            response = response[len(starter):].lstrip(' ,:-')
    
    # ===== RULE 2: Tone-aware modifications =====
    if user_tone == UserTone.FRUSTRATED:
        # Add empathy
        if not any(word in response.lower() for word in ["sorry", "understand", "appreciate"]):
            response = f"I understand your concern. {response}"
        
        # Shorter, more direct sentences
        response = self._shorten_sentences(response, max_length=100)
    
    elif user_tone == UserTone.CONFUSED:
        # Add clarity
        response = self._clarify_response(response)
    
    elif user_tone == UserTone.IMPATIENT:
        # Lead with the answer
        response = self._reorganize_for_speed(response)
    
    # ===== RULE 3: Personalization =====
    if session_context and session_context.user_name:
        # Use name naturally in response
        if "you" in response.lower():
            # Personalize greeting
            if is_first_turn:
                response = f"Hi {session_context.user_name}, {response}"
    
    # ===== RULE 4: Reference conversation history =====
    if session_context and len(session_context.turns) > 1:
        # Reference previous context naturally
        prev_turn = session_context.turns[-2]
        if prev_turn.role == "user":
            # Create bridge to previous context
            response = self._bridge_to_context(response, prev_turn.text)
    
    # ===== RULE 5: Add thinking delay metadata =====
    # (For speech systems to add natural pauses)
    thinking_delay = 0.5  # Default 500ms
    if user_intent == UserIntent.APPOINTMENT:
        thinking_delay = 1.0  # Longer for complex questions
    elif user_tone == UserTone.CONFUSED:
        thinking_delay = 0.8  # Show we're thinking
    
    # ===== Format final response =====
    # Max length enforcement
    if len(response) > self.config.max_response_length:
        response = self._truncate_intelligently(response, self.config.max_response_length)
    
    # Ensure proper punctuation
    if response and response[-1] not in '.!?':
        response += '.'
    
    return response
```

---

## Part 6: Integration with FastAPI

### Updated endpoint

```python
@app.post("/api/v1/query_with_context")
async def query_with_session_context(
    session_id: str,
    query: str,
    user_name: Optional[str] = None
):
    """
    Query endpoint that maintains conversation context.
    
    Returns: {answer, sources, confidence, metadata, thinking_delay}
    """
    
    # Get or create session
    if session_id not in app.state.sessions:
        app.state.sessions[session_id] = SessionContext(
            session_id=session_id,
            user_name=user_name
        )
    
    session = app.state.sessions[session_id]
    
    # Update user name if provided
    if user_name:
        session.user_name = user_name
    
    # Process query
    result = await app.state.rag_engine.process_query(
        query=query,
        session_context=session
    )
    
    # Add thinking delay for speech
    result['thinking_delay_ms'] = 500
    
    return QueryResponse(
        answer=result['answer'],
        sources=result['sources'],
        confidence=result['confidence'],
        timing_breakdown=result['timing_breakdown'],
        cached=False,
        metadata={
            **result['metadata'],
            'session_id': session_id,
            'interaction_count': session.interaction_count,
            'thinking_delay_ms': result.get('thinking_delay_ms', 500)
        }
    )
```

---

## Summary: What This Achieves

| Before | After |
|--------|-------|
| Response: "The office is open 9-5" | Response: "Thanks for your patience! The registrar's office is open 9-5 on weekdays. Would you like me to schedule an appointment?" |
| No personalization | Uses user's name, references past turns |
| Same tone regardless | Matches user tone (empathetic for frustrated, clearer for confused) |
| Crashes on edge cases | Graceful fallbacks + escalation |
| Forgets previous context | Maintains full conversation history |
| Generic confidence | Intent-aware confidence scoring |

---

## Implementation Timeline

**Phase 1 (2-3 hours):**
- Add SessionContext class
- Add IntentToneDetector
- Integrate into process_query()

**Phase 2 (2 hours):**
- Enhanced prompting with context
- Humanization layer

**Phase 3 (1-2 hours):**
- Fallback strategies
- Edge case handling

**Phase 4 (1 hour):**
- FastAPI integration
- Session management

**Total: 6-8 hours to production**

This transforms responses from **robotic retrieval** to **context-aware, human-like conversation**.
