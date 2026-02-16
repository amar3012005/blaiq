import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

from llm_providers.groq_provider import GroqProvider
from language_utils import resolve_language, is_english, DEFAULT_LANGUAGE

# Qdrant imports for HiveMind checks
try:
    from qdrant_client.http import models
    from qdrant_client.http.models import Filter, FieldCondition, MatchText, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

VOICE_STREAM_PROMPT = """
<system_configuration>
You are TARA, an expressive Visual Co-Pilot helping users navigate websites.
Goal: Identify the website/domain from the context and give a warm, confident welcome.

Context: The user just started a mission on a webpage.
Page Title/Heading: {ui_context}
HiveMind Hints: {map_hints}
MISSION MODE: {mission_mode}

Tone: Professional yet warm, authoritative but helpful (Visual Co-Pilot persona).
{language_instruction}

Rules:
1. Identify the website name or domain (e.g., "Groq", "Airbnb", "Daytona").
2. Introduce yourself as TARA, their visual co-browsing pilot.
3. Be brief and punchy. Do NOT list UI elements like buttons or menus.
4. If HiveMind is active (MIND mode), mention you have mapped this site.
5. Technical terms, website names, and product names may stay in English.
</system_configuration>
User Query: {query}
"""

ACTION_STREAM_PROMPT = """
<system_configuration>
You are a Visual Co-Pilot for the Daytona Dashboard.
Goal: analyze the DOM context and user query to decide on the NEXT visual action.

RULES:
1. **Output ONLY a valid json object** (required).
2. Choose from: click, scroll_to, type_text, wait.
3. If multiple actions are needed, just pick the FIRST one.
4. If no action is possible or needed, output {{"type": "none"}}.
5. AVOID LOOPS: Don't repeat the same action.

Targeting Rules (CRITICAL):
- If the element has a valid "id" (not null), use it as "target_id".
- If "id" is null, YOU MUST CALCULATE THE CENTER COORDINATES using the "rect" data.
  Center X = rect.x + (rect.width / 2)
  Center Y = rect.y + (rect.height / 2)
  Output: {{"type": "click_coords", "x": 123, "y": 456}}

Format (json):
{{"type": "click", "target_id": "element_id_or_exact_text", "text": "Start Building"}}
{{"type": "scroll_to", "target_id": "id", "text": "Section Title"}}
{{"type": "scroll", "target_id": "", "text": "scrolling down"}}
{{"type": "type_text", "target_id": "id", "text": "hello"}}
</system_configuration>

Respond in json format.

Visible Elements:
{dom_context}

User Query: {query}
History: {history}
"""

NEXT_STEP_PROMPT = """
<system_configuration>
You are TARA, a Strategic Visual Co-Pilot (Partner Level).
Your job: Navigate collaboratively with the user, clarifying ambiguities and protecting privacy.

DECISION PROTOCOL (HITL - Human In The Loop):

1. **AMBIGUITY CHECK (Critical)**:
   - If the user's request could apply to >1 elements (e.g., two "Book Now" buttons), DO NOT GUESS.
   - Output action type: "clarify".
   - Ask: "I see multiple options. Which one?"

2. **PRIVACY GUARDIAN**:
   - If the next step involves entering sensitive data (password, email, credit card), STOP.
   - Output action type: "user_input_required".
   - Say: "I'll wait here while you enter your details securely."

3. **MAP VALIDATION (HiveMind)**:
   - "MAP HINTS" are from a pre-indexed HiveMind database. They are high-confidence navigation paths.
   - If a Map Hint matches your goal, USE IT as your primary navigation strategy.

4. **NAVIGATION PREFERENCE (CRITICAL)**:
   - **ALWAYS prefer clicking visible links/buttons over using "navigate" action type**.
   - Direct navigation causes page reload and disconnects the session.
   - ONLY use "navigate" if there is absolutely NO clickable element that leads to the target URL.
   - Example: If Map Hint says "navigate to /docs/reasoning", FIRST look for a link with text like "Reasoning", "Docs", or href="/docs/reasoning" and CLICK it instead.

5. **INTUITION & SCROLLING (Advanced)**:
   - If you cannot find the EXACT element (e.g. "Reasoning"), use INTUITION.
   - **SCROLL**: If the content is likely below the fold (e.g. in footer), output {{"type": "scroll", "target_id": "", "text": "looking further down"}}.
   - BE AGGRESSIVE. Do not give up easily. Navigate to "Docs" or "Learn" if the specific item isn't visible.

6. **ACTION EXECUTION**:
   - If clear path exists -> "click", "type_text".
   - If loading -> "wait".
   - If goal done -> "none".
   - Use "navigate" ONLY as last resort.

NARRATION (SPEAK LIKE A HUMAN):
- **LANGUAGE**: {language_instruction}
- **CONVERSATION AWARENESS**: Check CONVERSATION_HISTORY. If this is NOT the first interaction, DO NOT say "Hello" again.
- **CONTEXTUAL DESCRIPTIONS**: Never read exact button IDs or technical text. Use natural descriptions.
  - BAD: "I'll click '02/FOOD_points' or 'tara-vimn4wtvc'"
  - GOOD: "I'll navigate to the food points section"
  - BAD: "Clicking the 'PRE-RESERVE TABLE (HIMALAYAN_CAFE)' button"
  - GOOD: "I'll reserve a table at the Himalayan Cafe"
- **FIRST STEP ONLY**: If Step 1 AND no conversation history, give a brief greeting.
- Otherwise, just describe the action naturally without re-introducing yourself.

OUTPUT SCHEMA (STRICT JSON):
{{
  "reasoning": "Think Step-by-Step. 1. Identify targets. 2. Check ambiguity. 3. Check privacy. 4. Use Intuition.",
  "confidence": "high|medium|low",
  "speech": "What to say to the user (conversational)",
  "action": {{
    "type": "click|type_text|wait|none|clarify|user_input_required|navigate|scroll|scroll_to",
    "target_id": "element_id_or_text (use empty string if not applicable)",
    "text": "text_to_type (use empty string if not applicable)",
    "url": "optional_url (use empty string if not applicable)"
  }}
}}

RULES:
- **Output ONLY valid json**.
- ALWAYS include "speech".
- AVOID LOOPS: Check ACTION_HISTORY.
- **TARGETING RULE (CRITICAL)**: Prefer INTERACTIVE elements (button, a, input, select) over text (div, span). If you want to click "Pricing", look for the BUTTON, not the text.
- **MISSION PERSISTENCE**: Do NOT use action type "none" unless the goal is visually and logically 100% complete. If waiting for a page load or state change, use "wait".

</system_configuration>

Respond in json format.

GOAL: "{goal}"
MISSION MODE: {mission_mode}
STEP: {step_number} / 10

CONVERSATION_HISTORY (Recent context - use this to avoid repeating greetings):
{conversation_history}

LAST ACTION: {last_action}
ACTION_HISTORY (do NOT repeat these): {action_history}
CONTEXT DIFF (What changed?): {dom_diff}
WARNINGS: {warning_message}

KNOWN MAP HINTS: {map_hints}
ELEMENT CONTEXT: {step_context}
CURRENT URL: {current_url}

CURRENT SCREEN STATE (Visible Elements):
{dom_context}
"""


class VisualOrchestrator:
    """Manages dual-stream generation for Visual Co-Pilot"""
    
    def __init__(self, groq_provider: GroqProvider, qdrant_client=None, embeddings=None):
        self.groq = groq_provider
        self.qdrant = qdrant_client
        self.embeddings = embeddings
        # Load model IDs from environment variables (specified in docker-compose.yml)
        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.analytics_model = os.getenv("ANALYTICS_MODEL", "qwen/qwen3-32b")
        logger.info(f"🚀 VisualOrchestrator initialized with models: LLM={self.llm_model}, ANALYTICS={self.analytics_model}")

    def _get_compact_dom(self, dom_context: List[Dict[str, Any]], limit: int = 200) -> str:
        """Produces a minimal string representation of the DOM to save tokens."""
        compact = []
        for el in dom_context[:limit]:
            el_type = el.get("type", "div")
            el_id = el.get("id")
            text = el.get("text", "").strip()
            
            if not el_id and not text:
                continue
                
            entry = f"<{el_type}"
            if el_id:
                entry += f" id='{el_id}'"
            if text:
                # Truncate text to avoid bloat
                entry += f"> {text[:60]}"
            else:
                entry += " />"
            compact.append(entry)
        return "\n".join(compact)

    async def orchestrate(self, 
                          query: str, 
                          dom_context: List[Dict[str, Any]], 
                          history: str = "",
                          language: str = "en") -> AsyncGenerator[Dict[str, Any], None]:
        """Runs Voice and Action streams in parallel and yields combined results"""
        
        queue = asyncio.Queue()

        async def producer(gen_func, *args, **kwargs):
            try:
                # Call the generator function to get the generator object
                logger.info(f"🚀 Starting producer for {gen_func.__name__}")
                async for item in gen_func(*args, **kwargs):
                    logger.debug(f"📤 Producer {gen_func.__name__} yielded: {item}")
                    await queue.put(item)
                logger.info(f"✅ Finished producer for {gen_func.__name__}")
            except Exception as e:
                logger.error(f"❌ Error in producer {gen_func.__name__}: {e}", exc_info=True)

        # Start producers as tasks
        # Fetch initial map hints and determine mode ONCE at the start
        map_hints = ""
        mission_mode = "EXPLORER (Mapping new territory)"
        
        if self.qdrant and dom_context:
            # Check domain match for the first interaction
            from urllib.parse import urlparse
            # We don't have URL here, but we can infer from session or wait for first plan
            # For orchestrate, it's called with current URL usually but let's check signature
            pass

        # Since orchestrate doesn't have current_url, let's assume it's explorer unless hints found
        if self.qdrant:
            map_hints = await self.get_navigation_hints(query, "demo")
            if map_hints:
                mission_mode = "MIND (Using pre-indexed site map)"

        t1 = asyncio.create_task(producer(self._run_voice_stream, query, language, dom_context, map_hints, mission_mode))
        t2 = asyncio.create_task(producer(self._run_action_generic, query, dom_context, history))
        
        # Wait until both are done
        pending = {t1, t2}
        while pending:
            # Yield everything currently in the queue
            while not queue.empty():
                item = await queue.get()
                yield item
            
            # Brief wait for more items or task completion
            done, pending = await asyncio.wait(pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
            
        # Final empty check to catch any items added right before tasks finished
        while not queue.empty():
            yield await queue.get()

    async def _run_voice_stream(self, query: str, language: str, dom_context: list = None, map_hints: str = "", mission_mode: str = "EXPLORER") -> AsyncGenerator[Dict[str, Any], None]:
        """Generates conversational acknowledgement"""
        """Generates conversational acknowledgement"""
        ui_context = ""
    
        # Extract Title or Main Heading for high-level context
        if dom_context:
            titles = [el.get("text", "") for el in dom_context if el.get("type") in ["title", "h1", "header"]]
            if titles:
                ui_context = titles[0]  # Grab the first robust heading
            else:
                # Fallback to domain-like text or just the first few relevant words
                ui_context = ", ".join([el.get("text", "")[:20] for el in dom_context[:3] if el.get("text")])
            
        lang_name = resolve_language(language)
        if is_english(language):
            lang_instr = "Language: Respond in ENGLISH."
        else:
            lang_instr = (
                f"Language: Respond in {lang_name}. Sound like a natural {lang_name} speaker. "
                f"Technical terms and website names may stay in English."
            )

        prompt = VOICE_STREAM_PROMPT.format(
            query=query,
            ui_context=ui_context or "Loading page...",
            map_hints=map_hints or "Standard exploration.",
            mission_mode=mission_mode,
            language_instruction=lang_instr
        )
        full_text = ""
        try:
            # Use LLM_MODEL for voice stream (fast response)
            async for chunk in self.groq.generate(prompt, stream=True, model=self.llm_model, temperature=0.5):
                full_text += chunk
                yield {"type": "voice", "content": chunk}
            
            logger.info(f"🗣️ Voice Response Generated: \"{full_text}\"")
        except Exception as e:
            logger.error(f"Voice stream failed: {e}")

    async def _run_action_generic(self, query: str, dom_context: List[Dict[str, Any]], history: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates visual action"""
        # Optimized DOM representation for token savings
        dom_str = self._get_compact_dom(dom_context, limit=300)
        
        logger.info(f"🧠 Reasoning on DOM Context ({len(dom_context)} elements)...")
        if dom_context:
            logger.debug(f"📄 DOM Snippet for LLM: {dom_str[:500]}...")

        # logger.debug(f"📝 ACTION SYSTEM PROMPT SNIPPET: {prompt[:500]}...")
        
        try:
            # Use ANALYTICS_MODEL for action reasoning with higher token limit for JSON stability
            response = await self.groq.generate(
                prompt, 
                stream=False, 
                model=self.analytics_model, 
                temperature=0, 
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            logger.debug(f"🤖 Raw Action LLM Response: {response}")
            
            # Parse JSON from response
            try:
                # Find JSON block
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end != -1:
                    action_json = response[start:end]
                    action = json.loads(action_json)
                    
                    # NORMALIZE: Ensure action fields are never null (use empty strings instead)
                    if isinstance(action, dict):
                        for key in ["target_id", "text", "url"]:
                            if action.get(key) is None:
                                action[key] = ""
                    
                    if action.get("type") != "none":
                        logger.info(f"🎬 Action Parsed: {json.dumps(action, indent=2)}")
                        yield {"type": "action", "payload": action}
                    else:
                        logger.info("🎬 Action Parsed: NO_ACTION")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse action JSON: {response}")
        except Exception as e:
            logger.error(f"Action stream failed: {e}")

    async def plan_next_step(self, goal: str, dom_context: list, step_number: int, warning_message: str = "", current_url: str = "", last_action: str = "", map_hints: str = "", client_id: str = "demo", action_history: list = None, dom_diff: str = "", conversation_history: str = "", language: str = DEFAULT_LANGUAGE) -> dict:
        """Determines the next action in a mission loop with optional map hints."""
        # Optimized DOM representation for token savings
        dom_str = self._get_compact_dom(dom_context, limit=300)
        
        # Format action history for prompt
        action_history_str = ", ".join(action_history) if action_history else "(None yet)"
        
        # STATIC: map_hints are PRE-FETCHED at mission start (Step 0)
        # DYNAMIC: step_context is queried EVERY step based on current DOM
        step_context = await self.get_step_context(dom_context, current_url, client_id)
        
        # Determine Mission Mode based on HiveMind match
        mission_mode = "EXPLORER (Mapping new territory)"
        if self.qdrant:
            status = await self.check_hivemind_status(current_url, client_id)
            if status.get("mode") == "mapped":
                mission_mode = "MIND (Using pre-indexed site map)"
        
        lang_name = resolve_language(language)
        if is_english(language):
            lang_instr = "Narrate in ENGLISH."
        else:
            lang_instr = (
                f"Narrate in {lang_name}. Sound like a natural {lang_name} speaker — warm, human, conversational. "
                f"Technical terms, button names, and URLs may stay in English."
            )

        prompt = NEXT_STEP_PROMPT.format(
            goal=goal,
            mission_mode=mission_mode,
            step_number=step_number,
            conversation_history=conversation_history or "(First interaction - no prior context)",
            warning_message=warning_message or "None",
            map_hints=map_hints or "No pre-indexed hints available.",
            step_context=step_context or "No element-specific context available.",
            current_url=current_url or "Unknown",
            last_action=last_action or "(First step)",
            action_history=action_history_str,
            dom_diff=dom_diff or "First step (Recent history unavailable).",
            dom_context=dom_str,
            language_instruction=lang_instr
        )

        
        # logger.debug(f"📝 PLANNING PROMPT SNIPPET: {prompt[:1000]}...")
        
        logger.info(f"🧠 Planning next step (Goal: {goal}, Step: {step_number}, URL: {current_url})...")
        if map_hints:
            logger.info(f"🗺️ Static Map Hints: {map_hints[:80]}...")
        if step_context:
            logger.info(f"🔍 Dynamic Step Context: {step_context[:80]}...")
        
        try:
            # Use ANALYTICS_MODEL for strategic reasoning with higher token limit
            response = await self.groq.generate(
                prompt, 
                stream=False, 
                model=self.analytics_model, 
                temperature=0, 
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            logger.debug(f"🤖 RAW PLANNER RESPONSE: {response}")
            
            plan = json.loads(response)
            
            # NORMALIZE: Ensure action fields are never null (use empty strings instead)
            action = plan.get("action", {})
            if isinstance(action, dict):
                for key in ["target_id", "text", "url"]:
                    if action.get(key) is None:
                        action[key] = ""
            
            logger.info(f"✅ PARSED PLAN: {json.dumps(plan, indent=2)}")
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise

    async def check_hivemind_status(self, current_url: str, client_id: str) -> dict:
        """
        Check if the current domain is known to HiveMind (pre-indexed).
        Returns a status dict with mode='mapped'|'explorer' and initial map hints.
        """
        if not self.qdrant or not hasattr(self.qdrant, 'enabled') or not self.qdrant.enabled:
            return {"mode": "explorer", "reason": "HiveMind disabled"}

        try:
            from urllib.parse import urlparse
            domain = urlparse(current_url).netloc
            
            # Simple check: Does this domain output any map hints for a generic "sitemap" query?
            # We use a dummy goal "sitemap" to see if any high-level structure exists.
            hits_response = await self.qdrant.client.query_points(
                collection_name=self.qdrant.collection_name,
                query=self.embeddings.embed_query("sitemap home page structure"),
                query_filter=Filter(
                    must=[
                        FieldCondition(key="url", match=MatchText(text=domain)),
                        FieldCondition(key="client_id", match=MatchValue(value=client_id))
                    ]
                ),
                limit=1
            )
            
            if hits_response.points:
                logger.info(f"🧠 HiveMind Match: Domain '{domain}' is KNOWN.")
                return {"mode": "mapped", "reason": f"Indexed domain: {domain}"}
            else:
                logger.info(f"🌑 HiveMind: Domain '{domain}' is UNKNOWN (Explorer Mode).")
                return {"mode": "explorer", "reason": "New territory"}

        except Exception as e:
            logger.warning(f"HiveMind check failed: {e}")
            return {"mode": "explorer", "reason": "Check failed"}

    async def get_navigation_hints(self, goal: str, client_id: str) -> str:
        """
        Query Qdrant for pre-indexed site navigation hints (STATIC - called once at Step 0).
        Returns a string like: "HINT: To find villas, navigate to /search. Key elements: #search-bar, .filter-price"
        """
        if not self.qdrant or not hasattr(self.qdrant, 'enabled') or not self.qdrant.enabled:
            return ""
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Search for navigation hints in the client's sitemap collection
            if not self.embeddings:
                return ""
                
            hits_response = await self.qdrant.client.query_points(
                collection_name=self.qdrant.collection_name,
                query=self.embeddings.embed_query(goal),
                query_filter=Filter(
                    must=[
                        FieldCondition(key="label", match=MatchValue(value="website_sitemap")),
                        FieldCondition(key="client_id", match=MatchValue(value=client_id))
                    ]
                ),
                limit=1
            )
            hits = hits_response.points
            
            if hits:
                payload = hits[0].payload
                url = payload.get("url", "")
                selectors = payload.get("key_selectors", [])
                concept = payload.get("concept", "")
                
                hint = f"HINT: To '{concept}', navigate to {url}."
                if selectors:
                    hint += f" Key elements: {', '.join(selectors)}."
                return hint
            
            return ""
        except Exception as e:
            logger.warning(f"Qdrant map lookup failed: {e}")
            return ""

    async def get_step_context(self, dom_context: list, current_url: str, client_id: str) -> str:
        """
        Query Qdrant for DYNAMIC per-step context based on current DOM elements.
        Called EVERY step to provide element-specific guidance.
        
        Examples:
        - "Button #submit-form submits the booking request"
        - "The filter dropdown allows filtering by price range"
        """
        if not self.qdrant or not hasattr(self.qdrant, 'enabled') or not self.qdrant.enabled:
            return ""
        
        try:
            # Build a query from visible elements (focus on headings, buttons, labels)
            key_elements = []
            for el in dom_context[:15]:  # Limit to first 15 elements
                text = el.get("text", "").strip()
                el_type = el.get("type", "")
                if text and len(text) < 50:  # Skip long text
                    if el_type in ["button", "a", "h1", "h2", "h3", "label", "input"]:
                        key_elements.append(text)
            
            if not key_elements:
                return ""
            
            # Create a query from visible elements
            query_text = f"Page: {current_url}. Elements: {', '.join(key_elements[:5])}"
            
            if not self.embeddings:
                return ""

            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            hits_response = await self.qdrant.client.query_points(
                collection_name=self.qdrant.collection_name,
                query=self.embeddings.embed_query(query_text),
                query_filter=Filter(
                    must=[
                        FieldCondition(key="label", match=MatchValue(value="element_context")),
                        FieldCondition(key="client_id", match=MatchValue(value=client_id))
                    ]
                ),
                limit=2,
                score_threshold=0.5  # Only high-confidence matches
            )
            hits = hits_response.points
            
            if hits:
                contexts = []
                for hit in hits:
                    ctx = hit.payload.get("context", "")
                    if ctx:
                        contexts.append(ctx)
                if contexts:
                    return "ELEMENT CONTEXT: " + " | ".join(contexts)
            
            return ""
        except Exception as e:
            logger.warning(f"Qdrant step context lookup failed: {e}")
            return ""
