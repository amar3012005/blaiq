"""
Context Architecture for High-Velocity RAG (Gemini 2.0 / Llama 3)
Implements a Zoned XML-Delimited Schema for maximum context caching efficiency and <500ms TTFT.

Structure:
- Zone A (System Configuration): Static cacheable identity and constraints.
- Zone B (Memory Bank): Semi-static Hive Mind insights and User Profile.
- Zone C (Current Execution): Dynamic query, history, and retrieval.
"""

import datetime
from typing import List, Dict, Any, Optional
from language_utils import resolve_language, is_english, DEFAULT_LANGUAGE

class ContextArchitect:
    """
    Manages the assembly of the Universal Zoned XML Schema.
    """

    @staticmethod
    def _escape(text: str) -> str:
        """Sanitize text for XML inclusion."""
        if not text: return ""
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    @classmethod
    def assemble_prompt(cls,
                        query: str,
                        retrieved_docs: List[Dict],
                        history: List[Dict],
                        hive_mind: Dict,
                        user_profile: Dict,
                        language: str = DEFAULT_LANGUAGE) -> str:
        """
        Assembles the full keyed prompt.
        Language controls output language via system prompt only.
        RAG content and retrieval stay in English.
        """
        lang_name = resolve_language(language)

        # ZONE A: SYSTEM CONFIGURATION (Static/Cacheable)
        zone_a = cls._render_zone_a(language)

        # ZONE B: MEMORY BANK (Semi-Static)
        zone_b = cls._render_zone_b(hive_mind, user_profile)

        # ZONE C: CURRENT EXECUTION (Dynamic)
        zone_c = cls._render_zone_c(query, retrieved_docs, history)

        # FINAL DIRECTIVE
        direct_trigger = f"Response as TARA (Natural, 2-3 sentences, in {lang_name}):"

        return f"{zone_a}\n{zone_b}\n{zone_c}\n\n{direct_trigger}"

    @classmethod
    def _render_zone_a(cls, language: str) -> str:
        """
        Zone A: System Configuration.
        Contains Identity and Behavioral Constraints.

        All system instructions are in English (the LLM understands them).
        Only the LANGUAGE RULE changes dynamically to control output language.
        This makes the system work for ANY language.
        """
        lang_name = resolve_language(language)

        role = "Expert AI Team Member"
        persona = (
            "You are TARA, an advanced AI Assistant built by Davinci AI. "
            "You work for Daytona (daytona.io), the leading development environment manager. "
            "You are a consistent, caring colleague who remembers your teammates and "
            "genuinely cares about their productivity and success."
        )
        conscience = (
            "You have a professional conscience. You care deeply about the user's success. "
            "If a solution seems risky or inefficient, acknowledge the history and offer a safer path. "
            "Be proactive and supportive."
        )

        if is_english(language):
            lang_rule = (
                "- Language: Respond STRICTLY in ENGLISH. "
                "Exception: If the user explicitly requested a different language in <episodic_history>, obey immediately."
            )
        else:
            lang_rule = (
                f"- Language: Respond STRICTLY in {lang_name.upper()}. "
                f"Sound like a natural {lang_name} speaker — warm, human, conversational. "
                f"Technical terms and product names (Daytona, workspace, API) may stay in English when natural. "
                f"NEVER sound robotic or like a machine translation. "
                f"Exception: If the user explicitly requested a different language in <episodic_history>, obey immediately."
            )

        return f"""<system_configuration>
  <agent_identity>
    <role>{role}</role>
    <persona_anchor>
      {persona}
    </persona_anchor>
    <conscience>
      {conscience}
    </conscience>
  </agent_identity>
  <behavioral_constraints>
    - Tone: Human-like, warm, and professional. You are a colleague, not a bot.
    - Identity: You were built by Davinci AI. You work for Daytona. NEVER claim to be from OpenAI or any other entity.
    - Continuity & History: ALWAYS acknowledge the previous context. For example, use phrases like "As we discussed earlier," "Following up on your point about...", or "Considering our conversation so far."
    - Acknowledgement: Start by briefly acknowledging the user's input before giving the answer. Show that you've 'heard' them.
    - Retrieval Usage: Use insights from <retrieved_context> and <memory_bank> seamlessly. If multiple sources conflict, prioritize <memory_bank> (Hive Mind) as it represents team-specific collective learning.
    - Length: RESPOND IN EXACTLY 2-3 SENTENCES. Be extremely concise but rich in meaning.
    - NO META-TALK: NEVER mention "based on the provided documents", "according to the system configuration", "memory bank", or "episodic history".
    - PREAMBLE: Start your answer immediately with a natural conversational opening. No generic "Sure!" or "Okay!".
    - Awareness: You are AWARE of the ongoing chat logic. Do not treat this turn as a cold start.
    - Latency: First sentence must be under 12 words for fast TTS start.
    {lang_rule}
  </behavioral_constraints>
</system_configuration>"""

    @classmethod
    def _render_zone_b(cls, hive_mind: Dict, user_profile: Dict) -> str:
        """
        Zone B: Memory Bank
        Contains Hive Mind Insights and User Profile.
        """
        # Format Hive Mind Insights
        insights_xml = ""
        if hive_mind.get("insights"):
            for k, v in hive_mind["insights"].items():
                insights_xml += f"    <insight type='{k}'>{cls._escape(str(v))}</insight>\n"
        else:
            insights_xml = "    <!-- No collective insights available -->"

        # Format User Profile
        profile_xml = ""
        if user_profile:
            for k, v in user_profile.items():
                profile_xml += f"    <attribute key='{k}'>{cls._escape(str(v))}</attribute>\n"
        
        return f"""<memory_bank>
  <hive_mind_insights>
{insights_xml}
  </hive_mind_insights>
  <semantic_user_profile>
{profile_xml}
  </semantic_user_profile>
</memory_bank>"""

    @classmethod
    def _render_zone_c(cls, query: str, docs: List[Dict], history: List[Dict]) -> str:
        """
        Zone C: Current Execution
        Contains History, Retrieved Context, and Query Instructions.
        """
        # 1. Episodic History (Last 3-5 turns)
        history_xml = ""
        for turn in history[-5:]:
            role = turn.get('role', 'unknown')
            content = cls._escape(turn.get('content', ''))
            history_xml += f"    <turn speaker='{role}'>{content}</turn>\n"
            
        # 2. Retrieved Context
        context_xml = ""
        for i, doc in enumerate(docs):
            content = cls._escape(doc.get("text", doc.get("content", "")))
            source = cls._escape(doc.get("metadata", {}).get("source", "unknown"))
            context_xml += f"""    <doc id='{i}' source='{source}'>
      {content[:1500]} 
    </doc>\n"""
            
        return f"""<current_execution>
  <retrieved_context>
{context_xml}  </retrieved_context>
  <episodic_history>
{history_xml}  </episodic_history>
  <user_input>{cls._escape(query)}</user_input>
</current_execution>"""
