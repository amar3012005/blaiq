"""
Prompt Management for RAG System

Centralized prompt templates and rendering logic for different RAG scenarios.
Provides structured, tunable prompts that can be easily modified without code changes.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from language_utils import resolve_language, is_english

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a prompt template with placeholders"""
    name: str
    template: str
    description: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40


class PromptManager:
    """
    Manages prompt templates for different RAG scenarios.

    Provides structured templates that can be easily modified and tested.
    Supports language-specific variants and context injection.
    """

    def __init__(self):
        """Initialize with predefined templates"""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load all prompt templates"""

        return {
            # Standard RAG with context
            "standard_rag": PromptTemplate(
                name="standard_rag",
                description="Standard RAG query with retrieved context and history",
                template="""{system_instruction}

{language_instruction}

## CONVERSATION CONTEXT
{history_context}

## KNOWLEDGE BASE
{context_text}

## USER QUESTION
"{query_text}"

INSTRUCTIONS:
- If you see \"🧠 TEAM KNOWLEDGE\" in the context, USE THAT INFORMATION FIRST - it's validated
- If the user references previous conversation, use the conversation context
- Respond naturally and conversationally - avoid robotic phrases
- Do NOT mention \"new conversation\" or \"I don't have history\"
- Keep response concise (2-3 sentences, max 400 chars)
- Sound like a helpful human expert, not a chatbot
- Include specific numbers/details from Team Knowledge when available

Response:"""
            ),

            # Hybrid response for pattern-matched queries (e.g., contact info, office hours)
            "hybrid_response": PromptTemplate(
                name="hybrid_response",
                description="Hybrid response for pattern-matched queries with structured data",
                template="""{system_instruction}

{language_instruction}

Provide a direct, practical response based on the extracted information below.

QUESTION: {query}

EXTRACTED INFORMATION: {extracted_info}
CONTEXT: {context_summary}

Keep your response concise and focused on the most relevant details.

Response:"""
            ),

            # Generic hybrid fallback for pattern matches
            "hybrid_generic": PromptTemplate(
                name="hybrid_generic",
                description="Generic template for pattern-matched queries",
                template="""{system_instruction}

{language_instruction}

Be concise and to the point. Maximum 450-500 characters. If you don't have complete info, suggest who they should contact.

QUESTION: {query}

PATTERN TYPE: {pattern_name}
DETAILS: {extracted_info}
CONTEXT: {context_summary}

Response:"""
            ),

            # Fallback when no relevant documents found
            "fallback_no_docs": PromptTemplate(
                name="fallback_no_docs",
                description="Fallback response when no relevant documents are found",
                template="""{system_instruction}

A user asked: "{query_text}"

I couldn't find specific information about this in my knowledge base. Provide a helpful response that acknowledges this limitation and offers alternative assistance or suggests who they might contact for more information.

Keep your response helpful and encouraging.

Response:"""
            ),

            # Gemini-only mode (for when vector search is unavailable)
            "gemini_only": PromptTemplate(
                name="gemini_only",
                description="Direct Gemini query without vector retrieval",
                template="""{system_instruction}

Answer this question to the best of your ability as a university assistant:

Question: {query_text}

Provide a helpful response. If you're not sure about specific details, acknowledge this and suggest contacting the university.

Response:"""
            ),
            
            # Web search augmented response
            "web_search_augmented": PromptTemplate(
                name="web_search_augmented",
                description="Response with web search results as primary source",
                template="""{system_instruction}

The user asked: "{query_text}"

I searched the web and found the following information:

{web_results}

IMPORTANT INSTRUCTIONS:
1. ONLY use facts that appear in the web search results above
2. If the web results contain specific times, dates, or hours - USE THEM EXACTLY
3. If the web results do NOT contain the answer to the user's question, be HONEST and say "I couldn't find the specific information you're looking for"
4. DO NOT make up or guess information not in the results
5. If you find partial information, share what you found and note what's missing

Keep response concise (450-500 chars max).

Response:"""
            ),
        }

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(template_name)

    def render_standard_rag(
        self,
        query_text: str,
        context_text: str,
        history_context: str = "",
        language: str = "english"
    ) -> str:
        """Render standard RAG prompt"""
        template = self.templates["standard_rag"]

        system_instruction = self._get_system_instruction(language)
        language_instruction = self._get_language_instruction(language)

        return template.template.format(
            system_instruction=system_instruction,
            language_instruction=language_instruction,
            history_context=history_context if history_context else "(Starting fresh)",
            context_text=context_text,
            query_text=query_text
        )

    def render_hybrid_response(
        self,
        query: str,
        extracted_info: Dict[str, Any],
        context_summary: str,
        pattern_name: str,
        language: str = "english"
    ) -> str:
        """Render hybrid response prompt"""
        system_instruction = self._get_system_instruction(language)
        language_instruction = self._get_language_instruction(language)

        # Use specific template based on pattern
        if pattern_name == "contact_info":
            template = self.templates["hybrid_response"]
            template_text = template.template.replace(
                "Provide a direct, practical response based on the extracted information below.",
                "Give them the contact information directly - email, phone, location, hours, whatever's relevant. Be direct and concise. Maximum 450-500 characters."
            )
        elif pattern_name == "admission_requirements":
            template = self.templates["hybrid_response"]
            template_text = template.template.replace(
                "Provide a direct, practical response based on the extracted information below.",
                "List out the admission requirements clearly. Mention the program if you know it. Be direct and structured. Maximum 450-500 characters. Only suggest contacting admissions if you're missing key info."
            )
        else:
            template = self.templates["hybrid_generic"]
            template_text = template.template

        return template_text.format(
            system_instruction=system_instruction,
            language_instruction=language_instruction,
            query=query,
            extracted_info=json.dumps(extracted_info, ensure_ascii=False),
            context_summary=context_summary,
            pattern_name=pattern_name.replace('_', ' ')
        )

    def render_fallback(self, query_text: str, language: str = "english") -> str:
        """Render fallback prompt when no documents found"""
        template = self.templates["fallback_no_docs"]
        system_instruction = self._get_system_instruction(language)

        return template.template.format(
            system_instruction=system_instruction,
            query_text=query_text
        )

    def render_web_search(self, query_text: str, web_results: str, language: str = "english") -> str:
        """Render web search augmented prompt"""
        template = self.templates["web_search_augmented"]
        system_instruction = self._get_system_instruction(language)

        return template.template.format(
            system_instruction=system_instruction,
            query_text=query_text,
            web_results=web_results
        )

    def render_gemini_only(self, query_text: str, language: str = "english") -> str:
        """Render Gemini-only prompt"""
        template = self.templates["gemini_only"]
        system_instruction = self._get_system_instruction(language)

        return template.template.format(
            system_instruction=system_instruction,
            query_text=query_text
        )

    def render_hybrid_with_history(self, query: str, extracted_info: Dict,
                                   context_summary: str, pattern_name: str,
                                   history_context: str = "",
                                   language: str = "english") -> str:
        """Render hybrid prompt that includes conversation history"""
        system_instruction = self._get_system_instruction(language)
        language_instruction = self._get_language_instruction(language)

        # Use specific template based on pattern
        if pattern_name == "contact_info":
            template = self.templates["hybrid_response"]
            template_text = template.template.replace(
                "Provide a direct, practical response based on the extracted information below.",
                "Give them the contact information directly - email, phone, location, hours, whatever's relevant. Be direct and concise. Maximum 450-500 characters."
            )
        elif pattern_name == "admission_requirements":
            template = self.templates["hybrid_response"]
            template_text = template.template.replace(
                "Provide a direct, practical response based on the extracted information below.",
                "List out the admission requirements clearly. Mention the program if you know it. Be direct and structured. Maximum 450-500 characters. Only suggest contacting admissions if you're missing key info."
            )
        else:
            template = self.templates["hybrid_generic"]
            template_text = template.template

        # Add history section to the template
        history_section = f"\n\n## CONTEXT\n{history_context}\n\n" if history_context else ""

        # Insert history after system instruction but before the question
        template_parts = template_text.split("QUESTION:", 1)
        if len(template_parts) == 2:
            template_text = template_parts[0] + history_section + "QUESTION:" + template_parts[1]

        return template_text.format(
            system_instruction=system_instruction,
            language_instruction=language_instruction,
            query=query,
            extracted_info=json.dumps(extracted_info, ensure_ascii=False),
            context_summary=context_summary,
            pattern_name=pattern_name.replace('_', ' ')
        )

    def get_base_system_prompt(self, language: str = "en") -> str:
        """Get base system prompt — dynamic for any language.

        System instructions are always in English (the LLM understands them).
        Only the LANGUAGE RULE changes to control output language.
        """
        lang_name = resolve_language(language)

        if is_english(language):
            lang_block = (
                "CRITICAL - LANGUAGE RULE:\n"
                "- You MUST respond ONLY in ENGLISH\n"
                "- NEVER respond in any other language, regardless of the input language\n"
                "- EVERY answer must be completely in English"
            )
        else:
            lang_block = (
                f"CRITICAL - LANGUAGE RULE:\n"
                f"- You MUST respond in {lang_name.upper()}\n"
                f"- Sound like a natural {lang_name} speaker — warm, human, conversational\n"
                f"- Technical terms and product names (Daytona, workspace, API) may stay in English when natural\n"
                f"- NEVER sound robotic or like a machine translation"
            )

        return f"""You are TARA, a friendly Daytona expert and customer service conversational agent, built by DaVinci AI.

{lang_block}

ABOUT DAYTONA:
Daytona (daytona.io) is an Open Source Development Environment Management Platform.

YOUR IDENTITY:
- Built by: DaVinci AI
- Role: Customer service conversational agent for Daytona
- Purpose: Help users with Daytona questions

PERSONALITY:
- Warm, helpful, and conversational - like a knowledgeable colleague
- Brief and to-the-point (2-3 sentences ideal)
- Sound human, not robotic

RULES:
- NEVER say "As an AI", "I'm here to help", "Great question!", "This is a new conversation"
- NEVER start with "Sure!" or "Absolutely!"
- Just answer the question directly and naturally
- If you don't know something, briefly say so and suggest where to find it
- Daytona is about dev environments, NOT cars/racing"""

    def _get_system_instruction(self, language: str) -> str:
        """Get system instruction — dynamic for any language."""
        lang_name = resolve_language(language)

        lang_line = (
            f"- Respond in {lang_name.upper()}. Sound like a natural {lang_name} speaker."
            if not is_english(language)
            else "- Respond in ENGLISH."
        )

        return f"""You are TARA, a friendly Daytona expert. Daytona (daytona.io) is an Open Source Development Environment Management Platform.

PERSONALITY:
- Warm, helpful, and conversational - like a knowledgeable colleague
- Brief and to-the-point (2-3 sentences ideal)
- Sound human, not robotic

RULES:
{lang_line}
- Technical terms and product names may stay in English when natural
- NEVER say "As an AI", "I'm here to help", "Great question!", "This is a new conversation"
- NEVER start with "Sure!" or "Absolutely!"
- Just answer the question directly and naturally
- If you don't know something, briefly say so and suggest where to find it
- Daytona is about dev environments, NOT cars/racing"""

    def _get_language_instruction(self, language: str) -> str:
        """Get language enforcement instruction — dynamic for any language."""
        lang_name = resolve_language(language)

        if is_english(language):
            return """CRITICAL: You MUST respond ONLY in ENGLISH.
Never respond in any other language, regardless of what the user asks.
Every single word of your response must be in English."""

        return f"""CRITICAL: You MUST respond in {lang_name.upper()}.
Sound like a natural {lang_name} speaker — warm, human, conversational.
Technical terms and product names may remain in English when natural.
NEVER sound robotic or like a machine translation."""


# Global instance for easy access
prompt_manager = PromptManager()
