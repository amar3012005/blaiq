"""
Distillation Prompt and Logic for Hive Mind Case Saving
Handles GDPR-compliant anonymization and multi-issue segmentation.
"""

import json
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DISTILL_PROMPT = """
Analyze the following conversation history between a User and TARA (a Daytona expert).
Your goal is to extract high-quality, professional knowledge cases for the "Hive Mind" collective intelligence system.

### RULES:
1. **Multi-Issue Segmentation**: If the conversation covers multiple distinct technical topics or problems, extract EACH one as a separate case.
2. **GDPR / Anonymization**: 
   - REMOVE all Personal Identifiable Information (PII).
   - Replace user names, emails, phone numbers, or specific company names with generic terms (e.g., "the user", "an external system").
   - Focus strictly on the technical problem and the solution provided.
3. **Structure**: 
   - 'issue': A clear, one-sentence description of the technical challenge or question.
   - 'solution': A concise, technical explanation of the answer or fix provided (2-3 sentences).
4. **Tone**: Professional, technical, and objective.
5. **No Noise**: Do not include greetings, small talk, or meta-comments about the conversation.

### OUTPUT FORMAT:
You MUST output ONLY a valid JSON list of objects.
Example:
[
  {"issue": "How to create a workspace using the Daytona CLI?", "solution": "Use the 'daytona create' command followed by the repository URL. This initializes a managed development environment automatically."},
  {"issue": "Where is the global configuration file stored?", "solution": "The Daytona configuration is typically located at ~/.daytona/config.yaml on Linux and macOS."}
]

### CONVERSATION HISTORY:
{history}

OUTPUT (JSON LIST ONLY):
"""

class CaseDistiller:
    @staticmethod
    def clean_json_response(response_text: str) -> List[Dict[str, str]]:
        """Extract and parse JSON list from LLM response."""
        try:
            # Handle potential markdown blocks
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            
            # Fallback: try parsing whole text
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            return [data] if isinstance(data, dict) else []
            
        except Exception as e:
            logger.error(f"Failed to parse distilled cases: {e}")
            return []

    @staticmethod
    def get_prompt(history: str) -> str:
        return DISTILL_PROMPT.format(history=history)
