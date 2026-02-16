"""
Anthropic Claude LLM Provider

Implementation of LLMProvider for Anthropic's Claude models.
"""

import logging
from typing import Generator
from anthropic import Anthropic
from .base import LLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def initialize(self) -> bool:
        """Initialize Claude client."""
        try:
            self._client = Anthropic(api_key=self.api_key)
            logger.info(f"✅ Claude initialized: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Claude initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """Generate response using Claude."""
        if not self.is_available():
            raise RuntimeError("Claude provider not initialized")
        
        try:
            if stream:
                # Streaming mode
                with self._client.messages.stream(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ) as stream_response:
                    def stream_generator():
                        for text in stream_response.text_stream:
                            yield text
                    
                    return stream_generator()
            else:
                # Non-streaming mode
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude is ready."""
        return self._client is not None
