"""
OpenAI LLM Provider

Implementation of LLMProvider for OpenAI models (GPT-3.5, GPT-4, etc.)
"""

import logging
from typing import Generator
from openai import OpenAI
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""
    
    def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            self._client = OpenAI(api_key=self.api_key)
            logger.info(f"✅ OpenAI initialized: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """Generate response using OpenAI."""
        if not self.is_available():
            raise RuntimeError("OpenAI provider not initialized")
        
        try:
            # System message can be customized
            system_message = kwargs.get("system_message", "You are TARA, a helpful AI assistant for Daytona.")
            
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            if stream:
                # Streaming mode
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return stream_generator()
            else:
                # Non-streaming mode
                return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is ready."""
        return self._client is not None
