"""
Google Gemini LLM Provider - Modern google-genai SDK
"""

import logging
from typing import Generator, Optional, Any

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation using the modern google-genai SDK."""
    
    _client_instance: Optional['genai.Client'] = None

    def initialize(self) -> bool:
        """Initialize Gemini client."""
        try:
            # Singleton-like initialization for the genai.Client
            if GeminiProvider._client_instance is None:
                GeminiProvider._client_instance = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(api_version="v1beta")
                )
            
            # For back-compat with the original structure
            self._client = GeminiProvider._client_instance
            logger.info(f"✅ Gemini initialized (google-genai): {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """Generate response using Gemini."""
        if not self.is_available():
            raise RuntimeError("Gemini provider not initialized")
        
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            if stream:
                # Streaming mode
                def stream_generator():
                    # Note: generate_content is the method in google-genai
                    for chunk in self._client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    ):
                        if chunk.text:
                            yield chunk.text
                
                return stream_generator()
            else:
                # Non-streaming mode
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
        
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Gemini is ready."""
        return self._client is not None
