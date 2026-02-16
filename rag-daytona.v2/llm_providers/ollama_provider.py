"""
Ollama LLM Provider

Implementation of LLMProvider for local Ollama models.
Runs models locally without API costs.
"""

import logging
from typing import Generator
import ollama
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local model provider implementation."""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        # Ollama doesn't need an API key, but we keep the interface consistent
        super().__init__(api_key or "local", model_name, **kwargs)
        self.host = kwargs.get("host", "http://localhost:11434")
    
    def initialize(self) -> bool:
        """Initialize Ollama client."""
        try:
            # Test connection by listing models
            models = ollama.list()
            self._client = True  # Flag that we're initialized
            logger.info(f"✅ Ollama initialized: {self.model_name} (host: {self.host})")
            logger.info(f"   Available models: {len(models.get('models', []))}")
            return True
        except Exception as e:
            logger.error(f"❌ Ollama initialization failed: {e}")
            logger.error("   Make sure Ollama is running: ollama serve")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """Generate response using Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama provider not initialized")
        
        try:
            options = {
                "temperature": temperature,
                "num_predict": max_tokens  # Ollama uses num_predict instead of max_tokens
            }
            
            if stream:
                # Streaming mode
                response_stream = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=True,
                    options=options
                )
                
                def stream_generator():
                    for chunk in response_stream:
                        if 'response' in chunk:
                            yield chunk['response']
                
                return stream_generator()
            else:
                # Non-streaming mode
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )
                return response['response']
        
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is ready."""
        return self._client is not None
