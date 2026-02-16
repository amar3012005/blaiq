"""
Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement.
This allows easy swapping between Gemini, OpenAI, Claude, etc.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations (Gemini, OpenAI, etc.) must inherit from this
    and implement the required methods.
    """
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for the provider
            model_name: Model identifier (e.g., "gpt-4", "gemini-2.0-flash-lite")
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model_name = model_name
        self.config = kwargs
        self._client = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """
        Generate response from the LLM.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stream: Enable streaming response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            For streaming: Generator yielding text chunks
            For non-streaming: Complete response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is initialized and ready.
        
        Returns:
            True if provider is ready to use
        """
        pass
    
    def get_provider_name(self) -> str:
        """
        Get the provider name.
        
        Returns:
            Provider name (e.g., "openai", "gemini")
        """
        return self.__class__.__name__.replace("Provider", "").lower()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get provider and model information.
        
        Returns:
            Dictionary with provider details
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.model_name,
            "available": self.is_available()
        }
