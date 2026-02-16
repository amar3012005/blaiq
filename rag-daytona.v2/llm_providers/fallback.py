"""
Fallback LLM Provider

Wraps multiple providers with automatic fallback on errors.
Primary provider with fallback to secondary provider(s).
"""

import logging
from typing import Generator, List
from .base import LLMProvider

logger = logging.getLogger(__name__)


class FallbackProvider(LLMProvider):
    """
    Provider that wraps multiple providers with fallback logic.
    
    Tries primary provider first, falls back to secondary on errors.
    Useful for:
    - Free tier quota limits
    - Rate limiting
    - Network errors
    - Provider downtime
    """
    
    def __init__(self, providers: List[LLMProvider]):
        """
        Initialize fallback provider.
        
        Args:
            providers: List of providers in priority order (first = primary)
        """
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        
        self.providers = providers
        self.primary = providers[0]
        self.fallbacks = providers[1:] if len(providers) > 1 else []
        
        # No direct API key/model since we wrap other providers
        super().__init__(
            api_key="fallback",
            model_name=self.primary.model_name
        )
    
    def initialize(self) -> bool:
        """Initialize all providers."""
        success_count = 0
        
        for i, provider in enumerate(self.providers):
            try:
                if provider.initialize():
                    success_count += 1
                    logger.info(
                        f"✅ Fallback tier {i+1} initialized: "
                        f"{provider.get_provider_name()} ({provider.model_name})"
                    )
                else:
                    logger.warning(
                        f"⚠️ Fallback tier {i+1} failed: "
                        f"{provider.get_provider_name()}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Fallback tier {i+1} initialization error: {e}"
                )
        
        # Consider initialized if at least one provider works
        self._client = success_count > 0
        
        if self._client:
            logger.info(
                f"🛡️ Fallback provider ready: {success_count}/{len(self.providers)} providers available"
            )
        
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """
        Generate with automatic fallback on errors.
        
        Tries providers in order until one succeeds.
        """
        if not self.is_available():
            raise RuntimeError("No providers available in fallback chain")
        
        last_error = None
        
        for i, provider in enumerate(self.providers):
            if not provider.is_available():
                logger.debug(f"Skipping unavailable provider tier {i+1}")
                continue
            
            try:
                tier_name = "PRIMARY" if i == 0 else f"FALLBACK-{i}"
                logger.info(
                    f"🔄 {tier_name}: {provider.get_provider_name()} "
                    f"({provider.model_name})"
                )
                
                result = provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwargs
                )
                
                # Success!
                if i > 0:
                    logger.warning(
                        f"⚠️ FALLBACK USED: Tier {i+1} succeeded after primary failed"
                    )
                
                return result
            
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Detect common error types
                if "quota" in error_msg or "limit" in error_msg or "rate" in error_msg:
                    logger.warning(
                        f"💸 Tier {i+1} quota/rate limit: {provider.get_provider_name()} - "
                        f"Trying next provider..."
                    )
                elif "timeout" in error_msg or "connection" in error_msg:
                    logger.warning(
                        f"⏱️ Tier {i+1} timeout: {provider.get_provider_name()} - "
                        f"Trying next provider..."
                    )
                else:
                    logger.warning(
                        f"❌ Tier {i+1} error: {provider.get_provider_name()} - {e} - "
                        f"Trying next provider..."
                    )
                
                # If this was the last provider, re-raise
                if i == len(self.providers) - 1:
                    logger.error(
                        f"🚨 ALL PROVIDERS FAILED in fallback chain ({len(self.providers)} tried)"
                    )
                    raise
        
        # Should never reach here, but just in case
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def is_available(self) -> bool:
        """Check if at least one provider is available."""
        return any(p.is_available() for p in self.providers)
    
    def get_provider_name(self) -> str:
        """Get fallback chain info."""
        provider_names = " → ".join([p.get_provider_name() for p in self.providers])
        return f"fallback({provider_names})"
    
    def get_model_info(self) -> dict:
        """Get info about all providers in chain."""
        return {
            "provider": "fallback",
            "chain": [p.get_model_info() for p in self.providers],
            "available": self.is_available()
        }
