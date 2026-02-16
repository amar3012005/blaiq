"""
LLM Providers Package

Modular LLM provider implementations for the RAG engine.
Supports multiple providers: Gemini, OpenAI, OpenRouter, Claude, Ollama.
"""

from .base import LLMProvider
from .gemini import GeminiProvider
from .openai_provider import OpenAIProvider
from .openrouter import OpenRouterProvider
from .claude import ClaudeProvider
from .ollama_provider import OllamaProvider
from .groq_provider import GroqProvider
from .fallback import FallbackProvider

__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ClaudeProvider",
    "OllamaProvider",
    "GroqProvider",
    "FallbackProvider"
]


def create_provider(provider_name: str, api_key: str, model_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider_name: Provider name ("gemini", "openai", "openrouter", "claude", "ollama")
        api_key: API key for the provider
        model_name: Model identifier
        **kwargs: Additional provider-specific config
        
    Returns:
        Initialized LLM provider instance
        
    Raises:
        ValueError: If provider_name is not supported
    """
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "claude": ClaudeProvider,
        "anthropic": ClaudeProvider,  # Alias
        "ollama": OllamaProvider,
        "groq": GroqProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: {', '.join(providers.keys())}"
        )
    
    provider = provider_class(api_key=api_key, model_name=model_name, **kwargs)
    if provider.initialize():
        return provider
    else:
        raise RuntimeError(f"Failed to initialize {provider_name} provider")


def create_fallback_provider(
    primary_provider: str,
    primary_api_key: str,
    primary_model: str,
    fallback_provider: str,
    fallback_api_key: str,
    fallback_model: str,
    **kwargs
) -> FallbackProvider:
    """
    Create a fallback provider with primary + fallback.
    
    Args:
        primary_provider: Primary provider name (e.g., "openrouter")
        primary_api_key: Primary API key
        primary_model: Primary model name
        fallback_provider: Fallback provider name (e.g., "gemini")
        fallback_api_key: Fallback API key
        fallback_model: Fallback model name
        **kwargs: Additional config
        
    Returns:
        FallbackProvider instance with both providers configured
        
    Example:
        # OpenRouter (free) with Gemini fallback
        provider = create_fallback_provider(
            primary_provider="openrouter",
            primary_api_key="sk-or-...",
            primary_model="nvidia/nemotron-3-nano-30b-a3b:free",
            fallback_provider="gemini",
            fallback_api_key="AIza...",
            fallback_model="gemini-2.0-flash-lite"
        )
    """
    import logging
    logger = logging.getLogger(__name__)
    
    providers_list = []
    
    # Create primary
    try:
        primary = create_provider(
            provider_name=primary_provider,
            api_key=primary_api_key,
            model_name=primary_model,
            **kwargs
        )
        providers_list.append(primary)
        logger.info(f"✅ Primary provider: {primary_provider} ({primary_model})")
    except Exception as e:
        logger.error(f"❌ Failed to create primary provider: {e}")
    
    # Create fallback
    try:
        fallback = create_provider(
            provider_name=fallback_provider,
            api_key=fallback_api_key,
            model_name=fallback_model
        )
        providers_list.append(fallback)
        logger.info(f"✅ Fallback provider: {fallback_provider} ({fallback_model})")
    except Exception as e:
        logger.error(f"❌ Failed to create fallback provider: {e}")
    
    if not providers_list:
        raise RuntimeError("Failed to create any providers for fallback chain")
    
    # Create and initialize fallback wrapper
    fallback_provider = FallbackProvider(providers=providers_list)
    fallback_provider.initialize()
    
    return fallback_provider

