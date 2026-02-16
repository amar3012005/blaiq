#!/usr/bin/env python3
"""
Script to integrate fallback provider into rag_engine.py
Replaces Gemini-only initialization with OpenRouter+Gemini fallback.
"""

import re

# Read the file
with open('rag_engine.py', 'r') as f:
    content = f.read()

# Pattern 1: Replace the import section (add new imports)
old_import = "import google.generativeai as genai"
new_import = """import google.generativeai as genai  # Keep for backwards compat
from llm_providers import create_provider, create_fallback_provider"""

content = content.replace(old_import, new_import)

# Pattern 2: Replace Gemini initialization (lines 65-75)
old_init = """        # Initialize Gemini model
        self.gemini_model = None
        if config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(config.gemini_model)
                logger.info(f" Gemini model initialized: {config.gemini_model}")
            except Exception as e:
                logger.error(f" Gemini initialization failed: {e}")
        else:
            logger.warning("️ No Gemini API key - response generation unavailable")"""

new_init = """        # Initialize LLM with fallback support
        self.llm = None
        if config.enable_llm_fallback and config.fallback_llm_provider and config.fallback_llm_api_key:
            # Use fallback provider (OpenRouter + Gemini)
            logger.info("🛡️ Initializing LLM with fallback support...")
            try:
                self.llm = create_fallback_provider(
                    primary_provider=config.llm_provider,
                    primary_api_key=config.llm_api_key,
                    primary_model=config.llm_model,
                    fallback_provider=config.fallback_llm_provider,
                    fallback_api_key=config.fallback_llm_api_key,
                    fallback_model=config.fallback_llm_model,
                    site_url=getattr(config, 'openrouter_site_url', 'https://tara.daytona.io'),
                    app_name=getattr(config, 'openrouter_app_name', 'TARA-Daytona')
                )
                logger.info(
                    f"✅ Fallback LLM: {config.llm_provider} ({config.llm_model}) → "
                    f"{config.fallback_llm_provider} ({config.fallback_llm_model})"
                )
            except Exception as e:
                logger.error(f"❌ Fallback provider init failed: {e}")
        elif config.llm_api_key:
            # Use single provider
            logger.info(f"🤖 Single LLM provider: {config.llm_provider}")
            try:
                kwargs = {}
                if config.llm_provider == "openrouter":
                    kwargs['site_url'] = getattr(config, 'openrouter_site_url', 'https://tara.daytona.io')
                    kwargs['app_name'] = getattr(config, 'openrouter_app_name', 'TARA-Daytona')
                
                self.llm = create_provider(
                    provider_name=config.llm_provider,
                    api_key=config.llm_api_key,
                    model_name=config.llm_model,
                    **kwargs
                )
                logger.info(f"✅ LLM initialized: {config.llm_provider} ({config.llm_model})")
            except Exception as e:
                logger.error(f"❌ LLM provider init failed: {e}")
        else:
            logger.warning("⚠️ No LLM API key - response generation unavailable")
        
        # Backwards compat: keep gemini_model reference
        self.gemini_model = self.llm if (self.llm and config.llm_provider == "gemini") else None"""

content = content.replace(old_init, new_init)

# Pattern 3: Replace all generate_content() calls with .generate()
# This is the most important part!

# Replace streaming calls
content = re.sub(
    r'(\s+)response_stream = self\.gemini_model\.generate_content\(\s*([^,]+),\s*generation_config=genai\.types\.GenerationConfig\(\s*temperature=([^,]+),\s*max_output_tokens=([^,]+),?\s*\),\s*stream=True\s*\)',
    r'\1# Use provider.generate() for streaming\n\1response_stream = self.llm.generate(\n\1    prompt=\2,\n\1    temperature=\3,\n\1    max_tokens=\4,\n\1    stream=True\n\1)',
    content
)

# Replace non-streaming calls
content = re.sub(
    r'(\s+)response = self\.gemini_model\.generate_content\(\s*([^,]+),?\s*generation_config=genai\.types\.GenerationConfig\(\s*temperature=([^,]+),\s*max_output_tokens=([^,]+),?\s*\)\s*\)',
    r'\1# Use provider.generate() for non-streaming\n\1response = self.llm.generate(\n\1    prompt=\2,\n\1    temperature=\3,\n\1    max_tokens=\4,\n\1    stream=False\n\1)',
    content
)

# Replace simple generate_content calls
content =re.sub(
    r'self\.gemini_model\.generate_content\(([^)]+)\)',
    r'self.llm.generate(prompt=\1, stream=False)',
    content
)

# Fix response access (.text → direct value)
content = re.sub(
    r'([^a-zA-Z_])response\.text([^a-zA-Z_])',
    r'\1response\2',  # provider.generate() returns string directly
    content
)

# For streaming chunks
content = re.sub(
    r'if chunk\.text:',
    r'if chunk:  # Provider yields text directly',
    content
)
content = re.sub(
    r'chunk\.text',
    r'chunk',
    content
)

# Write back
with open('rag_engine.py', 'w') as f:
    f.write(content)

print("✅ rag_engine.py updated successfully!")
print("   - Added fallback provider imports")
print("   - Replaced Gemini initialization")
print("   - Converted generate_content() → generate()")
print("   - Fixed response access patterns")
