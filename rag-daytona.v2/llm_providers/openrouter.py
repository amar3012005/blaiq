"""
OpenRouter LLM Provider with Reasoning Support

Implementation of LLMProvider for OpenRouter.ai
Supports Nvidia Nemotron's reasoning API with reasoning_details preservation.
"""

import logging
from typing import Generator, Dict, Any, Optional
import requests
import json
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation with reasoning support."""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = kwargs.get("site_url", "https://tara.daytona.io")
        self.app_name = kwargs.get("app_name", "TARA-Daytona")
        self.enable_reasoning = kwargs.get("enable_reasoning", True)  # Enable reasoning by default
        self.conversation_history = []  # Store conversation with reasoning_details
    
    def initialize(self) -> bool:
        """Initialize OpenRouter client."""
        try:
            # Test connection with a simple request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name
            }
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                self._client = True  # Flag that we're initialized
                logger.info(f"✅ OpenRouter initialized: {self.model_name} (reasoning={'enabled' if self.enable_reasoning else 'disabled'})")
                return True
            else:
                logger.error(f"OpenRouter initialization failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ OpenRouter initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        messages: Optional[list] = None,  # NEW: Allow passing full message history
        **kwargs
    ) -> str | Generator[str, None, None]:
        """
        Generate response using OpenRouter with reasoning support.
        
        Args:
            prompt: User prompt (used if messages is None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming
            messages: Full message history with reasoning_details (optional)
            **kwargs: Additional parameters (e.g., enable_reasoning override)
            
        Returns:
            Generated text or generator for streaming
        """
        if not self.is_available():
            raise RuntimeError("OpenRouter provider not initialized")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
                "Content-Type": "application/json"
            }
            
            # Build messages
            if messages is None:
                # Simple prompt without history
                messages_list = [{"role": "user", "content": prompt}]
            else:
                # Use provided messages (with reasoning_details)
                messages_list = messages
            
            # Build payload
            payload = {
                "model": self.model_name,
                "messages": messages_list,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            # Add reasoning if enabled
            enable_reasoning = kwargs.get("enable_reasoning", self.enable_reasoning)
            if enable_reasoning and "nemotron" in self.model_name.lower():
                payload["reasoning"] = {"enabled": True}
                logger.debug(f"🧠 Reasoning enabled for {self.model_name}")
            
            if stream:
                # Streaming mode
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                
                accumulated_content = ""
                reasoning_details = None
                
                def stream_generator():
                    nonlocal accumulated_content, reasoning_details
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: '
                                if data_str != '[DONE]':
                                    try:
                                        data = json.loads(data_str)
                                        if 'choices' in data and len(data['choices']) > 0:
                                            choice = data['choices'][0]
                                            delta = choice.get('delta', {})
                                            
                                            # Extract content
                                            content = delta.get('content')
                                            if content:
                                                accumulated_content += content
                                                yield content
                                            
                                            # Extract reasoning_details if present
                                            if 'reasoning_details' in delta:
                                                reasoning_details = delta['reasoning_details']
                                                logger.debug(f"🧠 Reasoning details: {reasoning_details}")
                                    
                                    except json.JSONDecodeError:
                                        continue
                    
                    # Store in conversation history after streaming completes
                    if accumulated_content:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": accumulated_content,
                            "reasoning_details": reasoning_details
                        })
                
                return stream_generator()
            else:
                # Non-streaming mode
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract message with reasoning_details
                message = data['choices'][0]['message']
                content = message.get('content', '')
                reasoning_details = message.get('reasoning_details')
                
                if reasoning_details:
                    logger.debug(f"🧠 Reasoning details: {reasoning_details}")
                
                # Store in conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning_details": reasoning_details
                })
                
                return content
        
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise
    
    def add_user_message(self, content: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def get_conversation_history(self) -> list:
        """Get full conversation history with reasoning_details."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def is_available(self) -> bool:
        """Check if OpenRouter is ready."""
        return self._client is not None
