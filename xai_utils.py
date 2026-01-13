"""
Utilities for working with XAI/Grok API.

This module provides helpers to:
- Create chat completions with message-style input
- Extract the completed assistant message
- Parse structured outputs into Pydantic models
- Enable web search for up-to-date information

Based on trader-agent project XAI integration patterns.
"""

from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, cast
from datetime import datetime, timedelta
import json
import logging
import requests
import time

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Model constants for different task complexities
MODEL_FAST = "grok-3-mini-fast"      # Quick, simple tasks
MODEL_STANDARD = "grok-3-mini"        # Standard analysis
MODEL_DEEP = "grok-3"                 # Complex reasoning
MODEL_PREMIUM = "grok-4-latest"       # Most capable, with search

XAI_BASE_URL = "https://api.x.ai/v1/chat/completions"


class XAIClient:
    """
    XAI/Grok API Client for Kalshi trading bot.
    
    Provides synchronous and asynchronous interfaces for XAI API calls with:
    - Model selection based on task complexity
    - Retry logic with exponential backoff
    - Response parsing and validation
    - Citation handling for search-enabled requests
    """
    
    def __init__(self, api_key: str, default_model: str = MODEL_PREMIUM):
        """
        Initialize XAI client with API key.
        
        Args:
            api_key: XAI API key
            default_model: Default model for requests
        """
        if not api_key:
            raise ValueError("XAI_API_KEY is required")
        
        self.api_key = api_key
        self.default_model = default_model
        self._total_tokens = 0
        self._request_count = 0
    
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        enable_search: bool = False,
        search_from_date: Optional[str] = None,
        search_to_date: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make a synchronous request to XAI API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            enable_search: Whether to enable web search
            search_from_date: Start date for search (YYYY-MM-DD)
            search_to_date: End date for search (YYYY-MM-DD)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            temperature: Temperature for response generation
            
        Returns:
            Raw response dict from XAI API
        """
        if model is None:
            model = self.default_model
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add search parameters for search-enabled requests
        if enable_search:
            today = datetime.now()
            payload["search_parameters"] = {
                "mode": "auto",
                "return_citations": True,
                "from_date": search_from_date or (today - timedelta(days=7)).strftime('%Y-%m-%d'),
                "to_date": search_to_date or today.strftime('%Y-%m-%d'),
                "max_search_results": 20,
                "sources": [
                    {"type": "web"},
                    {"type": "x"},
                    {"type": "news"}
                ]
            }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    XAI_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    self._request_count += 1
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                elif response.status_code >= 500:
                    # Server error - retry
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {timeout}s"
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                continue
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                continue
        
        raise RuntimeError(f"XAI API request failed after {max_retries} attempts: {last_error}")


def extract_message_text(response: Dict[str, Any]) -> str:
    """
    Extract plain text from XAI API response.
    
    Args:
        response: Raw response dict from XAI API
        
    Returns:
        Extracted text content
    """
    try:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "").strip()
    except Exception as e:
        logger.error(f"Error extracting message text: {e}")
    
    return ""


def chat_completion_text(
    client: XAIClient,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    enable_search: bool = False,
    search_from_date: Optional[str] = None,
    search_to_date: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Create a chat completion and extract the text response.
    
    Args:
        client: XAI client instance
        model: Model to use
        messages: Chat messages
        enable_search: Whether to enable web search
        search_from_date: Start date for search
        search_to_date: End date for search
        temperature: Temperature for generation
        
    Returns:
        Text content from the response
    """
    response = client._make_request(
        messages=list(messages),
        model=model,
        enable_search=enable_search,
        search_from_date=search_from_date,
        search_to_date=search_to_date,
        temperature=temperature,
    )
    
    return extract_message_text(response)


def chat_completion_parse_pydantic(
    client: XAIClient,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    enable_search: bool = False,
    search_from_date: Optional[str] = None,
    search_to_date: Optional[str] = None,
    temperature: Optional[float] = None,
) -> T:
    """
    Parse structured output from XAI API into a Pydantic model.
    
    Args:
        client: XAI client instance
        model: Model to use
        messages: Chat messages
        response_format: Pydantic model type for response
        enable_search: Whether to enable web search
        search_from_date: Start date for search
        search_to_date: End date for search
        temperature: Temperature for generation
        
    Returns:
        Parsed Pydantic model instance
    """
    # Build JSON schema from the Pydantic model
    try:
        schema = response_format.model_json_schema()
    except AttributeError:
        # Pydantic v1 fallback
        schema = response_format.schema()
    
    # Inject schema instruction as system message
    schema_str = json.dumps(schema)
    schema_instruction = {
        "role": "system",
        "content": (
            "You must respond with ONLY a single JSON object that validates against the following JSON Schema. "
            "Do not include any prose, code fences, markdown formatting, or additional text. "
            "If a field is optional, omit it instead of writing null.\n\n"
            f"JSON Schema: {schema_str}"
        ),
    }
    
    # Prepend schema instruction to messages
    messages_with_schema = [schema_instruction] + list(messages)
    
    response = client._make_request(
        messages=messages_with_schema,
        model=model,
        enable_search=enable_search,
        search_from_date=search_from_date,
        search_to_date=search_to_date,
        temperature=temperature,
    )
    
    # Extract text content
    text_value = extract_message_text(response)
    
    if not text_value:
        raise RuntimeError("Structured output parsing failed: no content in XAI API response")
    
    # Clean up JSON if wrapped in code blocks
    cleaned_text = text_value
    if "```json" in cleaned_text:
        start = cleaned_text.find("```json") + 7
        end = cleaned_text.find("```", start)
        if end > start:
            cleaned_text = cleaned_text[start:end].strip()
    elif "```" in cleaned_text:
        start = cleaned_text.find("```") + 3
        end = cleaned_text.find("```", start)
        if end > start:
            cleaned_text = cleaned_text[start:end].strip()
    
    # Parse JSON
    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from XAI response: {e}")
        logger.error(f"Raw text (first 500 chars): {text_value[:500]}")
        raise RuntimeError(f"Structured output parsing failed: invalid JSON: {e}")
    
    # Validate with Pydantic
    try:
        return cast(T, response_format.model_validate(data))
    except AttributeError:
        # Pydantic v1 fallback
        return cast(T, response_format.parse_obj(data))


# Async versions for compatibility
async def async_chat_completion_text(
    client: XAIClient,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    enable_search: bool = False,
    search_from_date: Optional[str] = None,
    search_to_date: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Async wrapper for chat_completion_text.
    
    Note: Currently uses sync implementation internally.
    For true async, consider using httpx or aiohttp.
    """
    return chat_completion_text(
        client,
        model=model,
        messages=messages,
        enable_search=enable_search,
        search_from_date=search_from_date,
        search_to_date=search_to_date,
        temperature=temperature,
    )


async def async_chat_completion_parse_pydantic(
    client: XAIClient,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    enable_search: bool = False,
    search_from_date: Optional[str] = None,
    search_to_date: Optional[str] = None,
    temperature: Optional[float] = None,
) -> T:
    """
    Async wrapper for chat_completion_parse_pydantic.
    
    Note: Currently uses sync implementation internally.
    For true async, consider using httpx or aiohttp.
    """
    return chat_completion_parse_pydantic(
        client,
        model=model,
        messages=messages,
        response_format=response_format,
        enable_search=enable_search,
        search_from_date=search_from_date,
        search_to_date=search_to_date,
        temperature=temperature,
    )
