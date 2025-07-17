import os
import logging
import json
import requests
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# Local inference configuration
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Provider priority: try local first, no API fallback
PROVIDER_PRIORITY = os.getenv(
    "PROVIDER_PRIORITY", "lmstudio,ollama").lower().split(",")

# Available local models for LMStudio (mapped to actual model IDs)
LMSTUDIO_MODELS = {
    "phi-4-reasoning-plus": "microsoft/phi-4-reasoning-plus",
    "phi-4-mini-reasoning": "microsoft/phi-4-mini-reasoning",
    "phi-4-mini-instruct": "phi-4-mini-instruct",  # This one doesn't have microsoft/ prefix
    "instruct": "phi-4-mini-instruct",  # Alias for instruct model
    "reasoning": "microsoft/phi-4-mini-reasoning",  # Alias for reasoning model
    "reasoning-plus": "microsoft/phi-4-reasoning-plus",  # Alias for reasoning-plus model
    "default": "microsoft/phi-4-reasoning-plus"  # Fallback to loaded model
}

# Available local models for Ollama - updated to match your actual models
OLLAMA_MODELS = {
    "phi-4-reasoning-plus": "phi4-mini-reasoning:latest",
    "phi-4-mini-reasoning": "phi4-mini-reasoning:latest",
    "phi-4-mini-instruct": "phi4-mini:latest",
    "instruct": "phi4-mini:latest",  # Alias for instruct model
    "reasoning": "phi4-mini-reasoning:latest",  # Alias for reasoning model
    "reasoning-plus": "phi4-mini-reasoning:latest",  # Alias for reasoning-plus model
    "default": "phi4-mini:latest"  # Fallback to instruct model
}


def _select_model_for_task(prompt: str, requested_model: str = None) -> str:
    """Enhanced model selection based on task complexity"""
    
    if requested_model:
        return requested_model
    
    # Code generation keywords → use reasoning for better planning
    code_generation_keywords = [
        "write", "create", "implement", "generate", "build", "make",
        "function", "class", "algorithm", "code", "program"
    ]
    
    # Decision-making keywords → use reasoning-plus for better choices
    decision_keywords = [
        "decide", "choose", "select", "determine", "analyze", "plan"
    ]
    
    # Complex reasoning keywords → use reasoning-plus
    complex_reasoning_keywords = [
        "complex", "architecture", "design", "strategy", "approach",
        "step by step", "analyze", "evaluate", "compare"
    ]
    
    prompt_lower = prompt.lower()
    
    # Check for complex reasoning needs
    if any(keyword in prompt_lower for keyword in complex_reasoning_keywords + decision_keywords):
        return "reasoning-plus"
    
    # Check for code generation or basic reasoning needs
    if any(keyword in prompt_lower for keyword in code_generation_keywords):
        return "reasoning"
    
    # Default to instruct for simple tasks
    return "instruct"
def call_lmstudio(prompt: str, model_name: str = "phi-4-reasoning-plus") -> str:
    """Call LMStudio local server with dynamic model detection"""
    try:
        # Find the best available model
        model_id = find_best_lmstudio_model(model_name)

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20000,
            "temperature": 0.7,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{LMSTUDIO_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=300  # 5 minutes timeout
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"LMStudio success using model {model_id}")
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(
                f"LMStudio API error: {response.status_code} - {response.text}")
            raise ConnectionError(
                f"LMStudio API error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"LMStudio connection error: {e}")
        raise ConnectionError(f"LMStudio connection failed: {e}")


def call_ollama(prompt: str, model_name: str = "phi-4-reasoning-plus") -> str:
    """Call Ollama local server"""
    try:
        # Use the model identifier from OLLAMA_MODELS
        model_id = OLLAMA_MODELS.get(model_name, model_name)

        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 20000
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            headers=headers,
            timeout=300  # 5 minutes timeout
        )

        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            logger.error(
                f"Ollama API error: {response.status_code} - {response.text}")
            raise ConnectionError(f"Ollama API error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama connection error: {e}")
        raise ConnectionError(f"Ollama connection failed: {e}")


def check_server_availability(url: str) -> bool:
    """Check if a server is available"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_available_lmstudio_models() -> list:
    """Get list of available models from LMStudio"""
    try:
        response = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        return []
    except requests.exceptions.RequestException:
        return []


def find_best_lmstudio_model(requested_model: str) -> str:
    """Find the best available model in LMStudio"""
    available_models = get_available_lmstudio_models()

    if not available_models:
        return LMSTUDIO_MODELS.get("default", "microsoft/phi-4-reasoning-plus")

    # First try exact match
    preferred_model = LMSTUDIO_MODELS.get(requested_model)
    if preferred_model and preferred_model in available_models:
        return preferred_model

    # Try to find a close match
    for model_name, model_id in LMSTUDIO_MODELS.items():
        if model_id in available_models:
            logger.info(
                f"Using available model {model_id} instead of requested {requested_model}")
            return model_id

    # Fall back to first available model
    fallback_model = available_models[0]
    logger.info(
        f"Using fallback model {fallback_model} instead of requested {requested_model}")
    return fallback_model


def _select_model_for_task(prompt: str, model_name: str = None) -> str:
    """Intelligently select model based on task type from prompt analysis"""
    if model_name:
        return model_name  # Use explicitly provided model

    # Convert to lowercase for analysis
    prompt_lower = prompt.lower()

    # Keywords that indicate reasoning tasks
    reasoning_keywords = [
        "analyze", "reasoning", "complex", "think", "step by step", "logic", "solve",
        "explain why", "because", "therefore", "conclusion", "deduce", "infer",
        "problem solving", "critical thinking", "reasoning chain", "thought process"
    ]

    # Keywords that indicate simple instruct tasks
    instruct_keywords = [
        "write", "create", "generate", "make", "build", "format", "convert",
        "list", "summarize", "translate", "rewrite", "edit", "modify",
        "simple", "basic", "quick", "direct", "straightforward"
    ]

    # Count keyword matches
    reasoning_score = sum(
        1 for keyword in reasoning_keywords if keyword in prompt_lower)
    instruct_score = sum(
        1 for keyword in instruct_keywords if keyword in prompt_lower)

    # Decision logic
    if reasoning_score > instruct_score:
        logger.info("Selected reasoning model based on prompt analysis")
        return "phi-4-reasoning-plus"  # Use the most capable reasoning model
    elif instruct_score > reasoning_score:
        logger.info("Selected instruct model based on prompt analysis")
        return "phi-4-mini-instruct"
    else:
        # For ambiguous cases, check prompt length and complexity
        if len(prompt) > 500 or "complex" in prompt_lower:
            logger.info("Selected reasoning model for complex/long prompt")
            return "phi-4-reasoning-plus"
        else:
            logger.info("Selected instruct model for simple/short prompt")
            return "phi-4-mini-instruct"

# Learn more about calling the LLM: https://the-pocket.github.io/PocketFlow/utility_function/llm.html


def call_llm(prompt: str, use_cache: bool = True, model_name: str = None, provider: str = None, model: str = None) -> str:
    # Support both model_name and model parameters for flexibility
    if model is not None:
        model_name = model

    # Intelligently select model based on task type
    selected_model = _select_model_for_task(prompt, model_name)

    # Log the prompt and selected model
    logger.info(f"PROMPT: {prompt[:100]}...")
    logger.info(f"SELECTED MODEL: {selected_model}")

    # Check cache if enabled
    if use_cache:
        cached_response = _get_from_cache(prompt)
        if cached_response:
            return cached_response

    # Try providers in order of priority, or specific provider if requested
    if provider:
        response_text = _try_single_provider(prompt, selected_model, provider)
    else:
        response_text = _try_providers(prompt, selected_model)

    # Log the response with full details
    logger.info(f"RESPONSE: {response_text}")
    print(f"DEBUG: Full LLM Response:\n{response_text}")
    print(f"DEBUG: Response length: {len(response_text)}")

    # Update cache if enabled
    if use_cache:
        _save_to_cache(prompt, response_text)

    return response_text


def _get_from_cache(prompt: str) -> str | None:
    """Get response from cache if available"""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to load cache: {e}, starting with empty cache")

    if prompt in cache:
        logger.info(f"Cache hit for prompt: {prompt[:50]}...")
        return cache[prompt]

    return None


def _save_to_cache(prompt: str, response: str) -> None:
    """Save response to cache"""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to reload cache: {e}")

    cache[prompt] = response
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        logger.info("Added to cache")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def _try_single_provider(prompt: str, model_name: str, provider: str) -> str:
    """Try a specific provider"""
    try:
        if provider == "lmstudio":
            if check_server_availability(LMSTUDIO_URL):
                logger.info(
                    f"Attempting to call LMStudio with model: {model_name}")
                return call_lmstudio(prompt, model_name)
            else:
                raise RuntimeError("LMStudio server not available")

        elif provider == "ollama":
            if check_server_availability(OLLAMA_URL):
                logger.info(
                    f"Attempting to call Ollama with model: {model_name}")
                return call_ollama(prompt, model_name)
            else:
                raise RuntimeError("Ollama server not available")

        else:
            raise RuntimeError(f"Unknown provider: {provider}")

    except Exception as e:
        logger.error(f"Provider {provider} failed: {e}")
        raise


def _try_providers(prompt: str, model_name: str) -> str:
    """Try each provider in order until one succeeds"""
    for provider in PROVIDER_PRIORITY:
        provider = provider.strip()

        try:
            if provider == "lmstudio":
                if check_server_availability(LMSTUDIO_URL):
                    logger.info(
                        f"Attempting to call LMStudio with model: {model_name}")
                    return call_lmstudio(prompt, model_name)
                else:
                    logger.warning("LMStudio server not available")

            elif provider == "ollama":
                if check_server_availability(OLLAMA_URL):
                    logger.info(
                        f"Attempting to call Ollama with model: {model_name}")
                    return call_ollama(prompt, model_name)
                else:
                    logger.warning("Ollama server not available")

            elif provider == "anthropic":
                logger.info("Calling Anthropic API")
                return call_anthropic(prompt)

        except Exception as e:
            logger.warning(f"{provider} failed: {e}, trying next provider")
            continue

    raise RuntimeError("All providers failed")


def clear_cache() -> None:
    """Clear the cache file if it exists."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info("Cache cleared")


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making first call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")

    # Second call - should hit cache
    print("\nMaking second call with same prompt...")
    response2 = call_llm(test_prompt, use_cache=True)
    print(f"Response: {response2}")
