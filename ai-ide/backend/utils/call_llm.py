"""
LLM calling utilities
"""
import requests
import json
import logging

logger = logging.getLogger(__name__)

LMSTUDIO_URL = "http://localhost:1234"

def check_server_availability(url: str) -> bool:
    """Check if LM Studio server is available"""
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_llm(prompt: str, model: str = None, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Call LM Studio API"""
    try:
        if not check_server_availability(LMSTUDIO_URL):
            raise Exception("LM Studio server not available")
        
        payload = {
            "model": model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(f"{LMSTUDIO_URL}/v1/chat/completions", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error: {e}"