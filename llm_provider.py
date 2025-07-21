import requests
import os
from dotenv import load_dotenv
load_dotenv()


def get_llm_config(model=None, provider=None):
    """
    Returns a dictionary with LLM provider, model, API URL, and API key.
    Priority: function argument > environment variable > default
    """
    # Provider-specific defaults
    default_provider = os.getenv("LLM_PROVIDER", "openrouter")
    default_model = os.getenv("LLM_MODEL", "mistralai/mixtral-8x7b-instruct")
    default_api_url = os.getenv(
        "LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    # API key: try all possible env vars
    api_key = (
        os.getenv("LLM_API_KEY") or
        os.getenv("OPENROUTER_API_KEY") or
        os.getenv("HF_API_KEY")
    )
    return {
        "provider": provider or default_provider,
        "model": model or default_model,
        "api_url": os.getenv("LLM_API_URL") or default_api_url,
        "api_key": api_key
    }


def call_llm(prompt, model=None, provider=None, temperature=0.1, max_tokens=512):
    cfg = get_llm_config(model, provider)
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": cfg["model"],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(cfg["api_url"], headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unexpected LLM response: {result}")
