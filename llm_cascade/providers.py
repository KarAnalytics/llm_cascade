"""
LLM Cascade: Automatic fallback across 8 free-tier LLM providers.

Usage:
    from llm_cascade import get_cascade

    llm = get_cascade()          # auto-detects available API keys
    response = llm.generate("What is Python?")
    print(response.text)
    print(response.provider)     # which provider answered
    print(response.model)        # which model was used
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    provider: str
    model: str


# Provider definitions in fallback order
PROVIDERS = [
    {
        'name': 'Gemini',
        'key_env': 'GEMINI_API_KEY',
        'style': 'gemini',
        'default_model': 'gemini-2.5-flash',
    },
    {
        'name': 'Ollama',
        'key_env': 'OLLAMA_API_KEY',
        'style': 'openai',
        'base_url': 'https://ollama.com/v1',
        'default_model': 'kimi-k2.5:cloud',
    },
    {
        'name': 'Groq',
        'key_env': 'GROQ_API_KEY',
        'style': 'openai',
        'base_url': 'https://api.groq.com/openai/v1',
        'default_model': 'llama-3.3-70b-versatile',
    },
    {
        'name': 'HuggingFace',
        'key_env': 'HF_TOKEN',
        'style': 'openai',
        'base_url': 'https://router.huggingface.co/v1',
        'default_model': 'meta-llama/Llama-3.3-70B-Instruct',
    },
    {
        'name': 'Cohere',
        'key_env': 'COHERE_API_KEY',
        'style': 'openai',
        'base_url': 'https://api.cohere.ai/compatibility/v1',
        'default_model': 'command-a-03-2025',
    },
    {
        'name': 'OpenRouter',
        'key_env': 'OPENROUTER_API_KEY',
        'style': 'openai',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'meta-llama/llama-3.3-70b-instruct:free',
    },
        {
        'name': 'OpenAI',
        'key_env': 'OPENAI_API_KEY',
        'style': 'openai',
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'gpt-4o-mini',
    },
    {
        'name': 'Grok (xAI)',
        'key_env': 'XAI_API_KEY',
        'style': 'openai',
        'base_url': 'https://api.x.ai/v1',
        'default_model': 'grok-3-mini',
    },
]


def _load_env():
    """Load .env file if it exists (for local development)."""
    for env_path in [Path.cwd() / '.env', Path.cwd().parent / '.env']:
        if env_path.exists():
            for raw_line in env_path.read_text(encoding='utf-8').splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            break


def _load_colab_secret(name):
    """Try to load a secret from Google Colab."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def _get_key(env_name):
    """Get an API key from env vars or Colab Secrets."""
    # Some users store Gemini key as GOOGLE_API_KEY instead of GEMINI_API_KEY
    aliases = {'GEMINI_API_KEY': ['GOOGLE_API_KEY']}
    names_to_try = [env_name] + aliases.get(env_name, [])
    for name in names_to_try:
        val = os.getenv(name) or _load_colab_secret(name)
        if val:
            return val
    return None


def _is_retriable_error(exc):
    """Check if an exception is worth retrying with the next provider."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        'quota', 'resource_exhausted', '429', 'rate limit',
        'exceeded your current quota', 'too many requests',
        'rate_limit_exceeded', 'tokens per minute',
        '503', 'unavailable', 'overloaded', 'high demand',
        '502', 'bad gateway', 'service unavailable',
        '500', 'internal server error',
        '401', 'invalid api key', 'expired_api_key', 'unauthorized',
        '403', 'forbidden', 'permission denied',
        'invalid_api_key', 'authentication',
    ))


class LLMCascade:
    """LLM with automatic fallback across multiple providers."""

    def __init__(self, providers=None, verbose=True):
        """
        Initialize the cascade.

        Args:
            providers: List of provider dicts (uses defaults if None)
            verbose: Print provider status and fallback messages
        """
        _load_env()
        self.verbose = verbose
        self.providers = providers or PROVIDERS
        self.available = [p for p in self.providers if _get_key(p['key_env'])]
        self._embedder = None
        self._embedder_name = None

        if verbose:
            if self.available:
                print('LLM Cascade - available providers:')
                for p in self.available:
                    print(f"  + {p['name']:<16} model={p['default_model']}")
                not_configured = [p for p in self.providers if p not in self.available]
                if not_configured:
                    print('Not configured (skipped):')
                    for p in not_configured:
                        print(f"  - {p['name']:<16} (set {p['key_env']})")
            else:
                print('WARNING: No API keys found. Set at least one of:')
                for p in self.providers:
                    print(f"  {p['key_env']}")

    def generate(self, prompt, system_prompt=None, model=None, max_tokens=1024):
        """
        Generate text using the first available provider, with automatic fallback.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Override the default model for the provider
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with .text, .provider, .model attributes
        """
        if not self.available:
            # Refresh in case keys were added after instantiation (common in Colab)
            _load_env()
            self.available = [p for p in self.providers if _get_key(p['key_env'])]
            if not self.available:
                raise ValueError(
                    'No LLM provider configured. Set at least one API key in Colab Secrets '
                    '(and toggle "Notebook access" ON), then re-run this cell.'
                )
            if self.verbose:
                print('Providers refreshed:')
                for p in self.available:
                    print(f"  + {p['name']:<16} ({p['default_model']})")

        last_exc = None
        for provider in self.available:
            api_key = _get_key(provider['key_env'])
            used_model = model or provider['default_model']
            try:
                if provider['style'] == 'gemini':
                    text = self._call_gemini(api_key, used_model, prompt, system_prompt)
                else:
                    text = self._call_openai(api_key, provider['base_url'], used_model,
                                            prompt, system_prompt, max_tokens)
                if self.verbose:
                    print(f"  [Response from {provider['name']} / {used_model}]")
                return LLMResponse(text=text, provider=provider['name'], model=used_model)
            except Exception as exc:
                last_exc = exc
                if _is_retriable_error(exc):
                    if self.verbose:
                        print(f"  [{provider['name']}] unavailable, trying next...")
                    continue
                else:
                    raise

        raise RuntimeError(f'All providers exhausted. Last error: {last_exc}')

    def get_embedding(self, text, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Get an embedding vector using a local HuggingFace model (no API key needed).

        Args:
            text: Text to embed
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)

        Returns:
            List of floats (the embedding vector)
        """
        if self._embedder is None or self._embedder_name != model_name:
            from sentence_transformers import SentenceTransformer
            if self.verbose:
                print(f'  Loading embedding model: {model_name}...')
            self._embedder = SentenceTransformer(model_name)
            self._embedder_name = model_name
            if self.verbose:
                print(f'  Embedding model ready (local, no API key needed)')

        embedding = self._embedder.encode(text)
        return embedding.tolist()

    def _call_gemini(self, api_key, model, prompt, system_prompt=None):
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)

        # Disable thinking mode so all token budget goes to the actual answer.
        # Without this, gemini-2.5-flash can burn the budget on internal
        # reasoning and return an empty or truncated .text field.
        config_kwargs = {
            'thinking_config': genai_types.ThinkingConfig(thinking_budget=0),
        }
        if system_prompt:
            config_kwargs['system_instruction'] = system_prompt

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        # resp.text can still be None if the model produced no text parts;
        # fall back to empty string so downstream code doesn't crash.
        return resp.text or ''

    def _call_openai(self, api_key, base_url, model, prompt, system_prompt=None, max_tokens=1024):
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content

    def has_provider(self):
        """Check if at least one provider is available."""
        return len(self.available) > 0

    def list_providers(self):
        """Return list of available provider names."""
        return [p['name'] for p in self.available]

    def list_models(self):
        """Return a dict of {provider_name: current_default_model} for all providers."""
        return {p['name']: p['default_model'] for p in self.providers}

    def set_model(self, provider_name, model):
        """
        Override the default model for a specific provider.

        Example:
            llm = get_cascade()
            llm.set_model('Gemini', 'gemini-2.5-pro')  # use Pro instead of Flash
            llm.set_model('OpenAI', 'gpt-4o')          # use full gpt-4o
            response = llm.generate('Summarize this')   # uses the new defaults
        """
        matched = False
        for p in self.providers:
            if p['name'].lower() == provider_name.lower():
                p['default_model'] = model
                matched = True
                if self.verbose:
                    print(f"  Set {p['name']} model -> {model}")
        if not matched:
            available_names = [p['name'] for p in self.providers]
            raise ValueError(
                f'Unknown provider: {provider_name}. Available: {available_names}'
            )


def get_cascade(verbose=True, models=None):
    """
    Create and return an LLMCascade instance.

    Args:
        verbose: Print provider status and per-call progress messages.
        models: Optional dict of {provider_name: model} to override the defaults.
                Example: get_cascade(models={'Gemini': 'gemini-2.5-pro', 'OpenAI': 'gpt-4o'})

    The simplest way to get started:
        from llm_cascade import get_cascade
        llm = get_cascade()
        response = llm.generate("Hello!")

    To use different models per application:
        llm = get_cascade(models={'Gemini': 'gemini-2.5-pro'})
        # or after creation:
        llm.set_model('Gemini', 'gemini-2.5-pro')
    """
    cascade = LLMCascade(verbose=verbose)
    if models:
        for name, model in models.items():
            cascade.set_model(name, model)
    return cascade
