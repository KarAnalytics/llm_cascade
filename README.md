# LLM Cascade

Automatic fallback across 8 free-tier LLM providers. When one provider hits its quota, the next one is tried automatically.

## Install

```bash
pip install git+https://github.com/KarAnalytics/llm_cascade.git
```

## Quick Start

```python
from llm_cascade import get_cascade

llm = get_cascade()
response = llm.generate("What is machine learning?")
print(response.text)       # the answer
print(response.provider)   # e.g., "Gemini"
print(response.model)      # e.g., "gemini-2.5-flash"
```

## Supported Providers

Set any of these API keys (env vars or Colab Secrets):

| Provider | Env Variable | Free Tier |
|---|---|---|
| Gemini | `GEMINI_API_KEY` | 500 req/day |
| Ollama Cloud | `OLLAMA_API_KEY` | Free tier |
| Groq | `GROQ_API_KEY` | 30 req/min |
| HuggingFace | `HF_TOKEN` | Free inference |
| Cohere | `COHERE_API_KEY` | 20 req/min |
| OpenRouter | `OPENROUTER_API_KEY` | Free models |
| OpenAI | `OPENAI_API_KEY` | Limited free credits |
| Grok (xAI) | `XAI_API_KEY` | $25/month free |

## Features

- **Auto-fallback:** If a provider hits quota, automatically tries the next
- **Embeddings:** `llm.get_embedding("text")` with provider fallback
- **Simple API:** Just `llm.generate(prompt)` — returns text, provider name, and model
- **Works everywhere:** Colab, local, VS Code — loads keys from env vars, `.env` files, or Colab Secrets
