"""Thin OpenAI-compatible client (works with Gemini, OpenAI, Groq, OpenRouter, Ollama...).

Resilience (free-tier quotas run out mid-demo):
  * LLM_API_KEYS  = comma-separated keys, LLM_MODEL_FALLBACKS = comma-separated models.
    On 429 / 503 / 404 / timeout we move to the next (model, key) pair after a short, growing wait,
    bounded by a total deadline so the UI falls back to the cited offline quote within seconds.
  * Answers are cached in SQLite (key = caller-supplied, e.g. standalone question + chunk ids + model),
    so repeated questions and eval re-runs do not spend quota.
  * last_status() reports the model actually used and the outcome of the last call (never a key).
Returns None when no key is configured / every attempt fails, so the app runs in offline mode."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time

from . import config

log = logging.getLogger(__name__)
RETRYABLE = {404, 408, 429, 500, 502, 503, 504}
_clients: dict[int, object] = {}
_acc = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}  # reset per question by the eval


def usage_reset() -> None:
    _acc.update(prompt_tokens=0, completion_tokens=0, calls=0)


def usage_total() -> dict:
    return dict(_acc)


def _account(usage: dict) -> None:
    _acc["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    _acc["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    _acc["calls"] += 1


_last: dict = {"model": None, "ok": None, "status": None, "cached": False, "ts": None, "fallback_used": False}


def available() -> bool:
    return bool(config.LLM_API_KEYS)


def last_status() -> dict:
    return dict(_last)


def _make_client(key: str):  # separate function so tests can swap in a fake client
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=config.LLM_BASE_URL or None, timeout=config.LLM_TIMEOUT, max_retries=0)


def _client(i: int):
    if i not in _clients:
        _clients[i] = _make_client(config.LLM_API_KEYS[i])
    return _clients[i]


def _status_of(e: Exception) -> int | None:
    code = getattr(e, "status_code", None)
    if code is None and "timeout" in type(e).__name__.lower():
        return 408
    return code


def candidates(model: str | None) -> list[tuple[str, int]]:
    """(model, key index) pairs in the order they are tried: each model on every key, then the next model."""
    models = [model] if model else [config.LLM_MODEL, *config.LLM_MODEL_FALLBACKS]
    models = list(dict.fromkeys(m for m in models if m))
    return [(m, k) for m in models for k in range(len(config.LLM_API_KEYS))]


def _cache_get(key: str | None):
    if not key or not config.LLM_CACHE:
        return None
    from . import store

    return store.cache_get(key)


def _cache_put(key: str | None, text: str, model: str, usage: dict) -> None:
    if key and config.LLM_CACHE and text:
        from . import store

        store.cache_put(key, text, model, usage)


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 800, model: str | None = None,
         cache_key: str | None = None) -> tuple[str | None, dict]:
    """Return (text, usage). usage carries "model" (actually used) and "cached". text is None if all attempts fail."""
    if not available():
        return None, {}
    hit = _cache_get(cache_key)
    if hit:
        _last.update(model=hit["model"], ok=True, status="cache", cached=True, ts=time.time(), fallback_used=False)
        _account(hit["usage"])
        return hit["text"], {**hit["usage"], "model": hit["model"], "cached": True}

    deadline = time.time() + config.LLM_DEADLINE
    tried = candidates(model)
    err = None
    for n, (m, k) in enumerate(tried):
        if n and time.time() + config.LLM_RETRY_WAIT * n > deadline:
            break
        if n:
            time.sleep(min(config.LLM_RETRY_WAIT * n, 8))  # short, growing pause between attempts
        kwargs = dict(model=m, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if config.LLM_REASONING_EFFORT and m == config.LLM_MODEL:  # effort is tuned for the primary model only
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
        try:
            resp = _client(k).chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            if not text:
                raise ValueError("empty completion")
            _last.update(model=m, ok=True, status=200, cached=False, ts=time.time(), fallback_used=n > 0)
            if n:
                log.warning("LLM answered by fallback model=%s (attempt %d)", m, n + 1)
            _account(usage)
            _cache_put(cache_key, text, m, usage)
            return text, {**usage, "model": m, "cached": False}
        except Exception as e:
            err = e
            code = _status_of(e)
            log.warning("LLM attempt %d failed: model=%s key#%d status=%s", n + 1, m, k + 1, code or type(e).__name__)
            if code is not None and code not in RETRYABLE:
                break  # e.g. 400/401: trying more of the same will not help
    _last.update(model=None, ok=False, status=_status_of(err) if err else None, cached=False, ts=time.time(), fallback_used=False)
    log.error("LLM unavailable after %d attempt(s); falling back to offline quote", len(tried))
    return None, {"error": f"{type(err).__name__}: {_status_of(err)}" if err else "no attempt"}


def hash_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def chat_json(messages: list[dict], max_tokens: int = 400, model: str | None = None) -> dict | None:
    key = hash_key("json", model or config.LLM_MODEL, messages)
    text, _ = chat(messages, temperature=0, max_tokens=max_tokens, model=model, cache_key=key)
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
