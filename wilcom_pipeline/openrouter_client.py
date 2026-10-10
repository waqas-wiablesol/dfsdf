"""OpenRouter client.

Implements the same surface as `GeminiClient`, so the stages do not know which
provider they are using:

    client.chat(model, request, cache_key=...)  ->  {"text", "usage", "model", "cached"}
    client.ledger                               ->  Ledger
    client.media(path, kind)                    ->  the provider's media handle

A *request* is the neutral `{system, text, media}` dict built by
`prompts.trajectory_request` / `prompts.grounding_request`. Converting it to this
provider's wire format is `to_messages` below.

Wraps the official `openrouter` SDK with what a data pipeline actually needs and
the raw SDK does not provide: retries with backoff, tolerant JSON extraction,
a cost ledger, and a disk cache so a crashed run does not re-pay for completed
work.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, payloads

# ============================================================================
# SDK import, with a clear failure
# ============================================================================

try:
    from openrouter import OpenRouter  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit(
        "The official OpenRouter SDK is not installed.\n"
        "    pip install openrouter"
    )


# ============================================================================
# Cost ledger
# ============================================================================


@dataclass
class Ledger:
    """Running spend, so a batch's cost is measured rather than guessed.

    OpenRouter reports usage per response and prices per model. Those prices are
    per-token in USD; the model list returns them as strings like "0.000001".
    """

    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, usage: dict, price_in: float, price_out: float) -> None:
        self.calls += 1
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        rt = int((usage.get("completion_tokens_details") or {})
                 .get("reasoning_tokens") or 0)

        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.reasoning_tokens += rt

        # Some OpenRouter responses carry the authoritative cost directly.
        direct = usage.get("cost")
        if direct is not None:
            cost = float(direct)
        else:
            cost = pt * price_in + ct * price_out
        self.cost_usd += cost

        m = self.by_model.setdefault(
            model, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        m["calls"] += 1
        m["in"] += pt
        m["out"] += ct
        m["cost"] += cost

    def add_failure(self) -> None:
        self.failures += 1

    def report(self) -> str:
        lines = [
            "calls           : %d  (%d failed)" % (self.calls, self.failures),
            "prompt tokens   : %s" % f"{self.prompt_tokens:,}",
            "completion tok  : %s" % f"{self.completion_tokens:,}",
            "  of which think: %s" % f"{self.reasoning_tokens:,}",
            "estimated cost  : $%.4f" % self.cost_usd,
        ]
        if len(self.by_model) > 1:
            lines.append("by model:")
            for m, d in sorted(self.by_model.items()):
                lines.append("   %-38s %3d calls  $%.4f"
                             % (m, d["calls"], d["cost"]))
        return "\n".join(lines)


# ============================================================================
# Prices
# ============================================================================

# The public model list. Read with urllib rather than the SDK because it needs no
# authentication, and a missing key should not stop someone from seeing prices.
MODELS_URL = "https://openrouter.ai/api/v1/models"

_PRICE_CACHE: dict = {}


def model_prices(model: str) -> tuple[float, float]:
    """(input $/token, output $/token) for a model, from the live model list.

    Cached per process. Falls back to the published Pro/Flash figures if the
    lookup fails, so a network hiccup degrades the cost estimate rather than
    breaking the run.
    """
    if model in _PRICE_CACHE:
        return _PRICE_CACHE[model]

    fallback = (1e-6, 6e-6) if "pro" in model else (3.75e-7, 1.875e-6)
    prices = fallback
    try:
        request = urllib.request.Request(
            MODELS_URL, headers={"User-Agent": "wilcom-pipeline"})
        # Context manager, so the socket is closed even if json.load raises.
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        for entry in data.get("data", []):
            if entry.get("id") == model:
                pricing = entry.get("pricing") or {}
                prices = (float(pricing.get("prompt") or fallback[0]),
                          float(pricing.get("completion") or fallback[1]))
                break
    except Exception:
        # A price lookup that fails degrades the cost estimate; it must not stop
        # a run that is otherwise working.
        pass

    _PRICE_CACHE[model] = prices
    return prices


# ============================================================================
# JSON extraction
# ============================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model reply.

    Models wrap JSON in prose, in fences, or in both. Tried in order of
    confidence, and each attempt is narrow so a failure is diagnosable:

      1. the whole reply parses
      2. a fenced block parses
      3. the first balanced [ ... ] parses
      4. the first balanced { ... } parses
    """
    if not text:
        raise ValueError("empty reply")

    t = text.strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    for m in _FENCE_RE.finditer(t):
        body = m.group(1).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            continue

    # Bracket scan, candidates tried in the order they *appear* rather than by
    # bracket type. Ordering by type fails on a reply that opens an object before
    # an array ("text {"a": [1]} tail"), which would return the inner array.
    starts = []
    for ch in ("{", "["):
        i = t.find(ch)
        while i != -1:
            starts.append(i)
            i = t.find(ch, i + 1)

    for i in sorted(starts):
        open_ch = t[i]
        close_ch = "}" if open_ch == "{" else "]"
        depth, in_str, esc = 0, False, False
        for j in range(i, len(t)):
            ch = t[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:j + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("no JSON found in reply (%d chars): %s"
                     % (len(text), text[:200]))


# Hashing of stage inputs lives in `hashing.py`, not here. It is used for cache
# keys and dataset fingerprints, not only for API calls, so it does not belong to
# the client.


# ============================================================================
# Client
# ============================================================================


class TransientError(RuntimeError):
    """Worth retrying: rate limit, cold provider, 5xx, timeout."""


class PermanentError(RuntimeError):
    """Not worth retrying: bad request, auth, content policy."""


class Client:
    """OpenRouter wrapper with retries, caching and a cost ledger."""

    # Kept as `Client` because that is the name the tests and the pipeline already
    # import. `OpenRouterClient` is the clearer alias for new code.

    def __init__(self, cache_dir: Path | None = None, use_cache: bool = True,
                 verbose: bool = True):
        self._sdk = OpenRouter(api_key=config.api_key())
        self.ledger = Ledger()
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path("wilcom_work/cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / ("%s.json" % key)

    def _cache_get(self, key: str):
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, value) -> None:
        if not self.use_cache:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------ call
    def to_messages(self, request: dict) -> list:
        """A neutral request -> this provider's chat messages.

        OpenAI-compatible APIs expect media as base64 `data:` URIs inside a chat
        message, and the system prompt as a message with role "system". Gemini
        does neither, which is exactly why the conversion lives in the client
        rather than in the stages.
        """
        content = []
        for item in request.get("media", []):
            path, kind = (item[0], item[1]) if isinstance(item, tuple) \
                else (item, "image")
            uri = payloads.video_uri(path) if kind == "video" \
                else payloads.image_uri(path)
            content.append(payloads.video_part(uri) if kind == "video"
                           else payloads.image_part(uri))
        content.append(payloads.text_part(request["text"]))

        messages = []
        if request.get("system"):
            messages.append({"role": "system", "content": request["system"]})
        messages.append({"role": "user", "content": content})
        return messages

    def media(self, path, kind: str) -> tuple:
        """A neutral media reference; `to_messages` turns it into a data URI."""
        return (str(path), kind)

    def chat(self, model: str, request: dict, *, max_tokens: int | None = None,
             temperature: float | None = None, cache_key: str | None = None,
             retries: int = 4) -> dict:
        """One completion. Returns {text, usage, model, cached}.

        `cache_key` should capture everything that affects the result -- model,
        prompt, and the content digests. Two calls with the same key return the
        stored value without a request.

        Accepts either a neutral request dict or a pre-built messages list, so a
        caller that needs a custom message shape is not blocked.
        """
        if isinstance(request, list):
            messages = request
        else:
            messages = self.to_messages(request)
        return self._chat(model, messages, max_tokens=max_tokens,
                          temperature=temperature, cache_key=cache_key,
                          retries=retries)

    def _chat(self, model: str, messages: list, *, max_tokens: int | None = None,
              temperature: float | None = None, cache_key: str | None = None,
              retries: int = 4) -> dict:
        if cache_key:
            hit = self._cache_get(cache_key)
            if hit is not None:
                if self.verbose:
                    print("      [cache] %s" % cache_key)
                hit["cached"] = True
                return hit

        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        temperature = config.TEMPERATURE if temperature is None else temperature

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._sdk.chat.send(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                out = self._normalise(resp, model)
                if cache_key:
                    self._cache_put(cache_key, out)
                return out

            except Exception as exc:
                msg = str(exc).lower()
                last = exc

                permanent = any(k in msg for k in (
                    "invalid api key", "unauthorized", "401", "403",
                    "content policy", "invalid request", "400",
                    "context length", "no endpoints found"))
                if permanent:
                    self.ledger.add_failure()
                    raise PermanentError(str(exc)[:400]) from exc

                transient = any(k in msg for k in (
                    "rate limit", "429", "timeout", "timed out",
                    "502", "503", "504", "500", "overloaded",
                    "temporarily", "connection"))
                if not transient and attempt >= retries:
                    self.ledger.add_failure()
                    raise

                wait = min(60, 2 ** attempt * 2)
                if self.verbose:
                    print("      [retry %d/%d in %ds] %s"
                          % (attempt + 1, retries, wait, str(exc)[:120]))
                time.sleep(wait)

        self.ledger.add_failure()
        raise TransientError(str(last)[:400])

    # ------------------------------------------------------------------ parse
    def _normalise(self, resp, model: str) -> dict:
        """SDK response -> {text, usage}, tolerant of shape differences.

        The SDK is auto-generated and its response objects have changed shape
        between versions, so this reads defensively rather than assuming a
        single layout.
        """
        text = ""
        usage = {}

        d = resp if isinstance(resp, dict) else None
        if d is None:
            d = getattr(resp, "model_dump", lambda: None)()
        if d is None:
            d = json.loads(json.dumps(resp, default=str))

        usage = d.get("usage") or {}
        choices = d.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
            # Some Gemini routes put the answer in reasoning_content when the
            # visible content is empty and thinking is on.
            if not text.strip() and msg.get("reasoning_content"):
                text = msg["reasoning_content"]
        if not text.strip() and d.get("output_text"):
            text = d["output_text"]

        price_in, price_out = model_prices(model)
        self.ledger.add(model, usage, price_in, price_out)

        if self.verbose:
            pt = usage.get("prompt_tokens", "?")
            ct = usage.get("completion_tokens", "?")
            print("      [ok] %s  in=%s out=%s" % (model, pt, ct))

        return {"text": text, "usage": usage, "model": model, "cached": False}


# `Client` is the historical name; `OpenRouterClient` says which provider it is
# now that there is more than one. Both point at the same class.
OpenRouterClient = Client
