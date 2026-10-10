"""Gemini client, via the official `google-genai` SDK.

The pipeline's only API client. The stages talk to a small provider-neutral
surface:

    client.chat(model, request, cache_key=...)  ->  {"text", "usage", "model", "cached"}
    client.ledger                               ->  Ledger
    client.media(path, kind)                    ->  the provider's media handle

A *request* is the neutral shape built by `prompts.trajectory_request` and
`prompts.grounding_request`:

    {"system": str, "text": str, "media": [handle, ...]}

That indirection is kept even with a single implementation, because it is what
keeps the stages free of wire-format details. Three of them matter here:

  * **Media is bytes, not URIs.** Gemini rejects `data:` URLs inside a chat
    message; a part is `types.Part.from_bytes(data=..., mime_type=...)`. So the
    client reads the file, and the old data-URI encoding step is gone entirely.
  * **The system prompt is a separate field**, not a message with role "system".
  * **Thinking cannot be disabled** on Gemini 3, and it bills as output tokens.
    `usage_metadata.thoughts_token_count` is what the ledger has to read.
"""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path

from . import config, hashing, ledger as ledger_mod, log, store
from .errors import PermanentError, TransientError
from .ledger import Ledger

# --- SDK import, with a clear failure ---------------------------------------

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit(
        "The Gemini SDK is not installed.\n"
        "    pip install google-genai"
    )


# ============================================================================
# Part construction
# ============================================================================


def _mime_for(path: Path) -> str:
    """MIME type for a local file.

    `mimetypes` returns None for some extensions on some platforms, so the
    important ones are pinned. A wrong MIME type is rejected by Gemini with a
    message about the file, not about the type, which is a confusing failure to
    debug.
    """
    suffix = path.suffix.lower()
    pinned = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg", ".webp": "image/webp",
              ".mov": "video/mov", ".webm": "video/webm"}
    if suffix in pinned:
        return pinned[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def media_part(path, kind: str, uploaded=None):
    """A Gemini content part for a local file.

    `uploaded` is an optional Files API object; when given it is referenced by
    URI instead of being re-encoded inline. Inside a 20 MB request the inline
    form is simpler and avoids the upload state machine, so that is the default.
    """
    path = Path(path)
    if uploaded is not None:
        return genai_types.Part.from_uri(file_uri=uploaded.uri,
                                         mime_type=uploaded.mime_type)
    return genai_types.Part.from_bytes(data=path.read_bytes(),
                                       mime_type=_mime_for(path))


# ============================================================================
# Response normalisation
# ============================================================================


def _usage_dict(usage) -> dict:
    """`usage_metadata` -> the dict shape `Ledger.add` expects.

    Gemini names things differently from OpenAI, and thinking tokens are the line
    item that matters most on this provider, so they are mapped explicitly rather
    than left to be silently dropped.
    """
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        # `Ledger` reads this nested key and adds it to the reasoning total.
        "completion_tokens_details": {
            "reasoning_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
        },
    }


# ============================================================================
# Client
# ============================================================================


class GeminiClient:
    """Gemini wrapper with retries, caching and a cost ledger."""

    def __init__(self, cache_dir=None, use_cache: bool = True,
                 verbose: bool = True, api_key: str | None = None,
                 ledger_work=None):
        # `ledger_work` is the work tree the SQLite ledger belongs to. Optional, so a
        # caller with no work tree (a test, a one-off script) simply records nothing.
        self.ledger_work = ledger_work
        self._sdk = genai.Client(api_key=api_key or config.api_key())
        self.ledger = Ledger()
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path("wilcom_work/cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    #
    # Keys are prefixed so a cache directory shared with anything else cannot
    # serve the wrong provider's answer.

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / ("gemini_%s.json" % key)

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

    # ------------------------------------------------------------- conversion
    def to_contents(self, request: dict) -> list:
        """A neutral request -> Gemini `contents`.

        Media first, then the text, matching the paper's guidance and Google's
        own recommendation to place the prompt after the media for a single
        media item.
        """
        contents = []
        for item in request.get("media", []):
            if isinstance(item, tuple):
                path, kind = item[0], item[1]
                contents.append(media_part(path, kind))
            else:
                contents.append(item)          # already a Part
        contents.append(request["text"])
        return contents

    # ------------------------------------------------------------------ call
    def _effective_key(self, cache_key: str, model: str, temperature: float,
                       max_tokens: int, request: dict) -> str:
        """The cache key, extended with everything that changes the answer.

        A cache key must cover *every* input that affects the response, or a
        changed setting returns a stale result that looks like the setting had no
        effect. `thinking_level` is exactly that kind of input and was initially
        missing from the key: switching level would have replayed the previous
        level's answer, which would have made this very comparison impossible to
        run and quietly wrong instead.

        Built here rather than in the stages so no caller can forget, and
        including the provider name so one cache directory can hold both.
        """
        return hashing.fingerprint(
            "gemini", cache_key or "nocache", model,
            "t=%s" % temperature, "m=%s" % max_tokens,
            "think=%s" % (request.get("thinking_level") or config.THINKING_LEVEL),
            "schema=%s" % bool(request.get("json_schema")),
        )

    def chat(self, model: str, request: dict, *, max_tokens: int | None = None,
             temperature: float | None = None, cache_key: str | None = None,
             retries: int = 5) -> dict:
        """One completion. Returns {text, usage, model, cached}.

        Retries are more aggressive than a default SDK setup: measured on this
        project, Gemini returns a transient 503 UNAVAILABLE on a large share of
        first attempts, including on a trivial 8-token call. A five-attempt
        exponential backoff absorbs that.

        `cache_key` is *extended* with the generation settings, so two calls that
        differ only in temperature or thinking level cannot collide.
        """
        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        temperature = config.TEMPERATURE if temperature is None else temperature

        raw_key = cache_key
        if cache_key:
            cache_key = self._effective_key(cache_key, model, temperature,
                                            max_tokens, request)
            hit = self._cache_get(cache_key)
            if hit is not None:
                log.detail("[cache] %s" % cache_key)
                hit["cached"] = True
                # Recorded as a replay, with no tokens and no cost: it was billed
                # when the original call was made, and counting it again would
                # double the reported spend.
                store.record_call(self.ledger_work, model, raw_key, None,
                                  cached=True)
                return hit

        contents = self.to_contents(request)
        cfg = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if request.get("system"):
            cfg.system_instruction = request["system"]
        if request.get("json_schema"):
            # Structured output. Measured working on this account, and it removes
            # the need to parse JSON out of prose.
            cfg.response_mime_type = "application/json"
            cfg.response_schema = request["json_schema"]

        # Thinking is the single largest cost and time item on Pro: measured at
        # 11,681 thinking tokens per clip against 4,101 of actual answer. Gemini 3
        # cannot switch it off, so the level is set explicitly rather than left at
        # the model default. `thinking_level` is the modern field; older models
        # take `thinking_budget`, which is only used if the level is rejected.
        level = request.get("thinking_level") or config.THINKING_LEVEL
        if level:
            cfg.thinking_config = genai_types.ThinkingConfig(thinking_level=level)

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                log.detail("calling %s (%s media part(s), %s max out)"
                           % (model, len(request.get("media", [])),
                              log.tokens(cfg.max_output_tokens)))
                started = time.time()
                # The heartbeat is what makes a multi-minute call legible: the
                # provider returns nothing until the answer is complete, so
                # without it a working call and a hung one look the same.
                with log.heartbeat("%s" % model):
                    resp = self._sdk.models.generate_content(
                        model=model, contents=contents, config=cfg)
                took = time.time() - started

                text = getattr(resp, "text", None) or ""
                usage = _usage_dict(getattr(resp, "usage_metadata", None))

                price_in, price_out = ledger_mod.price_for(model)
                self.ledger.add(model, usage, price_in, price_out)

                think = usage["completion_tokens_details"]["reasoning_tokens"]
                cost = ledger_mod.cost_of(
                    model, usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0), think)
                log.ok("%.1fs  in %s / out %s (think %s)  %s"
                       % (took, log.tokens(usage.get("prompt_tokens")),
                          log.tokens(usage.get("completion_tokens")),
                          log.tokens(think), log.money(cost)))

                out = {"text": text, "usage": usage, "model": model,
                       "cached": False, "seconds": round(took, 2),
                       "cost_usd": round(cost, 6)}
                if cache_key:
                    self._cache_put(cache_key, out)

                # Appended to the ledger under the RAW key -- the one the caller
                # passed, which the trajectory and grounding records also store. That
                # makes "what did this video cost" an exact join rather than a guess,
                # and survives any future change to how the effective key is built.
                # Never raises: this runs after money has been spent.
                store.record_call(self.ledger_work, model, raw_key, usage,
                                  seconds=took, cached=False, usd=cost)
                return out

            except Exception as exc:
                last = exc
                msg = str(exc).lower()
                code = getattr(exc, "code", None)

                # 402 and 403 will not improve by waiting; 429 might, if it is a
                # per-minute rate limit rather than a spent balance.
                permanent = any(k in msg for k in (
                    "api key not valid", "api_key_invalid", "permission denied",
                    "invalid argument", "not found", "prepayment credits",
                )) or code in (400, 401, 403, 404, 402)
                if permanent:
                    self.ledger.add_failure()
                    raise PermanentError(str(exc)[:300]) from exc

                if attempt >= retries:
                    break

                wait = min(30, 2 ** attempt * 2)
                log.warn("attempt %d/%d failed, retrying in %ds: %s"
                         % (attempt + 1, retries, wait, str(exc)[:110]))
                time.sleep(wait)

        self.ledger.add_failure()
        raise TransientError(str(last)[:300])

    # ------------------------------------------------------------ media refs
    def media(self, path, kind: str) -> tuple:
        """A neutral media reference for a request.

        Returns a tuple the request builder stores verbatim and `to_contents`
        turns into a Part. Reading the file is deferred to call time, so building
        a request does no I/O.
        """
        return (str(path), kind)
