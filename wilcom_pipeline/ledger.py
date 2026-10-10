"""Cost tracking and model pricing.

Two things that belong together and previously lived inside the OpenRouter client:

  * `Ledger`, which accumulates calls, tokens and spend for a run.
  * `price_for`, which turns a model id into a per-token rate.

Prices are now a local table rather than a live lookup, because Gemini's Developer
API exposes no pricing endpoint. That is a real trade: the table cannot follow a
price change, so it carries a date and a note to re-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per token, from the Gemini Developer API pricing page.
#
# Figures are for prompts up to 200k tokens. Above that Gemini roughly doubles the
# rate, and a 4-minute clip is ~19k tokens, so the low tier is the right one here.
# `price_for` warns rather than silently applying the wrong tier.
#
# Checked 2026-09-30. Prices change; this table does not follow them.
_PRICES_UNDER_200K = {
    "gemini-3.1-pro-preview": (2.00e-6, 12.00e-6),
    "gemini-3.1-pro-preview-customtools": (2.00e-6, 12.00e-6),
    "gemini-3.8-flash": (0.75e-6, 3.75e-6),
    "gemini-3.7-flash": (0.75e-6, 3.75e-6),
    "gemini-3.6-flash": (0.75e-6, 3.75e-6),
    "gemini-3.5-flash": (1.50e-6, 9.00e-6),
    "gemini-3.5-flash-lite": (0.30e-6, 2.50e-6),
    "gemini-3.1-flash-lite": (0.25e-6, 1.50e-6),
    "gemini-3-flash-preview": (0.50e-6, 3.00e-6),
}

# Applied to any model not in the table. The most expensive tier, so an unknown
# model makes the ledger pessimistic rather than flattering -- a cost report that
# understates spend is worse than one that overstates it.
_UNKNOWN_MODEL_PRICE = (2.00e-6, 12.00e-6)

# Above this prompt size Gemini charges roughly double.
_LONG_PROMPT_TOKENS = 200_000


def price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per token for a model id."""
    return _PRICES_UNDER_200K.get(model, _UNKNOWN_MODEL_PRICE)


def is_long_prompt(prompt_tokens: int) -> bool:
    """Whether a prompt crosses into Gemini's higher price tier."""
    return (prompt_tokens or 0) > _LONG_PROMPT_TOKENS


def cost_of(model: str, prompt_tokens: int, output_tokens: int,
            thinking_tokens: int = 0) -> float:
    """USD for one call.

    `thinking_tokens` is billed at the *output* rate and is not included in the
    provider's `completion_tokens`, so passing it separately is deliberate: on
    Gemini 3 it has been the largest single line item, and omitting it would
    understate a run by roughly 2x.
    """
    price_in, price_out = price_for(model)
    return ((prompt_tokens or 0) * price_in
            + ((output_tokens or 0) + (thinking_tokens or 0)) * price_out)


@dataclass
class Ledger:
    """Running spend, so a batch's cost is measured rather than guessed."""

    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, usage: dict, price_in: float,
            price_out: float) -> None:
        """Record one call.

        Takes the usage dict in the shape the Gemini client produces, which is
        normalised from `usage_metadata`: `prompt_tokens`, `completion_tokens`, and
        `completion_tokens_details.reasoning_tokens`.
        """
        self.calls += 1
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        thinking = int((usage.get("completion_tokens_details") or {})
                       .get("reasoning_tokens") or 0)

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.reasoning_tokens += thinking

        cost = (prompt * price_in
                + (completion + thinking) * price_out)
        self.cost_usd += cost

        entry = self.by_model.setdefault(
            model, {"calls": 0, "in": 0, "out": 0, "think": 0, "cost": 0.0})
        entry["calls"] += 1
        entry["in"] += prompt
        entry["out"] += completion
        entry["think"] += thinking
        entry["cost"] += cost

    def add_failure(self) -> None:
        self.failures += 1

    def report(self) -> str:
        think_share = ""
        if self.completion_tokens + self.reasoning_tokens:
            share = 100.0 * self.reasoning_tokens / (
                self.completion_tokens + self.reasoning_tokens)
            think_share = "  (%.0f%% of all output)" % share
        lines = [
            "calls           : %d  (%d failed)" % (self.calls, self.failures),
            "prompt tokens   : {:,}".format(self.prompt_tokens),
            "output tokens   : {:,}".format(self.completion_tokens),
            "thinking tokens : {:,}{}".format(self.reasoning_tokens, think_share),
            "estimated cost  : $%.4f" % self.cost_usd,
        ]
        if len(self.by_model) > 1:
            lines.append("by model:")
            for model, entry in sorted(self.by_model.items()):
                lines.append("   %-34s %3d calls  %9s tok  $%.4f"
                             % (model, entry["calls"],
                                "{:,}".format(entry["in"] + entry["out"]
                                              + entry["think"]),
                                entry["cost"]))
        return "\n".join(lines)
