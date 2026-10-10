"""Building the API client.

There is one provider now, so this is thin by design. It exists as a seam rather
than an abstraction: `cli` and `pipeline` both need a configured client, and
having each construct its own would let them drift on the cache directory or on
verbosity. If a second provider is ever added, this is the one place that changes.

The provider-neutral request shape (`{"system", "text", "media"}` from
`prompts.trajectory_request` and friends) is kept even with a single
implementation, because that is what keeps the stages free of wire-format details.
"""

from __future__ import annotations


def model_for(role: str) -> str:
    """The default model for a role: 'trajectory', 'grounding' or 'meta'."""
    from . import config
    return config.model_for(role)


def make_client(*, cache_dir=None, use_cache: bool = True, verbose: bool = True,
                ledger_work=None):
    """Build a Gemini client.

    The import is inside the function so that `cli check`, `cli plan` and the
    offline half of stage 3 do not require the Gemini SDK to be installed.

    `ledger_work` routes every call into `<work>/usage.sqlite`. It is passed in
    rather than inferred from `cache_dir`: the two happen to share a parent today,
    and a caller that moved one would otherwise lose its ledger silently.
    """
    from .gemini_client import GeminiClient
    return GeminiClient(cache_dir=cache_dir, use_cache=use_cache,
                        verbose=verbose, ledger_work=ledger_work)
