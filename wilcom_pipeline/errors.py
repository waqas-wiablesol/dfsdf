"""Error types shared by the client and the stages.

These lived in the OpenRouter client, which put a provider-specific module at the
bottom of the import graph for no reason -- every stage needed `PermanentError`
whether or not it cared which API was behind it. They belong somewhere neutral.

The distinction is the one that matters to a retry loop:

  * `TransientError`  waiting might help: rate limits, capacity spikes, timeouts.
  * `PermanentError`  waiting cannot help: bad key, no billing, malformed request,
                      content the provider refuses. Retrying these burns time and
                      makes a setup problem look like a flaky service.
"""

from __future__ import annotations


class TransientError(RuntimeError):
    """Worth retrying: rate limit, cold provider, 5xx, timeout."""


class PermanentError(RuntimeError):
    """Not worth retrying: bad request, auth, billing, content policy."""
