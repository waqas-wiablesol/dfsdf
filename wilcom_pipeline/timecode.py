"""Time-code parsing and formatting.

Timestamps arrive from a model as text ("01:23") and are converted to float
seconds at exactly two boundaries: when reading a reply, and when extracting a
frame. Everything in between is seconds.

This is its own module because a mis-parsed timestamp does not raise. It quietly
extracts the frame at the wrong moment, and the resulting row looks fine. That
failure mode is bad enough that the parsing deserves to be isolated and tested
on its own rather than buried in a stage.
"""

from __future__ import annotations

import re

# "mm:ss" or "hh:mm:ss". Minutes are allowed to exceed 59, because a model that
# has seen a long video will emit "75:30" meaning 1h15m30s.
_CLOCK_RE = re.compile(r"^\s*(\d{1,3}):(\d{2})(?::(\d{2}))?\s*$")

# Brackets and quotes that a model may wrap a timestamp in.
_TRIM = "[](){}'\" \t"


def to_seconds(stamp) -> float:
    """Convert 'mm:ss', 'hh:mm:ss', or bare seconds to float seconds.

    Raises ValueError on anything else. Deliberately strict: returning 0.0 for an
    unparseable stamp would silently extract the frame at the start of the video,
    which is exactly the failure this function exists to prevent. Callers that
    can tolerate a bad stamp must drop the action explicitly.
    """
    if stamp is None:
        raise ValueError("no timestamp")

    text = str(stamp).strip().strip(_TRIM)

    m = _CLOCK_RE.match(text)
    if m:
        first, second, third = m.group(1), m.group(2), m.group(3)
        if third is None:                       # mm:ss
            return int(first) * 60 + float(second)
        return int(first) * 3600 + int(second) * 60 + float(third)  # hh:mm:ss

    # Bare seconds, e.g. "195" or "195.5".
    try:
        return float(text)
    except ValueError:
        raise ValueError("unparseable timestamp: %r" % stamp) from None


def from_seconds(seconds: float) -> str:
    """Float seconds to 'mm:ss', or 'hh:mm:ss' past an hour.

    Used whenever a timestamp goes *into* a prompt, so the model sees the same
    shape the paper's examples use.
    """
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = int(total % 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)
