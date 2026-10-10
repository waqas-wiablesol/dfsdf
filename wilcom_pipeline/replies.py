"""Pulling a JSON value out of a model reply.

Models wrap JSON in prose, in fenced blocks, or in both, and the wrapping varies
run to run. This is the tolerant reader that was previously in the OpenRouter
client, where it had no business being: parsing a reply is not a provider concern,
and three stages plus the tests need it.

Each attempt is narrow and tried in order of confidence, so a failure is
diagnosable rather than a single opaque exception.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str):
    """Pull a JSON value out of a model reply.

    Tried in order:

      1. the whole reply parses
      2. a fenced block parses
      3. the first balanced `{...}` or `[...]`, whichever starts earliest

    Step 3 scans candidates in the order they *appear* rather than by bracket
    type. Ordering by type fails on a reply that opens an object before an array,
    such as `text {"a": [1]} tail`, where it would return the inner array.
    """
    if not text:
        raise ValueError("empty reply")

    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in _FENCE_RE.finditer(stripped):
        body = match.group(1).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            continue

    starts = []
    for char in ("{", "["):
        index = stripped.find(char)
        while index != -1:
            starts.append(index)
            index = stripped.find(char, index + 1)

    for start in sorted(starts):
        open_char = stripped[start]
        close_char = "}" if open_char == "{" else "]"
        depth, in_string, escaped = 0, False, False
        for position in range(start, len(stripped)):
            char = stripped[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start:position + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("no JSON found in reply (%d chars): %s"
                     % (len(text), text[:200]))
