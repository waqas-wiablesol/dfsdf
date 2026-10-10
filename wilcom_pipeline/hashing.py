"""Content hashing, for cache keys and change detection.

Every stage caches on the hash of its *inputs* -- clip bytes, frame bytes, prompt
version -- so a re-run after a crash reuses what is already on disk instead of
re-paying for it. That makes these two functions load-bearing rather than
incidental: a hash that is too weak silently serves a stale result, which is
worse than no cache at all.
"""

from __future__ import annotations

import hashlib

# Read files in 1 MB chunks. A 4-minute 720p clip is ~45 MB, and hashing it
# should not require it to be resident.
_CHUNK = 1 << 20


def file_digest(path) -> str:
    """Full sha256 of a file's bytes.

    Used where a *content* change must invalidate a cache: a clip re-encoded at a
    different CRF must produce a different key, or the cached trajectory would
    describe video that no longer exists.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(*parts) -> str:
    """Short, stable hash of a sequence of strings.

    Each part is NUL-terminated before hashing, so ("ab", "c") and ("a", "bc")
    hash differently. Plain concatenation would let those two collide, which for
    a cache key means returning the wrong stage's result.

    16 hex characters is 64 bits: collisions are not a practical concern for a
    corpus of thousands of clips, and short keys keep the cache directory
    readable.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
