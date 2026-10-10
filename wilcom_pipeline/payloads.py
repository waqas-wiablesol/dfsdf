"""Request payload pieces: file encoding and content-part shapes.

Everything that turns a local file into something the API accepts lives here,
for two reasons:

  * **One place to change.** The content-part shapes are defined by the provider,
    not by us. When they change, one file changes rather than four.
  * **The cost stays visible.** A data URI is base64 of the entire file, held in
    memory as a string. For a 4-minute clip that is ~60 MB, roughly a third
    larger than the file on disk. Keeping the encoding in a small module makes
    that overhead something you can see, instead of something that quietly
    happens inside video handling.

Nothing here reads a video's contents or knows what a frame is; it only encodes
bytes and builds the dicts the API expects.
"""

from __future__ import annotations

import base64
from pathlib import Path

# Extension to MIME type. Only the formats this pipeline actually produces are
# listed; anything else falls back to PNG, which is what frame extraction writes.
_IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}

VIDEO_MIME = "video/mp4"


def data_uri(path, mime: str) -> str:
    """Encode a file as a `data:` URI.

    Deliberately reads the whole file in one call. Streaming into base64 would
    halve peak memory, but the SDK needs a single string argument anyway, so the
    encoded form is what dominates and the streaming would only add complexity.
    """
    blob = Path(path).read_bytes()
    return "data:%s;base64,%s" % (mime, base64.b64encode(blob).decode("ascii"))


def image_uri(path) -> str:
    """Data URI for an image, with the MIME type taken from its extension."""
    suffix = Path(path).suffix.lower().lstrip(".")
    return data_uri(path, _IMAGE_MIME.get(suffix, "image/png"))


def video_uri(path) -> str:
    """Data URI for an MP4 clip."""
    return data_uri(path, VIDEO_MIME)


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------
#
# Written as small named functions rather than inline dicts at the call sites,
# so a change to the wire format has one home. The nested shape is
# {"type": ..., "<type>": {"url": ...}} and is not symmetric, which is the kind
# of detail that is easy to get subtly wrong from memory.


def text_part(text: str) -> dict:
    """A text content part."""
    return {"type": "text", "text": text}


def image_part(uri: str) -> dict:
    """An image content part, from a data URI."""
    return {"type": "image_url", "image_url": {"url": uri}}


def video_part(uri: str) -> dict:
    """A video content part, from a data URI.

    `video_url` rather than `image_url`. Verified against the live model list:
    a model must advertise `video` in its modalities for this to be accepted.
    """
    return {"type": "video_url", "video_url": {"url": uri}}
