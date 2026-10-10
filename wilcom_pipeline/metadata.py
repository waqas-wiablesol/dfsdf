"""Fetching video metadata and transcripts.

Paper Prompts 1 and 2 are *text-only*: they classify and score a candidate from
its metadata, with no video and no frames. That makes candidate screening cheap
enough to run on everything and expensive enough to justify doing properly, since
it is the paper's own coarse-to-fine filter (Section 3.1).

This module is the "tool" that supplies those fields. It uses `yt-dlp`, which
needs no API key and no billing, so a screening pass costs nothing until the two
prompts are called -- and those are a few hundred tokens each.

Nothing here downloads video. `--skip-download` is always set: screening a
candidate should not move 34 MB when 12 KB of text will do.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import log, video

# A metadata read is a text fetch and should take seconds. Measured healthy: 4-18 s.
# Anything approaching this means the network is being throttled, and a screening
# pass that cannot finish is worse than one that reports a failure.
METADATA_TIMEOUT_S = 120

# Captions are optional -- Prompt 1 ignores the transcript and Prompt 2 degrades
# without it -- so they get a much shorter leash than the metadata itself. This was
# the real cause of a 6.6-minute "fetching metadata" stall: two caption attempts
# each burning a full 180 s timeout before giving up.
CAPTION_TIMEOUT_S = 45

# The fields the prompts were written to consume.
FIELDS = ("title", "description", "tags", "channel", "category", "subtitle")


class MetadataError(RuntimeError):
    """yt-dlp could not read the URL."""


def _run(args: list, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run yt-dlp with a deadline that is actually enforced.

    Output goes to temporary FILES, not pipes. `capture_output=True` builds pipes,
    and every process yt-dlp starts inherits them; if such a grandchild outlives the
    timeout kill, reading the pipe never reaches EOF and the call blocks LONG past
    its deadline. Measured on this machine: `subprocess.run(timeout=3)` against a
    child that left a grandchild holding the pipe took 20.2 s. A file has no EOF to
    wait for, so `wait(timeout=)` is honoured.

    The network is bounded as well. yt-dlp defaults to 10 download retries, which
    turns a throttled video into minutes of silence before it admits failure, and
    silence is indistinguishable from a hang from the outside.

    Raises MetadataError on timeout rather than TimeoutExpired: the caller wants to
    record a reason for a screening decision, not to handle a subprocess detail.
    """
    cmd = [sys.executable, "-m", "yt_dlp",
           "--socket-timeout", "20",
           "--retries", "3",
           "--extractor-retries", "2"] + list(args)

    with tempfile.TemporaryDirectory(prefix="wilcom_ytdlp_") as tmp:
        out_path = Path(tmp) / "out.txt"
        err_path = Path(tmp) / "err.txt"
        timed_out = False
        with open(out_path, "w+", encoding="utf-8", errors="replace") as out, \
                open(err_path, "w+", encoding="utf-8", errors="replace") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                video.kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")

    if timed_out:
        raise MetadataError(
            "yt-dlp produced nothing within %ds and was stopped. Either the URL is "
            "throttled or the network is blocking it.\n%s"
            % (timeout, stderr[-400:]))

    return subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout, stderr)


# ============================================================================
# Metadata
# ============================================================================


def fetch(url: str, want_subtitles: bool = True) -> dict:
    """Metadata for a video URL, in the shape `prompts.format_metadata` expects.

    Returns keys: title, description, tags, channel, category, subtitle,
    duration_s, width, height, fps, video_id, webpage_url, upload_date.

    Raises MetadataError rather than returning a half-filled dict, because a
    screening decision made on empty metadata would be worse than no decision.

    `--no-playlist` is load-bearing. A URL copied from a playlist carries
    `&list=...`, and without this flag yt-dlp expands the WHOLE playlist and
    fetches full metadata for every entry before returning anything. Measured: a
    tutorial URL from a 44-video playlist hung past 43 s with no output, where the
    single video takes 4-18 s. Screening one candidate must never walk a playlist.
    """
    proc = _run(["--no-warnings", "--no-playlist", "--skip-download",
                 "--dump-json", url], timeout=METADATA_TIMEOUT_S)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise MetadataError("yt-dlp failed for %s:\n%s"
                            % (url, (proc.stderr or "")[-500:]))

    # A playlist URL yields one JSON object per line; take the first video and
    # note that others were ignored, rather than silently screening one of many.
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    try:
        info = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise MetadataError("could not parse yt-dlp output: %s" % exc) from exc

    categories = info.get("categories") or []
    meta = {
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "tags": info.get("tags") or [],
        "channel": (info.get("channel") or info.get("uploader") or ""),
        "category": ", ".join(str(c) for c in categories),
        "subtitle": "",
        "duration_s": info.get("duration"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "video_id": info.get("id"),
        "webpage_url": info.get("webpage_url") or url,
        "upload_date": info.get("upload_date"),
        "extra_entries": max(0, len(lines) - 1),
    }

    if want_subtitles:
        # A failure here is not fatal: Prompt 1 does not use the transcript, and
        # Prompt 2 degrades rather than breaks without it.
        #
        # Announced as its own step because it is the slow part. This used to run
        # silently under "fetching metadata", and caption attempts burning their
        # full timeout looked identical to a hung process -- 6.6 minutes of no
        # output for a field the prompts do not require.
        log.task("fetching captions (optional, not used by Prompt 1)")
        try:
            text, lang, kind = _subtitles(url, info)
            meta["subtitle"] = text
            meta["subtitle_language"] = lang
            meta["subtitle_kind"] = kind
            meta["subtitle_source"] = "captions"
            log.task("captions: %d characters in %r (%s-written)"
                     % (len(text), lang, kind))
        except Exception as exc:
            meta["subtitle"] = ""
            meta["subtitle_language"] = ""
            meta["subtitle_kind"] = ""
            # `str(exc)` distinguishes "no tracks at all" from "tracks exist but
            # none downloaded" -- a difference that matters, because only the
            # second is worth retrying.
            meta["subtitle_source"] = "unavailable: %s" % str(exc)[:120]
            log.task("captions unavailable: %s" % str(exc)[:90])

    return meta


# Every language after the preferred ones would be tried in turn, and YouTube
# advertises 157 automatic tracks per video, so an unbounded search could burn 157
# timeouts. Three attempts covers English, the video's own language, and a fallback.
MAX_CAPTION_LANGS = 3


def _subtitles(url: str, info: dict) -> tuple:
    """Best available caption track, as `(text, language, kind)`.

    Deliberately NOT English-only. Two things this got wrong:

      * **A Hindi tutorial has no English transcript.** YouTube advertises automatic
        captions in ~150 languages for every video, but those are machine
        TRANSLATIONS of whichever ASR ran. Measured on a Hindi Wilcom tutorial: the
        English track came back EMPTY while `hi` returned 4,309 characters and `ur`
        4,625 -- one error layer closer to what the speaker actually said. The old
        code looked only at `en`, found nothing, and discarded the rest.
      * **"No English caption track" was reported for every failure**, including a
        download that merely failed or timed out. That reads as "this video has no
        captions", which is false -- and it would have been recorded as evidence
        about instruction clarity.

    Order: human-written English, then human-written in the video's own language,
    then the same two for machine tracks. English first because the prompts are
    written for English text; Hindi second because this corpus rarely has English
    and almost always has Hindi. `format_metadata` labels which language was used,
    so the model is not left judging a Hindi transcript as unclear English.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    original = (info.get("language") or "").split("-")[0]

    # English first, then Hindi. This corpus is Hindi and Urdu embroidery
    # tutorials: a Hindi video has no real English transcript, and Hindi captions
    # are almost always available, so Hindi is the practical fallback. Urdu is
    # third for the minority of videos spoken in it, then the video's own detected
    # language, then anything -- so a video with only Bangla captions still yields
    # a transcript instead of nothing.
    preferred = ["en", "en-US", "en-GB", "hi"]
    for extra in ("ur", original):
        if extra and extra not in preferred:
            preferred.append(extra)

    tried: list = []
    for source, kind in ((manual, "manual"), (auto, "auto")):
        if not source:
            continue
        available = [lang for lang in preferred if lang in source]
        # Nothing preferred is present: fall back to whatever exists, so a video
        # with only, say, Bangla captions still yields a transcript.
        if not available:
            available = [l for l in source if l != "live_chat"]
        for lang in available[:MAX_CAPTION_LANGS]:
            for entry in source.get(lang) or []:
                ext = entry.get("ext")
                if ext not in ("vtt", "srt", "json3"):
                    continue
                try:
                    text = _download_caption(url, lang, ext,
                                             auto=(kind == "auto"))
                except Exception as exc:
                    tried.append("%s/%s %s" % (kind, lang, type(exc).__name__))
                    break
                if text:
                    return text, lang, kind
                tried.append("%s/%s empty" % (kind, lang))
                break

    if manual or auto:
        raise RuntimeError("caption tracks exist but none downloaded (%s)"
                           % (", ".join(tried) or "no usable format"))
    raise RuntimeError("video has no caption tracks at all")


def _download_caption(url: str, lang: str, ext: str, auto: bool) -> str:
    """Download one caption track into a temp dir and convert it to text.

    Bounded by `CAPTION_TIMEOUT_S` rather than the metadata timeout: this is
    optional data, so it must not be able to dominate a screening pass.
    """
    with tempfile.TemporaryDirectory(prefix="wilcom_subs_") as tmp:
        flag = "--write-auto-subs" if auto else "--write-subs"
        proc = _run([flag, "--sub-langs", lang, "--sub-format", ext,
                     "--skip-download", "--no-warnings", "--no-playlist",
                     "-o", "%(id)s.%(ext)s", "--paths", tmp, url],
                    timeout=CAPTION_TIMEOUT_S)
        if proc.returncode != 0:
            return ""
        candidates = list(__import__("pathlib").Path(tmp).glob("*"))
        for path in candidates:
            if path.suffix.lstrip(".") == ext:
                return caption_to_text(path.read_text(encoding="utf-8",
                                                      errors="replace"), ext)
    return ""


def caption_to_text(raw: str, ext: str) -> str:
    """Caption file -> readable prose, deduplicated.

    Auto-captions use a rolling-window format where each cue repeats the previous
    line, so the naive text contains every phrase two or three times. That inflates
    the transcript without adding information, and it makes the text read as
    stuttering -- which is precisely what Prompt 2 is asked to judge. Consecutive
    duplicates are therefore removed.
    """
    if ext == "json3":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ""
        pieces = []
        for event in data.get("events") or []:
            for seg in event.get("segs") or []:
                pieces.append(seg.get("utf8") or "")
        text = "".join(pieces)
        return re.sub(r"\s+", " ", text).strip()

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):        # srt index
            continue
        # Strip vtt inline timing tags and speaker annotations.
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"^\s*[A-Z][A-Za-z .]{0,24}:\s*", "", line)
        lines.append(line)

    out = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


# ============================================================================
# First-minute clip, for Prompt 2
# ============================================================================
#
# The paper is explicit that quality scoring is NOT text-only:
#
#   "we leverage an omnimodal model that supports diverse inputs including text,
#    video, and audio to assess video quality. For each candidate video, we
#    extract the first minute and evaluate it along three complementary
#    dimensions."
#
# This matters more than it sounds. Running Prompt 2 on metadata alone produced a
# `recording_quality` score of 4 whose own reasoning admitted "the visual file
# itself isn't viewable" -- a positive score on a dimension the model could not
# see. Text cannot support a judgement about blur, resolution or framing, so the
# dimension is unfalsifiable without pixels.

PROBE_SECONDS = 60


def fetch_probe_clip(url: str, seconds: int = PROBE_SECONDS,
                     dest_dir=None) -> Path:
    """Download the first `seconds` of a video, for Prompt 2.

    Reads the opening minute rather than the whole file, which is all the paper
    scores and keeps screening cheap: a minute of 1080p60 screen content is a few
    megabytes, against tens for a full tutorial.

    Section download needs ffmpeg, which `video.ffmpeg_path()` supplies. Returns
    the path, or raises MetadataError with the reason.
    """
    import pathlib
    import tempfile

    dest_dir = pathlib.Path(dest_dir or tempfile.mkdtemp(prefix="wilcom_probe_"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    # `*start-end` is yt-dlp's section syntax; force-keyframes-at-cuts makes the
    # first frame decodable rather than a grey smear from a partial GOP.
    mm, ss = divmod(int(seconds), 60)
    section = "*0:00-%d:%02d" % (mm, ss)

    proc = _run(["--no-warnings", "--no-playlist", "--newline",
                 "--download-sections", section,
                 "--force-keyframes-at-cuts",
                 "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
                 "--merge-output-format", "mp4",
                 "-o", str(dest_dir / "%(id)s.%(ext)s"), url], timeout=600)
    if proc.returncode != 0:
        raise MetadataError("probe clip download failed:\n%s"
                            % (proc.stderr or "")[-500:])

    videos = [p for p in dest_dir.iterdir() if p.suffix.lower() == ".mp4"
              and p.stat().st_size > 0]
    if not videos:
        raise MetadataError("probe clip produced no file in %s" % dest_dir)
    return max(videos, key=lambda p: p.stat().st_size)


# ============================================================================
# Convenience for the CLI
# ============================================================================


def summarise(meta: dict) -> str:
    """One-line human summary, for logs."""
    bits = [meta.get("title") or "(untitled)"]
    if meta.get("duration_s"):
        bits.append("%.0fs" % meta["duration_s"])
    if meta.get("width") and meta.get("height"):
        bits.append("%dx%d" % (meta["width"], meta["height"]))
    if meta.get("channel"):
        bits.append(meta["channel"])
    return "  ".join(str(b) for b in bits)
