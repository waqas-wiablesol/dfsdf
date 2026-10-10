"""Stage 0 -- candidate screening. The paper's Section 3.1 coarse-to-fine filter.

Two prompts, both **text only**:

    Prompt 1  is this GUI operation content at all?     (PDF page 18)
    Prompt 2  how good is it, on three 1-5 dimensions?   (PDF pages 19-20)

Running these first is what makes the rest affordable. They cost a few hundred
tokens and no video upload, so a candidate is screened for roughly a thousandth
of what processing it costs. The pipeline previously skipped this entirely and
went straight to a 4-minute video call, which meant every judgement about whether
a video was *worth* processing was being made by hand.

Two independent gates, and both are recorded whether or not they pass:

    Prompt 1 -> is_gui_content must be true, confidence >= MIN_CONFIDENCE
    Prompt 2 -> every dimension >= MIN_SCORE

The record keeps every field, including the rejections. A screening log that only
lists survivors cannot answer "why was this rejected", which is the question that
gets asked when a good video is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import (config, hashing, log, metadata as metadata_mod, prompts, records,
               timecode)
from .errors import PermanentError
from .replies import extract_json

PROMPT_VERSION = "filter-v1"

# Prompt 2's dimensions, in the order the paper lists them.
DIMENSIONS = ("topic_relevance", "instruction_clarity", "recording_quality")


# ============================================================================
# Parsing
# ============================================================================


def parse_meta_verdict(reply: str) -> dict:
    """Prompt 1's reply -> a normalised verdict.

    Structured output is requested, but parsed defensively anyway: a provider that
    ignores `response_schema` returns prose, and the whole screening pass would
    then fail on a formatting difference rather than a data one.
    """
    data = extract_json(reply)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("meta verdict is not an object")

    flag = data.get("is_gui_content")
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "yes", "1")

    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    # Some models answer confidence on a 0-100 scale despite the prompt.
    if confidence > 1.0:
        confidence = confidence / 100.0

    return {
        "is_gui_content": bool(flag),
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": str(data.get("reasoning") or "").strip(),
    }


def parse_quality_scores(reply: str) -> dict:
    """Prompt 2's reply -> {dimension: score} plus the summary and reasons."""
    data = extract_json(reply)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("quality verdict is not an object")

    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("quality verdict has no 'scores' object")

    scores, reasons = {}, {}
    for dim in DIMENSIONS:
        entry = raw_scores.get(dim)
        if isinstance(entry, dict):
            value = entry.get("score")
            reasons[dim] = str(entry.get("reasoning") or "").strip()
        else:
            value = entry
            reasons[dim] = ""
        try:
            scores[dim] = int(value)
        except (TypeError, ValueError):
            scores[dim] = 0

    return {
        "scores": scores,
        "reasons": reasons,
        "overall_summary": str(data.get("overall_summary") or "").strip(),
    }


# ============================================================================
# Decisions
# ============================================================================


def verdict(meta: dict, quality: dict, duration_s=None) -> tuple:
    """(accepted, reasons) for a screened candidate.

    Returns the list of reasons either way, so a rejection is explainable and an
    acceptance records what it passed on. Thresholds come from config so they can
    be argued with rather than buried.

    `duration_s` is passed separately because this function's `meta` is Prompt 1's
    *verdict* (`{is_gui_content, confidence, reasoning}`), not the video metadata --
    the two are easy to confuse and the duration is not in the verdict.

    The duration rule is the paper's own candidate selection:

        "we selected only those with durations under 12 minutes based on metadata
         information"

    It is checked first, because it is free and it is the one criterion that is
    knowable before spending anything. An unknown duration is NOT treated as a
    failure -- the metadata was unreadable, which is different from the video being
    too long, and blocking on it would reject videos for a reason that is not true.
    """
    reasons = []

    # ---- duration (the paper's selection rule, and the cheapest check) ------
    # Formatted mm:ss rather than "%.1f min": at one decimal, 719s rounds to
    # "12.0 min is under the 12-minute limit", which reads as a contradiction.
    too_long = False
    limit_min = config.MAX_DURATION_S // 60
    if duration_s is None:
        reasons.append("Duration: unknown, so the %d-minute limit could not be "
                       "checked" % limit_min)
    else:
        shown = timecode.from_seconds(duration_s)
        if duration_s >= config.MAX_DURATION_S:
            too_long = True
            reasons.append("Duration: %s exceeds the paper's %d-minute limit"
                           % (shown, limit_min))
        else:
            reasons.append("Duration: %s is under the %d-minute limit"
                           % (shown, limit_min))

    if not meta.get("is_gui_content"):
        reasons.append("Prompt 1: not GUI operation content")
    elif meta.get("confidence", 0) < config.MIN_META_CONFIDENCE:
        reasons.append("Prompt 1: confidence %.2f below %.2f"
                       % (meta.get("confidence", 0), config.MIN_META_CONFIDENCE))
    else:
        reasons.append("Prompt 1: GUI content, confidence %.2f"
                       % meta.get("confidence", 0))

    below = {d: quality["scores"].get(d, 0) for d in DIMENSIONS
             if quality["scores"].get(d, 0) < config.MIN_QUALITY_SCORE}
    if below:
        reasons.append("Prompt 2: below %d on %s"
                       % (config.MIN_QUALITY_SCORE,
                          ", ".join("%s=%d" % (k, v) for k, v in below.items())))
    else:
        reasons.append("Prompt 2: all dimensions >= %d (%s)"
                       % (config.MIN_QUALITY_SCORE,
                          ", ".join("%s=%d" % (d, quality["scores"].get(d, 0))
                                    for d in DIMENSIONS)))

    accepted = (not below and not too_long and meta.get("is_gui_content")
                and meta.get("confidence", 0) >= config.MIN_META_CONFIDENCE)
    return accepted, reasons


# ============================================================================
# Stage entry point
# ============================================================================


def _cache_key(kind: str, payload: str, model: str) -> str:
    return hashing.fingerprint(PROMPT_VERSION, kind, model,
                               prompts.screening_fingerprint(), payload)


def screen_one(url: str, client, paths: config.Paths,
               model: str | None = None, meta: dict | None = None,
               force: bool = False, verbose: bool = True) -> dict:
    """Screen one candidate. Returns the screening record and writes it.

    `meta` allows a caller that has already fetched metadata to avoid a second
    yt-dlp call, which is the common case when screening a list.
    """
    model = model or config.model_for("meta")
    paths.ensure()

    if meta is None:
        log.step("fetching metadata (yt-dlp, no API cost)")
        meta = metadata_mod.fetch(url)
        log.task("metadata read: %s  %ss  %sx%s"
                 % ((meta.get("title") or "")[:48],
                    int(meta.get("duration_s") or 0),
                    meta.get("width"), meta.get("height")))
    log.detail("%s" % metadata_mod.summarise(meta))

    stem = _record_name(url, meta)
    out_path = paths.screen / ("%s.json" % stem)
    if out_path.exists() and not force:
        cached = records.read_json(out_path)
        if isinstance(cached, dict) and cached.get("prompt_version") == PROMPT_VERSION:
            log.info("[skip] %s already screened: %s"
                     % (stem, "ACCEPTED" if cached.get("accepted") else "rejected"))
            return cached

    # Metadata is what feeds Prompt 1, so it is the natural cache payload for
    # that call. Prompt 2's key also folds in the probe clip's bytes, because its
    # answer depends on the video and not only on the text.
    payload = json.dumps({k: meta.get(k) for k in metadata_mod.FIELDS
                          + ("duration_s", "width", "height")},
                         sort_keys=True, ensure_ascii=False)
    quality_payload = payload

    record = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "url": url,
        "video_id": meta.get("video_id"),
        "title": meta.get("title"),
        "duration_s": meta.get("duration_s"),
        "resolution": [meta.get("width"), meta.get("height")],
        "subtitle_chars": len(meta.get("subtitle") or ""),
        "subtitle_source": meta.get("subtitle_source"),
        "subtitle_language": meta.get("subtitle_language"),
        "subtitle_kind": meta.get("subtitle_kind"),
        "metadata": {k: meta.get(k) for k in metadata_mod.FIELDS},
    }

    # ---- Prompt 1 ----------------------------------------------------------
    log.progress(1, 2, "Prompt 1 -- is this GUI operation content?")
    try:
        result = client.chat(model, prompts.meta_request(meta),
                             cache_key=_cache_key("meta", payload, model))
        record["meta_verdict"] = parse_meta_verdict(result["text"])
        v = record["meta_verdict"]
        log.ok("is_gui_content=%s confidence=%.2f  %s"
               % (v["is_gui_content"], v["confidence"], v["reasoning"][:70]))
    except (PermanentError, ValueError) as exc:
        record["meta_verdict"] = None
        record["meta_error"] = str(exc)[:300]
        log.error("Prompt 1 failed: %s" % str(exc)[:160])

    # ---- Prompt 2 ----------------------------------------------------------
    # Skipped when Prompt 1 already rejected, because the fine filter refines a
    # candidate rather than reviving one, and the paper's order is coarse first.
    #
    # Also skipped when the video is over the duration limit: that criterion is
    # already decided from metadata, so paying for a video upload to score a
    # candidate the paper would not accept would be waste.
    duration = record.get("duration_s")
    too_long = duration is not None and duration >= config.MAX_DURATION_S
    if too_long:
        log.warn("%.1f min is over the paper's %d-minute limit, so Prompt 2 is "
                 "skipped and nothing more is spent on this candidate."
                 % (duration / 60.0, config.MAX_DURATION_S // 60))

    if (record.get("meta_verdict") and record["meta_verdict"]["is_gui_content"]
            and not too_long):
        log.progress(2, 2, "Prompt 2 -- quality scoring, three dimensions")

        # The paper scores the FIRST MINUTE of video, not text:
        #   "we leverage an omnimodal model ... we extract the first minute and
        #    evaluate it along three complementary dimensions."
        # This is load-bearing. Run on metadata alone, the model gave
        # recording_quality a 4 while its own reasoning said "the visual file
        # itself isn't viewable" -- a passing score on a dimension it could not
        # see. So a missing clip is recorded as a failure of that dimension
        # rather than silently scored from a title.
        media = []
        probe_path = None
        try:
            log.detail("downloading the first %d s for the omnimodal scorer"
                       % metadata_mod.PROBE_SECONDS)
            probe_path = metadata_mod.fetch_probe_clip(
                url, dest_dir=paths.screen / "probes")
            size_mb = probe_path.stat().st_size / 1e6
            log.detail("probe clip: %s (%.1f MB)" % (probe_path.name, size_mb))
            record["probe_clip"] = probe_path.name
            record["probe_mb"] = round(size_mb, 2)
            media = [client.media(probe_path, "video")]
            # Prompt 2's answer depends on the video, not only on the text, so the
            # clip's bytes belong in its cache key. Without this, re-running after
            # a different clip was downloaded would replay the old verdict.
            quality_payload = payload + hashing.file_digest(probe_path)
        except Exception as exc:
            record["probe_error"] = str(exc)[:300]
            log.warn("could not obtain the first minute, so recording_quality "
                     "cannot be judged: %s" % str(exc)[:140])

        try:
            result = client.chat(model, prompts.quality_request(meta, media),
                                 cache_key=_cache_key("quality",
                                                      quality_payload, model))
            record["quality"] = parse_quality_scores(result["text"])
            sc = record["quality"]["scores"]
            log.ok("  ".join("%s=%d" % (d, sc.get(d, 0)) for d in DIMENSIONS))
            log.detail(record["quality"]["overall_summary"][:150])
            if not media:
                # Not a hard failure of the whole screening, but the score is not
                # trustworthy, so say so and mark it.
                record["quality"]["blind_dimensions"] = ["recording_quality"]
                log.warn("recording_quality was judged without the video, so it "
                         "is a guess and is marked as such")
        except (PermanentError, ValueError) as exc:
            record["quality"] = None
            record["quality_error"] = str(exc)[:300]
            log.error("Prompt 2 failed: %s" % str(exc)[:160])
    else:
        record["quality"] = None
        log.info("Prompt 2 skipped: Prompt 1 did not pass")

    # ---- decision ----------------------------------------------------------
    if record.get("meta_verdict") and record.get("quality"):
        accepted, reasons = verdict(record["meta_verdict"], record["quality"],
                                    duration)
    elif too_long and record.get("meta_verdict"):
        # Prompt 2 never ran, so the verdict is computed from what IS known. The
        # duration line is still produced, which is the reason that matters here.
        accepted, reasons = verdict(
            record["meta_verdict"],
            {"scores": {d: config.MIN_QUALITY_SCORE for d in DIMENSIONS}},
            duration)
        reasons = [r for r in reasons if not r.startswith("Prompt 2")]
        reasons.append("Prompt 2: not run, the video is over the limit")
    else:
        accepted, reasons = False, ["screening incomplete"]
    record["accepted"] = accepted
    record["reasons"] = reasons
    # Recorded so the decision can be audited later against the rule that was in
    # force at the time, rather than against today's constant.
    record["duration_limit_s"] = config.MAX_DURATION_S
    record["too_long"] = too_long

    records.write_json(record, out_path)
    if accepted:
        log.ok("ACCEPTED -- %s" % stem)
    else:
        log.warn("REJECTED -- %s: %s" % (stem, "; ".join(reasons)))
    for line in reasons:
        log.detail("%s" % line)
    return record


def _record_name(url: str, meta: dict) -> str:
    """A stable filename for a screening record.

    The video id when available, because it is stable across URL spellings and
    playlists, and a hash otherwise so a URL with odd characters cannot produce an
    unusable filename.
    """
    vid = meta.get("video_id")
    if vid:
        return str(vid)
    # `hashing` exposes `fingerprint` and `file_digest`; there is no `prompt_hash`.
    # This branch only runs when a video has no id, which is why the NameError sat
    # here unnoticed -- it is reachable only for unusual URLs.
    return hashing.fingerprint(url)[:12]


def screen(urls: list, client, paths: config.Paths, **kwargs) -> list:
    """Screen several candidates. One failure does not stop the batch."""
    paths.ensure()
    out = []
    for i, url in enumerate(urls, 1):
        log.step("screening %d/%d: %s" % (i, len(urls), url))
        try:
            out.append(screen_one(url, client, paths, **kwargs))
        except metadata_mod.MetadataError as exc:
            log.error("metadata failed: %s" % str(exc)[:200])
        except Exception as exc:
            log.error("%s: %s" % (type(exc).__name__, str(exc)[:200]))
    return out
