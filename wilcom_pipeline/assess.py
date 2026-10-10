"""Assess whether a recording is a good candidate for the pipeline. No API calls.

The first video processed was a "Basic Navigation" tour: it produced one task and
15 exploratory actions ("click the File menu", "click the Edit menu"), which are
individually low-value and not goal-directed. That is a property of the *source
video*, not of the pipeline, and it is cheap to detect before spending anything.

This module answers the question from the pixels, using ffmpeg only:

  * **Activity density.** Frames are sampled at 1 Hz, downscaled, and compared
    consecutively. A recording of someone *doing work* changes the screen
    constantly; a tour where a menu is opened and left open is static for long
    stretches. The share of sampled seconds that are essentially unchanged is a
    good proxy for how much of the runtime carries usable actions.
  * **Dead time.** The longest run of unchanged frames. Tutorials often open with
    a logo, a title card, or a spoken introduction over a still screen. None of
    that produces a row and all of it costs tokens.
  * **Screen-state variety.** How many *distinct* screen states appear. Twenty
    clicks that each change something look different from twenty clicks that
    change nothing.
  * **Clip plan and payload**, so the cost is known before committing.

What it deliberately does NOT do is judge content quality. Whether the actions are
goal-directed, whether the cursor is visible, whether a dialog is legible — those
need eyes. It prints a contact sheet for that.

Calibrate against a known-bad example: the basic-navigation tour measured
`still_ratio` around 0.5 with a long opening still. A hands-on workflow should sit
well below that.
"""

from __future__ import annotations

from pathlib import Path

from . import config, log, video


# ============================================================================
# Frame sampling
# ============================================================================


def sample_frames(src: Path, out_dir: Path, rate: float = 1.0,
                  working_width: int = 320) -> list:
    """Extract one small grayscale frame per `1/rate` seconds.

    Small and grayscale because these exist only to be *differenced*. Full colour
    frames at 1080p would be ~300 MB for a five-minute video and the comparison
    would be no more informative.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    pattern = out_dir / "s%05d.png"
    cmd = [
        video.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", "fps=%s,scale=%d:-2,format=gray" % (rate, working_width),
        str(pattern),
    ]
    result = video._run(cmd, timeout=900)
    if result.returncode != 0:
        raise RuntimeError("sampling failed:\n%s" % (result.stderr or "")[-600:])
    return sorted(out_dir.glob("s*.png"))


# ============================================================================
# Activity analysis
# ============================================================================


def _mean_abs_diff(a: Path, b: Path) -> float:
    """Mean absolute pixel difference, 0-255, between two same-size images."""
    from PIL import Image, ImageChops, ImageStat
    with Image.open(a) as ia, Image.open(b) as ib:
        return ImageStat.Stat(ImageChops.difference(ia, ib)).mean[0]


def analyse_activity(frames: list, rate: float = 1.0,
                     still_threshold: float = 1.0) -> dict:
    """Difference every consecutive pair of sampled frames.

    `still_threshold` is deliberately low (1/255). Any real UI change -- a menu
    opening, a panel toggling, a tool switching -- moves far more than that even
    after a 320px downscale, so a low bar means "unchanged" really means
    unchanged rather than "changed a little".
    """
    if len(frames) < 2:
        return {"samples": len(frames), "diffs": [], "still_ratio": 1.0,
                "longest_still_s": 0.0, "distinct_states": 0}

    diffs = []
    for a, b in zip(frames, frames[1:]):
        try:
            diffs.append(_mean_abs_diff(a, b))
        except Exception:
            diffs.append(0.0)

    still = [d < still_threshold for d in diffs]
    still_ratio = sum(still) / len(still)

    # Longest consecutive run of unchanged frames, in seconds.
    longest = run = 0
    for is_still in still:
        run = run + 1 if is_still else 0
        longest = max(longest, run)

    # Distinct screen states: count transitions into a changed state, which is
    # closer to "how many separate things happened" than the raw frame count.
    distinct = sum(1 for i, d in enumerate(diffs)
                   if d >= still_threshold
                   and (i == 0 or diffs[i - 1] < still_threshold))

    changed = [d for d in diffs if d >= still_threshold]
    return {
        "samples": len(frames),
        "diffs": diffs,
        "still_ratio": round(still_ratio, 4),
        "longest_still_s": round(longest / rate, 1),
        "active_seconds": round(len(changed) / rate, 1),
        "distinct_states": distinct,
        "mean_change": round(sum(changed) / len(changed), 2) if changed else 0.0,
        "max_change": round(max(diffs), 2) if diffs else 0.0,
    }


# ============================================================================
# Verdict
# ============================================================================


def verdict(info, activity: dict) -> tuple[str, list]:
    """A crude classification, with the reasons behind it.

    Thresholds are calibrated on one known-bad example, so this is a filter for
    *obviously* unsuitable sources, not an oracle. The reasons are returned so the
    judgement can be argued with rather than just trusted.
    """
    reasons = []
    score = 0

    ratio = activity["still_ratio"]
    if ratio <= 0.25:
        reasons.append("activity: only %.0f%% of sampled seconds are unchanged"
                       % (100 * ratio))
        score += 2
    elif ratio <= 0.45:
        reasons.append("activity: %.0f%% unchanged, a moderate amount of dead time"
                       % (100 * ratio))
        score += 1
    else:
        reasons.append("activity: %.0f%% unchanged, so much of the runtime "
                       "carries no action" % (100 * ratio))
        score -= 2

    if activity["longest_still_s"] <= 8:
        reasons.append("dead time: longest still stretch %.0fs"
                       % activity["longest_still_s"])
        score += 1
    elif activity["longest_still_s"] <= 25:
        reasons.append("dead time: longest still stretch %.0fs, likely an intro"
                       % activity["longest_still_s"])
    else:
        reasons.append("dead time: %.0fs with no screen change at all"
                       % activity["longest_still_s"])
        score -= 2

    if activity["distinct_states"] >= 40:
        reasons.append("variety: %d distinct screen states" % activity["distinct_states"])
        score += 1
    elif activity["distinct_states"] >= 15:
        reasons.append("variety: %d distinct screen states" % activity["distinct_states"])
    else:
        reasons.append("variety: only %d distinct screen states" % activity["distinct_states"])
        score -= 1

    if info.long_side < 1080:
        reasons.append("legibility: %dpx long side, below 1080p"
                       % info.long_side)
        score -= 1
    else:
        reasons.append("legibility: %dpx long side, comfortably readable"
                       % info.long_side)

    if score >= 3:
        return "STRONG", reasons
    if score >= 1:
        return "USABLE", reasons
    if score >= -1:
        return "WEAK", reasons
    return "POOR", reasons


# ============================================================================
# Contact sheet
# ============================================================================


def contact_sheet(src: Path, out_path: Path, count: int = 24,
                  tile_width: int = 480, cols: int = 3) -> Path:
    """A grid of evenly spaced full-resolution frames, for a human to read.

    This is the part the statistics cannot replace. Frame differences say *how
    much* changed; only looking says *what* changed and whether the resulting
    action would make a useful training row.
    """
    from PIL import Image, ImageDraw

    total = max(1, count)
    duration = video.probe(src).duration
    stamps = [duration * (i + 0.5) / total for i in range(total)]

    thumbs = []
    for t in stamps:
        tmp = out_path.parent / ("_cs_%06.1f.png" % t)
        try:
            video.extract_frame(src, t, tmp)
        except Exception:
            continue
        with Image.open(tmp) as im:
            im = im.convert("RGB")
            ratio = tile_width / im.width
            thumbs.append((t, im.resize((tile_width, int(im.height * ratio)),
                                        Image.LANCZOS)))
        try:
            tmp.unlink()
        except OSError:
            pass

    if not thumbs:
        raise RuntimeError("no frames could be extracted")

    tw, th = thumbs[0][1].size
    band = 20
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, (th + band) * rows), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(thumbs):
        x = (i % cols) * tw
        y = (i // cols) * (th + band)
        sheet.paste(im, (x, y))
        draw.rectangle([x, y + th, x + tw, y + th + band], fill=(28, 28, 32))
        draw.text((x + 6, y + th + 4), "t=%5.1fs" % t, fill=(210, 210, 210))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ============================================================================
# Entry point
# ============================================================================


def assess(src: Path, paths: config.Paths, samples: int = 24,
           sheet: bool = True) -> dict:
    """Full assessment of one file. Writes artefacts under <work>/logs."""
    src = Path(src)
    info = video.probe(src)
    spans = video.plan_clips(info)
    size = video.target_size(info.width, info.height, config.VIDEO_LONG_SIDE)

    log.step("assessing %s" % src.name)
    log.detail("%s" % info.summary())

    # Activity, from 1 Hz grayscale samples.
    frames = sample_frames(src, paths.root / "assess" / src.stem / "samples")
    activity = analyse_activity(frames)
    log.detail("sampled %d frame(s) at 1 Hz" % activity["samples"])

    # Payload, so cost is known before anything is sent.
    payload_mb = 0.0
    if spans:
        probe_clip = paths.root / "assess" / src.stem / "clip_probe.mp4"
        video.make_clip(src, spans[0][0], min(spans[0][1], spans[0][0] + 60),
                        probe_clip, size=size)
        per_second = probe_clip.stat().st_size / 60.0
        payload_mb = per_second * (spans[0][1] - spans[0][0]) * 4 / 3 / 1e6

    label, reasons = verdict(info, activity)

    log.info("")
    log.table([
        ["duration", "%.0f s (%.1f min)" % (info.duration, info.duration / 60)],
        ["resolution", "%dx%d @ %.0f fps" % (info.width, info.height, info.fps)],
        ["clips at %ds" % config.CLIP_SECONDS, "%d" % len(spans)],
        ["clip encode", "%dx%d" % size],
        ["~clip 0 payload", "%.1f MB base64" % payload_mb],
        ["", ""],
        ["sampled seconds", "%d" % activity["samples"]],
        ["unchanged seconds", "%.0f%%" % (100 * activity["still_ratio"])],
        ["active seconds", "%.0f" % activity["active_seconds"]],
        ["mean change when active", "%.1f / 255" % activity["mean_change"]],
        ["distinct screen states", "%d" % activity["distinct_states"]],
        ["longest still stretch", "%.0f s" % activity["longest_still_s"]],
    ], ["metric", "value"])

    log.info("")
    log.info("VERDICT: %s" % label)
    for reason in reasons:
        log.detail("- %s" % reason)

    sheet_path = None
    if sheet:
        sheet_path = contact_sheet(src, paths.logs / ("candidate_%s.png" % src.stem))
        log.info("")
        log.info("contact sheet: %s" % sheet_path)
        log.info("The statistics say how much changed. Only this says WHAT "
                 "changed, and whether it would make a useful row.")

    return {"file": str(src), "info": info, "spans": spans,
            "activity": activity, "verdict": label, "reasons": reasons,
            "payload_mb": round(payload_mb, 2),
            "sheet": str(sheet_path) if sheet_path else None}
