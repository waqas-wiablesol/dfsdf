"""Video handling: probe, clip, downscale, frame extraction.

Uses ffmpeg. Rather than requiring it on PATH, this falls back to the static
binary that `imageio-ffmpeg` ships, which means the pipeline works on a clean
machine with nothing but `pip install -r requirements.txt`.

This module deals in files only. Getting a file onto the wire is the client's job:
the Gemini path reads the bytes into a typed `Part`, so no data-URI encoding step
exists anywhere in the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


# ============================================================================
# ffmpeg discovery
# ============================================================================

_FFMPEG: str | None = None
_FFPROBE: str | None = None


def ffmpeg_path() -> str:
    """Path to an ffmpeg executable, or raise with install instructions."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG

    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return found

    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        pass

    raise SystemExit(
        "ffmpeg not found.\n"
        "  pip install imageio-ffmpeg   (bundles a static binary, easiest)\n"
        "  or install ffmpeg system-wide and put it on PATH"
    )


def ffprobe_path() -> str | None:
    """ffprobe if available.

    `imageio-ffmpeg` ships ffmpeg without ffprobe, so probing may have to fall
    back to parsing `ffmpeg -i` output. That is why this can return None.
    """
    global _FFPROBE
    if _FFPROBE:
        return _FFPROBE
    found = shutil.which("ffprobe")
    if found:
        _FFPROBE = found
        return found
    # Some installs ship ffprobe next to the bundled ffmpeg.
    ff = Path(ffmpeg_path())
    cand = ff.with_name("ffprobe" + (ff.suffix or ""))
    if cand.exists():
        _FFPROBE = str(cand)
        return _FFPROBE
    return None


def _run(cmd: list, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def kill_tree(proc) -> None:
    """Stop a process and everything it started.

    `proc.kill()` signals only the direct child, so an ffmpeg that yt-dlp spawned
    survives and keeps writing into the output directory. On Windows `taskkill /T`
    walks the tree; elsewhere the process is signalled directly.

    The direct child's liveness is deliberately NOT checked first. A venv launcher
    can exit while the interpreter it started keeps running, so "the child is gone"
    does not mean "the tree is gone" -- and skipping the kill in that case would
    orphan exactly the process that matters. taskkill on a dead PID is harmless.

    Best effort by design: this runs on a failure path, and raising here would
    replace a useful timeout message with a less useful one.
    """
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=20)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _parse_out_time(text: str) -> float | None:
    """Seconds from ffmpeg's `out_time=HH:MM:SS.microseconds`.

    Parsed from the formatted string rather than from `out_time_us` or
    `out_time_ms`, because those two report the SAME number in microseconds:
    `out_time_ms` is a long-standing ffmpeg misnomer. Verified against ffmpeg 7.1,
    where a 2.95 s position prints `out_time_us=2950000` and
    `out_time_ms=2950000`. Trusting the name over the value would be a 1000x
    error in exactly one of the two.
    """
    parts = text.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None


def _run_with_progress(cmd: list, total_seconds: float, on_progress,
                       timeout: int = 1800) -> tuple[int, str]:
    """Run ffmpeg, reporting how far through the OUTPUT it has got.

    ffmpeg cannot know how long an encode will take, but it does know the
    timestamp it has reached in the output. Against the requested duration that is
    a real percentage rather than a guess or a spinner.

    Returns (returncode, stderr).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    deadline = time.time() + timeout
    try:
        for raw in proc.stdout:
            if time.time() > deadline:
                proc.kill()
                break
            line = raw.strip()
            if line.startswith("out_time=") and total_seconds > 0:
                done = _parse_out_time(line.split("=", 1)[1])
                if done is not None:
                    pct = max(0.0, min(100.0, 100.0 * done / total_seconds))
                    try:
                        on_progress(pct)
                    except Exception:
                        # A reporting failure must never fail the encode.
                        pass
    finally:
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.stdout.close()
        proc.stderr.close()
        code = proc.wait()
    return code, stderr


# ============================================================================
# Probing
# ============================================================================


@dataclass
class VideoInfo:
    path: Path
    duration: float          # seconds
    width: int
    height: int
    fps: float

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    def summary(self) -> str:
        return ("%s  %.1fs  %dx%d  %.2f fps"
                % (self.path.name, self.duration, self.width, self.height,
                   self.fps))


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_VID_RE = re.compile(r"Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+\.?\d*)\s*fps")


def probe(path: Path) -> VideoInfo:
    """Duration, resolution and fps for a video.

    Prefers ffprobe's JSON output. When ffprobe is missing -- which is the case
    with the bundled binary -- parses `ffmpeg -i`, which writes its banner to
    stderr and exits non-zero because no output file was given. That non-zero
    exit is expected here and must not be treated as failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    fp = ffprobe_path()
    if fp:
        r = _run([fp, "-v", "quiet", "-print_format", "json",
                  "-show_format", "-show_streams", str(path)], timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            try:
                d = json.loads(r.stdout)
                streams = d.get("streams") or []
                v = next((s for s in streams
                          if s.get("codec_type") == "video"), None)
                if v:
                    dur = float(d.get("format", {}).get("duration")
                                or v.get("duration") or 0.0)
                    num, den = v.get("avg_frame_rate", "0/1").split("/")
                    fps = float(num) / float(den) if float(den) else 0.0
                    return VideoInfo(path, dur, int(v["width"]),
                                     int(v["height"]), fps)
            except Exception:
                pass  # fall through to the ffmpeg banner

    r = _run([ffmpeg_path(), "-hide_banner", "-i", str(path)], timeout=120)
    banner = (r.stderr or "") + (r.stdout or "")

    dur = 0.0
    m = _DUR_RE.search(banner)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    w = h = 0
    m = _VID_RE.search(banner)
    if m:
        w, h = int(m.group(1)), int(m.group(2))

    fps = 0.0
    m = _FPS_RE.search(banner)
    if m:
        fps = float(m.group(1))

    if not (w and h):
        raise RuntimeError(
            "could not read video dimensions from %s. Is this a real video "
            "file?\n--- ffmpeg said ---\n%s" % (path.name, banner[-1200:]))

    return VideoInfo(path, dur, w, h, fps)


# ============================================================================
# Clipping and downscaling
# ============================================================================


def plan_clips(info: VideoInfo,
               clip_seconds: int | None = None) -> list[tuple[float, float]]:
    """[(start, end)] covering the whole video in <= clip_seconds pieces.

    A video shorter than the limit yields one clip; a long one is split on exact
    boundaries. The paper carries annotation history across boundaries so tasks
    that span a cut still come out whole.
    """
    clip_seconds = clip_seconds or config.CLIP_SECONDS
    if info.duration <= 0:
        return [(0.0, clip_seconds)]

    spans = []
    t = 0.0
    while t < info.duration - 0.5:
        spans.append((t, min(t + clip_seconds, info.duration)))
        t += clip_seconds
    return spans or [(0.0, info.duration)]


def target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Output dimensions for a clip: fit inside long_side, never upscale.

    Computed here rather than in the ffmpeg filter, because the filter
    expression needs `min()`, whose comma the filtergraph parser reads as a
    filter separator unless it is quoted, and quoting rules differ between
    ffmpeg builds. Arithmetic in Python has no such ambiguity.

    Both dimensions are forced even: yuv420p with libx264 requires it, and an
    odd height is silently rejected rather than corrected.
    """
    longest = max(width, height)
    if longest <= 0 or longest <= long_side:
        w, h = width, height
    elif width >= height:
        w, h = long_side, max(2, round(height * long_side / width))
    else:
        w, h = max(2, round(width * long_side / height)), long_side
    return max(2, w - (w % 2)), max(2, h - (h % 2))


def make_clip(src: Path, start: float, end: float, dest: Path,
              long_side: int | None = None, crf: int | None = None,
              size: tuple[int, int] | None = None,
              progress=None) -> Path:
    """Cut [start, end) and downscale to fit `long_side`.

    Downscaling matters: the paper's 15,908 tokens for a 4-minute segment imply
    roughly 66 tokens/second, i.e. an already-compressed representation. Sending
    full 1080p would cost several times more for no gain in *temporal* reasoning,
    because precision is recovered later from full-resolution frames.

    `size` is accepted so a caller that has already probed the source does not
    pay for a second probe.

    `progress` is called with a 0-100 float as the encode advances. It is optional
    because cutting is the one slow local step, and a caller that does not want to
    report it should not pay for the extra pipe.
    """
    long_side = long_side or config.VIDEO_LONG_SIDE
    crf = config.VIDEO_CRF if crf is None else crf

    if size is None:
        info = probe(src)
        size = target_size(info.width, info.height, long_side)
    w, h = size
    vf = "scale=%d:%d" % (w, h)

    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"]
    if progress is not None:
        # `-progress pipe:1` writes key=value lines to stdout, so stderr is left
        # carrying only errors and the two streams cannot be confused.
        cmd += ["-nostats", "-progress", "pipe:1"]
    cmd += [
        "-ss", "%.3f" % start,
        "-i", str(src),
        "-t", "%.3f" % (end - start),
        "-vf", vf,
        "-c:v", "libx264", "-preset", config.VIDEO_PRESET, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(dest),
    ]
    if progress is None:
        r = _run(cmd)
        code, err = r.returncode, (r.stderr or "")
    else:
        code, err = _run_with_progress(cmd, end - start, progress)
    if code != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("clipping failed for %s [%.1f-%.1f]:\n%s"
                           % (src.name, start, end, err[-800:]))
    return dest


# ============================================================================
# Frame extraction
# ============================================================================


def extract_frame(src: Path, t: float, dest: Path) -> Path:
    """One frame at time t, at the video's native resolution.

    PNG, full resolution, no scaling: this is the stage where pixel precision is
    the point. Seeks before -i for speed, which on a GOP-aligned seek can land
    slightly off; the offsets we request are half a second apart, far wider than
    that error, so it does not matter here.
    """
    t = max(0.0, t)
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "%.3f" % t,
        "-i", str(src),
        "-frames:v", "1",
        "-pix_fmt", "rgb24",
        str(dest),
    ]
    r = _run(cmd, timeout=300)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("frame extraction failed at t=%.2f for %s:\n%s"
                           % (t, src.name, (r.stderr or "")[-600:]))
    return dest


def extract_frames_at(src: Path, t: float, offsets, dest_dir: Path,
                      stem: str) -> list[tuple[float, Path]]:
    """Frames at t+offset for each offset.

    Returns [(actual_time, path)]. A frame that fails to extract is skipped
    rather than aborting: the paper's rule is "first frame that grounds wins",
    so losing one candidate narrows the choice instead of the run.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, off in enumerate(offsets):
        when = max(0.0, t + off)
        dest = dest_dir / ("%s_f%d.%s" % (stem, i, config.FRAME_FORMAT))
        try:
            extract_frame(src, when, dest)
            out.append((when, dest))
        except Exception as exc:
            print("        frame t=%.2f failed: %s" % (when, str(exc)[:90]))
    return out


def frame_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image.

    PIL is imported inside the function so that importing this module, and
    therefore running `cli check` or `cli plan`, does not require it.
    """
    from PIL import Image
    with Image.open(path) as handle:
        return handle.size


def oversize_frame_warning(info: "VideoInfo", limit: int | None = None):
    """A warning string when a source is large enough to yield oversized frames.

    Returns None when the source is fine, which is the usual case.

    This is deliberately a warning and not a resize. Grounding frames are stored
    at the source's native resolution because a frame's pixels *are* the
    coordinate space -- `target_x` / `target_y` are pixels in that exact image.
    Rescaling a frame would silently invalidate every coordinate measured against
    it, so an outsized source is reported and left alone, and the decision stays
    with a human.

    The practical ceiling is about the provider's request size: three PNGs per
    action, each several megabytes at 4K and above.
    """
    limit = config.MAX_FRAME_LONG_SIDE if limit is None else limit
    if not limit or info.long_side <= limit:
        return None
    return ("%s is %dx%d, above MAX_FRAME_LONG_SIDE=%d. Grounding frames will "
            "be stored at that native size, so three PNGs per action may "
            "approach the provider's request limit. Frames are NOT resized -- "
            "raise or disable MAX_FRAME_LONG_SIDE to silence this."
            % (info.path.name, info.width, info.height, limit))


# Encoding to data URIs used to live here, then moved to a `payloads` module. Both
# are gone: the Gemini client reads a path into a typed `Part` directly, so there is
# no encoding step to own.
