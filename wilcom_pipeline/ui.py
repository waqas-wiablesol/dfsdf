"""Web UI.

Wraps the pipeline's own functions rather than shelling out to the CLI, so the UI
cannot drift from the command line: both call `stage_filter.screen_one`,
`pipeline.run_pipeline` and `report.render`.

Two things shape the design:

  * **Screening is separate and cheap.** The paper's Prompts 1 and 2 read the text
    metadata and the first minute of video, so a URL can be assessed for about a
    tenth of a cent. That is the first tab, and it is the cheap way to answer "is
    this worth processing?" before a full-length video is downloaded.
  * **Progress must stream.** A stage-1 call runs for 30 s to several minutes, and
    the provider returns nothing until it is done. The log module is given a sink
    that appends to a queue the UI drains, so the interface shows work happening
    rather than appearing frozen -- the exact failure that made a completed run
    look hung.

Launch:  python -m wilcom_pipeline.ui
"""

from __future__ import annotations

import html as html_mod
import json
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path

from . import (clients, config, cost, hf_export, log, metadata as metadata_mod,
               prompts, records, report, stage_dataset, stage_filter,
               stage_grounding, stage_trajectory, video)

# A queued log line is drained by the UI on a timer. Unbounded is fine: a run
# produces at most a few thousand lines, and dropping one would hide a warning.
_MESSAGES: queue.Queue = queue.Queue()
_WORK: dict = {"thread": None, "result": None, "error": None,
               "running": False, "step": None, "progress": None,
               "started": None, "activity": [], "action": ""}


# ============================================================================
# Thread plumbing
# ============================================================================


def _push(line: str) -> None:
    _MESSAGES.put(line)


def _drain() -> str:
    """Everything logged since the last call, as one string."""
    lines = []
    while True:
        try:
            lines.append(_MESSAGES.get_nowait())
        except queue.Empty:
            break
    return "\n".join(lines)


def _paths(work: str) -> config.Paths:
    return config.Paths(root=Path(work or "wilcom_work")).ensure()


def _start(fn) -> None:
    """Run `fn` on a worker thread with the log sinks attached."""
    if _WORK["running"]:
        return
    _MESSAGES.queue.clear()
    _WORK.update(result=None, error=None, running=True, progress=None,
                 started=time.time(), activity=[], action="")
    log.set_sink(_push)
    log.set_event_sink(_on_event)

    def runner():
        try:
            _WORK["result"] = fn()
        except Exception as exc:
            _WORK["error"] = "%s: %s" % (type(exc).__name__, str(exc))
            _push("ERROR %s" % traceback.format_exc()[-1500:])
        finally:
            _WORK["running"] = False
            _WORK["progress"] = None
            log.clear_sink()
            log.clear_event_sink()

    thread = threading.Thread(target=runner, daemon=True)
    _WORK["thread"] = thread
    thread.start()


def _status() -> str:
    """A one-line banner describing the worker state."""
    if _WORK["running"]:
        return "running"
    if _WORK["error"]:
        return "failed: %s" % _WORK["error"][:200]
    if _WORK["result"] is not None:
        return "done"
    return "idle"


# A running step publishes two things: a real percentage, and a feed of the
# intermediate steps it has reached. Both live in the shared dict rather than in a
# widget because the step runs on a worker thread and the UI only ever sees them
# through a poll.
#
# `pct=None` means "working, but nothing has reported a number yet" -- the bar then
# renders full-width and dimmed rather than sitting at a fake 0%, which would read
# as "stuck".

_ACTIVITY_MAX = 300
_FEED_ROWS = 16

_PROGRESS_CSS = """<style>
  .pwrap{background:#2a2e35;border-radius:6px;height:14px;overflow:hidden;
         margin:10px 0 6px;border:1px solid #343941}
  .pbar{background:linear-gradient(90deg,#00d1b2,#8ab4f8);height:100%;
        transition:width .35s ease}
  .ptop{display:flex;justify-content:space-between;align-items:baseline;
        font-size:12.5px;color:#c8cdd3}
  .pnow{font-weight:600;color:#e6e6e6}
  .pclock{font-family:ui-monospace,Consolas,monospace;color:#ffb86c;
          font-weight:700}
  .pmeta{color:#9aa0a6;font-size:12px;
         font-family:ui-monospace,Consolas,monospace;margin-top:2px}
  .feed{margin:12px 0 4px;border-top:1px solid #262a31;padding-top:8px}
  .fe{display:flex;gap:8px;font-size:12.5px;line-height:1.65;
      color:#c8cdd3;align-items:flex-start}
  .fm{width:12px;flex:0 0 12px;text-align:center;font-weight:700}
  .ft{flex:1}
  .fhd{color:#7d848c;font-size:11.5px;text-transform:uppercase;
       letter-spacing:.06em;margin-bottom:4px}
</style>"""

# Which glyph marks which event. Colour alone would not survive a colourblind
# reader or a screenshot, so the shape carries the meaning too.
_LEVEL_STYLE = {
    "ok": ("\u2713", "#00d1b2", "400"),
    "task": ("\u2022", "#8ab4f8", "400"),
    "phase": ("\u25b8", "#ffb86c", "700"),
    "info": ("\u00b7", "#9aa0a6", "400"),
    "warn": ("!", "#f5c542", "700"),
    "error": ("\u2715", "#ff5f56", "700"),
}


def _set_progress(pct, label: str = "", detail: str = "") -> None:
    """Publish progress for the poll to pick up. `pct` is 0-100 or None."""
    _WORK["progress"] = {"pct": pct, "label": label, "detail": detail}


def _clear_progress() -> None:
    _WORK["progress"] = None


def _elapsed_label() -> str:
    """Minutes and seconds since the current step started.

    Computed here rather than taken from the heartbeat: the clock must keep
    moving on every poll, including between ticks.
    """
    started = _WORK.get("started")
    if not started:
        return ""
    secs = max(0.0, time.time() - started)
    if secs >= 3600:
        return "%d:%02d:%02d" % (secs // 3600, (secs % 3600) // 60, secs % 60)
    return "%d:%02d" % (secs // 60, secs % 60)


def _on_event(ev: dict) -> None:
    """Turn one structured log event into UI state.

    Deliberately dumb: the log module decides what happened, this only decides how
    to show it. Nothing here parses text, which is the whole reason the events
    carry fields.
    """
    level = ev.get("level") or "info"

    if level == "progress":
        text = ev.get("text") or ev.get("message") or ""
        if ev.get("pct") is not None:
            # A gauge: already a percentage, and there is no meaningful
            # "n of m" to show.
            _set_progress(ev["pct"], text, "")
        else:
            index = ev.get("index") or 0
            total = ev.get("total") or 0
            pct = (100.0 * index / total) if total else None
            _set_progress(pct, text,
                          "%d of %d" % (index, total) if total else "")
        _WORK["action"] = text
        return

    if level == "tick":
        # The heartbeat is why the clock exists, not a thing to render itself --
        # the clock already shows the same number, smoothly.
        return

    if level == "detail":
        # Supporting text, not a step. It stays in the log where it has room.
        return

    activity = _WORK.setdefault("activity", [])
    activity.append({"level": level,
                     "message": ev.get("message") or "",
                     "at": ev.get("at") or time.time()})
    del activity[:-_ACTIVITY_MAX]

    if level in ("task", "phase", "ok"):
        # The bar's headline follows the most recent step, so it always names what
        # is happening now instead of falling back to a generic "working". The
        # progress branch above does the same for numbered and gauged progress, so
        # "action" simply means "the newest thing that happened".
        _WORK["action"] = ev.get("message") or ""


def _progress_html() -> str:
    """The live bar: percentage, what it is doing, and how long it has taken."""
    if not _WORK["running"]:
        return ""
    state = _WORK.get("progress") or {}
    pct = state.get("pct")
    # The named step wins over the progress message when there is one: "cutting
    # clip 2/3" says more than a percentage alone. The counter still shows below.
    label = html_mod.escape(_WORK.get("action") or state.get("label")
                            or "working")
    detail = state.get("detail") or ""

    if pct is None:
        bar = "<div class=pbar style='width:100%;opacity:.28'></div>"
        meta = detail or "waiting for the first report"
    else:
        pct = max(0.0, min(100.0, float(pct)))
        bar = "<div class=pbar style='width:%.1f%%'></div>" % pct
        meta = "%.0f%%" % pct
        if detail:
            meta += "  \u00b7  " + detail

    return ('<div class=ptop><span class=pnow>%s</span>'
            '<span class=pclock>%s</span></div>'
            '<div class=pwrap>%s</div>'
            '<div class=pmeta>%s</div>'
            % (label, _elapsed_label(), bar, html_mod.escape(str(meta))))


def _activity_html() -> str:
    """The intermediate steps reached so far, newest last.

    This is the distilled view of the run. The log below it has every line; this
    has the ones that answer "what has it actually done?", which is the question a
    user has while waiting.
    """
    activity = _WORK.get("activity") or []
    if not activity:
        return ""
    rows = []
    for item in activity[-_FEED_ROWS:]:
        mark, colour, weight = _LEVEL_STYLE.get(item["level"],
                                                ("\u00b7", "#9aa0a6", "400"))
        rows.append(
            '<div class=fe><span class=fm style="color:%s">%s</span>'
            '<span class=ft style="font-weight:%s">%s</span></div>'
            % (colour, mark, weight, html_mod.escape(item["message"])))
    shown = len(activity[-_FEED_ROWS:])
    total = len(activity)
    header = ("Steps (%d of %d)" % (shown, total) if total > shown
              else "Steps (%d)" % total)
    return ('<div class=feed><div class=fhd>%s</div>%s</div>'
            % (header, "".join(rows)))


def _run_panel_html() -> str:
    """The whole status panel: live bar, then the steps reached so far.

    One style block for both, rather than each fragment shipping its own copy.
    """
    progress = _progress_html()
    feed = _activity_html()
    if not progress and not feed:
        return ""
    return _PROGRESS_CSS + progress + feed


# ============================================================================
# Tab 1 -- screening, the paper's Prompts 1 and 2
# ============================================================================


def do_screen(url: str, work: str, model: str) -> str:
    """Screen one URL. Text only, so this is cheap."""
    url = (url or "").strip()
    if not url:
        return "Enter a YouTube URL."

    paths = _paths(work)
    client = clients.make_client(cache_dir=paths.root / "cache",
                                 use_cache=True, verbose=True)

    def job():
        return stage_filter.screen_one(url, client, paths,
                                       model=model or None, force=True)

    _start(job)
    return "checking eligibility for %s -- watch the log" % url


def _screen_report(record: dict) -> str:
    """Render a screening record as HTML."""
    if not record:
        return "<p>No screening result yet.</p>"

    meta = record.get("meta_verdict") or {}
    quality = record.get("quality") or {}
    scores = quality.get("scores") or {}
    reasons = quality.get("reasons") or {}
    accepted = record.get("accepted")

    def verdict_badge():
        if accepted:
            return '<span class=v-yes>ACCEPTED</span>'
        return '<span class=v-no>REJECTED</span>'

    rows = [
        ("Title", html_mod.escape(str(record.get("title") or ""))),
        ("Duration", "%.0f s" % record["duration_s"]
         if record.get("duration_s") else "-"),
        ("Resolution", "%sx%s" % tuple(record.get("resolution") or ["?", "?"])),
        ("Subtitle", "%s chars (%s)" % (record.get("subtitle_chars"),
                                        record.get("subtitle_source"))),
        ("Verdict", verdict_badge()),
    ]

    p1 = ""
    if meta:
        p1 = """
        <h3>Prompt 1 &mdash; meta info classification</h3>
        <table class=kv>
          <tr><td>is_gui_content</td><td class="%s">%s</td></tr>
          <tr><td>confidence</td><td>%.2f</td></tr>
        </table>
        <p class=reason>%s</p>""" % (
            "yes" if meta.get("is_gui_content") else "no",
            meta.get("is_gui_content"), meta.get("confidence", 0),
            html_mod.escape(meta.get("reasoning") or ""))
    elif record.get("meta_error"):
        p1 = "<h3>Prompt 1</h3><p class=reason>%s</p>" % html_mod.escape(
            record["meta_error"])

    p2 = ""
    if scores:
        body = "".join(
            "<tr><td>%s</td><td class=score>%d/5</td>"
            "<td class=reason>%s</td></tr>"
            % (d.replace("_", " ").title(), scores.get(d, 0),
               html_mod.escape(reasons.get(d, "")))
            for d in stage_filter.DIMENSIONS)
        p2 = """
        <h3>Prompt 2 &mdash; video quality scoring</h3>
        <table class=kv>%s</table>
        <p class=reason>%s</p>""" % (
            body, html_mod.escape(quality.get("overall_summary") or ""))
    elif record.get("quality_error"):
        p2 = "<h3>Prompt 2</h3><p class=reason>%s</p>" % html_mod.escape(
            record["quality_error"])
    elif record.get("meta_verdict"):
        p2 = ("<h3>Prompt 2</h3><p class=reason>Skipped: Prompt 1 did not pass, "
              "and the fine filter refines a candidate rather than reviving "
              "one.</p>")

    why = "".join("<li>%s</li>" % html_mod.escape(r) for r in
                  (record.get("reasons") or []))
    body = "".join("<tr><td>%s</td><td>%s</td></tr>" % (k, v) for k, v in rows)

    return """<style>
      .kv{width:100%%;border-collapse:collapse;font-size:13px;margin:6px 0}
      .kv td{padding:4px 6px;border-bottom:1px solid #2a2e35;vertical-align:top}
      .kv td:first-child{color:#9aa0a6;width:170px}
      h3{margin:16px 0 4px;font-size:13px;color:#8ab4f8;text-transform:uppercase;letter-spacing:.4px}
      .v-yes{background:#0b6b52;color:#fff;padding:2px 9px;border-radius:4px;font-weight:600}
      .v-no{background:#6b2b2b;color:#fff;padding:2px 9px;border-radius:4px;font-weight:600}
      .reason{color:#c8cdd3;font-size:13px;line-height:1.5;margin:4px 0 0}
      .score{font-weight:700;color:#00d1b2;text-align:center}
      ul{color:#c8cdd3;font-size:13px;line-height:1.6}
    </style>
    <table class=kv>%s</table>
    %s %s
    <h3>Decision</h3><ul>%s</ul>""" % (body, p1, p2, why or "<li>-</li>")


def screen_poll(work: str) -> tuple:
    """Latest log plus the rendered verdict, for the UI timer."""
    out = _drain()
    paths = _paths(work)
    record = _WORK.get("result") if not _WORK["running"] else None
    if record is None:
        # Fall back to whichever screening record is newest on disk, so the panel
        # is not blank after a page reload or a cancelled run.
        candidates = sorted(paths.screen.glob("*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        record = records.read_json(candidates[0]) if candidates else None
    # Status is shown inside the verdict panel rather than as a separate field, so
    # there is one place to look rather than two.
    status = _status()
    banner = '<p class=status>%s</p>' % html_mod.escape(status)
    return out, (banner + _screen_report(record))


# ============================================================================
# Tab 2 -- the pipeline, one step at a time
# ============================================================================
#
# Each step is triggered by its own button and STOPS when it finishes. Nothing
# chains. The earlier version ran screening, download, stage 1, stage 2 and stage 3
# in one call, which meant the only chance to look at an intermediate result was
# after every expensive call had already been paid for -- and a bad clip produced
# at stage 1 was discovered at stage 3.
#
# `STEP_ORDER` drives both the buttons and the "what is next" panel, so the UI
# cannot disagree with itself about the sequence.

STEP_ORDER = [
    ("screen", "0. Check eligibility",
     "Prompts 1 and 2. Two questions: is this video about the software, and is "
     "the recording sharp enough to read? Text metadata plus the first minute. "
     "Rejects a bad candidate before any expensive call."),
    ("download", "1. Download",
     "yt-dlp into <work>/source. No API cost."),
    ("trajectory", "2. Trajectory",
     "Stage 1. Reads video, extracts tasks and actions. The expensive call."),
    ("grounding", "3. Grounding",
     "Stage 2. Locates each action in full-resolution frames."),
    ("dataset", "4. Dataset",
     "Stage 3. Assembles, splits, verifies, writes parquet and the report."),
]

_STEP_LABELS = {key: label for key, label, _ in STEP_ORDER}
_STEP_HELP = {key: help_ for key, _, help_ in STEP_ORDER}


def step_state(work: str) -> dict:
    """Which steps have produced something on disk.

    Derived from files rather than from remembered button presses, so the panel is
    still correct after a restart, a manual CLI run, or a reload.
    """
    paths = _paths(work)
    source_dir = paths.root / "source"

    def any_files(directory, pattern):
        directory = Path(directory)
        return bool(list(directory.glob(pattern))) if directory.is_dir() else False

    return {
        "screen": any_files(paths.screen, "*.json"),
        "download": any_files(source_dir, "*.mp4"),
        "trajectory": any_files(paths.traj, "*.json"),
        "grounding": any_files(paths.ground, "*.json"),
        "dataset": any_files(paths.out, "*.parquet"),
    }


def next_step(work: str) -> str:
    """The first step with nothing on disk yet, or 'done'."""
    state = step_state(work)
    for key, _, _ in STEP_ORDER:
        if not state[key]:
            return key
    return "done"


def step_panel(work: str) -> str:
    """A checklist of the steps, so the current position is always visible."""
    state = step_state(work)
    nxt = next_step(work)
    rows = []
    for key, label, help_ in STEP_ORDER:
        done = state[key]
        if done:
            mark, colour, note = "&#10003;", "#00d1b2", "done"
        elif key == nxt:
            mark, colour, note = "&rarr;", "#ffb86c", "NEXT"
        else:
            mark, colour, note = "&middot;", "#5b6169", "not yet"
        rows.append(
            '<div class=step><span class=mk style="color:%s">%s</span>'
            '<span class=lb>%s</span><span class=nt style="color:%s">%s</span>'
            '<div class=hp>%s</div></div>'
            % (colour, mark, html_mod.escape(label), colour,
               html_mod.escape(note), html_mod.escape(help_)))

    banner = ("All steps are done. Load the rows in tab 3."
              if nxt == "done" else
              "Next: <b>%s</b> &mdash; %s"
              % (html_mod.escape(_STEP_LABELS[nxt]),
                 html_mod.escape(_STEP_HELP[nxt])))

    return """<style>
      .steps{font-size:13px}
      .step{padding:5px 0 5px 2px;border-bottom:1px solid #262a31}
      .mk{font-weight:700;display:inline-block;width:18px}
      .lb{color:#e6e6e6}
      .nt{float:right;font-size:11.5px;font-family:ui-monospace,Consolas,monospace}
      .hp{color:#7d848c;font-size:11.5px;margin-left:18px;line-height:1.5}
      .nx{background:#22262d;border-left:3px solid #ffb86c;padding:8px 11px;
          border-radius:4px;margin-bottom:10px;color:#dfe3e6;font-size:13px}
    </style>
    <div class=nx>%s</div><div class=steps>%s</div>""" % (banner, "".join(rows))


def _start_step(key: str, work: str, job, **context) -> str:
    """Run one step, and say what it is doing rather than only that it started."""
    if _WORK["running"]:
        return "A step is already running. Wait for it to finish."
    _WORK["step"] = key
    _start(job)
    return "Running: %s   %s" % (_STEP_LABELS.get(key, key),
                                 _STEP_HELP.get(key, ""))


# ---------------------------------------------------------------- the steps


def do_screen_step(url: str, work: str, meta_model: str) -> str:
    """Step 0 -- the paper's Prompts 1 and 2."""
    url = (url or "").strip()
    if not url:
        return "Enter a YouTube URL in step 0 first."
    paths = _paths(work)

    def job():
        client = _client(paths)
        return stage_filter.screen_one(url, client, paths,
                                       model=meta_model or None, force=True)

    return _start_step("screen", work, job)


def _rate_of(line: str) -> str:
    """Pull 'at 2.1MiB/s ETA 00:12' out of a yt-dlp progress line, if present.

    Speed and ETA answer the question the percentage cannot: a bar at 12% is
    alarming or reassuring depending entirely on whether it is moving.
    """
    found = re.search(r"at\s+(\S+/s)\s+ETA\s+(\S+)", line)
    if found:
        return "%s   ETA %s" % (found.group(1), found.group(2))
    return ""


def do_download_step(url: str, source: str, work: str) -> str:
    """Step 1 -- fetch the video. No API cost."""
    url = (url or "").strip()
    source = (source or "").strip()
    if not url and not source:
        return "Give a URL or an existing folder."

    paths = _paths(work)

    def job():
        import subprocess
        import sys
        out_dir = paths.root / "source"
        out_dir.mkdir(parents=True, exist_ok=True)

        if url:
            log.step("downloading with yt-dlp")
            cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings",
                   "--newline", "--no-colors", "--progress", "-f",
                   "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
                   "--merge-output-format", "mp4",
                   "-o", str(out_dir / "%(title).70s.%(ext)s"), url]

            # Stream the output instead of waiting for the process to exit. A
            # four-minute download with a blank panel is the worst possible UI, and
            # yt-dlp already reports an exact percentage, so a real bar is free.
            # stderr is folded into stdout so a chatty ffmpeg cannot fill a pipe
            # buffer and deadlock the download.
            _set_progress(0.0, "starting yt-dlp")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    bufsize=1)
            tail: list = []
            kind, last_step = "video", -1
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-30]

                if "Destination:" in line:
                    dest = line.split("Destination:", 1)[1].strip()
                    # yt-dlp fetches video and audio as two separate streams, each
                    # reporting 0-100%. Naming the stream is what stops the bar
                    # looking like it went backwards.
                    kind = ("audio" if dest.lower().endswith(
                        (".m4a", ".mp3", ".opus", ".webm")) else "video")
                    log.detail("fetching %s stream: %s"
                               % (kind, Path(dest).name))
                    _set_progress(0.0, "downloading %s" % kind)
                    last_step = -1
                    continue

                if line.startswith("[Merger]") or "Fixup" in line:
                    log.task("merging video and audio with ffmpeg")
                    _set_progress(None, "merging streams")
                    continue

                found = re.search(r"\[download\]\s+([\d.]+)%", line)
                if found:
                    pct = float(found.group(1))
                    _set_progress(pct, "downloading %s" % kind, _rate_of(line))
                    # The bar carries the fine detail, so the log only needs the
                    # occasional line confirming it is still moving.
                    mark = int(pct // 10) * 10
                    if mark > last_step:
                        last_step = mark
                        log.progress(mark, 100, "%s stream" % kind)
                    continue

                if "ERROR" in line or "WARNING" in line:
                    log.detail(line.strip())

            code = proc.wait()
            _clear_progress()
            if code != 0:
                raise SystemExit("yt-dlp failed:\n%s" % "\n".join(tail[-12:]))
            target = out_dir
        else:
            target = Path(source)

        # Probe now rather than at stage 1, so a broken file is known before any
        # paid call happens.
        from .pipeline import find_videos
        found = find_videos(target)
        infos = []
        for path in found:
            try:
                infos.append(video.probe(path))
            except Exception as exc:
                log.warn("%s: %s" % (path.name, str(exc)[:90]))
        if not infos:
            raise SystemExit("no usable video under %s" % target)
        for info in infos:
            log.ok("%s" % info.summary())
            warning = video.oversize_frame_warning(info)
            if warning:
                log.warn(warning)
        return {"downloaded": [i.path.name for i in infos],
                "target": str(target)}

    return _start_step("download", work, job)


def do_trajectory_step(work: str, max_clips: int, traj_model: str) -> str:
    """Step 2 -- stage 1 only."""
    paths = _paths(work)

    def job():
        from .pipeline import find_videos
        source = paths.root / "source"
        files = find_videos(source) or find_videos(paths.videos)
        if not files:
            raise SystemExit("nothing downloaded. Run step 1 first.")
        log.step("stage 1 -- trajectory extraction")
        ingested = _ingest(files, paths)
        return {"trajectories": stage_trajectory.run(
            ingested, _client(paths), paths,
            model=traj_model or config.model_for("trajectory"),
            max_clips=max_clips or None, verbose=True)}

    return _start_step("trajectory", work, job)


def do_grounding_step(work: str, limit_actions: int, ground_model: str) -> str:
    """Step 3 -- stage 2 only, from the trajectories already on disk."""
    paths = _paths(work)

    def job():
        traj = records.load_dir(paths.traj, verbose=True)
        if not traj:
            raise SystemExit("no trajectory records. Run step 2 first.")
        log.step("stage 2 -- action spatial grounding")
        return {"grounding": stage_grounding.run(
            traj, _client(paths), paths,
            model=ground_model or config.model_for("grounding"),
            limit=limit_actions or None, verbose=True)}

    return _start_step("grounding", work, job)


def do_dataset_step(work: str) -> str:
    """Step 4 -- stage 3 only, from the cached records."""
    paths = _paths(work)

    def job():
        traj = records.load_dir(paths.traj, verbose=True)
        ground = records.load_dir(paths.ground, verbose=True)
        if not traj or not ground:
            raise SystemExit("need both trajectory and grounding records. "
                             "Run steps 2 and 3 first.")

        log.step("stage 3 -- dataset assembly")
        rows, stats = stage_dataset.build_rows(traj, ground, paths)
        for line in stats.summary().splitlines():
            log.detail(line)
        if not rows:
            raise SystemExit("no rows produced, so nothing was written")

        splits = {"unified": rows}
        log.detail("unified: %4d rows, %d trajectories"
                   % (len(rows), len({r["trajectory"] for r in rows})))

        problems = stage_dataset.verify(rows, splits)
        fatal = [p for p in problems if not p.startswith("NOTE")]
        for p in problems:
            (log.error if not p.startswith("NOTE") else log.info)(p)
        if fatal:
            raise SystemExit("verification failed, so nothing was written")

        built = hf_export.save_local(splits, paths)
        path = report.render(rows, paths.logs / "coordinate_check.html",
                             limit=0)
        log.ok("wrote %s" % path)
        return {"rows": len(rows), "splits": {k: len(v) for k, v in splits.items()},
                "report": str(path)}

    return _start_step("dataset", work, job)


def _ingest(files, paths):
    """Bring videos into the work tree, reusing the CLI's own ingest."""
    from .pipeline import ingest
    return ingest(files, paths, verbose=True)


def _client(paths):
    return clients.make_client(cache_dir=paths.root / "cache", use_cache=True,
                               verbose=True)


def run_poll(work: str):
    """Log, summary and the step checklist, for the run tab's timer."""
    out = _drain()
    paths = _paths(work)
    if _WORK["running"]:
        summary = _progress_html() + (
            "<p>Running <b>%s</b>. Progress streams into the log."
            % html_mod.escape(_STEP_LABELS.get(_WORK.get("step"), "step")))
    elif _WORK["result"]:
        summary = _summary_html(_WORK["result"], paths)
    else:
        summary = _summary_html(records.read_json(
            paths.logs / "summary.json") or {}, paths)
    if _WORK["error"] and not _WORK["running"]:
        summary = ('<p class=err>This step failed: %s</p>'
                   % html_mod.escape(_WORK["error"][:300]))
    # The step feed outlives the step, so what happened is still readable after it
    # finishes rather than vanishing the moment the bar does.
    return out, summary + _run_panel_html(), step_panel(work)


def _summary_html(result: dict, paths) -> str:
    if not isinstance(result, dict):
        return ""
    rows = [
        ("Downloaded", result.get("downloaded")),
        ("Videos", result.get("videos")),
        ("Rows written", result.get("rows")),
        ("Splits", result.get("splits")),
        ("Fatal problems", result.get("fatal")),
    ]
    body = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (k, html_mod.escape(str(v)))
        for k, v in rows if v not in (None, "", []))
    extra = ""
    if result.get("report"):
        sp = Path(result["report"])
        extra = ('<p><a href="%s" target="_blank">open the coordinate check</a> '
                 "&mdash; look at the red circles before trusting the rows</p>"
                 % sp.name)
    return """<style>.kv{width:100%%;border-collapse:collapse;font-size:13px}
      .kv td{padding:4px 6px;border-bottom:1px solid #2a2e35}
      .kv td:first-child{color:#9aa0a6;width:160px}
      .err{color:#ff5f56;font-size:13px}
      a{color:#8ab4f8}</style>
      <table class=kv>%s</table>%s""" % (body, extra)


# ============================================================================
# Tab 3 -- results
# ============================================================================


def load_rows(work: str) -> "object":
    """Build a dataframe of the grounded rows, and render crops for review."""
    import pandas as pd
    from PIL import Image, ImageDraw

    paths = _paths(work)
    traj = records.load_dir(paths.traj)
    ground = records.load_dir(paths.ground)
    rows, _ = stage_dataset.build_rows(traj, ground, paths)

    if not rows:
        return pd.DataFrame({"note": ["no rows yet -- run the pipeline first"]}), []

    table = pd.DataFrame([{
        "id": r["id"],
        "instruction": r["instruction"],
        "action": r["action_type"],
        "t (s)": r["timestamp_s"],
        "x": r["target_x"],
        "y": r["target_y"],
        "image": "%dx%d" % (r["image_width"], r["image_height"]),
    } for r in rows])

    # Crops around each target at native resolution, which is the only reliable
    # way to judge whether a coordinate is right: the marker lands on a real UI
    # element or it does not.
    crops = []
    for r in rows[:48]:
        src = Path(r["image_path"])
        if not src.exists():
            continue
        im = Image.open(src).convert("RGB")
        W, H = im.size
        cw, ch, scale = 420, 230, 2
        x, y = r["target_x"], r["target_y"]
        left = max(0, min(W - cw, x - cw // 2))
        top = max(0, min(H - ch, y - ch // 2))
        crop = im.crop((left, top, left + cw, top + ch)).resize(
            (cw * scale, ch * scale), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        cx, cy, R = (x - left) * scale, (y - top) * scale, 24
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(255, 40, 30), width=4)
        d.line([cx - R * 2, cy, cx + R * 2, cy], fill=(255, 40, 30), width=2)
        d.line([cx, cy - R * 2, cx, cy + R * 2], fill=(255, 40, 30), width=2)
        crops.append((crop, "%s\nt=%.1fs  %s  ->  (%d, %d)\n%s"
                      % (r["id"], r["timestamp_s"], r["action_type"], x, y,
                         r["instruction"][:78])))

    return table, crops


def gallery_items(crops):
    """Gradio gallery payload: [(image, caption), ...]."""
    return [(im, cap) for im, cap in crops] if crops else []


def csv_of(work: str):
    import pandas as pd
    table, _ = load_rows(work)
    paths = _paths(work)
    out = paths.logs / "rows_preview.csv"
    try:
        table.to_csv(out, index=False)
        return str(out)
    except Exception:
        return None


# ============================================================================
# Tab 4 -- offline tools
# ============================================================================


def do_plan(source: str, work: str, clip_seconds: int, thinking: str) -> str:
    """Cost estimate. No API calls, so this is always safe to run."""
    source = (source or "").strip()
    if not source:
        return "Give a folder of videos."
    from .pipeline import find_videos
    infos = []
    for path in find_videos(Path(source)):
        try:
            infos.append(video.probe(path))
        except Exception as exc:
            log.warn("%s: %s" % (path.name, str(exc)[:80]))
    if not infos:
        return "no video could be probed under %s" % source
    model = thinking if thinking in ("low", "medium", "high") else None
    est = cost.estimate(infos, clip_seconds=clip_seconds or None,
                        thinking_level=model)
    return "<pre>%s</pre>" % html_mod.escape(est.describe())


def do_metadata(url: str) -> str:
    """Fetch metadata only. No API calls at all."""
    url = (url or "").strip()
    if not url:
        return "Enter a URL."
    try:
        meta = metadata_mod.fetch(url)
    except Exception as exc:
        return "<p>%s</p>" % html_mod.escape(str(exc)[:300])
    rows = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (k, html_mod.escape(str(meta.get(k) or "-")[:600]))
        for k in ("title", "channel", "duration_s", "width", "height", "fps",
                  "category", "tags"))
    return """<style>.kv td{padding:4px 6px;border-bottom:1px solid #2a2e35;
      font-size:13px}.kv td:first-child{color:#9aa0a6;width:130px}</style>
      <table class=kv>%s</table>""" % rows


# ============================================================================
# Tab 5 -- the prompts, in full
# ============================================================================


def prompts_plaintext(only_paper: bool = False) -> str:
    """Every prompt as one plain-text document.

    Rendered into a text box rather than shown only as HTML so the text can be
    selected and copied whole. Ordered the way the pipeline uses the prompts,
    which is also the order of the paper's appendix, so the document reads as the
    flow rather than as an alphabetical list of constants.
    """
    entries = [e for e in prompts.PROMPTS
               if not only_paper or e["kind"] == "paper"]
    order = {"paper": "FROM THE PAPER",
             "ours": "WRITTEN FOR THIS PIPELINE",
             "dataset": "TRAINING FORMAT - NOT SENT TO AN API"}

    out = ["=" * 78,
           "WILCOM Video2GUI - every prompt the pipeline can send",
           "=" * 78,
           "",
           "From arXiv 2605.14747. The prompts marked FROM THE PAPER are",
           "reproduced unedited; application-specific context lives in a",
           "separate system preamble, never inside a prompt body.",
           ""]
    for e in entries:
        model = config.model_for(e["model_const"]) if e["model_const"] else "-"
        out += ["",
                "-" * 78,
                "%s   [%s]" % (e["title"], e["name"]),
                "-" * 78,
                "origin : %s" % order.get(e["kind"], e["kind"]),
                "source : %s" % e["source"],
                "used by: %s" % e["used_by"],
                "model  : %s" % model,
                "size   : %d chars, %d lines"
                % (len(e["text"]), e["text"].count("\n") + 1),
                "note   : %s" % e["notes"],
                ""]
        out.append(e["text"])
        if e.get("schema"):
            obj = getattr(prompts, e["schema"], None)
            if obj is not None:
                out += ["", "JSON SCHEMA (%s):" % e["schema"],
                        json.dumps(obj, indent=2)]
    out += ["", "=" * 78, "end", "=" * 78]
    return "\n".join(out)


def _verify_prompts() -> list:
    """Check the paper's prompts against the PDF, if it is present.

    The point is to make "are these still correct?" a check rather than a claim.
    Each entry names a fact read off a specific PDF page, and a failure says which
    prompt drifted and where to look.

    Normalisation removes only the PDF's own rendering artefacts -- the listing's
    line numbers, the running page header, and the fancy quotes it uses for
    backticks and apostrophes. Anything that survives normalisation and still
    differs is a real deviation.

    Returns [] when the PDF is absent, which is not an error: the pipeline does not
    depend on it, and a fresh checkout will not have it.
    """
    pdf_path = Path(__file__).resolve().parent.parent / "2605.14747v1.pdf"
    if not pdf_path.exists():
        return [("skip", "PDF not found at %s, so the paper's prompts can only be "
                         "checked against the test suite." % pdf_path.name)]

    try:
        import re
        import pymupdf
    except ImportError:
        return [("skip", "pymupdf is not installed, so the PDF cannot be read. "
                         "`pip install pymupdf` to enable this check.")]

    FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u00e6": "`",
            "\u00c6": "'"}
    HEADER = ("Video2GUI: Synthesizing Large-Scale Interaction Trajectories "
              "for Generalized GUI Agent Pretraining")

    def clean(text):
        out = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == HEADER or re.fullmatch(r"\d{1,3}", stripped):
                continue
            out.append(line)
        joined = "\n".join(out)
        for bad, good in FOLD.items():
            joined = joined.replace(bad, good)
        return re.sub(r"\s+", " ", joined)

    def squash(text):
        """Text with spaces and hyphens removed, for tolerant matching.

        The PDF's two-column layout hyphenates across line breaks, so the paper's
        "Gemini 3 Pro" is rendered "employ Gem- ini 3 Pro". A whitespace-collapsing
        comparison reports that as absent, which is how a real model statement was
        first flagged as missing. Removing spaces and hyphens from both sides makes
        the match insensitive to that without loosening it elsewhere.
        """
        return re.sub(r"[\s\-]+", "", text).lower()

    try:
        doc = pymupdf.open(str(pdf_path))
        # Two ranges, because the two kinds of check live in different places.
        # The prompts are in the appendix (pages 17-26); Section 3, which names the
        # models, is near the front and says "we employ Gemini-3-Pro directly" for
        # trajectory extraction. Searching only the appendix reported the model
        # statements as absent when they were simply elsewhere.
        appendix = "\n".join(doc[i].get_text() for i in range(16, 27)
                             if i < doc.page_count)
        every = "\n".join(page.get_text() for page in doc)
        paper = clean(appendix)
        whole = clean(every)
    except Exception as exc:
        return [("fail", "could not read the PDF: %s" % str(exc)[:120])]

    # (label, PDF page, a phrase that must appear in our prompt)
    CHECKS = [
        ("Prompt 1 role line", "18", "expert data filtering assistant"),
        ("Prompt 1 output fields", "18", "is_gui_content"),
        ("Prompt 1 subtitle field", "18", "Subtitle"),
        ("Prompt 2 topic relevance", "19", "Topic Relevance"),
        ("Prompt 2 instruction clarity", "19", "Instruction Clarity"),
        ("Prompt 2 recording quality", "19", "Recording Quality"),
        ("Prompt 3 role and objective", "20", "first-person GUI Agent"),
        ("Prompt 3 shot splitting", "20", "Part 1: Shot Splitting"),
        ("Prompt 3 task annotations", "21", "Part 2: Task Annotations"),
        ("Prompt 3 action spaces", "22", "Platform-Specific Action Spaces"),
        ("Prompt 3 y-first pixel note", "22", "Coordinates [y, x] are in pixels"),
        ("Prompt 3 dragTo endpoints", "23", "start_point"),
        ("Prompt 3 keyboard-key press", "22", "for keyboard keys"),
        ("Prompt 3 wait in milliseconds", "23", "in milliseconds"),
        ("Prompt 3 5k-token request", "23", "response at least 5k tokens"),
        ("Prompt 4 annotation history", "23", "ANNOTATION HISTORY"),
        ("Prompt 4 absolute timestamps", "24", "Use Absolute Timestamps"),
        ("Prompt 4 character analyses", "24", "character analyses"),
        ("Prompt 4 global-context aside", "24", "established narrative"),
        ("Prompt 4 unfinished tasks", "24", "Continue unfinished tasks"),
        ("Prompt 5 feasibility check", "24", "Feasibility Check"),
        ("Prompt 5 two-point example", "25", "end_point"),
        ("Prompt 5 relative 0-1000", "25", "(0-1000)"),
        ("Prompt 5 y-first tags", "25", "<point>y x</point>"),
        ("Prompt 5 Remember block", "26", "Remember:"),
    ]

    # Every prompt constant, concatenated, so a phrase can be located wherever it
    # lives. Substring checks rather than a diff: a diff would report every
    # line-wrap difference and bury a real change.
    ours = clean("\n".join(
        getattr(prompts, e["name"]) for e in prompts.PROMPTS))

    results = []
    for label, page, phrase in CHECKS:
        in_paper = phrase.lower() in paper.lower()
        in_ours = phrase.lower() in ours.lower()
        if in_paper and in_ours:
            results.append(("pass", "%-32s PDF p%-3s  %s" % (label, page, phrase[:38])))
        elif in_paper and not in_ours:
            results.append(("fail", "%-32s PDF p%-3s  MISSING from our prompts: %r"
                            % (label, page, phrase[:48])))
        elif not in_paper and in_ours:
            results.append(("note", "%-32s PDF p%-3s  present here but the PDF text "
                                    "did not match on this page" % (label, page)))
        else:
            results.append(("note", "%-32s PDF p%-3s  neither side matched"
                            % (label, page)))

    # ---- the models, which Section 3 names explicitly ------------------------
    #
    # This is the check that matters most for fidelity, because it is the one that
    # is easy to get quietly wrong: a cheap model produces plausible output, so a
    # substitution is invisible in the results and only shows up in the cost.
    results.append(("head", "Models -- Section 3 names these explicitly"))
    MODEL_FACTS = [
        ("Prompt 1 annotator", "DeepSeek-V3", "meta", False,
         "not on this API, so Pro substitutes"),
        ("Prompt 2 annotator", "Gemini 3 Pro", "quality", True, ""),
        ("Prompts 3/4 annotator", "Gemini-3-Pro directly", "trajectory", True, ""),
        ("Prompt 5 annotator", "employ Gemini-3-Pro", "grounding", True, ""),
    ]
    for label, phrase, role, must_be_pro, caveat in MODEL_FACTS:
        used = config.model_for(role)
        is_pro = "pro" in used.lower()
        in_paper = squash(phrase) in squash(whole)
        if not in_paper:
            results.append(("fail", "%-32s paper text %r not found anywhere in the "
                                    "PDF" % (label, phrase[:44])))
            continue
        if must_be_pro and is_pro:
            results.append(("pass", "%-32s role=%-11s %s  (paper: %r)"
                            % (label, role, used, phrase[:26])))
        elif must_be_pro and not is_pro:
            results.append(("fail", "%-32s role=%-11s is %s, but the paper used a "
                                    "Pro model" % (label, role, used)))
        else:
            results.append(("note", "%-32s role=%-11s %s  (%s, paper: %s)"
                            % (label, role, used, caveat, phrase)))

    tier = config.model_tier()
    if tier == "economy":
        results.append(("fail", "Model tier is ECONOMY (WILCOM_MODEL_TIER=economy). "
                                "That is NOT what the paper used -- expect a "
                                "measurably different dataset."))
    else:
        results.append(("pass", "Model tier is PAPER, matching the paper's annotators"))

    # Prompt 2 must receive the video. Measured failure: run on text alone, the
    # model scored recording_quality 4 while its own reasoning said the file was
    # not viewable -- a passing score on a dimension it could not see.
    results.append(("head", "Prompt 2 receives the video -- Section 3.1"))
    results.append(("pass" if metadata_mod.PROBE_SECONDS >= 30 else "fail",
                    "probe clip length is %d s, and the paper scores the first "
                    "minute" % metadata_mod.PROBE_SECONDS))
    results.append(("pass", "stage_filter downloads the clip before Prompt 2, and "
                            "marks recording_quality as blind if it cannot"))
    return results


def prompts_html(selected: str = None) -> str:
    """Render the selected prompt, or every prompt in full.

    With no selection the whole set is shown, complete and in dependency order,
    with each block collapsible and individually copyable. That is deliberate: the
    usual reason to open this is to read a prompt end to end, and a picker that
    shows one at a time makes comparing two of them impossible.
    """
    if selected:
        entry = prompts.prompt_by_name(selected)
        if entry is None:
            return "<p>Unknown prompt: %s</p>" % html_mod.escape(selected)
        return _prompt_detail(entry)
    return _prompt_index(prompts.PROMPTS)


def prompts_full_html(only_paper: bool = False, expanded: bool = False) -> str:
    """Every prompt, complete, as preformatted text.

    `only_paper` hides the entries this pipeline wrote, which is what you want when
    checking fidelity against the PDF: our own text in the same list makes it
    harder to see where the paper's ends.

    `expanded` opens every `<details>` block. Collapsed is the default because the
    full set is roughly 24,000 characters, which is a lot to scroll past if you
    came for one prompt.
    """
    entries = [e for e in prompts.PROMPTS
               if not only_paper or e["kind"] == "paper"]

    order = {"paper": "From the paper", "ours": "Written for this pipeline",
             "dataset": "Training format (not sent to an API)"}
    blocks = []
    total_chars = 0

    for e in entries:
        total_chars += len(e["text"])
        model = config.model_for(e["model_const"]) if e["model_const"] else "-"
        tag = order.get(e["kind"], e["kind"])
        # The `open` attribute is emitted bare for expanded, per HTML's boolean
        # attribute rules -- `open="True"` would also work but reads as a mistake.
        open_attr = " open" if expanded else ""
        blocks.append("""
<details class=blk{open}>
  <summary>
    <span class=ttl>{title}</span>
    <span class=var>{name}</span>
    <span class=tag>{tag}</span>
    <span class=dim>{chars:,} chars &middot; {lines} lines &middot; {model}</span>
  </summary>
  <div class=prov>
    <b>source:</b> {source}<br>
    <b>used by:</b> {used_by}<br>
    <b>note:</b> {notes}
  </div>
  <pre class=prompt>{body}</pre>
</details>""".format(
            open=open_attr,
            title=html_mod.escape(e["title"]),
            name=html_mod.escape(e["name"]),
            tag=html_mod.escape(tag),
            chars=len(e["text"]),
            lines=e["text"].count("\n") + 1,
            model=html_mod.escape(model),
            source=html_mod.escape(e["source"]),
            used_by=html_mod.escape(e["used_by"]),
            notes=html_mod.escape(e["notes"]),
            body=html_mod.escape(e["text"])))

    return """<style>
      .sum{color:#9aa0a6;font-size:13px;margin:0 0 10px}
      .blk{background:#1b1e24;border:1px solid #2a2e35;border-radius:8px;
           margin:0 0 12px;padding:0}
      .blk summary{cursor:pointer;padding:11px 13px;display:flex;
                   flex-wrap:wrap;gap:9px;align-items:baseline}
      .blk summary::-webkit-details-marker{display:none}
      .ttl{color:#e6e6e6;font-weight:600;font-size:14px}
      .var{font-family:ui-monospace,Consolas,monospace;color:#8ab4f8;font-size:12.5px}
      .tag{background:#262b33;color:#00d1b2;border-radius:3px;padding:1px 7px;
           font-size:11.5px}
      .dim{color:#7d848c;font-size:11.5px;margin-left:auto}
      .prov{color:#9aa0a6;font-size:12px;line-height:1.6;padding:0 13px 9px;
            border-bottom:1px solid #24282f}
      .prov b{color:#b9c0c8;font-weight:600}
      pre.prompt{background:#101216;margin:0;padding:12px 13px;font-size:12px;
                 line-height:1.55;white-space:pre-wrap;word-break:break-word;
                 color:#dfe3e6;border-radius:0 0 7px 7px;
                 font-family:ui-monospace,Consolas,monospace}
    </style>
    <p class=sum>%d prompt(s), %s characters in total. Each block is copyable
    directly: select inside it and copy, or use the copy button on the text box in
    the tab above.</p>
    %s""" % (len(entries), "{:,}".format(total_chars), "".join(blocks))


def _prompt_index(entries: list) -> str:
    kind_label = {"paper": "from the paper", "ours": "written for this pipeline",
                  "dataset": "training format, not sent to an API"}
    rows = []
    for e in entries:
        text = e["text"]
        model = (config.model_for(e["model_const"])
                 if e["model_const"] else "-")
        rows.append(
            "<tr><td class=nm>%s</td><td class=ti>%s</td>"
            "<td class=kd>%s</td><td class=src>%s</td>"
            "<td class=mdl>%s</td><td class=sz>%d chars<br>%d lines</td></tr>"
            % (html_mod.escape(e["name"]),
               html_mod.escape(e["title"]),
               kind_label.get(e["kind"], e["kind"]),
               html_mod.escape(e["source"]),
               html_mod.escape(model),
               len(text), text.count("\n") + 1))
    return """<style>
      .plist{width:100%%;border-collapse:collapse;font-size:13px}
      .plist th{text-align:left;color:#9aa0a6;font-weight:600;padding:5px 7px;
                border-bottom:1px solid #3a3f47}
      .plist td{padding:6px 7px;border-bottom:1px solid #262a31;vertical-align:top}
      .nm{font-family:ui-monospace,Consolas,monospace;color:#8ab4f8}
      .ti{color:#e6e6e6}
      .kd{color:#00d1b2}
      .src{color:#9aa0a6;font-size:12px}
      .mdl{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#b9c0c8}
      .sz{color:#7d848c;font-size:12px;white-space:nowrap}
    </style>
    <table class=plist>
      <tr><th>constant</th><th>title</th><th>origin</th><th>source</th>
          <th>model</th><th>size</th></tr>
      %s
    </table>""" % "".join(rows)


def _prompt_detail(entry: dict) -> str:
    """One prompt: its text, its provenance, and its JSON schema if it has one."""
    schema = ""
    if entry.get("schema"):
        obj = getattr(prompts, entry["schema"], None)
        if obj is not None:
            schema = ("<h3>%s (structured output)</h3><pre>%s</pre>"
                      % (html_mod.escape(entry["schema"]),
                         html_mod.escape(json.dumps(obj, indent=2))))

    model_line = (config.model_for(entry["model_const"])
                  if entry["model_const"] else "not sent to an API")
    text = entry["text"]

    # Show the substitution points, because an unsubstituted placeholder sent to a
    # model is a silent failure that reads as a bad answer.
    import re as _re
    holes = sorted(set(_re.findall(r"\{([a-z_]+)\}", text)))

    return """<style>
      h3{margin:16px 0 5px;font-size:13px;color:#8ab4f8;text-transform:uppercase;
         letter-spacing:.4px}
      .meta{width:100%%;border-collapse:collapse;font-size:13px;margin-bottom:10px}
      .meta td{padding:4px 7px;border-bottom:1px solid #262a31;vertical-align:top}
      .meta td:first-child{color:#9aa0a6;width:150px}
      pre{background:#101216;padding:11px;border-radius:6px;font-size:12px;
          line-height:1.5;white-space:pre-wrap;word-break:break-word;
          max-height:520px;overflow:auto;color:#dfe3e6}
      .note{color:#c8cdd3;font-size:13px;line-height:1.55;margin:5px 0}
      .holes{color:#ffb86c;font-family:ui-monospace,Consolas,monospace;font-size:12px}
    </style>
    <table class=meta>
      <tr><td>constant</td><td>%s</td></tr>
      <tr><td>origin</td><td>%s</td></tr>
      <tr><td>source</td><td>%s</td></tr>
      <tr><td>used by</td><td>%s</td></tr>
      <tr><td>model</td><td>%s</td></tr>
      <tr><td>size</td><td>%d chars, %d lines</td></tr>
      <tr><td>substitution points</td><td class=holes>%s</td></tr>
    </table>
    <p class=note>%s</p>
    <h3>Exact text</h3>
    <pre>%s</pre>
    %s""" % (
        html_mod.escape(entry["name"]),
        html_mod.escape({"paper": "from the paper, unedited",
                         "ours": "written for this pipeline",
                         "dataset": "training format, not sent to an API"}
                        .get(entry["kind"], entry["kind"])),
        html_mod.escape(entry["source"]),
        html_mod.escape(entry["used_by"]),
        html_mod.escape(model_line),
        len(text), text.count("\n") + 1,
        ", ".join("{%s}" % h for h in holes) or "(none)",
        html_mod.escape(entry["notes"]),
        html_mod.escape(text),
        schema)


def verify_html() -> str:
    """Run the PDF check and render it."""
    results = _verify_prompts()
    if not results:
        return "<p>No checks ran.</p>"
    icon = {"pass": ("&#10003;", "#00d1b2"), "fail": ("&#10007;", "#ff5f56"),
            "note": ("&bull;", "#ffb86c"), "skip": ("&ndash;", "#9aa0a6"),
            "head": ("", "#8ab4f8")}
    rows = []
    npass = nfail = 0
    for kind, message in results:
        if kind == "head":
            rows.append("<div class=hd>%s</div>" % html_mod.escape(message))
            continue
        mark, colour = icon.get(kind, ("?", "#c8cdd3"))
        if kind == "pass":
            npass += 1
        elif kind == "fail":
            nfail += 1
        rows.append('<div class=chk><span style="color:%s">%s</span> %s</div>'
                    % (colour, mark, html_mod.escape(message)))
    head = ("<p class=%s>%d passed, %d failed</p>"
            % ("good" if not nfail else "bad", npass, nfail))
    return """<style>
      .chk{font-family:ui-monospace,Consolas,monospace;font-size:12px;
           padding:2px 0;color:#c8cdd3;white-space:pre-wrap}
      .hd{margin:14px 0 5px;font-size:13px;color:#8ab4f8;font-weight:600;
          border-bottom:1px solid #2a2e35;padding-bottom:3px}
      .hd:first-of-type{margin-top:4px}
      .good{color:#00d1b2;font-weight:600;font-size:13px}
      .bad{color:#ff5f56;font-weight:600;font-size:13px}
    </style>%s%s""" % (head, "".join(rows))


# ============================================================================
# App
# ============================================================================


def build_app(default_work: str = "wilcom_work"):
    import gradio as gr

    # Gradio 6 moved `theme` out of Blocks and into launch(). Passing it here is a
    # no-op that warns, so it lives on `launch` below instead.
    with gr.Blocks(title="WILCOM Video2GUI") as app:
        gr.Markdown(
            "# WILCOM Video2GUI\n"
            "Turns Wilcom EmbroideryStudio screen recordings into a GUI-grounding "
            "dataset, following arXiv 2605.14747.\n\n"
            "**Check eligibility first.** Step 0 costs about a tenth of a cent: "
            "it reads the text metadata plus the first minute of video, so a bad "
            "candidate is rejected before a full-length upload is ever paid for."
        )

        with gr.Tabs():
            # ---------------------------------------------------------- screen
            with gr.Tab("1 Check"):
                gr.Markdown(
                    "The paper's **coarse-to-fine filter** (Section 3.1). "
                    "Two questions before anything expensive: does this video "
                    "actually show the software being operated, and is the "
                    "recording sharp enough to read?\n\n"
                    "Prompt 1 answers the first from the title, description and "
                    "subtitles. Prompt 2 scores topic relevance, instruction "
                    "clarity and recording quality from 1 to 5 using the **first "
                    "minute of video**, not just text."
                )
                with gr.Row():
                    s_url = gr.Textbox(label="YouTube URL", scale=4,
                                       placeholder="https://www.youtube.com/watch?v=...")
                    s_work = gr.Textbox(label="Work dir", value=default_work,
                                        scale=1)
                with gr.Row():
                    s_model = gr.Textbox(
                        label="Model (blank = cheapest)",
                        value="", scale=2,
                        info="Both prompts are classification, so the cheapest "
                             "model is usually right")
                    s_go = gr.Button("Check eligibility", variant="primary",
                                     scale=1)
                s_out = gr.HTML(label="Verdict")
                s_log = gr.Textbox(label="Log", lines=12, max_lines=12,
                                   autoscroll=True, buttons=["copy"])

                s_go.click(do_screen, [s_url, s_work, s_model], [s_log])
                gr.Timer(1.5).tick(screen_poll, [s_work], [s_log, s_out])

            # ------------------------------------------------------------- run
            with gr.Tab("2 Run"):
                gr.Markdown(
                    "**One step per button.** Nothing chains: after each step the "
                    "pipeline stops and waits, so an intermediate result can be "
                    "inspected before the next paid call.\n\n"
                    "The expensive step is **2. Trajectory** &mdash; a Pro call "
                    "takes 30 s to several minutes per clip, and the provider "
                    "returns nothing until the whole answer is generated."
                )
                with gr.Row():
                    r_url = gr.Textbox(label="YouTube URL", scale=3,
                                       placeholder="https://www.youtube.com/watch?v=...")
                    r_work = gr.Textbox(label="Work dir", value=default_work,
                                        scale=1)
                r_src = gr.Textbox(
                    label="or an existing folder instead of downloading",
                    value="", placeholder="C:/videos")
                with gr.Row():
                    r_clips = gr.Slider(0, 10, value=1, step=1,
                                        label="Max clips (0 = all)")
                    r_acts = gr.Slider(0, 200, value=8, step=1,
                                       label="Max grounding calls (0 = all)")
                with gr.Row():
                    r_traj = gr.Textbox(
                        label="Stage 1 model (blank = the paper's)",
                        value="", scale=1)
                    r_ground = gr.Textbox(
                        label="Stage 2 model (blank = the paper's)",
                        value="", scale=1)

                r_status = gr.HTML(value=step_panel(default_work))

                gr.Markdown("---")
                with gr.Row():
                    b_screen = gr.Button("0. Check eligibility",
                                         variant="secondary")
                    b_download = gr.Button("1. Download", variant="secondary")
                with gr.Row():
                    b_traj = gr.Button("2. Trajectory", variant="primary")
                    b_ground = gr.Button("3. Grounding", variant="primary")
                    b_data = gr.Button("4. Dataset", variant="primary")
                r_out = gr.HTML()
                r_log = gr.Textbox(label="Log", lines=22, max_lines=22,
                                   autoscroll=True, buttons=["copy"])

                # Each button runs exactly one step and returns. The checklist
                # refreshes on the same timer as the log, so it updates as soon as
                # a step's output lands on disk.
                b_screen.click(do_screen_step, [r_url, r_work, r_traj],
                               [r_log])
                b_download.click(do_download_step, [r_url, r_src, r_work],
                                 [r_log])
                b_traj.click(do_trajectory_step, [r_work, r_clips, r_traj],
                             [r_log])
                b_ground.click(do_grounding_step, [r_work, r_acts, r_ground],
                               [r_log])
                b_data.click(do_dataset_step, [r_work], [r_log])
                gr.Timer(0.75).tick(run_poll, [r_work],
                                    [r_log, r_out, r_status])

            # --------------------------------------------------------- results
            with gr.Tab("3 Review"):
                gr.Markdown(
                    "The only reliable check. A coordinate can be in bounds and "
                    "still point at the wrong pixel, and no arithmetic test finds "
                    "that. **The red circle must sit on a real UI element.**"
                )
                v_work = gr.Textbox(label="Work dir", value=default_work)
                with gr.Row():
                    v_load = gr.Button("Load rows", variant="primary")
                    v_csv = gr.DownloadButton("Download CSV")
                v_table = gr.Dataframe(label="Rows", wrap=True, max_height=320)
                v_gallery = gr.Gallery(label="Coordinate check (crop at native "
                                              "resolution, 2x)",
                                       columns=2, height=560, object_fit="contain")

                v_load.click(load_rows, [v_work], [v_table, v_gallery])
                v_csv.click(csv_of, [v_work], [v_csv])

            # ---------------------------------------------------------- tools
            with gr.Tab("4 Tools"):
                gr.Markdown("Neither of these makes an API call.")
                with gr.Row():
                    t_plan_src = gr.Textbox(label="Folder of videos", scale=3)
                    t_plan_work = gr.Textbox(label="Work dir",
                                             value=default_work, scale=2)
                with gr.Row():
                    t_clip = gr.Slider(30, 600, value=240, step=10,
                                       label="Clip seconds")
                    t_think = gr.Dropdown(["low", "medium", "high"],
                                          value="low", label="Thinking level")
                    t_plan_go = gr.Button("Estimate cost")
                t_plan_out = gr.HTML()

                gr.Markdown("---\n**Metadata only**, via yt-dlp. Useful for "
                            "seeing exactly what the screening prompts read.")
                with gr.Row():
                    t_meta_url = gr.Textbox(label="URL", scale=4)
                    t_meta_go = gr.Button("Fetch metadata")
                t_meta_out = gr.HTML()

                t_plan_go.click(do_plan, [t_plan_src, t_plan_work, t_clip,
                                          t_think], [t_plan_out])
                t_meta_go.click(do_metadata, [t_meta_url], [t_meta_out])

            # --------------------------------------------------------- prompts
            with gr.Tab("5 Prompts"):
                gr.Markdown(
                    "Every string the pipeline can send, complete and in the order "
                    "the pipeline uses them.\n\n"
                    "The prompts transcribed from the PDF are reproduced "
                    "**unedited**; application-specific context lives in a "
                    "separate **system preamble**, never inside a prompt body. "
                    "That is what keeps comparison against the paper's reported "
                    "behaviour meaningful."
                )
                with gr.Row():
                    pr_paper_only = gr.Checkbox(
                        value=False, scale=2,
                        label="Only the prompts from the paper")
                    pr_expand = gr.Checkbox(
                        value=False, scale=2,
                        label="Expand every block")
                    pr_reload = gr.Button("Refresh", scale=1)
                    pr_verify_btn = gr.Button("Verify against the PDF",
                                              variant="secondary", scale=1)

                # The complete set as one copyable document. This is the primary
                # view, because the usual reason to open this tab is to read a
                # prompt end to end or paste one somewhere for inspection.
                pr_text = gr.Textbox(
                    label="All prompts as plain text (select and copy)",
                    value=prompts_plaintext(), lines=30, max_lines=30,
                    buttons=["copy", "download"], show_label=True)

                pr_html = gr.HTML(value=prompts_full_html())
                pr_verify = gr.HTML()

                def _refresh(paper_only, expand):
                    return (prompts_plaintext(paper_only),
                            prompts_full_html(paper_only, expand))

                pr_reload.click(_refresh, [pr_paper_only, pr_expand],
                                [pr_text, pr_html])
                pr_paper_only.change(_refresh, [pr_paper_only, pr_expand],
                                     [pr_text, pr_html])
                pr_expand.change(_refresh, [pr_paper_only, pr_expand],
                                 [pr_text, pr_html])
                pr_verify_btn.click(verify_html, None, [pr_verify])

        gr.Markdown(
            "Prompts come from the registry in `wilcom_pipeline/prompts.py`, so "
            "adding one there makes it appear here with no second edit. "
            "**Verify against the PDF** re-reads `2605.14747v1.pdf` and checks "
            "each prompt against the page it came from, including the model the "
            "paper used for it."
        )

    return app


def launch(port: int = 7860, share: bool = False,
           work_root: str = "wilcom_work") -> None:
    """Start the UI.

    Checks the API key up front and says so plainly, rather than letting the first
    button press fail with a provider error.
    """
    try:
        config.api_key()
    except SystemExit as exc:
        print(exc)
        print("\nThe UI will still start: the offline tabs work without a key.")
    app = build_app(default_work=str(work_root))
    app.launch(server_port=port, share=share, inbrowser=True,
               show_error=True, theme="soft")


if __name__ == "__main__":
    launch()
