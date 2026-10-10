"""One step at a time, with live progress.

This module owns the run: which step is executing, how far it has got, what it
has done so far, and what went wrong. It imports no UI framework, so the same code
drives the HTTP API, the command line and anything added later -- the previous
Gradio view was the only caller, and putting the logic inside it is what forced a
rewrite when the transport turned out to be unusable.

Two rules, both learned the hard way:

  * **Nothing chains.** A step runs, then stops. The earlier version ran all five
    stages inside one call, so the only chance to inspect an intermediate result
    was after every expensive call had already been paid for.
  * **Long silences are bugs.** A Pro call returns nothing for 30 s to 4 minutes,
    and cutting a 1080p60 clip took 94 s with no output at all. Every unit of work
    reports: numbered counters for counted things, a gauge for ffmpeg, and a
    heartbeat for a call that cannot report anything until it finishes.

The state lives behind a lock because a worker thread writes it and requests read
it, and `snapshot()` returns a plain dict that is safe to serialise.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from . import (clients, config, hf_export, log, records, report,
               stage_dataset, stage_filter, stage_grounding, stage_trajectory,
               video)

# The sequence drives the step rail, the "what is next" logic and the API routes.
# It exists once so those three cannot disagree.
STEP_ORDER = [
    ("screen", "0. Check eligibility",
     "Prompts 1 and 2. Two questions before anything expensive: is this video "
     "about the software, and is the recording sharp enough to read? Costs about "
     "a tenth of a cent and rejects a bad candidate early."),
    ("download", "1. Download",
     "yt-dlp into <work>/source, up to 1080p. No API cost, so it is safe to run "
     "against any candidate."),
    ("trajectory", "2. Trajectory",
     "Stage 1. Watches the video and writes down the tasks and the actions in "
     "them. A Pro call takes 30 s to 4 minutes per clip, and a video is at most "
     "3 clips because screening caps candidates at 12 minutes."),
    ("grounding", "3. Grounding",
     "Stage 2. Finds each action in the original full-resolution frames. Three "
     "offsets are tried and the first valid grounding wins; an action that never "
     "grounds is dropped rather than guessed."),
    ("dataset", "4. Dataset",
     "Stage 3. Joins the two stages, splits by whole trajectory, verifies the "
     "arithmetic and writes parquet plus a coordinate-check page."),
    ("review", "5. Review",
     "Recovers any screenshot that has gone missing, then reports what is "
     "available to inspect. Free -- no API calls, just ffmpeg."),
]

STEP_KEYS = [key for key, _, _ in STEP_ORDER]
STEP_LABELS = {key: label for key, label, _ in STEP_ORDER}
STEP_HELP = {key: help_ for key, _, help_ in STEP_ORDER}

# Bounded so a pathological run cannot grow without limit. A run produces a few
# thousand lines at most.
LOG_LINES = 4000
ACTIVITY_MAX = 400
FEED_ROWS = 200


# ============================================================================
# Paths and clients
# ============================================================================


def paths_for(work: str) -> config.Paths:
    return config.Paths(root=Path(work or "wilcom_work")).ensure()


def _client(paths: config.Paths):
    return clients.make_client(cache_dir=paths.root / "cache", use_cache=True,
                               verbose=True, ledger_work=paths.root)


def _ingest(files, paths):
    """Bring videos into the work tree, reusing the CLI's own ingest."""
    from .pipeline import ingest
    return ingest(files, paths, verbose=True)


# ============================================================================
# What is on disk, which is what "done" means
# ============================================================================


def _session_file(paths: config.Paths) -> Path:
    return paths.logs / "step_session.json"


def get_completed_steps(work: str) -> set[str]:
    """Return the set of step keys completed in the current run session."""
    paths = paths_for(work)
    sf = _session_file(paths)
    if sf.is_file():
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "completed" in data:
                return set(data["completed"])
        except Exception:
            pass
    return set()


def record_step_completed(work: str, step_key: str) -> None:
    """Record that a step has completed in this session."""
    paths = paths_for(work)
    sf = _session_file(paths)
    current = get_completed_steps(work)
    current.add(step_key)
    try:
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps({"completed": sorted(current), "updated_at": time.time()}), encoding="utf-8")
    except Exception:
        pass


def reset_step_session(work: str) -> None:
    """Clear session-completed steps so the pipeline resets back to Step 0."""
    paths = paths_for(work)
    sf = _session_file(paths)
    try:
        if sf.is_file():
            sf.unlink()
    except Exception:
        pass


def step_state(work: str) -> dict[str, bool]:
    """Which steps have completed in the current run session.

    A step is marked done only once it has run and completed in this session.
    Unrelated legacy files on disk do not cause premature checkmarks.
    """
    completed = get_completed_steps(work)
    return {
        key: (key in completed)
        for key in STEP_KEYS
    }


def next_step(work: str) -> str:
    """The first step not yet completed in this session, or 'done'."""
    state = step_state(work)
    for key in STEP_KEYS:
        if not state[key]:
            return key
    return "done"


def lock_state(work: str) -> dict[str, bool]:
    """Which steps may be run. Each step stays disabled until the step before it is completed.

    Step 0 is always open. Step i stays locked (disabled) until Step i-1 has completed.
    """
    state = step_state(work)
    unlocked = {}
    for index, key in enumerate(STEP_KEYS):
        unlocked[key] = True if index == 0 else bool(state[STEP_KEYS[index - 1]])
    return unlocked


def locked_reason(work: str, key: str) -> str:
    """Why a step is locked, phrased as the predecessor that must complete first."""
    if lock_state(work).get(key, True):
        return ""
    index = STEP_KEYS.index(key)
    previous = STEP_KEYS[index - 1]
    return "Complete %s first." % STEP_LABELS.get(previous, previous)


# ============================================================================
# The runner
# ============================================================================


class Runner:
    """Holds the state of the current (or last) step.

    Single job at a time, deliberately: two Pro calls at once would double the
    cost and make the progress display meaningless.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.running = False
            self.key: str | None = None
            self.started: float | None = None
            self.finished: float | None = None
            self.progress: dict | None = None
            self.action: str = ""
            self.activity: deque = deque(maxlen=ACTIVITY_MAX)
            self.lines: deque = deque(maxlen=LOG_LINES)
            self.result = None
            self.error: str | None = None

    # ---------------------------------------------------------------- events

    def _line(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def _event(self, ev: dict) -> None:
        """Translate one log event into runner state.

        Nothing here parses text. The log module decides what happened and carries
        the numbers as fields, because recovering "step 3 of 14" from a rendered
        line breaks the moment a message contains a bracket.
        """
        level = ev.get("level") or "info"
        text = ev.get("text") or ev.get("message") or ""

        with self._lock:
            if level == "progress":
                if ev.get("pct") is not None:
                    # A gauge: already a percentage, no meaningful "n of m".
                    self.progress = {"pct": float(ev["pct"]), "detail": ""}
                else:
                    index = ev.get("index") or 0
                    total = ev.get("total") or 0
                    self.progress = {
                        "pct": (100.0 * index / total) if total else None,
                        "detail": "%d of %d" % (index, total) if total else "",
                    }
                self.action = text
                return

            if level == "tick":
                # The clock already shows the same number, smoothly.
                return

            if level == "detail":
                # Supporting text: it belongs in the log, which has room for it.
                return

            self.activity.append({"level": level, "message": text,
                                  "at": ev.get("at") or time.time()})
            if level in ("task", "phase", "ok"):
                # The headline follows the newest step, so it always names what is
                # happening now instead of falling back to a generic "working".
                self.action = text

    # ---------------------------------------------------------------- running

    def start(self, key: str, fn, work: str = "") -> str:
        """Run `fn` on a worker thread. Returns the label of what started."""
        with self._lock:
            if self.running:
                return "busy"
            self.running = True
            self.key = key
            self.started = time.time()
            self.finished = None
            self.progress = None
            self.action = ""
            self.activity.clear()
            self.lines.clear()
            self.result = None
            self.error = None

        log.set_sink(self._line)
        log.set_event_sink(self._event)

        def runner():
            try:
                outcome = fn()
                with self._lock:
                    self.result = outcome
                if work and key in STEP_KEYS:
                    accepted = True
                    if isinstance(outcome, dict) and outcome.get("accepted") is False:
                        accepted = False
                    if accepted:
                        record_step_completed(work, key)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                with self._lock:
                    self.error = "%s: %s" % (type(exc).__name__, str(exc))
                self._line("ERROR %s" % traceback.format_exc()[-1500:])
            finally:
                log.clear_sink()
                log.clear_event_sink()
                with self._lock:
                    self.running = False
                    self.finished = time.time()
                    self.progress = None

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return STEP_LABELS.get(key, key)

    def elapsed(self) -> float:
        with self._lock:
            if self.started is None:
                return 0.0
            end = self.finished if not self.running else time.time()
            return max(0.0, end - self.started)

    def snapshot(self, work: str) -> dict:
        """Everything the UI needs, as plain JSON-safe data."""
        with self._lock:
            state = {
                "running": self.running,
                "step": self.key,
                "label": STEP_LABELS.get(self.key or "", ""),
                "error": self.error,
                "result": self.result,
                "progress": dict(self.progress) if self.progress else None,
                "action": self.action,
                "activity": list(self.activity)[-FEED_ROWS:],
                "activity_total": len(self.activity),
                "log": "\n".join(self.lines),
                "started": self.started,
            }
            end = self.finished if not self.running else time.time()
            state["elapsed"] = (max(0.0, end - self.started)
                                if self.started else 0.0)

        state["work"] = str(work)
        done = step_state(work)
        nxt = next_step(work)
        unlocked = lock_state(work)
        state["steps"] = [
            {"key": key, "label": label, "help": help_,
             "done": done[key],
             "is_next": (key == nxt),
             "is_running": bool(self.running and self.key == key),
             "locked": not unlocked[key],
             "locked_reason": locked_reason(work, key)}
            for key, label, help_ in STEP_ORDER
        ]
        state["next"] = nxt
        state["next_unlocked"] = nxt == "done" or unlocked.get(nxt, False)
        return state


RUNNER = Runner()


# ============================================================================
# The steps
# ============================================================================


def _running() -> bool:
    with RUNNER._lock:
        return RUNNER.running


def screen(url: str, work: str, model: str = "") -> str:
    """Step 0 -- the paper's Prompts 1 and 2."""
    url = (url or "").strip()
    if not url:
        return "Enter a YouTube URL first."
    paths = paths_for(work)

    def job():
        return stage_filter.screen_one(url, _client(paths), paths,
                                       model=model or None, force=True)

    return RUNNER.start("screen", job, work=work)


# `at 2.1MiB/s ETA 00:12`, if yt-dlp reported it. Speed and ETA answer the question
# the percentage cannot: a bar at 12% is alarming or reassuring depending entirely
# on whether it is moving.
_RATE_RE = re.compile(r"at\s+(\S+/s)\s+ETA\s+(\S+)")


def _rate_of(line: str) -> str:
    found = _RATE_RE.search(line)
    return "%s   ETA %s" % (found.group(1), found.group(2)) if found else ""


def download(url: str, source: str, work: str) -> str:
    """Step 1 -- fetch the video. No API cost."""
    url = (url or "").strip()
    source = (source or "").strip()
    if not url and not source:
        return "Give a URL, or a folder that already holds videos."
    paths = paths_for(work)

    def job():
        out_dir = paths.root / "source"
        out_dir.mkdir(parents=True, exist_ok=True)

        if url:
            _download_with_progress(url, out_dir)
            target = out_dir
        else:
            target = Path(source)

        # Probe now rather than at stage 1, so a broken file is known before any
        # paid call happens.
        from .pipeline import find_videos
        infos = []
        for path in find_videos(target):
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

    return RUNNER.start("download", job, work=work)


# yt-dlp reports progress several times a second while downloading, and a merge
# takes seconds, so a long silence means something is wrong rather than slow. This
# is what turns a stuck download into an error instead of an indefinite wait.
DOWNLOAD_SILENCE_S = 180.0


def _download_with_progress(url: str, out_dir: Path) -> None:
    """Run yt-dlp, turning its own progress lines into a real percentage.

    Streamed rather than waited on: a four-minute download behind a blank panel is
    the worst possible UI, and yt-dlp already reports an exact percentage, so the
    bar is free. stderr is folded into stdout so a chatty ffmpeg cannot fill a pipe
    buffer and deadlock the download.
    """
    log.step("downloading with yt-dlp")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--newline",
           "--no-colors", "--progress", "--no-playlist", "-f",
           "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
           "--merge-output-format", "mp4",
           "-o", str(out_dir / "%(title).70s.%(ext)s"), url]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None

    # Read on a thread and hand lines over a queue, rather than iterating the pipe.
    # Iterating blocks forever if a process yt-dlp started outlives it while holding
    # stdout -- the same shape that made a 3 s timeout take 20 s in `metadata._run`
    # -- and a plain loop offers no way to notice a stalled download.
    lines: queue.Queue = queue.Queue()

    def reader():
        try:
            for raw in proc.stdout:
                lines.put(raw)
        except Exception:
            # The pipe closing under us is the normal end of a download, not a
            # problem worth reporting.
            pass
        finally:
            # Sentinel, so the consumer can tell "stream finished" from "no output
            # yet". Without it a finished stream looks identical to a quiet one.
            lines.put(None)

    threading.Thread(target=reader, daemon=True).start()

    tail: deque = deque(maxlen=30)
    # yt-dlp fetches video and audio as two separate streams, each counting 0-100%.
    # Naming the stream is what stops the bar looking like it went backwards.
    kind, last_step = "video", -1
    log.gauge(0.0, "starting yt-dlp")

    while True:
        try:
            raw = lines.get(timeout=DOWNLOAD_SILENCE_S)
        except queue.Empty:
            video.kill_tree(proc)
            raise SystemExit(
                "the download stalled: no output for %.0f s, so yt-dlp was "
                "stopped. The URL may be throttled." % DOWNLOAD_SILENCE_S)
        if raw is None:
            break

        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)

        if "Destination:" in line:
            dest = line.split("Destination:", 1)[1].strip()
            kind = ("audio" if dest.lower().endswith(
                (".m4a", ".mp3", ".opus", ".webm")) else "video")
            log.detail("fetching %s stream: %s" % (kind, Path(dest).name))
            log.gauge(0.0, "downloading %s" % kind)
            last_step = -1
            continue

        if line.startswith("[Merger]") or "Fixup" in line:
            log.task("merging video and audio with ffmpeg")
            log.gauge(0.0, "merging streams")
            continue

        found = re.search(r"\[download\]\s+([\d.]+)%", line)
        if found:
            pct = float(found.group(1))
            log.gauge(pct, "downloading %s" % kind)
            # The bar carries the fine detail; the log just confirms it is moving.
            mark = int(pct // 10) * 10
            if mark > last_step:
                last_step = mark
                log.progress(mark, 100, "%s %s" % (kind, _rate_of(line)))
            continue

        if "ERROR" in line or "WARNING" in line:
            log.detail(line.strip())

    code = proc.wait()
    if code != 0:
        raise SystemExit("yt-dlp failed:\n%s" % "\n".join(tail))


def trajectory(work: str, max_clips: int = 0, model: str = "",
               thinking: str = "") -> str:
    """Step 2 -- stage 1 only.

    `thinking` selects the reasoning budget for this run. It is validated here
    rather than passed through, because an unrecognised level would be forwarded to
    the provider and either rejected mid-run or silently ignored, and silently
    ignoring it would make a level comparison report a difference that does not
    exist.
    """
    paths = paths_for(work)
    level = (thinking or "").strip().lower()
    if level and level not in config.THINKING_LEVELS:
        return "unknown thinking level %r. Choose one of: %s" % (
            thinking, ", ".join(config.THINKING_LEVELS))

    def job():
        from .pipeline import find_videos
        files = find_videos(paths.root / "source") or find_videos(paths.videos)
        if not files:
            raise SystemExit("nothing to read. Run step 1 first.")
        log.step("stage 1 -- trajectory extraction")
        ingested = _ingest(files, paths)
        return {"trajectories": stage_trajectory.run(
            ingested, _client(paths), paths,
            model=model or config.model_for("trajectory"),
            max_clips=max_clips or None, verbose=True,
            thinking_level=level or None)}

    return RUNNER.start("trajectory", job, work=work)


def grounding(work: str, limit: int = 0, model: str = "") -> str:
    """Step 3 -- stage 2 only, from the trajectories already on disk."""
    paths = paths_for(work)

    def job():
        traj = records.load_dir(paths.traj, verbose=True)
        if not traj:
            raise SystemExit("no trajectory records. Run step 2 first.")
        log.step("stage 2 -- action spatial grounding")
        return {"grounding": stage_grounding.run(
            traj, _client(paths), paths,
            model=model or config.model_for("grounding"),
            limit=limit or None, verbose=True)}

    return RUNNER.start("grounding", job, work=work)


def dataset(work: str) -> str:
    """Step 4 -- stage 3 only, from the cached records."""
    paths = paths_for(work)

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

        log.task("verifying row arithmetic before writing")
        problems = stage_dataset.verify(rows, splits)
        fatal = [p for p in problems if not p.startswith("NOTE")]
        for p in problems:
            (log.error if not p.startswith("NOTE") else log.info)(p)
        if fatal:
            raise SystemExit("verification failed, so nothing was written")

        built = hf_export.save_local(splits, paths)
        path = report.render(rows, paths.logs / "coordinate_check.html", limit=0)
        log.ok("wrote %s" % path)
        return {"rows": len(rows),
                "splits": {k: len(v) for k, v in splits.items()},
                "report": str(path),
                "parquet": str(built)}

    return RUNNER.start("dataset", job, work=work)


def review(work: str) -> str:
    """Step 5 -- recover missing screenshots, then report what can be inspected.

    Frames go missing for ordinary reasons: the `frames/` tree was deleted to free
    space, a work tree was copied without it, or one extraction failed quietly.
    When that happens the row does not merely lose its image --
    `stage_dataset._row_from_action` returns `frame_missing` and the row is DROPPED
    from the dataset, so the loss never appears in the parquet at all.

    Recovery therefore works from the GROUNDING records rather than from the rows,
    because the rows no longer mention the missing ones. Each grounded result
    records its frame filename and the absolute time that frame came from, so it can
    be regenerated at the exact moment the coordinate was measured against.
    Re-extracting at the action's own timestamp instead would produce a different
    frame and a subtly wrong check -- the frame that grounded may be up to half a
    second off the action.

    No API calls. ffmpeg and JSON only.
    """
    paths = paths_for(work)
    return RUNNER.start("review", lambda: run_review(paths), work=work)


def run_review(paths: config.Paths) -> dict:
    """Execute screenshot recovery and verification report for given paths."""
    traj = records.load_dir(paths.traj, verbose=True)
    ground = records.load_dir(paths.ground, verbose=True)
    if not ground:
        raise SystemExit("no grounding records. Run step 3 first.")

    log.step("stage 4 -- preparing the dataset for inspection")
    by_stem = {t.get("video_stem"): t for t in traj}

    checked = present = recovered = 0
    unrecoverable: list = []
    by_video: dict = {}

    for record in ground:
        stem = record.get("video_stem") or "?"
        source_name = (by_stem.get(stem) or {}).get("video")
        source = paths.videos / source_name if source_name else None
        stat = by_video.setdefault(stem, {"checked": 0, "recovered": 0,
                                          "missing": 0, "no_source": False})

        for result in record.get("results") or []:
            if not result.get("grounded"):
                continue
            frame_name = result.get("frame")
            if not frame_name:
                continue
            checked += 1
            stat["checked"] += 1
            dest = paths.frames / stem / frame_name

            if dest.exists():
                present += 1
                continue

            stat["missing"] += 1
            when = result.get("frame_time_s")
            if not source or not source.exists() or when is None:
                stat["no_source"] = True
                unrecoverable.append("%s/%s" % (stem, frame_name))
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                video.extract_frame(source, float(when), dest)
                recovered += 1
                stat["recovered"] += 1
                log.task("re-extracted %s at %.1fs from %s"
                         % (frame_name, float(when), source.name))
            except Exception as exc:
                unrecoverable.append("%s/%s" % (stem, frame_name))
                log.warn("could not re-extract %s: %s"
                         % (frame_name, str(exc)[:110]))

    for stem, stat in sorted(by_video.items()):
        log.detail("%-28s %d frame(s), %d missing, %d recovered%s"
                   % (stem, stat["checked"], stat["missing"],
                      stat["recovered"],
                      "  (no source video)" if stat["no_source"] else ""))

    if checked == 0:
        raise SystemExit("no grounded frames to review. Run step 3 first.")

    # What the dataset would hold now. This is the number that matters, because
    # a recovered frame turns a previously-dropped action back into a row.
    rows, stats = stage_dataset.build_rows(traj, ground, paths)
    log.task("rows available for review: %d" % len(rows))

    # If recovery changed the row count, the parquet on disk is stale: step 4
    # wrote it before these frames existed. Saying so is the difference between
    # a recovered dataset and a silently half-written one.
    stale = False
    parquet = sorted(paths.out.glob("*.parquet"))
    if recovered and parquet:
        stored = 0
        readable = True
        try:
            import pandas as pd
            for p in parquet:
                stored += len(pd.read_parquet(p))
        except Exception:
            readable = False
        if readable and stored != len(rows):
            stale = True
            log.warn("recovered %d frame(s), so the written dataset is now "
                     "stale (%d row(s) on disk vs %d available). Re-run step 4 "
                     "to include them." % (recovered, stored, len(rows)))

    coverage = (present + recovered) / checked if checked else 0.0
    if recovered:
        log.ok("recovered %d frame(s); coverage is now %.1f%% (%d of %d)"
               % (recovered, 100 * coverage, present + recovered, checked))
    elif present == checked:
        log.ok("all %d grounded frame(s) present" % checked)
    else:
        log.ok("%d of %d grounded frame(s) present" % (present, checked))
    if unrecoverable:
        log.warn("%d frame(s) could not be recovered" % len(unrecoverable))

    report = {
        "prompt_version": "review-v1",
        "videos": {k: v for k, v in sorted(by_video.items())},
        "frames_checked": checked,
        "frames_present": present,
        "frames_missing": checked - present,
        "frames_recovered": recovered,
        "frames_unrecoverable": len(unrecoverable),
        "unrecoverable": unrecoverable[:200],
        "coverage": round(coverage, 4),
        "rows": len(rows),
        "dataset_stale": stale,
    }
    records.write_json(report, paths.logs / "review.json")
    log.ok("wrote review.json")
    return report


def process_playlist_queue(work: str, model_traj: str = "",
                           model_ground: str = "", thinking: str = "") -> str:
    """Run sequential pipeline over all pending videos in the SQLite queue."""
    paths = paths_for(work)
    from . import queue as queue_mgr

    db_path = paths.root / "queue.db"
    pending = [item for item in queue_mgr.list_queue(db_path) if item["status"] in ("pending", "failed")]
    if not pending:
        raise SystemExit("No pending videos in the queue. Enqueue a playlist or retry failed videos first.")

    level = (thinking or "").strip().lower()

    def job():
        total_queued = len(pending)
        log.step("Playlist pipeline: processing %d queued video(s)" % total_queued)
        client = _client(paths)

        completed_in_run = 0
        failed_in_run = 0

        for idx, item in enumerate(pending, 1):
            vid = item["video_id"]
            title = item["title"] or vid
            url = item["url"]

            log.step("[Video %d/%d] %s" % (idx, total_queued, title[:60]))
            queue_mgr.update_item(db_path, vid, status="downloading", error_msg=None)

            try:
                # 1. Download
                target_dir = paths.videos
                target_dir.mkdir(parents=True, exist_ok=True)
                existing_before = set(target_dir.glob("*.mp4"))
                _download_with_progress(url, target_dir)
                existing_after = set(target_dir.glob("*.mp4"))
                new_files = list(existing_after - existing_before)

                if new_files:
                    video_file = new_files[0]
                else:
                    candidates = sorted(target_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if not candidates:
                        raise RuntimeError("Download completed but no MP4 file found in %s" % target_dir)
                    video_file = candidates[0]

                queue_mgr.update_item(db_path, vid, status="trajectory", local_video_path=str(video_file))

                # 2. Trajectory
                log.task("Extracting trajectory for %s" % video_file.name)
                ingested = _ingest([video_file], paths)
                traj_results = stage_trajectory.run(
                    ingested, client, paths,
                    model=model_traj or config.model_for("trajectory"),
                    thinking_level=level or None,
                    verbose=True
                )
                if not traj_results:
                    raise RuntimeError("Trajectory extraction produced no records.")

                traj_record = traj_results[0]
                actions_cnt = len(traj_record.get("actions", []))
                queue_mgr.update_item(db_path, vid, status="grounding", actions_count=actions_cnt)

                # 3. Grounding
                log.task("Grounding %d actions for %s" % (actions_cnt, video_file.name))
                ground_results = stage_grounding.run(
                    [traj_record], client, paths,
                    model=model_ground or config.model_for("grounding"),
                    verbose=True
                )
                grounded_cnt = sum(1 for g in ground_results if g.get("grounded")) if ground_results else 0
                queue_mgr.update_item(db_path, vid, status="completed", grounded_count=grounded_cnt)
                completed_in_run += 1
                log.ok("Completed video %d/%d: %d actions grounded" % (idx, total_queued, grounded_cnt))

            except Exception as exc:
                err = str(exc)
                log.error("Failed video %s: %s" % (title[:40], err[:200]))
                queue_mgr.update_item(db_path, vid, status="failed", error_msg=err[:400])
                failed_in_run += 1
                continue

        # 4. Assemble into unified dataset
        log.step("Reassembling unified dataset from all completed queue records...")
        traj_all = records.load_dir(paths.traj, verbose=False)
        ground_all = records.load_dir(paths.ground, verbose=False)
        if traj_all and ground_all:
            rows, _stats = stage_dataset.build_rows(traj_all, ground_all, paths)
            if rows:
                hf_export.save_local({"unified": rows}, paths)
                log.ok("Unified dataset updated: %d total rows on disk" % len(rows))

        return {
            "processed": total_queued,
            "completed": completed_in_run,
            "failed": failed_in_run,
        }

    return RUNNER.start("playlist", job, work=work)


def autoflow(url: str, source: str, work: str,
             thinking: str = "",
             max_clips: int = 0,
             model_traj: str = "",
             model_ground: str = "") -> str:
    """Run the complete single-video pipeline sequentially on its own:
    Download -> Trajectory -> Grounding -> Dataset -> Review.
    """
    paths = paths_for(work)
    paths.ensure()
    level = (thinking or "").strip().lower()
    if level and level not in config.THINKING_LEVELS:
        return "unknown thinking level %r. Choose one of: %s" % (
            thinking, ", ".join(config.THINKING_LEVELS))

    def job():
        log.step("Auto-flow single video pipeline started")
        t_start = time.time()

        # Phase 1: Download or Ingest
        from .pipeline import find_videos
        raw_url = (url or "").strip()
        raw_source = (source or "").strip()

        if raw_url:
            log.step("Auto-flow [1/5]: Downloading video...")
            out_dir = paths.root / "source"
            out_dir.mkdir(parents=True, exist_ok=True)
            _download_with_progress(raw_url, out_dir)
            target = out_dir
            record_step_completed(work, "download")
            log.ok("Download complete")
        elif raw_source:
            log.step("Auto-flow [1/5]: Ingesting videos from %s..." % raw_source)
            target = Path(raw_source)
            found = find_videos(target)
            if not found:
                raise SystemExit("No video files found in %s" % raw_source)
            _ingest(found, paths)
            record_step_completed(work, "download")
            log.ok("Ingest complete: %d video(s)" % len(found))
        else:
            target = paths.root / "source"
            existing = find_videos(target) or find_videos(paths.videos)
            if not existing:
                raise SystemExit("No video URL provided and no videos in %s/source" % work)
            record_step_completed(work, "download")
            log.ok("Using existing video(s) in %s/source" % work)

        # Phase 2: Trajectory
        log.step("Auto-flow [2/5]: Stage 1 Trajectory Extraction...")
        files = find_videos(target) or find_videos(paths.root / "source") or find_videos(paths.videos)
        if not files:
            raise SystemExit("No video found to extract trajectory from.")
        client = _client(paths)
        ingested = _ingest(files, paths)
        traj_results = stage_trajectory.run(
            ingested, client, paths,
            model=model_traj or config.model_for("trajectory"),
            max_clips=max_clips or None,
            thinking_level=level or None,
            verbose=True
        )
        if not traj_results:
            raise SystemExit("Trajectory extraction produced no records.")
        record_step_completed(work, "trajectory")
        log.ok("Trajectory extraction complete: %d record(s)" % len(traj_results))

        # Phase 3: Grounding
        log.step("Auto-flow [3/5]: Stage 2 Action Grounding...")
        traj_records = records.load_dir(paths.traj, verbose=True)
        if not traj_records:
            raise SystemExit("No trajectory records found for grounding.")
        ground_results = stage_grounding.run(
            traj_records, client, paths,
            model=model_ground or config.model_for("grounding"),
            verbose=True
        )
        record_step_completed(work, "grounding")
        grounded_cnt = sum(1 for g in ground_results if g.get("grounded")) if ground_results else 0
        log.ok("Grounding complete: %d action(s) grounded" % grounded_cnt)

        # Phase 4: Dataset
        log.step("Auto-flow [4/5]: Stage 3 Unified Dataset Assembly...")
        ground_records = records.load_dir(paths.ground, verbose=True)
        rows, stats = stage_dataset.build_rows(traj_records, ground_records, paths)
        for line in stats.summary().splitlines():
            log.detail(line)
        if not rows:
            raise SystemExit("Dataset assembly produced 0 rows.")
        hf_export.save_local({"unified": rows}, paths)
        record_step_completed(work, "dataset")
        log.ok("Dataset assembly complete: %d unified rows written" % len(rows))

        # Phase 5: Review & Screenshots
        log.step("Auto-flow [5/5]: Screenshot recovery & verification...")
        try:
            rev = run_review(paths)
            record_step_completed(work, "review")
            log.ok("Review complete: coverage %.1f%%" % (rev.get("coverage", 1.0) * 100))
        except (Exception, SystemExit) as exc:
            log.detail("Review step notice: %s" % exc)
            record_step_completed(work, "review")

        elapsed = time.time() - t_start
        log.ok("Auto-flow single video pipeline completed in %.1fs" % elapsed)
        return {
            "autoflow_completed": True,
            "elapsed_s": round(elapsed, 1),
            "rows": len(rows),
        }

    return RUNNER.start("autoflow", job, work=work)


STEP_LABELS["playlist"] = "Playlist Pipeline"
STEP_LABELS["autoflow"] = "Auto-Flow Pipeline"

STARTERS = {
    "screen": screen,
    "download": download,
    "trajectory": trajectory,
    "grounding": grounding,
    "dataset": dataset,
    "review": review,
    "playlist": process_playlist_queue,
    "autoflow": autoflow,
}


def start(key: str, work: str, payload: dict) -> str:
    """Start a step by key, with whatever parameters it needs.

    Refuses a locked step. Returned as a message rather than raised, because the
    API turns it into a response the UI can show next to the button.
    """
    if key not in STARTERS:
        return "unknown step: %s" % key
    if _running():
        return "busy"
    if key != "playlist" and key != "autoflow":
        reason = locked_reason(work, key)
        if reason:
            return "locked: %s" % reason
    payload = payload or {}
    if key == "screen":
        return screen(payload.get("url", ""), work, payload.get("model", ""))
    if key == "download":
        return download(payload.get("url", ""), payload.get("source", ""), work)
    if key == "trajectory":
        return trajectory(work, int(payload.get("max_clips") or 0),
                          payload.get("model", ""),
                          payload.get("thinking", ""))
    if key == "grounding":
        return grounding(work, int(payload.get("limit") or 0),
                         payload.get("model", ""))
    if key == "dataset":
        return dataset(work)
    if key == "playlist":
        return process_playlist_queue(
            work,
            model_traj=payload.get("model_traj", ""),
            model_ground=payload.get("model_ground", ""),
            thinking=payload.get("thinking", ""),
        )
    if key == "autoflow":
        return autoflow(
            payload.get("url", ""),
            payload.get("source", ""),
            work,
            thinking=payload.get("thinking", ""),
            max_clips=int(payload.get("max_clips") or 0),
            model_traj=payload.get("model_traj", ""),
            model_ground=payload.get("model_ground", ""),
        )
    return review(work)


