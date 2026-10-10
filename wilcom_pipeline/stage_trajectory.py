"""Stage 1 -- trajectory extraction.

Sends a video clip to Gemini with the paper's Prompt 3 and receives back:

  Part 1  a numbered shot list, mm:ss - mm:ss
  Part 2  a JSON list of sub-tasks, each holding several atomic actions

Then it normalises that into a single per-video record:
  tasks, each with an ordered list of actions carrying natural-language
  grounding instructions and a timestamp.

Nothing here produces coordinates we trust. The trajectory pass sees compressed
video, which is exactly why the paper added a separate grounding stage, and why
the coordinates it emits are kept only as a cross-check for stage 2.

Long videos are cut into <= 240 s clips and stitched with Prompt 4, which carries
the earlier analysis forward so a task spanning a cut survives.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from . import config, hashing, log, prompts, records, timecode, video
from .replies import extract_json

# Human-readable label recorded in the output. The cache key does NOT depend on
# this; it hashes the prompt text itself, via `prompts.trajectory_prompt_fingerprint`.
PROMPT_VERSION = "traj-v2"

# A sub-task with no `finish` action that is followed within this many seconds by
# another sub-task is treated as one task split by a clip boundary, and merged.
# The boundary is 240 s, so a gap this small cannot be a genuine new task.
MERGE_GAP_S = 8.0

# Timestamp parsing lives in `timecode`. A mis-parsed timestamp does not raise --
# it extracts the frame at the wrong moment and yields a row that looks correct --
# so that conversion is one small tested module rather than a helper per stage.


# ============================================================================
# Parsing
# ============================================================================

# `[ \t]*$` rather than `\s*$`: \s would also match the newline itself, and a
# greedy match could then run past the end of the line it was meant to anchor.
#
# The dash alternation contains a literal en-dash and em-dash (U+2013, U+2014).
# That is intentional, not stray typography: models emit "00:04 - 00:11" with
# whichever dash they prefer, and stripping them to ASCII would make the shot
# list unparseable for those replies.
_SHOT_LINE_RE = re.compile(
    r"^[ \t]*(?:\d+[.)][ \t]*|[-*][ \t]*)"     # "1." / "1)" / "-"
    r"(\d{1,3}:\d{2}(?::\d{2})?)"             # start
    r"[ \t]*(?:-|–|—|to|~)[ \t]*"              # en/em dash are deliberate
    r"(\d{1,3}:\d{2}(?::\d{2})?)"             # end
    r"[ \t]*$",
    re.MULTILINE,
)

# Phrases that only ever appear in the *template* inside the prompt, never in a
# real annotation. A model that echoes the example JSON back gets those tasks
# dropped instead of poisoning the dataset with placeholder coordinates.
TEMPLATE_MARKERS = (
    "a one-sentence summary of the user",
    "a detailed, causal description",
    "natural language instruction to precisely locate",
    "first-person reasoning",
    "timestamp (mm:ss) from part 1",
    "the operating system (e.g.",
    "parameters specific to the action_type",
)


def parse_shots(text: str) -> list[dict]:
    """Part 1 -> [{'index', 'start_s', 'end_s'}].

    Anchored to the start of a line and requiring the whole line to be a range,
    so a stray timestamp inside prose is not mistaken for a shot.
    """
    out = []
    for i, m in enumerate(_SHOT_LINE_RE.finditer(text)):
        try:
            start = timecode.to_seconds(m.group(1))
            end = timecode.to_seconds(m.group(2))
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        out.append({"index": i, "start_s": start, "end_s": end})
    return out


def strip_shot_section(text: str) -> str:
    """Everything from the 'Task Annotations' heading onward.

    Removing Part 1 before looking for JSON means a bracket in the shot list
    cannot be mistaken for the start of the JSON array.
    """
    for pattern in (r"^#{0,6}\s*Part\s*2\b.*$",
                    r"^#{0,6}\s*Task\s+Annotations?\b.*$"):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return text[m.start():]
    return text


def parse_tasks(text: str) -> list[dict]:
    """Part 2 -> list of task dicts. Raises ValueError if nothing parses."""
    if not text or not text.strip():
        raise ValueError("empty reply")

    body = strip_shot_section(text)
    value = extract_json(body)

    if isinstance(value, dict):
        # A single task returned unwrapped, or wrapped in {"tasks": [...]}
        for k in ("tasks", "task_annotations", "annotations", "data"):
            if isinstance(value.get(k), list):
                value = value[k]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError("expected a JSON list, got %s" % type(value).__name__)

    tasks = []
    echoed = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        actions = item.get("user_actions")
        if not isinstance(actions, list) or not actions:
            continue
        first = actions[0] if isinstance(actions[0], dict) else {}
        blob = " ".join(str(first.get(k) or "").lower() for k in (
            "grounding_instruction", "action_reason", "action_parameters",
            "core_change", "core_change_reason", "timestamp"))
        blob += " " + str(item.get("instruction") or "").lower()
        if any(m in blob for m in TEMPLATE_MARKERS):
            echoed += 1
            continue
        tasks.append(item)

    if echoed:
        print("      [warn] dropped %d task(s) that echoed the prompt template"
              % echoed)
    if not tasks:
        raise ValueError("JSON list contained no task with a user_actions array")
    return tasks


_INT_RE = re.compile(r"-?\d+")


def parse_numbers(value) -> list[int]:
    """Coordinates arrive as '[447, 512]' or [447, 512] or 'y=447 x=512'."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(parse_numbers(v))
        return out
    return [int(x) for x in _INT_RE.findall(str(value))]


def trajectory_point(params: dict) -> tuple[int, int] | None:
    """The paper's own coordinate for an action, as (y, x).

    Prompt 3 declares `Coordinates [y, x] are in pixels`, so the first number is
    y. Returns None when the action has no coordinate of its own.
    """
    if not isinstance(params, dict):
        return None
    for key in ("point", "start_point", "center_point"):
        nums = parse_numbers(params.get(key))
        if len(nums) >= 2:
            return nums[0], nums[1]
    return None


# ============================================================================
# Normalisation
# ============================================================================


def _norm_action(action: dict, task_id: int, seq: int, clip_index: int,
                 clip_start: float) -> dict | None:
    """One action -> a flat record, or None if it is unusable.

    Timestamps are converted to absolute seconds here. The model is told to emit
    absolute stamps, but a model that ignored that would otherwise have its
    frames extracted from the wrong place, and the failure would be invisible.
    """
    ts = action.get("timestamp")
    try:
        t = timecode.to_seconds(ts)
    except (ValueError, TypeError):
        return None

    # Guard against clip-relative output: if the stamp lands before this clip
    # starts, the model answered in clip-local time.
    if t + 1.0 < clip_start:
        t = clip_start + t

    atype = str(action.get("action_type") or "").strip()
    if not atype:
        return None

    instr = str(action.get("grounding_instruction") or "").strip()
    params = action.get("action_parameters")
    if not isinstance(params, dict):
        params = {}

    return {
        "task_id": task_id,
        "seq": seq,
        "clip": clip_index,
        "timestamp": ts,
        "t": round(t, 3),
        "action_type": atype,
        "grounding_instruction": instr,
        "action_reason": str(action.get("action_reason") or "").strip(),
        "action_parameters": params,
        "core_change_reason": str(action.get("core_change_reason") or "").strip(),
        "core_change": str(action.get("core_change") or "").strip(),
        "traj_point": trajectory_point(params),
        "needs_grounding": (
            atype not in prompts.NO_COORDINATE_ACTIONS
            and atype in prompts.COORDINATE_ACTIONS
        ),
        "grounding": None,
    }


def normalise_tasks(raw_tasks: list[dict], clip_index: int,
                    clip_start: float) -> list[dict]:
    """Raw task list -> flat list with sequential action seq numbers."""
    tasks = []
    for ti, raw in enumerate(raw_tasks):
        try:
            raw_task_id = int(raw.get("task_id", ti))
        except (TypeError, ValueError):
            raw_task_id = ti

        actions = []
        for action in raw.get("user_actions") or []:
            if not isinstance(action, dict):
                continue
            a = _norm_action(action, raw_task_id, len(actions), clip_index,
                             clip_start)
            if a is not None:
                actions.append(a)

        if not actions:
            continue

        tasks.append({
            "task_id": raw_task_id,
            "clip": clip_index,
            "clip_start_s": clip_start,
            "instruction": str(raw.get("instruction") or "").strip(),
            "dense_caption": str(raw.get("dense_caption") or "").strip(),
            "plan": str(raw.get("plan") or "").strip(),
            "platform": str(raw.get("platform") or "").strip().lower(),
            "software": str(raw.get("software") or "").strip(),
            "website": raw.get("website"),
            "actions": actions,
            "has_finish": any(a["action_type"] == "finish" for a in actions),
        })
    return tasks


def merge_across_clips(tasks: list[dict]) -> list[dict]:
    """Stitch sub-tasks that a clip boundary cut in half.

    Prompt 4 asks the model to continue the previous sub-task, and it usually
    does. When it instead starts a new one with a fresh index, the result is a
    task with no `finish` whose successor begins seconds later -- indistinguishable
    from a real truncation. Those are merged, and a global task_id is assigned in
    timeline order so ids are unique across the whole video.
    """
    ordered = sorted(tasks, key=lambda t: (t["actions"][0]["t"], t["clip"]))

    merged: list[dict] = []
    for task in ordered:
        if merged:
            prev = merged[-1]
            gap = task["actions"][0]["t"] - prev["actions"][-1]["t"]
            if (not prev["has_finish"] and gap <= MERGE_GAP_S
                    and prev["instruction"] and task["instruction"]):
                for a in task["actions"]:
                    a["seq"] = len(prev["actions"])
                    a["task_id"] = prev["task_id"]
                    prev["actions"].append(a)
                prev["has_finish"] = task["has_finish"]
                prev["clips"] = sorted(set(prev.get("clips", [prev["clip"]])
                                           + [task["clip"]]))
                continue
        merged.append(task)

    for i, task in enumerate(merged):
        task["task_id"] = i
        task["clips"] = sorted(set(task.get("clips", [task["clip"]])))
        for j, a in enumerate(task["actions"]):
            a["task_id"] = i
            a["seq"] = j

    return merged


# ============================================================================
# Stage entry point
# ============================================================================


def _clip_cache_key(clip_path: Path, model: str) -> str:
    """Cache key for one clip.

    Keyed on the clip's *bytes*, not its name, so re-encoding at a different CRF
    invalidates the cached trajectory instead of quietly reusing an annotation of
    video that no longer exists. The prompt text is hashed in for the same
    reason: an edited prompt must not be answered from the old prompt's cache.
    """
    return hashing.fingerprint(PROMPT_VERSION, model,
                               prompts.trajectory_prompt_fingerprint(),
                               hashing.file_digest(clip_path))


def _clip_meta(index: int, start: float, end: float, model: str,
               cache_key: str, shots: list, clip_tasks: list,
               reply_chars: int, cached: bool,
               error: str | None = None,
               usage: dict | None = None,
               seconds: float | None = None) -> dict:
    """One entry for the record's `clips` list.

    The success path and the failure path both come through here, so a clip that
    failed to parse records exactly the same keys as one that worked. Two separate
    dict literals is how those paths drift apart and one of them starts silently
    missing a field.

    `usage` and `seconds` are recorded rather than only logged. A number that
    exists solely in scrollback cannot be reported later, and re-deriving it means
    paying for the call again -- which is exactly what the Details view would have
    had to do.
    """
    meta = {
        "index": index,
        "start_s": start,
        "end_s": end,
        "model": model,
        "cache_key": cache_key,
        "shots": shots,
        "tasks": len(clip_tasks),
        "actions": sum(len(t["actions"]) for t in clip_tasks),
        "reply_chars": reply_chars,
        "cached": cached,
    }
    if usage:
        meta["usage"] = usage
    if seconds is not None:
        meta["seconds"] = round(float(seconds), 2)
    if error:
        meta["error"] = error
    return meta


def extract_video(src: Path, client, paths: config.Paths,
                  model: str | None = None, clip_seconds: int | None = None,
                  force: bool = False, verbose: bool = True,
                  max_clips: int | None = None,
                  thinking_level: str | None = None) -> dict:
    """Run stage 1 on one video. Returns the trajectory record and writes it.

    Resumable at two levels: a finished <video>.json short-circuits the whole
    call, and each clip's raw reply is cached by clip content hash, so a crash
    mid-video costs at most the clip in flight.

    `max_clips` caps how many segments are processed, which is how a full run is
    trialled on the first few minutes of a long recording before paying for all
    of it.
    """
    model = model or config.model_for("trajectory")
    src = Path(src)
    out_path = paths.traj / ("%s.json" % src.stem)
    level = thinking_level or config.THINKING_LEVEL

    if out_path.exists() and not force:
        cached = records.read_json(out_path)
        if (isinstance(cached, dict)
                and cached.get("prompt_version") == PROMPT_VERSION
                and cached.get("tasks")):
            # A finished record is only reusable at the SAME thinking level. It is
            # easy to miss because the level is not in the filename, and reusing a
            # `low` record for a `high` request would silently report the cheaper
            # run's output as if it were the expensive one's.
            recorded = cached.get("thinking_level")
            if recorded in (None, level):
                log.info("[skip] %s already extracted (%d tasks, thinking=%s)"
                         % (src.name, len(cached["tasks"]), recorded or "?"))
                return cached
            log.warn("%s was extracted at thinking=%s but %s was requested, so "
                     "it is being re-run and the previous record replaced"
                     % (src.stem, recorded, level))

    info = video.probe(src)
    log.info("%s" % info.summary())

    # Planned once rather than twice. The previous version called `plan_clips` again
    # inside the log line just to report the total, which is wasteful and invites the
    # two calls to disagree if the planning logic ever gains state.
    planned = video.plan_clips(info, clip_seconds)
    spans = planned[:max_clips] if max_clips else list(planned)
    truncated = len(spans) < len(planned)
    if truncated:
        covered = sum(end - start for start, end in spans)
        log.warn("TRUNCATED: processing the first %d of %d clip(s) -- %.0fs of "
                 "%.0fs (%.1f%%). The record says so; the dataset from it covers "
                 "only that much of the video."
                 % (len(spans), len(planned), covered, info.duration,
                    100 * covered / info.duration if info.duration else 0))
    paths.clips.mkdir(parents=True, exist_ok=True)

    clips_meta = []
    tasks: list[dict] = []
    history_parts: list[str] = []
    prev_end = 0.0

    for ci, (start, end) in enumerate(spans):
        clip_path = paths.clips / ("%s_%03d.mp4" % (src.stem, ci))
        span = "%s-%.0fs" % (timecode.from_seconds(start), end)
        if not clip_path.exists() or force:
            log.task("cutting clip %d/%d  %s" % (ci + 1, len(spans), span))
            # Cutting a 4-minute 1080p60 clip is the one slow local step. ffmpeg
            # reports its own position, so this is a real percentage rather than
            # a spinner, and `gauge` keeps it out of the log.
            video.make_clip(
                src, start, end, clip_path,
                size=video.target_size(info.width, info.height,
                                       config.VIDEO_LONG_SIDE),
                progress=lambda pct, ci=ci, span=span: log.gauge(
                    pct, "cutting clip %d/%d  %s" % (ci + 1, len(spans), span)))
        else:
            log.task("clip %d/%d already cut, reusing  %s"
                     % (ci + 1, len(spans), span))

        cache_key = _clip_cache_key(clip_path, model)
        size_mb = clip_path.stat().st_size / 1e6

        continuation = None
        if ci > 0 and history_parts:
            continuation = {
                "start_time": timecode.from_seconds(prev_end),
                "end_time": timecode.from_seconds(start),
                "history": "\n\n".join(history_parts),
            }

        if verbose:
            log.progress(ci + 1, len(spans),
                         "clip %s-%.0fs  %.1f MB%s"
                         % (timecode.from_seconds(start),
                            end, size_mb,
                            "  +history" if continuation else ""))

        # The clip is handed over as a path, and the client reads it into a typed
        # media part. No encoding happens here.
        request = prompts.trajectory_request(
            [client.media(clip_path, "video")], continuation,
            thinking_level=thinking_level)
        _t0 = time.time()
        result = client.chat(model, request, cache_key=cache_key)
        reply = result["text"]
        _seconds = time.time() - _t0

        shots = parse_shots(reply)
        try:
            raw_tasks = parse_tasks(reply)
        except ValueError as exc:
            # A clip that does not parse is recorded and skipped. Aborting the
            # video would throw away the clips that did work.
            log.warn("no tasks parsed from clip %d: %s" % (ci, str(exc)[:180]))
            clips_meta.append(_clip_meta(
                ci, start, end, model, cache_key, shots, [], len(reply),
                result.get("cached", False), error=str(exc)[:300],
                usage=result.get("usage"), seconds=_seconds))
            history_parts.append(reply[:20000])
            prev_end = end
            continue

        clip_tasks = normalise_tasks(raw_tasks, ci, start)
        tasks.extend(clip_tasks)

        # Only the text is carried forward: a video part cannot be replayed into a
        # later segment, and including its placeholder would invite confusion.
        history_parts.append(reply[:20000])
        prev_end = end

        clips_meta.append(_clip_meta(
            ci, start, end, model, cache_key, shots, clip_tasks, len(reply),
            result.get("cached", False), usage=result.get("usage"),
            seconds=_seconds))

        actions = sum(len(t["actions"]) for t in clip_tasks)
        log.task("clip %d/%d read: %d shot(s), %d task(s), %d action(s)%s"
                 % (ci + 1, len(spans), len(shots), len(clip_tasks), actions,
                    "  [cached]" if result.get("cached") else ""))
        log.ok("clip %d: %d shots, %d tasks, %d actions"
               % (ci, len(shots), len(clip_tasks), actions))

    merged = merge_across_clips(tasks)

    # The final task of a real trajectory ends with a `finish` action. If the model
    # never emitted one, the last task is still a task, so it is kept rather than
    # dropped -- the absence is recorded in `stats` for a reader to notice.
    record = {
        "video": src.name,
        "video_stem": src.stem,
        "duration_s": round(info.duration, 3),
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 3),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "clip_seconds": clip_seconds or config.CLIP_SECONDS,
        # Recorded so a run's reasoning budget is visible after the fact. Without
        # it, two records with very different cost and quality look identical.
        "thinking_level": thinking_level or config.THINKING_LEVEL,
        # Truncation is recorded, not just logged. A run capped at N clips produces a
        # record that is otherwise indistinguishable from a complete one -- and a
        # dataset built from it reads as "this video's trajectory" when it is the
        # first few minutes of it. The count is what makes that visible, and it is
        # the reason a log line was not enough.
        "planned_clips": len(planned),
        "processed_clips": len(spans),
        "truncated": truncated,
        "covered_s": round(sum(end - start for start, end in spans), 1),
        "clips": clips_meta,
        "tasks": merged,
        "stats": {
            "clips": len(clips_meta),
            "shots": sum(len(c["shots"]) for c in clips_meta),
            "tasks": len(merged),
            "actions": sum(len(t["actions"]) for t in merged),
            "groundable": sum(1 for t in merged for a in t["actions"]
                              if a["needs_grounding"]),
            "with_instruction": sum(1 for t in merged for a in t["actions"]
                                    if a["grounding_instruction"]),
        },
    }

    records.write_json(record, out_path)
    s = record["stats"]
    log.ok("%s: %d tasks, %d actions, %d groundable -> %s"
           % (src.name, s["tasks"], s["actions"], s["groundable"],
              out_path.name))
    return record


def run(sources: list[Path], client, paths: config.Paths,
        **kwargs) -> list[dict]:
    """Stage 1 over many videos. One bad video does not stop the batch."""
    paths.ensure()
    out = []
    level = kwargs.get("thinking_level") or config.THINKING_LEVEL
    for i, src in enumerate(sources, 1):
        log.step("trajectory %d/%d: %s  (thinking=%s)"
                 % (i, len(sources), Path(src).name, level))
        try:
            out.append(extract_video(Path(src), client, paths, **kwargs))
        except Exception as exc:
            log.error("%s: %s" % (Path(src).name, str(exc)[:300]))
    return out
