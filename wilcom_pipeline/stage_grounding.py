"""Stage 2 -- action spatial grounding.

Stage 1 reads a compressed video, so its coordinates are coarse by construction.
This stage recovers precision: for each action it grabs three full-resolution
frames around the action instant and asks for the element's location in them.

Three things about the returned numbers make this stage the riskiest in the
pipeline, and each has an explicit gate:

  1. **Axis order.** Prompt 5 asks for `<point>y x</point>` and
     `<bbox>y1 x1 y2 x2</bbox>`. Our dataset stores `[x, y]`. Reading them in the
     wrong order produces coordinates that look plausible and are on the wrong
     axis -- a mirror flip, not an obvious error.

  2. **Scale.** Prompt 5 returns 0-1000 relative coordinates; the dataset stores
     pixels. Mixing the two is silent and catastrophic, and this project has
     already lost a dataset to exactly that bug once.

  3. **Hallucination.** If the model returns a box and a point that disagree, or a
     box that does not contain its own centre point, one of them is invented.

Gates 1 and 2 are settled by arithmetic on known values. Gate 3 and the axis test
are settled by the one invariant the model cannot fake: a centre point must fall
inside its own bounding box.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config, hashing, log, prompts, records, timecode, video
from .errors import PermanentError
from .replies import extract_json

PROMPT_VERSION = "ground-v2"

_TAG_RE = {"point": r"<point>\s*([^<]*?)\s*</point>",
           "bbox": r"<bbox>\s*([^<]*?)\s*</bbox>"}


# ============================================================================
# Parsing
# ============================================================================


def parse_tagged(text: str, tag: str) -> list[int]:
    """Pull the integers out of a `<point>y x</point>` / `<bbox>...</bbox>` tag."""
    m = re.search(_TAG_RE[tag], str(text or ""), re.IGNORECASE)
    if not m:
        return []
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def parse_grounding(reply: str) -> dict:
    """Grounded reply -> {'feasible', 'reason', 'point_yx', 'bbox_yx'}, all raw.

    Coordinates come out in the model's own order (y first) and its own scale
    (0-1000). Conversion happens later, after validation, so a conversion bug can
    never be confused with a parsing bug.
    """
    data = extract_json(reply)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("grounding reply is not an object")

    feasible = data.get("feasible")
    if isinstance(feasible, str):
        feasible = feasible.strip().lower() in ("true", "yes", "1")

    out = {"feasible": bool(feasible),
           "reason": str(data.get("reason") or "").strip(),
           "point_name": None,
           "point_yx": None,
           "bbox_yx": None}

    preds = data.get("predictions")
    if isinstance(preds, dict):
        preds = [preds]
    if isinstance(preds, list) and preds:
        # More than one prediction means the action needed several points; we
        # take the one named `point`, which is what every action space we use
        # defines. Falling back to the first keeps the row rather than losing it.
        chosen = next((p for p in preds
                       if isinstance(p, dict)
                       and str(p.get("point_name", "")).strip().lower()
                       in ("point", "start_point", "")), preds[0])
        if isinstance(chosen, dict):
            out["point_name"] = str(chosen.get("point_name") or "").strip()
            pt = parse_tagged(chosen.get("center_point"), "point")
            bb = parse_tagged(chosen.get("bounding_box"), "bbox")
            if len(pt) >= 2:
                out["point_yx"] = pt[:2]
            if len(bb) >= 4:
                out["bbox_yx"] = bb[:4]

    return out


# ============================================================================
# Validation
# ============================================================================


class Rejected(Exception):
    """The record is usable as data but the coordinate is not trustworthy."""


def validate_and_convert(parsed: dict, width: int, height: int) -> dict:
    """Validate a raw grounding and convert to absolute pixels.

    Returns {'x','y','bbox','point_norm','bbox_norm'} in pixel space, `bbox` as
    [x1, y1, x2, y2] to match the dataset convention.

    Every failing check raises `Rejected` with a reason, rather than clamping or
    guessing. A coordinate that is wrong in an unknown way is worse than a dropped
    row: the model would be trained to reproduce it.
    """
    pt = parsed.get("point_yx")

    # Feasibility first: a model that said "not visible" also returned no point,
    # and reporting that as "no point in reply" would hide the real reason.
    if not parsed.get("feasible"):
        raise Rejected("model reported not feasible: %s"
                       % (parsed.get("reason") or "")[:120])

    if not pt or len(pt) < 2:
        raise Rejected("no point in reply")

    y_n, x_n = int(pt[0]), int(pt[1])

    lo, hi = config.NORM_MIN, config.NORM_MAX
    if not (lo <= x_n <= hi and lo <= y_n <= hi):
        raise Rejected(
            "point (%d, %d) outside %d-%d: scale is not normalized"
            % (y_n, x_n, lo, hi))

    bb_n = parsed.get("bbox_yx")
    if not bb_n:
        raise Rejected("no bounding box in reply")

    y1_n, x1_n, y2_n, x2_n = (int(v) for v in bb_n)
    for v in (y1_n, x1_n, y2_n, x2_n):
        if not (lo <= v <= hi):
            raise Rejected(
                "bbox value %d outside %d-%d: scale is not normalized"
                % (v, lo, hi))

    if y2_n < y1_n:
        y1_n, y2_n = y2_n, y1_n
    if x2_n < x1_n:
        x1_n, x2_n = x2_n, x1_n

    if y2_n - y1_n < 1 or x2_n - x1_n < 1:
        raise Rejected("degenerate bbox %d x %d" % (y2_n - y1_n, x2_n - x1_n))

    # Point must be inside its own box. This is the axis-swap test: if y and x
    # were transposed, a box near one corner of the screen will not contain the
    # point, because the screen is not square. On a 16:9 capture the mismatch is
    # far larger than the tolerance below.
    tol = 20   # normalized units, generous: a real centre sits within a few
    if not (y1_n - tol <= y_n <= y2_n + tol):
        raise Rejected(
            "point y=%d outside bbox y=[%d,%d]: axes likely swapped"
            % (y_n, y1_n, y2_n))
    if not (x1_n - tol <= x_n <= x2_n + tol):
        raise Rejected(
            "point x=%d outside bbox x=[%d,%d]: axes likely swapped"
            % (x_n, x1_n, x2_n))

    # A point that is inside the box but nowhere near its middle means the box is
    # about something else, e.g. the model boxed the wider panel.
    cy_n, cx_n = (y1_n + y2_n) / 2.0, (x1_n + x2_n) / 2.0
    disagree = max(abs(y_n - cy_n), abs(x_n - cx_n))
    if disagree > config.POINT_BBOX_MAX_DISAGREEMENT:
        raise Rejected(
            "point (y=%d,x=%d) vs bbox centre (y=%.0f,x=%.0f) disagree by %.0f"
            % (y_n, x_n, cy_n, cx_n, disagree))

    x = config.norm_to_abs(x_n, width)
    y = config.norm_to_abs(y_n, height)

    m = config.EDGE_MARGIN_PX
    if not (m <= x < width - m and m <= y < height - m):
        raise Rejected("point (%d, %d) on the frame edge %dx%d"
                       % (x, y, width, height))

    x1 = config.norm_to_abs(x1_n, width)
    y1 = config.norm_to_abs(y1_n, height)
    x2 = config.norm_to_abs(x2_n, width)
    y2 = config.norm_to_abs(y2_n, height)

    return {
        "x": x, "y": y,
        "bbox": [x1, y1, x2, y2],
        "point_norm": [x_n, y_n],
        "bbox_norm": [x1_n, y1_n, x2_n, y2_n],
        "point_bbox_disagreement": round(disagree, 1),
        "image_width": width, "image_height": height,
    }


def check_against_trajectory(grounded: dict, traj_point,
                             width: int, height: int) -> dict:
    """Compare the grounded point with stage 1's own coordinate.

    Stage 1 emits pixels in `[y, x]` order on the compressed video, so it is
    coarse and will disagree somewhat. A *large* disagreement is still useful: it
    flags rows where the two stages are looking at different things. Reported,
    never used to reject -- stage 1 is the less reliable of the two.
    """
    if not traj_point:
        return {"available": False}
    ty, tx = int(traj_point[0]), int(traj_point[1])

    def dist(ax: int, ay: int) -> float:
        return ((grounded["x"] - ax) ** 2 + (grounded["y"] - ay) ** 2) ** 0.5

    # A stage-1 coordinate that exceeds the frame is normalized, not pixels.
    scale = "pixels"
    if tx > width or ty > height:
        tx = config.norm_to_abs(tx, width)
        ty = config.norm_to_abs(ty, height)
        scale = "normalized"

    out = {
        "available": True,
        "traj_xy": [tx, ty],
        "traj_scale": scale,
        # False means stage 1 pointed outside the frame even after rescaling,
        # which on a 16:9 capture is the signature of x and y being transposed.
        "traj_in_frame": bool(0 <= tx < width and 0 <= ty < height),
        "distance_px": round(dist(tx, ty), 1),
        "distance_frac": round(dist(tx, ty) / max(width, height), 4),
    }

    # A value of 1000 or less is in range read as pixels *and* read as normalized,
    # and on a screen taller than 1000 px the two readings differ by a lot. Both
    # are reported, so a systematic offset shows up as a number rather than as a
    # mystery. Neither reading is used to reject: stage 1 is the coarser stage.
    if scale == "pixels" and tx <= 1000 and ty <= 1000:
        ax, ay = config.norm_to_abs(tx, width), config.norm_to_abs(ty, height)
        out["alt_normalized_xy"] = [ax, ay]
        out["alt_distance_px"] = round(dist(ax, ay), 1)

    return out


# ============================================================================
# One action
# ============================================================================


def _frame_cache_key(frame: Path, instruction: str, action_type: str,
                     model: str) -> str:
    """Cache key for one grounding call.

    The frame's bytes are part of the key, so a different image at the same
    timestamp -- which is what re-encoding a clip produces -- is a new call rather
    than a cache hit against the wrong picture. The prompt text is hashed in as
    well, so editing Prompt 5 cannot be answered from the previous cache.
    """
    return hashing.fingerprint(PROMPT_VERSION, model, instruction, action_type,
                               prompts.grounding_prompt_fingerprint(),
                               hashing.file_digest(frame))


def _attempt_frame(frame: Path, when: float, action: dict, instruction: str,
                   atype: str, width: int, height: int, client,
                   model: str) -> dict:
    """Try to ground one frame. Never raises.

    Returns a record with `ok` set and either the converted coordinate plus its
    cross-check, or a `rejected` / `error` string saying why not. Failing softly
    is the design: the caller moves to the next frame, because the paper's rule is
    that the first frame which grounds is the answer.
    """
    record = {"t": round(when, 3), "frame": frame.name}

    try:
        request = prompts.grounding_request(
            [client.media(frame, "image")], instruction, atype)
        result = client.chat(
            model, request,
            cache_key=_frame_cache_key(frame, instruction, atype, model),
        )
        reply = result["text"]
    except PermanentError as exc:
        record["ok"] = False
        record["error"] = "permanent: %s" % str(exc)[:160]
        return record
    except Exception as exc:
        record["ok"] = False
        record["error"] = "call: %s" % str(exc)[:160]
        return record

    record["cached"] = result.get("cached", False)
    # Kept on the record, not only logged: the Details view reports what each
    # action cost, and a number that only ever existed in scrollback cannot be
    # reported afterwards without paying for the call again.
    if result.get("usage"):
        record["usage"] = result["usage"]
    if result.get("seconds") is not None:
        record["seconds"] = round(float(result["seconds"]), 2)

    try:
        parsed = parse_grounding(reply)
        record["feasible"] = parsed["feasible"]
        converted = validate_and_convert(parsed, width, height)
        record["ok"] = True
        record.update(converted)
        record["cross_check"] = check_against_trajectory(
            converted, action.get("traj_point"), width, height)
    except Rejected as exc:
        record["ok"] = False
        record["rejected"] = str(exc)
    except Exception as exc:
        record["ok"] = False
        record["rejected"] = "parse: %s" % str(exc)[:200]

    return record


def ground_action(action: dict, clip_path: Path, client,
                  paths: config.Paths, video_stem: str, model: str,
                  verbose: bool = True) -> dict:
    """Ground one action. Extracts frames, tries each, validates, converts.

    Following the paper: the three frames are tried in order and the first that
    yields a valid grounding wins. If none does, the action is dropped rather than
    written with an approximate coordinate.
    """
    frame_dir = paths.frames / video_stem
    stem = "t%08.3f_t%d_a%d" % (action["t"], action["task_id"], action["seq"])

    instruction = action.get("grounding_instruction") or ""
    atype = action.get("action_type") or ""

    log.task("t=%.1fs extracting %d frame(s) from the original, full resolution"
             % (action["t"], len(config.FRAME_OFFSETS)))
    frames = video.extract_frames_at(clip_path, action["t"],
                                     config.FRAME_OFFSETS, frame_dir, stem)
    if not frames:
        log.task("t=%.1fs no frames could be extracted, dropping"
                 % action["t"])
        return {"grounded": False, "drop": True, "attempts": [],
                "instruction": instruction, "action_type": atype,
                "error": "no frames could be extracted"}

    # Frame dimensions come from the frame itself, never from the video record:
    # grounding runs on full-resolution originals while stage 1 saw a downscaled
    # clip, so the two disagree and only this one is the coordinate space.
    width, height = video.frame_size(frames[0][1])

    attempts: list[dict] = []
    winner = None
    for when, frame in frames:
        log.task("t=%.1fs asking for offset %+.1fs  (%s)"
                 % (action["t"], when - action["t"],
                    "first choice" if not attempts else "retry %d" % len(attempts)))
        attempt = _attempt_frame(frame, when, action, instruction, atype,
                                 width, height, client, model)
        attempts.append(attempt)
        if attempt.get("ok"):
            winner = attempt
            log.task("t=%.1fs grounded on the %+.1fs frame"
                     % (action["t"], when - action["t"]))
            break
        log.task("t=%.1fs offset %+.1fs rejected: %s"
                 % (action["t"], when - action["t"],
                    str(attempt.get("rejected") or attempt.get("error")
                        or "?")[:90]))

    if winner is None:
        last = attempts[-1]
        log.warn("t=%.1fs %s: dropped -- %s"
                 % (action["t"], atype,
                    (last.get("rejected") or last.get("error") or "?")[:110]))
        return {"grounded": False, "drop": True, "attempts": attempts,
                "instruction": instruction, "action_type": atype}

    # Flat rather than nested. Stage 3 reads these keys directly, and one level
    # less makes a bad record readable in the JSON without decoding anything.
    flat = {
        "grounded": True,
        "drop": False,
        "instruction": instruction,
        "action_type": atype,
        "attempts": attempts,
        "x": winner["x"], "y": winner["y"],
        "bbox": winner["bbox"],
        "point_norm": winner["point_norm"],
        "bbox_norm": winner["bbox_norm"],
        "image_width": winner["image_width"],
        "image_height": winner["image_height"],
        "frame": winner["frame"],
        "frame_time_s": winner["t"],
        "offset_s": round(winner["t"] - action["t"], 3),
        "n_attempts": len(attempts),
        "point_bbox_disagreement": winner.get("point_bbox_disagreement"),
        "cross_check": winner.get("cross_check", {}),
    }
    action["grounding"] = flat

    if verbose:
        cross = flat["cross_check"] or {}
        extra = ""
        if cross.get("available"):
            extra = "  traj off by %.0f px" % cross["distance_px"]
        log.ok("t=%.1fs %-12s -> (%d, %d)  frame %d/%d%s"
               % (action["t"], atype, winner["x"], winner["y"],
                  flat["n_attempts"], len(config.FRAME_OFFSETS), extra))

    return flat


# ============================================================================
# Stage entry point
# ============================================================================


def ground_video(traj_record: dict, client: Client, paths: config.Paths,
                 model: str | None = None, force: bool = False,
                 verbose: bool = True, limit: int | None = None) -> dict:
    """Ground every groundable action in one video's trajectory record.

    Writes a sibling record rather than mutating the trajectory file, so
    re-running grounding with a better model does not require re-reading the
    video.
    """
    model = model or config.model_for("grounding")
    stem = traj_record["video_stem"]
    out_path = paths.ground / ("%s.json" % stem)

    if out_path.exists() and not force:
        cached = records.read_json(out_path)
        if (isinstance(cached, dict)
                and cached.get("prompt_version") == PROMPT_VERSION
                and cached.get("model") == model):
            log.info("[skip] %s already grounded (%d/%d)"
                     % (stem, cached.get("stats", {}).get("grounded", 0),
                        cached.get("stats", {}).get("attempted", 0)))
            return cached

    # Frames come from the original video, not the downscaled clip. Full
    # resolution is the entire point of this stage.
    clip_path = paths.videos / traj_record["video"]
    if not clip_path.exists():
        raise FileNotFoundError(
            "original video missing for grounding: %s" % clip_path)

    groundable = [a for t in traj_record["tasks"] for a in t["actions"]
                  if a["needs_grounding"]]
    # Recorded before the cap, because after it the two are indistinguishable --
    # `8/8 grounded` reads as complete whether the video had 8 groundable actions
    # or 100. The trajectory stage records its own truncation the same way; this is
    # the equivalent, and it was missing.
    available = len(groundable)
    if limit and limit < available:
        groundable = groundable[:limit]
        log.warn("LIMITED: grounding the first %d of %d action(s). The rest are "
                 "not attempted and are absent from the dataset -- raise the "
                 "limit and re-run to include them." % (len(groundable), available))

    log.info("grounding %d action(s) from %s with %s"
             % (len(groundable), traj_record["video"], model))

    results = []
    kept = dropped = 0
    reasons: dict = {}
    for i, action in enumerate(groundable, 1):
        log.progress(i, len(groundable),
                     "t=%s %-12s %s"
                     % (timecode.from_seconds(action["t"]),
                        action["action_type"],
                        (action["grounding_instruction"] or "")[:60]))
        try:
            res = ground_action(action, clip_path, client, paths, stem, model,
                                verbose=verbose)
        except Exception as exc:
            res = {"grounded": False, "drop": True, "attempts": [],
                   "error": str(exc)[:300],
                   "instruction": action.get("grounding_instruction", ""),
                   "action_type": action.get("action_type", "")}
            log.error("%s" % str(exc)[:200])

        res["task_id"] = action["task_id"]
        res["seq"] = action["seq"]
        res["t"] = action["t"]
        res["traj_point"] = action.get("traj_point")
        results.append(res)

        if res.get("grounded"):
            kept += 1
        else:
            dropped += 1
            # Bucket the first rejection reason so the histogram says which gate
            # is actually doing the rejecting, rather than which one I assumed.
            attempts = res.get("attempts") or []
            why = (attempts[-1].get("rejected") if attempts else None) \
                or res.get("error") or "unknown"
            key = str(why).split(":")[0][:52]
            reasons[key] = reasons.get(key, 0) + 1

    record = {
        "video": traj_record["video"],
        "video_stem": stem,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "width": traj_record["width"],
        "height": traj_record["height"],
        "results": results,
        "stats": {
            "attempted": len(results),
            "grounded": kept,
            "dropped": dropped,
            "yield": round(kept / len(results), 4) if results else 0.0,
            "avg_attempts": (round(sum(len(r.get("attempts") or [])
                                       for r in results) / len(results), 2)
                             if results else 0.0),
            # `available` is the count *before* the cap, so `attempted < available`
            # is the only thing that distinguishes a limited run from a short video.
            # `limited_to` is null rather than 0 when nothing was capped, because 0
            # is a meaningful cap value and would read as "grounded nothing".
            "available": available,
            "limited_to": len(groundable) if len(groundable) < available else None,
        },
    }

    records.write_json(record, out_path)
    s = record["stats"]
    log.ok("%s: %d/%d grounded (%.1f%%), %.2f frame(s) per action"
           % (stem, s["grounded"], s["attempted"], 100 * s["yield"],
              s["avg_attempts"]))
    if s["limited_to"] is not None:
        log.warn("%s: %d of %d groundable action(s) were attempted. This record "
                 "is partial -- the dataset built from it covers part of the "
                 "video." % (stem, s["limited_to"], s["available"]))
    if reasons:
        log.info("rejection reasons, which is what should calibrate the gates:")
        log.table([[k, v] for k, v in sorted(reasons.items(),
                                             key=lambda kv: -kv[1])],
                  ["reason", "count"])
    return record


def run(traj_records: list[dict], client, paths: config.Paths,
        **kwargs) -> list[dict]:
    paths.ensure()
    out = []
    for i, rec in enumerate(traj_records, 1):
        log.step("grounding %d/%d: %s" % (i, len(traj_records), rec["video"]))
        try:
            out.append(ground_video(rec, client, paths, **kwargs))
        except Exception as exc:
            log.error("%s: %s" % (rec["video"], str(exc)[:300]))
    return out
