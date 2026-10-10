"""Per-video detail: slicing, actions, tokens and timings.

Everything here is derived from what is already on disk. No API calls, so opening
the view is free.

Two of the numbers need care:

**Tokens.** The client stores each reply in its cache together with the usage
metadata, but the cache FILENAME is the *effective* key -- the request key folded
with model, temperature, max output tokens, thinking level and whether a schema was
requested. That effective key is not written into the records, so it is recomputed
here from the same inputs. When it matches a cache file the token counts are real;
when it does not, the field is reported as `None` rather than guessed, because a
plausible wrong number is worse than a gap. Runs made after this change also store
`usage` directly in the record, which is preferred when present.

**Elapsed time.** Not stored before this change, and not derivable after the fact.
It is reported as `None` for older records, and `recorded: false` says why. New
runs write it.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, ledger, records


def _effective_cache_path(paths: config.Paths, cache_key: str, model: str,
                          request: dict) -> Path:
    """Where the client would have stored this reply.

    Mirrors `GeminiClient._effective_key` exactly. It has to: the direction of the
    mistake matters, because a key that is merely *different* reports "tokens
    unknown" rather than reporting another call's tokens.
    """
    from . import hashing
    effective = hashing.fingerprint(
        "gemini", cache_key or "nocache", model,
        "t=%s" % config.TEMPERATURE,
        "m=%s" % config.MAX_OUTPUT_TOKENS,
        "think=%s" % (request.get("thinking_level") or config.THINKING_LEVEL),
        "schema=%s" % bool(request.get("json_schema")),
    )
    return paths.root / "cache" / ("gemini_%s.json" % effective)


def _cached_usage(paths: config.Paths, cache_key: str, model: str,
                  request: dict) -> dict | None:
    """Usage for a past call, recovered from the client's cache. None if absent.

    FALLBACK ONLY. The ledger is the primary source; this exists so that runs made
    before the ledger existed still report real numbers. It is brittle by nature --
    it reproduces the client's effective-key derivation, and if that hashing ever
    changes this silently degrades to "unknown" (never to a wrong number).
    """
    p = _effective_cache_path(paths, cache_key, model, request)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _ledger_entry(calls: dict, cache_key: str | None) -> dict | None:
    """The ledger row for a raw cache key, if the database has one."""
    if not cache_key:
        return None
    return calls.get(cache_key)


def _usage_from_ledger(entry: dict | None) -> dict | None:
    """A ledger row expressed in the client's usage shape, so the rest of this
    module has one representation to handle."""
    if not entry or entry.get("prompt_tokens") is None:
        return None
    return {
        "prompt_tokens": entry.get("prompt_tokens"),
        "completion_tokens": entry.get("completion_tokens"),
        "completion_tokens_details": {
            "reasoning_tokens": entry.get("thinking_tokens") or 0,
        },
    }


def _tokens_in_out(usage: dict | None) -> dict:
    """Normalise a usage dict into plain counts, or all-None when unknown."""
    if not usage:
        return {"input": None, "output": None, "thinking": None, "total": None}
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    think = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    total = None
    if inp is not None or out is not None:
        total = (inp or 0) + (out or 0)
    return {"input": inp, "output": out, "thinking": think, "total": total}


def _known(bundle: dict) -> bool:
    """Did this bundle come from a real usage report, or is it a placeholder?"""
    return bundle.get("input") is not None or bundle.get("output") is not None


def _cost(model: str, tokens: dict) -> float | None:
    """Dollar cost of a token bundle, or None when the counts are unknown."""
    if tokens.get("input") is None and tokens.get("output") is None:
        return None
    try:
        price_in, price_out = ledger.price_for(model)
    except Exception:
        return None
    return ((tokens.get("input") or 0) * price_in
            + (tokens.get("output") or 0) * price_out)


def _add(a: dict, b: dict) -> dict:
    """Sum two token bundles, treating unknown as zero but remembering if it was."""
    out = {}
    for key in ("input", "output", "thinking", "total"):
        av, bv = a.get(key), b.get(key)
        if av is None and bv is None:
            out[key] = None
        else:
            out[key] = (av or 0) + (bv or 0)
    return out


def build(work: str) -> dict:
    """Per-video detail for everything currently in the work tree."""
    paths = config.Paths(root=Path(work or "wilcom_work")).ensure()

    traj_records = records.load_dir(paths.traj)
    ground_records = records.load_dir(paths.ground)
    ground_by_stem = {g.get("video_stem"): g for g in ground_records}

    # Screening records are keyed by video id, so map id -> record and look it up
    # per video rather than per stem.
    screen_by_id = {}
    for p in paths.screen.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("video_id"):
            screen_by_id[rec["video_id"]] = rec

    videos = []
    for traj in traj_records:
        stem = traj.get("video_stem") or "?"
        ground = ground_by_stem.get(stem)

        # ---- slicing ------------------------------------------------------
        clips = []
        traj_tokens = {"input": None, "output": None, "thinking": None, "total": None}
        traj_seconds = 0.0
        traj_seconds_known = False

        for clip in traj.get("clips") or []:
            model = clip.get("model") or traj.get("model") or ""
            usage = clip.get("usage")

            # Recover when the run predates usage being recorded. `request` is
            # rebuilt only to reproduce the effective key, so it stays minimal.
            if usage is None and clip.get("cache_key"):
                from . import prompts
                usage = _cached_usage(
                    paths, clip["cache_key"], model,
                    prompts.trajectory_request([], None))

            tokens = _tokens_in_out(usage)
            traj_tokens = _add(traj_tokens, tokens)

            seconds = clip.get("seconds")
            if isinstance(seconds, (int, float)):
                traj_seconds += float(seconds)
                traj_seconds_known = True

            start = float(clip.get("start_s") or 0.0)
            end = float(clip.get("end_s") or 0.0)
            clips.append({
                "index": clip.get("index"),
                "start_s": start,
                "end_s": end,
                "seconds": round(end - start, 1),
                "model": model,
                "tasks": clip.get("tasks"),
                "actions": clip.get("actions"),
                "shots": len(clip.get("shots") or []),
                "reply_chars": clip.get("reply_chars"),
                "cached": bool(clip.get("cached")),
                "tokens": tokens,
                "cost_usd": _cost(model, tokens),
                "elapsed_s": seconds if isinstance(seconds, (int, float)) else None,
                "error": clip.get("error"),
            })

        # ---- actions ------------------------------------------------------
        tasks = traj.get("tasks") or []
        actions = [a for t in tasks for a in t["actions"]]
        by_type: dict = {}
        for a in actions:
            by_type[a["action_type"]] = by_type.get(a["action_type"], 0) + 1

        ground_results = (ground or {}).get("results") or []
        grounded = [r for r in ground_results if r.get("grounded")]
        dropped = [r for r in ground_results if not r.get("grounded")]
        recovered_frames = sum(1 for r in ground_results if r.get("recovered"))

        # ---- grounding tokens --------------------------------------------
        ground_tokens = {"input": None, "output": None, "thinking": None, "total": None}
        ground_seconds = 0.0
        ground_seconds_known = False
        ground_model = (ground or {}).get("model") or ""
        for r in ground_results:
            stamp = {"input": None, "output": None, "thinking": None, "total": None}
            bundle = None
            for attempt in r.get("attempts") or []:
                if attempt.get("usage"):
                    bundle = _add(bundle or stamp, _tokens_in_out(attempt["usage"]))
                if isinstance(attempt.get("seconds"), (int, float)):
                    ground_seconds += float(attempt["seconds"])
                    ground_seconds_known = True
            if bundle is None and r.get("cached") is not None:
                # Older records: the per-frame cache key is not stored, and the
                # frame's bytes are part of it, so it cannot be reproduced without
                # re-extracting the exact same image. Reported as unknown.
                bundle = stamp
            ground_tokens = _add(ground_tokens, bundle or stamp)

        # ---- screening ----------------------------------------------------
        screen = screen_by_id.get(traj.get("video_id") or "")
        if screen is None:
            # Fall back to matching on the title, which is all older records carry.
            for rec in screen_by_id.values():
                if (rec.get("title") or "") and rec.get("title") == traj.get("video"):
                    screen = rec
                    break

        total_tokens = _add(traj_tokens, ground_tokens)
        traj_cost = _cost(traj.get("model") or "", traj_tokens)
        ground_cost = _cost(ground_model, ground_tokens)
        total_cost = (None if traj_cost is None and ground_cost is None
                      else (traj_cost or 0.0) + (ground_cost or 0.0))

        videos.append({
            "stem": stem,
            "video": traj.get("video"),
            "duration_s": traj.get("duration_s"),
            "width": traj.get("width"),
            "height": traj.get("height"),
            "fps": traj.get("fps"),
            "clip_seconds": traj.get("clip_seconds"),
            "trajectory_model": traj.get("model"),
            "grounding_model": ground_model,
            "prompt_version": traj.get("prompt_version"),
            "stats": traj.get("stats") or {},
            "slicing": {
                "clip_count": len(clips),
                "clips": clips,
                "total_clip_seconds": round(
                    sum(c["seconds"] for c in clips), 1),
                # Whether every clip the video needs was actually processed.
                # `None` for records written before this was tracked, which is not
                # the same as False -- an older record may well be complete, and
                # claiming either way would be a guess.
                "truncated": traj.get("truncated"),
                "planned_clips": traj.get("planned_clips"),
                "processed_clips": traj.get("processed_clips"),
                "covered_s": traj.get("covered_s"),
                # Coverage is computed from the clip ranges that ARE recorded, so it
                # is real information for old records too -- not a guess at the
                # `truncated` flag, just the arithmetic of what was processed.
                "covers_pct": (
                    round(100 * sum(c["seconds"] for c in clips)
                          / float(traj["duration_s"]), 1)
                    if clips and traj.get("duration_s") else None
                ),
            },
            "actions": {
                "total": len(actions),
                "by_type": dict(sorted(by_type.items(),
                                       key=lambda kv: -kv[1])),
                "tasks": len(tasks),
                "groundable": sum(1 for a in actions if a["needs_grounding"]),
                "grounded": len(grounded),
                "dropped": len(dropped),
                "yield_pct": (round(100 * len(grounded) / len(ground_results), 1)
                              if ground_results else None),
                "avg_attempts": (ground or {}).get("stats", {}).get("avg_attempts"),
                "recovered_frames": recovered_frames,
                # How many groundable actions existed before any cap was applied,
                # and the cap itself. `attempted` alone cannot say whether the video
                # was short or the run was limited. Both are None for records written
                # before this was tracked, which is not the same as "not limited".
                "available": (ground or {}).get("stats", {}).get("available"),
                "limited_to": (ground or {}).get("stats", {}).get("limited_to"),
            },
            "tokens": {
                "trajectory": traj_tokens,
                "grounding": ground_tokens,
                "total": total_tokens,
                "trajectory_usd": traj_cost,
                "grounding_usd": ground_cost,
                "total_usd": total_cost,
                # Which components are real. A total that silently omits a source
                # reads as complete, and for this run grounding's tokens are
                # genuinely unrecoverable -- so the omission is named instead.
                "unknown": [name for name, bundle in
                            (("trajectory", traj_tokens),
                             ("grounding", ground_tokens))
                            if not _known(bundle)],
            },
            "elapsed": {
                "trajectory_s": round(traj_seconds, 1) if traj_seconds_known else None,
                "grounding_s": round(ground_seconds, 1) if ground_seconds_known else None,
                "recorded": traj_seconds_known or ground_seconds_known,
            },
            "screening": {
                "title": (screen or {}).get("title"),
                "accepted": (screen or {}).get("accepted"),
                "subtitle_chars": (screen or {}).get("subtitle_chars"),
                "subtitle_language": (screen or {}).get("subtitle_language"),
                "subtitle_kind": (screen or {}).get("subtitle_kind"),
                "scores": ((screen or {}).get("quality") or {}).get("scores"),
            } if screen else None,
        })

    # ---- totals ----------------------------------------------------------
    group = {"input": None, "output": None, "thinking": None, "total": None}
    usd = 0.0
    usd_known = False
    unknown: list = []
    truncated: list = []
    limited: list = []
    for v in videos:
        group = _add(group, v["tokens"]["total"])
        if v["tokens"]["total_usd"] is not None:
            usd += v["tokens"]["total_usd"]
            usd_known = True
        for name in v["tokens"].get("unknown") or []:
            entry = "%s/%s" % (v["stem"], name)
            if entry not in unknown:
                unknown.append(entry)
        if v["slicing"].get("truncated"):
            truncated.append({
                "stem": v["stem"],
                "processed": v["slicing"].get("processed_clips"),
                "planned": v["slicing"].get("planned_clips"),
                "covers_pct": v["slicing"].get("covers_pct"),
            })
        # Only when the record actually says so. Absent means the run predates the
        # field, and listing every old run as "limited" would be a fabricated claim.
        if v["actions"].get("limited_to"):
            limited.append({
                "stem": v["stem"],
                "attempted": v["actions"].get("limited_to"),
                "available": v["actions"].get("available"),
            })

    return {
        "work": str(work),
        "videos": videos,
        "video_count": len(videos),
        "totals": {
            "tokens": group,
            "usd": round(usd, 5) if usd_known else None,
            "clips": sum(v["slicing"]["clip_count"] for v in videos),
            "actions": sum(v["actions"]["total"] for v in videos),
            "grounded": sum(v["actions"]["grounded"] for v in videos),
            # Non-empty means the figures above are a lower bound, not a total.
            "unknown_sources": unknown,
            "complete": not unknown,
            # Non-empty means at least one video was only partly processed, so the
            # dataset does not cover all of it.
            "truncated": truncated,
            # Non-empty means grounding was capped for at least one video, so actions
            # that could have grounded are absent from the dataset.
            "limited": limited,
        },
    }
