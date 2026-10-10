"""Orchestration: run the three stages in order, resumably.

Each stage writes its own record to disk and is skipped when that record already
exists and matches the current prompt version. So a crashed run resumes for free,
and re-running with a better grounding model only re-does grounding -- the video
is never re-sent to the expensive trajectory pass.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path

from . import (clients, config, hf_export, log, records, report, stage_dataset,
               stage_grounding, stage_trajectory, video)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv"}


# ============================================================================
# Ingest
# ============================================================================


def find_videos(source: Path) -> list[Path]:
    """Video files under a path, or the path itself if it is one file."""
    source = Path(source)
    if source.is_file():
        return [source]
    out = []
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return out


def ingest(sources: list[Path], paths: config.Paths, copy: bool = True,
           verbose: bool = True) -> list[Path]:
    """Bring source videos into the work tree, then probe them.

    Grounding reads the *original* frames, so the files have to stay reachable
    for the whole run. Copying keeps that true even if the originals are on a
    drive that goes away; the downside is disk, which is cheap here.

    A video that will not probe is reported and skipped now, rather than failing
    an hour later inside stage 2.
    """
    paths.ensure()
    ingested = []
    for src in sources:
        try:
            info = video.probe(src)
        except Exception as exc:
            if verbose:
                print("  [skip] %s: %s" % (src.name, str(exc)[:120]))
            continue

        dest = paths.videos / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            pass
        elif copy:
            shutil.copy2(src, dest)
        else:
            dest = src

        if verbose:
            print("  %s" % info.summary())
            # Reporting here rather than in stage 2 means an oversized source is
            # known before any API spend, not after the grounding calls have
            # already been paid for.
            warning = video.oversize_frame_warning(info)
            if warning:
                print("  [warn] %s" % warning)
        ingested.append(dest)
    return ingested


# ============================================================================
# Stages
# ============================================================================
#
# Reading stage records back from disk is `records.load_dir`. It used to live
# here, but the CLI needs it too, and a driver module is the wrong home for a
# file-format helper.


def run_pipeline(source: Path, paths: config.Paths,
                 traj_model: str | None = None,
                 ground_model: str | None = None,
                 limit_videos: int | None = None,
                 limit_actions: int | None = None,
                 max_clips: int | None = None,
                 clip_seconds: int | None = None,
                 thinking_level: str | None = None,
                 force: bool = False,
                 stages: tuple = ("trajectory", "grounding", "dataset"),
                 cache_dir: Path | None = None,
                 verbose: bool = True) -> dict:
    """Run the pipeline end to end. Returns a summary dict.

    `limit_actions` caps grounding calls per video, which is the only way to
    iterate on the grounding prompt without paying for the whole corpus.

    `thinking_level` applies to stage 1 only. It is the one stage where the
    reasoning budget changes the answer rather than only the cost -- stage 2
    locates a single element in a still frame, so there is nothing for a longer
    deliberation to improve.
    """
    t0 = time.time()
    paths.ensure()

    traj_model = traj_model or config.model_for("trajectory")
    ground_model = ground_model or config.model_for("grounding")

    videos = find_videos(source)
    if limit_videos:
        videos = videos[:limit_videos]
    if not videos:
        raise SystemExit("no video files found under %s" % source)

    if verbose:
        print("=" * 72)
        print("WILCOM Video2GUI pipeline")
        print("  source videos : %d" % len(videos))
        print("  work dir      : %s" % paths.root.resolve())
        print("  trajectory    : %s" % traj_model)
        print("  grounding     : %s" % ground_model)
        print("  stages        : %s" % ", ".join(stages))
        print("=" * 72)
        print()
        print("[ingest]")

    client = clients.make_client(
        cache_dir=cache_dir or (paths.root / "cache"),
        verbose=verbose)
    local = ingest(videos, paths, verbose=verbose)
    if not local:
        raise SystemExit("no usable videos after probing")

    # ------------------------------------------------------------------ stage 1
    if "trajectory" in stages:
        log.step("stage 1 -- trajectory extraction")
        traj = stage_trajectory.run(local, client, paths,
                                    model=traj_model,
                                    clip_seconds=clip_seconds,
                                    max_clips=max_clips,
                                    thinking_level=thinking_level,
                                    force=force, verbose=verbose)
    else:
        traj = records.load_dir(paths.traj, verbose=verbose)
        log.step("stage 1 skipped, using %d cached record(s)" % len(traj))

    # ------------------------------------------------------------------ stage 2
    if "grounding" in stages:
        log.step("stage 2 -- action spatial grounding")
        ground = stage_grounding.run(traj, client, paths,
                                     model=ground_model, force=force,
                                     limit=limit_actions, verbose=verbose)
    else:
        ground = records.load_dir(paths.ground, verbose=verbose)
        log.step("stage 2 skipped, using %d cached record(s)" % len(ground))

    # ------------------------------------------------------------------ stage 3
    log.step("stage 3 -- dataset assembly")
    rows, stats = stage_dataset.build_rows(traj, ground, paths)
    if verbose:
        print(stats.summary())
        if not stats.accounting_ok:
            # Every action should land in exactly one bucket. A mismatch means
            # one vanished without being counted, which is worth surfacing loudly
            # rather than discovering later as a smaller-than-expected dataset.
            print("  [warn] action accounting does not balance")

    if not rows:
        raise SystemExit(
            "no rows were produced. Check the trajectory and grounding records "
            "in %s" % paths.root.resolve())

    splits = {"unified": rows}
    if verbose:
        print()
        for name, part in splits.items():
            if part:
                print("  %-11s %5d rows  %3d trajectories"
                      % (name, len(part),
                         len(set(r["trajectory"] for r in part))))

    problems = stage_dataset.verify(rows, splits)
    fatal = [p for p in problems if not p.startswith("NOTE")]
    for p in problems:
        print("  [%s] %s" % ("FAIL" if not p.startswith("NOTE") else "note", p))

    if "dataset" in stages:
        built = hf_export.save_local(splits, paths)
        if verbose:
            print()
            print("  saved:")
            for name in built:
                print("    %s" % (paths.out / ("%s.parquet" % name)).resolve())
            print("    %s" % (paths.out / "manifest.json").resolve())

        # The local name is `check_path` rather than `report`, which would shadow
        # the `report` module imported above.
        check_path = report.render(rows,
                                   paths.logs / "coordinate_check.html")
        print("    %s  (open this and look at the red circles)"
              % check_path.resolve())

    summary = {
        "trajectory_model": traj_model,
        "grounding_model": ground_model,
        "max_clips": max_clips,
        "videos": len(local),
        "rows": len(rows),
        # asdict, so the summary file holds plain JSON-serialisable data rather
        # than an object json.dumps would refuse.
        "stats": asdict(stats),
        "accounting_ok": stats.accounting_ok,
        "splits": {n: len(p) for n, p in splits.items() if p},
        "fatal_problems": fatal,
        "seconds": round(time.time() - t0, 1),
        "cost": {
            "calls": client.ledger.calls,
            "failures": client.ledger.failures,
            "prompt_tokens": client.ledger.prompt_tokens,
            "completion_tokens": client.ledger.completion_tokens,
            "usd": round(client.ledger.cost_usd, 4),
        },
    }

    records.write_json(summary, paths.logs / "summary.json")

    print()
    print("=" * 72)
    print(client.ledger.report())
    print("elapsed: %.1f s" % summary["seconds"])
    print("=" * 72)

    return summary
