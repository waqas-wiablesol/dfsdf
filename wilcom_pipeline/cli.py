"""Command line entry point.

    python -m wilcom_pipeline.cli check --source "C:/videos"
    python -m wilcom_pipeline.cli plan  --source "C:/videos"
    python -m wilcom_pipeline.cli run   --source "C:/videos" --limit-videos 1

`check` and `plan` make no API calls, which makes them the safe way to start: one
proves ffmpeg works on the real files, the other prices the run before it happens.

Global options (`--work`, `--quiet`, `--no-cache`, `--force`) are accepted either
before or after the subcommand. They come from shared parent parsers rather than
being re-declared per subcommand. Re-declaring them meant the second copy silently
overrode the first, so `cli --work A run` and `cli run --work A` behaved
differently depending on which one argparse happened to apply last.

There is deliberately no `ui` subcommand. One used to be listed here pointing at a
module that does not exist, so selecting it produced an ImportError traceback. A
Gradio front end is still planned; it will get a subcommand when it exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (clients, config, cost, hf_export, log, records, report,
               stage_dataset, stage_filter, stage_grounding,
               stage_trajectory, video)
from .pipeline import find_videos, ingest, run_pipeline

# Every stage the pipeline knows, in dependency order. `filter` is the paper's
# Section 3.1 screening and runs before anything touches a video frame.
STAGE_CHOICES = ("filter", "trajectory", "grounding", "dataset")

# The stages a run can be limited to, in dependency order.
STAGE_CHOICES = ("trajectory", "grounding", "dataset")


# ============================================================================
# Shared helpers
# ============================================================================


def _paths(args) -> config.Paths:
    """The work tree, created if needed."""
    return config.Paths(root=Path(args.work)).ensure()


def _flag(args, name: str, default=False):
    """Read a flag that may not exist on the namespace.

    The shared option parsers use `argparse.SUPPRESS` defaults, which means an
    unpassed flag is *absent* from the namespace rather than set to a default.
    `args.force` therefore raises AttributeError when --force was not given. This
    is the one-line consequence of that choice, so it is handled in one place
    rather than with getattr at every call site.
    """
    return getattr(args, name, default)


def _client(args, paths: config.Paths):
    """An API client for this run's cache and verbosity."""
    return clients.make_client(cache_dir=paths.root / "cache",
                               use_cache=not args.no_cache,
                               verbose=not args.quiet)


def _print_splits(splits: dict) -> None:
    """Row and trajectory counts per split, skipping empty ones."""
    for name, part in splits.items():
        if part:
            print("  %-11s %5d rows  %3d trajectories"
                  % (name, len(part), len({r["trajectory"] for r in part})))


def _print_problems(problems: list) -> bool:
    """Print verification output; return True when something is fatal.

    A problem starting with "NOTE" is advisory -- a wide target spread is worth
    reading but is not a reason to withhold a dataset.
    """
    for problem in problems:
        fatal = not problem.startswith("NOTE")
        print("  [%s] %s" % ("FAIL" if fatal else "note", problem))
    return any(not p.startswith("NOTE") for p in problems)


def _probe_all(source: str, quiet: bool = False) -> list:
    """Probe every video under a path, skipping ones that will not probe.

    Shared by `check` and `plan`, which need the same tolerance: a single broken
    file should not stop the other files from being reported on.
    """
    infos = []
    for path in find_videos(Path(source)):
        try:
            infos.append(video.probe(path))
        except Exception as exc:
            if not quiet:
                print("  [skip] %s: %s" % (path.name, str(exc)[:90]))
    return infos


# ============================================================================
# Commands
# ============================================================================


def cmd_screen(args) -> int:
    """Screen one or more URLs with the paper's Prompts 1 and 2.

    Text-only, so this costs a few hundred tokens per candidate rather than a
    video upload. It is the cheapest way to answer "is this worth processing?".
    """
    paths = _paths(args)
    client = _client(args, paths)

    urls = [args.url] if isinstance(args.url, str) else list(args.url)
    records_out = stage_filter.screen(
        urls, client, paths,
        model=args.meta_model or config.model_for("meta"),
        force=_flag(args, "force"), verbose=not args.quiet)

    if not records_out:
        print("nothing was screened")
        return 1

    print()
    log.table(
        [[r.get("video_id") or "?",
          "YES" if r.get("accepted") else "no",
          "%.2f" % (r.get("meta_verdict") or {}).get("confidence", 0),
          " ".join(str((r.get("quality") or {}).get("scores", {}).get(d, "-"))
                   for d in stage_filter.DIMENSIONS),
          str((r.get("meta_verdict") or {}).get("reasoning", ""))[:58]]
         for r in records_out],
        ["video", "keep", "conf", "topic/clarity/rec", "reasoning"])

    print()
    print(client.ledger.report())
    accepted = sum(1 for r in records_out if r.get("accepted"))
    print("\n%s: %d of %d accepted" % (paths.screen, accepted, len(records_out)))
    return 0


def cmd_run(args) -> int:
    """All three stages."""
    paths = _paths(args)

    if args.stages:
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
        unknown = [s for s in stages if s not in STAGE_CHOICES]
        if unknown:
            # Previously a typo here silently ran nothing for that stage, which
            # looks identical to a stage that had no work to do.
            print("unknown stage(s): %s" % ", ".join(unknown))
            print("valid stages    : %s" % ", ".join(STAGE_CHOICES))
            return 2
    else:
        stages = STAGE_CHOICES

    summary = run_pipeline(
        Path(args.source), paths,
        traj_model=args.traj_model,
        ground_model=args.ground_model,
        limit_videos=args.limit_videos,
        limit_actions=args.limit_actions,
        max_clips=_flag(args, "max_clips"),
        clip_seconds=args.clip_seconds,
        thinking_level=_thinking(args) or None,
        force=_flag(args, "force"),
        stages=stages,
        verbose=not args.quiet,
    )
    return 1 if summary["fatal_problems"] else 0


def _thinking(args) -> str:
    """The validated `--thinking` value, or "" for the configured default.

    Validated here rather than in the stage so a typo fails before the API client
    is built and before any video is probed. The UI already offered this setting;
    the CLI did not, which made the thinking level unreachable from a script and
    left the two front ends able to do different things.
    """
    value = (getattr(args, "thinking", "") or "").strip().lower()
    if value and value not in config.THINKING_LEVELS:
        raise SystemExit(
            "unknown thinking level %r. Choose one of: %s\n"
            "  (or omit --thinking to use the default, %r)"
            % (value, ", ".join(config.THINKING_LEVELS), config.THINKING_LEVEL))
    return value


def cmd_trajectory(args) -> int:
    """Stage 1 only."""
    paths = _paths(args)
    client = _client(args, paths)

    videos = ingest(find_videos(Path(args.source)), paths,
                    verbose=not args.quiet)
    if not videos:
        print("no usable videos under %s" % args.source)
        return 1
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    records_written = stage_trajectory.run(
        videos, client, paths,
        model=args.traj_model or config.model_for("trajectory"),
        clip_seconds=args.clip_seconds, force=_flag(args, "force"),
        max_clips=_flag(args, "max_clips"),
        thinking_level=_thinking(args) or None,
        verbose=not args.quiet)

    print()
    print(client.ledger.report())
    return 0 if records_written else 1


def cmd_grounding(args) -> int:
    """Stage 2 only, from cached trajectories."""
    paths = _paths(args)
    client = _client(args, paths)

    trajectories = records.load_dir(paths.traj, verbose=not args.quiet)
    if not trajectories:
        print("no trajectory records in %s" % paths.traj)
        print("run the trajectory stage first")
        return 1

    if args.video:
        trajectories = [t for t in trajectories
                        if t.get("video_stem") == args.video]
        if not trajectories:
            print("no trajectory record for %r" % args.video)
            return 1

    grounded = stage_grounding.run(
        trajectories, client, paths,
        model=args.ground_model or config.model_for("grounding"),
        force=_flag(args, "force"), limit=args.limit_actions, verbose=not args.quiet)

    print()
    print(client.ledger.report())
    return 0 if grounded else 1


def cmd_dataset(args) -> int:
    """Stage 3 only, from cached records."""
    paths = _paths(args)

    trajectories = records.load_dir(paths.traj, verbose=not args.quiet)
    groundings = records.load_dir(paths.ground, verbose=not args.quiet)
    if not trajectories:
        print("no trajectory records in %s" % paths.traj)
        return 1

    rows, stats = stage_dataset.build_rows(trajectories, groundings, paths)
    print(stats.summary())
    if not stats.accounting_ok:
        print("  [warn] action accounting does not balance")
    if not rows:
        print()
        print("no rows produced, so nothing was written")
        return 1

    splits = {"unified": rows}
    print()
    _print_splits(splits)

    problems = stage_dataset.verify(rows, splits)
    print()
    if _print_problems(problems):
        print()
        print("refusing to write a dataset that failed verification")
        return 1

    built = hf_export.save_local(splits, paths)
    print()
    for name in built:
        print("wrote %s" % (paths.out / ("%s.parquet" % name)).resolve())
    check_path = report.render(rows, paths.logs / "coordinate_check.html")
    print("wrote %s  -- open it and look" % check_path.resolve())
    return 0


def cmd_check(args) -> int:
    """Probe the videos and report what the pipeline would do. No API calls."""
    print("ffmpeg  : %s" % video.ffmpeg_path())
    print("ffprobe : %s"
          % (video.ffprobe_path() or "absent (parsing the ffmpeg banner instead)"))
    print()

    infos = _probe_all(args.source, quiet=args.quiet)
    if not infos:
        print("no video could be probed under %s" % args.source)
        return 1

    for info in infos:
        # info.summary() starts with the filename, which is printed separately.
        detail = info.summary().split(None, 1)
        print("  ok   %-44s %s  clips=%d"
              % (info.path.name[:44], detail[-1],
                 len(video.plan_clips(info, args.clip_seconds))))
        warning = video.oversize_frame_warning(info)
        if warning:
            print("       [warn] %s" % warning)

    print()
    print("%d video(s) usable" % len(infos))
    return 0


def cmd_plan(args) -> int:
    """Price a run without making it. No API calls."""
    infos = _probe_all(args.source, quiet=args.quiet)
    if not infos:
        print("no video could be probed under %s, so there is nothing to price"
              % args.source)
        return 1

    print(cost.estimate(infos, clip_seconds=args.clip_seconds).describe())
    return 0


def cmd_report(args) -> int:
    """Rebuild the coordinate-check HTML from whatever is on disk."""
    paths = _paths(args)

    trajectories = records.load_dir(paths.traj, verbose=not args.quiet)
    groundings = records.load_dir(paths.ground, verbose=not args.quiet)
    rows, _ = stage_dataset.build_rows(trajectories, groundings, paths)
    if not rows:
        print("no rows to report; run the dataset stage first")
        return 1

    out_path = Path(args.out) if args.out else paths.logs / "coordinate_check.html"
    # limit=0 means every row. The default is capped because a full report embeds
    # one full-resolution PNG per row and gets large.
    print("wrote %s" % report.render(rows, out_path, limit=args.limit).resolve())
    return 0


# ============================================================================
# Parser
# ============================================================================


def _global_options(with_force: bool) -> argparse.ArgumentParser:
    """A parent parser carrying the global options, for subcommands to inherit.

    Every default here is `argparse.SUPPRESS`, and that is the whole point of this
    function. A subparser does not merge into the main namespace -- it parses into
    a *fresh* one and then copies every attribute across unconditionally:

        subnamespace, arg_strings = subparser.parse_known_args(arg_strings, None)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)

    So a subparser that declares `--work` with a default overwrites whatever the
    user already gave *before* the subcommand. `cli --work X run` would silently
    use `wilcom_work` instead of X, and nothing would report it. SUPPRESS means the
    subparser contributes nothing until the user actually types the flag, which
    makes both `cli --work X run` and `cli run --work X` behave identically.

    The real defaults therefore live once, on the top-level parser.

    `--force` is only on the writing subcommands. It means nothing to `check`,
    `plan` or `report`, which only read, so offering it would be misleading.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--work", metavar="DIR", default=argparse.SUPPRESS,
                        help="working directory (default: wilcom_work)")
    parent.add_argument("--quiet", action="store_true",
                        default=argparse.SUPPRESS, help="print less")
    parent.add_argument("--no-cache", action="store_true",
                        default=argparse.SUPPRESS,
                        help="ignore the on-disk API cache")
    if with_force:
        parent.add_argument("--force", action="store_true",
                            default=argparse.SUPPRESS,
                            help="redo work that is already on disk")
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI."""
    write_options = _global_options(with_force=True)
    read_options = _global_options(with_force=False)

    parser = argparse.ArgumentParser(
        prog="wilcom-pipeline",
        description="Turn curated screen recordings into a GUI-grounding dataset.")

    # The real defaults. The subparser copies inherit the same flags with
    # SUPPRESS defaults; see _global_options for why that is necessary.
    parser.add_argument("--work", metavar="DIR", default="wilcom_work",
                        help="working directory (default: wilcom_work)")
    parser.add_argument("--quiet", action="store_true", help="print less")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the on-disk API cache")

    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    run = sub.add_parser("run", parents=[write_options],
                         help="all three stages, in order")
    run.add_argument("--source", required=True, help="video file or folder")
    run.add_argument("--urls", nargs="*", default=None,
                     help="URLs to screen and download first; implies the "
                          "filter stage")
    run.add_argument("--traj-model", default=None,
                     help="override the stage 1 model")
    run.add_argument("--ground-model", default=None,
                     help="override the stage 2 model")
    run.add_argument("--limit-videos", type=int, default=None,
                     help="process only the first N videos")
    run.add_argument("--limit-actions", type=int, default=None,
                     help="cap grounding calls per video")
    run.add_argument("--clip-seconds", type=int, default=None,
                     help="clip length, default %d s" % config.CLIP_SECONDS)
    run.add_argument("--max-clips", type=int, default=None, dest="max_clips",
                     help="process only the first N clips of each video, e.g. 1 "
                          "for the first four minutes")
    run.add_argument("--thinking", default=None, metavar="LEVEL",
                     help="stage 1 reasoning budget: %s (default: %s)"
                          % (", ".join(config.THINKING_LEVELS),
                             config.THINKING_LEVEL))
    run.add_argument("--stages", default=None,
                     help="comma list, e.g. 'grounding,dataset' to reuse stage 1")
    run.set_defaults(func=cmd_run)

    traj = sub.add_parser("trajectory", parents=[write_options],
                          help="stage 1 only: video -> trajectory records")
    traj.add_argument("--source", required=True, help="video file or folder")
    traj.add_argument("--traj-model", default=None)
    traj.add_argument("--clip-seconds", type=int, default=None)
    traj.add_argument("--max-clips", type=int, default=None, dest="max_clips",
                      help="process only the first N clips of each video")
    traj.add_argument("--limit-videos", type=int, default=None)
    traj.add_argument("--thinking", default=None, metavar="LEVEL",
                      help="reasoning budget: %s (default: %s). Recorded in "
                           "each record, so two runs at different levels can be "
                           "compared and will not reuse each other's output."
                           % (", ".join(config.THINKING_LEVELS),
                              config.THINKING_LEVEL))
    traj.set_defaults(func=cmd_trajectory)

    ground = sub.add_parser("grounding", parents=[write_options],
                            help="stage 2 only, from cached trajectories")
    ground.add_argument("--ground-model", default=None)
    ground.add_argument("--limit-actions", type=int, default=None,
                        help="cap grounding calls per video")
    ground.add_argument("--video", default=None,
                        help="restrict to one video stem")
    ground.set_defaults(func=cmd_grounding)

    dataset = sub.add_parser("dataset", parents=[write_options],
                             help="stage 3 only, from cached records")
    dataset.set_defaults(func=cmd_dataset)

    check = sub.add_parser("check", parents=[read_options],
                           help="probe videos and report; no API calls")
    check.add_argument("--source", required=True)
    check.add_argument("--clip-seconds", type=int, default=None)
    check.set_defaults(func=cmd_check)

    screen = sub.add_parser("screen", parents=[write_options],
                            help="screen URL(s) with the paper's Prompts 1-2, "
                                 "text only")
    screen.add_argument("url", nargs="+", help="one or more video URLs")
    screen.add_argument("--meta-model", default=None,
                        help="model for screening (default: cheapest)")
    screen.set_defaults(func=cmd_screen)

    plan = sub.add_parser("plan", parents=[read_options],
                          help="estimate cost; no API calls")
    plan.add_argument("--source", required=True)
    plan.add_argument("--clip-seconds", type=int, default=None)
    plan.set_defaults(func=cmd_plan)

    rep = sub.add_parser("report", parents=[read_options],
                         help="rebuild the coordinate-check HTML")
    rep.add_argument("--out", default=None,
                     help="output path (default: <work>/logs/coordinate_check.html)")
    rep.add_argument("--limit", type=int, default=200,
                     help="max rows to include; 0 means all")
    rep.set_defaults(func=cmd_report)

    return parser


def main(argv=None) -> int:
    # Line-buffer stdout. When output is piped rather than a console, Python
    # block-buffers it, so a stage that runs for minutes prints nothing at all
    # until it exits -- which reads as a hang and invites killing a run that is
    # working. Measured: a single Pro clip takes 2-4 minutes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = build_parser().parse_args(argv)

    # One switch controls all progress output. Warnings and errors ignore it, so
    # --quiet can never hide a problem.
    log.set_quiet(getattr(args, "quiet", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted. Work already on disk is kept; re-run to resume.")
        return 130
    except SystemExit:
        # Raised deliberately by the config and video modules for setup problems,
        # which already print their own instructions.
        raise
    except Exception as exc:
        print("\nERROR: %s" % exc, file=sys.stderr)
        if "--debug" in (argv if argv is not None else sys.argv):
            raise
        print("(re-run with --debug for the full traceback)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
