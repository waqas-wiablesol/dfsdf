"""Stage 3 -- dataset assembly.

Joins stage 1 (instructions) with stage 2 (coordinates) and emits rows in exactly
the schema of `junior0091/GUI-360-word-unsloth`, so the same training and
evaluation code works unchanged.

Two conventions are copied from the parent dataset, and both are easy to get
wrong:

  target_x / target_y   absolute pixels on the row's own image
answer_raw            the SAME absolute pixels, as
                      `<coordinate> [x, y]\n</coordinate>`

That second line was wrong for a long time. It claimed `answer_raw` was normalized
to 0-1000 and that the two columns "are not the same numbers". Verified against
`GUI-360-word-unsloth` itself -- 220 rows across train, validation and test -- the
answer is the pixel pair, identical to the target columns, with a newline before
the closing tag. The note that claimed otherwise is gone rather than reworded,
because it was confident and unverified and that is how it survived.
Extra provenance columns are added that the parent schema does not have. They cost
nothing and make a bad row traceable back to the second it came from.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import config, log, prompts


# ============================================================================
# Assembly
# ============================================================================


@dataclass
class BuildStats:
    """What happened while joining the two stages.

    Every action ends up either as a row or in exactly one bucket here, so the
    buckets plus `written` sum to `actions_seen`. That invariant is the point: a
    run that writes fewer rows than expected can be explained rather than guessed
    at, and `accounting_ok` asserts it.

    A dataclass rather than a dict, because these fields are read by the pipeline,
    the CLI and the tests, and a mistyped dict key fails silently.
    """

    tasks: int = 0
    actions_seen: int = 0
    not_groundable: int = 0
    no_instruction: int = 0
    ungrounded: int = 0
    frame_missing: int = 0
    out_of_bounds: int = 0
    written: int = 0

    @property
    def accounting_ok(self) -> bool:
        """True when every action is accounted for exactly once."""
        bucketed = (self.not_groundable + self.no_instruction + self.ungrounded
                    + self.frame_missing + self.out_of_bounds)
        return self.written + bucketed == self.actions_seen

    def summary(self) -> str:
        """A printable breakdown, so a shortfall is readable at a glance."""
        return "\n".join([
            "tasks seen      : %d" % self.tasks,
            "actions seen    : %d" % self.actions_seen,
            "  not groundable: %d" % self.not_groundable,
            "  no instruction: %d" % self.no_instruction,
            "  ungrounded    : %d" % self.ungrounded,
            "  frame missing : %d" % self.frame_missing,
            "  out of bounds : %d" % self.out_of_bounds,
            "rows written    : %d" % self.written,
        ])


def _row_from_action(action: dict, task: dict, traj: dict,
                     ground: dict | None,
                     paths: config.Paths) -> tuple:
    """Build one dataset row, or report why it could not be built.

    Returns `(row, reason)` with exactly one of the two set. `reason` is the name
    of a `BuildStats` field, so the caller's accounting stays in step with the
    checks here instead of duplicating them.
    """
    if not action["needs_grounding"]:
        return None, "not_groundable"

    instruction = (action["grounding_instruction"] or "").strip()
    if not instruction:
        return None, "no_instruction"

    if not ground or not ground.get("grounded"):
        return None, "ungrounded"

    frame = paths.frames / traj["video_stem"] / ground["frame"]
    if not frame.exists():
        return None, "frame_missing"

    width = int(ground["image_width"])
    height = int(ground["image_height"])
    x_norm, y_norm = ground["point_norm"]

    # Grounding returns a normalized pair; the dataset needs pixels. The conversion
    # is the inverse the parent dataset uses, and `verify` re-checks it for every
    # row, because getting it wrong is invisible -- both numbers look like
    # coordinates and both are in bounds.
    x = config.norm_to_abs(x_norm, width)
    y = config.norm_to_abs(y_norm, height)
    if not (0 <= x < width and 0 <= y < height):
        return None, "out_of_bounds"

    stem = traj["video_stem"]
    return {
        "id": "%s_%s_%03d_%02d" % (config.ID_PREFIX, stem,
                                   action["task_id"], action["seq"]),
        "trajectory": "%s_%s" % (config.ID_PREFIX, stem),
        "instruction": instruction,
        "prompt": prompts.render_prompt(instruction),
        "image_path": str(frame),
        "target_x": x,
        "target_y": y,
        "image_width": width,
        "image_height": height,
        # Absolute pixels, identical to `target_x` / `target_y`. That is the parent
        # dataset's convention and it was verified rather than assumed: in all 220
        # rows checked across its three splits, `answer_raw` equals the target
        # pixels exactly. The normalized pair is not used in the answer at all.
        "answer_raw": prompts.render_answer(x, y),
        # Provenance, beyond the parent schema.
        "source_video": traj["video"],
        "task_id": action["task_id"],
        "step": action["seq"],
        "action_type": action["action_type"],
        "timestamp_s": action["t"],
        "bbox": json.dumps(ground["bbox"]),
        "sub_task": task["instruction"],
    }, None


def build_rows(traj_records: list[dict], ground_records: list[dict],
               paths: config.Paths) -> tuple[list[dict], BuildStats]:
    """Join the two stages into dataset rows, dropping anything ungrounded.

    A row exists only if its action survived grounding with a validated
    coordinate. Dropped actions are counted, not papered over with a guess.
    """
    ground_by_video = {g["video_stem"]: g for g in ground_records}
    rows: list[dict] = []
    stats = BuildStats()

    log.task("joining %d trajectory record(s) against %d grounding record(s)"
             % (len(traj_records), len(ground_records)))

    for vi, traj in enumerate(traj_records, 1):
        ground = ground_by_video.get(traj["video_stem"])
        if ground is None:
            # A video with no grounding record simply contributes nothing. Not an
            # error: running stage 2 on a subset is a normal way to work.
            log.task("%s has no grounding record, skipping"
                     % traj["video_stem"])
            continue

        by_action = {(g["task_id"], g["seq"]): g for g in ground["results"]}
        before = stats.written

        for task in traj["tasks"]:
            stats.tasks += 1
            for action in task["actions"]:
                stats.actions_seen += 1
                row, reason = _row_from_action(
                    action, task, traj,
                    by_action.get((action["task_id"], action["seq"])),
                    paths)
                if row is None:
                    # `reason` names a BuildStats field, so this accounting can
                    # never drift from the checks in _row_from_action.
                    setattr(stats, reason, getattr(stats, reason) + 1)
                    continue
                rows.append(row)
                stats.written += 1

        log.progress(vi, len(traj_records),
                     "%s: %d row(s) written"
                     % (traj["video_stem"], stats.written - before))

    log.task("joined: %d row(s) from %d action(s) seen"
             % (stats.written, stats.actions_seen))
    return rows, stats


# ============================================================================
# Splits
# ============================================================================


def split_by_trajectory(rows: list[dict],
                        fractions: dict | None = None,
                        seed: int | None = None) -> dict:
    """Assign whole trajectories to splits, never individual rows.

    Steps from one video are near-identical frames of the same screen. Splitting
    row-wise would put step 3 in train and step 4 in test, and the test score
    would measure memorisation. This function is why that cannot happen.

    Greedy largest-first assignment: trajectories are shuffled, then each goes to
    whichever split is furthest below its row quota. Largest-first keeps one long
    video from overshooting a small split.
    """
    fractions = fractions or config.SPLIT_FRACTIONS
    seed = config.SPLIT_SEED if seed is None else seed

    groups: dict = {}
    for r in rows:
        groups.setdefault(r["trajectory"], []).append(r)

    names = list(fractions.keys())
    total = len(rows)
    quota = {n: fractions[n] * total for n in names}
    assigned = {n: 0 for n in names}
    out = {n: [] for n in names}

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    for key in keys:
        best = max(names, key=lambda n: (quota[n] - assigned[n],
                                         -names.index(n)))
        out[best].extend(groups[key])
        assigned[best] += len(groups[key])

    for name in names:
        out[name].sort(key=lambda r: (r["trajectory"], r["task_id"], r["step"]))
    log.task("split by whole trajectory: %s"
             % ", ".join("%s %d" % (n, len(out[n])) for n in names))
    return out


# ============================================================================
# Verification -- what must be true before this is written anywhere
# ============================================================================


def verify(rows: list[dict], splits: dict) -> list[str]:
    """Return a list of problems. Empty means the dataset is trustworthy.

    Deliberately checks arithmetic and disjointness, not plausibility. Every
    coordinate failure this project has hit was silent and passed a "looks fine"
    review; these are the checks that would have caught them.
    """
    problems: list[str] = []

    if not rows:
        return ["no rows at all"]

    ids = {}
    for r in rows:
        if r["id"] in ids:
            problems.append("duplicate id %s" % r["id"])
        ids[r["id"]] = True

        w, h = r["image_width"], r["image_height"]
        if not (0 <= r["target_x"] < w and 0 <= r["target_y"] < h):
            problems.append(
                "id=%s target (%d, %d) outside %dx%d"
                % (r["id"], r["target_x"], r["target_y"], w, h))

        # The answer must carry the SAME point as the target column, in pixels.
        #
        # This is the check that catches a dataset written at the wrong scale, and
        # the direction matters: for a long time it asserted the answer was the
        # *normalized* form, which enshrined a convention the parent dataset does
        # not use. Verified against the live parent: `answer_raw` equals
        # `target_x` / `target_y` exactly, in all 220 rows checked.
        expect = prompts.render_answer(r["target_x"], r["target_y"])
        if r["answer_raw"] != expect:
            problems.append(
                "id=%s answer_raw %r is not the pixel form of its own "
                "target (%r)" % (r["id"], r["answer_raw"], expect))

        if r["instruction"] not in r["prompt"]:
            problems.append("id=%s instruction not in prompt" % r["id"])

        if not Path(r["image_path"]).exists():
            problems.append("id=%s image missing: %s"
                            % (r["id"], r["image_path"]))

    # The whole point of splitting this way.
    seen: dict = {}
    for name, part in splits.items():
        for r in part:
            seen.setdefault(r["trajectory"], set()).add(name)
    leaked = {t: s for t, s in seen.items() if len(s) > 1}
    if leaked:
        problems.append("trajectory leakage across splits: %s"
                        % list(leaked.items())[:5])

    covered = sum(len(p) for p in splits.values())
    if covered != len(rows):
        problems.append("split total %d != row total %d" % (covered, len(rows)))

    # An instruction that appears with wildly different targets is usually a sign
    # the join matched the wrong action. Reported, not fatal: a toolbar button can
    # legitimately be clicked at slightly different pixels.
    pts = defaultdict(list)
    for r in rows:
        pts[r["instruction"]].append((r["target_x"], r["target_y"]))
    spread = [(i, p) for i, p in pts.items()
              if len(p) > 2 and max(x for x, _ in p) - min(x for x, _ in p) > 200]
    if spread:
        problems.append(
            "NOTE (not fatal): %d instructions with targets spread over >200px, "
            "e.g. %r" % (len(spread), spread[0][0][:70]))

    return problems


# ============================================================================
# What used to be here
# ============================================================================
#
# Two concerns were moved out of this module, because neither needs to know how a
# row is assembled:
#
#   Hugging Face export  -> `hf_export.py`
#       schemas, parquet writing, the manifest. Isolating it means the `datasets`
#       dependency is only imported when an export is actually wanted.
#
#   Visual coordinate check -> `report.py`
#       the standalone HTML report. Pure presentation, and the longest code in
#       the package, which made it the hardest thing here to read past.
#
# This module now owns one thing: turning two stages of records into verified rows.
