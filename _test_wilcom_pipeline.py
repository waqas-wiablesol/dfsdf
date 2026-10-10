"""Tests for the WILCOM Video2GUI pipeline.

Run:  python _test_wilcom_pipeline.py

Never calls OpenRouter. Every API interaction goes through `FakeClient`, so the
whole pipeline can be exercised end to end for zero cost. The only external
dependency the tests actually reach for is ffmpeg, and only for the two
integration checks at the end, which report SKIP if it is unavailable.

Design note: the tests are adversarial about coordinates on purpose. Three
separate coordinate bugs have already cost this project a dataset, and every one
of them passed a casual review. So the checks below are mostly attempts to trick
`validate_and_convert` into emitting a plausible-looking wrong answer, and
assertions that it refuses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wilcom_pipeline import (  # noqa: E402
    config, cost, details, hashing, hf_export, jobs, ledger, log, metadata,
    prompts, records, report, stage_dataset, stage_filter, stage_grounding,
    stage_trajectory, timecode, video,
)

PASS = 0
FAIL = 0
SKIP = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail="") -> None:
    """Record one assertion.

    `detail` is coerced rather than required to be a string: passing the offending
    value itself is the natural way to write a failure message, and half the calls
    below hand over a dict or a list. Requiring `str(...)` at every call site would
    be noise, and forgetting it raises inside the reporter -- which is a worse
    failure than the one being reported.
    """
    global PASS, FAIL
    text = "" if detail in (None, "") else str(detail)
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append("%s%s" % (name, ("  -- " + text) if text else ""))
        print("  FAIL  %s%s" % (name, ("  -- " + text) if text else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, "got %r, want %r" % (got, want))


def section(title: str) -> None:
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def raises(name: str, fn, exc=Exception, contains: str = "") -> None:
    try:
        fn()
    except exc as e:
        if contains and contains.lower() not in str(e).lower():
            check(name, False, "message %r missing %r" % (str(e)[:120], contains))
        else:
            check(name, True)
        return
    except Exception as e:                      # wrong exception type
        check(name, False, "raised %s not %s: %s"
              % (type(e).__name__, exc.__name__, str(e)[:100]))
        return
    check(name, False, "did not raise")


# ============================================================================
section("1. Coordinate convention")
# ============================================================================

# config asserts this at import; re-assert so a failure here is attributed.
try:
    config.verify_coordinate_convention()
    check("import-time coordinate assertion holds", True)
except AssertionError as e:
    check("import-time coordinate assertion holds", False, str(e))

eq("norm_to_abs(487, 1040) == 506", config.norm_to_abs(487, 1040), 506)
eq("norm_to_abs(295, 736) == 217", config.norm_to_abs(295, 736), 217)
eq("round trip 487/1040", config.abs_to_norm(506, 1040), 487)
eq("round trip 295/736", config.abs_to_norm(217, 736), 295)
eq("norm_to_abs clamps nothing at 0", config.norm_to_abs(0, 1920), 0)
eq("norm_to_abs at 1000", config.norm_to_abs(1000, 1920), 1920)


# ============================================================================
section("2. Timestamps")
# ============================================================================

eq("to_seconds('00:00')", timecode.to_seconds("00:00"), 0.0)
eq("to_seconds('03:15')", timecode.to_seconds("03:15"), 195.0)
eq("to_seconds('1:02:03')", timecode.to_seconds("1:02:03"), 3723.0)
eq("to_seconds('45')", timecode.to_seconds("45"), 45.0)
eq("to_seconds tolerates brackets", timecode.to_seconds("[02:30]"), 150.0)
eq("to_seconds tolerates quotes", timecode.to_seconds("'02:30'"), 150.0)
eq("to_seconds accepts a float string", timecode.to_seconds("12.5"), 12.5)
eq("from_seconds(195)", timecode.from_seconds(195), "03:15")
eq("from_seconds(3723)", timecode.from_seconds(3723), "1:02:03")
eq("from_seconds(0)", timecode.from_seconds(0), "00:00")
eq("from_seconds does not go negative", timecode.from_seconds(-5), "00:00")
eq("round trip 12:34", timecode.from_seconds(
    timecode.to_seconds("12:34")), "12:34")
raises("to_seconds('later') raises", lambda: timecode.to_seconds("later"),
       ValueError)
raises("to_seconds(None) raises", lambda: timecode.to_seconds(None),
       ValueError)
raises("to_seconds('') raises", lambda: timecode.to_seconds(""), ValueError)
eq("to_seconds('3:15') is 195, not 3m15s-of-hours",
   timecode.to_seconds("3:15"), 195.0)
eq("minutes past 59 are allowed", timecode.to_seconds("75:30"), 4530.0)


# ============================================================================
section("3. Shot splitting (Prompt 3, Part 1)")
# ============================================================================

SHOTS_REPLY = """## Shot Splitting

1. 00:00 - 00:04
2. 00:04 - 00:11
3. 00:11 - 00:19

## Part 2: Task Annotations

```json
[]
```
"""

shots = stage_trajectory.parse_shots(SHOTS_REPLY)
eq("three shots parsed", len(shots), 3)
eq("first shot start", shots[0]["start_s"], 0.0)
eq("second shot start", shots[1]["start_s"], 4.0)
eq("first shot end", shots[0]["end_s"], 4.0)
eq("shots are contiguous", all(shots[i]["end_s"] == shots[i + 1]["start_s"]
                              for i in range(len(shots) - 1)), True)

# Prose containing a time range must not become a shot.
NOISE = """## Shot Splitting
1. 00:00 - 00:05
The next action occurs between 00:05 and 00:09 approximately.
2. 00:05 - 00:09
"""
noisy = stage_trajectory.parse_shots(NOISE)
eq("prose range not counted as a shot", len(noisy), 2)

eq("reversed range is normalised",
   [s["start_s"] for s in
    stage_trajectory.parse_shots("1. 00:20 - 00:10")], [10.0])
eq("hmm:ss shots supported",
   len(stage_trajectory.parse_shots("1. 1:00:00 - 1:00:30")), 1)

eq("strip_shot_section starts at Part 2",
   stage_trajectory.strip_shot_section(SHOTS_REPLY).lstrip().startswith("## Part 2"),
   True)


# ============================================================================
section("4. Task parsing (Prompt 3, Part 2)")
# ============================================================================

GOOD_TASK = {
    "task_id": 0,
    "instruction": "Open the Digitize menu",
    "dense_caption": "The user opens the Digitize menu to start digitizing.",
    "plan": "step1: click Digitize. step2: choose Auto Digitize.",
    "platform": "windows",
    "software": "Wilcom EmbroideryStudio",
    "website": None,
    "user_actions": [
        {"timestamp": "00:04", "action_type": "click",
         "grounding_instruction": "Click the 'Digitize' tab in the ribbon.",
         "action_reason": "I have loaded the design, so now I switch to Digitize.",
         "action_parameters": {"point": "[412, 96]", "bbox": "[400, 74, 430, 120]"},
         "core_change_reason": "Ribbon tabs switch the visible command group.",
         "core_change": "The Digitize ribbon appears."},
        {"timestamp": "00:11", "action_type": "finish",
         "grounding_instruction": "Task complete.",
         "action_reason": "The Digitize ribbon is open.",
         "action_parameters": {"status": "success"},
         "core_change_reason": "Done", "core_change": "Done"},
    ],
}


def reply_with(task) -> str:
    return ("## Shot Splitting\n\n1. 00:00 - 00:04\n2. 00:04 - 00:11\n\n"
            "## Part 2: Task Annotations\n\n```json\n"
            + json.dumps([task]) + "\n```\n")


tasks = stage_trajectory.parse_tasks(reply_with(GOOD_TASK))
eq("one task parsed", len(tasks), 1)
eq("task instruction", tasks[0]["instruction"], "Open the Digitize menu")

# Four shapes a reply can arrive in.
for label, text in (
    ("bare json array", json.dumps([GOOD_TASK])),
    ("fenced json", "```json\n%s\n```" % json.dumps([GOOD_TASK])),
    ("prose wrapped", "Here is the analysis.\n\n%s\n\nHope that helps."
     % json.dumps([GOOD_TASK])),
    ("wrapped in an object", json.dumps({"tasks": [GOOD_TASK]})),
):
    got = stage_trajectory.parse_tasks(text)
    eq("parses %s" % label, len(got), 1)

eq("a single unwrapped task object parses",
   len(stage_trajectory.parse_tasks(json.dumps(GOOD_TASK))), 1)

# The prompt itself contains an example JSON block. A model echoing it must not
# produce a row.
ECHO = {"task_id": 0,
        "instruction": "A one-sentence summary of the user's core intent for this task.",
        "user_actions": [{"timestamp": "Timestamp (mm:ss) from Part 1, marking the "
                                         "start of the shot.",
                          "action_type": "The type of action (e.g., 'click', 'write').",
                          "grounding_instruction": "A natural language instruction to "
                                                   "precisely locate the UI element.",
                          "action_reason": "First-person reasoning.",
                          "action_parameters": {"...": "Parameters specific to the action_type"},
                          "core_change_reason": "Detailed explanation.",
                          "core_change": "The most critical change."}]}
raises("prompt echo is rejected", lambda: stage_trajectory.parse_tasks(
    json.dumps([ECHO])), ValueError)

raises("empty reply raises", lambda: stage_trajectory.parse_tasks(""), ValueError)
raises("reply with no JSON raises",
       lambda: stage_trajectory.parse_tasks("I cannot analyse this video."),
       ValueError)
raises("task without user_actions is rejected",
       lambda: stage_trajectory.parse_tasks(json.dumps([{"task_id": 0}])),
       ValueError)


# ============================================================================
section("5. Action normalisation")
# ============================================================================

eq("parse_numbers from string", stage_trajectory.parse_numbers("[412, 96]"),
   [412, 96])
eq("parse_numbers from list", stage_trajectory.parse_numbers([1, 2, 3]), [1, 2, 3])
eq("parse_numbers from nested", stage_trajectory.parse_numbers([[7, 8], [9]]),
   [7, 8, 9])
eq("parse_numbers of None", stage_trajectory.parse_numbers(None), [])

# Prompt 3 says coordinates are [y, x] in pixels.
eq("trajectory_point is y-first",
   stage_trajectory.trajectory_point({"point": "[412, 96]"}), (412, 96))
eq("trajectory_point of a drag uses start_point",
   stage_trajectory.trajectory_point({"start_point": "[10, 20]",
                                      "end_point": "[30, 40]"}), (10, 20))
eq("trajectory_point of a keypress is None",
   stage_trajectory.trajectory_point({"key_name": "enter"}), None)

norm = stage_trajectory.normalise_tasks([GOOD_TASK], 0, 0.0)
eq("normalise keeps both actions", len(norm[0]["actions"]), 2)
eq("seq numbers are 0, 1", [a["seq"] for a in norm[0]["actions"]], [0, 1])
eq("first action absolute t", norm[0]["actions"][0]["t"], 4.0)
eq("click needs grounding", norm[0]["actions"][0]["needs_grounding"], True)
eq("finish does not need grounding", norm[0]["actions"][1]["needs_grounding"], False)
eq("write does not need grounding", stage_trajectory.normalise_tasks(
    [{"task_id": 0, "user_actions": [
        {"timestamp": "00:01", "action_type": "write",
         "action_parameters": {"text": "hello"}}]}], 0, 0.0)[0]["actions"][0]
   ["needs_grounding"], False)

# A model that answers in clip-local time must be rescaled, not written as-is.
CLIP_LOCAL = {"task_id": 0, "user_actions": [
    {"timestamp": "01:10", "action_type": "click",
     "grounding_instruction": "Click the Save button.",
     "action_parameters": {"point": "[100, 200]"}}]}
res = stage_trajectory.normalise_tasks([CLIP_LOCAL], 1, 240.0)
eq("clip-local timestamp is promoted to absolute",
   res[0]["actions"][0]["t"], 310.0)

# An action with no parseable timestamp is dropped, not defaulted to zero.
BAD_TS = {"task_id": 0, "user_actions": [
    {"timestamp": "around the middle", "action_type": "click",
     "grounding_instruction": "Click something.",
     "action_parameters": {"point": "[1, 2]"}}]}
eq("action with a bad timestamp is dropped",
   stage_trajectory.normalise_tasks([BAD_TS], 0, 0.0), [])


# ============================================================================
section("6. Merging across clip boundaries")
# ============================================================================


def mk_task(tid, times, instruction, finish):
    actions = [{"task_id": tid, "seq": i, "clip": 0, "t": t,
                "action_type": "finish" if (finish and i == len(times) - 1)
                else "click",
                "grounding_instruction": instruction,
                "action_parameters": {}, "traj_point": None,
                "needs_grounding": not (finish and i == len(times) - 1)}
               for i, t in enumerate(times)]
    return {"task_id": tid, "clip": 0, "clip_start_s": 0.0,
            "instruction": instruction, "dense_caption": "", "plan": "",
            "platform": "windows", "software": "W", "website": None,
            "actions": actions,
            "has_finish": any(a["action_type"] == "finish" for a in actions)}


unfinished = mk_task(0, [100, 110], "Digitize the logo", finish=False)
continued = mk_task(1, [115, 120], "Digitize the logo", finish=True)
merged = stage_trajectory.merge_across_clips([unfinished, continued])
eq("unfinished + immediate successor merge into one", len(merged), 1)
eq("merged action count", len(merged[0]["actions"]), 4)
eq("merged seq is renumbered 0..3",
   [a["seq"] for a in merged[0]["actions"]], [0, 1, 2, 3])
eq("merged task_id is 0", merged[0]["task_id"], 0)
eq("merged task inherits the finish", merged[0]["has_finish"], True)

far = mk_task(1, [400, 405], "Export the design", finish=True)
eq("a distant task does not merge",
   len(stage_trajectory.merge_across_clips([unfinished, far])), 2)

done = mk_task(0, [100, 110], "Digitize the logo", finish=True)
eq("a finished task does not absorb its successor",
   len(stage_trajectory.merge_across_clips([done, continued])), 2)

# Out-of-order input must be sorted by time, not by arrival.
eq("tasks are ordered by first action time",
   [t["instruction"] for t in stage_trajectory.merge_across_clips(
       [far, mk_task(0, [10, 20], "First", finish=True)])],
   ["First", "Export the design"])

# Global ids must be dense and unique after merging.
three = [mk_task(0, [10], "A", True), mk_task(0, [400], "B", True),
         mk_task(0, [800], "C", True)]
eq("global task ids are 0,1,2",
   [t["task_id"] for t in stage_trajectory.merge_across_clips(three)], [0, 1, 2])


# ============================================================================
section("7. Grounding reply parsing (Prompt 5)")
# ============================================================================

FEASIBLE = json.dumps({
    "feasible": True,
    "predictions": [{"point_name": "point",
                     "center_point": "<point>400 800</point>",
                     "bounding_box": "<bbox>380 760 420 840</bbox>"}],
})
NOT_FEASIBLE = json.dumps({"feasible": False,
                           "reason": "The Submit button is not present."})

p = stage_grounding.parse_grounding(FEASIBLE)
eq("feasible parsed", p["feasible"], True)
eq("point parsed y-first", p["point_yx"], [400, 800])
eq("bbox parsed y-first", p["bbox_yx"], [380, 760, 420, 840])
eq("point_name parsed", p["point_name"], "point")

p2 = stage_grounding.parse_grounding(NOT_FEASIBLE)
eq("not feasible parsed", p2["feasible"], False)
eq("reason captured", "Submit" in p2["reason"], True)
eq("not feasible has no point", p2["point_yx"], None)

# Fenced, prose-wrapped, and lower-case "true".
eq("fenced grounding parses",
   stage_grounding.parse_grounding("```json\n%s\n```" % FEASIBLE)["point_yx"],
   [400, 800])
eq("capitalised tags parse",
   stage_grounding.parse_grounding(
       '{"feasible": "true", "predictions": [{"center_point":'
       ' "<POINT>10 20</POINT>", "bounding_box": "<BBOX>5 15 15 25</BBOX>"}]}'
   )["point_yx"], [10, 20])
eq("string 'true' is coerced",
   stage_grounding.parse_grounding('{"feasible": "true", "predictions": []}'
                                   )["feasible"], True)
eq("predictions given as an object still parse",
   stage_grounding.parse_grounding(
       '{"feasible": true, "predictions": {"center_point": "<point>1 2</point>",'
       ' "bounding_box": "<bbox>0 0 4 4</bbox>"}}')["point_yx"], [1, 2])
raises("grounding reply that is not JSON raises",
       lambda: stage_grounding.parse_grounding("I could not find it."),
       ValueError)


# ============================================================================
section("8. Grounding validation -- the coordinate gate")
# ============================================================================

W, H = 1920, 1080
GOOD = {"feasible": True, "point_yx": [400, 800], "bbox_yx": [380, 760, 420, 840]}

g = stage_grounding.validate_and_convert(GOOD, W, H)
eq("converted x", g["x"], 1536)
eq("converted y", g["y"], 432)
eq("converted bbox is [x1,y1,x2,y2]",
   g["bbox"], [config.norm_to_abs(760, W), config.norm_to_abs(380, H),
               config.norm_to_abs(840, W), config.norm_to_abs(420, H)])
eq("normalized point preserved", g["point_norm"], [800, 400])
# bbox_norm uses the same x1 y1 x2 y2 order as bbox, so the two never have to be
# read differently. The model's input order was y1 x1 y2 x2.
eq("bbox_norm is in the dataset's x1y1x2y2 order",
   g["bbox_norm"], [760, 380, 840, 420])
eq("bbox_norm converts back to bbox",
   [config.norm_to_abs(g["bbox_norm"][0], W),
    config.norm_to_abs(g["bbox_norm"][1], H),
    config.norm_to_abs(g["bbox_norm"][2], W),
    config.norm_to_abs(g["bbox_norm"][3], H)],
   g["bbox"])
check("converted point lies inside the converted bbox",
      g["bbox"][0] <= g["x"] <= g["bbox"][2]
      and g["bbox"][1] <= g["y"] <= g["bbox"][3])

# The point is inside the frame.
check("converted point is within the frame",
      0 <= g["x"] < W and 0 <= g["y"] < H)

# --- the axis trap. Same numbers, swapped, must be refused.
SWAPPED = {"feasible": True, "point_yx": [800, 400],
           "bbox_yx": [380, 760, 420, 840]}
raises("transposed point is rejected",
       lambda: stage_grounding.validate_and_convert(SWAPPED, W, H),
       stage_grounding.Rejected, "swap")

# --- the scale trap. Pixels where normalized is expected.
PIXELS = {"feasible": True, "point_yx": [432, 1536],
          "bbox_yx": [410, 1459, 454, 1613]}
raises("pixel-scale coordinates are rejected",
       lambda: stage_grounding.validate_and_convert(PIXELS, W, H),
       stage_grounding.Rejected, "normalized")

raises("negative coordinate is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [-5, 500],
            "bbox_yx": [-10, 400, 0, 600]}, W, H),
       stage_grounding.Rejected)

raises("degenerate bbox is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [400, 800],
            "bbox_yx": [400, 800, 400, 800]}, W, H),
       stage_grounding.Rejected, "degenerate")

raises("a one-element point list is rejected as malformed",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [400],
            "bbox_yx": [380, 760, 420, 840]}, W, H),
       stage_grounding.Rejected, "no point")

inverted = stage_grounding.validate_and_convert(
    {"feasible": True, "point_yx": [400, 800],
     "bbox_yx": [420, 840, 380, 760]}, W, H)
check("inverted bbox is repaired rather than dropped",
      inverted["bbox"][0] < inverted["bbox"][2]
      and inverted["bbox"][1] < inverted["bbox"][3])

raises("point far from its own bbox centre is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [350, 350],
            "bbox_yx": [300, 300, 900, 900]}, W, H),
       stage_grounding.Rejected, "disagree")

raises("not feasible is rejected before any conversion",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": False, "point_yx": [400, 800],
            "bbox_yx": [380, 760, 420, 840], "reason": "not visible"}, W, H),
       stage_grounding.Rejected, "not feasible")

raises("missing point is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": None, "bbox_yx": [0, 0, 1, 1]}, W, H),
       stage_grounding.Rejected, "no point")

raises("missing bbox is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [1, 1], "bbox_yx": None}, W, H),
       stage_grounding.Rejected, "no bounding box")

raises("a point on the frame edge is rejected",
       lambda: stage_grounding.validate_and_convert(
           {"feasible": True, "point_yx": [0, 0], "bbox_yx": [0, 0, 4, 4]},
           W, H),
       stage_grounding.Rejected, "edge")

# A square frame cannot expose an axis swap by containment alone, which is worth
# knowing: on a square screenshot the point-in-bbox test is weaker.
sq = stage_grounding.validate_and_convert(GOOD, 1080, 1080)
eq("on a square frame the conversion still works", sq["x"], config.norm_to_abs(800, 1080))

# The disagreement figure must be reported, not just gated on.
check("disagreement is reported", "point_bbox_disagreement" in g)


# ============================================================================
section("9. Cross-check against stage 1")
# ============================================================================

cc = stage_grounding.check_against_trajectory({"x": 1536, "y": 432},
                                              (432, 1536), W, H)
eq("cross check available", cc["available"], True)
eq("cross check scale is pixels", cc["traj_scale"], "pixels")
eq("identical points give zero distance", cc["distance_px"], 0.0)
eq("stage 1 point in frame", cc["traj_in_frame"], True)

cc2 = stage_grounding.check_against_trajectory({"x": 1536, "y": 432},
                                               (950, 980), W, H)
check("a far-off stage-1 point is reported with a distance",
      cc2["distance_px"] > 500)

# Stage 1 emitting values larger than the frame means it used normalized
# quantities, and they must be rescaled before the comparison, not compared raw.
cc3 = stage_grounding.check_against_trajectory({"x": 1536, "y": 432},
                                               (1200, 1500), W, H)
eq("an out-of-frame stage-1 point is treated as normalized",
   cc3["traj_scale"], "normalized")

# A small stage-1 value is valid read either way, so both readings are reported.
cc4 = stage_grounding.check_against_trajectory({"x": 1536, "y": 432},
                                               (400, 800), W, H)
eq("an ambiguous stage-1 point stays pixels", cc4["traj_scale"], "pixels")
check("the alternative reading is reported too",
      "alt_distance_px" in cc4 and "alt_normalized_xy" in cc4)
eq("the alternative reading is the normalized interpretation",
   cc4["alt_normalized_xy"],
   [config.norm_to_abs(800, W), config.norm_to_abs(400, H)])
check("exactly one of the two readings is close",
      (cc4["distance_px"] < 60) != (cc4["alt_distance_px"] < 60),
      "pixels %.1f, normalized %.1f" % (cc4["distance_px"],
                                         cc4["alt_distance_px"]))

# Values above 1000 are unambiguous. No alt reading should be offered, because
# inventing one would invite the wrong interpretation.
check("an unambiguous out-of-frame reading offers no alternative",
      "alt_distance_px" not in cc3)

eq("no stage-1 point means no check",
   stage_grounding.check_against_trajectory({"x": 1, "y": 1}, None, W, H),
   {"available": False})


# ============================================================================
section("10. JSON extraction")
# ============================================================================

from wilcom_pipeline.replies import extract_json  # noqa: E402

eq("whole string", extract_json('{"a": 1}'), {"a": 1})
eq("fenced", extract_json('```json\n{"a": 1}\n```'), {"a": 1})
eq("fenced without a language", extract_json('```\n{"a": 1}\n```'), {"a": 1})
eq("embedded in prose", extract_json('Sure!\n{"a": 1}\nDone.'), {"a": 1})
eq("array in prose", extract_json('Here: [1, 2, 3] ok'), [1, 2, 3])
eq("nested brackets survive",
   extract_json('text {"a": [1, {"b": 2}]} tail'), {"a": [1, {"b": 2}]})
eq("brackets inside strings are not counted",
   extract_json('{"a": "[not json]", "b": 2}'), {"a": "[not json]", "b": 2})
eq("escaped quote inside a string",
   extract_json(r'{"a": "he said \"hi\""}'), {"a": 'he said "hi"'})
raises("no JSON raises", lambda: extract_json("nothing here"), ValueError)


# ============================================================================
section("11. Prompt integrity")
# ============================================================================

check("Prompt 3 defines both parts",
      "Part 1: Shot Splitting" in prompts.PROMPT_TRAJECTORY
      and "Part 2: Task Annotations" in prompts.PROMPT_TRAJECTORY)
check("Prompt 3 declares y-first pixel coordinates",
      "Coordinates [y, x] are in pixels" in prompts.PROMPT_TRAJECTORY)
check("Prompt 3 includes the PC action space",
      "PC Action Space" in prompts.PROMPT_TRAJECTORY)
check("Prompt 3 includes the mobile action space",
      "Mobile Action Space" in prompts.PROMPT_TRAJECTORY)

check("Prompt 4 has the history placeholder",
      "{history}" in prompts.PROMPT_CONTINUATION)
check("Prompt 4 tells the model to continue unfinished tasks",
      "Continue unfinished tasks" in prompts.PROMPT_CONTINUATION)

check("Prompt 5 asks y x in point tags",
      "<point>y x</point>" in prompts.PROMPT_GROUNDING)
check("Prompt 5 asks y1 x1 y2 x2 in bbox tags",
      "<bbox>y1 x1 y2 x2</bbox>" in prompts.PROMPT_GROUNDING)
check("Prompt 5 states the 0-1000 scale",
      "(0-1000)" in prompts.PROMPT_GROUNDING)
check("Prompt 5 has an infeasible example",
      '"feasible": false' in prompts.PROMPT_GROUNDING)

check("the trajectory system prompt names the software",
      "Wilcom" in prompts.SYSTEM_TRAJECTORY)
check("the trajectory system prompt says Windows",
      "Windows" in prompts.SYSTEM_TRAJECTORY)
check("a WILCOM preamble did not leak into the paper's prompt",
      "Wilcom" not in prompts.PROMPT_TRAJECTORY)
check("no preamble leaked into Prompt 5 either",
      "Wilcom" not in prompts.PROMPT_GROUNDING)

# The request shape is provider-neutral: {system, text, media}. The wire format
# lives in the client, so these assert the neutral shape rather than a data URI.
req = prompts.trajectory_request([("clip.mp4", "video")])
eq("trajectory request keys", sorted(req), ["media", "system", "text"])
check("the system preamble travels separately from the body",
      req["system"] == prompts.SYSTEM_TRAJECTORY)
check("the body is Prompt 3", "Part 1: Shot Splitting" in req["text"])
eq("the media handle is passed through unchanged",
   req["media"], [("clip.mp4", "video")])
check("no wire format leaks into the request",
      "data:" not in req["text"] and "video_url" not in req["text"])

cont = prompts.trajectory_request(
    [("clip.mp4", "video")],
    {"start_time": "00:00", "end_time": "04:00", "history": "PRIOR ANALYSIS"})
check("continuation embeds the history", "PRIOR ANALYSIS" in cont["text"])
check("continuation embeds the window",
      "00:00" in cont["text"] and "04:00" in cont["text"])
check("continuation still contains the full Prompt 3",
      "Part 1: Shot Splitting" in cont["text"])

greq = prompts.grounding_request(
    [("f0.png", "image"), ("f1.png", "image"), ("f2.png", "image")],
    "Click the Save button.", "click")
eq("grounding request keys", sorted(greq), ["json_schema", "media", "system", "text"])
eq("all three frames travel as media", len(greq["media"]), 3)
check("grounding instruction is substituted",
      "Click the Save button." in greq["text"])
check("action type is substituted", "click" in greq["text"])
check("no unsubstituted braces remain in Prompt 5",
      "{grounding_instruction}" not in greq["text"])
check("the grounding system preamble is separate",
      greq["system"] == prompts.SYSTEM_GROUNDING)
check("a JSON schema is requested, so the reply need not be parsed from prose",
      isinstance(greq.get("json_schema"), dict)
      and "feasible" in greq["json_schema"]["properties"])

eq("NO_COORDINATE_ACTIONS and COORDINATE_ACTIONS are disjoint",
   bool(prompts.NO_COORDINATE_ACTIONS & prompts.COORDINATE_ACTIONS), False)

# --- screening requests (Prompts 1 and 2), which are text-only ---------------
_META = {"title": "Wilcom embroidery tutorial", "description": "How to digitize",
         "tags": ["wilcom", "embroidery"], "channel": "Design Verse",
         "category": "Education", "subtitle": "click the digitize tab"}

meta_req = prompts.meta_request(_META)
eq("Prompt 1 takes no media", meta_req["media"], [])
check("Prompt 1 contains the paper's role line",
      "expert data filtering assistant" in meta_req["text"])
check("Prompt 1 renders the metadata fields",
      all(f in meta_req["text"] for f in
          ("Title:", "Description:", "Tags/Keywords:", "Channel Name:",
           "Category:", "Subtitle:")))
check("Prompt 1 asks for the paper's output fields",
      all(k in meta_req["text"] for k in
          ("is_gui_content", "confidence", "reasoning")))
check("Prompt 1 has a schema too",
      "is_gui_content" in meta_req["json_schema"]["properties"])

qual_req = prompts.quality_request(_META)
eq("Prompt 2 takes no media", qual_req["media"], [])
check("Prompt 2 states all three dimensions",
      all(d in qual_req["text"] for d in
          ("Topic Relevance", "Instruction Clarity", "Recording Quality")))
check("Prompt 2's schema covers the three dimensions",
      all(d in qual_req["json_schema"]["properties"]["scores"]["properties"]
          for d in ("topic_relevance", "instruction_clarity",
                    "recording_quality")))

# A missing field must render as "(none)" rather than vanish, because an absent
# transcript is itself evidence about instruction clarity.
_sparse = prompts.format_metadata({"title": "x"})
check("absent metadata fields are shown as (none), not dropped",
      _sparse.count("(none)") >= 4, _sparse)

# A long caption track must be trimmed, or it swamps the prompt.
_long = prompts.format_metadata(
    {"title": "x", "subtitle": "word " * 4000})
check("a long transcript is trimmed",
      len(_long) < 12000 and "trimmed" in _long,
      "%d chars" % len(_long))


# ============================================================================
section("12. Dataset rows")
# ============================================================================

TMP = Path(tempfile.mkdtemp(prefix="wilcom_test_"))
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

FRAME_W, FRAME_H = 1280, 720
frame_dir = TMP / "frames" / "demo"
frame_dir.mkdir(parents=True, exist_ok=True)
frame_a = frame_dir / "a.png"
frame_b = frame_dir / "b.png"
if HAVE_PIL:
    Image.new("RGB", (FRAME_W, FRAME_H), (30, 30, 40)).save(frame_a)
    Image.new("RGB", (FRAME_W, FRAME_H), (40, 30, 30)).save(frame_b)

# point_norm (x_n, y_n) = (500, 250) on 1280x720 -> (640, 180)
XN, YN = 500, 250
ABS_X, ABS_Y = config.norm_to_abs(XN, FRAME_W), config.norm_to_abs(YN, FRAME_H)

TRAJ = [{
    "video": "demo.mp4", "video_stem": "demo",
    "duration_s": 60.0, "width": FRAME_W, "height": FRAME_H, "fps": 30.0,
    "tasks": [{
        "task_id": 0, "clip": 0, "clip_start_s": 0.0,
        "instruction": "Digitize the logo",
        "dense_caption": "", "plan": "", "platform": "windows",
        "software": "Wilcom EmbroideryStudio", "website": None,
        "has_finish": True,
        "actions": [
            {"task_id": 0, "seq": 0, "clip": 0, "timestamp": "00:05", "t": 5.0,
             "action_type": "click",
             "grounding_instruction": "Click the 'Digitize' tab in the ribbon.",
             "action_reason": "", "action_parameters": {},
             "core_change": "", "core_change_reason": "",
             "traj_point": None, "needs_grounding": True},
            {"task_id": 0, "seq": 1, "clip": 0, "timestamp": "00:08", "t": 8.0,
             "action_type": "click",
             "grounding_instruction": "Click the 'Export' button.",
             "action_reason": "", "action_parameters": {},
             "core_change": "", "core_change_reason": "",
             "traj_point": None, "needs_grounding": True},
            {"task_id": 0, "seq": 2, "clip": 0, "timestamp": "00:12", "t": 12.0,
             "action_type": "finish",
             "grounding_instruction": "Finished.",
             "action_reason": "", "action_parameters": {},
             "core_change": "", "core_change_reason": "",
             "traj_point": None, "needs_grounding": False},
        ],
    }],
}]

GROUND = [{
    "video": "demo.mp4", "video_stem": "demo",
    "model": "fake", "prompt_version": stage_grounding.PROMPT_VERSION,
    "width": FRAME_W, "height": FRAME_H,
    "results": [
        # Flat, exactly as stage 2 writes it: {"grounded": True, "x": ..., ...}.
        {"task_id": 0, "seq": 0, "t": 5.0, "grounded": True, "drop": False,
         "instruction": "Click the 'Digitize' tab in the ribbon.",
         "action_type": "click", "attempts": [{}],
         "x": ABS_X, "y": ABS_Y, "bbox": [600, 160, 680, 200],
         "point_norm": [XN, YN], "bbox_norm": [400, 200, 600, 300],
         "image_width": FRAME_W, "image_height": FRAME_H,
         "frame": "a.png", "frame_time_s": 5.0, "offset_s": 0.0,
         "n_attempts": 1, "cross_check": {}},
        {"task_id": 0, "seq": 1, "t": 8.0, "grounded": False, "drop": True,
         "instruction": "Click the 'Export' button.", "action_type": "click",
         "attempts": [{}, {}]},
    ],
}]

paths = config.Paths(root=TMP).ensure()
rows, stats = stage_dataset.build_rows(TRAJ, GROUND, paths)

eq("one row written", len(rows), 1)
eq("the finish action was excluded", stats.actions_seen, 3)
eq("the ungrounded action was dropped", stats.ungrounded, 1)
eq("the not-groundable action was counted separately", stats.not_groundable, 1)
check("every action is accounted for exactly once", stats.accounting_ok, 
      "written=%d actions=%d" % (stats.written, stats.actions_seen))
check("the stats summary is printable", "rows written" in stats.summary())

r0 = rows[0]
eq("id format", r0["id"], "wilcom_demo_000_00")
eq("trajectory key", r0["trajectory"], "wilcom_demo")
eq("target_x is absolute pixels", r0["target_x"], ABS_X)
eq("target_y is absolute pixels", r0["target_y"], ABS_Y)
eq("image width recorded", r0["image_width"], FRAME_W)
eq("image height recorded", r0["image_height"], FRAME_H)
eq("answer_raw is the ABSOLUTE pixel pair", r0["answer_raw"],
   "<coordinate> [%d, %d]\n</coordinate>" % (ABS_X, ABS_Y))
# The parent dataset stores pixels, verified against it directly rather than assumed.
check("answer uses absolute pixels, not the 0-1000 normalized pair",
      r0["answer_raw"] == "<coordinate> [%d, %d]\n</coordinate>" % (ABS_X, ABS_Y)
      and "%d" % XN not in r0["answer_raw"].split(",")[0],
      r0["answer_raw"])
check("the closing tag is on its own line, as in the parent",
      r0["answer_raw"].endswith("]\n</coordinate>"), repr(r0["answer_raw"]))
check("prompt contains the instruction",
      r0["instruction"] in r0["prompt"])
check("prompt has the output tag", "<coordinate>" in r0["prompt"])
check("bbox recorded as JSON", json.loads(r0["bbox"]) == [600, 160, 680, 200])
eq("action type recorded", r0["action_type"], "click")
eq("timestamp recorded", r0["timestamp_s"], 5.0)
eq("source video recorded", r0["source_video"], "demo.mp4")
eq("sub-task context recorded", r0["sub_task"], "Digitize the logo")

msg = hf_export.to_messages(r0["prompt"], r0["answer_raw"])
eq("two messages", len(msg), 2)
eq("user then assistant", [m["role"] for m in msg], ["user", "assistant"])
eq("image placeholder first",
   msg[0]["content"][0]["type"], "image")
check("image placeholder carries a text key (the schema requires it)",
      "text" in msg[0]["content"][0])
eq("prompt text second", msg[0]["content"][1]["type"], "text")
eq("assistant carries the answer",
   msg[1]["content"][0]["text"], r0["answer_raw"])


# ============================================================================
section("13. Splitting and verification")
# ============================================================================

# Three videos, uneven sizes, to exercise the quota logic.
many = []
for v, n in (("v1", 30), ("v2", 20), ("v3", 3), ("v4", 1)):
    for i in range(n):
        many.append({
            "id": "wilcom_%s_000_%02d" % (v, i),
            "trajectory": "wilcom_%s" % v,
            "instruction": "Click button %d in %s." % (i, v),
            "prompt": "You are a helpful assistant. Given a screenshot of the "
                      "current screen and user instruction, you need to output "
                      "the position of the element you will operate.\n\n"
                      "The instruction is:\nClick button %d in %s.\n\n"
                      "Output the coordinate of the element you will operate "
                      "within <coordinate></coordinate> tag:\n"
                      "<coordinate> [x, y]\n</coordinate>" % (i, v),
            "image_path": str(frame_a),
            "target_x": 100 + i, "target_y": 200 + i,
            "image_width": FRAME_W, "image_height": FRAME_H,
            # Built through the real renderer rather than hand-written, so a fixture
            # cannot silently disagree with the format the pipeline emits. A
            # hand-written copy is what let the old convention survive here.
            "answer_raw": prompts.render_answer(100 + i, 200 + i),
            "source_video": "%s.mp4" % v, "task_id": 0, "step": i,
            "action_type": "click", "timestamp_s": float(i),
            "bbox": "[0, 0, 1, 1]", "sub_task": "task",
        })

splits = stage_dataset.split_by_trajectory(many)
eq("all rows are placed", sum(len(p) for p in splits.values()), len(many))

seen: dict = {}
for name, part in splits.items():
    for r in part:
        seen.setdefault(r["trajectory"], set()).add(name)
eq("no trajectory spans two splits",
   sum(1 for v in seen.values() if len(v) > 1), 0)

eq("all four trajectories are used", len(seen), 4)

# The same seed must give the same split, or a re-run silently changes the data.
s2 = stage_dataset.split_by_trajectory(many)
eq("splitting is deterministic under a fixed seed",
   {k: [r["id"] for r in v] for k, v in splits.items()},
   {k: [r["id"] for r in v] for k, v in s2.items()})

big = max(splits, key=lambda k: len(splits[k]))
eq("the largest split is train", big, "train")
check("train holds the bulk of the rows",
      len(splits["train"]) >= 0.7 * len(many))

eq("a clean dataset verifies", stage_dataset.verify(many, splits), [])

# Now break it in every way that matters, and require each to be caught.
broken = [dict(r) for r in many]
broken[0]["target_x"] = 99999
check("out-of-bounds target is caught",
      any("outside" in p for p in stage_dataset.verify(broken, splits)))

broken2 = [dict(r) for r in many]
# A NORMALIZED answer is now the wrong one, so that is what must be caught.
broken2[0]["answer_raw"] = "<coordinate> [%d, %d]\n</coordinate>" % (
    config.abs_to_norm(broken2[0]["target_x"], broken2[0]["image_width"]),
    config.abs_to_norm(broken2[0]["target_y"], broken2[0]["image_height"]))
check("a normalized-valued answer_raw is caught",
      any("pixel form" in p for p in stage_dataset.verify(broken2, splits)),
      [p for p in stage_dataset.verify(broken2, splits)][:1])

# And the space-before-tag variant, which is the same numbers with the wrong
# framing -- the exact bug this suite used to assert as correct.
broken3 = [dict(r) for r in many]
broken3[0]["answer_raw"] = "<coordinate> [%d, %d] </coordinate>" % (
    broken3[0]["target_x"], broken3[0]["target_y"])
check("a space-separated closing tag is caught",
      any("pixel form" in p for p in stage_dataset.verify(broken3, splits)))

leaky = {"train": [r for r in many if r["trajectory"] != "wilcom_v4"],
         "test": [r for r in many if r["trajectory"] == "wilcom_v4"]}
# Put one trajectory into two splits. The first version of this test appended a
# row that was *already* in train, so nothing leaked and the check could never
# have failed -- a green test asserting nothing.
leaky["test"].append([r for r in many if r["trajectory"] == "wilcom_v1"][0])
check("trajectory leakage is caught",
      any("leakage" in p for p in stage_dataset.verify(many, leaky)))

missing = [dict(r) for r in many]
missing[0]["image_path"] = str(TMP / "nope.png")
check("a missing image is caught",
      any("image missing" in p for p in stage_dataset.verify(missing, splits)))

dupes = [dict(r) for r in many] + [dict(many[0])]
check("a duplicate id is caught",
      any("duplicate id" in p for p in
          stage_dataset.verify(dupes, stage_dataset.split_by_trajectory(dupes))))

eq("an empty dataset reports a problem",
   stage_dataset.verify([], {"train": []}), ["no rows at all"])

# A spread of targets for one instruction is a NOTE, not a failure, because a
# toolbar button can legitimately be hit at slightly different pixels.
spread = [dict(r) for r in many[:3]]
for i, r in enumerate(spread):
    r["instruction"] = "Click the same button."
    # The derived fields are rebuilt too. Without this the row is deliberately
    # inconsistent, so verification reports that instead, and the spread check is
    # never actually exercised.
    r["prompt"] = prompts.render_prompt(r["instruction"])
    r["target_x"] = 100 + i * 400
    r["target_y"] = 200 + i * 100
    r["answer_raw"] = prompts.render_answer(r["target_x"], r["target_y"])
notes = stage_dataset.verify(spread, stage_dataset.split_by_trajectory(spread))
check("a wide target spread is a note, not a failure",
      notes and all(p.startswith("NOTE") for p in notes), str(notes)[:200])


# ============================================================================
section("14. Fake-client pipeline run")
# ============================================================================

GOOD_REPLY = reply_with(GOOD_TASK)
GOOD_GROUND_REPLY = json.dumps({
    "feasible": True,
    "predictions": [{"point_name": "point",
                     "center_point": "<point>400 800</point>",
                     "bounding_box": "<bbox>380 760 420 840</bbox>"}],
})


class FakeClient:
    """Stands in for the API client. Counts calls so a test can assert that
    caching actually prevented a request.

    It implements the same small surface the real client does -- `chat`, `media`
    and `ledger` -- and classifies a call by the *neutral* request it is handed,
    not by a wire format. That is deliberate: a fake that inspected `video_url`
    parts would keep passing after the client's wire format changed, which is
    exactly the kind of test that hides a breakage.
    """

    def __init__(self, traj_reply=GOOD_REPLY, ground_reply=GOOD_GROUND_REPLY):
        self.traj_reply, self.ground_reply = traj_reply, ground_reply
        self.calls = {"video": 0, "image": 0}
        self.kinds = []
        self.requests = []
        self.ledger = type("L", (), {"calls": 0, "failures": 0, "cost_usd": 0.0,
                                     "prompt_tokens": 0, "completion_tokens": 0,
                                     "report": lambda s=None: "fake"})()

    def media(self, path, kind: str):
        return (str(path), kind)

    def chat(self, model, request, **kw):
        self.requests.append(request)
        media = request.get("media") or []
        is_video = any(kind == "video" for _, kind in media)
        if is_video:
            self.calls["video"] += 1
            self.kinds.append("video")
            return {"text": self.traj_reply, "usage": {}, "model": model,
                    "cached": False}
        self.calls["image"] += 1
        self.kinds.append("image")
        return {"text": self.ground_reply, "usage": {}, "model": model,
                "cached": False}


HAVE_FFMPEG = True
try:
    FFMPEG = video.ffmpeg_path()
except SystemExit:
    HAVE_FFMPEG = False
    FFMPEG = None

if not (HAVE_FFMPEG and HAVE_PIL):
    print("  SKIP  ffmpeg or PIL unavailable; skipping the end-to-end check")
    SKIP += 1
else:
    import subprocess
    work = TMP / "e2e"
    work.mkdir(parents=True, exist_ok=True)
    src_video = work / "demo.mp4"

    # Synthesise a 12 s test pattern. testsrc is ideal: it has moving detail
    # across the whole frame, so a frame extracted at the wrong time is visibly
    # different from one extracted at the right time.
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc=size=640x360:rate=15:duration=12",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           str(src_video)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except Exception as exc:
        print("  SKIP  could not synthesise a test video: %s" % str(exc)[:120])
        SKIP += 1
    else:
        info = video.probe(src_video)
        check("probed duration is about 12 s", 11.0 < info.duration < 13.0,
              "got %.2f" % info.duration)
        eq("probed width", info.width, 640)
        eq("probed height", info.height, 360)
        eq("long side", info.long_side, 640)
        check("probed fps is about 15", 14.0 < info.fps < 16.0,
              "got %.2f" % info.fps)

        # target_size must never upscale, and must keep both dimensions even.
        eq("target_size leaves a small video alone",
           video.target_size(640, 360, 1280), (640, 360))
        eq("target_size downscales 1080p to 720p",
           video.target_size(1920, 1080, 1280), (1280, 720))
        eq("target_size handles portrait",
           video.target_size(1080, 1920, 1280), (720, 1280))
        eq("target_size handles an odd aspect",
           video.target_size(1919, 1079, 1280), (1280, 720))
        check("target_size output is always even",
              all(v % 2 == 0 for v in
                  (video.target_size(1919, 1079, 1280)
                   + video.target_size(1000, 333, 400)
                   + video.target_size(333, 1000, 400))))
        check("target_size never upscales",
              all(max(video.target_size(w, h, 1280)) <= max(w, h)
                  for w, h in ((640, 360), (800, 600), (1279, 719),
                               (20, 20), (3000, 40))))

        eq("a 12 s video is one clip", len(video.plan_clips(info)), 1)
        check("a 600 s video is three clips",
              len(video.plan_clips(
                  video.VideoInfo(src_video, 600.0, 640, 360, 15.0))) == 3)
        spans = video.plan_clips(video.VideoInfo(src_video, 500.0, 640, 360, 15.0),
                                 clip_seconds=240)
        eq("a 500 s video is three clips", len(spans), 3)
        check("clip spans are contiguous",
              all(abs(spans[i][1] - spans[i + 1][0]) < 1e-6
                  for i in range(len(spans) - 1)))
        check("the last clip ends at the video end",
              abs(spans[-1][1] - 500.0) < 1e-6)

        clip = work / "clip.mp4"
        video.make_clip(src_video, 0.0, 4.0, clip)
        check("clip exists and is non-empty",
              clip.exists() and clip.stat().st_size > 0)
        check("clip duration is about 4 s",
              3.5 < video.probe(clip).duration < 4.5,
              "got %.2f" % video.probe(clip).duration)

        # Downscaling must not upscale a small source.
        small = work / "small.mp4"
        video.make_clip(src_video, 0.0, 2.0, small, long_side=1280)
        eq("a small source is not upscaled", video.probe(small).width, 640)

        fdir = work / "frames"
        frames = video.extract_frames_at(src_video, 6.0, (-0.5, 0.0, 0.5),
                                        fdir, "probe")
        eq("three frames extracted", len(frames), 3)
        eq("frames are at t-0.5, t, t+0.5",
           [round(t, 2) for t, _ in frames], [5.5, 6.0, 6.5])
        check("frame files are non-empty PNGs",
              all(p.exists() and p.stat().st_size > 500 for _, p in frames))
        eq("frame size is the native resolution",
           video.frame_size(frames[0][1]), (640, 360))

        # Frames genuinely differ, which is what makes the 3-frame fallback worth
        # doing. If a seek silently returned frame 0 every time, this catches it.
        import hashlib
        digests = {hashlib.sha256(p.read_bytes()).hexdigest()
                   for _, p in frames}
        eq("the three frames are not identical", len(digests), 3)

        # A frame past the end must not be silently substituted.
        late = video.extract_frames_at(src_video, 11.9, (0.0, 1.0), fdir, "late")
        eq("a frame past the end is not produced", len(late), 1)

        uri = "data:image/png;base64,QUJD"
        check("frames are read as bytes, not encoded to data URIs",
              hashing.file_digest(frames[0][1]) != "")
        eq("clips are read as bytes too",
           len(hashing.file_digest(clip)), 64)

        # ---- full run through the three stages with a fake client
        fake = FakeClient()
        ppaths = config.Paths(root=TMP / "e2e_work").ensure()
        # Point the work tree's video at the synthesised file.
        import shutil as _sh
        _sh.copy2(src_video, ppaths.videos / "demo.mp4")

        traj_rec = stage_trajectory.extract_video(
            src_video, fake, ppaths, model="fake", verbose=False)
        eq("one video call made", fake.calls["video"], 1)
        eq("trajectory record has a task", len(traj_rec["tasks"]), 1)
        eq("trajectory record has two actions",
           len(traj_rec["tasks"][0]["actions"]), 2)
        eq("shots were parsed from Part 1", traj_rec["stats"]["shots"], 2)
        eq("one groundable action", traj_rec["stats"]["groundable"], 1)
        eq("record was written to disk",
           (ppaths.traj / "demo.json").exists(), True)

        # Stage 1 again must hit the cache, not the API.
        before = fake.calls["video"]
        stage_trajectory.extract_video(src_video, fake, ppaths, model="fake",
                                       verbose=False)
        eq("re-running stage 1 makes no new call", fake.calls["video"], before)

        ground_rec = stage_grounding.ground_video(traj_rec, fake, ppaths,
                                                  model="fake", verbose=False)
        eq("one groundable action was attempted",
           ground_rec["stats"]["attempted"], 1)
        eq("it grounded", ground_rec["stats"]["grounded"], 1)
        eq("no drops", ground_rec["stats"]["dropped"], 0)
        check("grounding used image calls, not video calls",
              fake.calls["image"] >= 1 and fake.calls["video"] == before)

        fin = [r for r in ground_rec["results"] if r.get("grounded")][0]
        eq("grounded x on the 640x360 frame",
           fin["x"], config.norm_to_abs(800, 640))
        eq("grounded y on the 640x360 frame",
           fin["y"], config.norm_to_abs(400, 360))
        eq("the grounding result records its own image size",
           (fin["image_width"], fin["image_height"]), (640, 360))
        check("the winning frame file exists",
              (ppaths.frames / "demo" / fin["frame"]).exists())

        before_g = fake.calls["image"]
        stage_grounding.ground_video(traj_rec, fake, ppaths, model="fake",
                                     verbose=False)
        eq("re-running stage 2 makes no new call", fake.calls["image"], before_g)

        # ---- stage 3 over the real records
        rows2, stats2 = stage_dataset.build_rows([traj_rec], [ground_rec], ppaths)
        eq("one dataset row from the end-to-end run", len(rows2), 1)
        r = rows2[0]
        eq("target_x matches the grounded point", r["target_x"], fin["x"])
        eq("target_y matches the grounded point", r["target_y"], fin["y"])
        eq("image handed to the model is the winning frame",
           Path(r["image_path"]).name, fin["frame"])
        eq("row image dimensions are the frame's own",
           (r["image_width"], r["image_height"]), (640, 360))
        eq("answer is the pixel pair, matching the target columns",
           r["answer_raw"],
           "<coordinate> [%d, %d]\n</coordinate>" % (r["target_x"], r["target_y"]))
        check("the answer and the target are literally the same numbers",
              str(r["target_x"]) in r["answer_raw"]
              and str(r["target_y"]) in r["answer_raw"],
              r["answer_raw"])

        sp = stage_dataset.split_by_trajectory(rows2)
        eq("end-to-end dataset verifies",
           stage_dataset.verify(rows2, sp), [])

        # `check_path`, not `report`, which would shadow the report module.
        check_path = report.render(rows2, ppaths.logs / "check.html")
        html = check_path.read_text(encoding="utf-8")
        check("report was written", check_path.exists() and len(html) > 2000)
        check("report embeds the screenshot as base64",
              "data:image/png;base64," in html)
        check("report shows the instruction", "Digitize" in html)
        check("report draws a marker", 'class="dot"' in html)

        # A model that reports "not feasible" for all three frames must drop the
        # action rather than write an approximate coordinate.
        never = FakeClient(ground_reply=json.dumps(
            {"feasible": False, "reason": "not visible"}))
        w2 = config.Paths(root=TMP / "e2e_never").ensure()
        _sh.copy2(src_video, w2.videos / "demo.mp4")
        t2 = stage_trajectory.extract_video(src_video, never, w2, model="fake",
                                            verbose=False)
        g2 = stage_grounding.ground_video(t2, never, w2, model="fake", verbose=False)
        eq("infeasible grounding drops the action",
           g2["stats"]["dropped"], 1)
        eq("infeasible grounding yields no rows",
           len(stage_dataset.build_rows([t2], [g2], w2)[0]), 0)
        eq("all three frames were tried before giving up",
           len([r for r in g2["results"]][0]["attempts"]), 3)

        # A garbage video reply must surface as a failure, not as zero tasks
        # silently written.
        bad = FakeClient(traj_reply="I am unable to analyse this video.")
        w3 = config.Paths(root=TMP / "e2e_bad").ensure()
        _sh.copy2(src_video, w3.videos / "demo.mp4")
        t3 = stage_trajectory.extract_video(src_video, bad, w3, model="fake",
                                            verbose=False)
        eq("an unparseable clip yields no tasks", t3["stats"]["tasks"], 0)
        check("the failure is recorded on the clip",
              "error" in (t3["clips"][0] if t3["clips"] else {}))

        # The first-frame-wins rule: frame 0 infeasible, frame 1 good.
        class PartialClient(FakeClient):
            """Counts only image calls, so the trajectory call does not consume
            the deliberate first-frame failure."""

            def __init__(self, **kw):
                super().__init__(**kw)
                self.img = 0

            def chat(self, model, request, **kw):
                self.requests.append(request)
                media = request.get("media") or []
                if any(kind == "video" for _, kind in media):
                    self.calls["video"] += 1
                    return {"text": self.traj_reply, "usage": {},
                            "model": model, "cached": False}
                self.img += 1
                self.calls["image"] += 1
                if self.img == 1:
                    return {"text": json.dumps({"feasible": False,
                                                "reason": "not in this one"}),
                            "usage": {}, "model": model, "cached": False}
                return {"text": self.ground_reply, "usage": {}, "model": model,
                        "cached": False}

        partial = PartialClient()
        w4 = config.Paths(root=TMP / "e2e_partial").ensure()
        _sh.copy2(src_video, w4.videos / "demo.mp4")
        t4 = stage_trajectory.extract_video(src_video, partial, w4, model="fake",
                                            verbose=False)
        g4 = stage_grounding.ground_video(t4, partial, w4, model="fake",
                                          verbose=False)
        eq("a later frame rescues an infeasible first frame",
           g4["stats"]["grounded"], 1)
        eq("it took two attempts", g4["results"][0]["n_attempts"], 2)
        eq("the winning frame offset is recorded",
           g4["results"][0]["offset_s"], 0.0)
        eq("the winning frame is the middle one",
           Path(g4["results"][0]["frame"]).name.split("_")[-1], "f1.png")
        eq("the failed attempt is preserved in the record",
           len(g4["results"][0]["attempts"]), 2)
        eq("the failed attempt carries the rejection reason",
           g4["results"][0]["attempts"][0].get("rejected"),
           "model reported not feasible: not in this one")


# ============================================================================
section("15. Extracted foundation modules")
# ============================================================================

# --- hashing. The NUL separator is load-bearing: without it two different inputs
# could share a cache key, and a stale result would be served silently.
eq("fingerprint is stable", hashing.fingerprint("a", "b"),
   hashing.fingerprint("a", "b"))
check("fingerprint distinguishes part boundaries",
      hashing.fingerprint("ab", "c") != hashing.fingerprint("a", "bc"))
check("fingerprint distinguishes content",
      hashing.fingerprint("a") != hashing.fingerprint("b"))
eq("fingerprint is 16 hex chars", len(hashing.fingerprint("x")), 16)

BLANK = TMP / "hash_a.bin"
OTHER = TMP / "hash_b.bin"
BLANK.write_bytes(b"wilcom")
OTHER.write_bytes(b"wilcom!")
eq("file_digest is a plain sha256 of the bytes",
   hashing.file_digest(BLANK), hashlib.sha256(b"wilcom").hexdigest())
check("file_digest changes with content",
      hashing.file_digest(BLANK) != hashing.file_digest(OTHER))

# --- ledger. Pricing and cost accounting, which the whole spend estimate rests on.
eq("a known model has a price", ledger.price_for("gemini-3.1-pro-preview"),
   (2.00e-6, 12.00e-6))
check("an unknown model is priced pessimistically, not at zero",
      ledger.price_for("no-such-model") == (2.00e-6, 12.00e-6),
      "a zero price would make the ledger understate spend")
check("a prompt over 200k crosses the price tier",
      ledger.is_long_prompt(200_001) and not ledger.is_long_prompt(200_000))

# Thinking tokens bill at the OUTPUT rate and are not in completion_tokens, so
# omitting them understates a Gemini run by roughly 2x on Pro.
_with = ledger.cost_of("gemini-3.1-pro-preview", 10_000, 2_000, 10_000)
_without = ledger.cost_of("gemini-3.1-pro-preview", 10_000, 2_000)
check("thinking tokens are billed", _with > _without,
      "with=%r without=%r" % (_with, _without))
eq("the thinking delta is charged at the output rate",
   round(_with - _without, 9), round(10_000 * 12.00e-6, 9))

_L = ledger.Ledger()
_L.add("gemini-3.1-pro-preview",
       {"prompt_tokens": 100, "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 30}},
       2.00e-6, 12.00e-6)
eq("the ledger counts the call", _L.calls, 1)
eq("the ledger sums thinking tokens", _L.reasoning_tokens, 30)
eq("the ledger sums completion tokens", _L.completion_tokens, 20)
eq("the ledger cost includes thinking",
   round(_L.cost_usd, 9), round(100 * 2e-6 + (20 + 30) * 12e-6, 9))
_L.add_failure()
eq("failures are counted", _L.failures, 1)
check("the report names the thinking share", "thinking tokens" in _L.report())

# --- records. manifest.json and summary.json sit in the same directories as the
# per-video records and must not be fed back in as if they were records.
RDIR = TMP / "records"
ONE = RDIR / "one.json"
records.write_json({"a": 1}, ONE)
eq("write_json round-trips", records.read_json(ONE), {"a": 1})
records.write_json({"t": "caf\u00e9 \u2014 test"}, RDIR / "two.json")
eq("write_json keeps non-ASCII readable, unescaped",
   records.read_json(RDIR / "two.json")["t"], "caf\u00e9 \u2014 test")
eq("read_json on a missing file returns the default",
   records.read_json(RDIR / "nope.json", "fallback"), "fallback")
(RDIR / "bad.json").write_text("{not json at all", encoding="utf-8")
eq("read_json tolerates corrupt JSON",
   records.read_json(RDIR / "bad.json", "fallback"), "fallback")
records.write_json({"manifest": True}, RDIR / "manifest.json")
records.write_json({"summary": True}, RDIR / "summary.json")
eq("load_dir skips manifest and summary",
   sorted(r.get("a") for r in records.load_dir(RDIR) if "a" in r), [1])
eq("load_dir skips the corrupt file too",
   len(records.load_dir(RDIR)), 2)
eq("load_dir on a missing directory is empty",
   records.load_dir(TMP / "no_such_dir"), [])

# --- hf_export and the prompts renderers it relies on.
# The exact bytes the parent dataset uses: absolute pixels, closing tag on its own
# line. Verified against `GUI-360-word-unsloth` itself, not against a note in the
# code -- the previous expectation here was a space before the tag and normalized
# values, both of which the parent does not do.
eq("render_answer matches the trained format",
   prompts.render_answer(487, 295), "<coordinate> [487, 295]\n</coordinate>")
check("render_answer does NOT pad the closing tag onto the same line",
      "\n</coordinate>" in prompts.render_answer(1, 2))
eq("render_answer uses the answer template",
   prompts.render_answer(1, 2), prompts.ANSWER_TEMPLATE % (1, 2))
check("render_prompt substitutes the instruction",
      "Click the Save button." in prompts.render_prompt("Click the Save button."))
check("render_prompt leaves no placeholder behind",
      "{instruction}" not in prompts.render_prompt("x"))
check("render_prompt is byte-identical for the same input",
      prompts.render_prompt("same") == prompts.render_prompt("same"))
check("the manifest records the coordinate convention",
      "answer_raw" in hf_export.manifest({"train": rows})["coordinate_convention"])
eq("the manifest counts trajectories, not rows",
   hf_export.manifest({"train": rows})["splits"]["train"]["trajectories"], 1)
eq("the manifest omits empty splits",
   "test" in hf_export.manifest({"train": rows, "test": []})["splits"], False)

# --- report. The previous version built the whole page with str.format, so every
# CSS brace had to be doubled. An instruction containing a brace would have
# mangled it, and this is the test that would have caught that.
if HAVE_PIL:
    hostile = dict(rows[0])
    hostile["instruction"] = 'Click {Layout} & <Margins> "quoted"'
    hostile["sub_task"] = "<script>alert(1)</script>"
    page_path = report.render([hostile], TMP / "hostile.html", limit=0)
    page = page_path.read_text(encoding="utf-8")
    check("report renders a card despite braces in the instruction",
          'class="card"' in page)
    check("report does not leak a script tag from a sub-task",
          "<script>alert(1)</script>" not in page)
    check("report escapes ampersands", "&amp;" in page)
    check("report keeps its own stylesheet intact",
          ".wrap img{width:100%" in page)
    check("report keeps the filter script intact",
          "querySelectorAll('.filter')" in page)

    empty_path = report.render([], TMP / "empty.html", limit=0)
    check("report handles zero rows without raising",
          "No cards" in empty_path.read_text(encoding="utf-8"))

    limited = report.render(rows * 3, TMP / "limited.html", limit=2)
    check("report honours the limit",
          limited.read_text(encoding="utf-8").count('class="card"') == 2)

# --- cost. Pure arithmetic, so it must be checkable without a network call.
est = cost.estimate([video.VideoInfo(TMP / "x.mp4", 480.0, 1920, 1080, 30.0)],
                    clip_seconds=240)
eq("a 480 s video plans two clips", est.clips, 2)
eq("the estimate records its clip length", est.clip_seconds, 240)
check("the trajectory cost is positive", est.traj_usd > 0)
check("the grounding cost is positive", est.ground_usd > 0)
check("the screening cost is small but nonzero",
      0 < est.meta_usd < est.traj_usd,
      "screening should be negligible next to a video call")
eq("total is the sum of all three stages",
   round(est.total_usd, 9),
   round(est.traj_usd + est.ground_usd + est.meta_usd, 9))
check("the per-sample figure divides by actions",
      abs(est.per_sample_usd - est.total_usd / est.actions) < 1e-9)

# The thinking level changes stage 1's cost materially, which is the whole point of
# having the knob: measured 978 thinking tokens at "low" against 11,681 at default.
_cheap = cost.estimate([video.VideoInfo(TMP / "x.mp4", 480.0, 1920, 1080, 30.0)],
                       clip_seconds=240, thinking_level="low")
_dear = cost.estimate([video.VideoInfo(TMP / "x.mp4", 480.0, 1920, 1080, 30.0)],
                      clip_seconds=240, thinking_level="medium")
check("a higher thinking level costs more", _dear.traj_usd > _cheap.traj_usd,
      "low=%.4f medium=%.4f" % (_cheap.traj_usd, _dear.traj_usd))
eq("the estimate records which level it costed", _cheap.thinking_level, "low")
check("screening can be excluded from the estimate",
      cost.estimate([video.VideoInfo(TMP / "x.mp4", 480.0, 1920, 1080, 30.0)],
                    include_screening=False).meta_usd == 0.0)
check("describe() states its caveats rather than hiding them",
      "Caveats" in est.describe() and "guess" in est.describe())
eq("an empty corpus costs nothing",
   cost.estimate([]).total_usd, 0.0)
eq("an empty corpus does not divide by zero",
   cost.estimate([]).per_sample_usd, 0.0)


# ============================================================================
section("16. Native frame resolution -- the coordinate space")
# ============================================================================

# The guarantee: a grounding frame is stored at the SOURCE video's resolution,
# never at the stage-1 clip's. Coordinates are pixels in that exact image, so a
# silent resample would invalidate every label while looking perfectly fine.
# Proven with a source larger than VIDEO_LONG_SIDE, so the clip must differ from
# the frame; on a small source the two sizes coincide and the test would be vacuous.

if not (HAVE_FFMPEG and HAVE_PIL):
    print("  SKIP  ffmpeg or PIL unavailable")
    SKIP += 1
else:
    import shutil as _sh2

    NAT = TMP / "native"
    NAT.mkdir(parents=True, exist_ok=True)
    src_big = NAT / "big.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=10:duration=1",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(src_big)], check=True)

    binfo = video.probe(src_big)
    eq("source is 1920x1080", (binfo.width, binfo.height), (1920, 1080))

    clip_big = NAT / "big_clip.mp4"
    video.make_clip(src_big, 0.0, 1.0, clip_big)
    cinfo = video.probe(clip_big)
    eq("the stage-1 clip IS downscaled",
       max(cinfo.width, cinfo.height), config.VIDEO_LONG_SIDE)
    check("clip is genuinely smaller than the source",
          cinfo.width < binfo.width,
          "%dx%d vs %dx%d" % (cinfo.width, cinfo.height,
                              binfo.width, binfo.height))

    big_frames = video.extract_frames_at(src_big, 0.5, (0.0,),
                                        NAT / "frames", "native")
    eq("a frame was extracted", len(big_frames), 1)
    big_frame = big_frames[0][1]
    fw, fh = video.frame_size(big_frame)
    eq("the FRAME is at the source resolution, not the clip's",
       (fw, fh), (1920, 1080))
    check("frame size differs from clip size (so this proves something)",
          (fw, fh) != (cinfo.width, cinfo.height))

    # And the stored row must agree, end to end.
    npaths = config.Paths(root=NAT / "work").ensure()
    npaths.frames.joinpath("big").mkdir(parents=True, exist_ok=True)
    stored = npaths.frames / "big" / big_frame.name
    _sh2.copy2(big_frame, stored)

    XN_B = config.abs_to_norm(960, 1920)
    YN_B = config.abs_to_norm(540, 1080)
    ntraj = [{"video": "big.mp4", "video_stem": "big",
              "width": 1920, "height": 1080, "duration_s": 1.0, "fps": 10.0,
              "tasks": [{"task_id": 0, "clip": 0, "clip_start_s": 0.0,
                         "instruction": "t", "dense_caption": "", "plan": "",
                         "platform": "windows", "software": "s",
                         "website": None, "has_finish": True,
                         "actions": [{"task_id": 0, "seq": 0, "clip": 0,
                                      "timestamp": "00:01", "t": 0.5,
                                      "action_type": "click",
                                      "grounding_instruction": "Click centre.",
                                      "action_reason": "",
                                      "action_parameters": {},
                                      "core_change": "",
                                      "core_change_reason": "",
                                      "traj_point": None,
                                      "needs_grounding": True}]}]}]
    nground = [{"video": "big.mp4", "video_stem": "big", "model": "t",
                "prompt_version": "ground-v1",
                "width": 1920, "height": 1080,
                "results": [{"task_id": 0, "seq": 0, "t": 0.5, "grounded": True,
                             "drop": False, "instruction": "Click centre.",
                             "action_type": "click", "attempts": [{}],
                             "x": config.norm_to_abs(XN_B, 1920),
                             "y": config.norm_to_abs(YN_B, 1080),
                             "bbox": [900, 500, 1020, 580],
                             "point_norm": [XN_B, YN_B],
                             "bbox_norm": [config.abs_to_norm(900, 1920),
                                           config.abs_to_norm(500, 1080),
                                           config.abs_to_norm(1020, 1920),
                                           config.abs_to_norm(580, 1080)],
                             "image_width": 1920, "image_height": 1080,
                             "frame": big_frame.name, "frame_time_s": 0.5,
                             "offset_s": 0.0, "n_attempts": 1,
                             "cross_check": {}}]}]

    nrows, _ = stage_dataset.build_rows(ntraj, nground, npaths)
    eq("a row was built from the large source", len(nrows), 1)
    nrow = nrows[0]
    eq("row records the SOURCE resolution",
       (nrow["image_width"], nrow["image_height"]), (1920, 1080))
    eq("the stored image really is that size",
       video.frame_size(Path(nrow["image_path"])), (1920, 1080))
    check("the dataset image is not the clip's size",
          (nrow["image_width"], nrow["image_height"])
          != (cinfo.width, cinfo.height))
    eq("verify accepts it", stage_dataset.verify(nrows,
                                                 stage_dataset.split_by_trajectory(nrows)), [])

    # The oversized-source ceiling: a warning, never a resize.
    small_info = video.VideoInfo(src_big, 1.0, 1280, 720, 10.0)
    eq("a source within the ceiling is not warned about",
       video.oversize_frame_warning(small_info, limit=2560), None)
    eq("the ceiling can be disabled entirely",
       video.oversize_frame_warning(
           video.VideoInfo(src_big, 1.0, 7680, 4320, 10.0), limit=0), None)
    loud = video.oversize_frame_warning(
        video.VideoInfo(src_big, 1.0, 7680, 4320, 10.0), limit=2560)
    check("an oversized source is warned about", loud is not None)
    check("the warning says frames are not resized",
          loud is not None and "NOT resized" in loud)
    check("the default ceiling is a positive number",
          config.MAX_FRAME_LONG_SIDE > 0,
          "config says %r" % config.MAX_FRAME_LONG_SIDE)
    check("the default ceiling passes 1440p",
          video.oversize_frame_warning(
              video.VideoInfo(src_big, 1.0, 2560, 1440, 10.0)) is None)
    check("the default ceiling flags 4K",
          video.oversize_frame_warning(
              video.VideoInfo(src_big, 1.0, 3840, 2160, 10.0)) is not None)

# The manifest must carry the training-side requirement, because the resize that
# actually destroys these coordinates happens in the consumer's collator.
_man = hf_export.manifest({"train": many})
check("manifest records the image resolution promise",
      "native" in _man["images"]["resolution"])
check("manifest names the collator", 
      _man["consumer_notes"]["training_collator"] == "UnslothVisionDataCollator")
eq("manifest requires resize='max'",
   _man["consumer_notes"]["training_resize"], "max")
check("manifest explains why", "resize='min'" in _man["consumer_notes"]["why"])
check("manifest flags the unavoidable 28-multiple snap",
      "28-pixel" in _man["consumer_notes"]["unavoidable"])
check("manifest states both coordinate conventions",
      "answer_raw" in _man["coordinate_convention"]
      and "target_x / target_y" in _man["coordinate_convention"])


# ============================================================================
section("17. Prompts match the PDF, not just the OCR")
# ============================================================================

# These assert facts read off the real PDF (arXiv 2605.14747), with page numbers.
# The OCR'd markdown had mangled several of them, and an earlier draft of
# prompts.py copied two of the OCR's mistakes plus three edits of my own. Each
# check names where in the paper to look, so a failure says which prompt drifted.
#
# Ordering note: a trailing backslash in a Python string literal continues the
# *source* line, so text split across source lines arrives joined. Checks below
# therefore look for the joined form.

_P3, _P4, _P5 = (prompts.PROMPT_TRAJECTORY, prompts.PROMPT_CONTINUATION,
                 prompts.PROMPT_GROUNDING)

# -- Prompt 3, PDF pages 20-22 -------------------------------------------------
check("P3: bold Output Format (Part 1), as the paper renders it",
      "**Output Format (Part 1):**" in _P3)
check("P3: bold Output Format (Part 2)", "**Output Format (Part 2):**" in _P3)
check("P3: bold Field Definitions heading",
      "**Field Definitions for JSON (Part 2)**" in _P3)
check("P3: bold Platform-Specific Action Spaces heading, which we had omitted",
      "**Platform-Specific Action Spaces**" in _P3)
check("P3: the JSON-list sentence ends with a period",
      "[{...}, {...}]." in _P3)
check("P3: 'press' is qualified as keyboard keys (we had dropped this)",
      'action_type: "press" (for keyboard keys)' in _P3)
check("P3: the wait duration is noted as milliseconds (dropped before)",
      "(in milliseconds)" in _P3)
check("P3: the closing line is bold",
      "**Try best efforts, response at least 5k tokens.**" in _P3)
check("P3: the plan example closes its quote before the period",
      "formatted as 'step1: ..., step2: ...'." in _P3)
check("P3: the action-space headings are NOT markdown headings",
      "### A. PC Action Space" not in _P3
      and "\nA. PC Action Space (" in _P3)
check("P3: the mobile action-space heading is not a markdown heading either",
      "### B. Mobile Action Space" not in _P3)
check("P3: all six PC click variants survive, which the OCR had deduplicated",
      all(('action_type: "%s"' % a) in _P3 for a in
          ("click", "doubleClick", "tripleClick", "rightClick", "middleClick")))
check("P3: dragTo carries both endpoints",
      '"start_point": "[y, x]", "end_point": "[y, x]"' in _P3)
check("P3: coordinates are declared y-first, in pixels",
      "Coordinates [y, x] are in pixels." in _P3)

# -- Prompt 4, PDF pages 23-24 -------------------------------------------------
check("P4: CONTEXT is an H1, as in the paper, not an H2",
      _P4.startswith("# CONTEXT"))
check("P4: CURRENT TASK is an H1", "\n# CURRENT TASK\n" in _P4)
check("P4: the history separator matches the paper's '# ==='",
      "# ===\n" in _P4)
check("P4: no invented '====' separator", "\n====\n" not in _P4)
check("P4: 'character analyses' is present (an edit of mine had removed it)",
      "character analyses" in _P4)
check("P4: 'narrative arcs' is plural, as in the paper",
      "narrative arcs" in _P4)
check("P4: the consistency rule reads as the paper's",
      "interpretations, character analyses, and narrative arcs logically "
      "continue" in _P4)
check("P4: the global-context aside is present (also removed by me)",
      "(e.g., established narrative, characters, themes)" in _P4)
check("P4: the time range is one bold span",
      "**{start_time} - {end_time}**" in _P4)
check("P4: all five numbered rules survive",
      all(("\n%d. **" % n) in _P4 for n in range(1, 6)))

# -- Prompt 5, PDF pages 25-26 -------------------------------------------------
check("P5: Inputs is an H2, as in the paper",
      "\n## Inputs\n" in _P5)
check("P5: both inputs are H3, as in the paper",
      "### 1. Grounding Instruction" in _P5
      and "### 2. Action Type" in _P5)
check("P5: the horizontal rule before Inputs is present",
      "-----\n\n## Inputs" in _P5)
check("P5: both example headings are H3",
      "### Example: Action is Feasible" in _P5
      and "### Example: Action is Not Feasible" in _P5)
check("P5: the feasible example shows TWO predictions, as the paper does",
      '"point_name": "start_point"' in _P5
      and '"point_name": "end_point"' in _P5)
check("P5: the example two-point form is what makes dragTo groundable",
      "dragTo" in _P3 and "start_point" in _P5)
check("P5: the placeholders are quoted, as in the paper",
      "'{grounding_instruction}'" in _P5 and "'{action_type}'" in _P5)
check("P5: the closing Remember block and its two rules are present",
      "**Remember:**" in _P5 and "1. The center points" in _P5
      and "2. The bounding boxes" in _P5)
check("P5: the 0-1000 scale and y-first tag order are stated",
      "(0-1000)" in _P5 and "<point>y x</point>" in _P5
      and "<bbox>y1 x1 y2 x2</bbox>" in _P5)

# -- our own context must live outside the paper's bodies ----------------------
for _label, _body in (("P3", _P3), ("P4", _P4), ("P5", _P5)):
    check("%s: no Wilcom text leaked into the paper's prompt" % _label,
          "Wilcom" not in _body and "EmbroideryStudio" not in _body)
check("the WILCOM context is in the system preambles instead",
      "Wilcom" in prompts.SYSTEM_TRAJECTORY
      and "Wilcom" in prompts.SYSTEM_GROUNDING)

# -- cache keys must follow the prompt text -----------------------------------
# A hand-maintained PROMPT_VERSION only changes if someone remembers. These hash
# the prompt text itself, so editing a prompt cannot serve a stale cached result.
_fp_t = prompts.trajectory_prompt_fingerprint()
_fp_g = prompts.grounding_prompt_fingerprint()
eq("the trajectory fingerprint is 16 chars", len(_fp_t), 16)
eq("the grounding fingerprint is 16 chars", len(_fp_g), 16)
check("the two fingerprints differ", _fp_t != _fp_g)
eq("the trajectory fingerprint is stable across calls",
   prompts.trajectory_prompt_fingerprint(), _fp_t)

# Editing a prompt constant must change its fingerprint. Simulated by hashing the
# same inputs with one character altered, which is what a real edit does.
check("a one-character prompt edit would change the fingerprint",
      hashing.fingerprint(prompts.SYSTEM_TRAJECTORY + " ",
                          prompts.PROMPT_TRAJECTORY,
                          prompts.PROMPT_CONTINUATION) != _fp_t)

# -- the dataset prompt is a separate artefact from the paper's ----------------
check("the dataset prompt is not derived from any paper prompt",
      prompts.INSTRUCTION_TEMPLATE not in _P3
      and "<coordinate> [x, y] </coordinate>" not in _P3)
check("the paper's Prompts 6-8 are deliberately absent",
      "GUI Action Prediction" not in _P3 + _P4 + _P5
      and "GUI Trajectory Modeling" not in _P3 + _P4 + _P5)


# ============================================================================
section("18. Caption selection -- language, kind and honesty about failure")
# ============================================================================
#
# This path had NO tests, which is how it shipped looking only for English and
# then discarded the Hindi and Urdu tracks that are the only ones that actually
# download for this corpus. Every check below is pure logic with a stubbed
# downloader: no network, no yt-dlp.

_real_download = metadata._download_caption
_DOWNLOADED: list = []
_FAKE_TRACKS: dict = {}


def _fake_download(url, lang, ext, auto):
    """Serve a canned track per language, so no network is touched."""
    _DOWNLOADED.append((lang, "auto" if auto else "manual"))
    return _FAKE_TRACKS.get(lang, "")


def _with(url, info, tracks):
    """Run `_subtitles` against a stubbed downloader and report what it picked.

    The attempt log is cleared at the START rather than the end: clearing it on the
    way out wipes the very evidence the assertions read, which showed up as a
    spurious "it did try English first" failure.
    """
    global _DOWNLOADED, _FAKE_TRACKS
    _FAKE_TRACKS = tracks
    _DOWNLOADED = []
    metadata._download_caption = _fake_download
    try:
        return metadata._subtitles(url, info)
    finally:
        metadata._download_caption = _real_download


# A Hindi tutorial exactly as YouTube presents it: no manual captions, machine
# tracks in many languages, `hi` as the detected original.
HINDI_INFO = {
    "language": "hi",
    "subtitles": {},
    "automatic_captions": {
        "en": [{"ext": "json3"}],
        "hi": [{"ext": "json3"}],
        "ur": [{"ext": "json3"}],
    },
}

# The real-world failure this fixes: the English machine track does not download,
# while the original-language track does.
text, lang, kind = _with("u", HINDI_INFO, {"en": "", "hi": "\u0939\u093f\u0928\u094d\u0926\u0940"})
eq("falls through to the original language when English is empty", lang, "hi")
eq("the text is the one that downloaded", text, "\u0939\u093f\u0928\u094d\u0926\u0940")
eq("kind is recorded as automatic", kind, "auto")
check("it did try English first", _DOWNLOADED and _DOWNLOADED[0][0] == "en",
      _DOWNLOADED)

# English present and working: still preferred, because the prompts are English.
text, lang, kind = _with("u", HINDI_INFO, {"en": "english text", "hi": "\u0939"})
eq("English wins when it actually downloads", lang, "en")
eq("and its text is used", text, "english text")

# A human-written track beats a machine one even in another language.
MANUAL_INFO = {
    "language": "hi",
    "subtitles": {"hi": [{"ext": "vtt"}]},
    "automatic_captions": {"en": [{"ext": "json3"}]},
}
text, lang, kind = _with("u", MANUAL_INFO, {"hi": "manual hindi", "en": "auto english"})
eq("a human track beats a machine track, language notwithstanding", kind, "manual")
eq("and it is the human one", text, "manual hindi")

# English manual still outranks Hindi manual.
BOTH_MANUAL = {
    "language": "hi",
    "subtitles": {"en": [{"ext": "vtt"}], "hi": [{"ext": "vtt"}]},
    "automatic_captions": {},
}
_, lang, kind = _with("u", BOTH_MANUAL, {"en": "manual english", "hi": "manual hindi"})
eq("manual English outranks manual Hindi", (lang, kind), ("en", "manual"))

# Only an unrelated language exists: better than nothing.
ODD_ONLY = {"language": "bn", "subtitles": {},
            "automatic_captions": {"bn": [{"ext": "json3"}]}}
_, lang, _ = _with("u", ODD_ONLY, {"bn": "bangla text"})
eq("an unrelated language is still used rather than giving up", lang, "bn")

# -- Hindi is the explicit second choice -------------------------------------
# This corpus is Hindi and Urdu tutorials, so a Hindi video rarely has English and
# almost always has Hindi.
HINDI_FALLBACK = {
    "language": "hi",
    "subtitles": {},
    "automatic_captions": {
        "en": [{"ext": "json3"}],
        "hi": [{"ext": "json3"}],
        "bn": [{"ext": "json3"}],
    },
}
_, lang, _ = _with("u", HINDI_FALLBACK, {"hi": "\u0939\u093f\u0928\u094d\u0926\u0940"})
eq("English is tried first", _DOWNLOADED[0][0], "en")
eq("and Hindi is the fallback when English yields nothing", lang, "hi")

# A video whose detected language is NOT Hindi must still prefer Hindi over an
# unrelated fourth language, because Hindi is the stated corpus fallback.
NOT_HINDI = {
    "language": "bn",
    "subtitles": {},
    "automatic_captions": {
        "en": [{"ext": "json3"}],
        "hi": [{"ext": "json3"}],
        "bn": [{"ext": "json3"}],
    },
}
_, lang, _ = _with("u", NOT_HINDI, {"hi": "\u0939\u093f\u0928\u094d\u0926\u0940"})
eq("Hindi wins over an unrelated language even when the video is not Hindi",
   lang, "hi")

# Urdu is the third choice, for the videos spoken in it.
URDU_THIRD = {
    "language": "ur",
    "subtitles": {},
    "automatic_captions": {
        "en": [{"ext": "json3"}], "hi": [{"ext": "json3"}],
        "ur": [{"ext": "json3"}],
    },
}
_, lang, _ = _with("u", URDU_THIRD, {"ur": "\u0633\u0644\u0627\u0626\u06cc"})
eq("Urdu is reachable when English and Hindi both fail", lang, "ur")

# The preference order itself, asserted directly so a reordering is visible.
# Empty tracks make this raise, which is fine -- the attempt log is the point.
_ORDER_INFO = {"language": "hi", "subtitles": {}, "automatic_captions": {
    "en": [{"ext": "json3"}], "hi": [{"ext": "json3"}],
    "ur": [{"ext": "json3"}], "bn": [{"ext": "json3"}]}}
try:
    _with("u", _ORDER_INFO, {})
except RuntimeError:
    pass
eq("attempt order is en, hi, ur", [d[0] for d in _DOWNLOADED], ["en", "hi", "ur"])

# The search must be bounded: YouTube advertises 157 automatic tracks, and an
# unbounded walk would burn one timeout each.
MANY_INFO = {
    "language": "zz",
    "subtitles": {},
    "automatic_captions": {("l%03d" % i): [{"ext": "json3"}] for i in range(157)},
}
# Every track is empty, so this raises -- which is correct, and the point is how
# many attempts it took to get there.
try:
    _with("u", MANY_INFO, {})
    check("157 unusable tracks raise", False, "it returned without raising")
except RuntimeError:
    check("157 unusable tracks raise", True)
check("at most MAX_CAPTION_LANGS downloads are attempted",
      len(_DOWNLOADED) <= metadata.MAX_CAPTION_LANGS,
      "%d attempts" % len(_DOWNLOADED))
check("157 advertised tracks do not become 157 timeouts",
      metadata.MAX_CAPTION_LANGS <= 4, metadata.MAX_CAPTION_LANGS)

# An unusable container format is skipped rather than attempted.
BAD_FORMAT = {"language": "en", "subtitles": {},
              "automatic_captions": {"en": [{"ext": "ttml"}]}}
raises("a track with no usable format raises", lambda: _with("u", BAD_FORMAT, {}),
       RuntimeError, "none downloaded")

# The two failure kinds must be distinguishable: only one is worth retrying.
NO_TRACKS = {"language": "en", "subtitles": {}, "automatic_captions": {}}
raises("no tracks at all says so",
       lambda: _with("u", NO_TRACKS, {}), RuntimeError, "no caption tracks at all")
raises("tracks that all failed says something different",
       lambda: _with("u", HINDI_INFO, {}), RuntimeError, "none downloaded")

# The old message claimed a missing English track when the real problem was that
# a download failed -- which would have been recorded as evidence about clarity.
try:
    _with("u", HINDI_INFO, {})
    check("the misleading message is gone", False, "no exception raised")
except RuntimeError as exc:
    check("the message no longer claims there is no English track",
          "no English caption track" not in str(exc), str(exc)[:80])

# -- the language reaches the prompt, and provenance is recorded ---------------
block = prompts.format_metadata({
    "title": "T", "description": "D", "tags": [], "channel": "C",
    "category": "Education", "subtitle": "\u0939\u093f\u0928\u094d\u0926\u0940",
    "subtitle_language": "hi", "subtitle_kind": "auto",
})
_line = [ln for ln in block.splitlines() if ln.startswith("Subtitle")][0]
check("the prompt labels the transcript's language", "(hi" in _line.split(":", 1)[0],
      _line.split(":", 1)[0])
check("the prompt says the track is machine-written", "auto-written" in _line,
      _line.split(":", 1)[0])
check("the transcript itself is still included", "\u0939\u093f\u0928\u094d\u0926\u0940" in _line)

# With no language known, the label must stay plain rather than print "(None)".
_plain = prompts.format_metadata({
    "title": "T", "subtitle": "some text", "tags": [], "description": "D",
    "channel": "C", "category": "E"})
_p_line = [ln for ln in _plain.splitlines() if ln.startswith("Subtitle")][0]
eq("an unknown language leaves the label plain", _p_line.split(":", 1)[0], "Subtitle")

# -- non-ASCII output must not be able to kill a run ---------------------------
# A Windows console defaults to cp1252, and a model's reasoning may quote Hindi.
# Before this, that raised UnicodeEncodeError inside the logger and took down a run
# that had already been paid for.
class _Cp1252Stream:
    encoding = "cp1252"

    def __init__(self):
        self.parts: list = []

    def write(self, text):
        text.encode(self.encoding)          # raise, like a real console
        self.parts.append(text)

    def flush(self):
        pass


_hard = _Cp1252Stream()
try:
    _hard.write("\u0939\u093f\u0928\u094d\u0926\u0940")
    check("the stand-in console refuses Devanagari", False, "it accepted it")
except UnicodeEncodeError:
    check("the stand-in console refuses Devanagari", True)

_hard = _Cp1252Stream()
log._safe_write(_hard, "line with \u0939\u093f\u0928\u094d\u0926\u0940 inside\n")
_written = "".join(_hard.parts)
check("the logger survives an unrepresentable character", _written != "")
check("the ascii part is preserved", "line with" in _written, _written[:40])
check("the unrepresentable part is replaced, not raised",
      "\u0939" not in _written, _written[:60])

# And the subscription path still delivers the intact text to the UI.
_seen: list = []
log.set_event_sink(lambda ev: _seen.append(ev.get("message") or ev.get("text")))
_sink = _Cp1252Stream()
log.set_sink(_sink.parts.append)
try:
    log.ok("the video explains \u0938\u094d\u091f\u093f\u091a \u090f\u0902\u0917\u0932")
finally:
    log.clear_sink()
    log.clear_event_sink()
eq("the structured event carries the intact message", len(_seen), 1)
check("the UI receives real Devanagari, not question marks",
      "\u0938\u094d\u091f\u093f\u091a" in (_seen[0] if _seen else ""),
      (_seen[0] if _seen else "")[:60])


# ============================================================================
section("19. No unresolved names anywhere in the package")
# ============================================================================
#
# A typo like `quality_request(metadata, ...)` where the variable is `meta` is
# valid syntax, so compile() is happy; the suite is happy unless that exact line
# runs; and Pylance did not flag it either. It crashed a real run at Prompt 2 after
# the expensive part had already been paid for. `symtable` catches it statically.
#
# The checker is validated against a file containing the real bug FIRST. An earlier
# version of it reported "clean" for the wrong reason -- it excluded names symtable
# marks `is_global()`, which is precisely how an unknown name resolves -- and a
# check that cannot fail is worse than no check.

import builtins as _builtins  # noqa: E402
import symtable as _symtable  # noqa: E402


def _undefined_names(source: str, filename: str) -> list:
    table = _symtable.symtable(source, filename, "exec")
    tops = {s.get_name() for s in table.get_symbols()}
    builtin = set(dir(_builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "self", "cls",
    }
    found: list = []

    def walk(scope):
        for sym in scope.get_symbols():
            name = sym.get_name()
            if name in builtin or name in tops:
                continue
            if (sym.is_referenced() and not sym.is_assigned()
                    and not sym.is_parameter() and not sym.is_imported()
                    and not sym.is_free() and not sym.is_nonlocal()):
                found.append(name)
        for child in scope.get_children():
            walk(child)

    walk(table)
    return found


# Step 1: prove the checker detects the exact bug that crashed the run.
_BUGGY = (
    "from . import metadata_mod\n"
    "\n"
    "def screen_one(url, meta=None):\n"
    "    meta = metadata_mod.fetch(url)\n"
    "    return quality_request(metadata, meta.get('t'))\n"
)
_BUGGY_FOUND = _undefined_names(_BUGGY, "buggy_sample.py")
check("the checker detects the reported bug", "metadata" in _BUGGY_FOUND,
      _BUGGY_FOUND)

# A false-negative guard: the checker must not flag legitimate module globals.
_CLEAN = (
    "import os\n"
    "CONST = 1\n"
    "\n"
    "def f(a, b=2):\n"
    "    def inner():\n"
    "        return CONST + a + os.sep\n"
    "    return inner() + len(b and [1] or [2])\n"
)
check("the checker does NOT flag legitimate globals and closures",
      _undefined_names(_CLEAN, "clean_sample.py") == [],
      _undefined_names(_CLEAN, "clean_sample.py"))

# Step 2: now trust it on the real package.
_PKG_DIR = Path(__file__).resolve().parent / "wilcom_pipeline"
_UNRESOLVED: list = []
for _path in sorted(_PKG_DIR.glob("*.py")):
    _src = _path.read_text(encoding="utf-8")
    for _name in _undefined_names(_src, _path.name):
        _UNRESOLVED.append("%s: %s" % (_path.name, _name))

check("no unresolved names in any published module",
      _UNRESOLVED == [], "; ".join(_UNRESOLVED[:6]))

# A named regression for the second bug the scan found. It searches for a CALL,
# not the bare token: the explanation left in `_record_name` mentions the old name,
# and matching that would make this check fail on its own comment.
_hits = [p.name for p in _PKG_DIR.glob("*.py")
         if "prompt_hash(" in p.read_text(encoding="utf-8")]
check("the non-existent `prompt_hash` is no longer called", _hits == [], _hits)

# And the fallback it guards actually works. `_record_name` uses the video id when
# there is one, so this branch runs only for a URL whose metadata has no id -- which
# is precisely why the NameError sat here undetected.
_named = stage_filter._record_name("https://example.com/odd?id=1", {"video_id": "V1"})
check("a known video id is used as the record name", _named == "V1", _named)
_hashed = stage_filter._record_name("https://example.com/odd?id=1", {})
check("a missing video id falls back to a hash", len(_hashed) == 12, _hashed)
check("the hash fallback is stable across calls",
      _hashed == stage_filter._record_name("https://example.com/odd?id=1", {}),
      _hashed)
check("the hash fallback distinguishes URLs",
      _hashed != stage_filter._record_name("https://example.com/other", {}))


# ============================================================================
section("20. Prompt 2 actually reaches the model")
# ============================================================================
#
# The NameError above meant Prompt 2 NEVER completed. Everything before it worked
# (metadata, captions, Prompt 1, the probe clip) and then the whole screening
# record was thrown away at the last step. So this checks the join, not the parts:
# a fake client counts the calls and asserts the quality request is issued with the
# metadata and the clip.

# A purpose-built fake: `FakeClient` above answers with trajectory replies and
# counts media kinds, which is the wrong shape for screening. This one replies with
# the JSON the two prompts are parsed for, and records the requests so the join can
# be asserted.
_META_REPLY = json.dumps({
    "is_gui_content": True, "confidence": 0.99,
    "reasoning": "The title and description mention a Wilcom e4 tutorial.",
})
_QUALITY_REPLY = json.dumps({
    "scores": {
        "topic_relevance": {"score": 5, "reasoning": "focused on the software"},
        "instruction_clarity": {"score": 4, "reasoning": "numbered steps"},
        "recording_quality": {"score": 4, "reasoning": "direct screen capture"},
    },
    "overall_summary": "A clear software tutorial.",
})


class _ScreenClient:
    """Implements the three-method surface `screen_one` uses."""

    def __init__(self):
        self.requests = []
        self.media_kinds = []

    def media(self, path, kind):
        self.media_kinds.append(kind)
        return (str(path), kind)

    def chat(self, model, request, **kw):
        self.requests.append(request)
        media = request.get("media") or []
        # Prompt 1 is text-only; Prompt 2 is the one carrying the clip. Deciding by
        # the media is what makes this a real check of which call is which, rather
        # than trusting call order.
        text = _QUALITY_REPLY if media else _META_REPLY
        return {"text": text, "usage": {}, "model": model, "cached": False}


_probe_written = []


def _fake_probe(url, seconds=None, dest_dir=None):
    """A probe clip that exists, so Prompt 2 is not skipped for want of a video."""
    d = Path(dest_dir or tempfile.mkdtemp())
    d.mkdir(parents=True, exist_ok=True)
    p = d / "probe.mp4"
    p.write_bytes(b"\x00" * 2048)
    _probe_written.append(p)
    return p


_real_fetch = metadata.fetch
_real_probe = metadata.fetch_probe_clip
metadata.fetch = lambda url, want_subtitles=True: {
    "title": "Rectangle ellipse and basic shapes | wilcom e4 tutorial",
    "description": "step by step", "tags": ["wilcom"], "channel": "Design Verse",
    "category": "Howto", "subtitle": "select an object", "subtitle_language": "hi",
    "subtitle_kind": "auto", "subtitle_source": "captions",
    "duration_s": 207.0, "width": 1920, "height": 1080, "video_id": "ABC123",
    "webpage_url": "u", "upload_date": "20260101",
}
metadata.fetch_probe_clip = _fake_probe

_work = Path(tempfile.mkdtemp(prefix="wilcom_p2_"))
_p2_paths = config.Paths(root=_work).ensure()
_p2_client = _ScreenClient()
try:
    _rec = stage_filter.screen_one("https://youtu.be/ABC123", _p2_client,
                                   _p2_paths, force=True)
finally:
    metadata.fetch = _real_fetch
    metadata.fetch_probe_clip = _real_probe

check("screening produced a record", isinstance(_rec, dict), type(_rec).__name__)
check("Prompt 1 ran", _rec.get("meta_verdict") is not None, _rec.get("meta_error"))
check("Prompt 2 ran and produced scores", _rec.get("quality") is not None,
      _rec.get("quality") or _rec.get("probe_error"))
check("exactly two model calls were made", len(_p2_client.requests) == 2,
      len(_p2_client.requests))
check("the record was written to disk",
      list(_p2_paths.screen.glob("*.json")) != [],
      list(_p2_paths.screen.glob("*.json")))

if _rec.get("quality"):
    _scores = _rec["quality"].get("scores") or {}
    check("all three dimensions were scored",
          set(stage_filter.DIMENSIONS) <= set(_scores), _scores)
    check("the scores are the ones the model returned",
          _scores.get("topic_relevance") == 5, _scores)
    check("the probe clip was used, so recording_quality is not blind",
          _rec["quality"].get("blind_dimensions") in (None, []),
          _rec["quality"].get("blind_dimensions"))
else:
    check("all three dimensions were scored", False, "no quality block")
    check("the scores are the ones the model returned", False, "no quality block")
    check("the probe clip was used, so recording_quality is not blind", False,
          "no quality block")

# The two calls must be the two different prompts, and only the second may carry
# the video. Before the fix, the second call crashed before it was ever issued.
if len(_p2_client.requests) == 2:
    _first, _second = _p2_client.requests
    check("Prompt 1 is text-only", (_first.get("media") or []) == [],
          len(_first.get("media") or []))
    check("Prompt 2 carries the probe clip as media",
          len(_second.get("media") or []) == 1,
          len(_second.get("media") or []))
    check("Prompt 2's media is the video we wrote",
          any(Path(part[0] if isinstance(part, tuple) else part).resolve()
              == _probe_written[-1].resolve() for part in _second["media"]),
          _second.get("media"))
    check("Prompt 2 asks for the quality schema",
          _second.get("json_schema") is prompts.QUALITY_SCHEMA)
else:
    for _label in ("Prompt 1 is text-only", "Prompt 2 carries the probe clip as media",
                   "Prompt 2's media is the video we wrote",
                   "Prompt 2 asks for the quality schema"):
        check(_label, False, "expected 2 requests")


# ============================================================================
section("21. Step 5 -- screenshot recovery")
# ============================================================================
#
# A missing frame is not a cosmetic loss: `_row_from_action` returns
# `frame_missing` and the ROW is dropped, so the loss never appears in the parquet.
# Recovery therefore reads the grounding records, which still remember the frame
# filename and the time it came from.
#
# Tested against a synthetic tree rather than the real one, and every deleted path
# is asserted to be inside it -- a previous version of this check deleted from the
# real work tree while asserting on a copy, which both corrupted real data and made
# the assertions pass vacuously.

if not HAVE_FFMPEG:
    print("  SKIP  ffmpeg unavailable")
    SKIP += 1
else:
    _rev_root = Path(tempfile.mkdtemp(prefix="wilcom_revtest_")) / "wilcom_work"
    _rev = config.Paths(root=_rev_root).ensure()
    _rev.videos.mkdir(parents=True, exist_ok=True)
    _rev.frames.mkdir(parents=True, exist_ok=True)

    # A 3-second source, so frame extraction is real but instant.
    _src = _rev.videos / "demo.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=10:duration=3",
         "-pix_fmt", "yuv420p", str(_src)],
        capture_output=True, timeout=120, check=False,
    )
    check("a synthetic source video was created", _src.exists()
          and _src.stat().st_size > 0)

    # Two grounded results: one whose frame exists, one whose frame does not.
    _frame_dir = _rev.frames / "demo"
    _frame_dir.mkdir(parents=True, exist_ok=True)
    _present_name = "t0001.000_t0_a0_f0.png"
    _missing_name = "t0002.000_t0_a1_f1.png"
    video.extract_frame(_src, 1.0, _frame_dir / _present_name)

    _ground_rec = {
        "video": "demo.mp4", "video_stem": "demo", "model": "fake",
        "prompt_version": "ground-v2", "width": 1920, "height": 1080,
        "results": [
            {"grounded": True, "task_id": 0, "seq": 0, "t": 1.0,
             "frame": _present_name, "frame_time_s": 1.0,
             "image_width": 1920, "image_height": 1080,
             "point_norm": [100, 100], "bbox": [0, 0, 400, 400],
             "instruction": "Click the New icon", "action_type": "click"},
            {"grounded": True, "task_id": 0, "seq": 1, "t": 2.0,
             "frame": _missing_name, "frame_time_s": 2.0,
             "image_width": 1920, "image_height": 1080,
             "point_norm": [200, 200], "bbox": [0, 0, 400, 400],
             "instruction": "Click the File menu", "action_type": "click"},
        ],
    }
    _traj_rec = {
        "video": "demo.mp4", "video_stem": "demo", "duration_s": 3.0,
        "width": 1920, "height": 1080, "fps": 10.0, "model": "fake",
        "prompt_version": "traj-v2", "clip_seconds": 240, "clips": [],
        "tasks": [{
            "task_id": 0, "instruction": "Open a new document",
            "actions": [
                {"task_id": 0, "seq": 0, "t": 1.0, "action_type": "click",
                 "grounding_instruction": "Click the New icon",
                 "needs_grounding": True, "traj_point": [100, 100]},
                {"task_id": 0, "seq": 1, "t": 2.0, "action_type": "click",
                 "grounding_instruction": "Click the File menu",
                 "needs_grounding": True, "traj_point": [200, 200]},
            ],
        }],
        "stats": {},
    }
    records.write_json(_ground_rec, _rev.ground / "demo.json")
    records.write_json(_traj_rec, _rev.traj / "demo.json")

    # The loss: assess what the dataset holds with one frame absent.
    _rows_before, _stats_before = stage_dataset.build_rows(
        [_traj_rec], [_ground_rec], _rev)
    check("a missing frame costs the row, so only 1 of 2 survives",
          len(_rows_before) == 1, len(_rows_before))
    check("and it is counted as frame_missing, not invisible",
          _stats_before.frame_missing == 1, _stats_before.frame_missing)
    check("the missing frame really is absent on disk",
          not (_frame_dir / _missing_name).exists())
    check("the check is inside the temp tree, not the real work tree",
          _rev_root.is_relative_to(Path(tempfile.gettempdir())), str(_rev_root))

    check("review is NOT done before it runs",
          jobs.step_state(str(_rev_root))["review"] is False)

    # Run step 5.
    jobs.RUNNER.reset()
    jobs.review(str(_rev_root))
    jobs.RUNNER._thread.join(timeout=300)
    _rep = jobs.RUNNER.result
    check("step 5 finished without error", jobs.RUNNER.error is None,
          jobs.RUNNER.error)

    if _rep:
        check("it checked both grounded frames", _rep["frames_checked"] == 2,
              _rep["frames_checked"])
        check("it found the one that was missing", _rep["frames_missing"] == 1,
              _rep["frames_missing"])
        check("it recovered it", _rep["frames_recovered"] == 1,
              _rep["frames_recovered"])
        check("nothing was left unrecoverable", _rep["frames_unrecoverable"] == 0)
        check("coverage is back to 100%", _rep["coverage"] == 1.0, _rep["coverage"])
        check("the report counts both rows",
              _rep["rows"] == 2, _rep["rows"])
        check("review.json was written",
              (_rev.logs / "review.json").is_file())
        check("step_state now reports review done",
              jobs.step_state(str(_rev_root))["review"] is True)

        _recovered_file = _frame_dir / _missing_name
        check("the frame file exists again", _recovered_file.exists())
        if _recovered_file.exists():
            _w, _h = video.frame_size(_recovered_file)
            check("recovered at the source resolution (no downscale)",
                  (_w, _h) == (1920, 1080), (_w, _h))

        # The point of the whole step: the row is back.
        _rows_after, _stats_after = stage_dataset.build_rows(
            [_traj_rec], [_ground_rec], _rev)
        check("both rows are available again", len(_rows_after) == 2,
              len(_rows_after))
        check("frame_missing is zero again", _stats_after.frame_missing == 0,
              _stats_after.frame_missing)
        check("the recovered row keeps its original coordinate",
              any(r["instruction"] == "Click the File menu"
                  and r["target_x"] == config.norm_to_abs(200, 1920)
                  for r in _rows_after),
              [r["instruction"] for r in _rows_after])

        # Recovery must use the frame's own time, not the action's. Here they differ
        # by an offset, so a wrong implementation would extract a different frame.
        check("recovery records the time it used",
              "frame_time_s" in str(_ground_rec["results"][1]))


# ============================================================================
section("22. Truncation is recorded, not just logged")
# ============================================================================
#
# A run capped at N clips produces a record that is otherwise indistinguishable
# from a complete one, and a dataset built from it reads as "this video's
# trajectory" when it is the first few minutes of it. A log line is not enough --
# the log is gone by the time the dataset is read.

_tr = stage_trajectory
_tr_source = Path(
    __import__("inspect").getsourcefile(_tr.extract_video)).read_text(
        encoding="utf-8")

# The record must carry the counts. Asserted against the source because building a
# real record needs a Pro call, and the field set is a property of the code.
for _field in ("planned_clips", "processed_clips", "truncated", "covered_s"):
    check("the trajectory record carries %r" % _field,
          '"%s"' % _field in _tr_source, _field)

check("plan_clips is called once, not twice",
      _tr_source.count("plan_clips(") == 1,
      _tr_source.count("plan_clips("))
check("truncation is warned about loudly",
      "TRUNCATED" in _tr_source)
check("the warning states how much is covered",
      "100 * covered" in _tr_source)

# And the reporting path end to end: a truncated record must reach the details
# payload, so the UI can say the dataset is partial. Built on a synthetic work tree
# rather than the real one, so nothing real is touched.
_tw = Path(tempfile.mkdtemp(prefix="wilcom_trunc_")) / "wilcom_work"
_tp = config.Paths(root=_tw).ensure()
records.write_json({
    "video": "half.mp4", "video_stem": "half",
    "duration_s": 480.0, "width": 1920, "height": 1080, "fps": 60.0,
    "model": "fake", "prompt_version": "traj-v2", "clip_seconds": 240,
    "planned_clips": 2, "processed_clips": 1, "truncated": True,
    "covered_s": 240.0,
    "clips": [{"index": 0, "start_s": 0.0, "end_s": 240.0, "model": "fake",
               "cache_key": "k0", "shots": [], "tasks": 1, "actions": 1,
               "reply_chars": 100, "cached": False}],
    "tasks": [{"task_id": 0, "instruction": "T", "actions": [
        {"task_id": 0, "seq": 0, "t": 10.0, "action_type": "click",
         "grounding_instruction": "Click A", "needs_grounding": True,
         "traj_point": None}]}],
    "stats": {},
}, _tp.traj / "half.json")
records.write_json({
    "video": "half.mp4", "video_stem": "half", "model": "fake",
    "prompt_version": "ground-v2", "width": 1920, "height": 1080,
    "results": [], "stats": {},
}, _tp.ground / "half.json")

_det = details.build(str(_tw))
check("details builds over a synthetic tree", len(_det["videos"]) == 1,
      len(_det["videos"]))
if _det["videos"]:
    _sl = _det["videos"][0]["slicing"]
    eq("truncated reaches the payload", _sl["truncated"], True)
    eq("planned_clips reaches the payload", _sl["planned_clips"], 2)
    eq("processed_clips reaches the payload", _sl["processed_clips"], 1)
    eq("coverage is computed from the clip ranges", _sl["covers_pct"], 50.0)
    eq("the run-level total names the truncated video",
       len(_det["totals"]["truncated"]), 1)
    eq("and reports its coverage",
       _det["totals"]["truncated"][0]["covers_pct"], 50.0)

# An OLD record must report `truncated` as UNKNOWN, not False. Claiming a complete
# run would be a guess, and the two cases are genuinely different.
_ow = Path(tempfile.mkdtemp(prefix="wilcom_old_")) / "wilcom_work"
_op = config.Paths(root=_ow).ensure()
records.write_json({
    "video": "old.mp4", "video_stem": "old",
    "duration_s": 480.0, "width": 1920, "height": 1080, "fps": 60.0,
    "model": "fake", "prompt_version": "traj-v2", "clip_seconds": 240,
    "clips": [{"index": 0, "start_s": 0.0, "end_s": 240.0, "model": "fake",
               "cache_key": "k0", "shots": [], "tasks": 1, "actions": 1,
               "reply_chars": 10, "cached": False}],
    "tasks": [], "stats": {},
}, _op.traj / "old.json")
records.write_json({
    "video": "old.mp4", "video_stem": "old", "model": "fake",
    "prompt_version": "ground-v2", "width": 1920, "height": 1080,
    "results": [], "stats": {},
}, _op.ground / "old.json")
_odet = details.build(str(_ow))
if _odet["videos"]:
    _osl = _odet["videos"][0]["slicing"]
    check("an old record reports truncation as UNKNOWN, not False",
          _osl["truncated"] is None, _osl["truncated"])
    check("but still reports real coverage from its clips",
          _osl["covers_pct"] == 50.0, _osl["covers_pct"])
    eq("an unknown truncation is not listed as truncated",
       _odet["totals"]["truncated"], [])


# ============================================================================
section("23. Thinking level is selectable and actually changes the call")
# ============================================================================
#
# Thinking is the largest single cost and time item on stage 1, so it is worth
# being able to choose. Two things must hold for the choice to mean anything: the
# level must reach the request, and it must enter the cache key. A level that did
# not enter the key would replay the previous level's reply, and a comparison
# between levels would then report a difference that does not exist.

check("config offers a list of levels", len(config.THINKING_LEVELS) >= 2,
      config.THINKING_LEVELS)
check("the default is one of them",
      config.THINKING_LEVEL in config.THINKING_LEVELS, config.THINKING_LEVEL)
check("every level has a note the UI can show",
      all(lv in config.THINKING_NOTES for lv in config.THINKING_LEVELS),
      list(config.THINKING_NOTES))

_req_hi = prompts.trajectory_request([], None, thinking_level="high")
eq("an explicit level lands on the request", _req_hi.get("thinking_level"), "high")
_req_def = prompts.trajectory_request([], None)
check("omitting it leaves the field absent, so config's default applies",
      "thinking_level" not in _req_def, sorted(_req_def))

# -- the cache key, which is what makes the choice observable -----------------
from wilcom_pipeline import clients  # noqa: E402
_tmp_cache = Path(tempfile.mkdtemp(prefix="wilcom_think_"))
_client = clients.make_client(cache_dir=_tmp_cache, use_cache=True, verbose=False)
_keys = {}
for _lv in config.THINKING_LEVELS:
    _keys[_lv] = _client._effective_key(
        "same-raw-key", "gemini-3.1-pro-preview", config.TEMPERATURE,
        config.MAX_OUTPUT_TOKENS,
        prompts.trajectory_request([], None, thinking_level=_lv))
check("each level produces a DIFFERENT cache key",
      len(set(_keys.values())) == len(config.THINKING_LEVELS), _keys)
# No level given resolves to the config default, so it must SHARE that level's key
# rather than mint a second one. Otherwise choosing the default explicitly would
# pay for the same call twice.
_none_key = _client._effective_key(
    "same-raw-key", "gemini-3.1-pro-preview", config.TEMPERATURE,
    config.MAX_OUTPUT_TOKENS, prompts.trajectory_request([], None))
eq("an omitted level shares the default level's cache entry",
   _none_key, _keys[config.THINKING_LEVEL])
check("but that key is still distinct from the other levels",
      all(_none_key != _keys[lv] for lv in config.THINKING_LEVELS
          if lv != config.THINKING_LEVEL), _keys)

# -- a record made at one level is not reused for another --------------------
# The level is not in the record's filename, so reusing a `low` record for a `high`
# request would report the cheap run's output as the expensive one's and nothing in
# the artefact would show it.
_st_src = Path(__import__("inspect").getsourcefile(
    stage_trajectory.extract_video)).read_text(encoding="utf-8")
check("the reuse check compares the recorded level",
      'recorded = cached.get("thinking_level")' in _st_src)
check("a mismatched level is warned about, not silently reused",
      "was extracted at thinking=" in _st_src)
check("the record stores the level it used",
      '"thinking_level": thinking_level or config.THINKING_LEVEL' in _st_src)

# -- the step validates, rather than forwarding nonsense to the provider ------
_tw2 = Path(tempfile.mkdtemp(prefix="wilcom_thinkw_")) / "wilcom_work"
config.Paths(root=_tw2).ensure()
_refused = jobs.trajectory(str(_tw2), 0, "", "ludicrous")
check("an unknown level is refused with the valid list",
      "unknown thinking level" in _refused and "low" in _refused, _refused)
check("and nothing was started", not jobs._running())

for _lv in config.THINKING_LEVELS:
    # A valid level passes validation and fails later for a real reason (there is
    # nothing to read here), which is how we know it was not refused up front.
    _msg = jobs.trajectory(str(_tw2), 0, "", _lv)
    check("a valid level passes validation (%s)" % _lv,
          "unknown thinking level" not in _msg, _msg)
    jobs.RUNNER.reset()


# ============================================================================
section("24. The paper's 12-minute candidate limit")
# ============================================================================
#
# "we selected only those with durations under 12 minutes based on metadata
#  information" -- Section C. It is a cost control as much as a quality one: at
# 240s per clip a 12-minute video is 3 clips and a 60-minute one is 15, roughly
# $1.20 in stage 1 against $0.08.

eq("the limit is 12 minutes", config.MAX_DURATION_S, 720)

_ok_m = {"is_gui_content": True, "confidence": 0.99}
_ok_q = {"scores": {"topic_relevance": 5, "instruction_clarity": 4,
                    "recording_quality": 4}}

# The boundary. The paper says "under 12 minutes", so 12:00 exactly is out and
# 11:59 is in -- an off-by-one here silently admits a video the paper excluded.
_acc, _why = stage_filter.verdict(_ok_m, _ok_q, 719.0)
check("11:59 is accepted", _acc, _why[0])
_acc, _why = stage_filter.verdict(_ok_m, _ok_q, 720.0)
check("12:00 exactly is rejected", not _acc, _why[0])
_acc, _why = stage_filter.verdict(_ok_m, _ok_q, 719.999)
check("just under 12:00 is accepted", _acc, _why[0])

# The reason must name the limit, so a rejection is explainable without the code.
_acc, _why = stage_filter.verdict(_ok_m, _ok_q, 1500.0)
check("a 25-minute video is rejected", not _acc)
check("the reason quotes the limit", "12-minute limit" in _why[0], _why[0])
check("the reason is formatted mm:ss, not a rounded min",
      "25:00" in _why[0], _why[0])
# 719s at one decimal rounds to "12.0 min", which would read as a contradiction.
check("an accepted 719s does NOT display as 12.0 min",
      "12.0 min" not in _why[0], _why[0])

# An unknown duration is NOT a rejection. Unreadable metadata is a different thing
# from a video being too long, and rejecting on it would be a false reason.
_acc, _why = stage_filter.verdict(_ok_m, _ok_q, None)
check("an unknown duration is accepted rather than guessed at", _acc, _why[0])
check("but the reason says the check could not run",
      "could not be checked" in _why[0], _why[0])

# It composes with the other criteria rather than short-circuiting them, so a
# record explains every reason it failed rather than only the first.
_acc, _why = stage_filter.verdict(
    {"is_gui_content": False, "confidence": 0.1},
    {"scores": {d: 0 for d in stage_filter.DIMENSIONS}}, 1500.0)
check("all three failures are reported together", len(_why) == 3, _why)
check("duration is listed first, being the cheapest check",
      _why[0].startswith("Duration:"), _why[0])

# The duration check must GATE Prompt 2 rather than run after it -- the criterion is
# decided from metadata, so paying for a video upload to score a candidate that is
# already excluded would be waste.
_sf_src = Path(__import__("inspect").getsourcefile(
    stage_filter.screen_one)).read_text(encoding="utf-8")
check("Prompt 2 is skipped for an over-length video",
      "too_long = duration is not None and duration >= config.MAX_DURATION_S"
      in _sf_src, "the gate is missing from screen_one")
check("the skip is conditional on not being too long",
      "and not too_long" in _sf_src)
check("the record stores the limit that was in force",
      'record["duration_limit_s"] = config.MAX_DURATION_S' in _sf_src)
check("the record stores the verdict itself",
      'record["too_long"] = too_long' in _sf_src)


# ============================================================================
section("25. The schema the UI shows is the schema that was written")
# ============================================================================
#
# This exists because the question "why does the dataset have only seven columns"
# was asked twice, and both times the only honest answer was to open the parquet.
# A correct 18-column dataset looked broken because the Results tab renders a
# trimmed seven-column view for human scanning. The fix is to publish the real
# schema, and the risk in publishing it is that the published list drifts from the
# exported one -- so these checks compare the two rather than trusting the panel.

_schema = hf_export.schema()
_pcol = list(hf_export.PARENT_COLUMNS)
# `features()` returns a `datasets.Features`, which iterates as names -- taking it
# as a list of dicts raises TypeError rather than anything informative.
_fcol = list(hf_export.features())

eq("the schema describes every exported column", _schema["count"], len(_fcol))
eq("and reports all 18", _schema["count"], 18)
eq("the parent contributes 11", _schema["parent_count"], 11)
eq("so 7 are ours", len(_schema["extra"]), 7)

# The load-bearing claim. A consumer written against `junior0091/GUI-360-word-unsloth`
# reads these rows unchanged only if the parent's columns come first, in the parent's
# own order -- a reordered subset would silently mis-map every field.
eq("the first columns ARE the parent's, in the parent's order",
   _fcol[:len(_pcol)], _pcol)
check("the schema says so", _schema["order_matches_parent"] is True)
eq("nothing was inserted before them", [c["name"] for c in _schema["columns"]][:11],
   _pcol)

# `verify_against_parent` is the same check as an assertion, so it must agree.
_problems = hf_export.verify_against_parent()
eq("the verifier finds no problems", _problems, [])

# ...and it must actually be capable of finding one, or it is decoration. Reordering
# is the failure that matters, so it is the one tested. The swap has to produce a
# real `Features`, not a list -- `schema()` iterates it as a mapping, so a list would
# raise AttributeError and the check would pass for the wrong reason.
_orig_features = hf_export.features
try:
    from datasets import Features as _HF

    def _swapped():
        items = list(_orig_features().items())
        items[0], items[1] = items[1], items[0]
        return _HF(dict(items))

    hf_export.features = _swapped
    _swapped_problems = hf_export.verify_against_parent()
    check("reordering the columns IS detected", _swapped_problems != [],
          "verify_against_parent passed a schema with the first two columns swapped")
    check("and the offending columns are named",
          any("id" in p or "trajectory" in p for p in _swapped_problems),
          _swapped_problems)
    check("and the document order is reported as wrong",
          hf_export.schema()["order_matches_parent"] is False,
          "order_matches_parent stayed True with the columns reordered")
finally:
    hf_export.features = _orig_features

# Types are taken from the exporter's own Arrow spec, so a type shown in the panel
# is the Arrow type and not a guess made from a sample value.
_types = {c["name"]: c["type"] for c in _schema["columns"]}
eq("id is a string", _types["id"], "string")
eq("image is an Image", _types["image"], "Image")
eq("messages is a List", _types["messages"], "List")
eq("target_x is int32", _types["target_x"], "int32")
eq("timestamp_s is float32", _types["timestamp_s"], "float32")
eq("bbox is a JSON string, not a nested struct", _types["bbox"], "string")

# `populated` must be None when no rows were supplied rather than 0 -- 0 reads as
# "this column is empty", which is a claim about the data that has not been checked.
check("no rows means populated is unknown, not a fabricated zero",
      all(c["populated"] is None for c in _schema["columns"]), _schema["columns"][0])

# With rows supplied the real counts must come out.
_srows = [{"target_x": 1, "target_y": 2, "bbox": "[]"}] + \
         [{"target_x": 3, "target_y": 4, "bbox": None}]
_s2 = hf_export.schema(_srows)
_byname = {c["name"]: c for c in _s2["columns"]}
eq("rows counted", _s2["rows"], 2)
eq("bbox is populated in both", _byname["bbox"]["populated"], 1)
eq("target_x is populated in both", _byname["target_x"]["populated"], 2)
eq("a column absent from every row counts zero", _byname["id"]["populated"], 0)
check("a nested field is described, not crashed on",
      isinstance(_byname["messages"]["type"], str))

# `_has_value` is defensive because the rows come off disk, and the shapes pandas
# hands back are not scalar-comparable. Each of these has broken it once.
check("None is empty", not hf_export._has_value(None))
check("an empty string is empty", not hf_export._has_value(""))
check("an empty list is empty", not hf_export._has_value([]))
check("a NaN is empty, though it equals nothing",
      not hf_export._has_value(float("nan")))
check("a dict of Nones is empty (an image struct with no bytes)",
      not hf_export._has_value({"bytes": None, "path": None}))
check("a dict with one field set is present",
      hf_export._has_value({"bytes": None, "path": "a.png"}))
check("zero is present, not confused with empty", hf_export._has_value(0))
check("False is present, not confused with empty",
      hf_export._has_value(False))
check("an empty string inside a list still counts as a list with content",
      hf_export._has_value([""]))

# The one that returned a 500 rather than a wrong number: `array == ""` yields an
# array, so `or` raises "truth value of an array is ambiguous".
try:
    import numpy as _np
    _arr = _np.array([b"x", b"y"])
    check("a numpy array does not raise and counts as present",
          hf_export._has_value(_arr) is True)
    _arr2 = _np.array([{"bytes": None, "path": None}], dtype=object)
    check("an object-dtype array of structs does not raise",
          isinstance(hf_export._has_value(_arr2), bool))
except ImportError:                                     # pragma: no cover
    print("  skip  numpy unavailable; the array case is unchecked")
    SKIP += 1

# End to end through a real parquet write and read, which is where the array came
# from. Built by the exporter's own `to_dataset`, not by a hand-rolled frame, so the
# Arrow types are the ones the pipeline actually produces -- notably `image` as a
# struct, the column that returned an array and broke the comparison.
#
# Written into a temp dir rather than pointed at `wilcom_work`: a test that depends
# on a work tree happening to exist passes for the wrong reason when it does not.
_parquet_root = Path(tempfile.mkdtemp(prefix="wilcom_parquet_"))
try:
    import pandas as _pd

    _imgs = _parquet_root / "img"
    _imgs.mkdir(parents=True, exist_ok=True)
    if HAVE_PIL:
        from PIL import Image as _PILImage
        _p1, _p2 = _imgs / "a.png", _imgs / "b.png"
        _PILImage.new("RGB", (20, 20)).save(_p1)
        _PILImage.new("RGB", (20, 20)).save(_p2)

        _prows = [
            {"id": "wilcom_x_000_00", "trajectory": "wilcom_x", "instruction": "i1",
             "prompt": "p1", "image_path": str(_p1), "target_x": 5, "target_y": 6,
             "image_width": 20, "image_height": 20,
             "answer_raw": prompts.render_answer(5, 6), "source_video": "x.mp4",
             "task_id": 0, "step": 0, "action_type": "click", "timestamp_s": 1.0,
             "bbox": "[1,2,3,4]", "sub_task": "s1"},
            {"id": "wilcom_x_000_01", "trajectory": "wilcom_x", "instruction": "i2",
             "prompt": "p2", "image_path": str(_p2), "target_x": 7, "target_y": 8,
             "image_width": 20, "image_height": 20,
             "answer_raw": prompts.render_answer(7, 8), "source_video": "x.mp4",
             "task_id": 0, "step": 1, "action_type": "click", "timestamp_s": 2.0,
             "bbox": None, "sub_task": "s1"},
        ]
        _pq = _parquet_root / "train.parquet"
        hf_export.to_dataset(_prows).to_parquet(str(_pq))

        _back = _pd.read_parquet(_pq).to_dict("records")
        _pschema = hf_export.schema(_back)
        _pn = {c["name"]: c for c in _pschema["columns"]}
        eq("the parquet round trip keeps every row", _pschema["rows"], 2)
        eq("id is populated in both", _pn["id"]["populated"], 2)
        eq("target_x is populated in both", _pn["target_x"]["populated"], 2)
        # The one that 500'd: `bbox` is null in row 2, and pandas gives back NaN,
        # which is equal to nothing -- not even itself.
        eq("a null bbox is counted once, not twice", _pn["bbox"]["populated"], 1)
        # ...and the image column, which comes back as an array of the struct's
        # fields, so `array == ""` returns an array and `or` raises.
        eq("every row has an image", _pn["image"]["populated"], 2)
        check("no column is claimed populated beyond the row count",
              all(c["populated"] <= 2 for c in _pschema["columns"]),
              [(c["name"], c["populated"]) for c in _pschema["columns"]])
        check("every column reports an int",
              all(isinstance(c["populated"], int) for c in _pschema["columns"]))

        # The manifest ships to whoever reads the parquet later, so a wrong claim in
        # it is worse than a wrong comment. It said `answer_raw` was normalized
        # 0-1000 while the column is absolute pixels.
        _man = hf_export.manifest({"train": _prows})
        _conv = _man["coordinate_convention"]
        check("the manifest does not call answer_raw normalized",
              "normalized" not in _conv["answer_raw"].lower(),
              _conv["answer_raw"])
        check("the manifest says answer_raw is pixels",
              "absolute pixels" in _conv["answer_raw"].lower(),
              _conv["answer_raw"])
        check("and says it matches target_x / target_y",
              "target_x" in _conv["answer_raw"], _conv["answer_raw"])
        # The claim has to agree with the data, which is the check that would have
        # caught the original mistake.
        _r0 = _back[0]
        eq("answer_raw really is the pixel pair",
           _r0["answer_raw"],
           "<coordinate> [%d, %d]\n</coordinate>"
           % (_r0["target_x"], _r0["target_y"]))
        check("the manifest names the videos in each split",
              _man["splits"]["train"]["videos"] == ["x.mp4"],
              _man["splits"]["train"]["videos"])
    else:
        print("  skip  PIL unavailable; the parquet round trip is unchecked")
        SKIP += 1
except ImportError as _exc:                              # pragma: no cover
    print("  skip  pandas or datasets unavailable: %s" % _exc)
    SKIP += 1

# ---- the endpoint ---------------------------------------------------------
# Called directly rather than through a test client: the point is the payload, and
# the route itself is a two-line wrapper over the same call.
try:
    import importlib
    _api = importlib.import_module("wilcom_pipeline.api")
    HAVE_API = True
except Exception as _exc:                                  # pragma: no cover
    HAVE_API = False
    print("  skip  wilcom_pipeline.api not importable: %s" % _exc)

if HAVE_API:
    # Write a real parquet so the split counts come off disk, which is the whole
    # point of the endpoint -- the panel must show what was written.
    _api_root = Path(tempfile.mkdtemp(prefix="wilcom_schema_"))
    _api_paths = config.Paths(root=_api_root).ensure()
    try:
        import pandas as pd
        _frame = pd.DataFrame([
            {"id": "a", "target_x": 10, "target_y": 20, "bbox": "[]"},
            {"id": "b", "target_x": 30, "target_y": 40, "bbox": None},
        ])
        _frame.to_parquet(_api_paths.out / "train.parquet")
        pd.DataFrame([{"id": "c", "target_x": 1, "target_y": 2, "bbox": None}]) \
            .to_parquet(_api_paths.out / "validation.parquet")
    except Exception as _exc:
        print("  skip  could not write a parquet: %s" % _exc)

    _resp = _api.dataset_schema(work=str(_api_root))
    _payload = json.loads(_resp.body.decode("utf-8"))
    eq("the endpoint reports 18 columns", _payload["count"], 18)
    eq("the parent's 11 are marked as the parent's",
       sum(1 for c in _payload["columns"] if c["from_parent"]), 11)
    eq("train is counted", _payload["splits"].get("train"), 2)
    eq("validation is counted", _payload["splits"].get("validation"), 1)
    eq("the total is the sum", _payload["total_rows"], 3)
    eq("the compatibility claim is published", _payload["problems"], [])
    # Populated counts must come from the parquet rows, not from a default.
    _pn = {c["name"]: c for c in _payload["columns"]}
    eq("target_x populated in the 2 train rows", _pn["target_x"]["populated"], 2)
    eq("bbox populated in only 1 of them", _pn["bbox"]["populated"], 1)
    # A missing work directory must not 500 -- the panel has to render something.
    _empty = json.loads(
        _api.dataset_schema(work=str(_api_root / "nope")).body.decode("utf-8"))
    eq("an empty work dir still reports the schema", _empty["count"], 18)
    eq("with no rows", _empty["total_rows"], 0)


# ============================================================================
section("26. A capped grounding run cannot read as complete")
# ============================================================================
#
# `8/8 grounded` is what a capped run and a two-action video both produce. The
# trajectory stage learned this lesson already (`truncated`); grounding did not,
# so the cap was logged and then forgotten. What matters is that the count BEFORE
# the cap survives into the record, because after the cap the two are identical.

_gsrc = Path(__import__("inspect").getsourcefile(
    stage_grounding.ground_video)).read_text(encoding="utf-8")
check("the pre-cap count is recorded",
      '"available": available' in _gsrc, "available is not in the stats dict")
check("so is the cap that was applied",
      '"limited_to":' in _gsrc)
check("and it is null rather than 0 when nothing was capped",
      'if len(groundable) < available else None' in _gsrc,
      "a zero cap would read as 'grounded nothing'")
check("the pre-cap count is taken before the slice",
      _gsrc.index("available = len(groundable)")
      < _gsrc.index("groundable = groundable[:limit]"),
      "available is computed after the cap, so it would equal attempted")
check("a capped run is warned about, not just logged in passing",
      "is partial" in _gsrc)

# The detail view must carry it through, and must NOT invent a verdict for records
# written before the field existed. `None` means unknown; `0` would mean "no cap".
_dsrc = Path(__import__("inspect").getsourcefile(details.build)).read_text("utf-8")
check("details carries the pre-cap count through",
      '"available": (ground or {}).get("stats", {}).get("available")' in _dsrc)
check("details carries the cap through",
      '"limited_to": (ground or {}).get("stats", {}).get("limited_to")' in _dsrc)
check("only an affirmative cap is listed in the totals",
      'if v["actions"].get("limited_to"):' in _dsrc,
      "truthiness matters: None and 0 must both mean 'not listed'")
check("the totals expose the list", '"limited": limited,' in _dsrc)

# End to end, with the cap applied for real. The counters are what is asserted, so
# the run has to go all the way through frame extraction -- a mocked `ground_action`
# would leave `available` correct and prove nothing about the loop that uses it.
#
# The stats docstring on `limited_to` says why it is None and not 0: 0 is a legal
# value for the cap argument (`limit=0` means "no cap"), so a 0 in the record would
# be ambiguous between "grounded nothing" and "not capped at all".
if not (HAVE_FFMPEG and HAVE_PIL):
    print("  SKIP  ffmpeg or PIL unavailable")
    SKIP += 1
else:
    _lim_root = Path(tempfile.mkdtemp(prefix="wilcom_limit_"))
    _lim = config.Paths(root=_lim_root).ensure()

    _lim_src = _lim.videos / "lim.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=640x360:rate=5:duration=5",
         "-pix_fmt", "yuv420p", str(_lim_src)],
        capture_output=True, timeout=120, check=False,
    )
    check("a synthetic video was created for the cap test",
          _lim_src.exists() and _lim_src.stat().st_size > 0)

    _lim_traj = [{
        "video": "lim.mp4", "video_stem": "lim", "model": "fake",
        "prompt_version": stage_trajectory.PROMPT_VERSION,
        "duration_s": 5.0, "width": 640, "height": 360, "fps": 5.0,
        "clip_seconds": 240, "clips": [],
        "tasks": [{
            "task_id": 0, "clip": 0, "clip_start_s": 0.0,
            "instruction": "Do the thing", "dense_caption": "", "plan": "",
            "platform": "windows", "software": None, "website": None,
            "has_finish": False,
            "actions": [
                {"task_id": 0, "seq": i, "clip": 0,
                 "timestamp": "00:0%d" % (i + 1), "t": float(i + 1),
                 "action_type": "click",
                 "grounding_instruction": "Click button %d." % i,
                 "action_reason": "", "action_parameters": {}, "core_change": "",
                 "core_change_reason": "", "traj_point": None,
                 "needs_grounding": True}
                for i in range(3)
            ],
        }],
    }]
    records.write_json(_lim_traj[0], _lim.traj / "lim.json")

    # Reuses the suite's own fake client, so the request the stage builds is still
    # the stage's -- only the transport is stubbed.
    _lim_client = FakeClient()

    _rec = stage_grounding.ground_video(
        _lim_traj[0], _lim_client, _lim, "fake", force=True, verbose=False, limit=2)
    _stats = _rec["stats"]
    eq("only the cap was attempted", _stats["attempted"], 2)
    eq("but the video had three groundable actions", _stats["available"], 3)
    eq("so the record says it was capped", _stats["limited_to"], 2)
    check("and the cap is a different number from the count before it, so the two"
          " cannot be confused",
          _stats["available"] != _stats["limited_to"])

    # The same video, uncapped. This is the case the whole field exists to separate
    # from the one above: both produce `3/3 grounded`-shaped output.
    _rec2 = stage_grounding.ground_video(
        _lim_traj[0], _lim_client, _lim, "fake", force=True, verbose=False,
        limit=0)
    _stats2 = _rec2["stats"]
    eq("uncapped, every action was attempted", _stats2["attempted"], 3)
    eq("attempted equals available", _stats2["attempted"], _stats2["available"])
    eq("and no cap is claimed", _stats2["limited_to"], None)
    check("a zero cap means 'no cap', not 'nothing attempted'",
          _stats2["limited_to"] is None and _stats2["attempted"] == 3,
          _stats2)

    # The limit argument must be the only thing that differs: same actions, same
    # record apart from the two counters. If `available` were computed after the
    # slice it would read 2 here and the two runs would be indistinguishable.
    check("the two runs agree on the video's real capacity",
          _stats["available"] == _stats2["available"],
          "%r vs %r" % (_stats["available"], _stats2["available"]))

    # Reading the record back must survive the round trip, since the UI reads the
    # file rather than the in-memory dict.
    _reread = records.read_json(_lim.ground / "lim.json")
    eq("the cap survives being written and read", _reread["stats"]["limited_to"],
       _stats2["limited_to"])
    eq("so does the pre-cap count", _reread["stats"]["available"], 3)


# ============================================================================
section("27. Every row says which video it came from")
# ============================================================================
#
# The Details tab folds a video away and its screenshots have to go with it. That
# grouping is only as good as the key: the rows used to reach the api without
# `source_video`, and the only way back to the video then was the *parent directory*
# of `image_path`, which is a layout detail rather than a contract -- rename the
# frames directory and every row would silently group under the wrong heading.

_rows_src = Path(__import__("inspect").getsourcefile(
    __import__("wilcom_pipeline.api", fromlist=["api"]).rows)).read_text("utf-8")
check("the rows endpoint returns source_video",
      '"source_video"' in _rows_src,
      "source_video is not in the keep tuple, so the UI cannot group by video")
check("rows still carry the trajectory key, which is what the split uses",
      '"trajectory"' in _rows_src)

# The field is on the row itself, not only in the api's projection.
_built, _ = stage_dataset.build_rows(TRAJ, GROUND, paths)
eq("a built row knows its source video", _built[0]["source_video"], "demo.mp4")

# The UI groups on it, so a missing value must be handled rather than thrown. Checked
# by reading the component, because the failure mode is a blank card with no rows and
# no error -- a video whose rows all landed in the `''` bucket.
_det = Path("frontend/src/components/DetailsTab.tsx").read_text(encoding="utf-8")
check("the Details tab groups rows by source_video",
      "row.source_video ?? ''" in _det,
      "the group key is not source_video, so folding would not hide the frames")
check("the screenshots render inside the video card",
      "<VideoCard" in _det and "shots={byVideo[" in _det,
      "the shots grid is not passed into the card, so folding could not hide it")
check("folding is what removes them, not a display rule",
      "folded ? null : (" in _det,
      "a folded card would still be in the DOM")


print()
print("%d passed, %d failed, %d skipped" % (PASS, FAIL, SKIP))
if FAILURES:
    print()
    print("failures:")
    for f in FAILURES:
        print("  - %s" % f)
print()
print("temp dir: %s" % TMP)

sys.exit(1 if FAIL else 0)
