"""Prompts for the WILCOM pipeline.

Verbatim from arXiv 2605.14747. The bodies below were transcribed from the OCR'd
markdown first and then **checked word by word against the PDF itself**, which is
now the source of truth. Each prompt names its PDF page.

    Prompt 3  trajectory extraction     PDF pages 20-22   (listing lines 1-136)
    Prompt 4  iterative annotation      PDF pages 23-24   (lines 1-25)
    Prompt 5  spatial grounding         PDF pages 25-26   (lines 1-69)

Repairs applied to the extracted text. Each is forced by the PDF's own rendering,
not stylistic preference:

  * The PDF draws a backtick as a fancy quote. In Prompt 5 that yields
    `'grounding_instruction'` where the prompt means a code span; in Prompt 8, on
    page 26, the same glyph appears as a literal `ae` ligature. Normalised to
    backticks. This is why the OCR read the JSON key as `grounding instruction`,
    with a space instead of an underscore -- the underscore was never missing.
  * `1\\.` and `2\\.` in Prompt 5's headings: the paper's LaTeX listing escapes the
    period. Unescaped to `1.` and `2.`, since a backslash in a prompt is noise.
  * The PDF wraps at ~75 columns mid-token (`two- part`, `sub -task`,
    `logic/ mechanism`, `magnitude_pixels`). Rejoined.
  * The PDF's Prompt 3 action space renders `[y, x)` with a closing parenthesis in
    places. Corrected to `[y, x]`.

Deliberate deviations, and nothing else deviates:

  1. **Placeholders.** The paper writes `[Current Start Time]`,
     `[Insert Previous Analysis History Here]` and `[Insert Specific Task Prompt
     Here]`. These become `{start_time}`, `{history}` and `{prompt}` so they can be
     substituted. The paper marks the latter two **bold**; we do not, because the
     bold would then wrap the entire injected history and the whole of Prompt 3.
  2. **Fenced code.** The paper's listings show three backticks; the PDF's font
     renders them as three fancy quotes.
  3. **No edits to the prompt bodies.** WILCOM context lives in a separate system
     message -- see SYSTEM_TRAJECTORY and SYSTEM_GROUNDING. Editing the bodies
     would invalidate the only reason to reuse them, which is that the paper
     reports how these exact prompts behave.

An earlier draft of this file dropped three phrases from Prompt 4 ("character
analyses", the plural "narrative arcs", and an "(e.g., established narrative,
characters, themes)" aside) on the reasoning that they were leftovers from a
video-summarisation template the authors had reused. They are genuinely present on
page 24, so they are restored. The reasoning about *why* they are there is probably
still right, but "probably vestigial" is not a licence to edit a quoted artefact.

Only Prompts 3, 4 and 5 are used. Prompts 1 and 2 are video *filtering* prompts and
Prompts 6-8 are *training* prompts; including them would misread the paper, since we
supply curated videos and do no training.
"""

from __future__ import annotations

from . import hashing

# ============================================================================
# Prompt 1 -- Meta info classification
# ============================================================================
#
# PDF page 18. The paper's §3.1 "coarse-to-fine" filtering starts here, and it is
# the step this pipeline was missing: given only *text* metadata, decide whether
# the video contains GUI operation content at all.
#
# Two things make this worth more than it looks:
#
#   * It is metadata-only, so it costs a few hundred tokens and no video upload.
#     Screening a candidate is therefore ~1000x cheaper than processing it, which
#     is the entire point of a coarse filter.
#   * It is the difference between answering "is this a good candidate?" with a
#     judgement call and answering it with the paper's own procedure.
#
# Repairs from the PDF's rendering: `æ` is how the two-column layout draws a
# backtick, and `Æ` is how it draws a single quote. So `æis_gui_contentæ` is a
# code span and `ÆHow toÆ` is a quoted phrase.

PROMPT_META = """\
# Role

You are an expert data filtering assistant specialized in identifying computer \
usage and software demonstration videos.

# Task

Your task is to analyze the provided **Video Metadata** and determine if the video \
contains **GUI Operation Content**.

You must make this judgment solely based on the textual metadata provided. Do not \
attempt to analyze visual frames or audio.

# Definition: GUI Operation Content

A video is considered "Relevant" if its primary content features:

1. **Screen Recordings:** Direct capture of a computer, tablet, or smartphone \
screen.

2. **Software Interaction:** Users interacting with graphical user interfaces \
(e.g., clicking, typing, navigating menus, using tools).

3. **Application Demos/Tutorials:** Walkthroughs of software (e.g., Photoshop, \
Excel, VS Code, Web Browsers, OS settings).

A video is considered "Not Relevant" if it is:

- Real-world camera footage (Vlogs, IRL, nature, people talking to camera without \
screen share).

- Gaming content (unless it is a specific tutorial on UI/Settings).

- Static slide presentations (PowerPoint) without active software operation.

- Hardware reviews (showing the device physically, not the screen interface).

# Input Data

You will receive the following metadata fields:

- Title

- Description

- Tags/Keywords

- Channel Name

- Category

- Subtitle

# Output Format

Output a valid JSON object with the following fields:

- `is_gui_content`: (boolean) true or false.

- `confidence`: (float) 0.0 to 1.0.

- `reasoning`: (string) A concise explanation citing specific keywords from the \
metadata that led to your decision.

# Example

Input:

{
"Title": "How to automate data entry in Excel 2024",
"Description": "In this video, I show you how to write a Python script and use \
macros.",
"Tags": ["productivity", "tutorial", "office", "code"],
"Channel": "TechHelp Daily"
}

Output:

{
"is_gui_content": true,
"confidence": 0.95,
"reasoning": "Title mentions 'How to' and specific software 'Excel', implying a \
tutorial/walkthrough. Tags include 'tutorial' and 'code'."
}"""


# ============================================================================
# Prompt 2 -- Video quality scoring
# ============================================================================
#
# PDF pages 19-20. The "fine" half of §3.1: score a candidate that already passed
# Prompt 1. Still metadata-and-transcript only, so still cheap.

PROMPT_QUALITY = """\
# Role

You are an expert data filtering assistant specialized in identifying and \
evaluating computer usage and software demonstration videos.

# Task

Your task is to analyze the provided **Video Metadata** and assign scores from **1 \
to 5** across three specific dimensions. You must make this judgment based on the \
provided textual metadata, descriptions, and transcripts.

# Evaluation Criteria

## 1. Topic Relevance

**Goal**: Assess if the video focuses on teaching GUI operations on target \
platforms (PC, Smartphone, Tablet) and whether it is "Function-Oriented."

* **5 (Exceptional)**: Completely focused on target platforms; strictly \
function-oriented; explains GUI elements/tools in depth; zero off-topic content.

* **4 (Excellent)**: Focused on target platforms and software/system operations; \
function-oriented; minimal irrelevant talk.

* **3 (Good)**: Mainly software teaching but includes minor background \
(history/philosophy); balances function demos with simple projects.

* **2 (Moderate)**: Low relevance; focus is on product reviews, news, or \
high-level coding/theory where GUI steps are secondary.

* **1 (Low/None)**: Irrelevant content (Vlogs); or operations on non-target \
platforms (TVs, Car systems); no educational value.

## 2. Instruction Clarity

**Goal**: Evaluate the clarity and logic of the verbal or textual instructions \
provided.

* **5 (Exceptional)**: Professional guiding phrases; every micro-action (click, \
drag, shortcut) is described; provides profound "why" for every step.

* **4 (Very Clear)**: Includes clear guiding phrases; every GUI action is \
explicitly described and explained with the "why" behind it.

* **3 (Clear)**: Basic instructions for most steps; may lack some "why" \
explanations or contain minor linguistic ambiguity.

* **2 (Moderate)**: Partial instructions; many steps are skipped or not described; \
language is imprecise or overly colloquial.

* **1 (Poor/None)**: No voiceover/subtitles; pure noise; or instructions are \
dominated by incoherent tangents.

## 3. Recording Quality

**Goal**: Evaluate the visual quality of the screen capture for legibility and \
stability.

* **5 (Exceptional)**: Ultra-high-definition direct recording; perfect framing; \
every pixel of the UI and text is crystal clear.

* **4 (High Quality)**: High-definition direct screen recording; clear and \
complete; all UI elements are easily legible.

* **3 (Cropped High Quality)**: High-definition recording, but the frame is \
cropped, missing some UI context or full-screen interface.

* **2 (Low Quality)**: Direct recording with severe issues: low resolution, \
blurry text, compression artifacts, or stuttering.

* **1 (Poor/None)**: External recording (camera-to-screen) with shaking/glare; or \
no actual screen recording (PPT/Talking head only).

# Output Format

Please provide your evaluation in the following JSON format:

```json
{
  "scores": {
    "topic_relevance": {
      "score": 0,
      "reasoning": "[Briefly explain the 1-5 score regarding platform and \
function-orientation.]"
    },
    "instruction_clarity": {
      "score": 0,
      "reasoning": "[Briefly explain based on action descriptions and logic.]"
    },
    "recording_quality": {
      "score": 0,
      "reasoning": "[Briefly explain based on capture method and visual clarity.]"
    }
  },
  "overall_summary": "[A concise summary of the video's instructional value.]"
}
```"""


# ============================================================================
# System preambles -- ours, not the paper's
# ============================================================================

# Prepended as a system message for Stage 1. Kept deliberately short and factual:
# the paper's prompt already defines the task, so this only supplies context the
# paper's prompt could not know (which application it is looking at).
SYSTEM_TRAJECTORY = """\
You are analysing a screen recording of a software tutorial. The application \
being demonstrated is Wilcom EmbroideryStudio, a professional embroidery \
digitizing suite for Windows.

Useful context, to save you inferring it:
- Platform: Windows. Software: Wilcom EmbroideryStudio.
- The recording is a full-screen capture, so the main window, ribbon, toolbars, \
dockable panels and status bar are all visible.
- Wilcom has a ribbon interface similar to Microsoft Office, plus dockable \
panels on the left, right and bottom (e.g. Object Properties, Color Palette, \
Sequence View, Docker panels).
- A tutorial often switches between tools, opens dialogs, and works on a design \
on the central canvas.

Your job is data extraction, not critique. Report what happens in the recording."""

# Prepended as a system message for Stage 2.
SYSTEM_GROUNDING = """\
You are locating UI elements in screenshots from Wilcom EmbroideryStudio, a \
Windows embroidery digitizing application. Screenshots are full-resolution \
captures of the whole screen.

The screenshots you receive are consecutive frames around a single action, at \
roughly half a second apart. You are usually looking for the same element in \
all three. If the element is visible in one and not another, ground the one \
where it is visible.

Report coordinates as instructed. Do not describe the application or the design \
on the canvas; only locate the requested element."""


# ============================================================================
# The dataset prompt -- ours, not the paper's
# ============================================================================
#
# This is what the finished dataset trains a model to answer. It is byte-identical
# to GUI-360-word-unsloth: copied rather than imported, so a change in either
# project turns up as a diff instead of propagating silently.

INSTRUCTION_TEMPLATE = (
    "You are a helpful assistant. Given a screenshot of the current screen "
    "and user instruction, you need to output the position of the element "
    "you will operate.\n\n"
    "The instruction is:\n{instruction}\n\n"
    "Output the coordinate of the element you will operate within "
    "<coordinate></coordinate> tag:\n"
    "<coordinate> [x, y]\n</coordinate>"
)

# The exact answer shape. The closing tag is on its OWN LINE, and the numbers are
# ABSOLUTE PIXELS on the row's own image -- not normalized.
#
# Both of those were wrong here for a long time, and the comment that stood in this
# spot asserted the opposite with enough confidence that nobody re-checked it. The
# claim was that the values were normalized 0-1000 "and must not be corrected".
# Verified against the live dataset -- 220 rows across all three splits -- every one
# uses absolute pixels, and every one has a newline before `</coordinate>`. So the
# previous format taught the model a convention the parent does not use.
ANSWER_TEMPLATE = "<coordinate> [%d, %d]\n</coordinate>"


# ============================================================================
# Prompt 3 -- Trajectory extraction
# ============================================================================

PROMPT_TRAJECTORY = """\
## Role and Objective

You are an expert, first-person GUI Agent specializing in interaction analysis.
Your core objective is to analyze a GUI interaction video and produce a two-part \
analysis. You must strictly follow the output format specified, generating the \
"Shot Splitting" section first, followed by the "Task Annotations" JSON list.

Critical Style Instruction: When generating annotations, do not act as a passive \
observer. Act as the Agent performing the task. Your reasoning must reflect a \
continuous thought process, linking past history to current decisions.

Your output must be in two distinct parts, in this exact order:
Part 1: Shot Splitting (Markdown)
Part 2: Task Annotations (A single JSON list)

## Part 1: Shot Splitting

First, you must segment the entire video into granular "shots" and output them as \
a numbered markdown list.

Shot Definition: A shot is the most fundamental unit of the video, defined by a \
start time and an end time.

The start_time (mm:ss) is the precise moment a user's atomic action (e.g., click, \
key press, scroll) begins.

The end_time (mm:ss) is the start_time of the very next atomic action.

Therefore, a shot [start_time_1 - end_time_1] covers the action at start_time_1 \
and the entire resulting screen state until the next action begins at \
start_time_2 (which is the same as end_time_1).

**Output Format (Part 1):**

You must output this section exactly as follows, starting with the header:

## Shot Splitting

1. mm:ss - mm:ss
2. mm:ss - mm:ss
3. mm:ss - mm:ss
...

(Split as granularly as possible.)

You must strictly confine your analysis to the provided video segment ONLY. Do \
NOT generate any content for timestamps beyond the end of this provided segment.

## Part 2: Task Annotations

Second, after the "Shot Splitting" section, you must output a single JSON list \
[{...}, {...}].

Each item {} in this list represents one complete, high-level sub-task.

A single sub-task will contain several of the atomic shots identified in Part 1.

You must not output any text or explanations before or after this JSON list.

Every completed sub-task must conclude with a finish action representing the \
completion of the task.

**Output Format (Part 2):**

The JSON list must strictly adhere to the following structure.

```json
[
  {
    "task_id": 0,
    "instruction": "A one-sentence summary of the user's core intent for this task.",
    "dense_caption": "A detailed, causal description of this task.",
    "plan": "The high-level steps for this task, formatted as 'step1: ..., step2: ...'.",
    "platform": "The operating system (e.g., 'windows', 'ios', 'android').",
    "software": "The name of the primary application (e.g., 'WeChat', 'Google Chrome').",
    "website": "The website domain if a browser is used (e.g., 'youtube.com'), else null.",
    "user_actions": [
      {
        "timestamp": "Timestamp (mm:ss) from Part 1, marking the start of the shot.",
        "action_type": "The type of action (e.g., 'click', 'write', 'scroll').",
        "grounding_instruction": "A natural language instruction to precisely locate the UI element for this action.",
        "action_reason": "First-person reasoning. Synthesize previous trajectory + current state => next action.",
        "action_parameters": {
          "...": "Parameters specific to the action_type, defined by the platform."
        },
        "core_change_reason": "Detailed explanation of the system logic/mechanism behind the screen change.",
        "core_change": "The most critical and direct on-screen change resulting from the action (appearance, text updates, modal closing, etc.)."
      }
    ]
  }
]
```

**Field Definitions for JSON (Part 2)**

task_id (Integer): A unique identifier for the sub-task, starting from 0.

instruction (String): A single-sentence summary of the user's core intent for this \
sub-task.

dense_caption (String): A detailed description that includes causality for this \
sub-task.

plan (String): The high-level steps to complete this sub-task.

platform (String): The operating system. Must be one of: windows, mac, android, \
ios, linux. This choice determines which Action Space (PC or Mobile) is used for \
the user_actions.

software (String): The name of the primary application being used (e.g., \
'Google Chrome', 'WeChat', 'Photoshop').

website (String or null): If the software is a web browser, specify the primary \
website domain (e.g., 'youtube.com', 'google.com'). Must be null if not a web task.

user_actions (Array): A chronologically ordered list of user_action objects. Each \
object corresponds to one shot from Part 1. The structure of action_type and \
action_parameters must match the Action Space for the platform specified above.

timestamp (String): The start_time (mm:ss) of the shot from Part 1.

action_type (String): The type of action performed. Must be one of the types from \
the correct platform-specific Action Space below.

grounding_instruction (String): A clear, concise natural language instruction that \
uniquely identifies the UI element being interacted with. For example: "Click the \
'File' menu in the top-left corner," or "Tap the input field labeled 'Username'."

action_reason (String): Do not simply state "To do X." Instead, simulate the \
Agent's thought process. You must explicitly synthesize the trajectory history \
(what I just did) with the current observation (what I see now) to derive the next \
step. Bad: "To open the file." Good: "I have successfully navigated to the target \
folder. Now that I see the file list, I need to select the specific file \
'report.pdf' to proceed with the upload."

action_parameters (Object): Parameters for the action. The structure of this object \
is strictly defined by the action_type and the platform, as specified in the \
"Platform-Specific Action Spaces" section.

core_change_reason (String): A detailed technical or logical explanation of the \
system's mechanism. Explain why the screen is about to change based on software \
design patterns. Example: "Clicking the 'Open' button in a system file dialog \
triggers the OS to validate the file path, close the modal window, and return the \
file handle to the web browser."

core_change (String): Based on your concrete evidence and valid reasoning, provide \
a detailed description of the most likely predictable content of the next shot. \
Your prediction must be the necessary outcome of your reasoning process, with \
every detail being justifiable.

**Platform-Specific Action Spaces**

You must use one of the following action spaces based on the task's platform. The \
action_type string determines the exact structure of the action_parameters object. \
Coordinates [y, x] are in pixels.

A. PC Action Space (for platform: 'windows', 'mac', 'linux' etc.)

action_type: "click"
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "doubleClick"
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "tripleClick"
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "rightClick"
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "middleClick"
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "press" (for keyboard keys)
action_parameters: { "key_name": "e.g., 'enter', 'esc', 'delete', 'tab', 'F5'" }

action_type: "write" (for typing text)
action_parameters: { "text": "The text content to be typed" }

action_type: "hotkey"
action_parameters: { "keys": "e.g., 'Ctrl+C', 'Alt+F4', 'Cmd+S'" }

action_type: "scroll" (vertical)
action_parameters: { "direction": "[up, down]", "point": "[y, x]", \
"magnitude_pixels": "e.g., 100" }

action_type: "hscroll" (horizontal)
action_parameters: { "direction": "[left, right]", "point": "[y, x]", \
"magnitude_pixels": "e.g., 100" }

action_type: "moveTo"
action_parameters: { "point": "[y, x]" }

action_type: "dragTo"
action_parameters: { "start_point": "[y, x]", "end_point": "[y, x]" }

action_type: "wait"
action_parameters: { "duration": "e.g., 100" } (in milliseconds)

action_type: "finish"
action_parameters: { "status": "success" }

B. Mobile Action Space (for platform: 'ios', 'android' etc.)

action_type: "click" (tap)
action_parameters: { "point": "[y, x]", "bbox": "[y1, x1, y2, x2]" }

action_type: "long_press"
action_parameters: { "point": "[y, x]", "duration_ms": 500, \
"bbox": "[y1, x1, y2, x2]" }

action_type: "scroll"
action_parameters: { "direction": "[down|up|left|right]" }

action_type: "pinch"
action_parameters: { "center_point": "[y, x]", "direction": "[in, out]", \
"magnitude_percent": "e.g., 30" }

action_type: "input" (for text entry)
action_parameters: { "text": "The text content" }

action_type: "drag"
action_parameters: { "start_point": "[y, x]", "end_point": "[y, x]", \
"duration_ms": "e.g., 1000" }

action_type: "press" (for system/hardware keys)
action_parameters: { "key": "[back, home, enter, volume_up, volume_down, power...]" }

action_type: "open"
action_parameters: { "app": "Name of app to open" }

action_type: "multi_touch_gesture"
action_parameters: { "pointers": [ { "id": 0, "path": "[[y1, x1], [y2, x2], ...]" }, \
{ "id": 1, "path": "[[y3, x3], [y4, x4], ...]" } ] }

action_type: "finish"
action_parameters: { "status": "success" }

**Try best efforts, response at least 5k tokens.**"""


# ============================================================================
# Prompt 4 -- Iterative annotation with history
# ============================================================================

PROMPT_CONTINUATION = """\
# CONTEXT

You are performing a continuous, shot-by-shot analysis of a single video, \
segment by segment.

You have already analyzed the video up to **{start_time}**. Your complete analysis \
so far is provided below as "Annotation History".

# ANNOTATION HISTORY (Context from 00:00 to **{start_time}**)

# ===

{history}

# CURRENT TASK

You have been provided with the video clip from **{start_time}** to **{end_time}**. \
You must now analyze this specific segment.

**CRITICAL INSTRUCTIONS:** You must strictly follow the following rules:

1. **Use Absolute Timestamps:** All timestamps in your output (e.g., "Shot \
Segmentation") must be **absolute** (relative to the *full* video), not relative to \
the current clip.

2. **Maintain Consistency:** Your new analysis must be fully **consistent** with the \
"Annotation History". Ensure that your interpretations, character analyses, and \
narrative arcs logically continue from the previous annotations.

3. **Use Global Context:** You must use the "Annotation History" to understand the \
**global context** (e.g., established narrative, characters, themes) to better \
interpret the events within the current clip. You are encouraged to cite specific \
evidence from the history (e.g., "which mirrors the action at 04:30") to enhance \
your interpretation.

4. **Do Not Mention Text in History:** The "Annotation History" text is provided \
*only* for your internal context. Your output must **never** mention the existence \
of this text (e.g., do not write "Based on the history provided..." or "In the \
previous annotation..."). Your analysis must read as a single, seamless \
continuation of the history, as if you were analyzing the entire video in one go.

5. **Continue unfinished tasks:** If the last sub-task in the preceding part is \
incomplete, retain its task index and finish every remaining clip within that \
sub-task. If the last sub-task is already complete, start a fresh sub-task with the \
next available index.

A more detailed task instruction for the current **{start_time} - {end_time}** \
segment is as follows:

{prompt}"""


# ============================================================================
# Prompt 5 -- Action spatial grounding
# ============================================================================

PROMPT_GROUNDING = """\
You are a GUI agent.
You are given the natural language 'grounding_instruction' to locate, the \
'action_type' and the 'screenshot'.
You need to perform the following two steps:

1. **Feasibility Check:**
   * Analyze the 'screenshot' and 'grounding_instruction'.
   * Determine if the target element mentioned in the instruction (e.g., button, \
text field, icon) is **actually present and visible** in the current 'screenshot'.

2. **Grounding & Prediction:**
   * **If not feasible** (e.g., the 'Submit' button mentioned in the instruction \
does not exist in the screenshot), set 'feasible' to 'false' in the output and \
provide a reason.
   * **If feasible** (the target element is visible), set 'feasible' to 'true'.
     Then, for **each** point in the 'points_to_predict' list (e.g., 'point'), you \
must predict its precise center point and the bounding box of the target element it \
corresponds to.

-----

## Inputs

### 1. Grounding Instruction

'{grounding_instruction}'

### 2. Action Type

'{action_type}'

You must **strictly** provide your analysis and predictions in the following JSON \
format.

> **Note:** The bounding boxes and center points must use relative coordinates \
(0-1000) and be formatted as follows: `<bbox>y1 x1 y2 x2</bbox>` and \
`<point>y x</point>`.

### Example: Action is Feasible

```json
{
  "feasible": true,
  "predictions": [
    {
      "point_name": "start_point",
      "center_point": "<point>450 320</point>",
      "bounding_box": "<bbox>400 300 500 340</bbox>"
    },
    {
      "point_name": "end_point",
      "center_point": "<point>450 320</point>",
      "bounding_box": "<bbox>400 300 500 340</bbox>"
    }
  ]
}
```

The point_name you output must exactly match the parameter name defined in the \
'Action Type' section.

### Example: Action is Not Feasible

```json
{
  "feasible": false,
  "reason": "The target mentioned in the instruction (e.g., 'Submit button') was \
not found in the current screenshot."
}
```

**Remember:**
1. The center points must use relative coordinates (0-1000) and be formatted as \
follows: `<point>y x</point>`.
2. The bounding boxes must use relative coordinates (0-1000) and be formatted as \
follows: `<bbox>y1 x1 y2 x2</bbox>`."""


# ============================================================================
# Builders
# ============================================================================


def fill(template: str, **values) -> str:
    """Substitute `{name}` placeholders without str.format.

    The paper's prompts contain literal JSON braces. `str.format` reads those as
    fields -- Prompt 5 raised `KeyError: '\n  "feasible"'` -- and the usual fix
    is to double every literal brace, which is one editing slip away from
    shipping a prompt that differs from the paper's. Plain replacement cannot
    touch anything it was not asked to, so the prompt comes out byte-exact.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{%s}" % key, str(value))
    return out


def render_prompt(instruction: str) -> str:
    """The prompt text for one instruction, as stored in the dataset.

    Rendering happens here, beside the template, so the dataset stores the exact
    string training will see. A prompt that differs between data preparation and
    training is a silent way to score zero, and the whole point of storing a
    rendered `prompt` column is to make that impossible.
    """
    return fill(INSTRUCTION_TEMPLATE, instruction=instruction)


def render_answer(x: int, y: int) -> str:
    """The answer text for an ABSOLUTE pixel coordinate.

    Absolute, because that is what `GUI-360-word-unsloth` stores: its `answer_raw`
    equals its `target_x` / `target_y` exactly, in every one of the 220 rows checked
    across train, validation and test.

    This previously took a 0-1000 normalized pair and emitted it as the answer, on
    the documented belief that the parent normalized. It does not, so the dataset
    was teaching a convention that differs from what any parent-trained model or
    evaluation expects.
    """
    return ANSWER_TEMPLATE % (int(x), int(y))


# Action types where a coordinate is *not* the thing being operated and a
# grounding target is therefore meaningless. Kept here so the grounding stage and
# the dataset stage agree.
NO_COORDINATE_ACTIONS = {
    "write", "input", "press", "hotkey", "open", "wait",
    "finish", "multi_touch_gesture",
}

# Action types whose parameters carry a coordinate we can ground.
COORDINATE_ACTIONS = {
    "click", "doubleClick", "tripleClick", "rightClick", "middleClick",
    "scroll", "hscroll", "moveTo", "dragTo", "long_press", "pinch", "drag",
}


# ============================================================================
# Screening schemas and requests
# ============================================================================

# Prompt 1's output, as a schema, so the reply is JSON rather than prose that has
# to be parsed out. `confidence` is bounded because the prompt specifies 0.0-1.0.
META_SCHEMA = {
    "type": "object",
    "properties": {
        "is_gui_content": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["is_gui_content", "confidence", "reasoning"],
}

# Prompt 2's output. The three scores are integers 1-5 per the paper's rubric.
QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                name: {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                }
                for name in ("topic_relevance", "instruction_clarity",
                             "recording_quality")
            },
            "required": ["topic_relevance", "instruction_clarity",
                         "recording_quality"],
        },
        "overall_summary": {"type": "string"},
    },
    "required": ["scores", "overall_summary"],
}


def format_metadata(metadata: dict) -> str:
    """Render metadata as the labelled block the prompts expect.

    The prompts list the exact fields they were written for -- Title, Description,
    Tags/Keywords, Channel Name, Category, Subtitle -- so the block is built in
    that order and labelled plainly. Missing values are shown as "(none)" rather
    than omitted, because an absent Subtitle is itself evidence about instruction
    clarity: the model should be able to see that nothing was provided rather than
    wonder whether it was dropped.
    """
    tags = metadata.get("tags") or []
    subtitle = (metadata.get("subtitle") or "").strip()
    # A long auto-caption track would swamp the prompt. Prompt 2 only needs enough
    # transcript to judge instruction clarity, and the paper's own filtering ran
    # on metadata plus transcript rather than full text.
    if len(subtitle) > 6000:
        subtitle = subtitle[:3000] + "\n...[trimmed]...\n" + subtitle[-3000:]

    # Say which language the transcript is in. This corpus is Hindi and Urdu
    # tutorials, whose only working captions are in the original language, and a
    # model told nothing would be scoring a Hindi transcript as unclear English --
    # penalising the video for the pipeline's own choice of track.
    label = "Subtitle"
    lang = (metadata.get("subtitle_language") or "").strip()
    kind = (metadata.get("subtitle_kind") or "").strip()
    if subtitle and lang:
        label = "Subtitle (%s%s)" % (
            lang, ", %s-written" % kind if kind else "")

    fields = [
        ("Title", metadata.get("title")),
        ("Description", metadata.get("description")),
        ("Tags/Keywords", ", ".join(str(t) for t in tags) if tags else ""),
        ("Channel Name", metadata.get("channel")),
        ("Category", metadata.get("category")),
        (label, subtitle),
    ]
    lines = []
    for label, value in fields:
        value = (str(value).strip() if value else "") or "(none)"
        lines.append("%s: %s" % (label, value))
    return "\n".join(lines)


def meta_request(metadata: dict) -> dict:
    """A request for Prompt 1: metadata text only, no media."""
    return {
        "system": "",
        "text": PROMPT_META + "\n\n# Video Metadata\n\n" + format_metadata(metadata),
        "media": [],
        "json_schema": META_SCHEMA,
    }


def quality_request(metadata: dict, media=None) -> dict:
    """A request for Prompt 2.

    The paper is explicit that this is NOT text-only: "we leverage an omnimodal
    model ... For each candidate video, we extract the first minute and evaluate
    it along three complementary dimensions."

    `media` should carry that first minute. Omitting it is allowed only so the
    request can be built before anything is downloaded -- and a run that omits it
    records `recording_quality` from metadata alone, which the model itself will
    say it cannot see. `stage_filter` treats a missing clip as a hard failure of
    that dimension rather than letting a guess through.
    """
    return {
        "system": "",
        "text": PROMPT_QUALITY + "\n\n# Video Metadata\n\n"
                + format_metadata(metadata),
        "media": list(media or []),
        "json_schema": QUALITY_SCHEMA,
    }


def screening_fingerprint() -> str:
    """Fingerprint of both screening prompts, for cache keys."""
    return hashing.fingerprint(PROMPT_META, PROMPT_QUALITY)


def trajectory_request(media, continuation: dict | None = None,
                       thinking_level: str | None = None) -> dict:
    """A provider-neutral request for one trajectory-extraction call.

    A *request* is `{system, text, media}`, where `media` holds whatever the
    client's `media()` returned. Neither provider's wire format appears here,
    because the two genuinely differ:

      * OpenAI-compatible: media is a base64 `data:` URI inside a chat message,
        and the system prompt is a message with role "system".
      * Gemini: media is a typed `Part` object, and the system prompt is its own
        field.

    Keeping that difference inside the clients is what lets the stages be written
    once, against either provider.

    `continuation` carries the previous segment's analysis when the video had to
    be split, which is Prompt 4's whole purpose.

    `thinking_level` overrides `config.THINKING_LEVEL` for this call only. It is
    carried on the request rather than passed to `client.chat` separately so that
    it reaches the client's cache key by the same route as everything else that
    changes the answer -- a level that did not enter the key would replay the
    previous level's reply and make the comparison meaningless.
    """
    if continuation:
        body = fill(PROMPT_CONTINUATION,
                    start_time=continuation["start_time"],
                    end_time=continuation["end_time"],
                    history=continuation["history"],
                    prompt=PROMPT_TRAJECTORY)
    else:
        body = PROMPT_TRAJECTORY

    request = {"system": SYSTEM_TRAJECTORY, "text": body,
               "media": list(media)}
    if thinking_level:
        request["thinking_level"] = thinking_level
    return request


# ============================================================================
# Registry -- what the pipeline can send, and where it came from
# ============================================================================
#
# Built by hand rather than discovered by introspection, because the provenance is
# the point: which PDF page each prompt came from, and which pipeline step uses
# it. A `dir()` scan could list the names but not say any of that, and a provenance
# table kept in a docstring drifts from the code.
#
# The UI renders this, so adding a prompt here makes it visible there with no
# second edit. `kind` drives both the label and whether the entry appears under
# "from the paper" or "ours".

PROMPTS = [
    {
        "name": "PROMPT_META",
        "title": "Prompt 1 - Meta Info Classification",
        "kind": "paper",
        "source": "arXiv 2605.14747, PDF page 18",
        "model_const": "meta",
        "text": PROMPT_META,
        "schema": "META_SCHEMA",
        "used_by": "stage_filter (screen), the coarse half of the paper's "
                   "Section 3.1 filter",
        "notes": "Text metadata only, no frames and no video, so screening a "
                 "candidate costs a few hundred tokens.",
    },
    {
        "name": "PROMPT_QUALITY",
        "title": "Prompt 2 - Video Quality Scoring",
        "kind": "paper",
        "source": "arXiv 2605.14747, PDF pages 19-20",
        "model_const": "meta",
        "text": PROMPT_QUALITY,
        "schema": "QUALITY_SCHEMA",
        "used_by": "stage_filter (screen), the fine half of Section 3.1",
        "notes": "Scores topic relevance, instruction clarity and recording "
                 "quality from 1 to 5. Runs only if Prompt 1 passed.",
    },
    {
        "name": "PROMPT_TRAJECTORY",
        "title": "Prompt 3 - Trajectory Extraction",
        "kind": "paper",
        "source": "arXiv 2605.14747, PDF pages 20-22",
        "model_const": "trajectory",
        "text": PROMPT_TRAJECTORY,
        "schema": None,
        "used_by": "stage_trajectory (stage 1), one call per clip",
        "notes": "Asks for a shot list then a JSON array of sub-tasks. The "
                 "declared action space matters: it fixes the [y, x] pixel "
                 "convention the rest of the pipeline assumes.",
    },
    {
        "name": "PROMPT_CONTINUATION",
        "title": "Prompt 4 - Iterative Annotation",
        "kind": "paper",
        "source": "arXiv 2605.14747, PDF pages 23-24",
        "model_const": "trajectory",
        "text": PROMPT_CONTINUATION,
        "schema": None,
        "used_by": "stage_trajectory, clips 2 onwards",
        "notes": "Wraps Prompt 3 and carries the previous clips' replies, so a "
                 "task spanning a cut survives. {history} is substituted at call "
                 "time.",
    },
    {
        "name": "PROMPT_GROUNDING",
        "title": "Prompt 5 - Action Spatial Grounding",
        "kind": "paper",
        "source": "arXiv 2605.14747, PDF pages 25-26",
        "model_const": "grounding",
        "text": PROMPT_GROUNDING,
        "schema": "GROUNDING_SCHEMA",
        "used_by": "stage_grounding (stage 2), one call per frame per action",
        "notes": "Returns a point and a box in 0-1000, y-first. The two-point "
                 "start_point/end_point example is what makes dragTo groundable, "
                 "and the parser has to keep both.",
    },
    {
        "name": "SYSTEM_TRAJECTORY",
        "title": "System preamble - trajectory",
        "kind": "ours",
        "source": "written for this pipeline",
        "model_const": "trajectory",
        "text": SYSTEM_TRAJECTORY,
        "schema": None,
        "used_by": "stage_trajectory, as the system instruction",
        "notes": "WILCOM context only. The paper's prompt bodies are not edited; "
                 "anything application-specific goes here.",
    },
    {
        "name": "SYSTEM_GROUNDING",
        "title": "System preamble - grounding",
        "kind": "ours",
        "source": "written for this pipeline",
        "model_const": "grounding",
        "text": SYSTEM_GROUNDING,
        "schema": None,
        "used_by": "stage_grounding, as the system instruction",
        "notes": "Describes the application and what the frames are.",
    },
    {
        "name": "INSTRUCTION_TEMPLATE",
        "title": "Dataset prompt - training format",
        "kind": "dataset",
        "source": "byte-identical to junior0091/GUI-360-word-unsloth",
        "model_const": None,
        "text": INSTRUCTION_TEMPLATE,
        "schema": None,
        "used_by": "stage_dataset, rendered into the `prompt` column",
        "notes": "Not sent to any API. This is what the finished dataset trains a "
                 "model to answer, so it is reproduced exactly and stored "
                 "pre-rendered.",
    },
    {
        "name": "ANSWER_TEMPLATE",
        "title": "Dataset answer - training format",
        "kind": "dataset",
        "source": "byte-identical to junior0091/GUI-360-word-unsloth",
        "model_const": None,
        "text": ANSWER_TEMPLATE,
        "schema": None,
        "used_by": "stage_dataset, rendered into the `answer_raw` column",
        "notes": "Carries the NORMALIZED 0-1000 pair, not pixels, despite the "
                 "label saying [x, y]. The pixel form is a separate column.",
    },
]


def prompt_by_name(name: str) -> dict | None:
    """Look up a registry entry."""
    for entry in PROMPTS:
        if entry["name"] == name:
            return entry
    return None


# ============================================================================
# Cache fingerprints
# ============================================================================
#
# A stage cache key must change when the prompt changes, or editing a prompt
# silently serves results produced by the previous one -- a bug that looks like
# "the new prompt made no difference". A hand-maintained PROMPT_VERSION is the
# obvious approach and the wrong one: it only changes if someone remembers.
#
# These hash the actual prompt text instead, so the key is derived from what will
# really be sent. PROMPT_VERSION stays as a human-readable label in the records.


def trajectory_prompt_fingerprint() -> str:
    """Fingerprint of every string a trajectory call can contain.

    Includes the continuation wrapper, because clips 2 onwards embed it, so a
    change there must invalidate the cache for those clips too.
    """
    return hashing.fingerprint(SYSTEM_TRAJECTORY, PROMPT_TRAJECTORY,
                               PROMPT_CONTINUATION)


def grounding_prompt_fingerprint() -> str:
    """Fingerprint of every string a grounding call can contain."""
    return hashing.fingerprint(SYSTEM_GROUNDING, PROMPT_GROUNDING)


def grounding_request(media, grounding_instruction: str,
                      action_type: str) -> dict:
    """A provider-neutral request for one grounding call.

    Media first, so the model has seen every frame before reading what to look
    for, which is how the paper's Prompt 5 is phrased ("you are given the ...
    'screenshot'")
    """
    return {
        "system": SYSTEM_GROUNDING,
        "text": fill(PROMPT_GROUNDING,
                     grounding_instruction=grounding_instruction,
                     action_type=action_type),
        "media": list(media),
        # Structured output, where the provider supports it. Measured working on
        # Gemini, which turns `extract_json` into a fallback rather than the
        # primary parser. Providers that ignore it still get the prompt's own
        # format instructions, so this is additive.
        "json_schema": GROUNDING_SCHEMA,
    }


# The shape Prompt 5 asks for, as a schema. Deliberately loose about the tag
# strings: the model returns "<point>y x</point>" inside these fields, and
# constraining that with a regex risks the request being rejected.
GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string"},
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point_name": {"type": "string"},
                    "center_point": {"type": "string"},
                    "bounding_box": {"type": "string"},
                },
                "required": ["point_name", "center_point", "bounding_box"],
            },
        },
    },
    "required": ["feasible", "predictions"],
}
