"""Configuration for the WILCOM Video2GUI pipeline.

Everything tunable lives here so the stages stay readable. The coordinate
assertion at the bottom is not decoration -- see `verify_coordinate_convention`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Models
# ============================================================================

# The paper uses "Gemini-3-Pro". On the Gemini Developer API the Pro tier that
# accepts video is this one. Verified against the live model list for this account:
# present, 1M input / 64k output, and callable.
MODEL_PRO = "gemini-3.1-pro-preview"

# Cheaper alternatives, also verified present and video-capable.
MODEL_FLASH = "gemini-3.8-flash"
MODEL_FLASH_MID = "gemini-3.5-flash"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite"

# Which model does which job.
#
# These follow the paper exactly. Section 3 says, in its own words:
#
#   Prompt 1, meta filtering    "we leverage DeepSeek-V3 to annotate relevance
#                               labels ... and subsequently fine-tune a
#                               lightweight Qwen2.5-7B with a classification
#                               head to enable scalable inference"
#   Prompt 2, quality scoring   "we curate approximately 200 hours of videos ...
#                               and employ Gemini 3 Pro to annotate each video
#                               with dimension-wise scores"
#   Prompt 3/4, trajectory      "we employ Gemini-3-Pro directly as the
#                               annotation model"
#   Prompt 5, grounding         "For each frame, we employ Gemini-3-Pro"
#
# So Gemini-3-Pro is the annotator for everything except Prompt 1. The small
# models in the paper (Qwen2.5-7B, Qwen2.5-Omni) are the distilled *students* they
# train afterwards for cheap inference at 20-million-video scale; they are not what
# the prompts run on. Since this pipeline does the annotation step, Pro is the
# faithful choice at every one of them.
#
# The one substitution is Prompt 1: DeepSeek-V3 is not on the Gemini Developer API,
# so Pro stands in. `deepseek_note` records that rather than hiding it.
#
# COST CONSEQUENCE, measured rather than guessed. Grounding at Pro is ~8x the
# Flash-Lite price per call, and it is the high-volume stage: ~45 actions per
# 4-minute clip. A full clip moves from roughly $0.12 to roughly $0.37. That is the
# price of fidelity, and ECONOMY_MODELS below is the way to trade it back.
ROLE_MODELS = {
    "trajectory": MODEL_PRO,
    "grounding": MODEL_PRO,
    "meta": MODEL_PRO,       # paper used DeepSeek-V3 here; see note below
    "quality": MODEL_PRO,
}

# The paper's annotator for Prompt 1. Not reachable through this API, so
# ROLE_MODELS uses Pro instead and this records what was actually used.
DEEPSEEK_NOTE = ("the paper annotated Prompt 1 with DeepSeek-V3; that model is "
                 "not on the Gemini Developer API, so Pro substitutes")

# A deliberately cheaper set, for iterating on prompts where fidelity does not
# matter. Grounding only has to locate one element in a still frame, and measured
# it did that to within 1.0 px on a synthetic test at this size -- but it is NOT
# what the paper used, so it is opt-in and never the default.
ECONOMY_MODELS = {
    "trajectory": MODEL_PRO,        # cutting this one changes the data most
    "grounding": MODEL_FLASH_LITE,
    "meta": MODEL_FLASH_LITE,
    "quality": MODEL_FLASH_LITE,
}


def model_for(role: str) -> str:
    """The model for a role: 'trajectory', 'grounding', 'meta' or 'quality'.

    `WILCOM_MODEL_TIER=economy` switches to the cheaper set. Read from the
    environment so a run's tier is visible in its shell history rather than buried
    in a file someone edited.
    """
    import os as _os
    table = (ECONOMY_MODELS if _os.environ.get("WILCOM_MODEL_TIER", "").lower()
             == "economy" else ROLE_MODELS)
    try:
        return table[role]
    except KeyError:
        raise SystemExit("unknown model role %r (expected one of: %s)"
                         % (role, ", ".join(sorted(table))))


def model_tier() -> str:
    """'paper' or 'economy', for logging and for the manifest."""
    import os as _os
    return ("economy" if _os.environ.get("WILCOM_MODEL_TIER", "").lower()
            == "economy" else "paper")

# Grounding never needs temporal context -- a few still frames and a short
# instruction -- and it runs once per *action* rather than once per clip, so it is
# the high-volume call. The `grounding` role in ROLE_MODELS reflects that.

TEMPERATURE = 0.0          # deterministic: this is data extraction, not writing
MAX_OUTPUT_TOKENS = 16384  # Prompt 3 asks for "at least 5k tokens" of output

# Thinking budget. The paper reports 1,452 thinking tokens per sample. Charging
# thought to output tokens is the largest single line item, so it is configurable.
#
# MEASURED on the first real clip: Gemini 3.1 Pro spent 11,681 thinking tokens to
# produce 4,101 tokens of answer. That is 74% of the output cost and the dominant
# share of the wall-clock time -- roughly 2-4 minutes per clip of pure generation.
# The paper's figure is far lower, so the default here is deliberately restrained.
#
# Gemini 3 cannot disable thinking entirely; "low" is the floor for these models.
# Levels: "low" (about 1,024), "medium" (about 8,192), "high" (about 24,576).
THINKING_LEVEL = "medium"

# The levels a caller may choose, cheapest first. Ordered because that is how the
# trade-off should be read, and because the UI lists them in this order.
THINKING_LEVELS = ("low", "medium", "high")

# Measured on one real 207s clip, so the UI can state the trade-off rather than
# just offering a word. Thinking is billed as output tokens and dominates both
# cost and wall-clock time.
THINKING_NOTES = {
    "low": "cheapest and fastest. Around 1,000 thinking tokens.",
    "medium": "a middle setting. Around 8,000 thinking tokens.",
    "high": "most thorough, several times slower. Around 24,000 tokens. "
            "Measured at 11,681 thinking tokens against 4,101 of answer on one "
            "clip -- roughly 2-4 minutes of generation per clip.",
}

# Older models take a token budget instead of a level. Used only when a model
# rejects the level form.
# ============================================================================
# Candidate eligibility -- the paper's own selection rule
# ============================================================================

# "we selected only those with durations under 12 minutes based on metadata
#  information" -- Section C, Trajectory Extraction Details.
#
# Enforced at screening, from metadata, before anything is downloaded. It is a
# COST control as much as a quality one: at 240s per clip a 12-minute video is 3
# clips, and a 60-minute one is 15 -- roughly $1.20 in stage 1 against $0.08, for
# a video the paper would not have accepted at all.
#
# The paper says "under 12 minutes", so the comparison is strictly less than.
MAX_DURATION_S = 12 * 60

# ============================================================================
# Video handling
# ============================================================================

# The paper clips to 4-minute segments and carries annotation history forward.
CLIP_SECONDS = 240

# Video is compressed for the temporal pass; grounding then recovers precision
# from full-resolution frames. This split is the paper's own design (Section 3.3).
VIDEO_LONG_SIDE = 1280       # 720p-ish, keeps base64 near 60 MB per 4 min
VIDEO_CRF = 30               # screen content tolerates a high CRF well
VIDEO_PRESET = "veryfast"

# Grounding frames: three around each action, per Prompt 5.
FRAME_OFFSETS = (-0.5, 0.0, 0.5)
FRAME_FORMAT = "png"         # PNG, not JPEG: UI text edges are the whole point

# Grounding frames are stored at the SOURCE video's native resolution, always.
# Nothing in this pipeline resamples them, because a frame's pixels are the
# coordinate space: `target_x` / `target_y` are pixels in this exact image.
#
# Verified, not assumed. A marker drawn at pixel (1000, 500) in a 1920x1080
# source is still at (1000, 500) in the extracted frame, and the stored row's
# `image_width` / `image_height` equal the source video's dimensions.
#
# The ONE resample in the pipeline is on the video clip for stage 1 (below), and
# that branch produces no coordinates. That split is deliberate -- see
# `VIDEO_LONG_SIDE`.
#
# MAX_FRAME_LONG_SIDE is a sanity ceiling, not a resize. Above it the pipeline
# warns loudly and records the oversized frame, because a 4K-or-larger capture
# yields multi-megabyte PNGs and three of them per action may exceed the
# provider's request size limit. 0 disables the check.
MAX_FRAME_LONG_SIDE = 2560      # 1440p passes; 4K and above is reported

# ============================================================================
# Grounding gates
# ============================================================================

# Gemini returns 0-1000 normalized. We store absolute pixels. Anything outside
# this range is a hallucination and the row is rejected.
NORM_MIN, NORM_MAX = 0, 1000

# If the returned point and the centre of the returned bbox disagree by more than
# this (in normalized units), the axes are probably swapped and the row is
# rejected rather than written. A genuine bbox centre and point agree closely.
POINT_BBOX_MAX_DISAGREEMENT = 120

# Reject a point that lands within this many pixels of the image edge. Actions
# near the border are usually misreads of docked panels or the cursor itself.
EDGE_MARGIN_PX = 2

# ============================================================================
# Dataset
# ============================================================================

# Our id scheme mirrors GUI-360: <app>_<video>_<step>, with the trajectory key
# being the id minus the trailing step index.
ID_PREFIX = "wilcom"
SOURCE_APP = "wilcom"
SOFTWARE_NAME = "Wilcom EmbroideryStudio"

# Split by trajectory, never by row. Steps inside one video are near-duplicates,
# so a row-wise split leaks. This is the single most important property to keep.
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
SPLIT_SEED = 42

# Schema version for the exported dataset. Bump it when the row shape changes, so
# a manifest written by an older run can be told apart from a current one.
DATASET_SCHEMA = "wilcom-ds-v1"

# The prompt and answer templates are NOT here. They belong with the rest of the
# prompting in `prompts.py`, which is also where they are rendered -- see
# `prompts.render_prompt` and `prompts.render_answer`.

# ============================================================================
# Screening thresholds -- the paper's Section 3.1 coarse-to-fine filter
# ============================================================================
#
# Prompt 1 asks for an is_gui_content flag plus a confidence in 0.0-1.0. Prompt 2
# scores three dimensions from 1 to 5. These are the cut-offs applied to those
# answers.
#
# Unlike the grounding gates, which reject on arithmetic, these are editorial: a
# video scoring 3 on instruction clarity is not wrong, it is merely less useful.
# So they are deliberately permissive here, and a screening record keeps every
# answer so the thresholds can be re-decided from the log rather than re-run.
MIN_META_CONFIDENCE = 0.5   # below this, treat "yes" as a guess
MIN_QUALITY_SCORE = 3       # out of 5, applied to all three dimensions

# ============================================================================
# Paths
# ============================================================================


@dataclass
class Paths:
    """Where each stage reads and writes.

    `root` is relative by default, so it resolves against the current working
    directory. That is what a CLI user expects, but it does mean a library caller
    should pass an absolute root rather than relying on the default.
    """

    root: Path = Path("wilcom_work")

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def traj(self) -> Path:
        return self.root / "trajectory"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def ground(self) -> Path:
        return self.root / "grounding"

    @property
    def out(self) -> Path:
        return self.root / "dataset"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def screen(self) -> Path:
        """Screening records, keyed by video id.

        Separate from `traj` because a screening record exists for videos that
        were *rejected* and never downloaded, which a trajectory directory cannot
        represent.
        """
        return self.root / "screening"

    def ensure(self) -> "Paths":
        for p in (self.root, self.clips, self.videos, self.traj,
                  self.frames, self.ground, self.out, self.logs, self.screen):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ============================================================================
# Credentials
# ============================================================================


def _key_from_local_secrets() -> str:
    """Read the gitignored key file, if it is there. Never raises."""
    try:
        from . import local_secrets
    except Exception:
        # Absent on a fresh clone, which is the normal case for anyone else.
        return ""
    return str(getattr(local_secrets, "GEMINI_API_KEY", "") or "").strip()


def _key_from_env_file() -> tuple[str, str]:
    """Read GEMINI_API_KEY or GOOGLE_API_KEY from .env file if present.
    
    Returns (source_label, key) or ("", "").
    """
    candidates = (
        Path(".env"),
        Path(__file__).resolve().parent.parent / ".env",
    )
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                            return (".env file", v)
            except Exception:
                pass
    return ("", "")


def key_source() -> tuple:
    """Where the key came from, and its last 4 characters.

    Returns `(source, tail)` with `("", "")` when no key is configured. The tail is
    enough to tell two keys apart without putting a usable credential in a log, a
    screenshot or a bug report.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name)
        if v and v.strip():
            return ("%s environment variable" % name, v.strip()[-4:])
    env_src, env_key = _key_from_env_file()
    if env_key:
        return (env_src, env_key[-4:])
    key = _key_from_local_secrets()
    if key:
        return ("wilcom_pipeline/local_secrets.py", key[-4:])
    return ("", "")


def api_key() -> str:
    """Google AI Studio API key.

    Looked up in this order:

      1. `GEMINI_API_KEY`, then `GOOGLE_API_KEY`, in the environment.
      2. `.env` file in the project directory.
      3. `wilcom_pipeline/local_secrets.py`, which is gitignored.

    The environment wins so that one run can use a different key without editing a
    file, and so a CI machine can inject its own.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()

    _src, env_key = _key_from_env_file()
    if env_key:
        return env_key

    key = _key_from_local_secrets()
    if key:
        return key

    raise SystemExit(
        "No API key found. Any ONE of these is enough:\n"
        "  1. Environment variable:\n"
        "       PowerShell:  $env:GEMINI_API_KEY = 'your_api_key'\n"
        "       bash:        export GEMINI_API_KEY=your_api_key\n"
        "  2. .env file in project root:\n"
        "       GEMINI_API_KEY=your_api_key\n"
        "  3. wilcom_pipeline/local_secrets.py:\n"
        "       GEMINI_API_KEY = 'your_api_key'\n"
        "Get one at https://aistudio.google.com/apikey"
    )


# ============================================================================
# Coordinate convention -- asserted, not assumed
# ============================================================================


def norm_to_abs(norm: float, size: int) -> int:
    """0-1000 normalized -> absolute pixels, matching GUI-360's own convention."""
    return int(round(norm / 1000.0 * size))


def abs_to_norm(px: float, size: int) -> int:
    """Absolute pixels -> 0-1000 normalized."""
    return int(round(px / float(size) * 1000.0))


def verify_coordinate_convention() -> None:
    """Assert the conversion against a known real row from GUI-360.

    Measured from `junior0091/GUI-360-office-5k`, first training row:

        target_x = 506   target_x_norm = 487   image_width = 1040
        target_y = 217   target_y_norm = 295   image_height = 736

    If this ever fails, the convention has changed somewhere and every coordinate
    this pipeline writes would be silently wrong. That exact class of mismatch is
    what made `office-5k` unusable, so it is a hard assert rather than a comment.
    """
    checks = [
        (487, 1040, 506),
        (295, 736, 217),
    ]
    for norm, size, expected in checks:
        got = norm_to_abs(norm, size)
        if got != expected:
            raise AssertionError(
                "coordinate convention broken: norm=%d size=%d gave %d, "
                "expected %d. Every downstream coordinate would be wrong."
                % (norm, size, got, expected)
            )

    # And the inverse must round-trip, or we would drift on re-export.
    for norm, size, _ in checks:
        px = norm_to_abs(norm, size)
        if abs_to_norm(px, size) != norm:
            raise AssertionError(
                "coordinate round-trip broken for norm=%d size=%d" % (norm, size)
            )


verify_coordinate_convention()
