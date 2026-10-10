"""Cost estimation, with no API calls.

The point is to know what a corpus costs *before* spending it. The trajectory
stage bills by video-second and is dominated by thinking tokens; the grounding
stage bills per action, which is not knowable without running stage 1.

The defaults here are **measured**, not assumed. An earlier version of this file
took the paper's reported 15,908 prompt tokens per 240 s clip and predicted ~$0.31
for stage 1; the real figure was 18,599 prompt tokens plus 978 thinking tokens, so
the estimate was out by roughly 2x. Everything below marked "measured" was read off
a real run, and can be re-read the same way from a stage record.

Every quantity is a field of `Assumptions`, so it can be argued with rather than
taken on faith, and the uncertain ones are repeated as caveats in the output. An
estimate that hides its assumptions is worse than no estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config, ledger, video

# The paper's reported average cost per sample. Printed as a reference point, not
# used in the arithmetic: its samples average fewer actions than a dense WILCOM
# tutorial, so matching it exactly is not the goal.
PAPER_USD_PER_SAMPLE = 0.0763


@dataclass
class Assumptions:
    """Every quantity the estimate depends on."""

    # MEASURED, one 240 s 1080p60 screen recording: 18,599 prompt tokens, i.e.
    # ~77.5 tokens/s. Higher than the paper's implied 66.3, which is consistent
    # with 1080p60 carrying more per second than their average source.
    video_tokens_per_240s: int = 18599

    # MEASURED at thinking_level="low": 2,566 visible + 978 thinking = 3,544.
    # Thinking cannot be disabled on Gemini 3, so it is part of the output cost
    # rather than overhead that can be trimmed to zero.
    output_tokens_per_clip_low: int = 3544

    # MEASURED at the model's default thinking: 4,101 visible + 11,681 thinking.
    output_tokens_per_clip_default: int = 15782

    # GUESS, and the widest one. Measured on the first Wilcom clip: 15 actions in
    # 240 s, i.e. 3.75/min -- but that was an exploratory tour, which is the
    # low-value case. A hands-on workflow should be denser. Kept mid-range so the
    # estimate neither flatters nor alarms.
    actions_per_minute: float = 8.0

    # MEASURED, flash-lite on one 1920x1080 frame: 1,844 prompt tokens.
    frame_tokens: int = 1844

    # MEASURED: the first frame grounded on 7 of 8 actions, so the mean is close
    # to 1. 1.15 allows occasional retries without assuming they are rare.
    frames_per_action: float = 1.15

    # MEASURED, flash-lite: 53-142 visible tokens, 79-114 thinking.
    ground_output_tokens: int = 100
    ground_thinking_tokens: int = 95

    # MEASURED, screening on metadata alone: a few hundred tokens in, tens out.
    meta_input_tokens: int = 1200
    meta_output_tokens: int = 250


@dataclass
class Estimate:
    """A costed plan. All money in USD."""

    videos: int
    seconds: float
    clips: int
    clip_seconds: int

    traj_input_tokens: int = 0
    traj_output_tokens: int = 0
    traj_usd: float = 0.0

    actions: int = 0
    ground_input_tokens: int = 0
    ground_usd: float = 0.0

    meta_usd: float = 0.0

    thinking_level: str = "low"
    assumptions: Assumptions = field(default_factory=Assumptions)

    @property
    def total_usd(self) -> float:
        return self.traj_usd + self.ground_usd + self.meta_usd

    @property
    def per_sample_usd(self) -> float:
        """Cost per grounded action, the unit the paper quotes in."""
        return self.total_usd / self.actions if self.actions else 0.0

    @property
    def per_video_usd(self) -> float:
        return self.total_usd / self.videos if self.videos else 0.0

    def describe(self) -> str:
        """A printable summary, assumptions included.

        The caveats are part of the output rather than a docstring because this is
        the text someone will use to decide whether to spend money.
        """
        traj_model = config.model_for("trajectory")
        ground_model = config.model_for("grounding")
        lines = [
            "videos        : %d" % self.videos,
            "total runtime : %.1f min" % (self.seconds / 60.0),
            "clips         : %d  (%.0f s each)" % (self.clips, self.clip_seconds),
            "",
            "stage 0, screening (%s, metadata only):" % config.model_for("meta"),
            "  est. cost    ~ $%.4f   (%d candidate(s))"
            % (self.meta_usd, self.videos),
            "",
            "stage 1, trajectory (%s, thinking=%s):"
            % (traj_model, self.thinking_level),
            "  video tokens ~ %s  (measured 18,599 per 240 s)"
            % f"{self.traj_input_tokens:,}",
            "  output+think ~ %s" % f"{self.traj_output_tokens:,}",
            "  est. cost    ~ $%.2f" % self.traj_usd,
            "",
            "stage 2, grounding (%s):" % ground_model,
            "  actions      ~ %s  (%.1f per minute as modelled)"
            % (f"{self.actions:,}", self.assumptions.actions_per_minute),
            "  frames/action: %.2f of a possible %d"
            % (self.assumptions.frames_per_action, len(config.FRAME_OFFSETS)),
            "  est. cost    ~ $%.2f" % self.ground_usd,
            "",
            "TOTAL          ~ $%.2f" % self.total_usd,
            "per video      ~ $%.4f" % self.per_video_usd,
            "per action     ~ $%.4f  (the paper reports $%.4f)"
            % (self.per_sample_usd, PAPER_USD_PER_SAMPLE),
            "",
            "Caveats, in order of how much they move the number:",
            "  * actions per minute is the widest guess and drives the grounding",
            "    cost. Measured: 3.75/min on an exploratory tour, which is the",
            "    low-value case.",
            "  * stage 1 is dominated by THINKING tokens, which bill at the output",
            "    rate. At the model's default thinking the stage-1 cost is ~2.9x",
            "    the figure above. Set THINKING_LEVEL in config.py.",
            "  * grounding assumes the first frame usually lands (measured 7/8).",
            "    If it does not, up to %.0fx more calls are made."
            % len(config.FRAME_OFFSETS),
            "  * prices come from a local table in ledger.py, so they do not",
            "    follow a price change. Re-check before a large run.",
            "",
            "Nothing here is billed. Stage 1 dominates: it sends whole video.",
        ]
        return "\n".join(lines)


def _tokens_per_clip(clip_seconds: int, assumptions: Assumptions) -> int:
    """Scale the measured token count for a different clip length."""
    return int(assumptions.video_tokens_per_240s * (clip_seconds / 240.0))


def _output_tokens_per_clip(clip_seconds: int, assumptions: Assumptions,
                            thinking_level: str) -> int:
    """Output tokens for a clip, thinking included.

    Scales with clip length, because both the answer and the reasoning grow as the
    segment contains more to describe.
    """
    base = (assumptions.output_tokens_per_clip_low
            if thinking_level in ("low", "none", None)
            else assumptions.output_tokens_per_clip_default)
    return int(base * (clip_seconds / 240.0))


def estimate(infos: list, clip_seconds: int | None = None,
             assumptions: Assumptions | None = None,
             thinking_level: str | None = None,
             include_screening: bool = True) -> Estimate:
    """Cost a set of already-probed videos.

    Takes `video.VideoInfo` objects rather than paths, because probing is the
    caller's job and a video that would not probe should be excluded before it ever
    reaches a cost estimate.
    """
    assumed = assumptions or Assumptions()
    clip_seconds = clip_seconds or config.CLIP_SECONDS
    thinking_level = thinking_level or config.THINKING_LEVEL

    clips = sum(len(video.plan_clips(info, clip_seconds)) for info in infos)
    seconds = sum(float(info.duration) for info in infos)

    traj_in = clips * _tokens_per_clip(clip_seconds, assumed)
    traj_out = clips * _output_tokens_per_clip(clip_seconds, assumed,
                                               thinking_level)

    actions = int(seconds / 60.0 * assumed.actions_per_minute)
    frames = actions * assumed.frames_per_action
    ground_in = int(frames * assumed.frame_tokens)
    ground_out = int(frames * (assumed.ground_output_tokens
                               + assumed.ground_thinking_tokens))

    traj_in_price, traj_out_price = ledger.price_for(config.model_for("trajectory"))
    ground_in_price, ground_out_price = ledger.price_for(config.model_for("grounding"))
    meta_in_price, meta_out_price = ledger.price_for(config.model_for("meta"))

    meta_usd = 0.0
    if include_screening:
        meta_usd = len(infos) * (
            assumed.meta_input_tokens * meta_in_price
            + assumed.meta_output_tokens * meta_out_price)

    return Estimate(
        videos=len(infos),
        seconds=seconds,
        clips=clips,
        clip_seconds=clip_seconds,
        traj_input_tokens=traj_in,
        traj_output_tokens=traj_out,
        traj_usd=traj_in * traj_in_price + traj_out * traj_out_price,
        actions=actions,
        ground_input_tokens=ground_in,
        ground_usd=ground_in * ground_in_price + ground_out * ground_out_price,
        meta_usd=meta_usd,
        thinking_level=thinking_level,
        assumptions=assumed,
    )
