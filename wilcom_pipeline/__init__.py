"""WILCOM Video2GUI pipeline.

Turns curated screen-recording tutorials into a GUI-grounding dataset, following
the three-stage structure of the Video2GUI paper (arXiv 2605.14747):

    stage_trajectory   video      -> sub-tasks and atomic actions with instructions
    stage_grounding    actions    -> pixel coordinates for each clickable element
    stage_dataset      both       -> GUI-360-schema rows, split, pushed to the Hub

The paper goes looking for videos on the internet first (Section 3.1). We skip
that entirely, because the input is curated Wilcom tutorials. So this package
implements Sections 3.2 and 3.3, and the filtering stage has no counterpart here.

Modules are layered, and imports only ever point downward:

    foundations   config      settings, paths, model roles, coordinate conversion
                  timecode    mm:ss <-> seconds
                  hashing     cache keys and content digests
                  records     reading and writing stage JSON
                  errors      transient vs permanent failures
                  replies     pulling JSON out of a model reply
                  ledger      cost tracking and model pricing
                  log         console progress, timestamps, heartbeat
                  video       ffmpeg: probe, clip, extract frames
                  metadata    yt-dlp: title, description, tags, transcript

    interface     gemini_client        the API client
                  prompts              the paper's prompts, verbatim
                  clients              builds a configured client
                  cost                 offline spend estimation

    stages        stage_filter         stage 0: the paper's Prompts 1-2 screening
                  stage_trajectory     stage 1
                  stage_grounding      stage 2
                  stage_dataset        stage 3: assembly, split, verify
                  stage_review         stage 4: screenshot recovery & review
                  hf_export            stage 3: export to Hugging Face
                  report               visual coordinate check
                  assess               offline candidate analysis

    drivers       pipeline             runs the stages in order
                  cli                  argparse entry point

Nothing in `foundations` imports from `interface` or `stages`. That is what keeps
the stages testable with no API key and no network.
"""

__version__ = "0.2.0"

__all__ = [
    # foundations
    "config",
    "timecode",
    "hashing",
    "records",
    "errors",
    "replies",
    "ledger",
    "log",
    "video",
    "metadata",
    # interface
    "gemini_client",
    "prompts",
    "clients",
    "cost",
    # stages
    "stage_filter",
    "stage_trajectory",
    "stage_grounding",
    "stage_dataset",
    "stage_review",
    "hf_export",
    "report",
    "assess",
    # drivers
    "pipeline",
    "cli",
]
