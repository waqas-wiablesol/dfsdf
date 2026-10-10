"""Stage 4 / Review: Screenshot recovery & inspection report.

Validates grounded frames on disk, extracts missing frames via ffmpeg if
needed, and produces a review summary report with dataset consistency checks.
"""

from __future__ import annotations

from typing import Any, Dict

from . import config


def run(paths: config.Paths, verbose: bool = True) -> Dict[str, Any]:
    """Execute screenshot recovery and verification report.
    
    Delegates to jobs.run_review(paths).
    """
    from . import jobs

    return jobs.run_review(paths)
