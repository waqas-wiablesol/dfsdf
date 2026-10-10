"""Reading and writing the JSON records each stage leaves on disk.

One module so that encoding, indentation, and the question of "which files in a
stage directory are records" are each decided exactly once.

That last point matters: a stage directory also holds `manifest.json` and
`summary.json`, which are not per-video records. Reading those back in as if
they were would fail confusingly several stages later.
"""

from __future__ import annotations

import json
from pathlib import Path

# Files a stage writes for humans or for the next stage, not per-video records.
NON_RECORD = frozenset({"manifest.json", "summary.json", "index.json"})


def write_json(obj, path) -> Path:
    """Write a record with a stable encoding and indentation.

    `ensure_ascii=False` so a non-English instruction stays readable, and
    `indent=2` so a bad record can be inspected in an editor without being
    reformatted first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def read_json(path, default=None):
    """Parse a JSON file, returning `default` when it is absent or malformed.

    A missing cache file and a corrupt cache file are both "no cached result",
    so they are treated identically. Callers that need to distinguish them
    should check `Path.exists()` themselves.
    """
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_dir(directory, verbose: bool = False) -> list[dict]:
    """Every stage record in a directory, ordered by filename.

    One unreadable file is reported and skipped rather than aborting the load:
    the remaining records are still perfectly usable, and refusing to proceed
    because one file is corrupt would throw away an expensive run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name in NON_RECORD:
            continue
        record = read_json(path)
        if record is None:
            if verbose:
                print("  [warn] unreadable record, skipped: %s" % path.name)
            continue
        if record:
            records.append(record)
    return records
