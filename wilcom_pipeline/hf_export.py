"""Export assembled rows to a Hugging Face dataset.

Split out from the assembly stage for one practical reason: `datasets` is a heavy
optional dependency, and assembling, splitting and verifying rows needs none of
it. Keeping the import inside this module means the pure-Python half of stage 3
works on a machine that has never installed it.
"""

from __future__ import annotations

from . import config, prompts, records

# The parent dataset's columns, in its order. Copied from
# `junior0091/GUI-360-word-unsloth` itself -- 11 columns, verified against the live
# dataset rather than against a note in this repository, because the note was wrong
# once already (it claimed `answer_raw` was normalized; it is pixels).
PARENT_COLUMNS = (
    "id", "trajectory", "instruction", "prompt", "image",
    "target_x", "target_y", "image_width", "image_height",
    "answer_raw", "messages",
)


def features():
    """The `datasets` schema, matching `GUI-360-word-unsloth`.

    `messages` is nested, and a `datasets` Sequence requires every element to
    carry the *same* keys. That is why the image placeholder below carries a
    `text` key at all, holding None: a bare {"type": "image"} cannot be expressed
    in this schema. The parent dataset is shaped the same way for the same reason,
    so this is compatibility as well as necessity.

    The nesting is written as Python **lists of dicts**, not as explicit
    `Sequence({...})` calls. Both look equivalent and they are not: with
    `Sequence({...})` at the outer level, `Dataset.from_list` encodes the column
    as a dict and calls `.get()` on what is actually the list of messages, giving

        AttributeError: 'list' object has no attribute 'get'

    The list form is the one the Word dataset ships and trains with, so it is both
    the working form and the compatible one.

    `datasets` is imported here rather than at module scope so that importing
    this module does not require it.
    """
    from datasets import Features, Image as HFImage, Value

    return Features({
        "id": Value("string"),
        "trajectory": Value("string"),
        "instruction": Value("string"),
        "prompt": Value("string"),
        "image": HFImage(),
        "target_x": Value("int32"),
        "target_y": Value("int32"),
        "image_width": Value("int32"),
        "image_height": Value("int32"),
        "answer_raw": Value("string"),
        "messages": [{
            "role": Value("string"),
            "content": [{
                "type": Value("string"),
                "text": Value("string"),
            }],
        }],
        # Provenance columns beyond the parent schema.
        "source_video": Value("string"),
        "task_id": Value("int32"),
        "step": Value("int32"),
        "action_type": Value("string"),
        "timestamp_s": Value("float32"),
        "bbox": Value("string"),
        "sub_task": Value("string"),
    })


def _has_value(value) -> bool:
    """Whether a column holds something, for rows read back off disk.

    Every comparison here is guarded, because the rows come from
    `pandas.read_parquet` and pandas hands back things that do not compare like
    scalars:

      - a null is `NaN`, not `None`, and `NaN` is not equal to anything including
        itself -- so `value in (None, "", [], {})` reports every nullable column as
        fully populated, which is the one thing this count exists to reveal;
      - the `image` column comes back as an array of the struct's fields, and
        `array == ""` returns an *array*, so `or` raises `ValueError: The truth
        value of an array with more than one element is ambiguous`. That is a 500
        on the endpoint, not a wrong number.

    So the body is inside the `try` rather than the comparisons being checked
    first. Anything that gets to the handler is a value that exists but does not
    compare to a string, which means present -- the safer of the two guesses, since
    calling a populated column empty would hide real data.
    """
    if value is None:
        return False
    try:
        # A struct column (the image, with `bytes` and `path`) is empty when all of
        # its fields are. Recursing handles the field-level None as well.
        if isinstance(value, dict):
            return any(_has_value(v) for v in value.values())
        if isinstance(value, (list, tuple, set, frozenset)) and not value:
            return False
        if value == "" or value == b"":
            return False
        # `NaN != NaN` is the test that catches a pandas null without importing
        # numpy into this module.
        return not bool(value != value)
    except Exception:
        return True


def schema(rows: list | None = None) -> dict:
    """The dataset's real columns, and what each one is for.

    Derived from `features()` rather than written out by hand, so this cannot drift
    from what is actually exported. The split into parent / ours is by name against
    a list copied from the live parent dataset, which is the same kind of claim the
    docstrings make -- except this one is checkable, because `verify_against_parent`
    compares it.

    `rows` is optional. When given, per-column presence is reported as well, so a
    column that exists in the schema but never holds a value is visible.
    """
    feats = features()
    parent = list(PARENT_COLUMNS)
    out = []
    present: dict = {}
    if rows:
        for name in feats:
            present[name] = sum(1 for r in rows if _has_value(r.get(name)))

    for name, spec in feats.items():
        entry = {
            "name": name,
            "type": _type_name(spec),
            "from_parent": name in parent,
            "populated": present.get(name) if rows else None,
        }
        out.append(entry)

    return {
        "columns": out,
        "count": len(out),
        "parent_columns": parent,
        "parent_count": len(parent),
        "extra": [c["name"] for c in out if not c["from_parent"]],
        # Order matters: a consumer that reads by position, or a diff against the
        # parent, sees a difference if the first eleven are not in the same order.
        "order_matches_parent": [c["name"] for c in out][:len(parent)] == parent,
        "rows": len(rows) if rows else None,
    }


def _type_name(spec) -> str:
    """A readable type for a `datasets` feature, including the nested ones."""
    cls = type(spec).__name__
    if cls == "Image":
        return "Image"
    inner = getattr(spec, "dtype", None)
    if inner is not None:
        return str(inner)
    # Something nested. Describe its shape rather than dumping a repr.
    try:
        if isinstance(spec, list) and spec and isinstance(spec[0], dict):
            return "list<{%s}>" % ", ".join(spec[0].keys())
    except Exception:
        pass
    return cls


def verify_against_parent() -> list:
    """Problems with the claim that our first columns match the parent, exactly.

    A plain list of strings, empty when the claim holds. Checks order as well as
    membership, because "the same columns" and "the same columns in the same places"
    are different promises and only the second one is compatible.
    """
    ours = [c["name"] for c in schema()["columns"]]
    problems = []
    if ours[:len(PARENT_COLUMNS)] != list(PARENT_COLUMNS):
        problems.append(
            "the first %d columns are not the parent's, in order: ours=%s parent=%s"
            % (len(PARENT_COLUMNS), ours[:len(PARENT_COLUMNS)],
               list(PARENT_COLUMNS)))
    missing = [c for c in PARENT_COLUMNS if c not in ours]
    if missing:
        problems.append("missing parent columns: %s" % missing)
    return problems


def to_messages(prompt: str, answer: str) -> list:
    """The conversational pair Unsloth expects.

    The image itself is *not* embedded here. It is stored once, in the `image`
    column, and this carries only a placeholder. A 1040x736 frame decodes to
    about 2.3 MB of RGB, so embedding decoded images for thousands of rows runs
    a Colab kernel out of memory; the training notebook converts to the inline
    form lazily instead.

    The placeholder's `text` is None, matching the parent dataset exactly. It was
    an empty string here, which is the same shape but not the same value -- the
    kind of difference that survives every schema check and only shows up as a diff
    between two real rows.
    """
    return [
        {"role": "user", "content": [
            {"type": "image", "text": None},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]


def to_dataset(rows: list[dict]):
    """Build one split.

    Only file *paths* are passed in, so `Dataset.from_list` stores the encoded
    bytes with no decoded pixel buffers held anywhere.
    """
    from datasets import Dataset

    data = [{
        "id": r["id"],
        "trajectory": r["trajectory"],
        "instruction": r["instruction"],
        "prompt": r["prompt"],
        "image": r["image_path"],
        "target_x": r["target_x"],
        "target_y": r["target_y"],
        "image_width": r["image_width"],
        "image_height": r["image_height"],
        "answer_raw": r["answer_raw"],
        "messages": to_messages(r["prompt"], r["answer_raw"]),
        "source_video": r["source_video"],
        "task_id": r["task_id"],
        "step": r["step"],
        "action_type": r["action_type"],
        "timestamp_s": float(r["timestamp_s"]),
        "bbox": r["bbox"],
        "sub_task": r["sub_task"],
    } for r in rows]
    return Dataset.from_list(data, features=features())


def manifest(splits: dict) -> dict:
    """Everything needed to interpret the parquet files later.

    Records the prompt template and both coordinate conventions explicitly. A
    dataset whose scale convention is only in someone's memory is a dataset that
    will be misused.

    `consumer_notes` exists because the single most damaging mistake available to
    a consumer happens *outside* this repository: passing images through Unsloth's
    default collator resizing. That silently shrinks a 1920px frame to 768px, and
    the grounding accuracy drop is attributed to the data rather than to the
    collator. Recording the required setting here means the training script can
    read it instead of relying on a README nobody opens.
    """
    present = {name: rows for name, rows in splits.items() if rows}
    return {
        "schema": config.DATASET_SCHEMA,
        "prompt_template": prompts.INSTRUCTION_TEMPLATE,
        "answer_template": prompts.ANSWER_TEMPLATE,
        "source_app": config.SOURCE_APP,
        "software": config.SOFTWARE_NAME,
        "coordinate_convention": {
            "target_x / target_y": "absolute pixels on the row's own image",
            # This said "normalized 0-1000" and was simply wrong. Every `answer_raw`
            # in every one of the parent dataset's three splits is the absolute pixel
            # pair, exactly equal to `target_x` / `target_y` -- verified across 220
            # rows, not inferred. A manifest is the file someone reads to interpret
            # the parquet months later, so a wrong claim here misleads a consumer
            # that the numbers themselves would never have misled.
            "answer_raw": "absolute pixels on the row's own image, identical to "
                          "target_x / target_y, as <coordinate> [x, y] with the "
                          "closing tag on its own line",
            "note": "the two agree exactly, and verify re-checks that on every row",
        },
        "images": {
            "resolution": "native, as captured. Never resampled by this "
                          "pipeline.",
            "encoding": "PNG",
            "dimensions_recorded_in": ["image_width", "image_height"],
        },
        "consumer_notes": {
            "training_collator": "UnslothVisionDataCollator",
            "training_resize": "max",
            "why": "The collator's default resize='min' downscales to the "
                   "model's vision_config.image_size (768 for many Qwen "
                   "checkpoints), discarding most of the pixels these "
                   "coordinates were measured on. resize='max' means "
                   "'no resizing, leave the image intact'.",
            "also_set": {
                "resize_dimension": "'max' relies on the longer side; the "
                                    "default 0 keys off width",
            },
            "unavoidable": "The image processor still snaps dimensions to "
                           "28-pixel multiples (smart_resize). That is "
                           "inherent to Qwen2.5-VL and cannot be disabled.",
        },
        "splits": {
            name: {
                "rows": len(rows),
                "trajectories": len({r["trajectory"] for r in rows}),
                "videos": sorted({r["source_video"] for r in rows}),
            } for name, rows in present.items()
        },
        "totals": {
            "rows": sum(len(r) for r in present.values()),
            "trajectories": len({r["trajectory"]
                                 for rows in present.values() for r in rows}),
        },
    }


def save_local(splits: dict, paths: config.Paths) -> dict:
    """Write the single unified parquet dataset of all rows, plus a manifest. Returns {split: Dataset}."""
    out = paths.out
    out.mkdir(parents=True, exist_ok=True)

    # Remove any existing parquet files other than unified.parquet so only one file exists
    for stale in out.glob("*.parquet"):
        if stale.name != "unified.parquet":
            try:
                stale.unlink()
            except OSError:
                pass

    all_rows = []
    if "unified" in splits:
        all_rows = splits["unified"]
    else:
        for rows in splits.values():
            all_rows.extend(rows)

    built = {}
    if all_rows:
        unified_dataset = to_dataset(all_rows)
        unified_dataset.to_parquet(str(out / "unified.parquet"))
        built["unified"] = unified_dataset

    mani = manifest({"unified": all_rows})
    records.write_json(mani, out / "manifest.json")
    return built


