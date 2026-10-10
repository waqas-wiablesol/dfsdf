# WILCOM Video2GUI — Implementation Plan

Derived from **Video2GUI** (arXiv 2605.14747), the pipeline behind WildGUI, which is
the upstream source for `junior0091/GUI-360-word-unsloth`.

Our variant takes **curated WILCOM tutorial videos as input** rather than crawling
YouTube, so the paper's filtering stage (§3.1) is dropped entirely. What remains is
trajectory extraction and spatial grounding, then dataset assembly.

---

## What the paper actually does

| Stage | Model | Input | Output |
|---|---|---|---|
| §3.1 Filtering | DeepSeek-V3 → Qwen2.5-7B, then Gemini-3-Pro → Qwen2.5-Omni | metadata, video | *skipped for us* |
| §3.2 Trajectory extraction | **Gemini-3-Pro**, Prompt 3 + 4 | video segment ≤ 4 min | shot list + task JSON |
| §3.3 Action spatial grounding | **Gemini-3-Pro**, Prompt 5 | 3 high-res frames at `t-0.5s, t, t+0.5s` | point + bbox, 0–1000 |

Two things the paper states that shape our design:

> *"the input videos are compressed to accommodate long-context processing, which
> inherently limits the visual resolution required for precise pixel-level
> localization. Therefore, we introduce the action spatial grounding stage"*

So **two different resolutions for two different jobs** — compressed video for
temporal reasoning, full-resolution frames for precision. We reproduce that split
exactly.

> *"Among the three candidate frames, we select the first frame for which the model
> successfully produces a valid spatial grounding result"*

And if all three fail, the action is **discarded**. That is a quality gate we keep.

---

## Verified environment facts

Checked against live APIs, not assumed:

| Fact | Value |
|---|---|
| OpenRouter models | 460 total, 33 Gemini-family, **85 accept video** |
| Pro model with video | **`google/gemini-3.1-pro-preview`** — modalities `[audio, file, image, text, video]` |
| Cheaper fallbacks with video | `google/gemini-3.8-flash`, `google/gemini-3.5-flash` |
| Video input format | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |
| Images | `data:image/png;base64,...` |
| Pricing (Pro) | ~$1/M input, $6/M output |
| Official Python SDK | **`pip install openrouter`** (v1.3.2, generated from their OpenAPI spec) |
| ffmpeg on this machine | **absent** — bundle `imageio-ffmpeg` static binary |

Payload reality for a 4-minute clip:

| Quality | Raw | base64 |
|---|---|---|
| 720p, modest bitrate | 45 MB | **60 MB** |
| 1080p typical | 90 MB | 120 MB |
| 1080p crisp | 180 MB | 240 MB |

→ **Downscale the video pass to 720p.** The paper's own 15,908 tokens for 240 s
works out to 66 tokens/s, which is already a compressed representation, so we are
matching their regime rather than degrading it.

---

## Two coordinate traps, both already seen in this project

### Trap 1 — axis order

Prompt 5 specifies `<point>y x</point>` and `<bbox>y1 x1 y2 x2</bbox>`. That is
**y-first**. GUI-360 and our dataset use **`[x, y]`**. Mixing these silently swaps
every coordinate.

**Mitigation:** parse into explicitly named variables, never index a tuple. Then
**verify** the point against the bbox centre — if they disagree by more than a
few units, the axes are swapped and the row is rejected rather than written.

### Trap 2 — normalized vs absolute

Gemini returns **0–1000 normalized**. `GUI-360-word-unsloth` stores **absolute
pixels**. `junior0091/GUI-360-office-5k` had exactly this mismatch — its `text`
column was normalized while `answer_raw` was absolute, and a model trained on it
would score zero on our metric.

**Conversion, verified against real data:** for `target_x=506`, `target_x_norm=487`,
`W=1040`, the relation is

```
abs = round(norm / 1000 * W)        487 / 1000 * 1040 = 506.48  →  506  ✓
```

The code asserts this round-trip on a known pair at startup rather than trusting it.

---

## Pipeline

```
video.mp4 ──┬─► probe (duration, fps, resolution)
            │
            ├─► Stage 0: clip to ≤4 min, downscale 720p  ──► work/clips/*.mp4
            │
            ├─► Stage 1: trajectory extraction            ──► work/traj/*.json
            │      video_url base64 + Prompt 3
            │      carry annotation history across clips (Prompt 4)
            │      → tasks[{instruction, actions[{timestamp, type,
            │                                    grounding_instruction, params}]}]
            │
            ├─► Stage 2: spatial grounding                ──► work/ground/*.json
            │      for each action: 3 full-res PNG frames at t±0.5s
            │      image_url ×3 + Prompt 5
            │      feasibility check → point + bbox, 0–1000
            │      first frame that grounds wins; all fail → drop action
            │
            └─► Stage 3: dataset assembly                 ──► out/dataset/
                   normalized → absolute pixels
                   → GUI-360 row schema
                   → split by trajectory (no leakage)
                   → push to Hub
```

### Per-artefact schema

Stage 1 output, per clip:
```json
{
  "clip": "wilcom_01_part0.mp4",
  "tasks": [{
    "task_id": 0,
    "instruction": "…",
    "platform": "windows",
    "software": "Wilcom EmbroideryStudio",
    "user_actions": [
      {"timestamp": "00:14", "action_type": "click",
       "grounding_instruction": "Click the 'Insert' tab…",
       "action_parameters": {"point": "[y, x]"}}
    ]
  }]
}
```

Stage 2 output, per action:
```json
{"timestamp": "00:14", "feasible": true, "frame": 0,
 "norm_x": 487, "norm_y": 295,
 "abs_x": 506, "abs_y": 217,
 "bbox_norm": [y1, x1, y2, x2], "frame_path": "frames/…_f0.png"}
```

Final dataset row — identical schema to `GUI-360-word-unsloth`:
```
id, trajectory, instruction, prompt, image,
target_x, target_y, image_width, image_height, answer_raw, messages
```

---

## Deliberate design decisions

| Decision | Why |
|---|---|
| **Keep the paper's prompts verbatim** | They are the validated part. Any rewording invalidates the comparison. WILCOM-specific context goes in a separate preamble, not inside the prompt body. |
| **720p for video, full-res for frames** | Matches the paper's own two-resolution design and keeps requests near 60 MB. |
| **PNG frames, not JPEG** | UI text has sharp edges; JPEG ringing around glyphs degrades exactly the detail grounding needs. |
| **Reject rather than guess** | A swapped axis or a point far from its bbox is dropped, not repaired. Silent coordinate corruption is the failure mode this project has already been bitten by twice. |
| **Split by trajectory, not row** | Steps in one video are near-duplicates. A row-wise split leaks and inflates validation. This is why `word-unsloth` was built the way it was. |
| **Idempotent stages** | Each stage caches to disk keyed by input hash. A crash at stage 2 does not re-pay for stage 1. |

---

## Cost

Per 4-minute clip, roughly 20 actions:

| Step | Tokens | Cost |
|---|---|---|
| Trajectory (video in ~19k, out ~6.5k) | ~25k | ~$0.058 |
| Grounding (3 frames × 20 actions) | ~76k | ~$0.14 |
| **Total per clip** | | **~$0.20** |

Grounding dominates because it runs per action, not per clip. The paper quotes
$0.0763/sample, but that figure is a *whole-video* average with fewer actions per
sample; for a dense tutorial with many clicks, expect more.

**A 20-clip pilot ≈ $4.** Worth doing before committing to a full corpus.

---

## Build order

1. `config.py` — settings, model IDs, paths, the coordinate assertion
2. `openrouter_client.py` — SDK wrapper: retries, JSON extraction, cost ledger
3. `video.py` — probe / clip / downscale / frame extraction via bundled ffmpeg
4. `prompts.py` — Prompts 3, 4, 5 verbatim + WILCOM preamble
5. `stage_trajectory.py`
6. `stage_grounding.py`
7. `stage_dataset.py`
8. `pipeline.py` + `cli.py`
9. `ui.py` — Gradio
10. tests at every stage, mock-first

---

## What we are NOT doing

- **Not** reimplementing §3.1 filtering — we have curated videos.
- **Not** using Prompts 1, 2, 6, 7, 8 — those are filtering and *training* prompts,
  not pipeline prompts. Including them would be a misreading of the paper.
- **Not** training anything. This pipeline produces a dataset. Training is a
  separate step, with `word-unsloth` as the format reference.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Gemini's video limits differ from OpenRouter's docs | Probe with one short clip before a batch |
| 60 MB base64 request rejected | Fall back to per-frame image input, which loses temporal context but still works |
| Model invents actions not in the video | Require `feasible=false` handling; drop actions whose bbox is implausible |
| WILCOM UI differs from generic desktop | The desktop action space in Table 6 is generic; if WILCOM needs embroidery-specific verbs we keep the generic set and note it |
