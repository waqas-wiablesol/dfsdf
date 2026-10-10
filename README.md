<h1 align="center">
Video2GUI: Synthesizing Large-Scale Interaction Trajectories<br>for Generalized GUI Agent Pretraining
</h1>

<div align="center">

![](https://img.shields.io/badge/Paper-arXiv-red)
![](https://img.shields.io/badge/Dataset-Released-blue)
![](https://img.shields.io/badge/Venue-ICML%202026-purple)
![](https://img.shields.io/badge/Code%20License-Apache%202.0-green)

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2605.14747"><b>[📜 arXiv]</b></a> •
  <a href="https://huggingface.co/datasets/xwm/WildGUI"><b>[🤗 WildGUI Dataset]</b></a> •
  <a href="https://github.com/WeiminXiong/Video2GUI"><b>[🐱 GitHub]</b></a> •
  <a href="INSTRUCTIONS.md"><b>[🛠️ Setup & Running Guide]</b></a>
</p>

This repository contains the code and data for the paper **"Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining"**, accepted at **ICML 2026**. For step-by-step instructions on setting up and running the interactive pipeline and web studio, see [INSTRUCTIONS.md](INSTRUCTIONS.md).

<p align="center">
<img src="static/images/teaser_figure.png" width="780"/>
</p>

Recent advances in multimodal large language models have driven growing interest in graphical user interface (GUI) agents, yet their generalization remains constrained by the scarcity of large-scale training data spanning diverse real-world applications. Existing datasets rely heavily on costly manual annotations and are typically confined to narrow domains.

To address this, we propose **Video2GUI**, a fully automated framework that extracts grounded GUI interaction trajectories directly from unlabeled Internet videos. Applying this pipeline to **500M+ video metadata entries**, we construct **WildGUI**, a large-scale dataset containing **12M interaction trajectories** spanning over **1,500 applications and websites**. Pre-training Qwen2.5-VL and MiMo-VL on WildGUI yields consistent improvements of **5–20%** across multiple GUI grounding and action benchmarks, matching or surpassing state-of-the-art performance.

## 🔥 News

- **[2026/06/14]** 🔥 The **WildGUI** dataset is fully open-sourced on 🤗 HuggingFace — [annotations + screenshots](https://huggingface.co/datasets/xwm/WildGUI).
- **[2026/05/01]** 🎉 Video2GUI has been accepted to **ICML 2026**!

## 📅 Come Find Us at ICML 2026

We will present Video2GUI as a **poster** at ICML 2026 — come and chat with us!

- **🗓️ When:** Tuesday, July 7, 2026 · 2:00 PM – 3:45 PM CST
- **📍 Where:** Hall A #4603

## 🧭 Method Overview

Video2GUI is a fully automated, three-stage pipeline that turns unlabeled Internet videos into grounded GUI interaction trajectories suitable for pretraining generalized GUI agents (see the figure above).

- **(A) Coarse-to-Fine Video Filtering.** Starting from 500M+ raw Internet videos, we apply meta-info filtering with a fine-tuned Qwen2.5-7B classifier (trained on DeepSeek-V3 annotations) to select ∼20M candidates, then a fine-grained content-based scorer rating instruction clarity, topic relevance, and screen-recording quality — yielding **4.2M high-quality tutorial videos (∼300k hours)**.
- **(B) Trajectory Extraction.** Each video is split into ≤4-minute segments and processed by Gemini-3-Pro under a sliding-window strategy with historical context, producing instruction–trajectory pairs with task instructions, action timestamps, action details, and visually grounded low-level instructions.
- **(C) Action Spatial Grounding.** For each interaction at timestamp *t*, we feed Gemini-3-Pro a triplet of high-resolution screenshots {*o*<sub>t−0.5s</sub>, *o*<sub>t</sub>, *o*<sub>t+0.5s</sub>} with the low-level instruction and predict the grounding target *b*<sub>t</sub> = (*x*₁, *y*₁, *x*₂, *y*₂). Manual verification on 200 sampled actions shows **>95%** are correctly parameterized.

## �️ Setup, Configuration & Running

The full walkthrough — every studio tab, every CLI subcommand, the output layout — is in [INSTRUCTIONS.md](INSTRUCTIONS.md). This is the short version.

### 1. Requirements

| | Version | Needed for |
|---|---|---|
| Python | **3.10+** | the pipeline and the API server |
| Node.js | **18+** | rebuilding the web UI only — *not* to run it |

The built React bundle is committed under `wilcom_pipeline/webui/`, so a plain install needs no Node at all.

### 2. Install

Dependencies are pinned in exactly two places, and both are the source of truth. Installed trees are gitignored: recreate them, never commit them.

```powershell
# Backend -- wilcom_pipeline/requirements.txt
python -m venv .venv
.venv\Scripts\Activate.ps1                 # Windows;  source .venv/bin/activate elsewhere
pip install -r wilcom_pipeline/requirements.txt

# Frontend -- frontend/package.json (only if you are changing the UI)
cd frontend; npm install; npm run build; cd ..
```

`npm run build` writes the bundle straight into `wilcom_pipeline/webui/`, which is what the API server serves.

### 3. Configure the API key

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). It is resolved at startup in this order (see [`config.py`](wilcom_pipeline/config.py)):

1. `GEMINI_API_KEY`, then `GOOGLE_API_KEY`, in the environment
2. a `.env` file in the project root
3. `wilcom_pipeline/local_secrets.py` — gitignored

The `.env` route is the least fiddly:

```powershell
copy .env.example .env      # then set GEMINI_API_KEY=... inside it
```

No key is ever read from a tracked file, so a fork cannot leak one by accident.

Model choice is separate from the key. `WILCOM_MODEL_TIER=economy` swaps the paper-faithful Gemini-3-Pro annotator for a cheaper set (the `grounding` role alone is ~8× cheaper per call). It is opt-in because it changes the data — see [`config.py`](wilcom_pipeline/config.py).

### 4. Run

```powershell
# Web studio: JSON API + the built UI on one port
.venv\Scripts\python.exe -m wilcom_pipeline.api --host 127.0.0.1 --port 8000
#   -> http://127.0.0.1:8000

# UI development, with hot reload and /api proxied to port 8000
cd frontend; npm run dev            #   -> http://127.0.0.1:5173
```

### 5. Command line

The pipeline also runs without the web UI:

```powershell
.venv\Scripts\python.exe -m wilcom_pipeline.cli check --source path\to\video.mp4   # probe only, no API calls
.venv\Scripts\python.exe -m wilcom_pipeline.cli plan  --source path\to\video.mp4   # cost estimate, no API calls
.venv\Scripts\python.exe -m wilcom_pipeline.cli run   --source path\to\video.mp4   # all three stages
```

`--source` takes a file **or a folder**. `--stages grounding,dataset` reuses a cached stage 1 instead of paying for it again, and `--work DIR` (default `wilcom_work`) is where every artifact lands — gitignored, and regenerable.

### 6. Tests

Offline: no API key and no network needed.

```powershell
.venv\Scripts\python.exe -m unittest discover wilcom_pipeline/tests/
```

## �📦 WildGUI Dataset

WildGUI is the largest open-source GUI pre-training dataset to date, with comprehensive coverage across **web, mobile, and desktop** platforms. It is released across two HuggingFace repositories:

<p align="center">
<img src="static/images/dataset_comparison.png" width="820"/>
</p>

| Repository | Annotations | Screenshots |
|---|---|---|
| [`xwm/WildGUI`](https://huggingface.co/datasets/xwm/WildGUI) | **All** parts (`part1`–`part19`, JSONL) | `part1`–`part15` |
| [`joker-112/WildGUI_Screenshots`](https://huggingface.co/datasets/joker-112/WildGUI_Screenshots) | — | `part16`–`part19` |

The annotations for every part live in `xwm/WildGUI`; the screenshot frames are split between the two repos. Each annotation record is a single GUI task with an ordered action trajectory, and each action points at one screenshot frame stored as `{video_id}/screenshot_{MM_SS}.jpg` inside the packed tar shards. See each dataset card for the full field schema and the annotation↔screenshot linking convention.

```python
from datasets import load_dataset

# Annotations (all parts)
dataset = load_dataset("xwm/WildGUI", data_files="wildgui_part*.jsonl", split="train")
```

```bash
# Screenshots for one part (e.g. part1), then unpack the tar shards
hf download xwm/WildGUI --repo-type dataset --include "screenshots/part1/*" --local-dir ./wildgui
for t in ./wildgui/screenshots/part1/*.tar; do tar -xf "$t" -C ./wildgui_frames; done
```

## 📊 Results

Pre-training on WildGUI yields consistent gains on the GUI grounding and action benchmarks.

<p align="center">
<img src="static/images/main_results.png" width="820"/>
</p>

<p align="center">
<img src="static/images/table3_androidcontrol.png" width="820"/>
</p>

## 📖 Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{xiong2026video2gui,
  title     = {Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining},
  author    = {Xiong, Weimin and Gu, Shuhao and Ye, Bowen and Yue, Zihao and Li, Lei and Song, Feifan and Li, Sujian and Tian, Hao},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.14747}
}
```

## 📧 Contact

For questions, please contact: wmxiong@pku.edu.cn, lisujian@pku.edu.cn

## License

This repository is released under the [Apache License 2.0](./LICENSE).
