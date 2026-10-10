# Video2GUI Pipeline & Web Studio — Setup and Running Guide

This guide covers how to configure, run, and use the **Video2GUI Pipeline** and its interactive web interface for turning screen-recording video tutorials into multimodal GUI grounding datasets.

---

## ⚡ Quickstart (TL;DR)

1. **Configure Gemini API Key** (pick any one):
   ```bash
   # Option A: Create a .env file in the repository root
   copy .env.example .env
   # Edit .env and paste your API key:
   # GEMINI_API_KEY=your_gemini_api_key_here
   ```
   *Alternatively, set it in your shell:*
   - **PowerShell:** `$env:GEMINI_API_KEY = "your_gemini_api_key_here"`
   - **Bash/Zsh:** `export GEMINI_API_KEY="your_gemini_api_key_here"`

2. **Start the Backend API Server**:
   ```bash
   # From the project root with the virtual environment activated:
   .venv\Scripts\python.exe -m wilcom_pipeline.api --host 127.0.0.1 --port 8000
   ```

3. **Open the Web Studio**:
   - Navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
   - The FastAPI backend serves both the REST endpoints and the built React frontend.

---

## 🔑 1. Gemini API Key Configuration

> [!IMPORTANT]
> **No API keys are hardcoded in the codebase.**
> Keys are loaded dynamically at runtime following an order of precedence.

You can get an API key from [Google AI Studio](https://aistudio.google.com/apikey). The pipeline supports three clean ways to supply the key:

### Method A: `.env` File (Recommended)
Copy the template `.env.example` in the project root to `.env`:
```ini
# .env (gitignored)
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
The pipeline automatically parses `.env` on startup.

### Method B: Environment Variable
Set the variable in your current terminal session:
- **Windows PowerShell:**
  ```powershell
  $env:GEMINI_API_KEY = "your_actual_gemini_api_key_here"
  ```
- **Windows Command Prompt (cmd):**
  ```cmd
  set GEMINI_API_KEY=your_actual_gemini_api_key_here
  ```
- **macOS / Linux (bash/zsh):**
  ```bash
  export GEMINI_API_KEY="your_actual_gemini_api_key_here"
  ```
*(Note: `GOOGLE_API_KEY` is also accepted as an alias).*

### Method C: `wilcom_pipeline/local_secrets.py` (Gitignored)
If you prefer a Python configuration file:
1. Copy the template:
   ```powershell
   copy wilcom_pipeline\local_secrets.example.py wilcom_pipeline\local_secrets.py
   ```
2. Open `wilcom_pipeline/local_secrets.py` and set:
   ```python
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
*(This file is explicitly listed in `.gitignore` and is never committed).*

---

## 🛠️ 2. Prerequisites & Environment Setup

### 2.1 Python Environment (Backend)
- Python **3.10+** (Python 3.11 or 3.12 recommended).
- Virtual environment setup:
  ```powershell
  # 1. Create a virtual environment (if not already created)
  python -m venv .venv

  # 2. Activate virtual environment
  # Windows PowerShell:
  .venv\Scripts\Activate.ps1
  # Linux/macOS:
  source .venv/bin/activate

  # 3. Install Python dependencies
  pip install -r wilcom_pipeline/requirements.txt
  ```

### 2.2 Node.js & React (Frontend)
- Node.js **v18+** and **npm**.
- The pre-built frontend is already embedded inside `wilcom_pipeline/webui/`.
- If you make frontend code changes or need to build from source:
  ```powershell
  cd frontend
  npm install
  npm run build
  cd ..
  ```
  `npm run build` bundles the React assets directly into `wilcom_pipeline/webui/` so FastAPI can serve them.

---

## 🚀 3. Running the Pipeline

### Mode 1: Production / All-In-One (Recommended)
FastAPI serves both the API endpoints (`/api/...`) and the production React Web UI:

```powershell
.venv\Scripts\python.exe -m wilcom_pipeline.api --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### Mode 2: Frontend Live Development
If you are modifying UI components and want Hot Module Replacement (HMR):
1. **Start the FastAPI backend** in one terminal:
   ```powershell
   .venv\Scripts\python.exe -m wilcom_pipeline.api --host 127.0.0.1 --port 8000
   ```
2. **Start the Vite dev server** in another terminal:
   ```powershell
   cd frontend
   npm run dev
   ```
3. Open **[http://localhost:5173](http://localhost:5173)**. Vite proxies `/api` calls directly to port 8000.

---

## 💻 4. Using the Web Studio

The web application provides tabs for every stage of dataset curation:

### A. Single Video Auto-Flow
1. Open the **Run** tab.
2. Enter a YouTube video URL (e.g. `https://youtube.com/watch?v=...`) or specify local videos in `wilcom_work/source`.
3. Choose your models (e.g. `gemini-3.1-pro-preview` or `gemini-2.5-flash`).
4. Click **Auto-Flow (Complete Pipeline)**.
5. The pipeline automatically advances through all 5 phases:
   - **Phase 1 [Download/Ingest]:** Downloads video and extracts metadata.
   - **Phase 2 [Trajectory Extraction]:** Identifies high-level tasks and low-level action steps.
   - **Phase 3 [Grounding]:** Locates exact screen coordinates for clickable UI elements.
   - **Phase 4 [Dataset Assembly]:** Converts records into unified HuggingFace-compatible Parquet & JSON datasets.
   - **Phase 5 [Review & Verification]:** Re-extracts missing frames and validates screenshot coverage.

### B. YouTube Playlist Batch Queue
1. In the **Run** tab, paste a YouTube Playlist URL (e.g., `https://youtube.com/playlist?list=...`).
2. The URL detection card automatically detects the playlist, queries its title, item count, and item previews.
3. Click **Enqueue Playlist & Auto-Flow**.
4. The background queue downloads, processes, trajectories, and grounds each tutorial sequentially.
5. Once complete, it merges all completed queue items into a single unified Parquet dataset.

### C. Queue Management Tab
- Monitor current queue progress, active video step, item status (`pending`, `download`, `trajectory`, `grounding`, `completed`, `failed`), and error logs.
- Retry failed items or restart paused jobs with single-click actions.

### D. Results & Inspection Tab
- Displays grounded GUI actions in a modern 2-column layout.
- View extracted screenshot frames with exact coordinate grounding dots.
- Inspect and expand full action instructions, timestamps, bounding boxes, and model confidence scores.

---

## ⌨️ 5. Command-Line Interface (CLI)

The pipeline can also be run entirely from the command line without the web UI:

```powershell
# Check configuration and API key status:
.venv\Scripts\python.exe -m wilcom_pipeline.cli check

# Run end-to-end pipeline on a video:
.venv\Scripts\python.exe -m wilcom_pipeline.cli run \
    --url "https://youtube.com/watch?v=VIDEO_ID" \
    --work wilcom_work \
    --model-traj gemini-3.1-pro-preview \
    --model-ground gemini-3.1-pro-preview
```

---

## 🧪 6. Running Tests

To verify pipeline integrity, queue endpoints, and stage modules without making external API calls:

```powershell
.venv\Scripts\python.exe -m unittest discover wilcom_pipeline/tests/
```
All tests use mock fixtures and temporary directories to ensure offline reliability.

---

## 📁 7. Output Directory Structure

Pipeline outputs are written under the configured work directory (default: `wilcom_work/`):

```
wilcom_work/
├── source/                  # Downloaded video files (.mp4)
├── videos/                  # Ingested video directory
├── frames/                  # Extracted screenshot frames ({stem}/{frame}.png)
├── stage_trajectory/        # JSON task & action trajectory records
├── stage_grounding/         # JSON coordinate & bounding box grounding records
├── dataset/                 # Unified final Parquet files (*.parquet)
├── logs/                    # Execution logs, review.json, cost reports
└── queue.db                 # SQLite batch queue database for playlist ingestion
```

---

## ❓ 8. Troubleshooting

- **Error: `No API key found`**
  Ensure you have set `GEMINI_API_KEY` in `.env`, your shell environment, or `wilcom_pipeline/local_secrets.py`. Check status via `.venv\Scripts\python.exe -m wilcom_pipeline.cli check`.
- **Port 8000 already in use**
  Specify a different port:
  ```powershell
  .venv\Scripts\python.exe -m wilcom_pipeline.api --host 127.0.0.1 --port 8080
  ```
- **Missing ffmpeg**
  `imageio-ffmpeg` is included in `requirements.txt` as a static binary fallback. You can also install system-wide ffmpeg (`winget install Gyan.FFmpeg` on Windows or `sudo apt install ffmpeg` on Ubuntu) and ensure it is on your `PATH`.
