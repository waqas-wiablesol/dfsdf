"""HTTP API for the pipeline.

Serves a JSON API and the built React app. Deliberately polling rather than
server-sent events: the browser aborted Gradio's SSE stream on every reconnect
(`ERR_ABORTED`), which is what made the previous UI appear to do nothing after a
button press. A short poll is a few hundred bytes against localhost and cannot
fail that way.

Every endpoint is a thin wrapper over an existing function -- `jobs` for the run,
`records` for the artefacts, `prompts` for the prompt text. No pipeline logic
lives here, so the API cannot drift from the command line.

Launch:  python -m wilcom_pipeline.api
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, details, jobs, prompts, records, stage_dataset

ROOT = Path(__file__).resolve().parent
# The Vite build output. If it is missing the API still works; the root route then
# explains how to build it rather than returning a bare 404.
WEB_DIR = ROOT / "webui"

# When this process started. Compared against the source files' timestamps so the
# server can tell that it is running older code than what is on disk -- see
# `code_status`. This has been the single most confusing failure mode in this
# project: a new endpoint or a gating rule appears not to work, and the cause is a
# process that predates the change.
_STARTED_AT = time.time()


def _newest_mtime(directory: Path, pattern: str) -> float:
    newest = 0.0
    for p in directory.glob(pattern):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def web_asset() -> str:
    """Filename of the built JS bundle the server would serve right now.

    The UI compares this against the bundle it was actually loaded from. A mismatch
    means the page is stale, which no amount of reloading the API can fix.
    """
    if not (WEB_DIR / "assets").is_dir():
        return ""
    files = sorted((WEB_DIR / "assets").glob("index-*.js"),
                   key=lambda p: p.stat().st_mtime)
    return files[-1].name if files else ""


def code_status() -> dict:
    """Whether this process is older than the Python code it is running.

    A module edited after the process started cannot have been imported by it, so
    any endpoint or rule added in that edit is simply absent -- and it fails by
    404ing or by silently omitting a field, which looks like a bug in the change
    rather than a stale process. Reporting it removes the guesswork.

    Only SOURCE is compared, not the built web assets. Static files are read from
    disk on every request, so a bundle written after this process started is served
    correctly and is not a staleness signal. An earlier version compared it anyway
    and reported `web_newer: true` right after a rebuild, which is noise that would
    train us to ignore the warning that matters.
    """
    newest_source = _newest_mtime(ROOT, "*.py")
    # Half a second of slack: a file written in the same second the process started
    # is not evidence of staleness, and a false alarm here would train us to ignore
    # the warning.
    return {
        "started": _STARTED_AT,
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "newest_source": newest_source,
        "stale_code": newest_source > _STARTED_AT + 0.5,
        "web_asset": web_asset(),
    }

app = FastAPI(title="WILCOM Video2GUI", version="2")

# The dev server runs on another port, so the browser makes cross-origin requests
# until the app is built and served from here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StepRequest(BaseModel):
    work: str = "wilcom_work"
    url: str = ""
    source: str = ""
    model: str = ""
    model_traj: str = ""
    model_ground: str = ""
    max_clips: int = 0
    limit: int = 0
    # Only step 2 uses this. Validated in `jobs.trajectory` against
    # `config.THINKING_LEVELS`, so an unknown value is refused rather than
    # forwarded to the provider.
    thinking: str = ""


# ============================================================================
# Run state
# ============================================================================


@app.get("/api/state")
def state(work: str = Query("wilcom_work")) -> JSONResponse:
    """Everything the run view needs. Polled while a step is in flight.

    One request rather than several, because a UI that fetches progress, log and
    step rail separately can render a frame where they disagree.
    """
    try:
        return JSONResponse(jobs.RUNNER.snapshot(work))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        raise HTTPException(500, "%s: %s" % (type(exc).__name__, exc))


@app.post("/api/run/{key}")
def run(key: str, body: StepRequest) -> JSONResponse:
    """Start one step. It runs to completion and stops; nothing chains.

    A locked step is refused here and not merely disabled in the UI, so the gate
    holds for any caller. The refusal carries its own `reason` so the interface can
    say which step to run instead of showing a generic failure.
    """
    if key not in jobs.STEP_KEYS and key not in ("playlist", "autoflow"):
        raise HTTPException(404, "unknown step %r" % key)
    if jobs._running():
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A step is already running."})

    reason = jobs.locked_reason(body.work, key)
    if reason:
        return JSONResponse({"started": False, "reason": "locked",
                             "message": reason})

    message = jobs.start(key, body.work, body.model_dump())
    if message == "busy":
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A step is already running."})
    if message.startswith("locked:"):
        return JSONResponse({"started": False, "reason": "locked",
                             "message": message.split(":", 1)[1].strip()})
    return JSONResponse({"started": True, "step": key, "message": message})


@app.post("/api/reset")
def reset(work: str = Query("wilcom_work")) -> JSONResponse:
    """Forget the finished run and reset step session progression for this work tree."""
    jobs.RUNNER.reset()
    jobs.reset_step_session(work)
    return JSONResponse({"ok": True})


# ============================================================================
# Playlist Queue
# ============================================================================


class EnqueueRequest(BaseModel):
    url: str
    work: str = "wilcom_work"


class InspectUrlRequest(BaseModel):
    url: str


@app.post("/api/url/inspect")
def inspect_url(body: InspectUrlRequest) -> JSONResponse:
    from . import queue as queue_mgr

    raw_url = body.url.strip()
    if not raw_url:
        raise HTTPException(400, "URL cannot be empty")
    try:
        data = queue_mgr.extract_playlist_items(raw_url)
        if not data:
            raise HTTPException(404, "Could not extract metadata from the provided URL.")
        return JSONResponse({"ok": True, "data": data})
    except Exception as exc:
        raise HTTPException(400, "Error inspecting URL: %s" % exc)


@app.get("/api/url/inspect")
def inspect_url_get(url: str = Query(...)) -> JSONResponse:
    from . import queue as queue_mgr

    raw_url = url.strip()
    if not raw_url:
        raise HTTPException(400, "URL cannot be empty")
    try:
        data = queue_mgr.extract_playlist_items(raw_url)
        if not data:
            raise HTTPException(404, "Could not extract metadata from the provided URL.")
        return JSONResponse({"ok": True, "data": data})
    except Exception as exc:
        raise HTTPException(400, "Error inspecting URL: %s" % exc)


class QueueRunRequest(BaseModel):
    work: str = "wilcom_work"
    model_traj: str = ""
    model_ground: str = ""
    thinking: str = ""


class QueueActionRequest(BaseModel):
    work: str = "wilcom_work"
    mode: str = "all"


@app.get("/api/queue")
def get_queue(work: str = Query("wilcom_work")) -> JSONResponse:
    from . import queue as queue_mgr

    paths = jobs.paths_for(work)
    db_path = paths.root / "queue.db"
    items = queue_mgr.list_queue(db_path)
    stats = queue_mgr.queue_stats(db_path)
    return JSONResponse({
        "items": items,
        "total": len(items),
        "stats": stats,
    })


@app.post("/api/queue/enqueue")
def enqueue(body: EnqueueRequest) -> JSONResponse:
    from . import queue as queue_mgr

    paths = jobs.paths_for(body.work)
    db_path = paths.root / "queue.db"
    try:
        res = queue_mgr.enqueue_playlist(db_path, body.url.strip())
        return JSONResponse({"ok": True, "result": res})
    except Exception as exc:
        raise HTTPException(400, "Failed to enqueue playlist: %s" % exc)


@app.post("/api/queue/run")
def run_queue(body: QueueRunRequest) -> JSONResponse:
    if jobs._running():
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A pipeline step is already running."})

    message = jobs.start("playlist", body.work, body.model_dump())
    if message == "busy":
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A step is already running."})
    return JSONResponse({"started": True, "step": "playlist", "message": message})


class QueueAutoflowRequest(BaseModel):
    url: str = ""
    work: str = "wilcom_work"
    model_traj: str = ""
    model_ground: str = ""
    thinking: str = ""


@app.post("/api/queue/autoflow")
def autoflow_queue(body: QueueAutoflowRequest) -> JSONResponse:
    """Atomic playlist autoflow: enqueues playlist videos if URL provided, then launches the background queue."""
    from . import queue as queue_mgr

    if jobs._running():
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A pipeline step is already running."})

    paths = jobs.paths_for(body.work)
    db_path = paths.root / "queue.db"

    enqueued_res = None
    if body.url.strip():
        try:
            enqueued_res = queue_mgr.enqueue_playlist(db_path, body.url.strip())
        except Exception as exc:
            raise HTTPException(400, "Failed to enqueue playlist: %s" % exc)

    queued_items = [i for i in queue_mgr.list_queue(db_path) if i["status"] in ("pending", "failed")]
    if not queued_items:
        return JSONResponse({"started": False, "reason": "empty",
                             "message": "Queue has no pending or failed items to process."})

    message = jobs.start("playlist", body.work, body.model_dump())
    if message == "busy":
        return JSONResponse({"started": False, "reason": "busy",
                             "message": "A step is already running."})
    return JSONResponse({
        "started": True,
        "step": "playlist",
        "message": message,
        "enqueued": enqueued_res,
    })


@app.post("/api/queue/retry")
def retry_failed_queue(body: QueueActionRequest) -> JSONResponse:
    from . import queue as queue_mgr

    paths = jobs.paths_for(body.work)
    db_path = paths.root / "queue.db"
    cnt = queue_mgr.reset_failed(db_path)
    return JSONResponse({"ok": True, "reset_count": cnt})


@app.post("/api/queue/clear")
def clear_queue_items(body: QueueActionRequest) -> JSONResponse:
    from . import queue as queue_mgr

    paths = jobs.paths_for(body.work)
    db_path = paths.root / "queue.db"
    cnt = queue_mgr.clear_queue(db_path, mode=body.mode)
    return JSONResponse({"ok": True, "cleared_count": cnt})



# ============================================================================
# Artefacts
# ============================================================================


@app.get("/api/rows")
def rows(work: str = Query("wilcom_work"), limit: int = Query(0)) -> JSONResponse:
    """The grounded rows from the single unified parquet dataset, preserving all 18 schema columns.

    Each row carries "source_video" for UI grouping and "trajectory" for trajectory linkage.
    """
    import numpy as np
    import pandas as pd

    paths = jobs.paths_for(work)
    unified_p = paths.out / "unified.parquet"

    def clean_val(v):
        if isinstance(v, np.ndarray):
            return [clean_val(x) for x in v.tolist()]
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, dict):
            return {k: clean_val(val) for k, val in v.items()}
        if isinstance(v, list):
            return [clean_val(x) for x in v]
        return v

    if unified_p.is_file():
        df_u = pd.read_parquet(unified_p)
        raw_rows = df_u.to_dict("records")
        for r in raw_rows:
            if isinstance(r.get("image"), dict) and r["image"].get("path"):
                r["image_path"] = r["image"]["path"]
        columns = list(df_u.columns)
        out = [{k: clean_val(v) for k, v in r.items()} for r in raw_rows]
        splits = {"unified": len(out)}
        problems = []
    else:
        traj = records.load_dir(paths.traj)
        ground = records.load_dir(paths.ground)
        built, _stats = stage_dataset.build_rows(traj, ground, paths)
        problems = stage_dataset.verify(built, {"unified": built}) if built else []

        columns = sorted(list(built[0].keys())) if built else []
        out = [{k: clean_val(v) for k, v in r.items()} for r in built]
        splits = {"unified": len(built)} if built else {}

    filtered = out
    if limit:
        filtered = filtered[:limit]

    return JSONResponse({
        "count": len(filtered),
        "total_count": len(out),
        "columns": columns,
        "rows": filtered,
        "splits": splits,
        "problems": problems,
    })


# The work dir is a QUERY parameter, never a path parameter. A work tree is often an
# absolute path, and `/api/details/C:/work/wilcom_work` does not match a single
# `{work}` path segment -- it 404s, and the UI then shows an empty panel rather than
# an error worth noticing. Every endpoint that takes a work dir takes it as `?work=`.


@app.get("/api/schema")
def dataset_schema(work: str = Query("wilcom_work")) -> JSONResponse:
    """The exported dataset's real columns, with per-split row counts.

    Read from `hf_export.features()`, not written out here, so it cannot describe
    something other than what is exported. The row counts come from the parquet on
    disk, so the panel shows what was actually written rather than what the schema
    would allow.
    """
    from . import hf_export

    paths = jobs.paths_for(work)
    splits: dict = {}
    rows_for_presence: list = []
    for path in sorted(paths.out.glob("*.parquet")):
        try:
            import pandas as pd
            frame = pd.read_parquet(path)
        except Exception:
            continue
        splits[path.stem] = len(frame)
        if not rows_for_presence and len(frame):
            rows_for_presence = frame.to_dict("records")

    payload = hf_export.schema(rows_for_presence or None)
    payload["splits"] = splits
    if "unified" in splits:
        payload["total_rows"] = splits["unified"]
    else:
        payload["total_rows"] = sum(splits.values())
    payload["problems"] = hf_export.verify_against_parent()
    return JSONResponse(payload)


@app.get("/api/options")
def options() -> JSONResponse:
    """Per-step choices the UI offers, so the list lives in one place.

    The thinking levels and their measured notes come from `config` rather than
    being written into the interface -- a second copy would drift, and the numbers
    they quote are measurements, not decoration.
    """
    return JSONResponse({
        "thinking_levels": [
            {"value": level, "note": config.THINKING_NOTES.get(level, "")}
            for level in config.THINKING_LEVELS
        ],
        "default_thinking": config.THINKING_LEVEL,
        "default_model": {
            "trajectory": config.model_for("trajectory"),
            "grounding": config.model_for("grounding"),
            "meta": config.model_for("meta"),
        },
    })


@app.get("/api/details")
def details_report(work: str = Query("wilcom_work")) -> JSONResponse:
    """Per-video detail: slicing, actions, tokens and timings.

    Derived entirely from what is on disk, so opening the view costs nothing.
    Token counts for runs that predate usage being recorded are recovered from the
    client's cache; anything still unknown is reported as null and named in
    `totals.unknown_sources` rather than folded in as zero.
    """
    try:
        return JSONResponse(details.build(work))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        raise HTTPException(500, "%s: %s" % (type(exc).__name__, exc))


@app.get("/api/review")
def review_report(work: str = Query("wilcom_work")) -> JSONResponse:
    """Step 5's frame-coverage report, so the UI can show it without re-running."""
    path = jobs.paths_for(work).logs / "review.json"
    if not path.is_file():
        return JSONResponse({"report": None})
    return JSONResponse({"report": records.read_json(path)})


@app.get("/api/frame")
def frame(path: str = Query(...)) -> FileResponse:
    """Serve one extracted frame, for the coordinate check.

    Restricted to the configured work tree: a query parameter that reads any file
    on disk is a directory-traversal hole, and this is the only place the API
    touches a path from the client.
    """
    target = Path(path).resolve()
    allowed = [Path.cwd().resolve(), Path("wilcom_work").resolve()]
    if not any(target.is_relative_to(a) for a in allowed):
        raise HTTPException(403, "path outside the work tree")
    if not target.is_file():
        raise HTTPException(404, "no such frame")
    return FileResponse(target)


@app.get("/api/screen")
def screening(work: str = Query("wilcom_work")) -> JSONResponse:
    """The newest screening record, so the verdict survives a reload."""
    paths = jobs.paths_for(work)
    found = sorted(paths.screen.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        return JSONResponse({"record": None})
    return JSONResponse({"record": records.read_json(found[0])})


# ============================================================================
# Prompts
# ============================================================================


@app.get("/api/prompts")
def prompt_list() -> JSONResponse:
    """Every prompt, with its provenance.

    `prompts.PROMPTS` is a LIST of entries, not a mapping keyed by name -- each
    entry carries its own `name`. The provenance key is `source`, not
    `source_page`.
    """
    out = []
    for entry in prompts.PROMPTS:
        text = entry.get("text") or ""
        out.append({
            "name": entry.get("name"),
            "title": entry.get("title"),
            "kind": entry.get("kind"),
            "source": entry.get("source"),
            "model_const": entry.get("model_const"),
            "used_by": entry.get("used_by"),
            "notes": entry.get("notes"),
            "text": text,
            "chars": len(text),
        })
    return JSONResponse({"prompts": out, "count": len(out)})


@app.get("/api/plan")
def plan(seconds: float = Query(600.0),
         clip_seconds: int = Query(0)) -> JSONResponse:
    """How a video of this length would be cut, and what it would cost.

    Costs a synthetic `VideoInfo` rather than a real file, so this is answerable
    before anything is downloaded. `cost.estimate` is a module-level function
    taking probed videos, which is why one is constructed here.
    """
    from . import cost, video

    clip_seconds = clip_seconds or config.CLIP_SECONDS
    # Dimensions do not affect the estimate (cost scales with duration), so a
    # nominal 1080p60 stands in for whatever the real video turns out to be.
    info = video.VideoInfo(path=Path("planned.mp4"), duration=float(seconds),
                           width=1920, height=1080, fps=60.0)

    def payload(level: str) -> dict:
        est = cost.estimate([info], clip_seconds=clip_seconds,
                            thinking_level=level)
        return {
            "clips": est.clips,
            "actions": est.actions,
            "traj_usd": round(est.traj_usd, 5),
            "ground_usd": round(est.ground_usd, 5),
            "meta_usd": round(est.meta_usd, 5),
            "total_usd": round(est.total_usd, 5),
        }

    return JSONResponse({
        "duration_s": seconds,
        "clip_seconds": clip_seconds,
        "clips": max(1, int(-(-seconds // clip_seconds))),
        "default": payload("default"),
        "low": payload("low"),
    })


@app.get("/api/health")
def health() -> JSONResponse:
    """Liveness, where the API key came from, and whether this process is stale.

    Only the source and the key's last 4 characters are reported, never the key:
    enough to confirm a run will authenticate, and enough to tell two keys apart in
    a screenshot, without putting a usable credential anywhere it could be read.
    """
    source, tail = config.key_source()
    payload = {
        "ok": True,
        "web_built": WEB_DIR.is_dir(),
        "key_source": source or None,
        "key_tail": tail or None,
        "steps": jobs.STEP_KEYS,
    }
    payload.update(code_status())
    return JSONResponse(payload)


# ============================================================================
# The React app
# ============================================================================


def _mount_web() -> None:
    """Serve the build, with index.html as the fallback so a refresh works."""
    if not WEB_DIR.is_dir():
        @app.get("/")
        def missing() -> PlainTextResponse:
            return PlainTextResponse(
                "The web UI has not been built yet.\n\n"
                "  cd frontend\n  npm install\n  npm run build\n\n"
                "or run the dev server (npm run dev) and open "
                "http://localhost:5173\n", status_code=503)
        return

    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        # Any non-API path is a client route, so hand back the shell and let React
        # decide what to render.
        if path.startswith("api/"):
            raise HTTPException(404, "no such endpoint")
        candidate = WEB_DIR / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIR / "index.html")


_mount_web()


def main() -> None:
    parser = argparse.ArgumentParser(description="WILCOM Video2GUI API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Say where the key came from before serving anything. A run that fails on a
    # missing key minutes later, after a download, is a much worse way to find out.
    source, tail = config.key_source()
    if source:
        print("API key   : %s  (.....%s)" % (source, tail))
    else:
        print("API key   : NOT FOUND -- steps 0, 2, 3 and 4 will fail.\n"
              "            Set GEMINI_API_KEY, or put it in "
              "wilcom_pipeline/local_secrets.py")

    status = code_status()
    if status["stale_code"]:
        # Printed, not merely exposed: this is the moment it matters, and the whole
        # point is that it is otherwise invisible.
        age = time.strftime("%H:%M:%S", time.localtime(status["newest_source"]))
        print("WARNING   : this process is older than the code on disk "
              "(a module was edited at %s). Restart it, or new endpoints and "
              "rules will be missing." % age)

    print("Web UI    : http://%s:%d" % (args.host, args.port))
    print("Work dir  : %s" % Path("wilcom_work").resolve())
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
