"""SQLite queue for handling YouTube playlists and sequential video processing.

Allows users to enqueue a full YouTube playlist, track per-video progress across
download, trajectory extraction, and grounding, and recover gracefully from errors
or rate limits without re-running completed videos.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from . import log


def get_db(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite and ensure the queue schema is initialized."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_queue (
            video_id         TEXT PRIMARY KEY,
            playlist_id      TEXT,
            playlist_title   TEXT,
            index_in_list    INTEGER,
            title            TEXT,
            url              TEXT NOT NULL,
            duration_s       REAL,
            status           TEXT DEFAULT 'pending',
            error_msg        TEXT,
            local_video_path TEXT,
            actions_count    INTEGER DEFAULT 0,
            grounded_count   INTEGER DEFAULT 0,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def extract_playlist_items(url: str) -> dict[str, Any] | None:
    """Extract metadata for all videos in a playlist (or a single video) without downloading."""
    import re
    import yt_dlp

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
    }

    # If the URL contains a list= parameter, prefer the playlist URL so yt-dlp extracts
    # the entire playlist rather than just the single video being watched within it.
    candidate_urls = []
    list_match = re.search(r'[?&]list=([a-zA-Z0-9_-]+)', url)
    if list_match:
        list_id = list_match.group(1)
        candidate_urls.append(f"https://www.youtube.com/playlist?list={list_id}")
    candidate_urls.append(url)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = None
        for candidate in candidate_urls:
            try:
                info = ydl.extract_info(candidate, download=False)
                if info and info.get("entries"):
                    break
            except Exception:
                continue

        if not info:
            # Fallback direct attempt
            try:
                info = ydl.extract_info(url, download=False)
            except Exception:
                return None

        if not info:
            return None

        entries = info.get("entries")
        if entries is not None:
            # Playlist detected
            items = []
            for idx, e in enumerate(entries):
                if not e:
                    continue
                v_id = e.get("id")
                if not v_id:
                    continue
                v_url = e.get("url") or f"https://www.youtube.com/watch?v={v_id}"
                items.append({
                    "video_id": str(v_id),
                    "title": e.get("title") or f"Video #{idx + 1}",
                    "url": v_url,
                    "duration_s": float(e.get("duration") or 0.0) if e.get("duration") else None,
                    "index_in_list": idx + 1,
                })
            return {
                "is_playlist": True,
                "playlist_id": info.get("id"),
                "playlist_title": info.get("title") or "YouTube Playlist",
                "channel": info.get("channel") or info.get("uploader"),
                "total_videos": len(items),
                "items": items,
            }
        else:
            # Single video URL passed
            v_id = str(info.get("id") or "video")
            v_url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={v_id}"
            return {
                "is_playlist": False,
                "playlist_id": None,
                "playlist_title": None,
                "channel": info.get("channel") or info.get("uploader"),
                "total_videos": 1,
                "items": [{
                    "video_id": v_id,
                    "title": info.get("title") or "Single Video",
                    "url": v_url,
                    "duration_s": float(info.get("duration") or 0.0) if info.get("duration") else None,
                    "index_in_list": 1,
                }],
            }



def enqueue_playlist(db_path: Path, url: str) -> dict[str, Any]:
    """Extract playlist items and insert into SQLite queue."""
    parsed = extract_playlist_items(url)
    if not parsed or not parsed["items"]:
        raise ValueError("Could not extract any video entries from the provided URL.")

    conn = get_db(db_path)
    added = 0
    total = len(parsed["items"])
    playlist_id = parsed["playlist_id"]
    playlist_title = parsed["playlist_title"]

    with conn:
        for item in parsed["items"]:
            cursor = conn.execute("""
                INSERT INTO playlist_queue (
                    video_id, playlist_id, playlist_title, index_in_list,
                    title, url, duration_s, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(video_id) DO UPDATE SET
                    playlist_id = excluded.playlist_id,
                    playlist_title = excluded.playlist_title,
                    index_in_list = excluded.index_in_list,
                    title = excluded.title,
                    url = excluded.url,
                    duration_s = excluded.duration_s
                WHERE status IN ('pending', 'failed')
            """, (
                item["video_id"],
                playlist_id,
                playlist_title,
                item["index_in_list"],
                item["title"],
                item["url"],
                item["duration_s"],
            ))
            if cursor.rowcount > 0:
                added += 1

    conn.close()
    return {
        "playlist_id": playlist_id,
        "playlist_title": playlist_title,
        "total_extracted": total,
        "enqueued_or_updated": added,
        "items": list_queue(db_path),
    }


def list_queue(db_path: Path) -> list[dict[str, Any]]:
    """Return all queue items ordered by index."""
    conn = get_db(db_path)
    cursor = conn.execute("""
        SELECT video_id, playlist_id, playlist_title, index_in_list,
               title, url, duration_s, status, error_msg, local_video_path,
               actions_count, grounded_count, created_at, updated_at
        FROM playlist_queue
        ORDER BY index_in_list ASC, created_at ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_item(db_path: Path, video_id: str, **fields) -> None:
    """Update status, error, or counts for a specific video."""
    if not fields:
        return
    conn = get_db(db_path)
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    set_clause += ", updated_at = CURRENT_TIMESTAMP"
    values = list(fields.values()) + [video_id]

    with conn:
        conn.execute(f"UPDATE playlist_queue SET {set_clause} WHERE video_id = ?", values)
    conn.close()


def get_next_pending(db_path: Path) -> dict[str, Any] | None:
    """Get the next item needing processing."""
    conn = get_db(db_path)
    cursor = conn.execute("""
        SELECT video_id, playlist_id, playlist_title, index_in_list,
               title, url, duration_s, status, error_msg, local_video_path,
               actions_count, grounded_count, created_at, updated_at
        FROM playlist_queue
        WHERE status = 'pending'
        ORDER BY index_in_list ASC, created_at ASC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def reset_failed(db_path: Path) -> int:
    """Reset all failed queue items to pending."""
    conn = get_db(db_path)
    with conn:
        cursor = conn.execute("""
            UPDATE playlist_queue
            SET status = 'pending', error_msg = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE status = 'failed'
        """)
        count = cursor.rowcount
    conn.close()
    return count


def clear_queue(db_path: Path, mode: str = "all") -> int:
    """Clear all or only completed items from the queue."""
    conn = get_db(db_path)
    with conn:
        if mode == "finished":
            cursor = conn.execute("DELETE FROM playlist_queue WHERE status IN ('completed', 'failed')")
        else:
            cursor = conn.execute("DELETE FROM playlist_queue")
        count = cursor.rowcount
    conn.close()
    return count


def queue_stats(db_path: Path) -> dict[str, int]:
    """Get summary counts of queue statuses."""
    conn = get_db(db_path)
    cursor = conn.execute("SELECT status, COUNT(*) as cnt FROM playlist_queue GROUP BY status")
    stats = {r["status"]: r["cnt"] for r in cursor.fetchall()}
    conn.close()
    return stats
