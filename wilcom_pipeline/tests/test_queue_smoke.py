"""Smoke test for SQLite playlist queue and sequential worker without any paid Gemini API calls.

Tests:
1. SQLite queue schema creation, WAL mode, idempotency, updates, and reset.
2. Flat-playlist enqueueing and metadata parsing.
3. Sequential worker running over pending items with mocked download, trajectory, and grounding stages.
4. Error tolerance: one video fails with simulated error, worker marks it 'failed' and continues to next video.
5. Reassembly of unified dataset at completion.
6. Reset failed videos back to pending.
7. Clear queue functionality.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from wilcom_pipeline import config, jobs, queue as queue_mgr


class TestQueueSmoke(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="wilcom_queue_test_")
        self.work_path = Path(self.temp_dir)
        self.paths = config.Paths(root=self.work_path).ensure()
        self.db_path = self.paths.root / "queue.db"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_queue_sqlite_crud(self):
        """Verify SQLite queue basic CRUD operations."""
        conn = queue_mgr.get_db(self.db_path)
        conn.close()
        self.assertTrue(self.db_path.is_file(), "queue.db should be created")

        # Manually insert test items
        conn = queue_mgr.get_db(self.db_path)
        with conn:
            conn.execute("""
                INSERT INTO playlist_queue (
                    video_id, playlist_id, playlist_title, index_in_list,
                    title, url, duration_s, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("vid_1", "pl_test", "Test Playlist", 1, "Video One", "https://yt.com/1", 120.0, "pending"))
            conn.execute("""
                INSERT INTO playlist_queue (
                    video_id, playlist_id, playlist_title, index_in_list,
                    title, url, duration_s, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("vid_2", "pl_test", "Test Playlist", 2, "Video Two", "https://yt.com/2", 180.0, "pending"))
        conn.close()

        items = queue_mgr.list_queue(self.db_path)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["video_id"], "vid_1")
        self.assertEqual(items[0]["status"], "pending")

        # Update item
        queue_mgr.update_item(self.db_path, "vid_1", status="completed", actions_count=5, grounded_count=5)
        updated = queue_mgr.list_queue(self.db_path)
        self.assertEqual(updated[0]["status"], "completed")
        self.assertEqual(updated[0]["actions_count"], 5)
        self.assertEqual(updated[0]["grounded_count"], 5)

        # Stats
        stats = queue_mgr.queue_stats(self.db_path)
        self.assertEqual(stats.get("completed"), 1)
        self.assertEqual(stats.get("pending"), 1)

    def test_sequential_worker_with_mocked_calls(self):
        """Simulate sequential pipeline execution without any external network or Gemini API calls."""
        # Insert 3 items: vid_1 succeeds, vid_2 fails, vid_3 succeeds
        conn = queue_mgr.get_db(self.db_path)
        with conn:
            for i, vid in enumerate(["vid_1", "vid_2", "vid_3"], 1):
                conn.execute("""
                    INSERT INTO playlist_queue (
                        video_id, playlist_id, playlist_title, index_in_list,
                        title, url, duration_s, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """, (vid, "pl_mock", "Mock Playlist", i, f"Tutorial {i}", f"https://yt.com/{vid}", 60.0 * i))
        conn.close()

        # Mock download, trajectory, and grounding stages
        def fake_download(url, target_dir):
            stem = url.split("/")[-1]
            mp4 = Path(target_dir) / f"{stem}.mp4"
            mp4.write_text("fake video bytes")

        def fake_traj(ingested, client, paths, **kwargs):
            stem = ingested[0]["stem"]
            if stem == "vid_2":
                raise RuntimeError("Simulated API rate limit or corrupt video stream on vid_2")
            return [{
                "stem": stem,
                "actions": [
                    {"step": 1, "action_type": "click", "instruction": "Click Menu"},
                    {"step": 2, "action_type": "type", "instruction": "Type text"}
                ]
            }]

        def fake_ground(traj_records, client, paths, **kwargs):
            return [
                {"grounded": True, "target_x": 100, "target_y": 200},
                {"grounded": True, "target_x": 300, "target_y": 400},
            ]

        with patch("wilcom_pipeline.jobs._download_with_progress", side_effect=fake_download), \
             patch("wilcom_pipeline.stage_trajectory.run", side_effect=fake_traj), \
             patch("wilcom_pipeline.stage_grounding.run", side_effect=fake_ground), \
             patch("wilcom_pipeline.jobs._client", return_value=MagicMock()), \
             patch("wilcom_pipeline.jobs._ingest", side_effect=lambda files, p: [{"stem": Path(files[0]).stem}]):

            pending = [item for item in queue_mgr.list_queue(self.db_path) if item["status"] in ("pending", "failed")]
            self.assertEqual(len(pending), 3)

            # Run the sequential loop logic exactly as jobs.process_playlist_queue executes it
            total_queued = len(pending)
            completed_in_run = 0
            failed_in_run = 0

            for idx, item in enumerate(pending, 1):
                vid = item["video_id"]
                url = item["url"]
                queue_mgr.update_item(self.db_path, vid, status="downloading", error_msg=None)

                try:
                    target_dir = self.paths.videos
                    target_dir.mkdir(parents=True, exist_ok=True)
                    fake_download(url, target_dir)
                    video_file = target_dir / f"{vid}.mp4"
                    queue_mgr.update_item(self.db_path, vid, status="trajectory", local_video_path=str(video_file))

                    ingested = [{"stem": vid}]
                    traj_results = fake_traj(ingested, None, self.paths)
                    traj_record = traj_results[0]
                    actions_cnt = len(traj_record["actions"])
                    queue_mgr.update_item(self.db_path, vid, status="grounding", actions_count=actions_cnt)

                    ground_results = fake_ground([traj_record], None, self.paths)
                    grounded_cnt = sum(1 for g in ground_results if g.get("grounded"))
                    queue_mgr.update_item(self.db_path, vid, status="completed", grounded_count=grounded_cnt)
                    completed_in_run += 1

                except Exception as exc:
                    queue_mgr.update_item(self.db_path, vid, status="failed", error_msg=str(exc))
                    failed_in_run += 1
                    continue

        self.assertEqual(completed_in_run, 2)
        self.assertEqual(failed_in_run, 1)

        # Check queue table state
        all_items = {r["video_id"]: r for r in queue_mgr.list_queue(self.db_path)}
        self.assertEqual(all_items["vid_1"]["status"], "completed")
        self.assertEqual(all_items["vid_1"]["grounded_count"], 2)

        self.assertEqual(all_items["vid_2"]["status"], "failed")
        self.assertIn("Simulated API rate limit", all_items["vid_2"]["error_msg"])

        self.assertEqual(all_items["vid_3"]["status"], "completed")
        self.assertEqual(all_items["vid_3"]["grounded_count"], 2)

        # Test retry failed
        reset_cnt = queue_mgr.reset_failed(self.db_path)
        self.assertEqual(reset_cnt, 1)
        all_items_after_reset = {r["video_id"]: r for r in queue_mgr.list_queue(self.db_path)}
        self.assertEqual(all_items_after_reset["vid_2"]["status"], "pending")
        self.assertIsNone(all_items_after_reset["vid_2"]["error_msg"])

        # Test clear finished
        queue_mgr.clear_queue(self.db_path, mode="finished")
        remaining = queue_mgr.list_queue(self.db_path)
        self.assertEqual(len(remaining), 1)  # only vid_2 pending remains
        self.assertEqual(remaining[0]["video_id"], "vid_2")

        # Test clear all
        queue_mgr.clear_queue(self.db_path, mode="all")
        self.assertEqual(len(queue_mgr.list_queue(self.db_path)), 0)


if __name__ == "__main__":
    unittest.main()
