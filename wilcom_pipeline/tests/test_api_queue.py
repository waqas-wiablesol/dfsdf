"""Smoke test for FastAPI queue endpoints using TestClient with mocked API calls."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from wilcom_pipeline import api, config, queue as queue_mgr


class TestQueueEndpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="wilcom_api_test_")
        self.work_dir = Path(self.temp_dir)
        self.paths = config.Paths(root=self.work_dir).ensure()
        self.db_path = self.paths.root / "queue.db"
        self.client = TestClient(api.app)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_queue_endpoints_flow(self):
        # 1. GET /api/queue empty
        res = self.client.get(f"/api/queue?work={self.work_dir}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["items"], [])

        # 2. Mock enqueue
        mock_playlist_extract = {
            "is_playlist": True,
            "playlist_id": "pl_123",
            "playlist_title": "Fake Ingestion Playlist",
            "items": [
                {
                    "video_id": "mock_vid_1",
                    "title": "Tutorial 1",
                    "url": "https://youtube.com/watch?v=mock_vid_1",
                    "duration_s": 150.0,
                    "index_in_list": 1,
                },
                {
                    "video_id": "mock_vid_2",
                    "title": "Tutorial 2",
                    "url": "https://youtube.com/watch?v=mock_vid_2",
                    "duration_s": 240.0,
                    "index_in_list": 2,
                },
            ]
        }
        with patch("wilcom_pipeline.queue.extract_playlist_items", return_value=mock_playlist_extract):
            enqueue_res = self.client.post("/api/queue/enqueue", json={
                "url": "https://youtube.com/playlist?list=pl_123",
                "work": str(self.work_dir),
            })
            self.assertEqual(enqueue_res.status_code, 200)
            res_data = enqueue_res.json()
            self.assertTrue(res_data["ok"])
            self.assertEqual(res_data["result"]["total_extracted"], 2)

        # 3. GET /api/queue populated
        res = self.client.get(f"/api/queue?work={self.work_dir}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["stats"]["pending"], 2)

        # 4. Mark one item as failed to test retry endpoint
        queue_mgr.update_item(self.db_path, "mock_vid_1", status="failed", error_msg="Mock failure")
        res = self.client.get(f"/api/queue?work={self.work_dir}")
        self.assertEqual(res.json()["stats"]["failed"], 1)

        retry_res = self.client.post("/api/queue/retry", json={"work": str(self.work_dir)})
        self.assertEqual(retry_res.status_code, 200)
        self.assertEqual(retry_res.json()["reset_count"], 1)

        res_after_retry = self.client.get(f"/api/queue?work={self.work_dir}")
        self.assertEqual(res_after_retry.json()["stats"]["pending"], 2)

        # 5. Clear queue
        clear_res = self.client.post("/api/queue/clear", json={"work": str(self.work_dir), "mode": "all"})
        self.assertEqual(clear_res.status_code, 200)
        self.assertEqual(clear_res.json()["cleared_count"], 2)

        res_empty = self.client.get(f"/api/queue?work={self.work_dir}")
        self.assertEqual(res_empty.json()["total"], 0)

    def test_url_inspect_endpoints(self):
        mock_playlist = {
            "is_playlist": True,
            "playlist_id": "PLtest123",
            "playlist_title": "Test Playlist Title",
            "channel": "Test Channel",
            "total_videos": 3,
            "items": [
                {"video_id": "v1", "title": "Vid 1", "url": "https://youtu.be/v1", "duration_s": 60.0, "index_in_list": 1},
                {"video_id": "v2", "title": "Vid 2", "url": "https://youtu.be/v2", "duration_s": 120.0, "index_in_list": 2},
                {"video_id": "v3", "title": "Vid 3", "url": "https://youtu.be/v3", "duration_s": 180.0, "index_in_list": 3},
            ],
        }

        # Test POST /api/url/inspect
        with patch("wilcom_pipeline.queue.extract_playlist_items", return_value=mock_playlist):
            res = self.client.post("/api/url/inspect", json={"url": "https://youtube.com/playlist?list=PLtest123"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["ok"])
            self.assertTrue(data["data"]["is_playlist"])
            self.assertEqual(data["data"]["playlist_title"], "Test Playlist Title")
            self.assertEqual(data["data"]["total_videos"], 3)
            self.assertEqual(len(data["data"]["items"]), 3)

        # Test GET /api/url/inspect
        with patch("wilcom_pipeline.queue.extract_playlist_items", return_value=mock_playlist):
            res_get = self.client.get("/api/url/inspect?url=https%3A%2F%2Fyoutube.com%2Fplaylist%3Flist%3DPLtest123")
            self.assertEqual(res_get.status_code, 200)
            data_get = res_get.json()
            self.assertTrue(data_get["ok"])
            self.assertEqual(data_get["data"]["channel"], "Test Channel")

        # Test empty URL validation
        res_empty = self.client.post("/api/url/inspect", json={"url": "   "})
        self.assertEqual(res_empty.status_code, 400)

    def test_autoflow_endpoints(self):
        # 1. Single video autoflow route POST /api/run/autoflow
        with patch("wilcom_pipeline.jobs.start", return_value="Auto-Flow Pipeline") as mock_start:
            res = self.client.post("/api/run/autoflow", json={
                "work": str(self.work_dir),
                "url": "https://www.youtube.com/watch?v=demo123",
                "thinking": "medium",
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["started"])
            self.assertEqual(data["step"], "autoflow")
            self.assertEqual(data["message"], "Auto-Flow Pipeline")
            mock_start.assert_called_once()

        # 2. Queue autoflow with empty queue and no URL
        res_empty = self.client.post("/api/queue/autoflow", json={
            "work": str(self.work_dir),
            "url": "",
        })
        self.assertEqual(res_empty.status_code, 200)
        self.assertFalse(res_empty.json()["started"])
        self.assertEqual(res_empty.json()["reason"], "empty")

        # 3. Queue autoflow with playlist URL
        mock_playlist_extract = {
            "is_playlist": True,
            "playlist_id": "pl_autoflow",
            "playlist_title": "Auto-flow Playlist",
            "items": [
                {
                    "video_id": "vid_af_1",
                    "title": "Tutorial 1",
                    "url": "https://youtube.com/watch?v=vid_af_1",
                    "duration_s": 100.0,
                    "index_in_list": 1,
                },
            ]
        }
        with patch("wilcom_pipeline.queue.extract_playlist_items", return_value=mock_playlist_extract):
            with patch("wilcom_pipeline.jobs.start", return_value="Playlist Pipeline") as mock_queue_start:
                res_pl = self.client.post("/api/queue/autoflow", json={
                    "work": str(self.work_dir),
                    "url": "https://youtube.com/playlist?list=pl_autoflow",
                })
                self.assertEqual(res_pl.status_code, 200)
                data_pl = res_pl.json()
                self.assertTrue(data_pl["started"])
                self.assertEqual(data_pl["step"], "playlist")
                self.assertIsNotNone(data_pl["enqueued"])
                mock_queue_start.assert_called_once()

    def test_stage_review_import_and_execution(self):
        from wilcom_pipeline import records, stage_review
        stem = "test_video"
        traj = {
            "video_stem": stem,
            "video": "test_video.mp4",
            "tasks": [
                {
                    "task_id": 1,
                    "description": "Test task",
                    "instruction": "Do the task",
                    "actions": [
                        {
                            "task_id": 1,
                            "seq": 1,
                            "action_type": "click",
                            "grounding_instruction": "Click the icon",
                            "t": 1.0,
                            "needs_grounding": True,
                        }
                    ],
                }
            ],
        }
        records.write_json(traj, self.paths.traj / f"{stem}.json")

        ground = {
            "video_stem": stem,
            "results": [
                {
                    "task_id": 1,
                    "seq": 1,
                    "grounded": True,
                    "frame": "frame_001.png",
                    "frame_time_s": 1.0,
                    "point": [100, 200],
                    "point_norm": [0.2, 0.3],
                    "image_width": 1000,
                    "image_height": 1000,
                    "bbox": [90, 190, 110, 210],
                }
            ],
        }
        records.write_json(ground, self.paths.ground / f"{stem}.json")

        frame_path = self.paths.frames / stem / "frame_001.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"dummy_frame")

        rep = stage_review.run(self.paths)
        self.assertEqual(rep["frames_checked"], 1)
        self.assertEqual(rep["coverage"], 1.0)
        self.assertTrue((self.paths.logs / "review.json").exists())


if __name__ == "__main__":
    unittest.main()


