import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.detector import VehicleDetectorTracker
from backend.app.schemas import TARGET_VEHICLE_CLASSES


def create_synthetic_test_video(path: str, num_frames: int = 15, width: int = 640, height: int = 480):
    """Creates a temporary synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 15.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw some station geometry
        cv2.rectangle(frame, (100, 100), (300, 400), (60, 60, 60), -1)
        # Draw moving object
        x = 120 + i * 5
        y = 200
        cv2.rectangle(frame, (x, y), (x + 80, y + 50), (180, 180, 180), -1)
        out.write(frame)
    out.release()


def test_detector_on_synthetic_video(tmp_path: Path):
    """Test full process_video execution on a temporary synthetic video."""
    video_path = str(tmp_path / "synthetic_test.mp4")
    out_video_path = str(tmp_path / "synthetic_out.mp4")
    create_synthetic_test_video(video_path, num_frames=10)

    # Initialize tracker (uses pretrained yolo11n.pt)
    tracker = VehicleDetectorTracker(model_name="yolo11n.pt", conf_threshold=0.25)
    summary, records = tracker.process_video(
        video_path=video_path,
        output_path=out_video_path,
        max_frames=10,
    )

    assert summary.total_frames == 10
    assert summary.width == 640
    assert summary.height == 480
    assert summary.fps == 15.0
    assert Path(out_video_path).exists()
    assert isinstance(summary.counts_by_class, dict)
