import sys
from pathlib import Path

# Add project root to sys.path so direct script execution succeeds
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from backend.app.schemas import (
    TARGET_VEHICLE_CLASSES,
    VehicleTrackRecord,
    VideoAnalysisSummary,
)


def test_target_vehicle_classes():
    """Verify COCO classes required by Pillar 1 specifications."""
    assert 2 in TARGET_VEHICLE_CLASSES and TARGET_VEHICLE_CLASSES[2] == "car"
    assert 3 in TARGET_VEHICLE_CLASSES and TARGET_VEHICLE_CLASSES[3] == "motorcycle"
    assert 5 in TARGET_VEHICLE_CLASSES and TARGET_VEHICLE_CLASSES[5] == "bus"
    assert 7 in TARGET_VEHICLE_CLASSES and TARGET_VEHICLE_CLASSES[7] == "truck"
    assert len(TARGET_VEHICLE_CLASSES) == 4


def test_vehicle_track_record_validation():
    """Ensure VehicleTrackRecord accepts valid tracking inputs."""
    rec = VehicleTrackRecord(
        track_id=1,
        class_id=2,
        vehicle_class="car",
        confidence=0.89,
        first_seen_frame=0,
        first_seen_timestamp=0.0,
        last_seen_frame=50,
        last_seen_timestamp=2.0,
        frames_tracked=51,
    )
    assert rec.track_id == 1
    assert rec.vehicle_class == "car"
    assert rec.frames_tracked == 51
    assert rec.data_quality_flag == "phase1_tracking_only"


def test_video_analysis_summary_serialization():
    """Ensure VideoAnalysisSummary serializes correctly to JSON dict."""
    rec = VehicleTrackRecord(
        track_id=10,
        class_id=3,
        vehicle_class="motorcycle",
        confidence=0.92,
        first_seen_frame=10,
        first_seen_timestamp=0.4,
        last_seen_frame=60,
        last_seen_timestamp=2.4,
        frames_tracked=50,
    )
    summary = VideoAnalysisSummary(
        video_source="test.mp4",
        output_annotated_video="test_out.mp4",
        width=1920,
        height=1080,
        fps=25.0,
        total_frames=100,
        duration_seconds=4.0,
        total_unique_vehicles=1,
        counts_by_class={"car": 0, "motorcycle": 1, "bus": 0, "truck": 0},
        tracked_vehicles=[rec],
    )
    dumped = summary.model_dump()
    assert dumped["total_unique_vehicles"] == 1
    assert dumped["counts_by_class"]["motorcycle"] == 1
    assert len(dumped["tracked_vehicles"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])

