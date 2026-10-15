from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# COCO Vehicle Classes relevant to Pillar 1
TARGET_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleTrackRecord(BaseModel):
    """Record for an individual tracked vehicle."""
    track_id: int = Field(..., description="Unique ByteTrack ID")
    class_id: int = Field(..., description="COCO class index (2, 3, 5, 7)")
    vehicle_class: str = Field(..., description="Human-readable vehicle class name")
    confidence: float = Field(..., description="Highest detection confidence")
    first_seen_frame: int = Field(..., description="First video frame index")
    first_seen_timestamp: float = Field(..., description="First seen timestamp in seconds")
    last_seen_frame: int = Field(..., description="Last video frame index")
    last_seen_timestamp: float = Field(..., description="Last seen timestamp in seconds")
    frames_tracked: int = Field(..., description="Total frames vehicle was tracked")
    # Phase 2: License Plate Recognition (LPR)
    license_plate: Optional[str] = Field(None, description="Recognized vehicle license plate number")
    plate_confidence: Optional[float] = Field(None, description="Confidence of plate OCR recognition")
    # Placeholders for future phases (Phase 2 & 3)
    zones_entered: List[str] = Field(default_factory=list, description="Zones vehicle entered")
    estimated_dwell_seconds: float = Field(0.0, description="Dwell time estimate in active bay")
    data_quality_flag: str = Field("phase1_tracking_only", description="Quality and verification status")


class VideoAnalysisSummary(BaseModel):
    """Summary of video vehicle tracking results."""
    video_source: str
    output_annotated_video: Optional[str] = None
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    total_unique_vehicles: int
    counts_by_class: Dict[str, int]
    tracked_vehicles: List[VehicleTrackRecord]
