import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .schemas import TARGET_VEHICLE_CLASSES, VehicleTrackRecord, VideoAnalysisSummary

logger = logging.getLogger("emission_x.detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VehicleDetectorTracker:
    """YOLO11n Vehicle Detector with ByteTrack multi-object tracking.
    
    Filters COCO classes:
      - 2: car
      - 3: motorcycle
      - 5: bus
      - 7: truck
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_lpr: bool = False,
        lpr_interval: int = 15,
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_lpr = enable_lpr
        self.lpr_interval = lpr_interval
        logger.info(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.target_class_ids = list(TARGET_VEHICLE_CLASSES.keys())

        self.lpr = None
        if self.enable_lpr:
            from .lpr import LicensePlateRecognizer
            self.lpr = LicensePlateRecognizer(gpu=True)

        # Supervision Annotators styled in Emission X cream/white and orange theme
        # BGR colors: Orange is (12, 88, 234) or (24, 115, 234), White is (255, 255, 255)
        self.orange_color = sv.Color(r=234, g=88, b=12)
        self.cream_color = sv.Color(r=253, g=248, b=244)
        
        self.box_annotator = sv.BoxAnnotator(
            color=self.orange_color,
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self.orange_color,
            text_color=self.cream_color,
            text_scale=0.45,
            text_thickness=1,
            text_padding=4,
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[VideoAnalysisSummary, Dict[int, VehicleTrackRecord]]:
        """Processes a video file frame-by-frame, runs YOLO11n + ByteTrack,
        annotates frames, and produces tracking records and summary.
        """
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video source at: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            output_p = Path(output_path).resolve()
            output_p.parent.mkdir(parents=True, exist_ok=True)
            # Use mp4v or avc1 codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_p), fourcc, fps, (width, height))

        vehicle_records: Dict[int, VehicleTrackRecord] = {}
        frame_idx = 0

        logger.info(
            f"Starting video processing: {video_path} "
            f"({width}x{height} @ {fps:.1f} FPS, total frames: {total_frames})"
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                current_timestamp = frame_idx / fps

                # Run YOLO11n with ByteTrack tracking
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=self.target_class_ids,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )[0]

                # Convert to Supervision Detections
                detections = sv.Detections.from_ultralytics(results)

                # Collect track records
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    labels = []
                    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
                        if tracker_id is None:
                            labels.append(f"{TARGET_VEHICLE_CLASSES.get(class_id, 'vehicle')} {confidence:.2f}")
                            continue

                        tracker_id = int(tracker_id)
                        cls_name = TARGET_VEHICLE_CLASSES.get(int(class_id), "unknown")

                        if tracker_id not in vehicle_records:
                            vehicle_records[tracker_id] = VehicleTrackRecord(
                                track_id=tracker_id,
                                class_id=int(class_id),
                                vehicle_class=cls_name,
                                confidence=float(confidence),
                                first_seen_frame=frame_idx,
                                first_seen_timestamp=round(current_timestamp, 3),
                                last_seen_frame=frame_idx,
                                last_seen_timestamp=round(current_timestamp, 3),
                                frames_tracked=1,
                            )
                        else:
                            rec = vehicle_records[tracker_id]
                            rec.last_seen_frame = frame_idx
                            rec.last_seen_timestamp = round(current_timestamp, 3)
                            rec.frames_tracked += 1
                            if float(confidence) > rec.confidence:
                                rec.confidence = round(float(confidence), 3)

                        rec = vehicle_records[tracker_id]

                        # License Plate Recognition (if enabled)
                        if self.enable_lpr and self.lpr:
                            # Attempt OCR when:
                            # 1) Vehicle has no plate yet and tracked for a few frames
                            # 2) Or periodically every lpr_interval frames to improve reading
                            should_run_ocr = (
                                (rec.license_plate is None and rec.frames_tracked % 5 == 0) or
                                (rec.frames_tracked % self.lpr_interval == 0)
                            )
                            if should_run_ocr:
                                x1, y1, x2, y2 = [int(v) for v in xyxy]
                                box_w = max(0, x2 - x1)
                                box_h = max(0, y2 - y1)
                                # Only process if bounding box is sufficiently large
                                if box_w >= 80 and box_h >= 60:
                                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                                    plate_res = self.lpr.read_vehicle_plate(crop)
                                    if plate_res:
                                        plate_txt, plate_conf = plate_res
                                        if rec.plate_confidence is None or plate_conf > rec.plate_confidence:
                                            rec.license_plate = plate_txt
                                            rec.plate_confidence = round(float(plate_conf), 3)
                                            logger.info(
                                                f"Frame {frame_idx}: Detected Plate [{plate_txt}] "
                                                f"(conf: {plate_conf:.2f}) for Track #{tracker_id} ({cls_name})"
                                            )

                        # Build display label (with license plate if recognized)
                        if rec.license_plate:
                            labels.append(f"#{tracker_id} {cls_name} [{rec.license_plate}] {confidence:.2f}")
                        else:
                            labels.append(f"#{tracker_id} {cls_name} {confidence:.2f}")

                    # Annotate frame
                    annotated_frame = self.box_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections,
                    )
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels,
                    )
                else:
                    annotated_frame = frame.copy()

                # Add sleek Emission X HUD overlay (cream & orange)
                plates_count = sum(1 for r in vehicle_records.values() if r.license_plate is not None)
                hud_text = (
                    f"Emission X | Frame: {frame_idx}/{total_frames} | "
                    f"Tracked: {len(vehicle_records)} | Plates: {plates_count}"
                )
                cv2.rectangle(annotated_frame, (10, 10), (620, 48), (20, 20, 20), -1)
                cv2.rectangle(annotated_frame, (10, 10), (620, 48), (12, 88, 234), 2)  # Orange border
                cv2.putText(
                    annotated_frame,
                    hud_text,
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (244, 248, 253),  # Cream white
                    2,
                    cv2.LINE_AA,
                )

                if writer:
                    writer.write(annotated_frame)

                frame_idx += 1
                if frame_idx % 50 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames ({len(vehicle_records)} vehicles tracked)...")

        finally:
            cap.release()
            if writer:
                writer.release()
                logger.info(f"Annotated output video saved to: {output_path}")

        # Compute summary
        counts_by_class: Dict[str, int] = {name: 0 for name in TARGET_VEHICLE_CLASSES.values()}
        for rec in vehicle_records.values():
            if rec.vehicle_class in counts_by_class:
                counts_by_class[rec.vehicle_class] += 1

        duration_sec = frame_idx / fps if fps > 0 else 0.0
        summary = VideoAnalysisSummary(
            video_source=video_path,
            output_annotated_video=output_path,
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=frame_idx,
            duration_seconds=round(duration_sec, 2),
            total_unique_vehicles=len(vehicle_records),
            counts_by_class=counts_by_class,
            tracked_vehicles=list(vehicle_records.values()),
        )

        return summary, vehicle_records
