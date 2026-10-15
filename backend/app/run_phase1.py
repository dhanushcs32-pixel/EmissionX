import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path so direct script execution succeeds
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.detector import VehicleDetectorTracker

logger = logging.getLogger("emission_x.phase1")


def main():
    parser = argparse.ArgumentParser(
        description="Emission X - Pillar 1: Phase 1 Vehicle Detection & ByteTrack Tracking"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="fuel_station.mp4",
        help="Path to input CCTV video file (default: fuel_station.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/annotated_fuel_station.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="backend/data/phase1_summary.json",
        help="Path to save JSON summary file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (useful for rapid testing)",
    )
    parser.add_argument(
        "--lpr",
        action="store_true",
        help="Enable Automatic License Plate Recognition (OCR) for tracked vehicles",
    )
    parser.add_argument(
        "--lpr-interval",
        type=int,
        default=15,
        help="Interval of frames between OCR attempts per tracked vehicle (default: 15)",
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EMISSION X - PILLAR 1: VEHICLE DETECTION & TRACKING")
    if args.lpr:
        print("  [PHASE 2: LICENSE PLATE RECOGNITION (OCR) ENABLED]")
    print("=" * 65)
    print(f" Video Source : {video_path}")
    print(f" Model        : {args.model}")
    print(f" LPR Enabled  : {args.lpr}")
    print(f" Output Video : {output_path}")
    print(f" Summary JSON : {summary_path}")
    print("-" * 65)

    tracker = VehicleDetectorTracker(
        model_name=args.model,
        conf_threshold=args.conf,
        enable_lpr=args.lpr,
        lpr_interval=args.lpr_interval,
    )

    summary, records = tracker.process_video(
        video_path=str(video_path),
        output_path=str(output_path),
        max_frames=args.max_frames,
    )

    # Save summary JSON
    summary_dict = summary.model_dump()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 65)
    print("  EXECUTION COMPLETE")
    print("=" * 65)
    print(f" Total Frames Processed  : {summary.total_frames}")
    print(f" Video Duration (s)      : {summary.duration_seconds:.2f}s")
    print(f" Total Unique Vehicles   : {summary.total_unique_vehicles}")
    print("-" * 65)
    print(" Counts by Class:")
    for cls_name, count in summary.counts_by_class.items():
        print(f"   • {cls_name.capitalize():<12} : {count}")
    
    plates_found = [r for r in summary.tracked_vehicles if r.license_plate]
    if args.lpr:
        print("-" * 65)
        print(f" Vehicles with Plates Recognized: {len(plates_found)} / {summary.total_unique_vehicles}")
        for r in plates_found:
            print(f"   • Track #{r.track_id:<3} ({r.vehicle_class:<10}) : [{r.license_plate}] (conf: {r.plate_confidence})")

    print("-" * 65)
    print(f" Annotated Video saved   : {output_path}")
    print(f" JSON Summary saved      : {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
