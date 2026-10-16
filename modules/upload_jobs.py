"""Background analysis of clips uploaded from the website (POST /api/upload/{sector}).

One job runs at a time. The detector writes into uploads/work/, then the results are copied onto
the files the API already serves, so the website, digital twin and report all pick up the new clip:

  industrial -> cache/factory_annotated.mp4, cache/factory_telemetry.json (+ _meta.json)
  vehicular  -> cache/fuel_annotated.mp4, cache/fuel_telemetry.json, backend/data/phase1_summary.json
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("UploadJobs")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = WORKSPACE_ROOT / "uploads"
WORK_DIR = UPLOAD_DIR / "work"
CACHE_DIR = WORKSPACE_ROOT / "cache"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB
SECTORS = ("industrial", "vehicular")

_lock = threading.Lock()
_job: Dict[str, Any] = {"state": "idle"}


def status() -> Dict[str, Any]:
    with _lock:
        return dict(_job)


def is_running() -> bool:
    with _lock:
        return _job.get("state") == "running"


def _update(**fields: Any) -> None:
    with _lock:
        _job.update(fields)


def new_upload_path(sector: str, filename: str) -> Path:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext or 'none'}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return UPLOAD_DIR / f"{sector}_{stamp}{ext}"


def start(sector: str, video_path: Path, original_name: str) -> Dict[str, Any]:
    """Starts analysing an uploaded clip in a background thread. Caller checks is_running() first."""
    with _lock:
        if _job.get("state") == "running":
            raise RuntimeError("Another clip is still being analysed")
        _job.clear()
        _job.update({
            "state": "running", "sector": sector, "filename": original_name, "stage": "starting",
            "frames_done": 0, "total_frames": 0, "started_at": datetime.now(timezone.utc).isoformat(),
        })
    threading.Thread(target=_run, args=(sector, video_path, original_name), daemon=True).start()
    return status()


def _progress(done: int, total: int) -> None:
    _update(frames_done=int(done), total_frames=int(max(total, 0)))


def _publish(src: Path, dst: Path, attempts: int = 5) -> None:
    """Copies a finished output over the served file. Copying (not renaming) still works on Windows
    while a browser is streaming the old video; retry briefly in case the file is busy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        try:
            shutil.copyfile(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def _run(sector: str, video_path: Path, original_name: str) -> None:
    t0 = time.time()
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        if sector == "industrial":
            message = _run_industrial(video_path, original_name)
        else:
            message = _run_vehicular(video_path, original_name)
        _update(state="done", stage="done", message=message, elapsed_sec=round(time.time() - t0, 1),
                finished_at=datetime.now(timezone.utc).isoformat())
        logger.info(f"Upload job finished: {message}")
    except Exception as e:
        logger.error(f"Upload job failed: {e}\n{traceback.format_exc()}")
        _update(state="error", stage="failed", error=str(e), elapsed_sec=round(time.time() - t0, 1),
                finished_at=datetime.now(timezone.utc).isoformat())


def _run_industrial(video_path: Path, original_name: str) -> str:
    from modules.industrial_detector import process_video

    _update(stage="segmenting smoke plume")
    work_video = WORK_DIR / "factory_annotated.mp4"
    work_json = WORK_DIR / "factory_telemetry.json"
    stats = process_video(
        input_video_path=str(video_path),
        output_video_path=str(work_video),
        output_json_path=str(work_json),
        model_path=str(WORKSPACE_ROOT / "models" / "smoke.pt"),
        progress_cb=_progress,
    )
    meta_src = Path(stats.get("meta_json") or WORK_DIR / "factory_telemetry_meta.json")
    meta = json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.is_file() else {}
    meta.update({"source_video": original_name, "uploaded_from_website": True})

    _update(stage="publishing results")
    _publish(work_video, CACHE_DIR / "factory_annotated.mp4")
    _publish(work_json, CACHE_DIR / "factory_telemetry.json")
    (CACHE_DIR / "factory_telemetry_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return (f"{stats['frames_processed']} frames analysed with {stats.get('detector_mode') or stats.get('profile')}; "
            f"{stats['records_count']} telemetry records")


def _vehicle_model() -> str:
    """Local YOLO weights if present, else the ultralytics name (downloaded on first use)."""
    for p in (WORKSPACE_ROOT / "yolo11n.pt", WORKSPACE_ROOT / "models" / "yolo11n.pt"):
        if p.is_file():
            return str(p)
    return "yolo11n.pt"


def _run_vehicular(video_path: Path, original_name: str) -> str:
    _update(stage="loading vehicle detector and plate reader")
    from backend.app.detector import VehicleDetectorTracker

    tracker = VehicleDetectorTracker(model_name=_vehicle_model(), enable_lpr=True, lpr_interval=10)
    _update(stage="tracking vehicles and reading plates")
    work_video = WORK_DIR / "fuel_annotated.mp4"
    summary, _ = tracker.process_video(video_path=str(video_path), output_path=str(work_video), progress_cb=_progress)

    summary_dict = summary.model_dump()
    summary_dict["video_source"] = original_name
    summary_dict["output_annotated_video"] = "cache/fuel_annotated.mp4"
    records = tracks_to_fuel_telemetry(summary_dict)

    _update(stage="publishing results")
    _publish(work_video, CACHE_DIR / "fuel_annotated.mp4")
    (CACHE_DIR / "fuel_telemetry.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    plates = sum(1 for t in summary_dict["tracked_vehicles"] if t.get("license_plate"))
    return (f"{summary_dict['total_frames']} frames analysed; {summary_dict['total_unique_vehicles']} vehicles tracked, "
            f"{plates} number plates read")


def tracks_to_fuel_telemetry(summary: Dict[str, Any], node_id: str = "BLR_FUEL_STATION_01") -> List[Dict[str, Any]]:
    """Turns per-vehicle tracks into the per-second vehicular telemetry contract served by /api/telemetry.

    Each record counts the vehicles visible in that second of the clip. CO2 uses the same simple
    forecourt model as the receiver stub in main.py (running burn + per-class load + dwell waste).
    """
    fps = float(summary.get("fps") or 25.0)
    total = int(summary.get("total_frames") or 0)
    tracks = summary.get("tracked_vehicles") or []
    step = max(1, int(round(fps)))
    records: List[Dict[str, Any]] = []
    for f in range(0, max(total, 1), step):
        active = [t for t in tracks if t["first_seen_frame"] <= f <= t["last_seen_frame"]]
        counts = {"car": 0, "motorcycle": 0, "auto_rickshaw": 0, "bus_truck": 0}
        for t in active:
            cls = t.get("vehicle_class")
            counts["bus_truck" if cls in ("bus", "truck") else "motorcycle" if cls == "motorcycle" else "car"] += 1
        durations = [(t["last_seen_frame"] - t["first_seen_frame"]) / fps for t in active]
        dwell = round(sum(durations) / len(durations), 1) if durations else 0.0
        cars, bikes, heavy = counts["car"], counts["motorcycle"], counts["bus_truck"]
        base_co2 = round(110.0 + cars * 6.5 + bikes * 1.8 + heavy * 18.0, 1)
        waste_co2 = round((dwell / 30.0) * (cars * 3.5 + heavy * 9.0), 1)

        logs = [f" [VEH-CV] {len(active)} vehicles in view (cars {cars}, bikes {bikes}, bus/truck {heavy})"]
        for t in [t for t in active if t.get("license_plate")][:2]:
            logs.append(f" [LPR] Plate {t['license_plate']} ({t['vehicle_class']}, conf {t.get('plate_confidence')})")
        logs.append(f" [DWELL] Avg time in view: {dwell}s at t={f / fps:.0f}s of {summary.get('video_source', 'clip')}")

        records.append({
            "node_id": node_id,
            "sector": "vehicular",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": {},
            "cv_telemetry": {
                "active_counts": counts,
                "bay_dwell_sec": dwell,
                "queue_idle_sec": None,
                "smoke_opacity_pct": None,
                "ringelmann_level": 0,
                "opacity_multiplier": 1.0,
                "recent_logs": logs,
            },
            "emissions": {"baseline_co2_kg_hr": base_co2, "waste_co2_kg_hr": waste_co2},
            "simulation": {
                "active": False,
                "projected_co2_kg_hr": base_co2,
                "reduction_pct": 0.0,
                "applied_params": {"ev_adoption_pct": 0.0, "e20_blend_pct": 0.0,
                                   "queue_optimization_pct": 0.0, "ban_bs3_bs4_heavy": False},
            },
        })
    return records
