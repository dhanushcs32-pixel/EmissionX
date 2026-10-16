"""Background analysis of clips uploaded from the website (POST /api/upload/{sector}).

One job runs at a time. The detector writes into uploads/work/, then the results are copied onto
the files the API already serves, so the website, digital twin and report all pick up the new clip:

  industrial -> cache/factory_annotated.mp4, cache/factory_telemetry.json (+ _meta.json)
  vehicular  -> cache/fuel_annotated.mp4, cache/fuel_telemetry.json, backend/data/phase1_summary.json

Before the first upload for a sector replaces those files, the originals are saved in
uploads/originals/<sector>/ (manifest.json says which existed); remove() puts them back.
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

from modules import forecourt_fuel, vehicle_registry

logger = logging.getLogger("UploadJobs")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = WORKSPACE_ROOT / "uploads"
WORK_DIR = UPLOAD_DIR / "work"
ORIG_DIR = UPLOAD_DIR / "originals"
CACHE_DIR = WORKSPACE_ROOT / "cache"

# Files an upload replaces, per sector (restored by remove())
PUBLISHED = {
    "industrial": [CACHE_DIR / "factory_annotated.mp4", CACHE_DIR / "factory_telemetry.json",
                   CACHE_DIR / "factory_telemetry_meta.json"],
    "vehicular": [CACHE_DIR / "fuel_annotated.mp4", CACHE_DIR / "fuel_telemetry.json",
                  WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json"],
}

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


def _delete(path: Path, attempts: int = 10) -> None:
    """Deletes a served file; on Windows it can stay locked briefly while a video response closes."""
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.3)


def _manifest_path(sector: str) -> Path:
    return ORIG_DIR / sector / "manifest.json"


def _backup_originals(sector: str, source_video: str) -> None:
    """Saves the files this upload is about to replace, once per sector; later uploads keep that backup."""
    mp = _manifest_path(sector)
    if mp.is_file():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    else:
        mp.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"files": {}}
        for p in PUBLISHED[sector]:
            rel = p.relative_to(WORKSPACE_ROOT).as_posix()
            if p.is_file():
                shutil.copyfile(p, mp.parent / p.name)
                manifest["files"][rel] = "saved"
            else:
                manifest["files"][rel] = "absent"
    manifest["source_video"] = source_video
    manifest["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def uploaded_info(sector: str) -> Optional[Dict[str, Any]]:
    """The uploaded clip currently replacing this sector's data, or None if the originals are in place."""
    try:
        m = json.loads(_manifest_path(sector).read_text(encoding="utf-8"))
        return {"source_video": m.get("source_video"), "uploaded_at": m.get("uploaded_at")}
    except (OSError, ValueError):
        return None


def remove(sector: str) -> bool:
    """Puts back the data that existed before the first upload for this sector. False if nothing to remove."""
    if is_running():
        raise RuntimeError("A clip is still being analysed")
    mp = _manifest_path(sector)
    if not mp.is_file():
        return False
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for rel, state in manifest.get("files", {}).items():
        target = WORKSPACE_ROOT / rel
        if state == "saved":
            _publish(mp.parent / target.name, target)
        else:
            _delete(target)
    shutil.rmtree(mp.parent, ignore_errors=True)
    for _ in range(5):   # OneDrive or antivirus can hold the emptied folder for a moment
        if not mp.parent.exists():
            break
        time.sleep(0.3)
        shutil.rmtree(mp.parent, ignore_errors=True)
    for clip in UPLOAD_DIR.glob(f"{sector}_*"):
        clip.unlink(missing_ok=True)
    with _lock:
        if _job.get("sector") == sector and _job.get("state") in ("done", "error"):
            _job.clear()
            _job["state"] = "idle"
    logger.info(f"Removed uploaded {sector} clip; original data restored")
    return True


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
    _backup_originals("industrial", original_name)
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
    summary, _ = tracker.process_video(video_path=str(video_path), output_path=str(work_video),
                                       progress_cb=_progress, source_name=original_name)

    summary_dict = summary.model_dump()
    summary_dict["video_source"] = original_name
    summary_dict["output_annotated_video"] = "cache/fuel_annotated.mp4"
    _update(stage="computing fuel, idle and VOC emissions")
    forecourt_fuel.enrich_summary(summary_dict, temp_c=forecourt_fuel.current_temp_c())
    records = tracks_to_fuel_telemetry(summary_dict)

    _update(stage="publishing results")
    _backup_originals("vehicular", original_name)
    _publish(work_video, CACHE_DIR / "fuel_annotated.mp4")
    (CACHE_DIR / "fuel_telemetry.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    plates = sum(1 for t in summary_dict["tracked_vehicles"] if t.get("license_plate"))
    fc = summary_dict.get("forecourt") or {}
    return (f"{summary_dict['total_frames']} frames analysed; {summary_dict['total_unique_vehicles']} vehicles tracked, "
            f"{plates} number plates read, {fc.get('refuels', 0)} refuels ({fc.get('litres_total', 0)} L)")


def tracks_to_fuel_telemetry(summary: Dict[str, Any], node_id: str = "BLR_FUEL_STATION_01") -> List[Dict[str, Any]]:
    """Turns per-vehicle tracks into the per-second vehicular telemetry contract served by /api/telemetry.

    Counts are the vehicles visible in that second. bay_dwell_sec / queue_idle_sec are the average time the
    vehicles currently in Bay_01 / Queue_01 have been there. Refuel events (summary["forecourt"], from
    modules.forecourt_fuel.enrich_summary) are logged in the second they trigger. CO2 rates are the clip's
    fuel CO2 (scope 3) and queue idling CO2 (scope 1) per hour of footage.
    """
    fps = float(summary.get("fps") or 25.0)
    total = int(summary.get("total_frames") or 0)
    tracks = summary.get("tracked_vehicles") or []
    fc = summary.get("forecourt") or {}
    step = max(1, int(round(fps)))
    clip_hr = max(total / fps, 1.0) / 3600.0
    idle_rate = round(float(fc.get("co2_idle_kg") or 0.0) / clip_hr, 1)
    base_rate = round((float(fc.get("co2_fuel_kg") or 0.0) + float(fc.get("co2_idle_kg") or 0.0)) / clip_hr, 1)
    voc_rate = round(float(fc.get("voc_kg") or 0.0) / clip_hr, 3)

    def in_span(t: Dict[str, Any], first: str, last: str, f: int) -> bool:
        return t.get(first) is not None and t[first] <= f <= (t.get(last) or t[first])

    records: List[Dict[str, Any]] = []
    for f in range(0, max(total, 1), step):
        active = [t for t in tracks if t["first_seen_frame"] <= f <= t["last_seen_frame"]]
        counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
        for t in active:
            cls = t.get("vehicle_class")
            counts[cls if cls in counts else "car"] += 1
        in_bay = [t for t in tracks if in_span(t, "bay_first_frame", "bay_last_frame", f)]
        in_queue = [t for t in tracks if in_span(t, "queue_first_frame", "queue_last_frame", f)]
        bay_dwell = round(sum((f - t["bay_first_frame"]) / fps for t in in_bay) / len(in_bay), 1) if in_bay else 0.0
        queue_waits = [(f - t["queue_first_frame"]) / fps for t in in_queue]
        queue_idle = round(sum(queue_waits) / len(queue_waits), 1) if queue_waits else 0.0

        logs = [f" [VEH-CV] {len(active)} vehicles in view (cars {counts['car']}, bikes {counts['motorcycle']}, "
                f"trucks {counts['truck']}, buses {counts['bus']}) | Bay_01: {len(in_bay)} | Queue_01: {len(in_queue)}"]
        for t in tracks:   # refuel events that trigger in this second
            trig = t.get("refuel_trigger_frame")
            ev = (t.get("forecourt") or {}).get("refuel")
            if ev and trig is not None and f <= trig < f + step:
                logs.append(ev["log"])
        long_wait = [w for w in queue_waits if w > 40.0]
        if long_wait:
            wasted = sum((t.get("forecourt") or {}).get("co2_idle_kg", 0.0) for t, w in zip(in_queue, queue_waits) if w > 40.0)
            logs.append(f" [QUEUE-WARN] {len(long_wait)} vehicles idling > 40s in Queue_01 -> +{wasted:.2f} kg wasted CO2")
        elif in_queue:
            logs.append(f" [QUEUE] {len(in_queue)} vehicles waiting in Queue_01, avg {queue_idle}s")
        for t in [t for t in active if t.get("license_plate")][:2]:
            reg = vehicle_registry.lookup(t["license_plate"], t.get("vehicle_class"))
            logs.append(f" [LPR] Plate {t['license_plate']} ({t['vehicle_class']}, conf {t.get('plate_confidence')}) -> "
                        f"MOCK VAHAN: {reg['maker_model']}, {reg['fuel_type']} {reg['emission_norm']}, "
                        f"{reg['registration_year']}, PUC {reg['puc_status']}")

        records.append({
            "node_id": node_id,
            "sector": "vehicular",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": {},
            "cv_telemetry": {
                "active_counts": counts,
                "bay_dwell_sec": bay_dwell,
                "queue_idle_sec": queue_idle,
                "smoke_opacity_pct": None,
                "ringelmann_level": 0,
                "opacity_multiplier": 1.0,
                "recent_logs": logs,
            },
            "emissions": {"baseline_co2_kg_hr": base_rate, "waste_co2_kg_hr": idle_rate, "voc_kg_hr": voc_rate},
            "simulation": {
                "active": False,
                "projected_co2_kg_hr": base_rate,
                "reduction_pct": 0.0,
                "applied_params": {"ev_adoption_pct": 0.0, "e20_blend_pct": 0.0,
                                   "queue_optimization_pct": 0.0, "ban_bs3_bs4_heavy": False},
            },
        })
    return records
