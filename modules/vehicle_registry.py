"""Mock vehicle registry (VAHAN-style) for number plates read from CCTV.

MOCK DATA ONLY. Real VAHAN access needs an authorised provider, so this looks plates up in a
20-record invented database (backend/data/mock_vahan.json):

  - a plate that is in the database returns that record (match = "exact_plate");
  - any other vehicle, including one whose plate was not read, is given one of the records of the
    same vehicle class, chosen by a hash of its plate (or track id), so the same vehicle always gets
    the same record (match = "assigned_by_hash").

Every result carries data_source = "mock" and a notice, and the website shows it as mock data.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = WORKSPACE_ROOT / "backend" / "data" / "mock_vahan.json"
PHASE1_SUMMARY = WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json"

MOCK_NOTICE = "MOCK DATA: invented registry records, not from VAHAN or Parivahan"

# Detector / telemetry class names -> registry vehicle_class
CLASS_ALIASES = {"bike": "motorcycle", "motorbike": "motorcycle", "bus_truck": "truck", "auto": "auto_rickshaw"}

STATE_NAMES = {"KA": "Karnataka", "MH": "Maharashtra", "DL": "Delhi", "TN": "Tamil Nadu", "AP": "Andhra Pradesh",
               "TS": "Telangana", "KL": "Kerala", "GJ": "Gujarat", "RJ": "Rajasthan", "UP": "Uttar Pradesh"}

# Indicative tailpipe CO2 by class and fuel (g/km) and BS-norm multipliers; demo-grade like the records
CO2_G_KM = {("car", "petrol"): 140, ("car", "diesel"): 150, ("car", "cng"): 115,
            ("motorcycle", "petrol"): 45, ("auto_rickshaw", "cng"): 70, ("auto_rickshaw", "petrol"): 80,
            ("bus", "diesel"): 820, ("truck", "diesel"): 750}
NORM_FACTOR = {"BS-III": 1.25, "BS-IV": 1.1, "BS-VI": 1.0}

_cache: Dict[str, Any] = {"mtime": None, "records": []}


def records() -> List[Dict[str, Any]]:
    """The database records, re-read if the JSON file changes."""
    mtime = DB_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["records"] = json.loads(DB_PATH.read_text(encoding="utf-8"))["records"]
        _cache["mtime"] = mtime
    return _cache["records"]


def normalize_plate(plate: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (plate or "").upper())


def _state_of(plate: str) -> str:
    if re.match(r"^\d{2}BH", plate):
        return "Bharat (BH) series"
    return STATE_NAMES.get(plate[:2], plate[:2])


def _decorate(rec: Dict[str, Any], plate_read: Optional[str], match: str, today: date) -> Dict[str, Any]:
    out = dict(rec)
    fuel, cls = rec["fuel_type"], rec["vehicle_class"]
    out.update({
        "plate_read": plate_read,
        "match": match,
        "data_source": "mock",
        "notice": MOCK_NOTICE,
        "state": _state_of(rec["plate"]),
        "vehicle_age_years": max(0, today.year - int(rec["registration_year"])),
    })
    if fuel == "electric":
        out["puc_status"] = "not required (EV)"
        out["est_co2_g_km"] = 0
    else:
        puc = rec.get("puc_valid_until")
        out["puc_status"] = "expired" if puc and puc < today.isoformat() else "valid"
        base = CO2_G_KM.get((cls, fuel))
        out["est_co2_g_km"] = round(base * NORM_FACTOR.get(rec["emission_norm"], 1.0)) if base else None
    return out


def lookup(plate: Optional[str] = None, vehicle_class: Optional[str] = None, key: Optional[str] = None,
           today: Optional[date] = None) -> Dict[str, Any]:
    """Registry record for a plate (or, without a readable plate, for a stable key such as a track id)."""
    today = today or date.today()
    recs = records()
    norm = normalize_plate(plate)
    exact = next((r for r in recs if r["plate"] == norm), None) if norm else None
    if exact:
        return _decorate(exact, norm, "exact_plate", today)
    cls = CLASS_ALIASES.get(vehicle_class or "", vehicle_class)
    pool = [r for r in recs if r["vehicle_class"] == cls] or recs
    seed = norm or str(key or "")
    idx = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % len(pool)
    return _decorate(pool[idx], norm or None, "assigned_by_hash", today)


def enrich_tracks(tracks: Iterable[Dict[str, Any]], today: Optional[date] = None) -> List[Dict[str, Any]]:
    """One registry record per tracked vehicle from the detector summary."""
    out = []
    for t in tracks:
        rec = lookup(t.get("license_plate"), t.get("vehicle_class"), key=f"track-{t.get('track_id')}", today=today)
        rec["track_id"] = t.get("track_id")
        rec["detected_class"] = t.get("vehicle_class")
        out.append(rec)
    return out


def fleet_summary(vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
    fuel_mix: Dict[str, int] = {}
    norm_mix: Dict[str, int] = {}
    for v in vehicles:
        fuel_mix[v["fuel_type"]] = fuel_mix.get(v["fuel_type"], 0) + 1
        norm_mix[v["emission_norm"]] = norm_mix.get(v["emission_norm"], 0) + 1
    ages = [v["vehicle_age_years"] for v in vehicles]
    return {
        "vehicles": len(vehicles),
        "plates_read": sum(1 for v in vehicles if v.get("plate_read")),
        "exact_matches": sum(1 for v in vehicles if v["match"] == "exact_plate"),
        "fuel_mix": fuel_mix,
        "emission_norm_mix": norm_mix,
        "electric": fuel_mix.get("electric", 0),
        "puc_expired": sum(1 for v in vehicles if v["puc_status"] == "expired"),
        "bs3_bs4_heavy": sum(1 for v in vehicles
                             if v["vehicle_class"] in ("bus", "truck") and v["emission_norm"] in ("BS-III", "BS-IV")),
        "avg_age_years": round(sum(ages) / len(ages), 1) if ages else None,
    }


def fleet_from_phase1(summary_path: Path = PHASE1_SUMMARY, today: Optional[date] = None) -> Dict[str, Any]:
    """Registry view of the vehicles in the current fuel-station clip (backend/data/phase1_summary.json)."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    vehicles = enrich_tracks(summary.get("tracked_vehicles") or [], today=today)
    return {
        "data_source": "mock",
        "notice": MOCK_NOTICE,
        "database_size": len(records()),
        "source_video": Path(str(summary.get("video_source") or "")).name or None,
        "vehicles": vehicles,
        "summary": fleet_summary(vehicles),
    }
