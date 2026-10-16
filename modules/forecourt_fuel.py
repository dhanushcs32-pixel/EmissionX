"""Tri-modal synthetic fuel data and forecourt emissions for tracked vehicles (Pillar 1 blueprint).

SIMULATED DATA. A vehicle that stands still in Bay_01 for longer than the trigger (3 s) gets a refuel
event, tagged with one of three verification methods for diversity:

  A  FCC dispenser IoT sync   litres = nozzle flow (0.6 L/s car, 1.3 L/s truck/bus) x nozzle time
  B  Optical dwell estimate   litres = max(0, bay_seconds - 15) x flow_rate
  C  FASTag / RFID handshake  plate from the tag -> mock VAHAN record; litres = flow x nozzle time,
                              capped at a typical fill for the class

Fuel type and BS norm come from the mock VAHAN registry (modules/vehicle_registry.py).

  Scope 3 (dispensed fuel):  CO2_fuel = L_petrol x 2.31 + L_diesel x 2.68   (CNG: 2.75 per kg-equivalent)
  Scope 1 (queue idling):    CO2_idle = queue_seconds x class idle rate x BS multiplier (BS3 1.35, BS4 1.15, BS6 1.0)
  VOC evaporative loss:      refuelling vapour per litre, raised 5 % per degree C above 30 C
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modules import vehicle_registry

logger = logging.getLogger("ForecourtFuel")

DATA_NOTICE = "SIMULATED fuel data (tri-modal generator) with MOCK VAHAN records; zone timings come from the video"

FLOW_L_PER_S = {"car": 0.6, "motorcycle": 0.25, "auto_rickshaw": 0.4, "bus": 1.3, "truck": 1.3}
TYPICAL_FILL_L = {"car": 18.0, "motorcycle": 4.0, "auto_rickshaw": 8.0, "bus": 45.0, "truck": 65.0}
NOZZLE_SETUP_S = 8.0           # bay time before the nozzle starts flowing (methods A and C)
OPTICAL_OFFSET_S = 15.0        # method B, as in the blueprint
CO2_PER_L = {"petrol": 2.31, "diesel": 2.68, "cng": 2.75}
IDLE_KG_PER_S = {"car": 0.00075, "motorcycle": 0.00025, "auto_rickshaw": 0.0004, "bus": 0.0020, "truck": 0.0022}
BS_IDLE_MULT = {"BS-III": 1.35, "BS-IV": 1.15, "BS-VI": 1.0}
VOC_G_PER_L = {"petrol": 1.3, "diesel": 0.05, "cng": 0.0}     # refuelling vapour displacement, indicative
VOC_HOT_THRESHOLD_C = 30.0
VOC_PCT_PER_DEG = 5.0

METHODS = {
    "A": ("DISPENSER-IOT", "FCC dispenser IoT sync"),
    "B": ("OPTICAL-DWELL", "Optical dwell estimation"),
    "C": ("FASTAG-SYNC", "FASTag / RFID handshake"),
}
SHORT_NORM = {"BS-III": "BS3", "BS-IV": "BS4", "BS-VI": "BS6"}
DEFAULT_TEMP_C = 29.5


def method_for(track_id: Any) -> str:
    """Spread the three verification methods across vehicles (stable per track)."""
    try:
        return "ABC"[int(track_id) % 3]
    except (TypeError, ValueError):
        return "A"


def litres_for(method: str, vehicle_class: str, bay_seconds: float) -> float:
    flow = FLOW_L_PER_S.get(vehicle_class, 0.6)
    if method == "B":
        return max(0.0, bay_seconds - OPTICAL_OFFSET_S) * flow
    nozzle = max(0.0, bay_seconds - NOZZLE_SETUP_S) * flow
    if method == "C":
        return min(nozzle, TYPICAL_FILL_L.get(vehicle_class, 18.0))
    return nozzle


def voc_factor(temp_c: Optional[float]) -> float:
    if temp_c is None or temp_c <= VOC_HOT_THRESHOLD_C:
        return 1.0
    return 1.0 + VOC_PCT_PER_DEG * (temp_c - VOC_HOT_THRESHOLD_C) / 100.0


def idle_co2_kg(queue_seconds: float, vehicle_class: str, fuel_type: str, emission_norm: str) -> float:
    if fuel_type == "electric" or queue_seconds <= 0:
        return 0.0
    return queue_seconds * IDLE_KG_PER_S.get(vehicle_class, 0.00075) * BS_IDLE_MULT.get(emission_norm, 1.0)


def _plate_text(plate: str) -> str:
    """KA01MJ4821 -> KA-01-MJ-4821 and 22BH1234AA -> 22-BH-1234-AA for log lines; others unchanged."""
    p = vehicle_registry.normalize_plate(plate)
    if len(p) >= 9 and p[:2].isdigit() and p[2:4] == "BH":
        return f"{p[:2]}-BH-{p[4:8]}-{p[8:]}"
    if len(p) >= 8 and p[:2].isalpha() and p[2:4].isdigit():
        return f"{p[:2]}-{p[2:4]}-{p[4:-4]}-{p[-4:]}"
    return p


def refuel_event(track: Dict[str, Any], reg: Dict[str, Any], temp_c: Optional[float]) -> Optional[Dict[str, Any]]:
    """Refuel event for a track that passed the stationary-in-bay trigger, else None."""
    if track.get("refuel_trigger_frame") is None:
        return None
    cls = track.get("vehicle_class") or reg["vehicle_class"]
    fuel, norm = reg["fuel_type"], reg["emission_norm"]
    method = method_for(track.get("track_id"))
    tag, label = METHODS[method]
    if fuel == "electric":
        litres, co2, voc = 0.0, 0.0, 0.0
    else:
        litres = litres_for(method, cls, float(track.get("estimated_dwell_seconds") or 0.0))
        co2 = litres * CO2_PER_L.get(fuel, 2.31)
        voc = litres * VOC_G_PER_L.get(fuel, 0.0) / 1000.0 * voc_factor(temp_c)
    who = _plate_text(reg["plate"]) if method == "C" else (
        _plate_text(track["license_plate"]) if track.get("license_plate") else f"Track #{track.get('track_id')} {cls}")
    fuel_txt = f"{fuel.capitalize()} {SHORT_NORM.get(norm, norm)}"
    if fuel == "electric":
        outcome = "EV, no fuel dispensed"
    elif litres <= 0:
        outcome = f"fill under way, bay time too short to estimate litres yet ({label})"
    else:
        outcome = f"{litres:.1f}L dispensed -> {co2:.1f} kg CO2"
    log = f" [{tag}] {who} ({fuel_txt}) in Bay_01 -> {outcome}"
    return {"method": method, "method_label": label, "litres": round(litres, 2), "fuel_type": fuel,
            "emission_norm": norm, "co2_fuel_kg": round(co2, 3), "voc_kg": round(voc, 4), "log": log}


def enrich_summary(summary: Dict[str, Any], temp_c: Optional[float] = None) -> Dict[str, Any]:
    """Adds per-track 'forecourt' data and clip totals ('forecourt') to a phase-1 summary dict, in place."""
    tracks: List[Dict[str, Any]] = summary.get("tracked_vehicles") or []
    by_method = {"A": 0, "B": 0, "C": 0}
    litres_by_fuel: Dict[str, float] = {}
    co2_fuel = co2_idle = voc = 0.0
    bay, queue = [], []
    for t in tracks:
        reg = vehicle_registry.lookup(t.get("license_plate"), t.get("vehicle_class"), key=f"track-{t.get('track_id')}")
        q_s = float(t.get("queue_idle_seconds") or 0.0)
        b_s = float(t.get("estimated_dwell_seconds") or 0.0)
        idle = idle_co2_kg(q_s, t.get("vehicle_class") or reg["vehicle_class"], reg["fuel_type"], reg["emission_norm"])
        ev = refuel_event(t, reg, temp_c)
        t["forecourt"] = {"bay_dwell_s": round(b_s, 2), "queue_idle_s": round(q_s, 2), "co2_idle_kg": round(idle, 4),
                          "registry_plate": reg["plate"], "refuel": ev}
        co2_idle += idle
        if b_s > 0:
            bay.append(b_s)
        if q_s > 0:
            queue.append(q_s)
        if ev:
            by_method[ev["method"]] += 1
            litres_by_fuel[ev["fuel_type"]] = round(litres_by_fuel.get(ev["fuel_type"], 0.0) + ev["litres"], 2)
            co2_fuel += ev["co2_fuel_kg"]
            voc += ev["voc_kg"]
    summary["forecourt"] = {
        "data_source": "simulated",
        "notice": DATA_NOTICE,
        "refuels": sum(by_method.values()),
        "refuels_by_method": by_method,
        "litres_by_fuel": litres_by_fuel,
        "litres_total": round(sum(litres_by_fuel.values()), 2),
        "co2_fuel_kg": round(co2_fuel, 3),
        "co2_idle_kg": round(co2_idle, 4),
        "voc_kg": round(voc, 4),
        "temp_c": temp_c,
        "voc_penalty_pct": round((voc_factor(temp_c) - 1.0) * 100.0, 1),
        "vehicles_in_bay": len(bay),
        "vehicles_in_queue": len(queue),
        "avg_bay_dwell_s": round(sum(bay) / len(bay), 1) if bay else None,
        "avg_queue_idle_s": round(sum(queue) / len(queue), 1) if queue else None,
    }
    return summary


def current_temp_c(lat: float = 13.03, lon: float = 77.52, timeout: float = 4.0) -> float:
    """Live air temperature from Open-Meteo for the VOC penalty; falls back to the default offline."""
    try:
        import httpx

        r = httpx.get("https://api.open-meteo.com/v1/forecast",
                      params={"latitude": lat, "longitude": lon, "current": "temperature_2m"}, timeout=timeout)
        return float(r.json()["current"]["temperature_2m"])
    except Exception as e:  # offline venue, rate limit, schema change
        logger.info(f"Open-Meteo temperature unavailable ({e}); using {DEFAULT_TEMP_C} C")
        return DEFAULT_TEMP_C
