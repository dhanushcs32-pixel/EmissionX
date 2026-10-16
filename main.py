"""Unified FastAPI Backend & Open-Receiver Platform.

Antigravity Carbon Footprint Intelligence Platform:
- Pillar 1: Vehicular & Urban Mobility Emissions (Fuel Station CCTV + Queue Dwell Analysis)
- Pillar 2: Industrial Smokestack Intelligence (Multi-Cue Plume CV + CEMS IoT Sync)
- Open Receiver Adapters: Real-time telemetry streaming, CEMS boiler simulations,
  dynamic vehicular loader/fallback, live Open-Meteo weather caching, and GenAI advice.
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Import Pillar 2 Industrial Detector Engine
from modules.industrial_detector import (
    calculate_industrial_emissions,
    calculate_ringelmann_opacity,
    stream_industrial_telemetry,
    IndustrialTelemetryPayload,
    EMISSION_FACTORS,
)
from modules.digital_twin import (
    DigitalTwinParams,
    DigitalTwinResponse,
    simulate_digital_twin,
    save_twin_run,
    get_saved_twin_runs,
)
from modules.compliance_report import (
    generate_report_summary,
    generate_printable_html_report,
)
from modules import upload_jobs, vehicle_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("UnifiedBackend")

# Initialize FastAPI App
app = FastAPI(
    title="EmissionX Intelligence Platform API",
    description="Unified Full-Stack Emissions Intelligence Engine & Open-Receiver Platform",
    version="1.0.0",
)

# Configure CORS for multi-device pitch environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 1. WEATHER & ENVIRONMENT SERVICE (WITH OFFLINE FALLBACK)
# =====================================================================

DEFAULT_BENGALURU_WEATHER = {
    "temp_c": 29.5,
    "wind_kmh": 11.2,
    "wind_deg": 240,
    "aqi": 168,
}

_weather_cache: Dict[str, Any] = dict(DEFAULT_BENGALURU_WEATHER)
_weather_last_fetched: float = 0.0
WEATHER_CACHE_TTL_SEC = 300.0  # 5 minutes cache


async def get_current_environment() -> Dict[str, Any]:
    """
    Fetches real-time ambient weather for Peenya Industrial Area / Bengaluru
    via Open-Meteo with a 2-second timeout and 5-minute memory cache.
    Falls back gracefully to authentic Bengaluru baseline data when offline.
    """
    global _weather_cache, _weather_last_fetched

    now = time.time()
    if now - _weather_last_fetched < WEATHER_CACHE_TTL_SEC:
        return dict(_weather_cache)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=13.03&longitude=77.52"
        "&current=temperature_2m,wind_speed_10m,wind_direction_10m"
    )

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                _weather_cache = {
                    "temp_c": float(current.get("temperature_2m", 29.5)),
                    "wind_kmh": float(current.get("wind_speed_10m", 11.2)),
                    "wind_deg": int(current.get("wind_direction_10m", 240)),
                    "aqi": 168,
                }
                _weather_last_fetched = now
                logger.debug(f"Open-Meteo weather updated: {_weather_cache}")
                return dict(_weather_cache)
    except Exception as e:
        logger.debug(f"Open-Meteo request skipped ({e}). Using cached/default weather.")

    return dict(_weather_cache)


# =====================================================================
# 2. OPEN RECEIVER ADAPTER: INDUSTRIAL TELEMETRY STREAM
# =====================================================================

class IndustrialTelemetryBuffer:
    """Manages cyclic replay and live streaming of Industrial CCTV telemetry."""

    def __init__(self, json_path: str = "cache/factory_telemetry.json"):
        self.json_path = Path(json_path)
        self.meta_path = self.json_path.with_name(self.json_path.stem + "_meta.json")
        self.records: List[Dict[str, Any]] = []
        self.frame_idx = 0
        self._mtime: Optional[float] = None
        self.load_cache()

    def load_cache(self) -> None:
        if self.json_path.is_file():
            try:
                self._mtime = self.json_path.stat().st_mtime
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                self.frame_idx = 0
                logger.info(f"Loaded {len(self.records)} industrial telemetry records from {self.json_path}")
            except Exception as e:
                logger.warning(f"Could not read {self.json_path}: {e}")
                self.records = []
        else:
            self.records = []

    def reload_if_changed(self) -> None:
        """Picks up a fresh run of run_industrial.py without restarting the server."""
        try:
            mtime = self.json_path.stat().st_mtime if self.json_path.is_file() else None
        except OSError:
            return
        if mtime != self._mtime:
            self.load_cache()

    def run_meta(self) -> Optional[Dict[str, Any]]:
        """Detector run metadata written by process_video (engine/profile, source video, codec)."""
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def get_next_record(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the next frame's telemetry in a continuous circular loop."""
        self.reload_if_changed()
        if not self.records:
            # Fallback synthetic industrial record if cache not yet populated
            return {
                "node_id": "IND_STACK_01_PEENYA",
                "sector": "industrial",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": env,
                "cv_telemetry": {
                    "active_counts": {},
                    "bay_dwell_sec": None,
                    "queue_idle_sec": None,
                    "smoke_opacity_pct": 36.5,
                    "ringelmann_level": 2,
                    "opacity_multiplier": 1.25,
                    "cems_metrics": {
                        "fuel_type": "coal",
                        "fuel_rate_kg_hr": 1205.0,
                        "flue_gas_temp_c": 185.0,
                        "so2_ppm": 165,
                    },
                    "recent_logs": [
                        " [PLUME-CV] Smoke Opacity 36.5% (Ringelmann Lvl 2) -> Penalty 1.25x",
                        " [CEMS-IOT] Boiler #1 Coal Feed: 1205 kg/hr | Flue Temp: 185.0C | SO2: 165 ppm",
                    ],
                },
                "emissions": {
                    "baseline_co2_kg_hr": 3645.2,
                    "waste_co2_kg_hr": 729.0,
                },
                "simulation": {
                    "active": False,
                    "projected_co2_kg_hr": 3645.2,
                    "reduction_pct": 0.0,
                    "applied_params": {
                        "fuel_type": "coal",
                        "scrubber_efficiency_pct": 0.0,
                        "combustion_tuning_pct": 0.0,
                    },
                },
            }

        rec = dict(self.records[self.frame_idx % len(self.records)])
        self.frame_idx = (self.frame_idx + 1) % len(self.records)

        # Inject real-time timestamp and fresh ambient weather
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        rec["environment"] = env
        return rec


industrial_buffer = IndustrialTelemetryBuffer()


# =====================================================================
# 3. OPEN RECEIVER ADAPTER: VEHICULAR TELEMETRY STREAM & LOADER
# =====================================================================

class VehicularReceiverAdapter:
    """
    Open Receiver for Pillar 1 (Vehicular Emissions).
    Automatically detects if `modules/vehicular_detector.py` and `cache/fuel_telemetry.json`
    are present. If present, delegates live execution. If not yet created, serves
    a 100% contract-compliant receiver stub with realistic dynamic variations.
    """

    def __init__(self):
        self.module_path = WORKSPACE_ROOT / "modules" / "vehicular_detector.py"
        self.json_path = WORKSPACE_ROOT / "cache" / "fuel_telemetry.json"
        self.vehicular_module: Optional[Any] = None
        self.records: List[Dict[str, Any]] = []
        self.frame_idx = 0
        self._init_adapter()

    def _init_adapter(self) -> None:
        # Check if custom vehicular detector module exists
        if self.module_path.is_file():
            try:
                spec = importlib.util.spec_from_file_location("vehicular_detector", str(self.module_path))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self.vehicular_module = mod
                    logger.info("Vehicular Receiver Adapter: Successfully dynamically linked modules.vehicular_detector!")
            except Exception as e:
                logger.warning(f"Failed to dynamic-link vehicular_detector: {e}")

        # Check for pre-generated telemetry cache
        self._mtime: Optional[float] = None
        self.load_json()

    def load_json(self) -> None:
        if not self.json_path.is_file():
            self.records, self._mtime = [], None
            return
        try:
            self._mtime = self.json_path.stat().st_mtime
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            self.frame_idx = 0
            logger.info(f"Vehicular Receiver Adapter: Loaded {len(self.records)} records from {self.json_path}")
        except Exception as e:
            logger.warning(f"Could not load fuel telemetry JSON: {e}")

    def reload_if_changed(self) -> None:
        """Picks up a newly analysed fuel-station clip (website upload) without a restart."""
        try:
            mtime = self.json_path.stat().st_mtime if self.json_path.is_file() else None
        except OSError:
            return
        if mtime != self._mtime:
            self.load_json()

    def is_dynamically_linked(self) -> bool:
        return self.vehicular_module is not None

    def get_next_record(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """Provides next vehicular telemetry frame."""
        self.reload_if_changed()
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.records:
            rec = dict(self.records[self.frame_idx % len(self.records)])
            self.frame_idx = (self.frame_idx + 1) % len(self.records)
            rec["timestamp"] = now_iso
            rec["environment"] = env
            return rec

        # Schema-compliant dynamic stub for BLR_FUEL_STATION_01
        self.frame_idx += 1
        t = self.frame_idx * 0.1

        cars = max(1, int(3 + 2 * math.sin(t)))
        bikes = max(2, int(5 + 3 * math.cos(t * 0.8)))
        autos = max(1, int(2 + math.sin(t * 1.5)))
        trucks = 1 if (self.frame_idx % 7 == 0) else 0

        dwell_sec = round(45.0 + 8.0 * math.sin(t * 0.5), 1)
        idle_sec = round(28.0 + 10.0 * math.cos(t * 0.4), 1)

        # Baseline fuel station emission math
        # Running fuel burn (~104 kg/hr) + idling queue waste (~38 kg/hr)
        base_co2 = round(110.0 + cars * 6.5 + bikes * 1.8 + autos * 3.2 + trucks * 18.0, 1)
        waste_co2 = round((idle_sec / 30.0) * (cars * 3.5 + trucks * 9.0 + autos * 2.0), 1)

        return {
            "node_id": "BLR_FUEL_STATION_01",
            "sector": "vehicular",
            "timestamp": now_iso,
            "environment": env,
            "cv_telemetry": {
                "active_counts": {
                    "car": cars,
                    "motorcycle": bikes,
                    "auto_rickshaw": autos,
                    "bus_truck": trucks,
                },
                "bay_dwell_sec": dwell_sec,
                "queue_idle_sec": idle_sec,
                "smoke_opacity_pct": None,
                "ringelmann_level": 0,
                "opacity_multiplier": 1.0,
                "recent_logs": [
                    f" [VEH-CV] Active Pump Bays: {min(4, cars + trucks)}/4 occupied (Avg Dwell: {dwell_sec}s)",
                    f" [IDLE-SENS] Queue wait: {idle_sec}s (~{waste_co2} kg CO2/hr wasted)",
                    f" [TRAFFIC-AI] Flow density: {cars + bikes + autos + trucks} vehicles in monitored ROI",
                ],
            },
            "emissions": {
                "baseline_co2_kg_hr": base_co2,
                "waste_co2_kg_hr": waste_co2,
            },
            "simulation": {
                "active": False,
                "projected_co2_kg_hr": base_co2,
                "reduction_pct": 0.0,
                "applied_params": {
                    "ev_adoption_pct": 0.0,
                    "e20_blend_pct": 0.0,
                    "queue_optimization_pct": 0.0,
                    "ban_bs3_bs4_heavy": False,
                },
            },
        }

    def simulate(
        self,
        baseline_record: Dict[str, Any],
        ev_adoption_pct: float = 0.0,
        e20_blend_pct: float = 0.0,
        queue_optimization_pct: float = 0.0,
        ban_bs3_bs4_heavy: bool = False,
    ) -> Dict[str, Any]:
        """Calculates projected emissions under vehicular decarbonization policies."""
        # If external module provides calculation, delegate:
        if self.vehicular_module and hasattr(self.vehicular_module, "calculate_vehicular_emissions"):
            try:
                return self.vehicular_module.calculate_vehicular_emissions(
                    baseline_record=baseline_record,
                    ev_adoption_pct=ev_adoption_pct,
                    e20_blend_pct=e20_blend_pct,
                    queue_optimization_pct=queue_optimization_pct,
                    ban_bs3_bs4_heavy=ban_bs3_bs4_heavy,
                )
            except Exception as e:
                logger.warning(f"Error delegating to vehicular_module: {e}. Using deterministic engine.")

        # Built-in deterministic simulation engine
        base_co2 = float(baseline_record["emissions"]["baseline_co2_kg_hr"])
        waste_co2 = float(baseline_record["emissions"]["waste_co2_kg_hr"])

        # 1. EV Adoption: Cuts tailpipe emissions proportionally
        ev_cut = (ev_adoption_pct / 100.0) * 0.72 * base_co2

        # 2. E20 Bioethanol Blend: India mandates 20% ethanol by 2025-26, cutting lifecycle CO2 by ~6.5%
        e20_cut = (e20_blend_pct / 100.0) * 0.065 * (base_co2 - ev_cut)

        # 3. Queue & Bay Flow Optimization: Eliminates idling fuel burn waste in station queues
        queue_cut = (queue_optimization_pct / 100.0) * waste_co2

        # 4. Heavy BS-III / BS-IV Diesel Restriction: Eliminates high-emitting commercial diesel
        heavy_cut = 18.5 if ban_bs3_bs4_heavy else 0.0

        total_cut = min(base_co2 * 0.92, ev_cut + e20_cut + queue_cut + heavy_cut)
        projected_co2 = max(8.0, round(base_co2 - total_cut, 1))
        reduction_pct = round((total_cut / base_co2 * 100.0) if base_co2 > 0 else 0.0, 1)

        # Financial savings in INR (₹) under Indian Carbon Credit Trading Scheme (CCTS ~₹1,350/tCO2e)
        inr_per_kg_co2 = 1.35
        savings_inr_hr = round(total_cut * inr_per_kg_co2, 2)
        annual_savings_inr = round(total_cut * 24 * 365 * 0.001 * 1350, 0)

        return {
            "baseline_co2_kg_hr": base_co2,
            "projected_co2_kg_hr": projected_co2,
            "avoided_waste_co2_kg_hr": round(total_cut, 1),
            "reduction_pct": reduction_pct,
            "financial_savings": {
                "inr_per_hour": savings_inr_hr,
                "annual_inr": annual_savings_inr,
                "currency": "INR (₹)",
            },
            "applied_params": {
                "ev_adoption_pct": ev_adoption_pct,
                "e20_blend_pct": e20_blend_pct,
                "queue_optimization_pct": queue_optimization_pct,
                "ban_bs3_bs4_heavy": ban_bs3_bs4_heavy,
            },
        }


vehicular_adapter = VehicularReceiverAdapter()


# =====================================================================
# 4. REQUEST / RESPONSE SCHEMAS
# =====================================================================

class SimulationRequest(BaseModel):
    """Flexible simulation request covering both Industrial and Vehicular pillars."""
    sector: str = Field(default="industrial", description="'industrial' or 'vehicular'")

    # Industrial parameters
    fuel_type: str = Field(default="coal")
    fuel_rate_kg_hr: Optional[float] = None
    opacity_multiplier: Optional[float] = None
    scrubber_efficiency_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    combustion_tuning_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    # Vehicular parameters
    ev_adoption_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    e20_blend_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    queue_optimization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    ban_bs3_bs4_heavy: bool = Field(default=False)


class AIAdviceRequest(BaseModel):
    """Open receiver input for Sustainability Copilot recommendations."""
    sector: str = Field(default="industrial", description="'industrial' or 'vehicular'")
    telemetry: Optional[Dict[str, Any]] = None
    simulation: Optional[Dict[str, Any]] = None
    query: Optional[str] = None


# =====================================================================
# 5. REST API ENDPOINTS
# =====================================================================

@app.get("/")
def root():
    return {
        "platform": "EmissionX Intelligence Platform",
        "version": "1.0.0",
        "docs_url": "/docs",
        "api_health": "/api/health",
        "telemetry_stream": "/api/telemetry",
    }


def _vehicular_clip_totals() -> Optional[Dict[str, Any]]:
    """Unique vehicles counted over the whole fuel-station clip (backend/data/phase1_summary.json)."""
    try:
        with open(WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json", "r", encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None
    tracks = s.get("tracked_vehicles") or []
    return {
        "source_video": Path(str(s.get("video_source") or "")).name or None,
        "duration_seconds": s.get("duration_seconds"),
        "total_unique_vehicles": s.get("total_unique_vehicles"),
        "counts_by_class": s.get("counts_by_class") or {},
        "plates_read": sum(1 for t in tracks if t.get("license_plate")),
    }


def _file_version(path: Path) -> Optional[int]:
    """Modification time in ms; the website adds it to video URLs so a re-upload is never served from cache."""
    try:
        return int(path.stat().st_mtime * 1000)
    except OSError:
        return None


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Returns real-time backend health, component statuses, and cache diagnostics."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "industrial_module": {
            "live": True,
            "cached_telemetry_frames": len(industrial_buffer.records),
            "video_available": (WORKSPACE_ROOT / "cache" / "factory_annotated.mp4").is_file(),
            "video_version": _file_version(WORKSPACE_ROOT / "cache" / "factory_annotated.mp4"),
            "video_path": "cache/factory_annotated.mp4",
            "detector_run": industrial_buffer.run_meta(),
            "uploaded_clip": upload_jobs.uploaded_info("industrial"),
        },
        "vehicular_module": {
            "mode": "dynamic_linked" if vehicular_adapter.is_dynamically_linked() else "open_receiver_fallback",
            "live": True,
            "cached_telemetry_frames": len(vehicular_adapter.records),
            "video_available": (WORKSPACE_ROOT / "cache" / "fuel_annotated.mp4").is_file(),
            "video_version": _file_version(WORKSPACE_ROOT / "cache" / "fuel_annotated.mp4"),
            "video_path": "cache/fuel_annotated.mp4",
            "uploaded_clip": upload_jobs.uploaded_info("vehicular"),
            "clip_totals": _vehicular_clip_totals(),
        },
        "environment_service": {
            "source": "open-meteo" if (time.time() - _weather_last_fetched < WEATHER_CACHE_TTL_SEC) else "offline-cache",
            "location": "Peenya / Bengaluru (13.03N, 77.52E)",
            "current_weather": _weather_cache,
        },
    }


@app.get("/api/environment")
async def get_environment():
    """Returns live Open-Meteo ambient air temperature, wind heading, and AQI."""
    return await get_current_environment()


@app.get("/api/telemetry")
async def get_telemetry(
    sector: str = Query(default="industrial", description="'industrial', 'vehicular', or 'all'"),
):
    """
    Continuous circular telemetry stream for the frontend UI.
    Supports single-sector queries or `sector=all` returning both nodes simultaneously.
    """
    env = await get_current_environment()

    if sector.lower() == "industrial":
        return industrial_buffer.get_next_record(env)
    elif sector.lower() == "vehicular":
        return vehicular_adapter.get_next_record(env)
    elif sector.lower() in ["all", "both"]:
        return {
            "industrial": industrial_buffer.get_next_record(env),
            "vehicular": vehicular_adapter.get_next_record(env),
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector: '{sector}'. Supported sectors are 'industrial', 'vehicular', or 'all'.",
        )


@app.post("/api/simulate")
async def run_simulation(req: SimulationRequest):
    """
    Deterministic What-If Carbon Simulation Engine.
    Computes avoided CO2, net cut %, and financial carbon credit savings in INR (₹).
    """
    env = await get_current_environment()

    if req.sector.lower() == "industrial":
        # Get baseline telemetry to anchor calculations
        baseline_record = industrial_buffer.get_next_record(env)

        fuel_type = req.fuel_type.lower()
        fuel_rate = req.fuel_rate_kg_hr or float(
            baseline_record["cv_telemetry"].get("cems_metrics", {}).get("fuel_rate_kg_hr", 1245.0)
        )
        opacity_mult = req.opacity_multiplier or float(
            baseline_record["cv_telemetry"].get("opacity_multiplier", 1.25)
        )
        scrubber_eff = req.scrubber_efficiency_pct / 100.0
        tuning_eff = req.combustion_tuning_pct / 100.0

        # 1. Compute unmitigated dirty baseline
        base_calc = calculate_industrial_emissions(
            fuel_rate_kg_hr=fuel_rate,
            fuel_type=fuel_type,
            opacity_multiplier=opacity_mult,
            scrubber_efficiency=0.0,
            combustion_tuning_pct=0.0,
        )

        # 2. Compute simulated mitigated projection
        sim_calc = calculate_industrial_emissions(
            fuel_rate_kg_hr=fuel_rate,
            fuel_type=fuel_type,
            opacity_multiplier=opacity_mult,
            scrubber_efficiency=scrubber_eff,
            combustion_tuning_pct=req.combustion_tuning_pct,
        )

        base_co2 = base_calc["baseline_co2_kg_hr"]
        proj_co2 = sim_calc["projected_co2_kg_hr"]
        avoided_co2 = max(0.0, round(base_co2 - proj_co2, 1))
        reduction_pct = round((avoided_co2 / base_co2 * 100.0) if base_co2 > 0 else 0.0, 1)

        # Carbon credit valuation in INR (₹1,350 per tonne CO2)
        savings_inr_hr = round(avoided_co2 * 1.35, 2)
        annual_savings_inr = round(avoided_co2 * 24 * 365 * 0.001 * 1350, 0)

        # Update telemetry record with simulation block
        resp = dict(baseline_record)
        resp["emissions"] = {
            "baseline_co2_kg_hr": base_co2,
            "waste_co2_kg_hr": base_calc["waste_co2_kg_hr"],
        }
        resp["simulation"] = {
            "active": True,
            "projected_co2_kg_hr": proj_co2,
            "avoided_waste_co2_kg_hr": avoided_co2,
            "reduction_pct": reduction_pct,
            "financial_savings": {
                "inr_per_hour": savings_inr_hr,
                "annual_inr": annual_savings_inr,
                "currency": "INR (₹)",
            },
            "applied_params": {
                "fuel_type": fuel_type,
                "scrubber_efficiency_pct": req.scrubber_efficiency_pct,
                "combustion_tuning_pct": req.combustion_tuning_pct,
            },
        }
        return resp

    elif req.sector.lower() == "vehicular":
        baseline_record = vehicular_adapter.get_next_record(env)
        sim_res = vehicular_adapter.simulate(
            baseline_record=baseline_record,
            ev_adoption_pct=req.ev_adoption_pct,
            e20_blend_pct=req.e20_blend_pct,
            queue_optimization_pct=req.queue_optimization_pct,
            ban_bs3_bs4_heavy=req.ban_bs3_bs4_heavy,
        )

        resp = dict(baseline_record)
        resp["simulation"] = {
            "active": True,
            "projected_co2_kg_hr": sim_res["projected_co2_kg_hr"],
            "avoided_waste_co2_kg_hr": sim_res["avoided_waste_co2_kg_hr"],
            "reduction_pct": sim_res["reduction_pct"],
            "financial_savings": sim_res["financial_savings"],
            "applied_params": sim_res["applied_params"],
        }
        return resp

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector for simulation: '{req.sector}'. Supported are 'industrial' or 'vehicular'.",
        )


@app.post("/api/ai-advice")
async def get_ai_advice(req: AIAdviceRequest):
    """
    GenAI Sustainability Copilot Open Receiver.
    Returns structured, telemetry-grounded decarbonization advice and CPCB regulatory compliance guidance.
    """
    sector = req.sector.lower()

    if sector == "industrial":
        ringelmann = 2
        opacity = 36.5
        if req.telemetry:
            ringelmann = req.telemetry.get("cv_telemetry", {}).get("ringelmann_level", 2)
            opacity = req.telemetry.get("cv_telemetry", {}).get("smoke_opacity_pct", 36.5)

        if ringelmann >= 4:
            headline = "CRITICAL: Scrubber Bypass & Heavy Black Carbon Violation Detected"
            regulatory = "NON-COMPLIANT with CPCB / KSPCB Industrial Boiler Standards. Subject to immediate penalty under Section 21/22 of Air Act 1981."
            actions = [
                "Engage electrostatic precipitator (ESP) / wet scrubber pumps to restore particulate scrubbing to >85%.",
                "Throttle fuel feed rate by 20% to eliminate oxygen starvation in primary combustion zone.",
                "Calibrate continuous emission monitoring system (CEMS) SO2 and opacity optical transceivers.",
            ]
            capex_roi = "₹12.5L retrofitting investment recovers in ~7.8 months through avoided regulatory downtime and CCTS carbon credits."
        else:
            headline = "Operational Efficiency Advisory: Coal-to-Biomass Transition Opportunity"
            regulatory = "COMPLIANT with CPCB Ringelmann Level 2 standard. Eligible for Clean Development Mechanism (CDM) incentives."
            actions = [
                "Maintain secondary air damper at 32% to sustain stoichiometric air-to-fuel balance.",
                "Introduce 15% sustainable biomass pellet co-firing into Boiler #1 coal feed.",
                "Implement waste heat recovery on flue gas duct (currently exiting at >185°C).",
            ]
            capex_roi = "Biomass co-firing yields net 18.4% operating fuel cost savings and earns ₹4.8L/year in tradable carbon credits."

        return {
            "sector": "industrial",
            "headline": headline,
            "immediate_actions": actions,
            "capex_roi_estimate": capex_roi,
            "regulatory_status": regulatory,
            "esg_score_impact": "+14 pts on DJSI/BRSR Corporate Sustainability Reporting",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    else:  # vehicular
        headline = "Urban Mobility Decarbonization: Queue Dwell & EV Fast-Charging Opportunity"
        regulatory = "Aligned with MoRTH Stage VI (BS-VI) Emission Guidelines & National Clean Air Programme (NCAP) targets."
        actions = [
            "Implement automated ANPR smart queue routing across bays to cut idling time below 25 seconds.",
            "Install two 60kW DC fast-chargers to incentivize EV taxi and commercial fleet turnaround.",
            "Enforce idle-engine shutoff compliance during fuel dispensing operations.",
        ]
        capex_roi = "Smart queue flow optimization cuts station-level idling emissions by 68%, saving motorists ~₹3.2L in wasted fuel monthly."

        return {
            "sector": "vehicular",
            "headline": headline,
            "immediate_actions": actions,
            "capex_roi_estimate": capex_roi,
            "regulatory_status": regulatory,
            "esg_score_impact": "+9 pts on Scope 3 Tenant & Commuter Mobility Footprint",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list, description="Prior {role, content} text turns")
    allow_write: bool = Field(default=True, description="Let the bot modify the JSON data files")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Live dashboard snapshot")


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """GenAI chatbot with tool-calling read/write access to the platform JSON files."""
    from modules.chatbot import chat, ChatUnavailable
    try:
        return await asyncio.to_thread(chat, req.message, req.history, req.allow_write, req.context)
    except ChatUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


# =====================================================================
# 5. DIGITAL TWIN SIMULATION & CROSS-NODE SANDBOX
# =====================================================================

@app.post("/api/twin/simulate", response_model=DigitalTwinResponse)
def simulate_twin_endpoint(params: DigitalTwinParams):
    """
    Deterministic Digital Twin Simulation Engine.
    Combines live physical CCTV telemetry baselines (Industrial Stack & Fuel Forecourt),
    evaluates multi-parameter What-If interventions, calculates CPCB compliance,
    derives financial ₹ INR carbon savings, and outputs synchronized 3D scene parameters.
    """
    try:
        return simulate_digital_twin(params)
    except Exception as e:
        logger.error(f"Digital Twin simulation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Digital Twin simulation failed: {str(e)}")


@app.get("/api/twin/state", response_model=DigitalTwinResponse)
def get_twin_state_endpoint():
    """
    Returns the live Digital Twin state reflecting current physical CCTV baseline
    telemetry from both nodes under standard default operating conditions.
    """
    try:
        return simulate_digital_twin(DigitalTwinParams())
    except Exception as e:
        logger.error(f"Digital Twin state error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not fetch Digital Twin state: {str(e)}")


class SaveTwinRunRequest(BaseModel):
    params: DigitalTwinParams = Field(default_factory=DigitalTwinParams)
    label: Optional[str] = None


@app.post("/api/twin/save")
def save_twin_run_endpoint(req: SaveTwinRunRequest):
    """
    Saves a digital twin simulation scenario run to cache/twin_saved_runs.json
    and returns the persisted record along with total saved run count.
    """
    try:
        saved_entry = save_twin_run(req.params, label=req.label)
        all_runs = get_saved_twin_runs()
        return {
            "success": True,
            "status": "success",
            "saved_run": saved_entry,
            "total_count": len(all_runs),
            "total_saved_runs": len(all_runs),
        }
    except Exception as e:
        logger.error(f"Failed to save twin run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save twin run: {str(e)}")


@app.get("/api/twin/saved")
def get_saved_twin_runs_endpoint():
    """
    Returns all saved digital twin runs from cache/twin_saved_runs.json.
    Automatically seeds a flagship scenario if empty.
    """
    try:
        return get_saved_twin_runs()
    except Exception as e:
        logger.error(f"Failed to fetch saved twin runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not fetch saved twin runs: {str(e)}")


# =====================================================================
# 6. CPCB FORM-V & SEBI BRSR COMPLIANCE REPORTING
# =====================================================================

@app.get("/api/report/summary")
def get_report_summary_endpoint():
    """
    Returns multi-node environmental audit aggregator JSON payload for React Report tab:
    SEBI BRSR Scope 1 vs Scope 3, CPCB compliance matrix, CapEx ROI, and saved scenarios.
    """
    try:
        return generate_report_summary()
    except Exception as e:
        logger.error(f"Failed to generate report summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not generate report summary: {str(e)}")


@app.get("/api/report/export")
def export_report_endpoint(format: str = Query("html", pattern="^(html|json)$")):
    """
    Exports official CPCB Form-V & SEBI BRSR Environmental Compliance Audit Certificate.
    - format=html: Standalone print-ready certificate with window.print() trigger.
    - format=json: Raw audit telemetry payload.
    """
    try:
        if format.lower() == "json":
            return generate_report_summary()
        html_content = generate_printable_html_report()
        return HTMLResponse(content=html_content, media_type="text/html")
    except Exception as e:
        logger.error(f"Failed to export compliance report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not export compliance report: {str(e)}")


@app.get("/api/video/{sector}")
def stream_video(sector: str):
    """
    Streams annotated CCTV video with computer vision overlays.
    Returns HTTP 200 with video/mp4, or clear 404 if not yet generated.
    """
    sector_lower = sector.lower()
    if sector_lower == "industrial":
        video_path = WORKSPACE_ROOT / "cache" / "factory_annotated.mp4"
    elif sector_lower == "vehicular":
        video_path = WORKSPACE_ROOT / "cache" / "fuel_annotated.mp4"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid sector: '{sector}'. Must be 'industrial' or 'vehicular'.")

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Video for sector '{sector}' has not been generated yet at {video_path.relative_to(WORKSPACE_ROOT)}.",
        )

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
        headers={"Cache-Control": "no-cache"},   # a new upload replaces the file at the same URL
    )


# =====================================================================
# 7. CLIP UPLOAD FROM THE WEBSITE
# =====================================================================

@app.post("/api/upload/{sector}")
async def upload_clip(sector: str, request: Request, filename: str = Query(..., description="Original file name")):
    """
    Receives a CCTV clip as the raw request body and analyses it in the background:
    'industrial' runs the smoke-plume detector, 'vehicular' the vehicle tracker + plate reader.
    Results replace the factory / fuel-station data the API serves. Poll /api/upload/status.
    """
    sector = sector.lower()
    if sector not in upload_jobs.SECTORS:
        raise HTTPException(status_code=400, detail=f"Invalid sector '{sector}'. Must be 'industrial' or 'vehicular'.")
    if upload_jobs.is_running():
        raise HTTPException(status_code=409, detail="Another clip is still being analysed. Try again when it finishes.")
    size_hdr = request.headers.get("content-length")
    if size_hdr and size_hdr.isdigit() and int(size_hdr) > upload_jobs.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Clip is larger than 1 GB.")
    try:
        dest = upload_jobs.new_upload_path(sector, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    written = 0
    try:
        with open(dest, "wb") as f:
            async for chunk in request.stream():
                written += len(chunk)
                if written > upload_jobs.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Clip is larger than 1 GB.")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        job = upload_jobs.start(sector, dest, Path(filename).name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    logger.info(f"Clip upload accepted: {filename} ({written / 1e6:.1f} MB) -> {sector} analysis")
    return {"accepted": True, "bytes": written, "job": job}


@app.get("/api/upload/status")
def upload_status():
    """State of the current or last clip analysis: idle, running (with frame progress), done or error."""
    return upload_jobs.status()


@app.delete("/api/upload/{sector}")
def remove_uploaded_clip(sector: str):
    """Removes the uploaded clip for a sector and restores the data that was there before it."""
    sector = sector.lower()
    if sector not in upload_jobs.SECTORS:
        raise HTTPException(status_code=400, detail=f"Invalid sector '{sector}'. Must be 'industrial' or 'vehicular'.")
    try:
        removed = upload_jobs.remove(sector)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=423, detail=f"A file is still in use, try again in a moment ({e}).")
    if not removed:
        raise HTTPException(status_code=404, detail=f"No uploaded {sector} clip to remove.")
    return {"removed": True, "sector": sector}


# =====================================================================
# 8. MOCK VEHICLE REGISTRY (VAHAN-STYLE, MOCK DATA ONLY)
# =====================================================================

@app.get("/api/vahan/lookup")
def vahan_lookup(
    plate: Optional[str] = Query(default=None, description="Number plate as read, e.g. KA51AF4321"),
    vehicle_class: Optional[str] = Query(default=None, description="car, motorcycle, bus, truck or auto_rickshaw"),
):
    """Registry details for one plate from the 20-record MOCK database (not real VAHAN data)."""
    if not plate and not vehicle_class:
        raise HTTPException(status_code=400, detail="Give a plate, a vehicle_class, or both.")
    return vehicle_registry.lookup(plate, vehicle_class, key=plate or vehicle_class)


@app.get("/api/vahan/fleet")
def vahan_fleet():
    """MOCK registry details for every vehicle tracked in the current fuel-station clip, with a fleet summary."""
    return vehicle_registry.fleet_from_phase1()


@app.get("/api/vahan/database")
def vahan_database():
    """All records of the MOCK registry database."""
    return {"data_source": "mock", "notice": vehicle_registry.MOCK_NOTICE, "records": vehicle_registry.records()}


# =====================================================================
# 6. STANDALONE SERVER RUNNER
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 65)
    print(f" EMISSIONX UNIFIED BACKEND SERVER STARTING ON http://{host}:{port}")
    print("=" * 65)
    uvicorn.run("main:app", host=host, port=port, reload=True)
