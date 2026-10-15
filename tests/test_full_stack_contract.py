import pytest
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import numpy as np
import cv2

# Define the exact contract your frontend expects
class EnvironmentContext(BaseModel):
    temp_c: float
    wind_kmh: float
    wind_deg: float
    aqi: int

class CVTelemetry(BaseModel):
    active_counts: Dict[str, int] = {}
    bay_dwell_sec: Optional[float] = None
    queue_idle_sec: Optional[float] = None
    smoke_opacity_pct: Optional[float] = None
    recent_logs: List[str]

class EmissionsSummary(BaseModel):
    baseline_co2_kg_hr: float
    waste_co2_kg_hr: float

class PlatformStatePayload(BaseModel):
    node_id: str
    sector: str
    timestamp: str
    environment: EnvironmentContext
    cv_telemetry: CVTelemetry
    emissions: EmissionsSummary

def test_json_payload_schema():
    """Validates that a sample backend payload conforms to what the UI expects."""
    sample_payload = {
        "node_id": "BLR_STATION_01",
        "sector": "vehicular",
        "timestamp": "2026-10-15T11:00:00Z",
        "environment": {
            "temp_c": 28.0,
            "wind_kmh": 12.0,
            "wind_deg": 180.0,
            "aqi": 140
        },
        "cv_telemetry": {
            "active_counts": {"car": 3, "truck": 1},
            "bay_dwell_sec": 45.0,
            "queue_idle_sec": 20.0,
            "smoke_opacity_pct": None,
            "recent_logs": ["Vehicle #1 refueling"]
        },
        "emissions": {
            "baseline_co2_kg_hr": 120.5,
            "waste_co2_kg_hr": 25.0
        }
    }
    # If the schema raises an error, CI fails automatically
    validated = PlatformStatePayload(**sample_payload)
    assert validated.sector == "vehicular"
    assert validated.emissions.baseline_co2_kg_hr > 0


def test_industrial_emissions_engine():
    """Tests the Industrial Carbon Math Engine calculations."""
    from modules.industrial_detector import calculate_industrial_emissions, calculate_ringelmann_opacity

    # Baseline dirty coal with 1.6x penalty
    calc = calculate_industrial_emissions(
        fuel_rate_kg_hr=1245.0,
        fuel_type="coal",
        opacity_multiplier=1.60,
        scrubber_efficiency=0.0,
    )
    assert calc["baseline_co2_kg_hr"] == 4820.6
    assert calc["waste_co2_kg_hr"] == 1807.7

    # Test scrubber retrofit (85% efficiency)
    scrub = calculate_industrial_emissions(
        fuel_rate_kg_hr=1245.0,
        fuel_type="coal",
        opacity_multiplier=1.60,
        scrubber_efficiency=0.85,
    )
    assert scrub["baseline_co2_kg_hr"] == 723.1
    assert scrub["waste_co2_kg_hr"] == 0.0

    # Test Ringelmann Opacity calculation on black vs white ROI
    black_roi = np.zeros((50, 50, 3), dtype=np.uint8)
    opacity, level, mult, status = calculate_ringelmann_opacity(black_roi)
    assert opacity == 100.0
    assert level == 5
    assert mult == 1.60
    assert status == "SCRUBBER VIOLATION"

    white_roi = np.ones((50, 50, 3), dtype=np.uint8) * 255
    opacity_w, level_w, mult_w, status_w = calculate_ringelmann_opacity(white_roi)
    assert opacity_w == 0.0
    assert level_w == 0
    assert mult_w == 1.0
    assert status_w == "NORMAL"

    # Test True Masked Ringelmann Opacity: white sky with a black plume inside mask
    sky_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255  # 100% white sky
    plume_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(plume_mask, (50, 50), 25, 255, -1)  # circular plume mask
    sky_frame[plume_mask > 0] = 0  # 100% black smoke inside mask

    # Masked calculation should evaluate ONLY the smoke pixels (100% opacity)
    op_m, lvl_m, mult_m, stat_m = calculate_ringelmann_opacity(sky_frame, mask=plume_mask)
    assert op_m == 100.0
    assert lvl_m == 5
    assert mult_m == 1.60
    assert stat_m == "SCRUBBER VIOLATION"


def test_api_health():
    """Validates FastAPI system health and component diagnostics."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "industrial_module" in data
    assert "vehicular_module" in data
    assert data["industrial_module"]["live"] is True
    assert data["vehicular_module"]["live"] is True


def test_api_telemetry_endpoints():
    """Validates real-time circular telemetry streaming for industrial and vehicular sectors."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. Industrial telemetry stream
    r_ind = client.get("/api/telemetry?sector=industrial")
    assert r_ind.status_code == 200
    data_ind = r_ind.json()
    assert data_ind["sector"] == "industrial"
    assert "emissions" in data_ind
    assert "cv_telemetry" in data_ind
    assert "environment" in data_ind

    # 2. Vehicular telemetry stream (Open-Receiver Fallback)
    r_veh = client.get("/api/telemetry?sector=vehicular")
    assert r_veh.status_code == 200
    data_veh = r_veh.json()
    assert data_veh["sector"] == "vehicular"
    assert "active_counts" in data_veh["cv_telemetry"]
    assert data_veh["emissions"]["baseline_co2_kg_hr"] > 0

    # 3. Dual-node simultaneous fetch (sector=all)
    r_all = client.get("/api/telemetry?sector=all")
    assert r_all.status_code == 200
    data_all = r_all.json()
    assert "industrial" in data_all
    assert "vehicular" in data_all
    assert data_all["industrial"]["sector"] == "industrial"
    assert data_all["vehicular"]["sector"] == "vehicular"


def test_api_simulation_endpoints():
    """Validates What-If Carbon math and financial savings in INR for both sectors."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. Industrial What-If Simulation: Wet Scrubber Retrofit (85% efficiency)
    sim_ind = client.post("/api/simulate", json={
        "sector": "industrial",
        "fuel_type": "coal",
        "fuel_rate_kg_hr": 1245.0,
        "opacity_multiplier": 1.6,
        "scrubber_efficiency_pct": 85.0,
        "combustion_tuning_pct": 0.0,
    })
    assert sim_ind.status_code == 200
    res_ind = sim_ind.json()
    assert res_ind["simulation"]["active"] is True
    assert res_ind["simulation"]["projected_co2_kg_hr"] == 723.1
    assert res_ind["simulation"]["financial_savings"]["inr_per_hour"] > 0
    assert res_ind["simulation"]["financial_savings"]["currency"] == "INR (₹)"

    # 2. Vehicular What-If Simulation: 40% EV Adoption + E20 Ethanol Blend
    sim_veh = client.post("/api/simulate", json={
        "sector": "vehicular",
        "ev_adoption_pct": 40.0,
        "e20_blend_pct": 20.0,
        "queue_optimization_pct": 50.0,
        "ban_bs3_bs4_heavy": True,
    })
    assert sim_veh.status_code == 200
    res_veh = sim_veh.json()
    assert res_veh["simulation"]["active"] is True
    assert res_veh["simulation"]["reduction_pct"] > 0
    assert res_veh["simulation"]["financial_savings"]["annual_inr"] > 0


def test_api_ai_advice_endpoint():
    """Validates GenAI Sustainability Copilot recommendation schema."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. Industrial Advice
    adv_ind = client.post("/api/ai-advice", json={"sector": "industrial"})
    assert adv_ind.status_code == 200
    res_ind = adv_ind.json()
    assert "headline" in res_ind
    assert len(res_ind["immediate_actions"]) == 3
    assert "capex_roi_estimate" in res_ind
    assert "regulatory_status" in res_ind

    # 2. Vehicular Advice
    adv_veh = client.post("/api/ai-advice", json={"sector": "vehicular"})
    assert adv_veh.status_code == 200
    res_veh = adv_veh.json()
    assert "headline" in res_veh
    assert len(res_veh["immediate_actions"]) == 3
    assert "capex_roi_estimate" in res_veh


def test_api_video_streaming():
    """Validates video streaming endpoints for both generated and ungenerated assets."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Industrial video is generated and should return 200
    r_ind = client.get("/api/video/industrial")
    assert r_ind.status_code == 200
    assert r_ind.headers["content-type"] == "video/mp4"

    # Vehicular video is not yet generated and should return clean 404
    r_veh = client.get("/api/video/vehicular")
    assert r_veh.status_code == 404

    # Invalid sector returns 400
    r_inv = client.get("/api/video/unknown_sector")
    assert r_inv.status_code == 400
