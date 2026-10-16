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

    # Vehicular video (returns 200 if generated, or clean 404 if not yet generated)
    r_veh = client.get("/api/video/vehicular")
    assert r_veh.status_code in [200, 404]
    if r_veh.status_code == 200:
        assert r_veh.headers["content-type"] == "video/mp4"

    # Invalid sector returns 400
    r_inv = client.get("/api/video/unknown_sector")
    assert r_inv.status_code == 400


def test_api_digital_twin_endpoints():
    """Validates Digital Twin simulation & state endpoints."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. Test GET /api/twin/state
    r_state = client.get("/api/twin/state")
    assert r_state.status_code == 200
    data_state = r_state.json()
    assert "baseline_summary" in data_state
    assert "projected_summary" in data_state
    assert "comparative_chart_metrics" in data_state
    assert "financial_savings" in data_state
    assert "cpcb_compliance" in data_state
    assert "scene_3d_industrial" in data_state
    assert "scene_3d_vehicular" in data_state

    # 2. Test POST /api/twin/simulate with multi-parameter interventions
    r_sim = client.post("/api/twin/simulate", json={
        "ev_adoption_pct": 40.0,
        "idle_time_cut_pct": 50.0,
        "e20_blend_pct": 20.0,
        "ban_bs3_bs4_heavy": True,
        "scrubber_efficiency_pct": 85.0,
        "fuel_type": "biomass",
        "combustion_tuning_pct": 80.0,
    })
    assert r_sim.status_code == 200
    sim_data = r_sim.json()
    assert sim_data["projected_summary"]["total_avoided_co2_kg_hr"] > 0
    assert sim_data["projected_summary"]["total_reduction_pct"] > 50.0
    assert sim_data["financial_savings"]["annual_inr"] > 10000000.0  # > ₹1 Crore annual carbon savings
    assert sim_data["scene_3d_industrial"]["scrubber_active"] is True
    assert sim_data["scene_3d_industrial"]["plume_color_hex"] == "#e2e8f0"  # Clean white steam
    assert len(sim_data["comparative_chart_metrics"]) == 3
    assert len(sim_data["cpcb_compliance"]) == 2
    assert sim_data["cpcb_compliance"][0]["status"] == "COMPLIANT"


def test_digital_twin_physics_engine():
    """Validates baseline extraction fallbacks and calculation mechanics."""
    from modules.digital_twin import simulate_digital_twin, DigitalTwinParams

    res_default = simulate_digital_twin()
    assert res_default.baseline_summary["total_baseline_co2_kg_hr"] > 0
    assert res_default.financial_savings.hourly_inr == 0.0  # Zero intervention = zero avoided carbon

    # Fuel switch to biomass alone cuts industrial baseline by >80%
    res_biomass = simulate_digital_twin(DigitalTwinParams(fuel_type="biomass"))
    ind_metric = [m for m in res_biomass.comparative_chart_metrics if m.category == "Industrial Smokestack"][0]
    assert ind_metric.reduction_pct > 75.0


def test_saved_twin_runs_endpoints():
    """Validates the Saved Digital Twin Runs persistence bridge and auto-seeding."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. GET /api/twin/saved (auto-seeds default flagship run if empty)
    r_get = client.get("/api/twin/saved")
    assert r_get.status_code == 200
    runs = r_get.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1
    assert "run_id" in runs[0]
    assert "params" in runs[0]
    assert "projected_summary" in runs[0]

    # 2. POST /api/twin/save (saves new scenario)
    r_save = client.post("/api/twin/save", json={
        "params": {
            "ev_adoption_pct": 50.0,
            "scrubber_efficiency_pct": 90.0,
            "fuel_type": "biomass",
            "combustion_tuning_pct": 75.0,
        },
        "label": "Test Scenario: 90% Scrubber + Biomass"
    })
    assert r_save.status_code == 200
    save_data = r_save.json()
    assert save_data["success"] is True
    assert "saved_run" in save_data
    assert save_data["saved_run"]["label"] == "Test Scenario: 90% Scrubber + Biomass"
    assert save_data["total_saved_runs"] >= 2


def test_compliance_report_endpoints():
    """Validates CPCB Form-V & SEBI BRSR compliance report generation & export endpoints."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. GET /api/report/summary (React tab JSON payload)
    r_sum = client.get("/api/report/summary")
    assert r_sum.status_code == 200
    summary = r_sum.json()
    assert "audit_id" in summary
    assert summary["audit_id"].startswith("CPCB-BRSR-")
    assert "facility_metadata" in summary
    assert "executive_kpis" in summary
    assert "sebi_brsr_breakdown" in summary
    assert "scope_1_direct_kg_hr" in summary["sebi_brsr_breakdown"]
    assert "scope_3_downstream_kg_hr" in summary["sebi_brsr_breakdown"]
    assert "node_1_vehicular_forecourt" in summary
    assert "node_2_industrial_smokestack" in summary
    assert "carbon_economics_inr" in summary
    assert summary["carbon_economics_inr"]["annual_inr"] > 0
    assert summary["carbon_economics_inr"]["trees_equivalent_annual"] > 0
    assert summary["saved_scenarios_count"] >= 1

    # 2. GET /api/report/export?format=html (Print-ready certificate)
    r_html = client.get("/api/report/export?format=html")
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers.get("content-type", "")
    html_content = r_html.text
    assert "Central Pollution Control Board" in html_content
    assert "SEBI BRSR" in html_content
    assert "window.print()" in html_content
    assert "CPCB-BRSR-" in html_content
    assert "Environmental Compliance Audit Certificate" in html_content

    # 3. GET /api/report/export?format=json (JSON export)
    r_json = client.get("/api/report/export?format=json")
    assert r_json.status_code == 200
    json_data = r_json.json()
    assert json_data["audit_id"] == summary["audit_id"]


