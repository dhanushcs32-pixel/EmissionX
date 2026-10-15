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
