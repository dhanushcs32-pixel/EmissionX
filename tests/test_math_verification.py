import math
import pytest
from modules.industrial_detector import (
    calculate_industrial_emissions,
    opacity_to_ringelmann,
    EMISSION_FACTORS,
)
from modules.digital_twin import DigitalTwinParams, simulate_digital_twin


def test_tc01_ringelmann_opacity_tiers():
    """TC-01: Verify optical opacity (%) maps to exact Ringelmann grades and penalty multipliers."""
    lvl_a, mult_a = opacity_to_ringelmann(14.5)
    assert lvl_a == 1 and math.isclose(mult_a, 1.00, rel_tol=1e-3)

    lvl_b, mult_b = opacity_to_ringelmann(36.0)
    assert lvl_b == 2 and math.isclose(mult_b, 1.25, rel_tol=1e-3)

    lvl_c, mult_c = opacity_to_ringelmann(71.2)
    assert lvl_c == 4 and math.isclose(mult_c, 1.60, rel_tol=1e-3)


def test_tc02_baseline_coal_stack_math():
    """TC-02: Verify baseline 1200 kg/hr coal burn at Ringelmann Lvl 4 (1.60x) = 4646.40 kg CO2/hr."""
    res = calculate_industrial_emissions(
        fuel_type="coal",
        fuel_rate_kg_hr=1200.0,
        opacity_multiplier=1.60,
        scrubber_efficiency_pct=0.0,
        combustion_tuning_pct=0.0,
    )
    # 1200 * 2.42 * 1.60 = 4646.40
    assert math.isclose(res["baseline_co2_kg_hr"], 4646.40, abs_tol=0.1)


def test_tc03_and_tc04_industrial_interventions():
    """TC-03 & TC-04: Verify combustion tuning, fuel switch to biomass, and 85% scrubber math."""
    res = calculate_industrial_emissions(
        fuel_type="biomass",
        fuel_rate_kg_hr=1200.0,
        opacity_multiplier=1.60,
        scrubber_efficiency_pct=85.0,
        combustion_tuning_pct=80.0,
    )
    # Expected projected stack emissions must be >95% lower than coal baseline
    assert res["projected_co2_kg_hr"] < 200.0
    assert res["net_reduction_pct"] > 90.0


def test_tc05_and_tc06_digital_twin_economics_and_compliance():
    """TC-05 & TC-06: Verify Digital Twin INR carbon credit arithmetic and CPCB compliance transition."""
    params = DigitalTwinParams(
        ev_adoption_pct=0.0,
        idle_time_cut_pct=0.0,
        e20_blend_pct=0.0,
        ban_bs3_bs4_heavy=False,
        scrubber_efficiency_pct=85.0,
        fuel_type="biomass",
        combustion_tuning_pct=80.0,
    )
    out = simulate_digital_twin(params)

    # 1. Verify exact conservation of mass: saved_kg_hr == baseline - projected
    base_total = out["baseline_summary"]["total_co2_kg_hr"]
    proj_total = out["projected_summary"]["total_co2_kg_hr"]
    saved_hr = out["projected_summary"]["avoided_co2_kg_hr"]
    assert math.isclose(base_total - proj_total, saved_hr, abs_tol=0.2)

    # 2. Verify INR Carbon Credit Math (₹1,350/tonne = ₹1.35/kg)
    fin = out["financial_savings"]
    expected_hourly_inr = saved_hr * 1.35
    expected_daily_inr = expected_hourly_inr * 24.0
    expected_annual_inr = expected_daily_inr * 365.0

    assert math.isclose(fin["hourly_inr"], expected_hourly_inr, rel_tol=1e-2)
    assert math.isclose(fin["daily_inr"], expected_daily_inr, rel_tol=1e-2)
    assert math.isclose(fin["annual_inr"], expected_annual_inr, rel_tol=1e-2)

    # 3. Verify CPCB Industrial Compliance flips to COMPLIANT (#22c55e) and plume turns clean steam (#e2e8f0)
    assert out["cpcb_compliance"]["industrial_status"] == "COMPLIANT"
    assert out["scene_3d_industrial"]["plume_color_hex"] == "#e2e8f0"
    assert out["scene_3d_industrial"]["scrubber_active"] is True
