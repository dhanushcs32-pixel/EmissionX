"""Digital Twin Simulation Engine Backend (Pillar 1 & Pillar 2 Cross-Node Integration).

Deterministic, high-speed physics and economics sandbox:
- Node 1: Fuel Station Forecourt (from backend/data/phase1_summary.json)
- Node 2: Industrial Chimney Stack (from cache/factory_telemetry.json)

Applies multi-parameter 'What-If' interventions, calculates baseline and projected emissions,
computes CPCB compliance statuses, evaluates financial INR carbon credit savings,
and outputs synchronized 3D scene parameters and comparative bar chart metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("DigitalTwin")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# =====================================================================
# 1. EMISSION FACTORS & INDUSTRIAL CARBON CONSTANTS
# =====================================================================

INDUSTRIAL_EMISSION_FACTORS: Dict[str, float] = {
    "coal": 2.42,          # kg CO2 per kg fuel
    "furnace_oil": 3.15,   # kg CO2 per kg fuel
    "natural_gas": 1.93,   # kg CO2 per m3 fuel
    "biomass": 0.45,       # kg CO2 per kg fuel (net lifecycle)
}

VEHICULAR_EMISSION_FACTORS: Dict[str, float] = {
    "petrol": 2.31,        # kg CO2 per liter
    "diesel": 2.68,        # kg CO2 per liter
}

# Real-world idling fuel burn rates (kg CO2 per second)
VEHICLE_IDLE_CO2_PER_SEC: Dict[str, float] = {
    "car": 0.00075,
    "truck": 0.0022,
    "bus": 0.0020,
    "motorcycle": 0.00025,
}

# Standard average dispensed fuel per refueling visit (liters)
DISPENSED_FUEL_PER_VISIT: Dict[str, float] = {
    "car": 18.0,           # Petrol
    "motorcycle": 4.0,     # Petrol
    "bus": 45.0,           # Diesel
    "truck": 65.0,         # Diesel
}

# Carbon credit price in India (CCTS market benchmark: ₹1,350 per tonne CO2)
INR_PER_TONNE_CO2 = 1350.0
INR_PER_KG_CO2 = INR_PER_TONNE_CO2 / 1000.0  # ₹1.35 per kg


# =====================================================================
# 2. STRICT PYDANTIC SCHEMAS FOR DIGITAL TWIN
# =====================================================================

class DigitalTwinParams(BaseModel):
    """Controllable levers for multi-parameter What-If sandbox."""
    # Vehicular Levers
    ev_adoption_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Proportionally cuts tailpipe and dispensed fuel carbon for cars/bikes to 0",
    )
    idle_time_cut_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Directly cuts forecourt queue idling time",
    )
    e20_blend_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Reduces petrol emission factor by (0.18 * (e20_blend_pct / 100))",
    )
    ban_bs3_bs4_heavy: bool = Field(
        default=False,
        description="Cuts heavy vehicle (truck/bus) emissions by 35% (simulating BS6/LNG retrofit)",
    )

    # Industrial Levers
    scrubber_efficiency_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=95.0,
        description="Particulate capture efficiency: stack_emissions *= (1 - scrubber_efficiency / 100)",
    )
    fuel_type: Literal["coal", "furnace_oil", "natural_gas", "biomass"] = Field(
        default="coal",
        description="Fuel type: coal (2.42), furnace_oil (3.15), natural_gas (1.93), biomass (0.45)",
    )
    combustion_tuning_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Bridges Ringelmann penalty multiplier down to clean stoichiometric 1.0x",
    )


class NodeBaselineData(BaseModel):
    source: str
    fuel_rate_or_dispense: float
    unit: str
    active_factor: float
    penalty_multiplier: float
    raw_co2_kg_hr: float


class NodeSimulatedData(BaseModel):
    projected_co2_kg_hr: float
    avoided_co2_kg_hr: float
    reduction_pct: float


class ComparativeBarMetric(BaseModel):
    category: str
    baseline_co2: float
    projected_co2: float
    avoided_co2: float
    reduction_pct: float


class FinancialSavingsData(BaseModel):
    currency: str = "INR (₹)"
    carbon_price_per_tonne_inr: float = INR_PER_TONNE_CO2
    hourly_inr: float
    daily_inr: float
    annual_inr: float


class CPCBComplianceItem(BaseModel):
    node: str
    standard_name: str
    measured_metric: str
    measured_value: str
    prescribed_limit: str
    status: Literal["COMPLIANT", "MODERATE CONCERN", "NON-COMPLIANT / VIOLATION"]
    color_code: Literal["#22c55e", "#f59e0b", "#ef4444"]
    recommendation: str


class IndustrialScene3D(BaseModel):
    plume_opacity: float = Field(description="Visual opacity of plume [0.05, 1.0]")
    plume_color_hex: str = Field(description="Hex color code for smoke plume")
    plume_scale: float = Field(description="Particle spread multiplier [0.2, 1.5]")
    particle_rate: int = Field(description="Particle spawn rate per second [5, 60]")
    flue_temp_c: float
    so2_ppm: int
    scrubber_active: bool
    scrubber_glow_color: str
    chimney_tip_glow: float


class VehicularScene3D(BaseModel):
    active_vehicle_counts: Dict[str, int]
    ev_vehicle_counts: Dict[str, int]
    avg_dwell_sec: float
    queue_density_pct: float
    station_smog_opacity: float
    charger_bays_active: int


class DigitalTwinResponse(BaseModel):
    timestamp: str
    parameters_applied: DigitalTwinParams
    baseline_summary: Dict[str, Any]
    projected_summary: Dict[str, Any]
    comparative_chart_metrics: List[ComparativeBarMetric]
    financial_savings: FinancialSavingsData
    cpcb_compliance: List[CPCBComplianceItem]
    scene_3d_industrial: IndustrialScene3D
    scene_3d_vehicular: VehicularScene3D


# =====================================================================
# 3. BASELINE EXTRACTION WITH ULTRA-RELIABLE FALLBACKS
# =====================================================================

def extract_vehicular_baseline() -> Dict[str, Any]:
    """
    Extracts vehicular baseline from backend/data/phase1_summary.json.
    Falls back gracefully to realistic forecourt standards if absent or empty.
    """
    summary_path = WORKSPACE_ROOT / "backend" / "data" / "phase1_summary.json"
    fallback_counts = {"car": 7, "truck": 3, "bus": 2, "motorcycle": 5}
    fallback_dwell = 42.0

    counts = dict(fallback_counts)
    avg_dwell = fallback_dwell

    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cbc = data.get("counts_by_class", {})
                for k in ["car", "truck", "bus", "motorcycle"]:
                    if k in cbc and cbc[k] > 0:
                        counts[k] = int(cbc[k])
                # Check tracked vehicles for dwell times
                tv = data.get("tracked_vehicles", [])
                dwells = [t.get("estimated_dwell_seconds", 0.0) for t in tv if t.get("estimated_dwell_seconds", 0.0) > 5.0]
                if dwells:
                    avg_dwell = float(sum(dwells) / len(dwells))
        except Exception as e:
            logger.warning(f"Could not read phase1_summary.json ({e}). Using forecourt baseline.")

    # Calculate throughput: station turnover cycles per hour
    # 4 fueling bays operating simultaneously
    cycles_per_hr = max(10.0, 3600.0 / (avg_dwell + 150.0))

    # 1. Forecourt Idling Emissions (kg CO2 / hr)
    idling_co2_hr = 0.0
    for vtype, count in counts.items():
        idle_rate = VEHICLE_IDLE_CO2_PER_SEC.get(vtype, 0.00075)
        # Idling emissions per cycle * cycles per hour
        idling_co2_hr += count * idle_rate * avg_dwell * (cycles_per_hr / 4.0)

    # 2. Dispensed Fuel Burn Emissions (kg CO2 / hr)
    dispensed_co2_hr = 0.0
    for vtype, count in counts.items():
        liters = DISPENSED_FUEL_PER_VISIT.get(vtype, 15.0)
        fuel = "diesel" if vtype in ["truck", "bus"] else "petrol"
        factor = VEHICULAR_EMISSION_FACTORS[fuel]
        # Dispensed fuel consumed per hour across active fleet turnaround
        dispensed_co2_hr += count * liters * factor * (cycles_per_hr / 12.0)

    total_vehicular_co2 = round(idling_co2_hr + dispensed_co2_hr, 1)

    return {
        "vehicle_counts": counts,
        "avg_dwell_sec": round(avg_dwell, 1),
        "idling_co2_kg_hr": round(idling_co2_hr, 1),
        "dispensed_co2_kg_hr": round(dispensed_co2_hr, 1),
        "total_vehicular_co2_kg_hr": total_vehicular_co2,
    }


def extract_industrial_baseline() -> Dict[str, Any]:
    """
    Extracts industrial baseline from cache/factory_telemetry.json.
    Falls back gracefully to authentic Peenya boiler standards if absent or empty.
    """
    telemetry_path = WORKSPACE_ROOT / "cache" / "factory_telemetry.json"
    fallback_fuel_rate = 1200.0
    fallback_opacity = 71.2
    fallback_ringelmann = 4
    fallback_mult = 1.60
    fallback_temp = 224.0
    fallback_so2 = 295

    fuel_rate = fallback_fuel_rate
    opacity = fallback_opacity
    ringelmann = fallback_ringelmann
    mult = fallback_mult
    temp_c = fallback_temp
    so2_ppm = fallback_so2

    if telemetry_path.is_file():
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if records and isinstance(records, list):
                    # Pick sample from active run (75% frame index)
                    idx = min(len(records) - 1, int(len(records) * 0.75))
                    sample = records[idx]
                    cv_tel = sample.get("cv_telemetry", {})
                    cems = cv_tel.get("cems_metrics", {})
                    opacity = float(cv_tel.get("smoke_opacity_pct", fallback_opacity))
                    ringelmann = int(cv_tel.get("ringelmann_level", fallback_ringelmann))
                    mult = float(cv_tel.get("opacity_multiplier", fallback_mult))
                    fuel_rate = float(cems.get("fuel_rate_kg_hr", fallback_fuel_rate))
                    temp_c = float(cems.get("flue_gas_temp_c", fallback_temp))
                    so2_ppm = int(cems.get("so2_ppm", fallback_so2))
        except Exception as e:
            logger.warning(f"Could not read factory_telemetry.json ({e}). Using Peenya boiler baseline.")

    base_factor = INDUSTRIAL_EMISSION_FACTORS["coal"]
    ideal_co2 = fuel_rate * base_factor
    baseline_stack_co2 = round(ideal_co2 * mult, 1)
    waste_stack_co2 = round(max(0.0, baseline_stack_co2 - ideal_co2), 1)

    return {
        "fuel_type": "coal",
        "fuel_rate_kg_hr": round(fuel_rate, 1),
        "smoke_opacity_pct": round(opacity, 1),
        "ringelmann_level": ringelmann,
        "opacity_multiplier": round(mult, 2),
        "flue_gas_temp_c": round(temp_c, 1),
        "so2_ppm": so2_ppm,
        "baseline_stack_co2_kg_hr": baseline_stack_co2,
        "waste_stack_co2_kg_hr": waste_stack_co2,
        "ideal_stack_co2_kg_hr": round(ideal_co2, 1),
    }


# =====================================================================
# 4. DIGITAL TWIN SIMULATION ENGINE
# =====================================================================

def simulate_digital_twin(params: Optional[DigitalTwinParams] = None) -> DigitalTwinResponse:
    """
    Executes cross-node What-If simulation combining physical CCTV node telemetry,
    interventions, CPCB compliance metrics, 3D parameters, and financial savings.
    """
    if params is None:
        params = DigitalTwinParams()

    # 1. Fetch live or fallback baselines
    ind_base = extract_industrial_baseline()
    veh_base = extract_vehicular_baseline()

    # -----------------------------------------------------------------
    # A. Industrial Intervention Math
    # -----------------------------------------------------------------
    chosen_fuel = params.fuel_type.lower()
    sim_fuel_factor = INDUSTRIAL_EMISSION_FACTORS.get(chosen_fuel, 2.42)

    # Combustion tuning bridges Ringelmann penalty multiplier down to stoichiometric 1.0x
    tuning_ratio = max(0.0, min(1.0, params.combustion_tuning_pct / 100.0))
    sim_multiplier = max(1.0, 1.0 + (ind_base["opacity_multiplier"] - 1.0) * (1.0 - tuning_ratio))

    # Wet Scrubber capture efficiency
    scrubber_eff = max(0.0, min(0.95, params.scrubber_efficiency_pct / 100.0))

    raw_projected_industrial = (
        ind_base["fuel_rate_kg_hr"] * sim_fuel_factor * sim_multiplier * (1.0 - scrubber_eff)
    )
    projected_industrial_co2 = round(raw_projected_industrial, 1)
    avoided_industrial_co2 = max(0.0, round(ind_base["baseline_stack_co2_kg_hr"] - projected_industrial_co2, 1))
    ind_reduction_pct = round(
        (avoided_industrial_co2 / ind_base["baseline_stack_co2_kg_hr"] * 100.0)
        if ind_base["baseline_stack_co2_kg_hr"] > 0 else 0.0,
        1,
    )

    # Simulated stack optical characteristics
    sim_opacity = max(
        5.0,
        ind_base["smoke_opacity_pct"] * (1.0 - scrubber_eff * 0.85) * (1.0 - tuning_ratio * 0.45)
    )
    sim_so2 = int(max(20.0, ind_base["so2_ppm"] * (1.0 - scrubber_eff * 0.92)))
    sim_flue_temp = round(max(140.0, ind_base["flue_gas_temp_c"] - scrubber_eff * 35.0), 1)

    # -----------------------------------------------------------------
    # B. Vehicular Intervention Math
    # -----------------------------------------------------------------
    ev_cut = max(0.0, min(1.0, params.ev_adoption_pct / 100.0))
    idle_cut = max(0.0, min(1.0, params.idle_time_cut_pct / 100.0))
    e20_ratio = max(0.0, min(1.0, params.e20_blend_pct / 100.0))
    e20_petrol_factor = VEHICULAR_EMISSION_FACTORS["petrol"] * (1.0 - 0.18 * e20_ratio)

    sim_dwell_sec = max(12.0, veh_base["avg_dwell_sec"] * (1.0 - idle_cut))
    sim_cycles_per_hr = max(10.0, 3600.0 / (sim_dwell_sec + 150.0))

    sim_idling_co2 = 0.0
    sim_dispensed_co2 = 0.0

    counts = veh_base["vehicle_counts"]
    ev_counts: Dict[str, int] = {}

    for vtype, count in counts.items():
        # EV adoption applies to cars and motorcycles
        if vtype in ["car", "motorcycle"]:
            ev_n = int(round(count * ev_cut))
            ev_counts[vtype] = ev_n
            ice_n = count - ev_n
        else:
            ev_counts[vtype] = 0
            ice_n = count

        # Heavy ban cuts truck/bus emissions by 35%
        heavy_factor = 0.65 if (params.ban_bs3_bs4_heavy and vtype in ["truck", "bus"]) else 1.0

        # Idling calculation
        idle_rate = VEHICLE_IDLE_CO2_PER_SEC.get(vtype, 0.00075)
        sim_idling_co2 += ice_n * idle_rate * sim_dwell_sec * (sim_cycles_per_hr / 4.0) * heavy_factor

        # Dispensed fuel calculation
        liters = DISPENSED_FUEL_PER_VISIT.get(vtype, 15.0)
        eff_factor = VEHICULAR_EMISSION_FACTORS["diesel"] if vtype in ["truck", "bus"] else e20_petrol_factor
        sim_dispensed_co2 += ice_n * liters * eff_factor * (sim_cycles_per_hr / 12.0) * heavy_factor

    projected_vehicular_co2 = round(sim_idling_co2 + sim_dispensed_co2, 1)
    avoided_vehicular_co2 = max(0.0, round(veh_base["total_vehicular_co2_kg_hr"] - projected_vehicular_co2, 1))
    veh_reduction_pct = round(
        (avoided_vehicular_co2 / veh_base["total_vehicular_co2_kg_hr"] * 100.0)
        if veh_base["total_vehicular_co2_kg_hr"] > 0 else 0.0,
        1,
    )

    # -----------------------------------------------------------------
    # C. Cross-Node Comparative Totals & Economics
    # -----------------------------------------------------------------
    total_baseline_co2 = round(ind_base["baseline_stack_co2_kg_hr"] + veh_base["total_vehicular_co2_kg_hr"], 1)
    total_projected_co2 = round(projected_industrial_co2 + projected_vehicular_co2, 1)
    total_avoided_co2 = max(0.0, round(total_baseline_co2 - total_projected_co2, 1))
    total_reduction_pct = round((total_avoided_co2 / total_baseline_co2 * 100.0) if total_baseline_co2 > 0 else 0.0, 1)

    # Carbon credit financial valuation
    hourly_inr = round(total_avoided_co2 * INR_PER_KG_CO2, 2)
    daily_inr = round(hourly_inr * 24.0, 2)
    annual_inr = round(total_avoided_co2 * 24.0 * 365.0 * 0.001 * INR_PER_TONNE_CO2, 0)

    comparative_metrics: List[ComparativeBarMetric] = [
        ComparativeBarMetric(
            category="Industrial Smokestack",
            baseline_co2=ind_base["baseline_stack_co2_kg_hr"],
            projected_co2=projected_industrial_co2,
            avoided_co2=avoided_industrial_co2,
            reduction_pct=ind_reduction_pct,
        ),
        ComparativeBarMetric(
            category="Vehicular Forecourt",
            baseline_co2=veh_base["total_vehicular_co2_kg_hr"],
            projected_co2=projected_vehicular_co2,
            avoided_co2=avoided_vehicular_co2,
            reduction_pct=veh_reduction_pct,
        ),
        ComparativeBarMetric(
            category="Combined Facility Total",
            baseline_co2=total_baseline_co2,
            projected_co2=total_projected_co2,
            avoided_co2=total_avoided_co2,
            reduction_pct=total_reduction_pct,
        ),
    ]

    # -----------------------------------------------------------------
    # D. CPCB (Central Pollution Control Board) Compliance Assessment
    # -----------------------------------------------------------------
    compliance_items: List[CPCBComplianceItem] = []

    # 1. Industrial Opacity & SO2
    if sim_opacity <= 20.0 and sim_so2 <= 100:
        ind_cpcb_status = "COMPLIANT"
        ind_cpcb_color = "#22c55e"
        ind_cpcb_rec = "Complies fully with CPCB / KSPCB Stack Emission Norms (Ringelmann Level 1)."
    elif sim_opacity <= 45.0 and sim_so2 <= 180:
        ind_cpcb_status = "MODERATE CONCERN"
        ind_cpcb_color = "#f59e0b"
        ind_cpcb_rec = "Minor particulate threshold warning; recommend scrubber trim or damper tuning."
    else:
        ind_cpcb_status = "NON-COMPLIANT / VIOLATION"
        ind_cpcb_color = "#ef4444"
        ind_cpcb_rec = "Violation under Section 21/22 Air Act 1981. Engage wet scrubber >80% immediately."

    compliance_items.append(
        CPCBComplianceItem(
            node="Peenya Smokestack Node",
            standard_name="CPCB Industrial Boiler Emission Norms",
            measured_metric="Optical Opacity & SO2",
            measured_value=f"{sim_opacity:.1f}% Opacity | {sim_so2} ppm SO2",
            prescribed_limit="<= 20% Opacity (Ringelmann 1) | <= 100 ppm SO2",
            status=ind_cpcb_status,
            color_code=ind_cpcb_color,
            recommendation=ind_cpcb_rec,
        )
    )

    # 2. Vehicular Idling & Heavy Fleet
    if sim_dwell_sec <= 25.0 and (params.ban_bs3_bs4_heavy or params.ev_adoption_pct >= 20.0):
        veh_cpcb_status = "COMPLIANT"
        veh_cpcb_color = "#22c55e"
        veh_cpcb_rec = "Exceeds MoRTH Clean Corridor and NCAP urban mobility guidelines."
    elif sim_dwell_sec <= 50.0:
        veh_cpcb_status = "MODERATE CONCERN"
        veh_cpcb_color = "#f59e0b"
        veh_cpcb_rec = "Elevated queue dwell detected; optimize pump dispatch or open reserve bays."
    else:
        veh_cpcb_status = "NON-COMPLIANT / VIOLATION"
        veh_cpcb_color = "#ef4444"
        veh_cpcb_rec = "Urban forecourt bottleneck exceeding 50s idling standard."

    compliance_items.append(
        CPCBComplianceItem(
            node="Bengaluru Fuel Forecourt Node",
            standard_name="MoRTH / NCAP Urban Air Quality Protocol",
            measured_metric="Queue Dwell & Fleet BS-VI Ratio",
            measured_value=f"{sim_dwell_sec:.1f}s Dwell | {params.ev_adoption_pct:.0f}% EV Share",
            prescribed_limit="<= 25s Average Queue Dwell Time",
            status=veh_cpcb_status,
            color_code=veh_cpcb_color,
            recommendation=veh_cpcb_rec,
        )
    )

    # -----------------------------------------------------------------
    # E. Synchronized 3D Scene Parameters (Three.js / Canvas 2.5D Twin)
    # -----------------------------------------------------------------
    # Plume color shifts from dark charcoal to clean white steam as opacity drops
    if sim_opacity > 60.0:
        plume_hex = "#27272a"  # Dark soot
        plume_scale = 1.35
        particle_rate = 50
    elif sim_opacity > 30.0:
        plume_hex = "#78716c"  # Amber/ash gray
        plume_scale = 1.05
        particle_rate = 32
    else:
        plume_hex = "#e2e8f0"  # Clean white steam
        plume_scale = 0.65
        particle_rate = 14

    scene_ind = IndustrialScene3D(
        plume_opacity=round(min(1.0, max(0.08, sim_opacity / 100.0)), 2),
        plume_color_hex=plume_hex,
        plume_scale=round(plume_scale, 2),
        particle_rate=particle_rate,
        flue_temp_c=sim_flue_temp,
        so2_ppm=sim_so2,
        scrubber_active=params.scrubber_efficiency_pct > 0.0,
        scrubber_glow_color="#22c55e" if params.scrubber_efficiency_pct > 0.0 else "#475569",
        chimney_tip_glow=round(max(0.1, min(1.0, sim_flue_temp / 260.0)), 2),
    )

    # Vehicular scene parameters
    total_active_ice = sum(counts.values()) - sum(ev_counts.values())
    queue_density = min(100.0, (sim_dwell_sec / 60.0) * 100.0)
    station_smog = min(0.85, (total_active_ice / 15.0) * (sim_dwell_sec / 60.0))

    scene_veh = VehicularScene3D(
        active_vehicle_counts=counts,
        ev_vehicle_counts=ev_counts,
        avg_dwell_sec=round(sim_dwell_sec, 1),
        queue_density_pct=round(queue_density, 1),
        station_smog_opacity=round(station_smog, 2),
        charger_bays_active=max(1, int(round(params.ev_adoption_pct / 25.0))),
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    return DigitalTwinResponse(
        timestamp=now_iso,
        parameters_applied=params,
        baseline_summary={
            "industrial": ind_base,
            "vehicular": veh_base,
            "total_baseline_co2_kg_hr": total_baseline_co2,
        },
        projected_summary={
            "industrial_co2_kg_hr": projected_industrial_co2,
            "vehicular_co2_kg_hr": projected_vehicular_co2,
            "total_projected_co2_kg_hr": total_projected_co2,
            "total_avoided_co2_kg_hr": total_avoided_co2,
            "total_reduction_pct": total_reduction_pct,
        },
        comparative_chart_metrics=comparative_metrics,
        financial_savings=FinancialSavingsData(
            hourly_inr=hourly_inr,
            daily_inr=daily_inr,
            annual_inr=annual_inr,
        ),
        cpcb_compliance=compliance_items,
        scene_3d_industrial=scene_ind,
        scene_3d_vehicular=scene_veh,
    )
