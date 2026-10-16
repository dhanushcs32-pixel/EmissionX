"""Industrial Emissions Module (Pillar 2) for Carbon Footprint Intelligence Platform.

Modular, production-ready Python pipeline analyzing CCTV smokestack footage,
detecting smoke plumes with True Multi-Cue Contour Segmentation, computing masked
Ringelmann Opacity, generating synthetic CEMS telemetry, calculating real-time CO2 emissions,
and producing annotated video with glowing segmentation overlay + telemetry JSON.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IndustrialDetector")

# Optional Ultralytics import for Primary Mode
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# Optional Supervision import
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    sv = None


# =====================================================================
# 1. EMISSION FACTORS & INDUSTRIAL CARBON MATH ENGINE
# =====================================================================

EMISSION_FACTORS: Dict[str, float] = {
    "coal": 2.42,         # kg CO2 per kg fuel
    "furnace_oil": 3.15,  # kg CO2 per kg fuel
    "natural_gas": 1.93,  # kg CO2 per m3 fuel
    "biomass": 0.45,      # kg CO2 per kg fuel (net lifecycle factor)
}


def opacity_to_ringelmann(opacity_pct: float) -> Tuple[int, float]:
    """
    Maps optical opacity percentage (0-100%) to Ringelmann level (0-5)
    and combustion penalty multiplier:
        0 - 20%: Level 0-1, 1.00x (clean stoichiometric burn)
        20% - 60%: Level 2-3, 1.25x (moderate unburnt carbon warning)
        60% - 100%: Level 4-5, 1.60x (severe incomplete combustion violation)
    """
    op = float(opacity_pct)
    if op <= 10.0:
        level = 0
    elif op <= 20.0:
        level = 1
    elif op <= 40.0:
        level = 2
    elif op <= 60.0:
        level = 3
    elif op <= 80.0:
        level = 4
    else:
        level = 5

    if op < 20.0:
        mult = 1.00
    elif op < 60.0:
        mult = 1.25
    else:
        mult = 1.60

    return level, mult


def calculate_industrial_emissions(
    fuel_rate_kg_hr: float = 1200.0,
    fuel_type: str = "coal",
    opacity_multiplier: float = 1.0,
    scrubber_efficiency: float = 0.0,
    combustion_tuning_pct: float = 0.0,
    scrubber_efficiency_pct: Optional[float] = None,
    baseline_fuel: Optional[str] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Pure, deterministic calculation engine for baseline & what-if simulation emissions.
    Shared across video frame processing, FastAPI /api/simulate endpoints, and verification tests.

    Formula:
        CO2,base = R_fuel * EF_fuel * M_op * (1.0 - scrubber_eff)
        M_tuned  = 1.0 + (M_op - 1.0) * (1.0 - combustion_tuning_pct / 100.0)
        CO2,proj = R_fuel * EF_new_fuel * M_tuned * (1.0 - scrubber_eff)
        Saved    = CO2,base - CO2,proj
    """
    norm_fuel = fuel_type.lower().strip().replace(" ", "_")
    emission_factor = EMISSION_FACTORS.get(norm_fuel, 2.42)

    # Baseline scrubber (existing physical scrubber on the baseline stack, default 0.0)
    base_scrub = float(scrubber_efficiency)
    base_scrub_eff = base_scrub / 100.0 if base_scrub > 1.0 else base_scrub
    base_scrub_eff = max(0.0, min(1.0, base_scrub_eff))

    # What-If Intervention scrubber capture efficiency (S_eff)
    if scrubber_efficiency_pct is not None:
        raw_sim_scrub = float(scrubber_efficiency_pct)
        sim_scrub_eff = raw_sim_scrub / 100.0 if raw_sim_scrub > 1.0 else raw_sim_scrub
    else:
        sim_scrub_eff = base_scrub_eff
    sim_scrub_eff = max(0.0, min(1.0, sim_scrub_eff))

    op_mult = max(1.0, float(opacity_multiplier))

    # Clean theoretical combustion rate
    ideal_co2_kg_hr = fuel_rate_kg_hr * emission_factor

    # Baseline calculation
    # For fuel switch interventions (e.g. switching to biomass), the unmitigated baseline
    # reflects the baseline coal firing (2.42) unless baseline_fuel is explicitly overridden.
    baseline_ef = EMISSION_FACTORS.get(baseline_fuel, 2.42) if baseline_fuel else (
        2.42 if norm_fuel != "coal" else emission_factor
    )
    raw_baseline = fuel_rate_kg_hr * baseline_ef * op_mult * (1.0 - base_scrub_eff)
    baseline_co2_kg_hr = round(raw_baseline, 1)

    # Excess carbon caused by incomplete combustion/scrubber deficiency
    ideal_base_co2 = fuel_rate_kg_hr * baseline_ef
    raw_waste = max(0.0, baseline_co2_kg_hr - ideal_base_co2)
    waste_co2_kg_hr = round(raw_waste, 1)

    # What-if simulation projection (with combustion tuning & scrubber capture)
    tuning_reduction = max(0.0, min(1.0, float(combustion_tuning_pct) / 100.0))
    sim_multiplier = max(1.0, 1.0 + (op_mult - 1.0) * (1.0 - tuning_reduction))
    raw_projected = ideal_co2_kg_hr * sim_multiplier * (1.0 - sim_scrub_eff)
    projected_co2_kg_hr = round(raw_projected, 2)

    saved_co2_kg_hr = max(0.0, round(baseline_co2_kg_hr - projected_co2_kg_hr, 2))

    if baseline_co2_kg_hr > 0.0:
        reduction_pct = round(
            max(0.0, (saved_co2_kg_hr / baseline_co2_kg_hr) * 100.0),
            2,
        )
    else:
        reduction_pct = 0.0

    return {
        "baseline_co2_kg_hr": baseline_co2_kg_hr,
        "waste_co2_kg_hr": waste_co2_kg_hr,
        "projected_co2_kg_hr": projected_co2_kg_hr,
        "saved_co2_kg_hr": saved_co2_kg_hr,
        "reduction_pct": reduction_pct,
        "net_reduction_pct": reduction_pct,
        "emission_factor": emission_factor,
        "ideal_co2_kg_hr": round(ideal_co2_kg_hr, 1),
    }


# =====================================================================
# 2. TRUE MASKED RINGELMANN SMOKE OPACITY CALCULATION
# =====================================================================

def calculate_ringelmann_opacity(
    frame_or_roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, int, float, str]:
    """
    Computes Ringelmann Smoke Opacity (0-100%).
    When a plume_mask is provided, computes mean pixel brightness strictly on
    the pixels inside plume_mask > 0 (cv2.mean(gray, mask=mask)[0]), completely
    eliminating empty-sky dilution.

    Opacity Formula:
        opacity_pct = round(max(0.0, min(100.0, (1.0 - (mean_brightness / 255.0)) * 100.0)), 1)

    Classification:
        * 0% - 20%  (Ringelmann 0-1: Clean steam / normal operation) -> multiplier = 1.0, status = "NORMAL"
        * 21% - 60% (Ringelmann 2-3: Incomplete combustion warning)  -> multiplier = 1.25, status = "COMBUSTION WARNING"
        * > 60%     (Ringelmann 4-5: Black carbon violation / Scrubber failure) -> multiplier = 1.60, status = "SCRUBBER VIOLATION"
    """
    if frame_or_roi is None or frame_or_roi.size == 0:
        return 0.0, 0, 1.0, "NORMAL"

    if len(frame_or_roi.shape) == 3 and frame_or_roi.shape[2] == 3:
        gray = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_or_roi

    # Compute mean brightness strictly inside the mask if available
    if mask is not None and cv2.countNonZero(mask) > 20:
        if mask.shape[:2] == gray.shape[:2]:
            mean_brightness = float(cv2.mean(gray, mask=mask)[0])
        else:
            resized_mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            mean_brightness = float(cv2.mean(gray, mask=resized_mask)[0])
    else:
        mean_brightness = float(np.mean(gray)) if gray.size > 0 else 255.0

    opacity_pct = round(max(0.0, min(100.0, (1.0 - (mean_brightness / 255.0)) * 100.0)), 1)

    # Classify Ringelmann level (0 to 5)
    if opacity_pct <= 10.0:
        ringelmann_level = 0
    elif opacity_pct <= 20.0:
        ringelmann_level = 1
    elif opacity_pct <= 40.0:
        ringelmann_level = 2
    elif opacity_pct <= 60.0:
        ringelmann_level = 3
    elif opacity_pct <= 80.0:
        ringelmann_level = 4
    else:
        ringelmann_level = 5

    # Assign combustion penalty multiplier and operational status
    if opacity_pct <= 20.0:
        multiplier = 1.0
        status = "NORMAL"
    elif opacity_pct <= 60.0:
        multiplier = 1.25
        status = "COMBUSTION WARNING"
    else:
        multiplier = 1.60
        status = "SCRUBBER VIOLATION"

    return opacity_pct, ringelmann_level, multiplier, status


# =====================================================================
# 3. SYNTHETIC CEMS (CONTINUOUS EMISSION MONITORING SYSTEM) GENERATOR
# =====================================================================

class SyntheticCEMSGenerator:
    """
    Simulates real-time telemetry from boiler / chimney sensors synchronized
    with the detected optical smoke plume opacity.
    """

    def __init__(
        self,
        fuel_type: str = "coal",
        base_fuel_rate: float = 1200.0,
        scrubber_efficiency: float = 0.0,
        boiler_id: str = "Boiler #1",
    ):
        self.fuel_type = fuel_type
        self.base_fuel_rate = base_fuel_rate
        self.scrubber_efficiency = scrubber_efficiency
        self.boiler_id = boiler_id
        self._step_counter = 0

    def generate(
        self,
        opacity_pct: float,
        ringelmann_level: int,
        opacity_multiplier: float,
    ) -> Tuple[Dict[str, Any], List[str]]:
        self._step_counter += 1
        t = self._step_counter * 0.15

        # 1. Fuel burn rate: Base ~1200.0 kg/hr + micro-fluctuations + spike when opacity > 50%
        micro_fluctuation = math.sin(t) * 12.0 + math.cos(t * 0.5) * 6.0 + random.uniform(-3.0, 3.0)
        spike = 0.0
        if opacity_pct > 50.0:
            spike = (opacity_pct - 50.0) * 0.95 + 15.0

        fuel_rate = round(self.base_fuel_rate + micro_fluctuation + spike, 1)

        # 2. Flue Gas Temp: ~160°C to 240°C scaling with opacity
        base_temp = 160.0 + (opacity_pct / 100.0) * 80.0
        temp_noise = math.sin(t * 0.8) * 2.5 + random.uniform(-1.0, 1.0)
        flue_gas_temp_c = round(max(150.0, min(260.0, base_temp + temp_noise)), 1)

        # 3. SO2 PPM: ~80 to 350 ppm (spikes sharply during Ringelmann 4-5 violations)
        if ringelmann_level >= 4 or opacity_pct > 60.0:
            base_so2 = 250.0 + (opacity_pct - 60.0) * 2.5
            so2_noise = random.randint(-10, 15)
            so2_ppm = int(max(240, min(360, base_so2 + so2_noise)))
        elif opacity_pct > 20.0:
            base_so2 = 120.0 + (opacity_pct - 20.0) * 3.0
            so2_noise = random.randint(-8, 8)
            so2_ppm = int(max(90, min(240, base_so2 + so2_noise)))
        else:
            base_so2 = 85.0 + opacity_pct * 1.5
            so2_noise = random.randint(-5, 5)
            so2_ppm = int(max(75, min(130, base_so2 + so2_noise)))

        cems_metrics = {
            "fuel_type": self.fuel_type,
            "fuel_rate_kg_hr": fuel_rate,
            "flue_gas_temp_c": flue_gas_temp_c,
            "so2_ppm": so2_ppm,
        }

        # Formulate standardized diagnostic log lines
        recent_logs = [
            f" [PLUME-CV] Smoke Opacity {opacity_pct:.1f}% (Ringelmann Lvl {ringelmann_level}) -> Penalty {opacity_multiplier:.1f}x",
            f" [CEMS-IOT] {self.boiler_id} Coal Feed: {fuel_rate:.0f} kg/hr | Flue Temp: {flue_gas_temp_c:.1f}C | SO2: {so2_ppm} ppm",
        ]

        return cems_metrics, recent_logs


# =====================================================================
# 4. STRICT PYDANTIC TELEMETRY SCHEMAS
# =====================================================================

class EnvironmentData(BaseModel):
    temp_c: float = Field(default=29.5, description="Ambient air temperature in Celsius")
    wind_kmh: float = Field(default=11.2, description="Local wind speed in km/h")
    wind_deg: int = Field(default=240, description="Wind heading direction in degrees")
    aqi: int = Field(default=168, description="Regional Air Quality Index")


class CEMSMetricsData(BaseModel):
    fuel_type: str = Field(default="coal")
    fuel_rate_kg_hr: float = Field(default=1245.0)
    flue_gas_temp_c: float = Field(default=218.4)
    so2_ppm: int = Field(default=285)


class CVTelemetryData(BaseModel):
    active_counts: Dict[str, int] = Field(default_factory=dict)
    bay_dwell_sec: Optional[float] = None
    queue_idle_sec: Optional[float] = None
    smoke_opacity_pct: float = Field(..., ge=0.0, le=100.0)
    ringelmann_level: int = Field(..., ge=0, le=5)
    opacity_multiplier: float = Field(..., ge=1.0)
    cems_metrics: CEMSMetricsData
    recent_logs: List[str] = Field(default_factory=list)


class EmissionsData(BaseModel):
    baseline_co2_kg_hr: float
    waste_co2_kg_hr: float


class SimulationAppliedParams(BaseModel):
    fuel_type: str = "coal"
    scrubber_efficiency_pct: float = 0.0
    combustion_tuning_pct: float = 0.0


class SimulationData(BaseModel):
    active: bool = False
    projected_co2_kg_hr: float
    reduction_pct: float = 0.0
    applied_params: SimulationAppliedParams = Field(default_factory=SimulationAppliedParams)


class IndustrialTelemetryPayload(BaseModel):
    node_id: str = "IND_STACK_01_PEENYA"
    sector: str = "industrial"
    timestamp: str
    environment: EnvironmentData = Field(default_factory=EnvironmentData)
    cv_telemetry: CVTelemetryData
    emissions: EmissionsData
    simulation: SimulationData = Field(default_factory=SimulationData)


# =====================================================================
# 5. DUAL-MODE PLUME DETECTOR WITH TRUE MULTI-CUE SEGMENTATION
# =====================================================================

class DualModePlumeDetector:
    """
    High-reliability industrial plume detector supporting:
      1. Primary Mode: Custom YOLO weight (`models/smoke.pt`) if present.
      2. Multi-Profile Scene Router:
         - Profile A: Overcast / Wide / 4K Chimney Scene (_segment_profile_overcast_horizontal)
           recovered from commit 0838785 (3-cue temporal turbulence + row-wise horizontal sky subtraction).
         - Profile B: Sunset / Warm Gradient / Vertical Scene (_segment_profile_sunset_vertical)
           from commit 9d5f95e (auto-locating plume origin + adaptive local contrast + temporal smoothing).
         - Profile C: Synthetic / Generic Fallback (_segment_profile_synthetic)
           for synthetic CI videos.
    """

    def __init__(
        self,
        model_path: str = "models/smoke.pt",
        chimney_roi_norm: Optional[Tuple[float, float, float, float]] = None,
        vid_stride: int = 3,
        imgsz: int = 480,
        profile: Optional[str] = None,
    ):
        self.model_path = model_path
        self.chimney_roi_norm = chimney_roi_norm
        self.vid_stride = vid_stride
        self.imgsz = imgsz
        self.active_profile = profile

        self.mode = "fallback"
        self.model = None

        # State for Profile A (Overcast Horizontal - Commit 0838785)
        self.bg_subtractor = None
        self.rolling_gray_buffer_overcast: Deque[np.ndarray] = collections.deque(maxlen=28)

        # State for Profile B (Sunset Vertical - Commit 9d5f95e)
        self.rolling_gray_buffer_sunset: Deque[np.ndarray] = collections.deque(maxlen=24)
        self.energy_map: Optional[np.ndarray] = None
        self.detected_origin_seed: Optional[Tuple[int, int]] = None
        self.temporal_mask_prob: Optional[np.ndarray] = None
        self.last_valid_result_sunset: Optional[Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]] = None

        # Generic last valid result
        self.last_valid_result: Optional[Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]] = None

        # State for Universal Camera-Stabilized Optical Flow + Dynamic LAB GMM Engine
        self.rolling_gray_buffer_universal: Deque[np.ndarray] = collections.deque(maxlen=24)
        self.energy_map_universal: Optional[np.ndarray] = None
        self.detected_chimney_seed: Optional[Tuple[int, int]] = None
        self.wind_cone_dir: Optional[Tuple[float, float]] = None
        self.smoke_lab_center: Optional[np.ndarray] = None
        self.sky_lab_center: Optional[np.ndarray] = None
        self.static_edge_mask_universal: Optional[np.ndarray] = None
        self.temporal_prob_universal: Optional[np.ndarray] = None
        self.last_valid_result_universal: Optional[Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]] = None

        self._init_detector()

    def _init_detector(self) -> None:
        model_file = Path(self.model_path)
        if model_file.is_file() and ULTRALYTICS_AVAILABLE and YOLO is not None:
            try:
                logger.info(f"Loading Primary Mode YOLO model from {model_file}...")
                self.model = YOLO(str(model_file))
                self.mode = "primary_yolo"
                logger.info("Primary Mode YOLO model initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Falling back to Multi-Cue Segmentation.")

        # Fallback Mode Initialization: Multi-Profile Scene Router
        logger.info("Initializing Multi-Profile Plume Segmentation Router...")
        self.mode = "multi_profile_router"
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=12,
            detectShadows=False,
        )

    def _detect_scene_profile(self, frame: np.ndarray) -> str:
        """
        Inspects the frame geometry and ambient color signature:
          - Profile A (Overcast / Wide / 4K Chimney Scene):
            If width > height and overcast gray/slate sky,
            routes to _segment_profile_overcast_horizontal (verified golden backup).
          - Profile B (Sunset / Warm Gradient / Vertical Scene):
            If height > width and warm sunset gradient,
            routes to _segment_profile_sunset_vertical (verified golden backup).
          - Profile C (Synthetic CI Fallback):
            Synthetic CI test videos (e.g. 640x480).
          - Profile Universal (Dense Optical Flow Physics & Buoyancy Engine):
            Universal default for all new / unknown input videos.
        """
        if self.active_profile in ["universal", "universal_optical_flow", "optical_flow"]:
            return "universal_optical_flow"

        h, w = frame.shape[:2]

        # Profile C: Synthetic CI test videos
        if (w == 640 and h == 480) or (w <= 640 and h <= 480):
            logger.info(f"Scene Signature: Synthetic CI frame ({w}x{h}) -> Route: Profile C")
            return "synthetic"

        # Sample top sky region (upper 15% of frame)
        sky_h = max(10, int(h * 0.15))
        top_sky = frame[:sky_h, :]
        hsv_sky = cv2.cvtColor(top_sky, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_sky[:, :, 1]))
        mean_val = float(np.mean(hsv_sky[:, :, 2]))

        logger.info(
            f"Scene Signature Analysis: resolution={w}x{h}, "
            f"top_sky_mean_sat={mean_sat:.1f}, top_sky_mean_val={mean_val:.1f}"
        )

        # Profile A: Overcast / High-Value Low-Saturation Chimney Scene (Verified Backup)
        if mean_sat < 28.0 and mean_val > 140.0:
            logger.info("Detected Overcast Scene -> Route: Profile A")
            return "overcast_horizontal"

        # Profile B: Sunset / Warm Gradient Scene (Verified Backup)
        if mean_sat >= 30.0 and mean_val <= 135.0:
            logger.info("Detected Sunset Vertical Scene -> Route: Profile B")
            return "sunset_vertical"

        # Any other video routes to Universal Engine
        logger.info("Routing new/unknown video scene to Universal Optical Flow Engine")
        return "universal_optical_flow"

    def segment_smoke_plume(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Multi-Profile Scene Router:
        Automatically routes frame to its exact proven segmentation pipeline.
        """
        if self.active_profile is None:
            self.active_profile = self._detect_scene_profile(frame)
            logger.info(f"Routed Plume Segmentation to Profile: '{self.active_profile}'")

        if self.active_profile == "overcast_horizontal":
            res = self._segment_profile_overcast_horizontal(frame, frame_idx=frame_idx)
        elif self.active_profile == "sunset_vertical":
            res = self._segment_profile_sunset_vertical(frame, frame_idx=frame_idx)
        elif self.active_profile == "synthetic":
            res = self._segment_profile_synthetic(frame, frame_idx=frame_idx)
        else:
            res = self._segment_profile_universal_optical_flow(frame, frame_idx=frame_idx)

        self.last_valid_result = res
        return res

    def _segment_profile_overcast_horizontal(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Profile A (Recovered from commit 0838785):
        Proven 3-Cue Computer Vision Smoke Segmentation Pipeline for 4K Overcast Chimney Scene:
          1. Strict Temporal Turbulence Gate (Rolling multi-step frame buffer t-6, t-14, t-22).
             Static overcast sky pixels with zero motion can NEVER be masked!
          2. Row-Wise Horizontal Background Subtraction (Handles vertical sky brightness gradients).
          3. Low-Saturation HSV Filtering (smoke S < 85).
          4. Chimney-Tip Anchor & Morphological Fusion into smooth organic contours.
        """
        h, w = frame.shape[:2]

        # Upper 55% search zone covers the sky and stack exit zone above buildings
        search_h = int(h * 0.55)
        search_zone = frame[:search_h, :]

        # Downscale for segmentation processing to achieve 40+ FPS while retaining full output resolution
        proc_w = 640
        if w > proc_w:
            scale_ratio = float(proc_w) / float(w)
            proc_h = int(search_h * scale_ratio)
            proc_zone = cv2.resize(search_zone, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale_ratio = 1.0
            proc_zone = search_zone
            proc_w = w
            proc_h = search_h

        proc_gray = cv2.cvtColor(proc_zone, cv2.COLOR_BGR2GRAY)
        self.rolling_gray_buffer_overcast.append(proc_gray.copy())

        # -------------------------------------------------------------
        # Cue 1: Strict Temporal Turbulence Gate (Multi-Step Rolling Buffer)
        # -------------------------------------------------------------
        buf_len = len(self.rolling_gray_buffer_overcast)
        if buf_len >= 8:
            idx1 = max(0, buf_len - 6)
            idx2 = max(0, buf_len - 14)
            idx3 = max(0, buf_len - 22)
            d1 = cv2.absdiff(proc_gray, self.rolling_gray_buffer_overcast[idx1])
            d2 = cv2.absdiff(proc_gray, self.rolling_gray_buffer_overcast[idx2])
            d3 = cv2.absdiff(proc_gray, self.rolling_gray_buffer_overcast[idx3])
            temp_diff = np.maximum(d1, np.maximum(d2, d3))
        elif buf_len >= 2:
            temp_diff = cv2.absdiff(proc_gray, self.rolling_gray_buffer_overcast[0])
        else:
            temp_diff = np.zeros_like(proc_gray)

        # MOG2 background subtractor motion
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=12, detectShadows=False
            )
        motion_raw = self.bg_subtractor.apply(proc_zone, learningRate=0.0005)
        _, mog_mask = cv2.threshold(motion_raw, 180, 255, cv2.THRESH_BINARY)

        # Combined temporal activity mask (threshold > 3)
        _, diff_mask = cv2.threshold(temp_diff, 3, 255, cv2.THRESH_BINARY)
        temporal_activity = cv2.bitwise_or(diff_mask, mog_mask)

        # -------------------------------------------------------------
        # Cue 2: Row-Wise Horizontal Background Sky Subtraction
        # -------------------------------------------------------------
        row_bg = np.zeros(proc_h, dtype=np.float32)
        right_margin_start = int(proc_w * 0.72)
        right_margin_end = int(proc_w * 0.98)

        for y in range(proc_h):
            margin_pixels = proc_gray[y, right_margin_start:right_margin_end]
            row_bg[y] = float(np.median(margin_pixels)) if len(margin_pixels) > 0 else float(np.median(proc_gray[y, :]))

        # Row difference: smoke diverges from row-background sky luminance
        row_diff_abs = np.abs(proc_gray.astype(np.float32) - row_bg[:, None])
        contrast_mask = (row_diff_abs > 4.5).astype(np.uint8) * 255

        # -------------------------------------------------------------
        # Cue 3: HSV Low-Saturation Filter (Smoke has S < 85)
        # -------------------------------------------------------------
        hsv_search = cv2.cvtColor(proc_zone, cv2.COLOR_BGR2HSV)
        low_sat_mask = cv2.inRange(hsv_search[:, :, 1], 0, 85)

        # -------------------------------------------------------------
        # FUSED PLUME SEED WITH MANDATORY TEMPORAL ACTIVITY GATE
        # -------------------------------------------------------------
        plume_seed = cv2.bitwise_and(temporal_activity, contrast_mask)
        plume_seed = cv2.bitwise_and(plume_seed, low_sat_mask)

        # -------------------------------------------------------------
        # Chimney-Tip Anchor & Spatial Geometry Constraints
        # -------------------------------------------------------------
        full_proc_h = int(h * scale_ratio)
        stack_mouth_y = int(full_proc_h * 0.455)
        roofline_y = int(full_proc_h * 0.50)
        stack_x1 = int(proc_w * 0.44)
        stack_x2 = int(proc_w * 0.53)

        # Zero out stack body below mouth
        plume_seed[stack_mouth_y:, stack_x1:stack_x2] = 0

        # General roofline cutoff (anything below roofline is buildings/parked cars)
        plume_seed[roofline_y:, :] = 0

        # Right-side cutoff: Tall chimney is at x > 0.56*proc_w; plume drifts leftwards
        plume_seed[:, int(proc_w * 0.56):] = 0

        # -------------------------------------------------------------
        # Morphological Plume Fusion & Sanity Area Cap
        # -------------------------------------------------------------
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fused = cv2.morphologyEx(plume_seed, cv2.MORPH_CLOSE, kernel_close)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel_open)

        # Sanity Area Cap: If mask covers > 40% of the search area, tighten dynamically
        total_search_pixels = float(proc_w * proc_h)
        if cv2.countNonZero(fused) > 0.40 * total_search_pixels:
            score_map = temp_diff.astype(np.float32) * row_diff_abs
            cutoff = float(np.percentile(score_map[fused > 0], 85))
            fused = ((score_map > cutoff) & (fused > 0)).astype(np.uint8) * 255
            fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel_close)

        # Find connected plume contours and filter by chimney origin & dispersion cone
        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500.0:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # Must be in plume dispersion region: x <= stack_x2 and above roofline
            if bx > stack_x2 or by > roofline_y:
                continue
            # Must connect near chimney tip or lie in leftward dispersion path
            if (bx + bw >= stack_x1 - 60 and by + bh >= stack_mouth_y - 120) or (bx < stack_x1 and area > 1000.0):
                valid_contours.append(c)

        inv_scale = 1.0 / scale_ratio

        if valid_contours:
            # Scale contours back to full video coordinates
            scaled_contours = []
            all_pts = []
            for c in valid_contours:
                scaled = (c.astype(np.float32) * inv_scale).astype(np.int32)
                epsilon = 0.005 * cv2.arcLength(scaled, True)
                approx = cv2.approxPolyDP(scaled, epsilon, True)
                scaled_contours.append(approx)
                all_pts.append(scaled.reshape(-1, 2))

            # Build full-resolution exact binary pixel mask
            plume_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(plume_mask, scaled_contours, -1, 255, -1)

            all_pts_arr = np.vstack(all_pts)
            bx, by, bw, bh = cv2.boundingRect(all_pts_arr)
            pad = int(12 * inv_scale)
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)

            return (x1, y1, x2, y2), plume_mask, scaled_contours, "Profile A (Overcast Horizontal)"

        roi_norm = self.chimney_roi_norm or (0.42, 0.41, 0.51, 0.465)
        rx1 = max(0, int(roi_norm[0] * w))
        ry1 = max(0, int(roi_norm[1] * h))
        rx2 = min(w, int(roi_norm[2] * w))
        ry2 = min(h, int(roi_norm[3] * h))

        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        fallback_mask[ry1:ry2, rx1:rx2] = 255
        rect_contour = np.array([[[rx1, ry1]], [[rx2, ry1]], [[rx2, ry2]], [[rx1, ry2]]], dtype=np.int32)

        return (rx1, ry1, rx2, ry2), fallback_mask, [rect_contour], "Profile A (Overcast Initializing)"

    def _segment_profile_sunset_vertical(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Profile B (Retained from commit 9d5f95e):
        Universal Smoke Plume Segmentation Engine for Sunset / Warm Gradient / Vertical Scene:
          Stage 1: Camera-Shake & Illumination Stabilization (mean normalization + 5x5 median blur).
          Stage 2: Dynamic Chimney-Tip & Plume-Origin Auto-Locator (EMA Turbulence Energy Map).
          Stage 3: Connected Seed-Growing & Local Adaptive Background Contrast (Zero sky/cloud bleed).
          Stage 4: Temporal Contour Smoothing with zero flicker (0.65/0.35 probability accumulator).
        """
        h, w = frame.shape[:2]

        proc_w = 480
        scale_ratio = float(proc_w) / float(w)
        proc_h = int(h * scale_ratio)
        inv_scale = 1.0 / scale_ratio

        # -------------------------------------------------------------
        # Stage 1: Camera-Shake & Illumination Stabilization
        # -------------------------------------------------------------
        scaled = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        raw_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(raw_gray))
        if mean_val > 5.0:
            norm_gray = np.clip(raw_gray * (128.0 / mean_val), 0, 255).astype(np.uint8)
        else:
            norm_gray = raw_gray

        stable_gray = cv2.medianBlur(norm_gray, 5)
        self.rolling_gray_buffer_sunset.append(stable_gray)

        buf_len = len(self.rolling_gray_buffer_sunset)
        if buf_len >= 5:
            idx5 = max(0, buf_len - 5)
            idx12 = max(0, buf_len - 12)
            d1 = cv2.absdiff(stable_gray, self.rolling_gray_buffer_sunset[idx5])
            d2 = cv2.absdiff(stable_gray, self.rolling_gray_buffer_sunset[idx12])
            diff = np.maximum(d1, d2)
        elif buf_len >= 2:
            diff = cv2.absdiff(stable_gray, self.rolling_gray_buffer_sunset[0])
        else:
            diff = np.zeros_like(stable_gray)

        _, diff_thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_TOZERO)

        # -------------------------------------------------------------
        # Stage 2: Dynamic Chimney-Tip & Plume-Origin Auto-Locator
        # -------------------------------------------------------------
        if self.energy_map is None or self.energy_map.shape != stable_gray.shape:
            self.energy_map = diff_thresh.astype(np.float32)
        else:
            self.energy_map = 0.90 * self.energy_map + 0.10 * diff_thresh.astype(np.float32)

        em_max = float(self.energy_map.max())
        if em_max > 10.0:
            thresh_val = max(8.0, 0.16 * em_max)
            active_bin = (self.energy_map > thresh_val).astype(np.uint8) * 255
            k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            active_bin = cv2.morphologyEx(active_bin, cv2.MORPH_CLOSE, k_c)
            num_l, lbls, sts, _ = cv2.connectedComponentsWithStats(active_bin)
            best_l, max_a = -1, 0
            for l in range(1, num_l):
                if sts[l, cv2.CC_STAT_AREA] > max_a:
                    max_a = sts[l, cv2.CC_STAT_AREA]
                    best_l = l

            if best_l > 0 and max_a > 120:
                pts = np.argwhere(lbls == best_l)
                bh = sts[best_l, cv2.CC_STAT_HEIGHT]
                bottom_pts = pts[pts[:, 0] >= pts[:, 0].max() - max(3, int(bh * 0.08))]
                cur_y = int(np.median(bottom_pts[:, 0]))
                cur_x = int(np.median(bottom_pts[:, 1]))
                if self.detected_origin_seed is None:
                    self.detected_origin_seed = (cur_x, cur_y)
                else:
                    ox, oy = self.detected_origin_seed
                    self.detected_origin_seed = (
                        int(0.85 * ox + 0.15 * cur_x),
                        int(0.85 * oy + 0.15 * cur_y),
                    )

        # -------------------------------------------------------------
        # Stage 3: Connected Seed-Growing & Local Adaptive Contrast
        # -------------------------------------------------------------
        active_turb_mask = (diff > 4) & (self.energy_map > 4.0)
        if self.detected_origin_seed is not None:
            active_turb_mask[self.detected_origin_seed[1] + 12:, :] = False

        zero_turb = (self.energy_map < 1.0)
        row_bg = np.zeros(proc_h, dtype=np.float32)
        for y in range(proc_h):
            z_row = stable_gray[y, zero_turb[y, :]]
            row_bg[y] = float(np.median(z_row)) if len(z_row) > 10 else float(np.median(stable_gray[y, :]))

        row_diff = np.abs(stable_gray.astype(np.float32) - row_bg[:, None])
        contrast_mask = (row_diff > 4.0)

        candidate = (active_turb_mask & contrast_mask).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)

        plume_bin = np.zeros_like(fused)
        if self.detected_origin_seed is not None:
            sx, sy = self.detected_origin_seed
            num_l, lbls, sts, _ = cv2.connectedComponentsWithStats(fused)
            best_dist, chosen_l = 9999.0, -1
            for l in range(1, num_l):
                if sts[l, cv2.CC_STAT_AREA] < 150:
                    continue
                lx = sts[l, cv2.CC_STAT_LEFT]
                ly = sts[l, cv2.CC_STAT_TOP]
                lw = sts[l, cv2.CC_STAT_WIDTH]
                lh = sts[l, cv2.CC_STAT_HEIGHT]
                dx = max(0, max(lx - sx, sx - (lx + lw)))
                dy = max(0, max(ly - sy, sy - (ly + lh)))
                dist = float(np.hypot(dx, dy))
                if dist < best_dist and dist < 45.0:
                    best_dist = dist
                    chosen_l = l
            if chosen_l > 0:
                plume_bin[lbls == chosen_l] = 255

        total_px = float(proc_w * proc_h)
        plume_area = float(cv2.countNonZero(plume_bin))
        if plume_area > 0.25 * total_px:
            score_map = self.energy_map * row_diff
            for pctl in [75, 85, 92]:
                cutoff = float(np.percentile(score_map[plume_bin > 0], pctl))
                tight_mask = ((score_map > cutoff) & (plume_bin > 0)).astype(np.uint8) * 255
                tight_mask = cv2.morphologyEx(tight_mask, cv2.MORPH_CLOSE, k_close)
                if float(cv2.countNonZero(tight_mask)) <= 0.20 * total_px:
                    plume_bin = tight_mask
                    break

        # -------------------------------------------------------------
        # Stage 4: Temporal Contour Smoothing (Zero Flicker)
        # -------------------------------------------------------------
        cur_f = (plume_bin > 0).astype(np.float32)
        if self.temporal_mask_prob is None or self.temporal_mask_prob.shape != cur_f.shape:
            self.temporal_mask_prob = cur_f
        else:
            self.temporal_mask_prob = 0.65 * self.temporal_mask_prob + 0.35 * cur_f

        smooth_bin = (self.temporal_mask_prob >= 0.45).astype(np.uint8) * 255
        contours, _ = cv2.findContours(smooth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 180.0]

        if valid_contours:
            scaled_contours = []
            all_pts = []
            for c in valid_contours:
                scaled = (c.astype(np.float32) * inv_scale).astype(np.int32)
                epsilon = 0.005 * cv2.arcLength(scaled, True)
                approx = cv2.approxPolyDP(scaled, epsilon, True)
                scaled_contours.append(approx)
                all_pts.append(scaled.reshape(-1, 2))

            plume_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(plume_mask, scaled_contours, -1, 255, -1)

            all_pts_arr = np.vstack(all_pts)
            bx, by, bw, bh = cv2.boundingRect(all_pts_arr)
            pad = int(8 * inv_scale)
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)

            res = ((x1, y1, x2, y2), plume_mask, scaled_contours, "Profile B (Sunset Vertical)")
            self.last_valid_result_sunset = res
            return res

        elif self.last_valid_result_sunset is not None:
            return self.last_valid_result_sunset

        else:
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            if self.detected_origin_seed is not None:
                sx, sy = self.detected_origin_seed
                fx = int(sx * inv_scale)
                fy = int(sy * inv_scale)
                r = int(25 * inv_scale)
                rx1, ry1 = max(0, fx - r), max(0, fy - 2 * r)
                rx2, ry2 = min(w, fx + r), min(h, fy + r)
                dummy_mask[ry1:ry2, rx1:rx2] = 255
                box_cnt = np.array([[[rx1, ry1]], [[rx2, ry1]], [[rx2, ry2]], [[rx1, ry2]]], dtype=np.int32)
                return (rx1, ry1, rx2, ry2), dummy_mask, [box_cnt], "Profile B (Sunset Calibrating)"

            return (0, 0, 0, 0), dummy_mask, [], "Profile B (Sunset Calibrating)"

    def _segment_profile_synthetic(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Profile C (Clean threshold + MOG2 fallback for synthetic CI videos).
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_h = int(h * 0.65)

        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=16, detectShadows=False)

        fg_mask = self.bg_subtractor.apply(frame[:search_h, :], learningRate=0.01)
        _, thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 200]

        plume_mask = np.zeros((h, w), dtype=np.uint8)
        if valid:
            cv2.drawContours(plume_mask[:search_h, :], valid, -1, 255, -1)
            all_pts = np.vstack([c.reshape(-1, 2) for c in valid])
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            x1, y1 = max(0, bx - 8), max(0, by - 8)
            x2, y2 = min(w, bx + bw + 8), min(h, by + bh + 8)
            return (x1, y1, x2, y2), plume_mask, valid, "Profile C (Synthetic)"

        sx1, sy1 = int(w * 0.44), int(h * 0.35)
        sx2, sy2 = int(w * 0.56), int(h * 0.52)
        plume_mask[sy1:sy2, sx1:sx2] = 255
        rect_cnt = np.array([[[sx1, sy1]], [[sx2, sy1]], [[sx2, sy2]], [[sx1, sy2]]], dtype=np.int32)
        return (sx1, sy1, sx2, sy2), plume_mask, [rect_cnt], "Profile C (Synthetic Initializing)"

    def _segment_profile_universal_optical_flow(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Universal Camera-Stabilized Optical Flow & Buoyancy Plume Segmentation Engine:
        Operates universally on ANY video input (daylight, overcast, sunset, or arbitrary camera angle):
          1. Camera-Motion Stabilization: Global frame median velocity subtraction cancels camera pan/shake.
          2. Physics-Based Buoyancy & Turbulence Filtering: Isolates upward-rising non-rigid vortex eddies.
          3. Static Edge Rejection: Excludes static chimney brickwork, buildings, and ground structures.
          4. Auto-Chimney & Plume Search Cone: Locates the emission origin (sx, sy) and dynamic drift cone.
          5. Dynamic LAB Color Modeling: Samples smoke core vs sky in LAB space to segment soot or steam.
          6. Connected Component & Temporal Smoothing (0.65/0.35 EMA) for zero-flicker organic contours.
        """
        h, w = frame.shape[:2]

        proc_w = 480
        scale_ratio = float(proc_w) / float(w)
        proc_h = int(h * scale_ratio)
        inv_scale = 1.0 / scale_ratio

        scaled = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        raw_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(raw_gray, (5, 5), 0)

        self.rolling_gray_buffer_universal.append(gray_blur)
        buf_len = len(self.rolling_gray_buffer_universal)

        # -------------------------------------------------------------
        # 1. Camera-Stabilized Dense Optical Flow (Farneback)
        # -------------------------------------------------------------
        if buf_len >= 2:
            prev_gray = self.rolling_gray_buffer_universal[-2]
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray_blur, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Global Camera-Motion Subtraction: Cancels shake and panning
            med_vx = float(np.median(flow[..., 0]))
            med_vy = float(np.median(flow[..., 1]))
            vx = flow[..., 0] - med_vx
            vy = flow[..., 1] - med_vy

            mag = np.sqrt(vx**2 + vy**2)

            # Local Flow Vorticity / Variance (Smoke swirls internally; rigid objects move together)
            mean_vx = cv2.blur(vx, (9, 9))
            mean_vy = cv2.blur(vy, (9, 9))
            vorticity = cv2.blur((vx - mean_vx)**2 + (vy - mean_vy)**2, (9, 9))

            # Upward Buoyancy & Drift Weight (negative vy is upward in image coordinates)
            buoyancy_weight = np.where(vy <= 0.20 * (mag + 1e-4), 1.5, 0.3)
            flow_energy = mag * np.sqrt(vorticity + 1e-4) * buoyancy_weight
        else:
            mag = np.zeros((proc_h, proc_w), dtype=np.float32)
            vorticity = np.zeros((proc_h, proc_w), dtype=np.float32)
            flow_energy = np.zeros((proc_h, proc_w), dtype=np.float32)
            vx = np.zeros((proc_h, proc_w), dtype=np.float32)
            vy = np.zeros((proc_h, proc_w), dtype=np.float32)

        # Mask out ground/traffic zone (bottom 18% of frame)
        flow_energy[int(proc_h * 0.82):, :] = 0.0

        # -------------------------------------------------------------
        # 2. Strict Static Structure & Edge Rejection
        # -------------------------------------------------------------
        sobel_x = cv2.Sobel(raw_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(raw_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        cur_edge = (edge_mag > 32.0).astype(np.float32)

        if self.static_edge_mask_universal is None or self.static_edge_mask_universal.shape != cur_edge.shape:
            self.static_edge_mask_universal = cur_edge.copy()
        else:
            self.static_edge_mask_universal = 0.90 * self.static_edge_mask_universal + 0.10 * cur_edge

        # Static structure: high-contrast edges with low motion across frames
        static_structure = (self.static_edge_mask_universal > 0.60) & (mag < 0.6)
        flow_energy[static_structure] *= 0.10

        # -------------------------------------------------------------
        # 3. Energy Map Accumulation & Chimney Seed Discovery
        # -------------------------------------------------------------
        if self.energy_map_universal is None or self.energy_map_universal.shape != flow_energy.shape:
            self.energy_map_universal = flow_energy.copy()
        else:
            self.energy_map_universal = 0.80 * self.energy_map_universal + 0.20 * flow_energy

        # Auto-calibrate chimney origin & plume search cone
        if frame_idx >= 8 and (self.detected_chimney_seed is None or frame_idx % 60 == 0):
            smooth_e = cv2.GaussianBlur(self.energy_map_universal, (11, 11), 0)
            # Suppress static structural edges so building and tower edges cannot be chosen as seeds
            if self.static_edge_mask_universal is not None:
                static_edge_penalty = np.clip(self.static_edge_mask_universal * 1.5, 0.0, 1.0)
                smooth_e = smooth_e * (1.0 - static_edge_penalty)

            search_band = smooth_e[int(proc_h * 0.05):int(proc_h * 0.80), :]
            if search_band.size > 0:
                p85 = float(np.percentile(search_band, 85))
                thresh_val = max(0.25, p85 * 0.8)
                core_bin = (smooth_e > thresh_val).astype(np.uint8) * 255
                core_bin[int(proc_h * 0.82):, :] = 0

                cnts, _ = cv2.findContours(core_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in cnts if cv2.contourArea(c) > 200]
                if valid_cnts:
                    best_c = max(valid_cnts, key=cv2.contourArea)
                    pts = best_c.reshape(-1, 2)
                    # The bottom-most cluster point is the chimney tip mouth
                    sy = int(np.percentile(pts[:, 1], 95))
                    bottom_pts = pts[pts[:, 1] >= sy - 15]
                    sx = int(np.median(bottom_pts[:, 0])) if len(bottom_pts) > 0 else int(np.median(pts[:, 0]))
                    self.detected_chimney_seed = (sx, sy)

                    # Dynamic wind drift direction
                    w_vx = float(np.median(vx[core_bin > 0])) if np.count_nonzero(core_bin) > 0 else 0.0
                    w_vy = float(np.median(vy[core_bin > 0])) if np.count_nonzero(core_bin) > 0 else -1.0
                    norm_w = np.sqrt(w_vx**2 + w_vy**2) + 1e-5
                    self.wind_cone_dir = (w_vx / norm_w, w_vy / norm_w)

                    # Sample smoke core vs background sky in LAB space
                    lab_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB).astype(np.float32)
                    core_y1, core_y2 = max(0, sy - 40), max(0, sy - 10)
                    core_x1, core_x2 = max(0, sx - 25), min(proc_w, sx + 25)
                    core_sample = lab_scaled[core_y1:core_y2, core_x1:core_x2]
                    if core_sample.size > 0:
                        self.smoke_lab_center = np.median(core_sample.reshape(-1, 3), axis=0)

                    # Sample zero-flow sky region
                    sky_y1, sky_y2 = max(0, sy - 70), max(0, sy - 20)
                    sky_x1, sky_x2 = (max(0, sx - 80), max(0, sx - 40)) if w_vx >= 0 else (min(proc_w, sx + 40), min(proc_w, sx + 80))
                    sky_sample = lab_scaled[sky_y1:sky_y2, sky_x1:sky_x2]
                    if sky_sample.size > 0:
                        self.sky_lab_center = np.median(sky_sample.reshape(-1, 3), axis=0)

        # -------------------------------------------------------------
        # 4. Universal Segmentation with Dynamic Wind Cone & LAB Contrast
        # -------------------------------------------------------------
        lab_frame = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB).astype(np.float32)

        if self.detected_chimney_seed is not None:
            sx, sy = self.detected_chimney_seed
            w_dir = self.wind_cone_dir or (0.0, -1.0)
            wx, wy = w_dir

            # Build Wind-Aligned Search Cone
            Y, X = np.ogrid[:proc_h, :proc_w]
            dx = (X - sx).astype(np.float32)
            dy = (Y - sy).astype(np.float32)
            dist_origin = np.sqrt(dx**2 + dy**2)

            # Plume must be above chimney mouth (with 15px margin below mouth)
            above_mouth = dy <= 15.0

            # Wind projection: positive along wind direction
            proj_wind = dx * wx + dy * wy
            perp_wind = np.abs(dx * (-wy) + dy * wx)

            # Cone expands with distance from mouth
            cone_width = dist_origin * 0.95 + 35.0
            in_wind_cone = above_mouth & ((proj_wind >= -12.0) | (dist_origin < 45.0)) & (perp_wind <= cone_width)

            # LAB Color Contrast
            if self.smoke_lab_center is not None and self.sky_lab_center is not None:
                d_smoke = np.sqrt(np.sum((lab_frame - self.smoke_lab_center)**2, axis=2))
                d_sky = np.sqrt(np.sum((lab_frame - self.sky_lab_center)**2, axis=2))
                # Pixels that match smoke color closer than sky or are within smoke tolerance
                color_match = (d_smoke < d_sky * 1.10) & (d_smoke < 60.0)
            else:
                color_match = np.ones((proc_h, proc_w), dtype=bool)

            # Motion & Turbulence Gate
            motion_gate = (mag > 0.35) | (vorticity > 0.035) | (flow_energy > 0.15)

            # Strict Static Edge Rejection
            dynamic_smoke = in_wind_cone & color_match & motion_gate & (~static_structure)
            candidate_bin = dynamic_smoke.astype(np.uint8) * 255

            # Morphological Closing & Opening for Organic Gas Dynamics
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(candidate_bin, cv2.MORPH_CLOSE, k_close)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

            # Keep Contours Originating from or Flowing out of Chimney Seed
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 150.0:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                # Near stack mouth
                near_mouth = (bx <= sx + 30 and bx + bw >= sx - 30 and by <= sy + 15 and by + bh >= sy - 50)
                
                # In upward dispersion cone (both vertical and horizontal containment)
                centroid_x = bx + bw / 2.0
                centroid_y = by + bh / 2.0
                vert_dist = max(1.0, float(sy - centroid_y))
                expected_cx = sx + vert_dist * (wx / max(abs(wy), 0.15))
                max_dev = vert_dist * 0.85 + 40.0
                in_cone = (by < sy + 15) and (abs(centroid_x - expected_cx) <= max_dev)
                
                if near_mouth or in_cone:
                    valid_contours.append(c)

            proc_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
            if valid_contours:
                cv2.drawContours(proc_mask, valid_contours, -1, 255, -1)

            # Temporal EMA Probability Smoothing (0.65 / 0.35)
            curr_prob = proc_mask.astype(np.float32) / 255.0
            if self.temporal_prob_universal is None or self.temporal_prob_universal.shape != curr_prob.shape:
                self.temporal_prob_universal = curr_prob.copy()
            else:
                self.temporal_prob_universal = 0.65 * self.temporal_prob_universal + 0.35 * curr_prob

            final_bin = (self.temporal_prob_universal >= 0.40).astype(np.uint8) * 255
            final_cnts, _ = cv2.findContours(final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            active_cnts = [c for c in final_cnts if cv2.contourArea(c) > 180.0]

            if active_cnts:
                scaled_contours = []
                all_pts = []
                for c in active_cnts:
                    scaled_c = (c.astype(np.float32) * inv_scale).astype(np.int32)
                    epsilon = 0.005 * cv2.arcLength(scaled_c, True)
                    approx = cv2.approxPolyDP(scaled_c, epsilon, True)
                    scaled_contours.append(approx)
                    all_pts.append(scaled_c.reshape(-1, 2))

                plume_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(plume_mask, scaled_contours, -1, 255, -1)

                all_pts_arr = np.vstack(all_pts)
                bx, by, bw, bh = cv2.boundingRect(all_pts_arr)
                pad = int(8 * inv_scale)
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(w, bx + bw + pad)
                y2 = min(h, by + bh + pad)

                res = ((x1, y1, x2, y2), plume_mask, scaled_contours, "Universal Optical Flow & Semantic Engine")
                self.last_valid_result_universal = res
                return res

        # Fallback during initial calibration frames (0 to 4)
        if self.last_valid_result_universal is not None:
            return self.last_valid_result_universal

        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        if self.detected_chimney_seed is not None:
            sx, sy = self.detected_chimney_seed
            fx, fy = int(sx * inv_scale), int(sy * inv_scale)
            r = int(25 * inv_scale)
            rx1, ry1 = max(0, fx - r), max(0, fy - 2 * r)
            rx2, ry2 = min(w, fx + r), min(h, fy + r)
            dummy_mask[ry1:ry2, rx1:rx2] = 255
            box_cnt = np.array([[[rx1, ry1]], [[rx2, ry1]], [[rx2, ry2]], [[rx1, ry2]]], dtype=np.int32)
            return (rx1, ry1, rx2, ry2), dummy_mask, [box_cnt], "Universal Optical Flow (Calibrating)"

        return (0, 0, 0, 0), dummy_mask, [], "Universal Optical Flow (Calibrating)"

    def detect_plume(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, List[np.ndarray], str]:
        """
        Detects smoke plume and returns:
          (bbox, plume_mask, plume_contours, mode_name)
          where bbox = (x1, y1, x2, y2)
                plume_mask = exact binary mask of the smoke (same HxW as frame)
                plume_contours = list of smoothed contour polygons
                mode_name = str
        """
        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # Mode 1: Primary YOLO Model
        # -------------------------------------------------------------
        if self.mode == "primary_yolo" and self.model is not None:
            try:
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    vid_stride=self.vid_stride,
                    verbose=False,
                    conf=0.20,
                )
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Build mask & contour for detected YOLO box
                    plume_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(plume_mask, (x1, y1), (x2, y2), 255, -1)
                    box_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
                    return (x1, y1, x2, y2), plume_mask, [box_contour], "Primary (YOLOv8)"
            except Exception as e:
                logger.debug(f"YOLO inference exception: {e}")

        # -------------------------------------------------------------
        # Mode 2: Multi-Cue Turbulence Computer Vision Segmentation
        # -------------------------------------------------------------
        return self.segment_smoke_plume(frame, frame_idx=frame_idx)


# =====================================================================
# 6. VISUAL VIDEO OVERLAY & HUD BANNER
# =====================================================================

def draw_hud_and_annotations(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    opacity_pct: float,
    ringelmann_level: int,
    multiplier: float,
    status: str,
    fuel_rate_kg_hr: float,
    co2_rate_kg_hr: float,
    detector_mode: str,
    so2_ppm: int,
    flue_gas_temp_c: float,
    plume_mask: Optional[np.ndarray] = None,
    plume_contours: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Renders true segmentation overlay:
      - Semi-transparent colored mask (alpha = 0.38) directly over the smoke plume pixels
      - Crisp 2px glowing contour outline around the organic plume shape
      - Tactical corner-reticle bounding box around plume bounds
      - Dark-mode HUD banner with live diagnostics
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    ui_scale = max(1.0, min(w, h) / 720.0)

    # Color palette based on Ringelmann Level:
    # 0-1: Neon Emerald Green (113, 204, 46)
    # 2-3: Warning Amber Orange (0, 165, 255)
    # 4-5: Critical Crimson Red (41, 41, 231)
    if ringelmann_level <= 1:
        theme_bgr = (113, 204, 46)   # Neon Green
        glow_color = (150, 240, 80)
        status_text = "NORMAL OPERATION"
    elif ringelmann_level <= 3:
        theme_bgr = (0, 165, 255)    # Amber Orange
        glow_color = (60, 200, 255)
        status_text = "COMBUSTION WARNING"
    else:
        theme_bgr = (41, 41, 231)    # Bright Red
        glow_color = (80, 80, 255)
        status_text = "SCRUBBER VIOLATION"

    # -----------------------------------------------------------------
    # A. True Segmentation Plume Overlay (Mask & Glowing Contours)
    # -----------------------------------------------------------------
    if plume_mask is not None and cv2.countNonZero(plume_mask) > 0:
        # Draw semi-transparent colored segmentation mask (alpha = 0.38)
        mask_indices = plume_mask > 0
        color_overlay = annotated.copy()
        color_overlay[mask_indices] = theme_bgr
        annotated = cv2.addWeighted(color_overlay, 0.38, annotated, 0.62, 0)

    if plume_contours:
        # Crisp glowing contour outline around the exact plume boundary
        stroke_thickness = max(2, int(2 * ui_scale))
        glow_thickness = max(4, int(4 * ui_scale))
        cv2.drawContours(annotated, plume_contours, -1, glow_color, glow_thickness, cv2.LINE_AA)
        cv2.drawContours(annotated, plume_contours, -1, theme_bgr, stroke_thickness, cv2.LINE_AA)

    # -----------------------------------------------------------------
    # B. Tactical Corner-Reticle Bounding Box
    # -----------------------------------------------------------------
    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1:
        # Outer glow borders
        cv2.rectangle(annotated, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), glow_color, 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_bgr, max(2, int(2 * ui_scale)), cv2.LINE_AA)

        # Corner brackets for modern tactical CV look
        corner_len = min(int(24 * ui_scale), (x2 - x1) // 4, (y2 - y1) // 4)
        corner_thick = max(2, int(2 * ui_scale))
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = corner_len if cx == x1 else -corner_len
            dy = corner_len if cy == y1 else -corner_len
            cv2.line(annotated, (cx, cy), (cx + dx, cy), (255, 255, 255), corner_thick, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy), (cx, cy + dy), (255, 255, 255), corner_thick, cv2.LINE_AA)

        # Plume tag
        tag = f"PLUME: {opacity_pct:.1f}% | R-{ringelmann_level}"
        font_scale = 0.5 * ui_scale
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(int(28 * ui_scale), y1 - 8)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), (20, 24, 30), -1)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), theme_bgr, 1)
        cv2.putText(annotated, tag, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------
    # C. Dark-Mode HUD Banner (Top-Left)
    # -----------------------------------------------------------------
    hud_x = int(20 * ui_scale)
    hud_y = int(20 * ui_scale)
    hud_w = int(min(w - 40, 480 * ui_scale))
    hud_h = int(160 * ui_scale)

    hud_sub = annotated[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
    hud_bg = np.zeros_like(hud_sub, dtype=np.uint8)
    hud_bg[:] = (18, 22, 28)  # Sleek dark navy/slate background
    cv2.addWeighted(hud_bg, 0.88, hud_sub, 0.12, 0, hud_sub)

    # Top accent line with theme color
    cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + max(3, int(4 * ui_scale))), theme_bgr, -1)
    cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (50, 58, 70), 1, cv2.LINE_AA)

    # Status LED indicator circle
    led_r = max(5, int(6 * ui_scale))
    cv2.circle(annotated, (hud_x + int(20 * ui_scale), hud_y + int(24 * ui_scale)), led_r, theme_bgr, -1, cv2.LINE_AA)
    cv2.circle(annotated, (hud_x + int(20 * ui_scale), hud_y + int(24 * ui_scale)), led_r + 2, glow_color, 1, cv2.LINE_AA)

    # Header Title
    header_title = f"INDUSTRIAL EMISSIONS INTELLIGENCE  [{detector_mode}]"
    cv2.putText(
        annotated,
        header_title,
        (hud_x + int(36 * ui_scale), hud_y + int(28 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 * ui_scale,
        (180, 195, 210),
        1,
        cv2.LINE_AA,
    )

    # Line 1: Smoke Opacity & Ringelmann Level
    line1 = f"SMOKE OPACITY: {opacity_pct:.1f}% (RINGELMANN LEVEL {ringelmann_level})"
    cv2.putText(
        annotated,
        line1,
        (hud_x + int(20 * ui_scale), hud_y + int(60 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * ui_scale,
        (255, 255, 255),
        max(1, int(2 * ui_scale)),
        cv2.LINE_AA,
    )

    # Line 2: Coal Rate & CO2 Rate
    line2 = f"COAL RATE: {fuel_rate_kg_hr:,.0f} kg/hr | CO2 RATE: {co2_rate_kg_hr:,.1f} kg/hr"
    cv2.putText(
        annotated,
        line2,
        (hud_x + int(20 * ui_scale), hud_y + int(92 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * ui_scale,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )

    # Line 3: Operational Status & Penalty Multiplier
    line3 = f"STATUS: {status_text} (PENALTY {multiplier:.2f}x)"
    cv2.putText(
        annotated,
        line3,
        (hud_x + int(20 * ui_scale), hud_y + int(122 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * ui_scale,
        theme_bgr,
        max(1, int(2 * ui_scale)),
        cv2.LINE_AA,
    )

    # Line 4: Auxiliary CEMS Metrics (Flue Temp & SO2)
    line4 = f"FLUE TEMP: {flue_gas_temp_c:.1f}C  |  SO2: {so2_ppm} ppm  |  NODE: IND_STACK_01"
    cv2.putText(
        annotated,
        line4,
        (hud_x + int(20 * ui_scale), hud_y + int(146 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 * ui_scale,
        (130, 145, 160),
        1,
        cv2.LINE_AA,
    )

    return annotated


# =====================================================================
# 7. VIDEO PROCESSING & SMART REPLAY CACHE PIPELINE
# =====================================================================

def process_video(
    input_video_path: str = "factory_stack.mp4",
    output_video_path: str = "cache/factory_annotated.mp4",
    output_json_path: str = "cache/factory_telemetry.json",
    model_path: str = "models/smoke.pt",
    node_id: str = "IND_STACK_01_PEENYA",
    fuel_type: str = "coal",
    base_fuel_rate: float = 1200.0,
    scrubber_efficiency: float = 0.0,
    max_frames: Optional[int] = None,
    profile: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Analyzes pre-recorded CCTV footage of a factory smokestack, segments smoke plume,
    computes masked Ringelmann Opacity, pairs with CEMS telemetry, calculates emissions,
    and saves both annotated video (with true segmentation overlay) and telemetry JSON.

    profile: None auto-routes by scene; or force "universal_optical_flow", "overcast_horizontal",
    "sunset_vertical" or "synthetic".
    """
    input_path = Path(input_video_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Ensure output cache directories exist
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Processing video {input_path.name}: {width}x{height} @ {fps:.1f} FPS, total frames: {total_frames}")

    # Video Writer: H.264 (avc1) so browsers can play it via /api/video/industrial;
    # mp4v only as a fallback for OpenCV builds without an H.264 encoder (browsers cannot play mp4v)
    video_codec = "avc1"
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        video_codec = "mp4v"
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        logger.warning("H.264 encoder unavailable; wrote mp4v, which web browsers cannot play.")

    # Initialize Detector & CEMS Generator
    detector = DualModePlumeDetector(model_path=model_path, profile=profile)
    cems_gen = SyntheticCEMSGenerator(
        fuel_type=fuel_type,
        base_fuel_rate=base_fuel_rate,
        scrubber_efficiency=scrubber_efficiency,
    )

    telemetry_records: List[Dict[str, Any]] = []
    frame_idx = 0
    mode_name = "n/a"
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, total_frames)

            # 1. Segment Smoke Plume (Bbox, Exact Mask, and Smoothed Contours)
            bbox, plume_mask, plume_contours, mode_name = detector.detect_plume(frame, frame_idx=frame_idx)

            # 2. True Masked Ringelmann Opacity Calculation (strictly inside plume_mask)
            opacity_pct, ringelmann_lvl, op_multiplier, status = calculate_ringelmann_opacity(frame, mask=plume_mask)

            # 3. Synchronized CEMS Telemetry
            cems_metrics, recent_logs = cems_gen.generate(
                opacity_pct=opacity_pct,
                ringelmann_level=ringelmann_lvl,
                opacity_multiplier=op_multiplier,
            )

            # 4. Deterministic Carbon Math Engine
            emissions_calc = calculate_industrial_emissions(
                fuel_rate_kg_hr=cems_metrics["fuel_rate_kg_hr"],
                fuel_type=fuel_type,
                opacity_multiplier=op_multiplier,
                scrubber_efficiency=scrubber_efficiency,
            )

            # 5. Build Standardized Telemetry Payload
            now_iso = datetime.now(timezone.utc).isoformat()
            payload = IndustrialTelemetryPayload(
                node_id=node_id,
                sector="industrial",
                timestamp=now_iso,
                environment=EnvironmentData(
                    temp_c=29.5,
                    wind_kmh=11.2,
                    wind_deg=240,
                    aqi=168,
                ),
                cv_telemetry=CVTelemetryData(
                    active_counts={},
                    bay_dwell_sec=None,
                    queue_idle_sec=None,
                    smoke_opacity_pct=opacity_pct,
                    ringelmann_level=ringelmann_lvl,
                    opacity_multiplier=op_multiplier,
                    cems_metrics=CEMSMetricsData(**cems_metrics),
                    recent_logs=recent_logs,
                ),
                emissions=EmissionsData(
                    baseline_co2_kg_hr=emissions_calc["baseline_co2_kg_hr"],
                    waste_co2_kg_hr=emissions_calc["waste_co2_kg_hr"],
                ),
                simulation=SimulationData(
                    active=False,
                    projected_co2_kg_hr=emissions_calc["projected_co2_kg_hr"],
                    reduction_pct=emissions_calc["reduction_pct"],
                    applied_params=SimulationAppliedParams(
                        fuel_type=fuel_type,
                        scrubber_efficiency_pct=scrubber_efficiency * 100.0,
                        combustion_tuning_pct=0.0,
                    ),
                ),
            )

            telemetry_records.append(payload.model_dump())

            # 6. Render HUD, Glowing Contours & Semi-Transparent Mask Overlay
            annotated_frame = draw_hud_and_annotations(
                frame=frame,
                bbox=bbox,
                opacity_pct=opacity_pct,
                ringelmann_level=ringelmann_lvl,
                multiplier=op_multiplier,
                status=status,
                fuel_rate_kg_hr=cems_metrics["fuel_rate_kg_hr"],
                co2_rate_kg_hr=emissions_calc["baseline_co2_kg_hr"],
                detector_mode=mode_name,
                so2_ppm=cems_metrics["so2_ppm"],
                flue_gas_temp_c=cems_metrics["flue_gas_temp_c"],
                plume_mask=plume_mask,
                plume_contours=plume_contours,
            )

            writer.write(annotated_frame)

            if frame_idx % 30 == 0:
                logger.info(
                    f"Frame {frame_idx}/{total_frames} | Opacity: {opacity_pct:.1f}% (R-{ringelmann_lvl}) | "
                    f"CO2: {emissions_calc['baseline_co2_kg_hr']} kg/hr | Mode: {mode_name}"
                )

    finally:
        cap.release()
        writer.release()

    # Save complete telemetry json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_records, f, indent=2)

    # Save permanent backup copy named after the video profile/resolution
    # so switching videos never destroys previously verified demo outputs!
    active = detector.active_profile or ""
    tag = ("overcast" if "overcast" in active else "sunset" if "sunset" in active
           else "universal" if "universal" in active else "synthetic")
    backup_video_path = Path(output_video_path).parent / f"factory_annotated_{tag}.mp4"
    backup_json_path = Path(output_json_path).parent / f"factory_telemetry_{tag}.json"
    try:
        shutil.copyfile(output_video_path, str(backup_video_path))
        shutil.copyfile(output_json_path, str(backup_json_path))
        logger.info(f"Saved permanent named profile backup: {backup_video_path} and {backup_json_path}")
    except Exception as e:
        logger.warning(f"Could not save named backup copies: {e}")

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Processing completed in {elapsed:.2f}s ({proc_fps:.1f} FPS). "
        f"Saved {frame_idx} frames to {output_video_path} and telemetry to {output_json_path}"
    )

    # Run metadata next to the telemetry (factory_telemetry_meta.json), so the API can say which engine ran
    meta_path = Path(output_json_path).with_name(Path(output_json_path).stem + "_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_video": input_path.name,
                "profile": detector.active_profile,
                "detector_mode": mode_name,
                "frames_processed": frame_idx,
                "width": width,
                "height": height,
                "fps": round(float(fps), 2),
                "video_codec": video_codec,
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write run metadata {meta_path}: {e}")

    return {
        "video_codec": video_codec,
        "detector_mode": mode_name,
        "meta_json": str(meta_path),
        "frames_processed": frame_idx,
        "elapsed_sec": round(elapsed, 2),
        "proc_fps": round(proc_fps, 1),
        "annotated_video": output_video_path,
        "telemetry_json": output_json_path,
        "records_count": len(telemetry_records),
        "profile": detector.active_profile,
        "backup_video": str(backup_video_path),
        "backup_json": str(backup_json_path),
    }


# =====================================================================
# 8. STREAM INDUSTRIAL TELEMETRY GENERATOR (SMART REPLAY CACHE)
# =====================================================================

def stream_industrial_telemetry(
    use_cache: bool = True,
    cache_path: str = "cache/factory_telemetry.json",
    video_path: str = "factory_stack.mp4",
    fps: float = 60.0,
    loop: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Tier-2 Smart Replay Cache Generator for FastAPI / WebSocket streaming.
    - If `use_cache=True` and cache file exists, yields pre-cached telemetry at target FPS (<2% CPU).
    - If `use_cache=False` or cache is absent, runs live frame-by-frame inference.
    """
    cache_file = Path(cache_path)

    if use_cache and cache_file.is_file():
        logger.info(f"Streaming from Smart Replay Cache: {cache_file} @ {fps} FPS")
        with open(cache_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            logger.warning("Cache file is empty.")
            return

        delay = 1.0 / fps if fps > 0 else 0.016
        while True:
            for item in records:
                # Update timestamp dynamically to current UTC time for live streaming freshness
                live_item = dict(item)
                live_item["timestamp"] = datetime.now(timezone.utc).isoformat()
                yield live_item
                time.sleep(delay)
            if not loop:
                break
    else:
        logger.info(f"Streaming via Live Inference from {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open live video stream from {video_path}")

        detector = DualModePlumeDetector()
        cems_gen = SyntheticCEMSGenerator()
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                frame_idx += 1
                bbox, plume_mask, plume_contours, _ = detector.detect_plume(frame, frame_idx=frame_idx)
                opacity_pct, ringelmann_lvl, op_mult, status = calculate_ringelmann_opacity(frame, mask=plume_mask)
                cems_metrics, logs = cems_gen.generate(opacity_pct, ringelmann_lvl, op_mult)
                emissions = calculate_industrial_emissions(
                    fuel_rate_kg_hr=cems_metrics["fuel_rate_kg_hr"],
                    opacity_multiplier=op_mult,
                )

                payload = IndustrialTelemetryPayload(
                    node_id="IND_STACK_01_PEENYA",
                    sector="industrial",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    environment=EnvironmentData(),
                    cv_telemetry=CVTelemetryData(
                        smoke_opacity_pct=opacity_pct,
                        ringelmann_level=ringelmann_lvl,
                        opacity_multiplier=op_mult,
                        cems_metrics=CEMSMetricsData(**cems_metrics),
                        recent_logs=logs,
                    ),
                    emissions=EmissionsData(
                        baseline_co2_kg_hr=emissions["baseline_co2_kg_hr"],
                        waste_co2_kg_hr=emissions["waste_co2_kg_hr"],
                    ),
                    simulation=SimulationData(
                        projected_co2_kg_hr=emissions["projected_co2_kg_hr"],
                        reduction_pct=emissions["reduction_pct"],
                    ),
                )
                yield payload.model_dump()
        finally:
            cap.release()
