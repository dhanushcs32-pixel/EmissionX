"""Industrial Emissions Module (Pillar 2) for Carbon Footprint Intelligence Platform.

Modular, production-ready Python pipeline analyzing CCTV smokestack footage,
detecting smoke plumes, computing Ringelmann Opacity, generating synthetic CEMS telemetry,
calculating real-time CO2 emissions, and producing annotated video + telemetry JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

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


def calculate_industrial_emissions(
    fuel_rate_kg_hr: float,
    fuel_type: str = "coal",
    opacity_multiplier: float = 1.0,
    scrubber_efficiency: float = 0.0,
    combustion_tuning_pct: float = 0.0,
) -> Dict[str, float]:
    """
    Pure, deterministic calculation engine for baseline & what-if simulation emissions.
    Shared across video frame processing and FastAPI /api/simulate endpoints.

    Formula:
        baseline_co2_kg_hr = (fuel_rate_kg_hr * emission_factor) * opacity_multiplier * (1.0 - scrubber_efficiency)
        waste_co2_kg_hr = baseline_co2_kg_hr - (fuel_rate_kg_hr * emission_factor)
    """
    norm_fuel = fuel_type.lower().strip().replace(" ", "_")
    emission_factor = EMISSION_FACTORS.get(norm_fuel, 2.42)

    # Scrubber efficiency clamp [0.0, 1.0]
    scrubber_eff = max(0.0, min(1.0, float(scrubber_efficiency)))
    op_mult = max(1.0, float(opacity_multiplier))

    # Clean theoretical combustion rate
    ideal_co2_kg_hr = fuel_rate_kg_hr * emission_factor

    # Actual baseline emission rate taking opacity penalty and scrubber into account
    raw_baseline = ideal_co2_kg_hr * op_mult * (1.0 - scrubber_eff)
    baseline_co2_kg_hr = round(raw_baseline, 1)

    # Excess carbon caused by incomplete combustion/scrubber deficiency
    raw_waste = max(0.0, baseline_co2_kg_hr - ideal_co2_kg_hr)
    waste_co2_kg_hr = round(raw_waste, 1)

    # What-if simulation projection (if combustion tuning or scrubber upgrade is simulated)
    tuning_reduction = max(0.0, min(1.0, float(combustion_tuning_pct) / 100.0))
    sim_multiplier = max(1.0, 1.0 + (op_mult - 1.0) * (1.0 - tuning_reduction))
    raw_projected = ideal_co2_kg_hr * sim_multiplier * (1.0 - scrubber_eff)
    projected_co2_kg_hr = round(raw_projected, 1)

    if baseline_co2_kg_hr > 0.0:
        reduction_pct = round(
            max(0.0, ((baseline_co2_kg_hr - projected_co2_kg_hr) / baseline_co2_kg_hr) * 100.0),
            1,
        )
    else:
        reduction_pct = 0.0

    return {
        "baseline_co2_kg_hr": baseline_co2_kg_hr,
        "waste_co2_kg_hr": waste_co2_kg_hr,
        "projected_co2_kg_hr": projected_co2_kg_hr,
        "reduction_pct": reduction_pct,
        "emission_factor": emission_factor,
        "ideal_co2_kg_hr": round(ideal_co2_kg_hr, 1),
    }


# =====================================================================
# 2. RINGELMANN SMOKE OPACITY CALCULATION
# =====================================================================

def calculate_ringelmann_opacity(
    roi_bgr: np.ndarray,
) -> Tuple[float, int, float, str]:
    """
    Extract the Region of Interest (ROI) of the detected smoke plume,
    convert to grayscale, and compute mean pixel brightness.

    Opacity Formula:
        opacity_pct = round(max(0.0, min(100.0, (1.0 - (mean_brightness / 255.0)) * 100.0)), 1)

    Classification:
        * 0% - 20%  (Ringelmann 0-1: Clean steam / normal operation) -> multiplier = 1.0, status = "NORMAL"
        * 21% - 60% (Ringelmann 2-3: Incomplete combustion warning)  -> multiplier = 1.25, status = "COMBUSTION WARNING"
        * > 60%     (Ringelmann 4-5: Black carbon violation / Scrubber failure) -> multiplier = 1.60, status = "SCRUBBER VIOLATION"
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0, 1.0, "NORMAL"

    if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr

    mean_brightness = float(np.mean(gray))
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
            # Scale smoothly up towards 1245 - 1280 kg/hr during high opacity
            spike = (opacity_pct - 50.0) * 0.95 + 15.0

        fuel_rate = round(self.base_fuel_rate + micro_fluctuation + spike, 1)

        # 2. Flue Gas Temp: ~160°C to 240°C scaling with opacity
        base_temp = 160.0 + (opacity_pct / 100.0) * 80.0
        temp_noise = math.sin(t * 0.8) * 2.5 + random.uniform(-1.0, 1.0)
        flue_gas_temp_c = round(max(150.0, min(260.0, base_temp + temp_noise)), 1)

        # 3. SO2 PPM: ~80 to 350 ppm (spikes sharply during Ringelmann 4-5 violations)
        if ringelmann_level >= 4 or opacity_pct > 60.0:
            # Severe violation range: 260 - 350 ppm
            base_so2 = 250.0 + (opacity_pct - 60.0) * 2.5
            so2_noise = random.randint(-10, 15)
            so2_ppm = int(max(240, min(360, base_so2 + so2_noise)))
        elif opacity_pct > 20.0:
            # Combustion warning range: 130 - 240 ppm
            base_so2 = 120.0 + (opacity_pct - 20.0) * 3.0
            so2_noise = random.randint(-8, 8)
            so2_ppm = int(max(90, min(240, base_so2 + so2_noise)))
        else:
            # Normal clean range: 80 - 130 ppm
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
# 5. DUAL-MODE PLUME DETECTOR (PRIMARY YOLO / FALLBACK OPENCV MOG2)
# =====================================================================

class DualModePlumeDetector:
    """
    High-reliability industrial plume detector supporting:
      1. Primary Mode: Custom YOLO weight (`models/smoke.pt`) if present.
      2. Fallback Mode: Zero-weight OpenCV ROI + MOG2 Background Subtraction
         + Pixel Intensity Analysis, operating reliably on any footage.
    """

    def __init__(
        self,
        model_path: str = "models/smoke.pt",
        chimney_roi_norm: Tuple[float, float, float, float] = (0.28, 0.08, 0.72, 0.50),
        vid_stride: int = 3,
        imgsz: int = 480,
    ):
        self.model_path = model_path
        self.chimney_roi_norm = chimney_roi_norm
        self.vid_stride = vid_stride
        self.imgsz = imgsz

        self.mode = "fallback"
        self.model = None
        self.bg_subtractor = None

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
                logger.warning(f"Failed to load YOLO model: {e}. Falling back to OpenCV MOG2 mode.")

        # Fallback Mode Initialization
        logger.info("Initializing Fallback Mode: OpenCV MOG2 + Chimney ROI Analysis...")
        self.mode = "fallback_opencv"
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=False,
        )

    def detect_plume(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray, str]:
        """
        Detects smoke plume bounding box and returns:
          (bbox, plume_roi, mode_name)
          where bbox = (x1, y1, x2, y2)
        """
        h, w = frame.shape[:2]

        # Calculate default Chimney ROI in pixel coordinates
        rx1 = max(0, int(self.chimney_roi_norm[0] * w))
        ry1 = max(0, int(self.chimney_roi_norm[1] * h))
        rx2 = min(w, int(self.chimney_roi_norm[2] * w))
        ry2 = min(h, int(self.chimney_roi_norm[3] * h))

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
                    # Bound to frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        return (x1, y1, x2, y2), roi, "Primary (YOLOv8)"
            except Exception as e:
                logger.debug(f"YOLO inference exception: {e}")

        # -------------------------------------------------------------
        # Mode 2: Zero-Weight Fallback OpenCV MOG2 + Motion ROI
        # -------------------------------------------------------------
        roi_zone = frame[ry1:ry2, rx1:rx2]
        if roi_zone.size == 0:
            return (rx1, ry1, rx2, ry2), frame, "Fallback (ROI)"

        # Update MOG2 background subtractor
        fg_mask = self.bg_subtractor.apply(roi_zone)

        # Morphological filter to isolate plume motion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.dilate(fg_clean, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        max_area = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area > 350.0:
                bx, by, bw, bh = cv2.boundingRect(c)
                if area > max_area:
                    max_area = area
                    best_bbox = (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh)

        if best_bbox is not None:
            bx1, by1, bx2, by2 = best_bbox
            # Smooth padding around detected motion
            pad = 12
            bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
            bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)
            plume_roi = frame[by1:by2, bx1:bx2]
            return (bx1, by1, bx2, by2), plume_roi, "Fallback (MOG2 Motion)"

        # If no significant motion is detected, fall back to default stack zone
        return (rx1, ry1, rx2, ry2), roi_zone, "Fallback (Stack ROI)"


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
) -> np.ndarray:
    """
    Renders high-contrast dark-mode HUD banner and glowing bounding box overlay.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Color palette based on Ringelmann Level:
    # 0-1: Neon Emerald Green (46, 204, 113)
    # 2-3: Warning Amber Orange (0, 165, 255)
    # 4-5: Critical Crimson Red (41, 41, 231)
    if ringelmann_level <= 1:
        theme_bgr = (113, 204, 46)   # Green
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
    # A. Glowing Bounding Box around Smoke Plume ROI
    # -----------------------------------------------------------------
    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1:
        # Create glowing semi-transparent overlay
        glow_overlay = annotated.copy()
        cv2.rectangle(glow_overlay, (x1, y1), (x2, y2), theme_bgr, -1)
        cv2.addWeighted(glow_overlay, 0.12, annotated, 0.88, 0, annotated)

        # Outer glow borders
        cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), glow_color, 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_bgr, 3, cv2.LINE_AA)

        # Corner brackets for modern tactical CV look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = corner_len if cx == x1 else -corner_len
            dy = corner_len if cy == y1 else -corner_len
            cv2.line(annotated, (cx, cy), (cx + dx, cy), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy), (cx, cy + dy), (255, 255, 255), 2, cv2.LINE_AA)

        # Plume tag
        tag = f"PLUME: {opacity_pct:.1f}% | R-{ringelmann_level}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(24, y1 - 8)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + 2), (20, 24, 30), -1)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + 2), theme_bgr, 1)
        cv2.putText(annotated, tag, (x1 + 5, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------
    # B. Dark-Mode HUD Banner (Top-Left)
    # -----------------------------------------------------------------
    hud_x, hud_y = 20, 20
    hud_w, hud_h = min(480, w - 40), 160

    hud_sub = annotated[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
    hud_bg = np.zeros_like(hud_sub, dtype=np.uint8)
    hud_bg[:] = (18, 22, 28)  # Sleek dark navy/slate background
    cv2.addWeighted(hud_bg, 0.88, hud_sub, 0.12, 0, hud_sub)

    # Top accent line with theme color
    cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + 4), theme_bgr, -1)
    cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (50, 58, 70), 1, cv2.LINE_AA)

    # Status LED indicator circle
    cv2.circle(annotated, (hud_x + 20, hud_y + 24), 6, theme_bgr, -1, cv2.LINE_AA)
    cv2.circle(annotated, (hud_x + 20, hud_y + 24), 8, glow_color, 1, cv2.LINE_AA)

    # Header Title
    header_title = f"INDUSTRIAL EMISSIONS INTELLIGENCE  [{detector_mode}]"
    cv2.putText(annotated, header_title, (hud_x + 36, hud_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 210), 1, cv2.LINE_AA)

    # Line 1: Smoke Opacity & Ringelmann Level
    line1 = f"SMOKE OPACITY: {opacity_pct:.1f}% (RINGELMANN LEVEL {ringelmann_level})"
    cv2.putText(annotated, line1, (hud_x + 20, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: Coal Rate & CO2 Rate
    line2 = f"COAL RATE: {fuel_rate_kg_hr:,.0f} kg/hr | CO2 RATE: {co2_rate_kg_hr:,.1f} kg/hr"
    cv2.putText(annotated, line2, (hud_x + 20, hud_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1, cv2.LINE_AA)

    # Line 3: Operational Status & Penalty Multiplier
    line3 = f"STATUS: {status_text} (PENALTY {multiplier:.2f}x)"
    cv2.putText(annotated, line3, (hud_x + 20, hud_y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, theme_bgr, 2, cv2.LINE_AA)

    # Line 4: Auxiliary CEMS Metrics (Flue Temp & SO2)
    line4 = f"FLUE TEMP: {flue_gas_temp_c:.1f}C  |  SO2: {so2_ppm} ppm  |  NODE: IND_STACK_01"
    cv2.putText(annotated, line4, (hud_x + 20, hud_y + 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 145, 160), 1, cv2.LINE_AA)

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
) -> Dict[str, Any]:
    """
    Analyzes pre-recorded CCTV footage of a factory smokestack, detects smoke plume,
    computes Ringelmann Opacity, pairs with CEMS telemetry, calculates emissions,
    and saves both annotated video and telemetry JSON.
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

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize Detector & CEMS Generator
    detector = DualModePlumeDetector(model_path=model_path)
    cems_gen = SyntheticCEMSGenerator(
        fuel_type=fuel_type,
        base_fuel_rate=base_fuel_rate,
        scrubber_efficiency=scrubber_efficiency,
    )

    telemetry_records: List[Dict[str, Any]] = []
    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame_idx += 1

            # 1. Detect Plume Box and Extract ROI
            bbox, plume_roi, mode_name = detector.detect_plume(frame, frame_idx=frame_idx)

            # 2. Ringelmann Opacity Calculation
            opacity_pct, ringelmann_lvl, op_multiplier, status = calculate_ringelmann_opacity(plume_roi)

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

            # 6. Render HUD & Glow Overlay
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
            )

            writer.write(annotated_frame)

            if frame_idx % 30 == 0:
                logger.info(
                    f"Frame {frame_idx}/{total_frames} | Opacity: {opacity_pct:.1f}% | "
                    f"CO2: {emissions_calc['baseline_co2_kg_hr']} kg/hr | Mode: {mode_name}"
                )

    finally:
        cap.release()
        writer.release()

    # Save complete telemetry json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_records, f, indent=2)

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Processing completed in {elapsed:.2f}s ({proc_fps:.1f} FPS). "
        f"Saved {frame_idx} frames to {output_video_path} and telemetry to {output_json_path}"
    )

    return {
        "frames_processed": frame_idx,
        "elapsed_sec": round(elapsed, 2),
        "proc_fps": round(proc_fps, 1),
        "annotated_video": output_video_path,
        "telemetry_json": output_json_path,
        "records_count": len(telemetry_records),
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
                bbox, plume_roi, _ = detector.detect_plume(frame, frame_idx=frame_idx)
                opacity_pct, ringelmann_lvl, op_mult, status = calculate_ringelmann_opacity(plume_roi)
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
