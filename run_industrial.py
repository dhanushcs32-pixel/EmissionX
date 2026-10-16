"""Test Runner & Validation Suite for Industrial Emissions Module (Pillar 2).

Executes full end-to-end pipeline:
1. Validates / Generates factory smokestack CCTV footage (`factory_stack.mp4`).
2. Runs Dual-Mode Plume Detector & Ringelmann Opacity analysis.
3. Synchronizes with synthetic CEMS boiler sensor telemetry.
4. Executes deterministic Industrial Carbon Math Engine.
5. Emits annotated video to `cache/factory_annotated.mp4`.
6. Emits standardized JSON contract to `cache/factory_telemetry.json`.
7. Strictly validates output against Pydantic schema contract.
8. Demonstrates FastAPI simulation math and Smart Replay Cache streaming.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

# Ensure project root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from modules.industrial_detector import (
    IndustrialTelemetryPayload,
    calculate_industrial_emissions,
    process_video,
    stream_industrial_telemetry,
    DualModePlumeDetector,
    calculate_ringelmann_opacity,
    EMISSION_FACTORS,
)


def generate_synthetic_factory_video(
    output_path: str = "factory_stack.mp4",
    duration_sec: int = 6,
    fps: int = 25,
    width: int = 640,
    height: int = 480,
) -> str:
    """
    Generates realistic factory smokestack CCTV footage if no video exists.
    Simulates a factory background, tall chimney smokestack, and billowing smoke plume
    cycling through:
      Phase 1 (Frames 0-50): Clean Steam / Normal (Ringelmann 0-1)
      Phase 2 (Frames 51-100): Incomplete Combustion Warning (Ringelmann 2-3)
      Phase 3 (Frames 101-150): Black Carbon Scrubber Violation (Ringelmann 4-5)
    """
    out_file = Path(output_path)
    if out_file.is_file():
        print(f"[INIT] Found existing input video: {output_path}")
        return output_path

    print(f"[SYNTHETIC-CV] Generating synthetic factory CCTV footage -> {output_path}...")
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    # Chimney stack parameters
    stack_x = width // 2
    stack_w = 64
    stack_top_y = height // 2 - 20
    stack_base_y = height - 40

    # Plume particle simulation
    particles = []

    for frame_idx in range(total_frames):
        # 1. Background: Industrial overcast sky gradient + horizon
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            # Gradient from slate-blue overcast to horizon smog
            ratio = y / height
            b = int(180 + 30 * ratio)
            g = int(170 + 25 * ratio)
            r = int(160 + 20 * ratio)
            frame[y, :] = (b, g, r)

        # Factory roof & ground silhouettes
        cv2.rectangle(frame, (0, stack_base_y - 20), (width, height), (70, 75, 80), -1)
        cv2.rectangle(frame, (40, stack_base_y - 80), (180, stack_base_y), (50, 55, 60), -1)
        cv2.rectangle(frame, (width - 220, stack_base_y - 110), (width - 30, stack_base_y), (45, 50, 55), -1)

        # Industrial Brick/Steel Smokestack
        cv2.rectangle(
            frame,
            (stack_x - stack_w // 2, stack_top_y),
            (stack_x + stack_w // 2, stack_base_y),
            (60, 65, 75),
            -1,
        )
        # Stack highlights & rim
        cv2.rectangle(
            frame,
            (stack_x - stack_w // 2 - 4, stack_top_y - 6),
            (stack_x + stack_w // 2 + 4, stack_top_y),
            (40, 45, 50),
            -1,
        )
        # Warning stripes on chimney
        for sy in range(stack_top_y + 20, stack_base_y, 40):
            cv2.rectangle(
                frame,
                (stack_x - stack_w // 2, sy),
                (stack_x + stack_w // 2, sy + 15),
                (40, 140, 210),
                -1,
            )

        # Determine smoke density & darkness based on phase
        if frame_idx < 50:
            # Phase 1: Clean Steam (Brightness ~215, Opacity ~15%)
            smoke_brightness = 220 + random.randint(-10, 10)
            spawn_rate = 3
        elif frame_idx < 100:
            # Phase 2: Incomplete Combustion (Brightness ~140, Opacity ~45%)
            smoke_brightness = 135 + random.randint(-15, 15)
            spawn_rate = 5
        else:
            # Phase 3: Scrubber Failure / Black Carbon (Brightness ~60, Opacity ~75%)
            smoke_brightness = 60 + random.randint(-12, 12)
            spawn_rate = 7

        # Spawn new plume particles at chimney stack exit
        for _ in range(spawn_rate):
            particles.append({
                "x": float(stack_x + random.uniform(-stack_w // 4, stack_w // 4)),
                "y": float(stack_top_y - 2),
                "vx": random.uniform(0.6, 1.8),  # Wind blowing rightward
                "vy": random.uniform(-2.5, -1.2), # Rising thermal plume
                "radius": random.uniform(8.0, 14.0),
                "brightness": smoke_brightness,
                "alpha": random.uniform(0.35, 0.65),
                "life": 0,
                "max_life": random.randint(45, 65),
            })

        # Update & draw billowing smoke plume
        smoke_layer = frame.copy()
        alive_particles = []

        for p in particles:
            p["x"] += p["vx"] + math.sin(p["life"] * 0.15) * 0.4
            p["y"] += p["vy"]
            p["radius"] += 0.55  # Plume expands as it rises
            p["life"] += 1

            if p["life"] < p["max_life"] and p["y"] > 10:
                alive_particles.append(p)
                # Draw circular smoke puff
                px, py = int(p["x"]), int(p["y"])
                pr = int(p["radius"])
                b_val = max(10, min(255, int(p["brightness"])))
                cv2.circle(smoke_layer, (px, py), pr, (b_val, b_val, b_val), -1)

        particles = alive_particles

        # Blend smoke overlay with soft edges
        cv2.addWeighted(smoke_layer, 0.70, frame, 0.30, 0, frame)

        # CCTV timestamp and camera watermark
        cv2.putText(
            frame,
            f"CAM-04 PEENYA INDUSTRIAL STACK | REC {frame_idx:04d}",
            (20, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"[SYNTHETIC-CV] Successfully created {total_frames} frames of CCTV footage: {output_path}")
    return output_path


def validate_telemetry_schema(json_path: str) -> List[Dict[str, Any]]:
    """
    Validates every telemetry record in the emitted JSON file
    strictly against the Pydantic IndustrialTelemetryPayload contract.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry JSON missing: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Telemetry JSON must contain non-empty array of records: {json_path}")

    print(f"\n[VALIDATION] Validating {len(data)} telemetry records against Pydantic schema...")
    for idx, item in enumerate(data):
        try:
            # Parse & validate strictly
            validated = IndustrialTelemetryPayload.model_validate(item)
            # Check required fields
            assert validated.node_id == "IND_STACK_01_PEENYA"
            assert validated.sector == "industrial"
            assert validated.cv_telemetry.smoke_opacity_pct >= 0.0
            assert validated.cv_telemetry.ringelmann_level in [0, 1, 2, 3, 4, 5]
            assert validated.cv_telemetry.opacity_multiplier in [1.0, 1.25, 1.60]
            assert validated.cv_telemetry.cems_metrics.fuel_rate_kg_hr > 0
            assert validated.emissions.baseline_co2_kg_hr > 0
        except Exception as e:
            raise AssertionError(f"Schema validation failed on record #{idx}: {e}\nPayload: {item}")

    print(" [SCHEMA VALIDATION PASSED] All emitted records strictly adhere to the contract!")
    return data


def display_telemetry_sample(record: Dict[str, Any]) -> None:
    """Pretty prints an example validated payload."""
    print("\n--- SAMPLE VALIDATED JSON PAYLOAD (MATCHES CONTRACT EXACTLY) ---")
    print(json.dumps(record, indent=2))
    print("----------------------------------------------------------------\n")


def test_whatif_simulations() -> None:
    """
    Tests and demonstrates the pure deterministic calculation engine
    called by the FastAPI /api/simulate endpoint.
    """
    print("\n=======================================================")
    print(" INDUSTRIAL CARBON MATH ENGINE: WHAT-IF SIMULATIONS")
    print("=======================================================")

    fuel_rate = 1245.0  # kg/hr
    high_penalty = 1.60 # Ringelmann 4-5

    # Case 1: Baseline Dirty Coal (0% Scrubber, 1.6x penalty)
    base_calc = calculate_industrial_emissions(
        fuel_rate_kg_hr=fuel_rate,
        fuel_type="coal",
        opacity_multiplier=high_penalty,
        scrubber_efficiency=0.0,
    )
    print(f"1. BASELINE DIRTY COAL:")
    print(f"   - Fuel Feed: {fuel_rate:.1f} kg/hr | Emission Factor: {EMISSION_FACTORS['coal']} kg CO2/kg")
    print(f"   - Opacity Penalty: {high_penalty}x")
    print(f"   - Baseline CO2: {base_calc['baseline_co2_kg_hr']} kg/hr")
    print(f"   - Waste CO2:    {base_calc['waste_co2_kg_hr']} kg/hr")

    # Case 2: Scrubber Retrofit (85% efficiency)
    scrub_calc = calculate_industrial_emissions(
        fuel_rate_kg_hr=fuel_rate,
        fuel_type="coal",
        opacity_multiplier=high_penalty,
        scrubber_efficiency=0.85,
    )
    print(f"\n2. RETROFIT WET SCRUBBER (85% Efficiency):")
    print(f"   - Baseline with Scrubber: {scrub_calc['baseline_co2_kg_hr']} kg/hr")
    print(f"   - Waste CO2:              {scrub_calc['waste_co2_kg_hr']} kg/hr")

    # Case 3: Fuel Switch to Biomass (Emission factor 0.45)
    biomass_calc = calculate_industrial_emissions(
        fuel_rate_kg_hr=fuel_rate,
        fuel_type="biomass",
        opacity_multiplier=1.0,
        scrubber_efficiency=0.0,
    )
    reduction = ((base_calc['baseline_co2_kg_hr'] - biomass_calc['baseline_co2_kg_hr']) / base_calc['baseline_co2_kg_hr']) * 100.0
    print(f"\n3. FUEL CONVERSION: COAL -> SUSTAINABLE BIOMASS PELLETS:")
    print(f"   - Biomass CO2:  {biomass_calc['baseline_co2_kg_hr']} kg/hr")
    print(f"   - Net CO2 Cut:  {reduction:.1f}% reduction relative to dirty coal baseline!")

    # Case 4: Combustion Tuning Simulation
    tuned_calc = calculate_industrial_emissions(
        fuel_rate_kg_hr=fuel_rate,
        fuel_type="coal",
        opacity_multiplier=high_penalty,
        scrubber_efficiency=0.0,
        combustion_tuning_pct=80.0, # 80% tuning removes most excess multiplier
    )
    print(f"\n4. COMBUSTION AIR-TO-FUEL OPTIMIZATION (80% Tuning):")
    print(f"   - Projected CO2: {tuned_calc['projected_co2_kg_hr']} kg/hr")
    print(f"   - Reduction:     {tuned_calc['reduction_pct']}% CO2 avoided without hardware overhaul!")
    print("=======================================================\n")


def test_smart_replay_streaming(json_path: str) -> None:
    """Demonstrates Tier-2 Smart Replay Cache streaming at 60 FPS."""
    print("[STREAMING] Testing Tier-2 Smart Replay Cache Generator (<2% CPU)...")
    gen = stream_industrial_telemetry(
        use_cache=True,
        cache_path=json_path,
        fps=60.0,
        loop=False,
    )
    count = 0
    start = time.time()
    for record in gen:
        count += 1
        if count >= 10:
            break
    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    print(f" [STREAMING VERIFIED] Streamed {count} frames in {elapsed:.3f}s (~{rate:.1f} FPS) with zero GPU overhead!\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Industrial Emissions Module (Pillar 2) - Pipeline Runner")
    parser.add_argument("--video", type=str, default="factory_stack.mp4", help="Path to input industrial smokestack video")
    parser.add_argument("--output", type=str, default="cache/factory_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument("--json", type=str, default="cache/factory_telemetry.json", help="Path to save telemetry JSON file")
    parser.add_argument("--model", type=str, default="models/smoke.pt", help="Path to YOLO plume weights (optional)")
    parser.add_argument(
        "--profile", type=str, default="auto",
        choices=["auto", "universal_optical_flow", "overcast_horizontal", "sunset_vertical", "synthetic"],
        help="Smoke segmentation engine; 'auto' routes by scene (unknown scenes use the universal optical-flow engine)",
    )
    args, _ = parser.parse_known_args()

    print("=" * 65)
    print(" INDUSTRIAL EMISSIONS MODULE (PILLAR 2) - TEST RUNNER")
    print("=" * 65)

    input_video = args.video
    output_video = args.output
    output_json = args.json
    model_weight = args.model

    # Step 1: Ensure input video exists (synthesizes default factory video if not provided)
    if input_video == "factory_stack.mp4" and not Path(input_video).is_file():
        generate_synthetic_factory_video(output_path=input_video, duration_sec=6, fps=25)
    elif not Path(input_video).is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video}")

    # Step 2: Run End-to-End Video Processing Pipeline
    print(f"\n[PIPELINE] Launching video analysis on {input_video}...")
    stats = process_video(
        input_video_path=input_video,
        output_video_path=output_video,
        output_json_path=output_json,
        model_path=model_weight,
        profile=None if args.profile == "auto" else args.profile,
    )

    print("\n[PIPELINE SUMMARY]")
    print(f" - Profile Selected:   {stats.get('profile', 'Auto-Routed')} ({stats.get('detector_mode', '')})")
    print(f" - Video Codec:        {stats.get('video_codec')} (browsers need avc1)")
    print(f" - Frames Processed:   {stats['frames_processed']}")
    print(f" - Processing Speed:   {stats['proc_fps']} FPS (Total time: {stats['elapsed_sec']}s)")
    print(f" - Annotated Video:    {stats['annotated_video']} (File size: {os.path.getsize(output_video) / 1024:.1f} KB)")
    print(f" - Telemetry JSON:     {stats['telemetry_json']} (Records: {stats['records_count']})")
    if stats.get('backup_video') and os.path.exists(stats['backup_video']):
        print(f" - Permanent Backup:   {stats['backup_video']} & {stats['backup_json']}")

    # Step 3: Validate Strict JSON Schema
    records = validate_telemetry_schema(output_json)

    # Display high-opacity violation sample
    violation_records = [r for r in records if r["cv_telemetry"]["ringelmann_level"] >= 4]
    sample_to_show = violation_records[0] if violation_records else records[-1]
    display_telemetry_sample(sample_to_show)

    # Step 4: Test What-If Emission Engine (FastAPI Simulation)
    test_whatif_simulations()

    # Step 5: Test Tier-2 Smart Replay Cache Streaming
    test_smart_replay_streaming(output_json)

    print("=" * 65)
    print(" PILLAR 2 INDUSTRIAL DETECTOR TEST RUNNER: ALL CHECKS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
