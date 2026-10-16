import json

import pytest

from backend.app.zones import load_forecourt_zones
from modules import forecourt_fuel as ff
from modules.upload_jobs import tracks_to_fuel_telemetry


def test_methods_spread_across_tracks():
    assert [ff.method_for(i) for i in (3, 1, 2)] == ["A", "B", "C"]


def test_litres_follow_blueprint_formulas():
    # Method B: max(0, bay_seconds - 15) x flow (car 0.6 L/s, truck 1.3 L/s)
    assert ff.litres_for("B", "car", 45) == pytest.approx(30 * 0.6)
    assert ff.litres_for("B", "truck", 10) == 0.0
    # Method A: nozzle time after setup; C: capped at a typical fill
    assert ff.litres_for("A", "truck", 28) == pytest.approx(20 * 1.3)
    assert ff.litres_for("C", "car", 600) == ff.TYPICAL_FILL_L["car"]


def test_idle_bs_multipliers_and_ev():
    base = 100 * ff.IDLE_KG_PER_S["truck"]
    assert ff.idle_co2_kg(100, "truck", "diesel", "BS-III") == pytest.approx(base * 1.35)
    assert ff.idle_co2_kg(100, "truck", "diesel", "BS-IV") == pytest.approx(base * 1.15)
    assert ff.idle_co2_kg(100, "truck", "diesel", "BS-VI") == pytest.approx(base)
    assert ff.idle_co2_kg(100, "car", "electric", "Zero tailpipe") == 0.0


def test_voc_penalty_only_above_30c():
    assert ff.voc_factor(29.0) == 1.0 and ff.voc_factor(None) == 1.0
    assert ff.voc_factor(34.0) == pytest.approx(1.20)


def _summary():
    # track 2 -> method C (FASTag); its mock record comes from the registry by track key
    return {
        "video_source": "clip.mp4", "fps": 10.0, "total_frames": 600,
        "tracked_vehicles": [
            {"track_id": 2, "vehicle_class": "car", "license_plate": None, "first_seen_frame": 0, "last_seen_frame": 599,
             "estimated_dwell_seconds": 50.0, "queue_idle_seconds": 0.0, "bay_stationary_seconds": 45.0,
             "bay_first_frame": 50, "bay_last_frame": 550, "queue_first_frame": None, "queue_last_frame": None,
             "refuel_trigger_frame": 90},
            {"track_id": 7, "vehicle_class": "truck", "license_plate": "KA09C2233", "first_seen_frame": 0, "last_seen_frame": 599,
             "estimated_dwell_seconds": 0.0, "queue_idle_seconds": 50.0, "bay_stationary_seconds": 0.0,
             "bay_first_frame": None, "bay_last_frame": None, "queue_first_frame": 0, "queue_last_frame": 500,
             "refuel_trigger_frame": None},
        ],
    }


def test_enrich_summary_events_and_totals():
    s = ff.enrich_summary(_summary(), temp_c=34.0)
    car, truck = s["tracked_vehicles"]
    ev = car["forecourt"]["refuel"]
    assert ev["method"] == "C" and ev["log"].startswith(" [FASTAG-SYNC] ")
    assert "-" in ev["log"].split("]")[1].split("(")[0]          # plate shown as KA-01-..-....
    assert truck["forecourt"]["refuel"] is None
    # BS-III mock truck idling 50 s in the queue
    assert truck["forecourt"]["co2_idle_kg"] == pytest.approx(50 * ff.IDLE_KG_PER_S["truck"] * 1.35, rel=1e-3)
    fc = s["forecourt"]
    assert fc["refuels"] == 1 and fc["refuels_by_method"]["C"] == 1
    assert fc["voc_penalty_pct"] == 20.0 and fc["data_source"] == "simulated"


def test_telemetry_uses_blueprint_keys_and_zone_timing():
    s = ff.enrich_summary(_summary(), temp_c=25.0)
    recs = tracks_to_fuel_telemetry(s)
    assert set(recs[0]["cv_telemetry"]["active_counts"]) == {"car", "motorcycle", "truck", "bus"}
    at_9s = recs[9]["cv_telemetry"]            # frame 90: refuel trigger second
    assert any("[FASTAG-SYNC]" in l for l in at_9s["recent_logs"])
    assert at_9s["bay_dwell_sec"] == pytest.approx(4.0)       # in bay since frame 50
    at_45s = recs[45]["cv_telemetry"]
    assert any("[QUEUE-WARN]" in l for l in at_45s["recent_logs"])   # truck waiting > 40 s
    assert recs[0]["emissions"]["baseline_co2_kg_hr"] > 0


def test_zone_override_by_clip_name(tmp_path):
    cfg = {"default": {"Bay_01": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
           "videos": {"special": {"Bay_01": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}},
           "bay_stationary_trigger_s": 3.0, "stationary_px_frac": 0.02}
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(cfg))
    default, trig, frac = load_forecourt_zones("normal.mp4", 100, 50, path=path)
    special, _, _ = load_forecourt_zones("my_special_clip.mp4", 100, 50, path=path)
    assert default["Bay_01"][1].tolist() == [50, 0] and special["Bay_01"][0].tolist() == [50, 0]
    assert trig == 3.0 and frac == 0.02
