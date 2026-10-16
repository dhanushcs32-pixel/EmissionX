from datetime import date

from fastapi.testclient import TestClient

from modules import vehicle_registry as vr

TODAY = date(2026, 10, 16)


def test_database_has_20_mock_records():
    recs = vr.records()
    assert len(recs) == 20
    assert len({r["plate"] for r in recs}) == 20


def test_exact_plate_match_ignores_spacing():
    r = vr.lookup("KA 51 AF 4321", "bus", today=TODAY)
    assert r["match"] == "exact_plate"
    assert r["maker_model"] == "Volvo 8400"
    assert r["data_source"] == "mock" and "MOCK" in r["notice"]


def test_unknown_plate_gets_stable_record_of_same_class():
    a = vr.lookup("ZZ99ZZ9999", "truck", today=TODAY)
    b = vr.lookup("ZZ99ZZ9999", "truck", today=TODAY)
    assert a["match"] == "assigned_by_hash"
    assert a["plate"] == b["plate"]
    assert a["vehicle_class"] == "truck"
    assert a["plate_read"] == "ZZ99ZZ9999"


def test_unread_plate_uses_track_key():
    a = vr.lookup(None, "motorcycle", key="track-3", today=TODAY)
    assert a["vehicle_class"] == "motorcycle" and a["plate_read"] is None


def test_derived_fields():
    ev = vr.lookup("KA02MB3310", today=TODAY)
    assert ev["puc_status"] == "not required (EV)" and ev["est_co2_g_km"] == 0
    old = vr.lookup("KA09C2233", today=TODAY)          # BS-III truck, PUC ran out in May 2026
    assert old["puc_status"] == "expired"
    assert old["est_co2_g_km"] == round(750 * 1.25)
    assert vr.lookup("22BH1234AA", today=TODAY)["state"] == "Bharat (BH) series"


def test_fleet_summary_counts():
    tracks = [
        {"track_id": 1, "vehicle_class": "bus", "license_plate": "KA51AF4321"},
        {"track_id": 2, "vehicle_class": "truck", "license_plate": "KA09C2233"},
        {"track_id": 3, "vehicle_class": "car", "license_plate": None},
    ]
    vehicles = vr.enrich_tracks(tracks, today=TODAY)
    s = vr.fleet_summary(vehicles)
    assert s["vehicles"] == 3 and s["plates_read"] == 2 and s["exact_matches"] == 2
    assert s["bs3_bs4_heavy"] == 2 and s["puc_expired"] >= 1


def test_api_endpoints():
    from main import app

    c = TestClient(app)
    r = c.get("/api/vahan/lookup", params={"plate": "MH12DE1433"})
    assert r.status_code == 200 and r.json()["match"] == "exact_plate"
    assert c.get("/api/vahan/lookup").status_code == 400
    fleet = c.get("/api/vahan/fleet").json()
    assert fleet["data_source"] == "mock" and fleet["database_size"] == 20
    assert len(c.get("/api/vahan/database").json()["records"]) == 20
