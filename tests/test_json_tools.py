import json
import pytest
from modules import json_tools as jt


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    t = tmp_path / "t.json"
    t.write_text(json.dumps([{"a": {"v": 1}}, {"a": {"v": 3}}]))
    monkeypatch.setitem(jt.ALLOWED_FILES, "factory_telemetry", t)
    return t


def test_read_and_query(sandbox):
    assert jt.read_json("factory_telemetry", "1.a.v")["value"] == 3
    assert jt.query_json("factory_telemetry", "a.v")["mean"] == 2.0


def test_write_modes_and_backup(sandbox):
    jt.write_json("factory_telemetry", "0.a.v", 9, "set")
    jt.write_json("factory_telemetry", "", {"a": {"v": 5}}, "append")
    jt.write_json("factory_telemetry", "1", {"note": "x"}, "merge")
    jt.write_json("factory_telemetry", "0", None, "delete")
    data = json.loads(sandbox.read_text())
    assert len(data) == 2 and data[0]["note"] == "x"
    assert sandbox.with_suffix(".json.bak").exists()


def test_errors_are_data(sandbox):
    assert "error" in jt.run_tool("read_json", {"file": "../etc/passwd"})
    assert "error" in jt.run_tool("write_json", {"file": "factory_telemetry", "path": "", "mode": "delete"})
