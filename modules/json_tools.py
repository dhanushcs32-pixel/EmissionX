"""Sandboxed JSON read/write tools for the EmissionX chatbot.

Only files in ALLOWED_FILES can be touched. Paths inside a document use dot
notation with integer list indices, e.g. ``0.cv_telemetry.smoke_opacity_pct``
or ``counts_by_class.car``. An empty path means the whole document.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_FILES: Dict[str, Path] = {
    "factory_telemetry": ROOT / "cache" / "factory_telemetry.json",
    "phase1_summary": ROOT / "backend" / "data" / "phase1_summary.json",
}

MAX_READ_CHARS = 20000
_lock = threading.RLock()


class JsonToolError(Exception):
    pass


def _file(name: str) -> Path:
    if name not in ALLOWED_FILES:
        raise JsonToolError(f"Unknown file '{name}'. Allowed: {sorted(ALLOWED_FILES)}")
    return ALLOWED_FILES[name]


def _load(name: str) -> Any:
    p = _file(name)
    if not p.exists():
        raise JsonToolError(f"File for '{name}' does not exist at {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(name: str, data: Any) -> None:
    p = _file(name)
    if p.exists():
        shutil.copyfile(p, p.with_suffix(p.suffix + ".bak"))
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _split(path: str) -> List[str]:
    return [s for s in (path or "").replace("[", ".").replace("]", "").split(".") if s != ""]


def _step(node: Any, key: str) -> Any:
    if isinstance(node, list):
        try:
            return node[int(key)]
        except (ValueError, IndexError):
            raise JsonToolError(f"Bad list index '{key}' (length {len(node)})")
    if isinstance(node, dict):
        if key not in node:
            raise JsonToolError(f"Key '{key}' not found. Keys: {list(node)[:30]}")
        return node[key]
    raise JsonToolError(f"Cannot descend into {type(node).__name__} with '{key}'")


def _get(data: Any, keys: List[str]) -> Any:
    for k in keys:
        data = _step(data, k)
    return data


def _describe(v: Any) -> Any:
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"object{{{len(v)} keys}}"
    return v


# ---------------------------------------------------------------- tools

def list_json_files() -> Dict[str, Any]:
    out = {}
    for name, p in ALLOWED_FILES.items():
        info: Dict[str, Any] = {"exists": p.exists()}
        if p.exists():
            d = _load(name)
            info["type"] = type(d).__name__
            info["size"] = len(d)
            if isinstance(d, dict):
                info["top_level_keys"] = {k: _describe(v) for k, v in d.items()}
            elif d and isinstance(d[0], dict):
                info["first_item_keys"] = list(d[0])
        out[name] = info
    return out


def read_json(file: str, path: str = "", start: int = 0, limit: int = 20) -> Dict[str, Any]:
    """Read a value. For lists, returns the slice [start, start+limit)."""
    with _lock:
        value = _get(_load(file), _split(path))
    total = None
    if isinstance(value, list):
        total = len(value)
        value = value[start:start + limit]
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > MAX_READ_CHARS:
        raise JsonToolError(
            f"Result too large ({len(text)} chars). Narrow the path or lower limit."
        )
    res: Dict[str, Any] = {"file": file, "path": path, "value": value}
    if total is not None:
        res.update(total_items=total, start=start, returned=len(value))
    return res


def query_json(file: str, field: str, op: str = "stats", path: str = "") -> Dict[str, Any]:
    """Aggregate a numeric field across all dict items of a list.

    field is a dot path inside each item (e.g. cv_telemetry.smoke_opacity_pct).
    op: stats (min/max/mean/count) | latest | first.
    """
    with _lock:
        items = _get(_load(file), _split(path))
    if not isinstance(items, list):
        raise JsonToolError("query_json needs a path that points to a list")
    vals = []
    for it in items:
        try:
            vals.append(_get(it, _split(field)))
        except JsonToolError:
            continue
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if op == "latest":
        return {"field": field, "value": vals[-1] if vals else None}
    if op == "first":
        return {"field": field, "value": vals[0] if vals else None}
    if op == "stats":
        if not nums:
            raise JsonToolError(f"No numeric values for '{field}'")
        return {"field": field, "count": len(nums), "min": min(nums),
                "max": max(nums), "mean": round(sum(nums) / len(nums), 3)}
    raise JsonToolError("op must be stats, latest or first")


def write_json(file: str, path: str, value: Any, mode: str = "set") -> Dict[str, Any]:
    """Modify a document. mode: set | append | delete | merge.

    set    - replace the value at path (path required; last key may be new in an object)
    append - append value to the list at path
    merge  - shallow-merge dict value into the object at path
    delete - remove the key / list element at path
    """
    keys = _split(path)
    with _lock:
        data = _load(file)
        work = copy.deepcopy(data)
        if mode in ("append", "merge"):
            target = _get(work, keys)
            if mode == "append":
                if not isinstance(target, list):
                    raise JsonToolError("append target is not a list")
                target.append(value)
            else:
                if not (isinstance(target, dict) and isinstance(value, dict)):
                    raise JsonToolError("merge needs an object target and object value")
                target.update(value)
        elif mode in ("set", "delete"):
            if not keys:
                if mode == "delete":
                    raise JsonToolError("Refusing to delete the whole document")
                work = value
            else:
                parent, last = _get(work, keys[:-1]), keys[-1]
                if isinstance(parent, list):
                    try:
                        idx = int(last)
                        if mode == "delete":
                            parent.pop(idx)
                        else:
                            parent[idx] = value
                    except (ValueError, IndexError):
                        raise JsonToolError(f"Bad list index '{last}'")
                elif isinstance(parent, dict):
                    if mode == "delete":
                        if last not in parent:
                            raise JsonToolError(f"Key '{last}' not found")
                        del parent[last]
                    else:
                        parent[last] = value
                else:
                    raise JsonToolError("Parent is not an object or list")
        else:
            raise JsonToolError("mode must be set, append, delete or merge")
        _save(file, work)
    return {"ok": True, "file": file, "path": path, "mode": mode,
            "backup": str(_file(file).with_suffix(_file(file).suffix + ".bak").name)}


# ------------------------------------------------- tool-calling schemas

_FILES = sorted(ALLOWED_FILES)

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "list_json_files",
        "description": "List the JSON data files the assistant may read/write, with their structure.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_json",
        "description": "Read a value from a JSON file by dot path (list indices as numbers). Lists are paginated with start/limit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "enum": _FILES},
                "path": {"type": "string", "description": "e.g. '0.emissions' or 'counts_by_class'. Empty = whole doc."},
                "start": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "query_json",
        "description": "Aggregate a field over a list of records: stats (min/max/mean), latest, or first. Use for questions like 'average opacity'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "enum": _FILES},
                "field": {"type": "string", "description": "Dot path inside each record, e.g. cv_telemetry.smoke_opacity_pct"},
                "op": {"type": "string", "enum": ["stats", "latest", "first"]},
                "path": {"type": "string", "description": "Path to the list; empty if the file root is the list."},
            },
            "required": ["file", "field"],
        },
    },
    {
        "name": "write_json",
        "description": "Modify a JSON file (a .bak backup is made). Only call when the user explicitly asks to change data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "enum": _FILES},
                "path": {"type": "string"},
                "value": {"description": "New value (any JSON). Unused for delete."},
                "mode": {"type": "string", "enum": ["set", "append", "merge", "delete"]},
            },
            "required": ["file", "path", "mode"],
        },
    },
]

TOOL_FUNCS = {
    "list_json_files": list_json_files,
    "read_json": read_json,
    "query_json": query_json,
    "write_json": lambda **kw: write_json(value=kw.pop("value", None), **kw),
}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool; errors are returned as data so the model can recover."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**(args or {}))
    except JsonToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"File error: {e}"}
