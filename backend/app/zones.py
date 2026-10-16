"""Forecourt zones (Bay_01 refuelling bay, Queue_01 waiting lane) from backend/data/forecourt_zones.json."""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ZONES_PATH = Path(__file__).resolve().parents[1] / "data" / "forecourt_zones.json"


def load_forecourt_zones(
    source_name: str, width: int, height: int, path: Optional[Path] = None
) -> Tuple[Dict[str, np.ndarray], float, float]:
    """Pixel polygons for this clip, plus (bay stationary trigger seconds, stationary movement fraction).

    A 'videos' entry whose key is part of the clip's file name overrides the default zones.
    """
    cfg = json.loads((path or ZONES_PATH).read_text(encoding="utf-8"))
    zones = cfg["default"]
    name = Path(source_name or "").name
    for key, override in (cfg.get("videos") or {}).items():
        if key and key in name:
            zones = override
            break
    polys = {
        zone: np.array([[int(round(x * width)), int(round(y * height))] for x, y in pts], dtype=np.int32)
        for zone, pts in zones.items()
    }
    return polys, float(cfg.get("bay_stationary_trigger_s", 3.0)), float(cfg.get("stationary_px_frac", 0.02))
