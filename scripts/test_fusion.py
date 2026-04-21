"""
Quick check: dual-camera payloads fuse with max(occupied) per zone id.

Run from repo root:
  source venv/bin/activate
  python scripts/test_fusion.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.server as srv  # noqa: E402


def _ingest(node_id: str, camera_id: str, seq: int, zones: list[dict]) -> None:
    key = srv._source_key(node_id, camera_id)
    srv._sources[key] = {
        "node_id": node_id,
        "camera_id": camera_id,
        "ts_ms": int(time.time() * 1000),
        "seq": seq,
        "zones": list(zones),
    }


def main() -> int:
    srv._sources.clear()

    # Cam 1: both zones empty
    _ingest(
        "test-node",
        "cam_1",
        1,
        [
            {"id": "zone_1", "name": "Area 1", "total_seats": 1, "occupied": 0, "available": 1},
            {"id": "zone_2", "name": "Area 2", "total_seats": 1, "occupied": 0, "available": 1},
        ],
    )
    # Cam 2: only cam2 sees zone_1 occupied
    _ingest(
        "test-node",
        "cam_2",
        2,
        [
            {"id": "zone_1", "name": "Area 1", "total_seats": 1, "occupied": 1, "available": 0},
            {"id": "zone_2", "name": "Area 2", "total_seats": 1, "occupied": 0, "available": 1},
        ],
    )

    fused, meta = srv._fuse()
    assert meta["active_sources"] == 2, meta
    by_id = {z["id"]: z for z in fused}
    assert by_id["zone_1"]["occupied"] == 1, by_id["zone_1"]
    assert by_id["zone_1"]["fusion"] == "max_occupied", by_id["zone_1"]
    assert by_id["zone_2"]["occupied"] == 0, by_id["zone_2"]

    # Multi-seat: max of counts, capped by total
    srv._sources.clear()
    _ingest(
        "test-node",
        "cam_1",
        1,
        [{"id": "zone_big", "name": "Hall", "total_seats": 5, "occupied": 1, "available": 4}],
    )
    _ingest(
        "test-node",
        "cam_2",
        2,
        [{"id": "zone_big", "name": "Hall", "total_seats": 5, "occupied": 3, "available": 2}],
    )
    z = {z["id"]: z for z in srv._fuse()[0]}["zone_big"]
    assert z["occupied"] == 3 and z["total_seats"] == 5, z

    print("fusion tests OK (dual-camera aggregate = max occupied per zone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
