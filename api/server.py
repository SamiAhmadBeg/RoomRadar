"""
RoomRadar API: counts-only backend. GET /occupancy returns current zone counts.
Detection process (or a background task) can POST /occupancy to update state.
Dashboard served at /.

Fusion (env ROOMRADAR_FUSION): ``max`` (default) = redundant/overlapping cameras;
``sum`` = complementary views of the same zone id (counts are summed, capped by total_seats).
Stale cutoff: ROOMRADAR_STALE_MS (default 10000).
"""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# In-memory state: list of { id, name, available, occupied } per zone.
_occupancy: list[dict] = []
_sources: dict[str, dict] = {}
DEFAULT_STALE_MS = int(os.environ.get("ROOMRADAR_STALE_MS", "10000"))
# max: overlapping cameras (same physical seats). sum: partitioned views sharing a zone id.
DEFAULT_FUSION = os.environ.get("ROOMRADAR_FUSION", "max").strip().lower()


def set_occupancy(zones: list[dict]) -> None:
    """Update current occupancy (called by detection loop or worker)."""
    global _occupancy
    _occupancy = list(zones)


def get_occupancy() -> list[dict]:
    """Return current occupancy snapshot."""
    return list(_occupancy)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _source_key(node_id: str, camera_id: str) -> str:
    return f"{node_id}::{camera_id}"


def _is_fresh(ts_ms: int, stale_ms: int) -> bool:
    now_ms = int(time.time() * 1000)
    return (now_ms - _to_int(ts_ms)) <= stale_ms


def _normalize_zone(zone: dict) -> dict:
    total = max(1, _to_int(zone.get("total_seats", 1), 1))
    occupied = _to_int(zone.get("occupied", 0), 0)
    occupied = max(0, min(occupied, total))
    return {
        "id": str(zone.get("id", "unknown")),
        "name": str(zone.get("name", zone.get("id", "unknown"))),
        "total_seats": total,
        "occupied": occupied,
        "available": max(0, total - occupied),
    }


def _fuse(
    stale_ms: int = DEFAULT_STALE_MS,
    fusion: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Fuse multi-camera occupancy into one snapshot per zone id.

    Each physical stream posts as (node_id, camera_id) with its own zone rows.

    ``max`` (default): take the maximum reported ``occupied`` across *fresh*
    sources — best when cameras overlap on the same physical seats (avoids
    double-counting the same person).

    ``sum``: add occupied counts across sources, capped by ``total_seats`` — use
    when the same zone id is intentionally split across cameras (partitioned
    seating). Prefer assigning distinct zone ids or ``camera_id`` in zones.json
    instead when possible.
    """
    mode = (fusion or DEFAULT_FUSION or "max").lower()
    if mode not in ("max", "sum"):
        mode = "max"
    active = [s for s in _sources.values() if _is_fresh(_to_int(s.get("ts_ms", 0)), stale_ms)]
    acc: dict[str, dict] = {}
    for src in active:
        for z in src.get("zones", []):
            nz = _normalize_zone(z)
            zid = nz["id"]
            item = acc.setdefault(
                zid,
                {
                    "id": zid,
                    "name": nz["name"],
                    "total_seats": nz["total_seats"],
                    "occupied_max": 0,
                    "occupied_sum": 0,
                    "sources": 0,
                },
            )
            if mode == "sum":
                item["occupied_sum"] = _to_int(item.get("occupied_sum", 0), 0) + nz["occupied"]
            else:
                item["occupied_max"] = max(_to_int(item.get("occupied_max", 0), 0), nz["occupied"])
            item["total_seats"] = max(item["total_seats"], nz["total_seats"])
            item["sources"] += 1

    fused: list[dict] = []
    fusion_label = "sum_occupied" if mode == "sum" else "max_occupied"
    for zid, item in acc.items():
        total = max(1, _to_int(item["total_seats"], 1))
        if mode == "sum":
            occupied = min(total, _to_int(item.get("occupied_sum", 0), 0))
        else:
            occupied = min(total, _to_int(item.get("occupied_max", 0), 0))
        fused.append(
            {
                "id": zid,
                "name": item["name"],
                "occupied": occupied,
                "total_seats": total,
                "available": max(0, total - occupied),
                "sources": item["sources"],
                "fusion": fusion_label,
            }
        )
    fused.sort(key=lambda z: z["id"])
    return fused, {
        "active_sources": len(active),
        "tracked_sources": len(_sources),
        "stale_ms": stale_ms,
        "fusion": mode,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: optional init (e.g. load default zones)
    yield
    # Shutdown
    pass


app = FastAPI(title="RoomRadar", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "dashboard"


@app.get("/")
def dashboard():
    """Serve the occupancy dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


class ZoneUpdate(BaseModel):
    zones: list[dict]
    node_id: str = "unknown-node"
    camera_id: str = "cam_1"
    ts_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    seq: int = 0


@app.get("/occupancy")
def occupancy(fusion: str | None = None):
    """Returns fused per-zone counts. Query ``fusion=max|sum`` overrides ROOMRADAR_FUSION for this request."""
    zones, meta = _fuse(fusion=fusion)
    if zones:
        set_occupancy(zones)
        return {"zones": zones, "meta": meta}
    # Avoid empty responses during transient camera startup/drop windows.
    return {"zones": get_occupancy(), "meta": meta}


@app.post("/occupancy")
def update_occupancy(payload: ZoneUpdate):
    """Accept occupancy update from detection node (e.g. edge device or run_camera)."""
    key = _source_key(payload.node_id, payload.camera_id)
    _sources[key] = {
        "node_id": payload.node_id,
        "camera_id": payload.camera_id,
        "ts_ms": payload.ts_ms,
        "seq": payload.seq,
        "zones": list(payload.zones),
    }
    fused, _meta = _fuse()
    if fused:
        set_occupancy(fused)
    else:
        set_occupancy(payload.zones)
    return {"status": "ok", "source": key}


@app.get("/occupancy/raw")
def occupancy_raw(stale_ms: int = DEFAULT_STALE_MS):
    """Returns latest per-source occupancy and freshness."""
    now_ms = int(time.time() * 1000)
    nodes = []
    for key, src in _sources.items():
        age_ms = max(0, now_ms - _to_int(src.get("ts_ms", 0)))
        nodes.append(
            {
                "source": key,
                "node_id": src.get("node_id"),
                "camera_id": src.get("camera_id"),
                "seq": src.get("seq", 0),
                "age_ms": age_ms,
                "fresh": age_ms <= stale_ms,
                "zones": src.get("zones", []),
            }
        )
    return {"nodes": nodes, "tracked_sources": len(_sources), "stale_ms": stale_ms}


@app.get("/health")
def health():
    return {"status": "ok"}
