"""
RoomRadar API: counts-only backend. GET /occupancy returns current zone counts.
Detection process (or a background task) can POST /occupancy to update state.
Dashboard served at /.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# In-memory state: list of { id, name, available, occupied } per zone.
_occupancy: list[dict] = []
_sources: dict[str, dict] = {}
DEFAULT_STALE_MS = 10_000


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


def _fuse(stale_ms: int = DEFAULT_STALE_MS) -> tuple[list[dict], dict]:
    """
    Fuse multi-camera occupancy into one snapshot per zone id.

    Each physical stream posts as (node_id, camera_id) with its own zone rows.
    Aggregation rule (OR across cameras): for each zone id, take the maximum
    reported ``occupied`` across *fresh* sources, capped by the largest
    ``total_seats`` seen for that zone. So if any camera sees occupancy in that
    zone, the fused count reflects at least that maximum (two cameras cannot
    "vote down" a seat another camera still sees as taken).
    """
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
                    "sources": 0,
                },
            )
            item["occupied_max"] = max(item["occupied_max"], nz["occupied"])
            item["total_seats"] = max(item["total_seats"], nz["total_seats"])
            item["sources"] += 1

    fused: list[dict] = []
    for zid, item in acc.items():
        total = max(1, _to_int(item["total_seats"], 1))
        occupied = min(total, _to_int(item["occupied_max"], 0))
        fused.append(
            {
                "id": zid,
                "name": item["name"],
                "occupied": occupied,
                "total_seats": total,
                "available": max(0, total - occupied),
                "sources": item["sources"],
                "fusion": "max_occupied",
            }
        )
    fused.sort(key=lambda z: z["id"])
    return fused, {"active_sources": len(active), "tracked_sources": len(_sources), "stale_ms": stale_ms}


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
def occupancy():
    """Returns fused per-zone counts: max(occupied) across fresh camera sources per zone id."""
    zones, meta = _fuse()
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
