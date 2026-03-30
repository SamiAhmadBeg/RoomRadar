"""
RoomRadar API: counts-only backend. GET /occupancy returns current zone counts.
Detection process (or a background task) can POST /occupancy to update state.
Dashboard served at /.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# In-memory state: list of { id, name, available, occupied } per zone.
_occupancy: list[dict] = []


def set_occupancy(zones: list[dict]) -> None:
    """Update current occupancy (called by detection loop or worker)."""
    global _occupancy
    _occupancy = list(zones)


def get_occupancy() -> list[dict]:
    """Return current occupancy snapshot."""
    return list(_occupancy)


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


@app.get("/occupancy")
def occupancy():
    """Returns current per-zone counts. No video, no PII."""
    return {"zones": get_occupancy()}


@app.post("/occupancy")
def update_occupancy(payload: ZoneUpdate):
    """Accept occupancy update from detection node (e.g. edge device or run_camera)."""
    set_occupancy(payload.zones)
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
