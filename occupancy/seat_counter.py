"""
RoomRadar occupancy: person boxes + ROI zones -> available/occupied counts.
ROI format in zones.json: normalized [x1, y1, x2, y2] in 0-1 range (relative to frame).
"""
import json
from pathlib import Path


def load_zones(config_path: str | Path) -> list[dict]:
    """Load zones from config/zones.json. Each zone has id, name, roi (norm), total_seats."""
    path = Path(config_path)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("zones", [])


def box_center(box_xyxy: list[float]) -> tuple[float, float]:
    """(x1,y1,x2,y2) -> (cx, cy)."""
    x1, y1, x2, y2 = box_xyxy[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_roi(cx_norm: float, cy_norm: float, roi: list[float]) -> bool:
    """Check if normalized point (cx, cy) lies inside roi [x1, y1, x2, y2] (normalized 0-1)."""
    x1, y1, x2, y2 = roi[:4]
    return x1 <= cx_norm <= x2 and y1 <= cy_norm <= y2


def boxes_to_normalized(boxes_xyxy: list, width: int, height: int) -> list[tuple[float, float]]:
    """Convert pixel boxes to normalized (cx, cy) centers."""
    out = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2 / width if width else 0
        cy = (y1 + y2) / 2 / height if height else 0
        out.append((cx, cy))
    return out


def count_occupied_per_zone(
    boxes_xyxy: list,
    frame_width: int,
    frame_height: int,
    zones: list[dict],
) -> list[dict]:
    """
    For each zone, count how many person centers fall inside the zone ROI.
    Returns list of { "id", "name", "occupied", "total_seats", "available" }.
    """
    if not frame_width or not frame_height:
        return [
            {
                "id": z["id"],
                "name": z.get("name", z["id"]),
                "occupied": 0,
                "total_seats": z.get("total_seats", 1),
                "available": z.get("total_seats", 1),
            }
            for z in zones
        ]
    centers = boxes_to_normalized(boxes_xyxy, frame_width, frame_height)
    out = []
    for zone in zones:
        roi = zone.get("roi", [0, 0, 1, 1])
        total = zone.get("total_seats", 1)
        occupied = sum(1 for cx, cy in centers if point_in_roi(cx, cy, roi))
        occupied = min(occupied, total)
        out.append({
            "id": zone["id"],
            "name": zone.get("name", zone["id"]),
            "occupied": occupied,
            "total_seats": total,
            "available": total - occupied,
        })
    return out


def compute_occupancy(
    boxes_xyxy: list,
    frame_width: int,
    frame_height: int,
    config_path: str | Path = "config/zones.json",
) -> list[dict]:
    """Load zones and return per-zone occupancy. Root dir is project root."""
    zones = load_zones(config_path)
    return count_occupied_per_zone(boxes_xyxy, frame_width, frame_height, zones)
