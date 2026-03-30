"""
Run full pipeline: webcam -> YOLO person detection -> ROI occupancy -> print counts.
Run from project root: python -m scripts.run_camera
Optional: run API in another terminal (uvicorn api.server:app) and use --api-url to POST counts.
"""
import argparse
import json
import urllib.request
from pathlib import Path

# Project root (RoomRadar/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def post_occupancy(api_url: str, zones: list[dict]) -> None:
    """POST zone counts to RoomRadar API (no deps beyond stdlib)."""
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/occupancy",
        data=json.dumps({"zones": zones}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # ignore if API not running


def main():
    from detection.detect import run_detection, get_person_boxes
    from occupancy.seat_counter import compute_occupancy

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Webcam index or video path")
    parser.add_argument("--config", default=None, help="Path to zones.json")
    parser.add_argument("--no-show", action="store_true", help="No preview window")
    parser.add_argument("--api-url", default="", help="e.g. http://127.0.0.1:8000 to POST counts")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    config_path = args.config or (PROJECT_ROOT / "config" / "zones.json")

    for frame, results in run_detection(source=source, show=not args.no_show):
        boxes = get_person_boxes(results)
        h, w = frame.shape[:2]
        zones = compute_occupancy(boxes, w, h, config_path=config_path)
        if args.api_url:
            post_occupancy(args.api_url, zones)
        for z in zones:
            print(
                f"  {z['name']}: {z['available']}/{z['total_seats']} available, "
                f"{z['occupied']} occupied",
                end="",
            )
        print("", end="\r")
    print("\nDone.")


if __name__ == "__main__":
    main()
