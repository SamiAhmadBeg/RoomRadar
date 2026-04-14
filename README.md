# RoomRadar

Edge occupancy detection for study spaces. Camera → local inference → anonymous counts only (no video leaves the device).

## Setup (Mac)

```bash
cd RoomRadar
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
``` (Windows)
.\venv\Scripts\Activate.ps1
```
## Run

**Detection + occupancy (webcam, prints counts):**
```bash
python -m scripts.run_camera
```
Optional: `--no-show` to hide preview; `--source path/to/video.mp4` for a file.

**API + dashboard:**
```bash
uvicorn api.server:app --reload --host 0.0.0.0
```
- Dashboard: http://127.0.0.1:8000
- Occupancy JSON: http://127.0.0.1:8000/occupancy
- Swagger: http://127.0.0.1:8000/docs

**Everything at once:**
```bash
python -m scripts.run_all
```
This starts the API first, then the camera pipeline posting counts to it.

**Push counts to API while running camera:**
```bash
# Terminal 1
uvicorn api.server:app --reload --host 0.0.0.0
# Terminal 2
python -m scripts.run_camera --api-url http://127.0.0.1:8000
```

## Config

Edit `config/zones.json` to define ROI zones (normalized 0–1: `[x1, y1, x2, y2]`) and `total_seats` per zone. Person detections whose center falls inside a zone count as occupied.

## Project layout

- `detection/` — YOLOv8 person detection
- `occupancy/` — ROI overlap → available/occupied counts
- `api/` — FastAPI, GET/POST `/occupancy`
- `config/zones.json` — zone definitions
