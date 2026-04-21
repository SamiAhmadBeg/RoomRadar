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

**Chair + person overlap (COCO, no extra dataset download):** occupied only if a person box overlaps a chair box (IoU). Uses pretrained `person` (0) and `chair` (56).

```bash
python -m scripts.run_camera --mode chairs --source videos/your.mp4
python -m scripts.run_camera --mode chairs --iou-threshold 0.2 --api-url http://127.0.0.1:8000
```

Default `--mode zones` uses `config/zones.json`. For library accuracy you can still fine-tune on your own images later.

**Both methods at once (split view on one video or webcam):**
```bash
python -m scripts.run_camera --mode both --source videos/your.mp4
```
Left panel shows ROIs + person boxes for zone counts. Right panel is the usual YOLO plot (person + chair). Footer shows both summaries. API gets zone rows plus the chair-overlap summary row if you pass `--api-url`.

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

**Two camera inputs (same node):** pass a second source; each stream uses its own `--camera-id` and the API **aggregates** zones. Fusion rule: for each zone id, take the **maximum** `occupied` across fresh cameras, capped by the largest `total_seats` for that zone — **if any camera still sees seats taken, the fused snapshot does not drop below that maximum** (no majority vote that clears another camera’s occupied view).

```bash
python -m scripts.run_all --source 0 --source-2 1 --camera-id cam_1 --camera-id-2 cam_2 --node-id rubik-pi --mjpg
```
Use `--width`, `--height`, `--fps` on `run_all` to keep USB bandwidth reasonable on edge boards.

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
- `occupancy/` — ROI overlap (`seat_counter.py`) or chair+person IoU (`chair_overlap.py`)
- `api/` — FastAPI, GET/POST `/occupancy`
- `config/zones.json` — zone definitions
