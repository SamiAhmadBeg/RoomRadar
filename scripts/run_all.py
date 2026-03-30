"""
Start RoomRadar end-to-end:
1) API + dashboard
2) Camera pipeline posting counts to the API

Usage:
    source venv/bin/activate
    python -m scripts.run_all
"""
import argparse
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoomRadar API and camera pipeline")
    parser.add_argument("--source", default="0", help="Webcam index or video path")
    parser.add_argument("--no-show", action="store_true", help="Do not show preview window")
    parser.add_argument("--config", default=None, help="Path to zones.json")
    args = parser.parse_args()

    api_cmd = [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    camera_cmd = [sys.executable, "-m", "scripts.run_camera", "--source", str(args.source), "--api-url", "http://127.0.0.1:8000"]
    if args.no_show:
        camera_cmd.append("--no-show")
    if args.config:
        camera_cmd.extend(["--config", args.config])

    api_proc = subprocess.Popen(api_cmd)
    time.sleep(2)
    camera_proc = subprocess.Popen(camera_cmd)

    def shutdown(*_):
        for proc in (camera_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (camera_proc, api_proc):
            try:
                proc.wait(timeout=5)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
        return 0

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        return camera_proc.wait()
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
