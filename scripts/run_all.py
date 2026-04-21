"""
Start RoomRadar end-to-end:
1) API + dashboard
2) Camera pipeline posting counts to the API

Usage:
    source venv/bin/activate
    python -m scripts.run_all
"""
import argparse
import socket
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoomRadar API and camera pipeline")
    parser.add_argument("--source", default="0", help="Primary webcam index or video path")
    parser.add_argument(
        "--source-2",
        default="",
        help="Optional second camera index/video path (for dual-camera Pi setup)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show preview window")
    parser.add_argument("--config", default=None, help="Path to zones.json")
    parser.add_argument("--node-id", default=socket.gethostname(), help="Node id sent by both camera streams")
    parser.add_argument("--camera-id", default="cam_1", help="Camera id for --source stream")
    parser.add_argument("--camera-id-2", default="cam_2", help="Camera id for --source-2 stream")
    parser.add_argument(
        "--mode",
        choices=("zones", "chairs", "both"),
        default="zones",
        help="Passed through to run_camera (zones, chairs, or split both)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.28,
        help="chairs/both: overlap threshold (meaning depends on --chair-metric)",
    )
    parser.add_argument(
        "--chair-metric",
        choices=("blend", "iou", "ioa", "center+ioa"),
        default="blend",
        help="chairs/both: overlap metric forwarded to run_camera",
    )
    parser.add_argument("--person-conf", type=float, default=0.35, help="chairs/both: min person conf")
    parser.add_argument("--chair-conf", type=float, default=0.35, help="chairs/both: min chair conf")
    parser.add_argument(
        "--rank-with-det-conf",
        action="store_true",
        help="chairs/both: rank greedy matches using geom * det_conf",
    )
    parser.add_argument(
        "--require-foot-in-chair",
        action="store_true",
        help="chairs/both: stricter seated heuristic forwarded to run_camera",
    )
    parser.add_argument(
        "--chair-expand-frac",
        type=float,
        default=0.12,
        help="chairs/both: chair bbox expansion for foot test forwarded to run_camera",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="chairs/both: EMA alpha")
    parser.add_argument("--smooth-window", type=int, default=5, help="chairs/both: vote window")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO imgsz forwarded to run_camera")
    parser.add_argument("--half", action="store_true", help="YOLO half precision forwarded to run_camera")
    parser.add_argument("--max-det", type=int, default=50, help="YOLO max_det forwarded to run_camera")
    parser.add_argument("--agnostic-nms", action="store_true", help="YOLO agnostic_nms forwarded to run_camera")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (forwarded to both run_camera procs)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (forwarded to both run_camera procs)")
    parser.add_argument("--fps", type=int, default=15, help="Camera capture FPS (forwarded to both run_camera procs)")
    parser.add_argument(
        "--mjpg",
        action="store_true",
        help="Request MJPG from both cameras (forwarded to both run_camera procs)",
    )
    args = parser.parse_args()

    api_cmd = [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    camera_cmd = [
        sys.executable,
        "-m",
        "scripts.run_camera",
        "--source",
        str(args.source),
        "--api-url",
        "http://127.0.0.1:8000",
        "--node-id",
        str(args.node_id),
        "--camera-id",
        str(args.camera_id),
        "--mode",
        args.mode,
        "--iou-threshold",
        str(args.iou_threshold),
        "--chair-metric",
        str(args.chair_metric),
        "--person-conf",
        str(args.person_conf),
        "--chair-conf",
        str(args.chair_conf),
        "--smooth-alpha",
        str(args.smooth_alpha),
        "--smooth-window",
        str(args.smooth_window),
        "--imgsz",
        str(args.imgsz),
        "--max-det",
        str(args.max_det),
        "--chair-expand-frac",
        str(args.chair_expand_frac),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
    ]
    camera2_cmd = []
    if str(args.source_2):
        camera2_cmd = [
            sys.executable,
            "-m",
            "scripts.run_camera",
            "--source",
            str(args.source_2),
            "--api-url",
            "http://127.0.0.1:8000",
            "--node-id",
            str(args.node_id),
            "--camera-id",
            str(args.camera_id_2),
            "--mode",
            args.mode,
            "--iou-threshold",
            str(args.iou_threshold),
            "--chair-metric",
            str(args.chair_metric),
            "--person-conf",
            str(args.person_conf),
            "--chair-conf",
            str(args.chair_conf),
            "--smooth-alpha",
            str(args.smooth_alpha),
            "--smooth-window",
            str(args.smooth_window),
            "--imgsz",
            str(args.imgsz),
            "--max-det",
            str(args.max_det),
            "--chair-expand-frac",
            str(args.chair_expand_frac),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
        ]
    if args.no_show:
        camera_cmd.append("--no-show")
        if camera2_cmd:
            camera2_cmd.append("--no-show")
    if args.config:
        camera_cmd.extend(["--config", args.config])
        if camera2_cmd:
            camera2_cmd.extend(["--config", args.config])
    if args.rank_with_det_conf:
        camera_cmd.append("--rank-with-det-conf")
        if camera2_cmd:
            camera2_cmd.append("--rank-with-det-conf")
    if args.half:
        camera_cmd.append("--half")
        if camera2_cmd:
            camera2_cmd.append("--half")
    if args.agnostic_nms:
        camera_cmd.append("--agnostic-nms")
        if camera2_cmd:
            camera2_cmd.append("--agnostic-nms")
    if args.require_foot_in_chair:
        camera_cmd.append("--require-foot-in-chair")
        if camera2_cmd:
            camera2_cmd.append("--require-foot-in-chair")
    if args.mjpg:
        camera_cmd.append("--mjpg")
        if camera2_cmd:
            camera2_cmd.append("--mjpg")

    api_proc = subprocess.Popen(api_cmd)
    time.sleep(2)
    camera_proc = subprocess.Popen(camera_cmd)
    camera2_proc = subprocess.Popen(camera2_cmd) if camera2_cmd else None

    def shutdown(*_):
        procs = [camera_proc, api_proc]
        if camera2_proc is not None:
            procs.insert(1, camera2_proc)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
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
