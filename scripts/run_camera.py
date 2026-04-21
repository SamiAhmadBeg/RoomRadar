"""
Run full pipeline: webcam/video -> YOLO -> occupancy -> print counts (and optional API POST).

Modes:
  zones  (default)  person boxes + config/zones.json ROIs
  chairs             COCO person + chair; occupied only if IoU overlap (no zones file)
  both               one YOLO pass; split view left=zones+ROIs, right=person+chair boxes; both counts

Run from project root: python -m scripts.run_camera
"""
import argparse
import json
import socket
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

# Project root (RoomRadar/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def post_occupancy(
    api_url: str,
    zones: list[dict],
    *,
    node_id: str,
    camera_id: str,
    seq: int,
) -> None:
    """POST zone counts to RoomRadar API (no deps beyond stdlib)."""
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/occupancy",
        data=json.dumps(
            {
                "node_id": node_id,
                "camera_id": camera_id,
                "ts_ms": int(time.time() * 1000),
                "seq": int(seq),
                "zones": zones,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # ignore if API not running


class _ChairOccSmoother:
    """Lightweight temporal smoothing for per-chair occupancy (reduces flicker)."""

    def __init__(self, alpha: float, window: int) -> None:
        self.alpha = float(alpha)
        self.window = int(window)
        self._hist: dict[int, deque[int]] = defaultdict(deque)
        self._ema: dict[int, float] = {}

    def smooth(self, occ_scores: dict[int, float], n_chairs: int) -> dict[int, float]:
        out: dict[int, float] = {}
        for i in range(max(0, int(n_chairs))):
            x = float(occ_scores.get(i, 0.0))
            prev = float(self._ema.get(i, 0.0))
            a = self.alpha
            ema = (a * x) + ((1.0 - a) * prev)
            self._ema[i] = ema

            if self.window > 1:
                dq = self._hist[i]
                dq.append(1 if x >= 0.5 else 0)
                while len(dq) > self.window:
                    dq.popleft()
                vote = (sum(dq) / len(dq)) if dq else 0.0
                # EMA shapes the displayed "confidence", vote stabilizes flicker.
                out[i] = max(ema, vote)
            else:
                out[i] = ema
        return out


def _overlay_font_scale(frame_w: int, frame_h: int) -> tuple[float, int]:
    """
    OpenCV label sizing tuned for high-res inputs (4K screenshots looked "invisible"
    at fixed fontScale=0.45). This tracks Ultralytics-ish readability: scale up with
    the shorter image dimension, clamp to sane bounds.
    """
    short = max(1, min(int(frame_w), int(frame_h)))
    # ~0.9 at 720p short side, ~2.7 at 2160p short side (4K portrait-ish), capped.
    font_scale = float(max(0.75, min(2.75, (short / 800.0) * 0.9)))
    thickness = 2 if font_scale < 1.35 else 3
    return font_scale, thickness


def _put_cv2_label(
    img,
    text: str,
    x: int,
    y_top: int,
    y_bottom: int,
    color_bgr: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """YOLO-ish label: filled strip behind text for contrast."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 6, 6
    h, w = img.shape[:2]

    # Prefer above the box (like Ultralytics), but fall back below if we'd clip at y=0.
    label_h = th + pad_y * 2 + max(0, baseline - 1)
    place_above = (y_top - label_h) >= 2
    if place_above:
        y1 = y_top - label_h
        y2 = y_top
    else:
        y1 = y_bottom
        y2 = min(h - 1, y_bottom + label_h)

    x1 = max(0, x)
    x2 = min(w - 1, x1 + tw + pad_x * 2)
    y1 = max(0, y1)
    y2 = min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness=-1)
    # White text reads well on saturated bbox colors.
    cv2.putText(
        img,
        text,
        (x1 + pad_x, y2 - pad_y - 1),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def main():
    from detection.detect import (
        CHAIR_CLASS_ID,
        PERSON_CLASS_ID,
        run_detection,
        get_detections_by_class,
        get_person_boxes,
    )
    from occupancy.chair_overlap import (
        ChairMatchConfig,
        occupied_chair_scores_matched,
        occupied_seat_pairs_matched,
        union_xyxy,
    )
    from occupancy.seat_counter import compute_occupancy, load_zones

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Webcam index or video path")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights (.pt), e.g. runs/detect/inferx_lab/weights/best.pt after fine-tune",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=PERSON_CLASS_ID,
        help="Class id for person in the weights file (0 for COCO and InferX 2-class)",
    )
    parser.add_argument(
        "--chair-class-id",
        type=int,
        default=CHAIR_CLASS_ID,
        help="Class id for chair (56 COCO; use 1 for InferX 2-class person+chair fine-tune)",
    )
    parser.add_argument("--config", default=None, help="Path to zones.json (zones mode only)")
    parser.add_argument("--no-show", action="store_true", help="No preview window")
    parser.add_argument(
        "--save",
        default="",
        help="Optional: path to save the last annotated frame (useful for images).",
    )
    parser.add_argument("--api-url", default="", help="e.g. http://127.0.0.1:8000 to POST counts")
    parser.add_argument(
        "--node-id",
        default=socket.gethostname(),
        help="Unique sender id for this inference node (e.g. rubik-pi hostname)",
    )
    parser.add_argument(
        "--camera-id",
        default="cam_1",
        help="Unique camera id under this node (e.g. cam_1, cam_2)",
    )
    parser.add_argument(
        "--mode",
        choices=("zones", "chairs", "both"),
        default="zones",
        help="zones | chairs | both (split screen, one video, two methods)",
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
        help="chairs/both: blend=max(iou,ioa) default for seated; iou alone often fails seated vs chair",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.35,
        help="chairs/both: min YOLO confidence for person boxes",
    )
    parser.add_argument(
        "--chair-conf",
        type=float,
        default=0.35,
        help="chairs/both: min YOLO confidence for chair boxes",
    )
    parser.add_argument(
        "--rank-with-det-conf",
        action="store_true",
        help="chairs/both: when greedy-matching, rank pairs using geom * det_conf product",
    )
    parser.add_argument(
        "--require-foot-in-chair",
        action="store_true",
        help="chairs/both: require person foot-point inside expanded chair box (cuts walk-by FP)",
    )
    parser.add_argument(
        "--chair-expand-frac",
        type=float,
        default=0.12,
        help="chairs/both: expand chair bbox fraction for foot-in-chair test (with --require-foot-in-chair)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.35,
        help="chairs/both: EMA alpha for occupancy score smoothing (0 disables EMA effect if also 0?)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="chairs/both: majority-vote window over frames (1 disables voting)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO inference size (multiple of 32). Smaller=faster, less accurate.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="YOLO half precision inference (GPU/MPS dependent; faster if supported)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=50,
        help="YOLO max detections per frame",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="YOLO class-agnostic NMS (sometimes helps when boxes overlap a lot)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (dual-cam Pi: keep low)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (dual-cam Pi: keep low)")
    parser.add_argument("--fps", type=int, default=15, help="Camera capture FPS (dual-cam Pi: keep low)")
    parser.add_argument(
        "--mjpg",
        action="store_true",
        help="Request MJPG camera format (reduces USB bandwidth pressure on many webcams)",
    )
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    config_path = args.config or (PROJECT_ROOT / "config" / "zones.json")

    pid, cid = int(args.person_class_id), int(args.chair_class_id)
    coco_person_chair = [pid, cid]
    det_classes = coco_person_chair if args.mode in ("chairs", "both") else None

    predict_kw: dict = {
        "imgsz": int(args.imgsz),
        "half": bool(args.half),
        "max_det": int(args.max_det),
        "agnostic_nms": bool(args.agnostic_nms),
    }
    camera_kw: dict = {
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "mjpg": bool(args.mjpg),
    }

    chair_cfg = ChairMatchConfig(
        metric=str(args.chair_metric),
        threshold=float(args.iou_threshold),
        use_detection_score=bool(args.rank_with_det_conf),
        require_foot_in_chair=bool(args.require_foot_in_chair),
        chair_expand_frac=float(args.chair_expand_frac),
    )
    chair_smoother = _ChairOccSmoother(alpha=float(args.smooth_alpha), window=int(args.smooth_window))
    seq = 0

    if args.mode == "both":
        import cv2
        import numpy as np

        state: dict = {}

        def annotate_dual(frame, results):
            person_dets = get_detections_by_class(
                results, pid, conf_min=float(args.person_conf)
            )
            chair_dets = get_detections_by_class(
                results, cid, conf_min=float(args.chair_conf)
            )
            persons = [d["xyxy"] for d in person_dets]
            chairs = [d["xyxy"] for d in chair_dets]
            h, w = frame.shape[:2]
            box_font_scale, box_thickness = _overlay_font_scale(w, h)
            banner_font_scale, banner_thickness = _overlay_font_scale(w, h)
            # Slightly smaller than bbox labels for top banners, but still readable in 4K.
            banner_font_scale = max(0.75, banner_font_scale * 0.85)
            banner_thickness = 2 if banner_font_scale < 1.35 else 3
            state["zones"] = compute_occupancy(persons, w, h, config_path=config_path)
            seat_pairs = occupied_seat_pairs_matched(person_dets, chair_dets, chair_cfg)
            raw_occ = {ci: geom for ci, (_pi, geom) in seat_pairs.items()}
            sm = chair_smoother.smooth(raw_occ, n_chairs=len(chair_dets))
            n_chairs = len(chair_dets)
            occupied = sum(1 for i in range(n_chairs) if sm.get(i, 0.0) >= 0.5)
            state["chairs"] = [
                {
                    "id": "chair_overlap",
                    "name": "Seats (chair + person match)",
                    "total_seats": n_chairs,
                    "occupied": occupied,
                    "available": max(0, n_chairs - occupied),
                }
            ]
            # Store smoothed 0..1 scores for visualization; threshold at 0.5 for occupied styling.
            state["occupied_chair_scores"] = {i: float(sm[i]) for i in range(n_chairs) if sm.get(i, 0.0) >= 0.5}

            left = frame.copy()
            for z in load_zones(config_path):
                roi = z.get("roi", [0, 0, 1, 1])
                x1, y1 = int(roi[0] * w), int(roi[1] * h)
                x2, y2 = int(roi[2] * w), int(roi[3] * h)
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 200, 100), 2)
                _put_cv2_label(
                    left,
                    z.get("name", z["id"]),
                    x1,
                    y1,
                    y2,
                    (0, 200, 100),
                    box_font_scale,
                    box_thickness,
                )
            for box in persons:
                bx = [int(v) for v in box[:4]]
                cv2.rectangle(left, (bx[0], bx[1]), (bx[2], bx[3]), (0, 165, 255), 2)
            banner_y = int(28 * (min(h, w) / 720.0))
            _put_cv2_label(
                left,
                "Zones (person center in ROI)",
                8,
                banner_y,
                banner_y + 6,
                (0, 165, 255),
                banner_font_scale,
                banner_thickness,
            )

            right = frame.copy()
            occ_scores = state.get("occupied_chair_scores", {})
            # Persons who are part of a smoothed "occupied seat" (drawn as union, not separately).
            seated_person_idx: set[int] = set()
            for ci in range(n_chairs):
                if sm.get(ci, 0.0) < 0.5:
                    continue
                if ci not in seat_pairs:
                    continue
                pi, _geom = seat_pairs[ci]
                seated_person_idx.add(pi)
                pbox = person_dets[pi]["xyxy"]
                cbox = chairs[ci]
                u = union_xyxy(pbox, cbox)
                ux = [int(v) for v in u[:4]]
                pct = int(round(float(sm.get(ci, 0.0)) * 100))
                color = (0, 200, 255)  # cyan: unified occupied seat
                cv2.rectangle(right, (ux[0], ux[1]), (ux[2], ux[3]), color, 3)
                _put_cv2_label(
                    right,
                    f"OCCUPIED SEAT {pct}%",
                    ux[0],
                    ux[1],
                    ux[3],
                    color,
                    box_font_scale,
                    box_thickness,
                )
            # Smoothed "occupied" but no raw person-chair pair this frame (temporal hold).
            for ci in range(n_chairs):
                if sm.get(ci, 0.0) < 0.5 or ci in seat_pairs:
                    continue
                bx = [int(v) for v in chairs[ci][:4]]
                pct = int(round(float(sm.get(ci, 0.0)) * 100))
                color = (0, 200, 255)
                cv2.rectangle(right, (bx[0], bx[1]), (bx[2], bx[3]), color, 3)
                _put_cv2_label(
                    right,
                    f"OCCUPIED SEAT {pct}%",
                    bx[0],
                    bx[1],
                    bx[3],
                    color,
                    box_font_scale,
                    box_thickness,
                )
            # Empty / available chairs (not smoothed-occupied).
            for i, c in enumerate(chairs):
                if sm.get(i, 0.0) >= 0.5:
                    continue
                bx = [int(v) for v in c[:4]]
                color = (60, 200, 80)
                cv2.rectangle(right, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                _put_cv2_label(
                    right,
                    "AVAILABLE SEAT",
                    bx[0],
                    bx[1],
                    bx[3],
                    color,
                    box_font_scale,
                    box_thickness,
                )
            # Standing / walking people (not matched to a seat this frame).
            for pi, p in enumerate(persons):
                if pi in seated_person_idx:
                    continue
                bx = [int(v) for v in p[:4]]
                cv2.rectangle(right, (bx[0], bx[1]), (bx[2], bx[3]), (255, 120, 0), 2)
                _put_cv2_label(
                    right,
                    "person (not seated)",
                    bx[0],
                    bx[1],
                    bx[3],
                    (255, 120, 0),
                    box_font_scale,
                    box_thickness,
                )
            target_h = left.shape[0]
            if right.shape[0] != target_h:
                scale = target_h / right.shape[0]
                right = cv2.resize(
                    right,
                    (int(right.shape[1] * scale), target_h),
                    interpolation=cv2.INTER_AREA,
                )
            gap_w = 6
            sep = np.zeros((target_h, gap_w, 3), dtype=np.uint8)
            sep[:] = (45, 45, 45)
            combined = np.hstack([left, sep, right])
            cw, ch = combined.shape[1], combined.shape[0]
            c_banner_scale, c_banner_thick = _overlay_font_scale(cw, ch)
            c_banner_scale = max(0.75, c_banner_scale * 0.85)
            c_banner_thick = 2 if c_banner_scale < 1.35 else 3
            c_banner_y = int(28 * (min(ch, cw) / 720.0))
            _put_cv2_label(
                combined,
                f"Chairs: metric={args.chair_metric} thr={args.iou_threshold:.2f} (smoothed)",
                left.shape[1] + gap_w + 8,
                c_banner_y,
                c_banner_y + 6,
                (255, 120, 0),
                c_banner_scale,
                c_banner_thick,
            )

            bar_h = max(52, int(round(44 * (min(ch, cw) / 720.0))))
            footer = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
            footer[:] = (28, 28, 28)
            zparts = [f"{z['name'][:12]}:{z['occupied']}/{z['total_seats']}" for z in state["zones"]]
            cz = state["chairs"][0] if state["chairs"] else {}
            cpart = f"chairs {cz.get('occupied', 0)}/{cz.get('total_seats', 0)}"
            fw, fh = footer.shape[1], footer.shape[0]
            f_scale, f_thick = _overlay_font_scale(fw, fh)
            f_scale = max(0.65, f_scale * 0.75)
            f_thick = 2 if f_scale < 1.2 else 3
            y1 = int(fh * 0.42)
            y2 = int(fh * 0.82)
            cv2.putText(
                footer,
                " | ".join(zparts)[:100],
                (10, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                f_scale,
                (200, 220, 200),
                f_thick,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                footer,
                cpart,
                (10, y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                f_scale,
                (200, 200, 255),
                f_thick,
                lineType=cv2.LINE_AA,
            )
            annotated = np.vstack([combined, footer])
            state["last_annotated"] = annotated
            return annotated

        for _frame, _results in run_detection(
            source=source,
            model_path=str(args.model),
            show=not args.no_show,
            classes=coco_person_chair,
            annotate_frame=annotate_dual,
            window_title="RoomRadar zones | chairs (ESC)",
            predict_kw=predict_kw,
            camera_kw=camera_kw,
        ):
            merged = list(state.get("zones", [])) + list(state.get("chairs", []))
            if args.api_url and merged:
                seq += 1
                post_occupancy(
                    args.api_url,
                    merged,
                    node_id=args.node_id,
                    camera_id=args.camera_id,
                    seq=seq,
                )
            zt = state.get("zones", [])
            ct = state.get("chairs", [{}])[0] if state.get("chairs") else {}
            zline = "  ".join(f"{z['name'][:8]}:{z['occupied']}/{z['total_seats']}" for z in zt)
            cline = f"chairs:{ct.get('occupied', 0)}/{ct.get('total_seats', 0)}"
            print(f"  {zline}  |  {cline}", end="\r")
        if args.save and state.get("last_annotated") is not None:
            out_path = Path(args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), state["last_annotated"])
        print("\nDone.")
        return

    for frame, results in run_detection(
        source=source,
        model_path=str(args.model),
        show=not args.no_show,
        classes=det_classes,
        predict_kw=predict_kw,
        camera_kw=camera_kw,
    ):
        if args.mode == "chairs":
            person_dets = get_detections_by_class(
                results, pid, conf_min=float(args.person_conf)
            )
            chair_dets = get_detections_by_class(
                results, cid, conf_min=float(args.chair_conf)
            )
            raw_occ = occupied_chair_scores_matched(person_dets, chair_dets, chair_cfg)
            sm = chair_smoother.smooth(raw_occ, n_chairs=len(chair_dets))
            n_chairs = len(chair_dets)
            occupied = sum(1 for i in range(n_chairs) if sm.get(i, 0.0) >= 0.5)
            zones = [
                {
                    "id": "chair_overlap",
                    "name": "Seats (chair + person match)",
                    "total_seats": n_chairs,
                    "occupied": occupied,
                    "available": max(0, n_chairs - occupied),
                }
            ]
        else:
            boxes = get_person_boxes(results)
            h, w = frame.shape[:2]
            zones = compute_occupancy(boxes, w, h, config_path=config_path)
        if args.api_url:
            seq += 1
            post_occupancy(
                args.api_url,
                zones,
                node_id=args.node_id,
                camera_id=args.camera_id,
                seq=seq,
            )
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
