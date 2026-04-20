"""
RoomRadar detection: webcam/video -> YOLOv8 person detection -> bounding boxes.
Run: python -m detection.detect [--source 0]   (0 = webcam, or path to video/file)
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class ids (YOLOv8n COCO)
PERSON_CLASS_ID = 0
CHAIR_CLASS_ID = 56

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _is_image_source(source) -> bool:
    if isinstance(source, Path):
        return source.suffix.lower() in _IMAGE_EXTS
    if isinstance(source, str) and not str(source).isdigit():
        return Path(source).suffix.lower() in _IMAGE_EXTS
    return False


def get_detector(model_path: str = "yolov8n.pt"):
    """Load YOLO model (downloads on first run)."""
    return YOLO(model_path)


def run_detection(
    source=0,
    model_path: str = "yolov8n.pt",
    show: bool = True,
    classes: list[int] | None = None,
    annotate_frame=None,
    window_title: str = "RoomRadar",
    predict_kw: dict | None = None,
):
    """
    Run person detection on webcam (source=0), video file, or a single image path.
    Yields (frame, results) per frame. results[0].boxes has xyxy, conf, cls.

    If annotate_frame is set, it is called as annotate_frame(frame, results) and must
    return a BGR image to show instead of results.plot().
    """
    if classes is None:
        classes = [PERSON_CLASS_ID]
    model = get_detector(model_path)
    predict_kw = dict(predict_kw or {})

    # Static image: run inference once, hold window until ESC (VideoCapture is unreliable for images).
    if _is_image_source(source):
        path = str(source)
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        results = model(frame, classes=classes, verbose=False, **predict_kw)
        try:
            if show:
                while True:
                    if annotate_frame is not None:
                        annotated = annotate_frame(frame, results)
                    else:
                        annotated = results[0].plot()
                    cv2.imshow(window_title, annotated)
                    if cv2.waitKey(50) & 0xFF == 27:
                        break
            yield frame, results
        finally:
            if show:
                cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, classes=classes, verbose=False, **predict_kw)
            if annotate_frame is not None:
                annotated = annotate_frame(frame, results)
                if show:
                    cv2.imshow(window_title, annotated)
                    if cv2.waitKey(1) == 27:  # ESC
                        break
            elif show:
                annotated = results[0].plot()
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) == 27:  # ESC
                    break
            yield frame, results
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def get_person_boxes(results):
    """Extract person bounding boxes (xyxy) from YOLO results."""
    if not results or len(results) == 0:
        return []
    boxes = results[0].boxes
    if boxes is None:
        return []
    return boxes.xyxy.cpu().numpy().tolist()


def get_boxes_by_class(results, class_id: int, conf_min: float = 0.0) -> list[list[float]]:
    """Extract xyxy boxes for a single COCO class id from YOLO results."""
    dets = get_detections_by_class(results, class_id, conf_min=conf_min)
    return [d["xyxy"] for d in dets]


def get_detections_by_class(results, class_id: int, conf_min: float = 0.0) -> list[dict]:
    """Extract detections {xyxy, conf} for a single COCO class id."""
    if not results or len(results) == 0:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    out: list[dict] = []
    for i in range(len(cls)):
        if cls[i] != class_id:
            continue
        c = float(conf[i])
        if c < conf_min:
            continue
        out.append({"xyxy": xyxy[i].tolist(), "conf": c})
    return out


def main():
    parser = argparse.ArgumentParser(description="RoomRadar person detection")
    parser.add_argument(
        "--source",
        default=0,
        help="Webcam index (0) or path to video/image",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show preview window (e.g. for headless)",
    )
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    for frame, results in run_detection(
        source=source,
        model_path=args.model,
        show=not args.no_show,
    ):
        boxes = get_person_boxes(results)
        n = len(boxes)
        if n > 0:
            print(f"People detected: {n}", end="\r")
    print("\nDone.")


if __name__ == "__main__":
    main()
