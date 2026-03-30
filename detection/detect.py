"""
RoomRadar detection: webcam/video -> YOLOv8 person detection -> bounding boxes.
Run: python -m detection.detect [--source 0]   (0 = webcam, or path to video/file)
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class 0 = person
PERSON_CLASS_ID = 0


def get_detector(model_path: str = "yolov8n.pt"):
    """Load YOLO model (downloads on first run)."""
    return YOLO(model_path)


def run_detection(
    source=0,
    model_path: str = "yolov8n.pt",
    show: bool = True,
    classes: list[int] | None = None,
):
    """
    Run person detection on webcam (source=0) or video file.
    Yields (frame, results) per frame. results[0].boxes has xyxy, conf, cls.
    """
    if classes is None:
        classes = [PERSON_CLASS_ID]
    model = get_detector(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, classes=classes, verbose=False)
            if show:
                annotated = results[0].plot()
                cv2.imshow("RoomRadar", annotated)
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
