#!/usr/bin/env python3
"""
Fine-tune YOLO on datasets/inferx_lab (pseudo-labeled InferX footage).

Run after:
  python scripts/build_inferx_dataset.py

Then use the exported weights with run_camera:
  python -m scripts.run_camera --mode both --source 0 --model runs/detect/train/weights/best.pt
(extend run_camera with --model if not present; see README note.)

Defaults favor small-dataset overfitting (high epochs, low patience).
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "inferx_lab" / "data.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=0, help="0 = no early stop (overfit small sets)")
    parser.add_argument("--device", default="", help="mps | cpu | 0 | (empty = auto)")
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"Missing {args.data}. Run: python scripts/build_inferx_dataset.py", flush=True)
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kw = dict(
        data=str(args.data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        lr0=float(args.lr0),
        patience=int(args.patience),
        augment=True,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.7,
        name="inferx_lab",
        exist_ok=True,
        plots=True,
    )
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)
    print("Training finished. Weights under runs/detect/inferx_lab/weights/best.pt (typical Ultralytics layout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
