#!/usr/bin/env python3
"""
Build a YOLO-format dataset under datasets/inferx_lab/ from InferX lab media.

- Extracts frames from videos (MOV/MP4).
- Converts iPhone HEIC to JPEG via macOS `sips` when OpenCV cannot read them.
- Writes pseudo-labels using a teacher YOLO model (COCO classes person=0, chair=56),
  remapped to dataset classes person=0, chair=1.

Example:
  cd RoomRadar && source venv/bin/activate
  python scripts/build_inferx_dataset.py --stride 5 --conf 0.22

Then train:
  python scripts/train_inferx.py
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_GLOB = "occupiedandunoccupied/*.MOV"
COCO_PERSON = 0
COCO_CHAIR = 56


def _xyxy_to_yolo_line(xyxy: list[float], iw: int, ih: int, cls_id: int) -> str:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / iw
    cy = (y1 + y2) / 2.0 / ih
    w = bw / iw
    h = bh / ih
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(1e-6, w))
    h = min(1.0, max(1e-6, h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def _map_coco_cls(c: int) -> int | None:
    if c == COCO_PERSON:
        return 0
    if c == COCO_CHAIR:
        return 1
    return None


def _heic_to_jpeg(src: Path, dst: Path) -> bool:
    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(src), "--out", str(dst)],
            check=True,
            capture_output=True,
        )
        return dst.is_file() and dst.stat().st_size > 0
    except Exception:
        return False


def _collect_videos(project_root: Path, extra: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in extra:
        for p in project_root.glob(pat):
            if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v", ".avi"}:
                paths.append(p)
    return sorted(set(paths))


def _collect_heic_dirs(project_root: Path, dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        root = project_root / d
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.suffix.lower() in {".heic", ".heif"}:
                out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build inferx_lab YOLO dataset (pseudo-labeled)")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "inferx_lab",
        help="Output dataset root (YOLO layout)",
    )
    parser.add_argument(
        "--teacher",
        default="yolov8m.pt",
        help="Teacher weights for pseudo-labels (downloads if missing)",
    )
    parser.add_argument(
        "--video-glob",
        action="append",
        default=[],
        help=f"Glob relative to project root (repeatable). Default: {DEFAULT_VIDEO_GLOB}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Save every Nth frame from each video",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.22,
        help="Min confidence to keep a pseudo box",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Cap frames per video (0 = no cap)",
    )
    parser.add_argument(
        "--include-heic-dirs",
        nargs="*",
        default=["person_two_standing(not_seated)"],
        help="Folder names under project root containing HEIC stills",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Train/val split seed",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.12,
        help="Fraction of images for validation",
    )
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    out_root: Path = args.out
    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    globs = args.video_glob if args.video_glob else [DEFAULT_VIDEO_GLOB]
    videos = _collect_videos(PROJECT_ROOT, globs)
    if not videos:
        print("No videos matched. Pass --video-glob 'occupiedandunoccupied/*.MOV'", file=sys.stderr)
        return 1

    print(f"Teacher: {args.teacher}")
    model = YOLO(str(PROJECT_ROOT / args.teacher) if (PROJECT_ROOT / args.teacher).is_file() else args.teacher)

    tmp_dir = out_root / "_tmp_heic"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(bgr, stem: str) -> None:
        h, w = bgr.shape[:2]
        r = model.predict(bgr, classes=[COCO_PERSON, COCO_CHAIR], conf=float(args.conf), verbose=False)[0]
        lines: list[str] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(cls)):
                mapped = _map_coco_cls(int(cls[i]))
                if mapped is None:
                    continue
                lines.append(_xyxy_to_yolo_line(xyxy[i].tolist(), w, h, mapped))
        # write to train first; split later
        ip = img_train / f"{stem}.jpg"
        lp = lbl_train / f"{stem}.txt"
        cv2.imwrite(str(ip), bgr)
        lp.write_text("".join(lines))

    # Videos
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"Skip (cannot open): {vid}", file=sys.stderr)
            continue
        base = vid.stem.replace(" ", "_")
        fi = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % int(args.stride) != 0:
                fi += 1
                continue
            stem = f"{base}_f{fi:06d}"
            add_frame(frame, stem)
            saved += 1
            if args.max_frames_per_video and saved >= args.max_frames_per_video:
                break
            fi += 1
        cap.release()
        print(f"Video {vid.name}: saved ~{saved} frames (stride={args.stride})")

    # HEIC folders
    heics = _collect_heic_dirs(PROJECT_ROOT, list(args.include_heic_dirs))
    for hp in heics:
        jp = tmp_dir / f"{hp.stem}.jpg"
        if not _heic_to_jpeg(hp, jp):
            print(f"Skip HEIC (sips failed): {hp}", file=sys.stderr)
            continue
        im = cv2.imread(str(jp))
        if im is None:
            print(f"Skip HEIC (imread failed): {jp}", file=sys.stderr)
            continue
        stem = f"heic_{hp.parent.name}_{hp.stem}".replace(" ", "_")
        add_frame(im, stem)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Train/val split: move files
    pairs = sorted(img_train.glob("*.jpg"))
    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * float(args.val_fraction)))
    val_set = set(p.name for p in pairs[:n_val])
    for ip in pairs:
        name = ip.name
        stem = ip.stem
        lp = lbl_train / f"{stem}.txt"
        if not lp.is_file():
            lp.write_text("")
        if name in val_set:
            shutil.move(str(ip), str(img_val / name))
            shutil.move(str(lp), str(lbl_val / f"{stem}.txt"))

    yaml = out_root / "data.yaml"
    yaml.write_text(
        f"""path: {out_root.as_posix()}
train: images/train
val: images/val
nc: 2
names:
  0: person
  1: chair
""",
        encoding="utf-8",
    )

    n_tr = len(list(img_train.glob("*.jpg")))
    n_va = len(list(img_val.glob("*.jpg")))
    print(f"Done. images train={n_tr} val={n_va}")
    print(f"Wrote {yaml}")
    print("Next: python scripts/train_inferx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
