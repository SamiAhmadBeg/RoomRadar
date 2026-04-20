"""
Occupancy from COCO person + chair boxes: a seat is occupied only if a person
box overlaps that chair box above an IoU threshold. No zones.json required.

Uses pretrained YOLO (person=0, chair=56 on COCO). Fine-tune on your space later.
"""
from __future__ import annotations

from dataclasses import dataclass


def iou_xyxy(a: list[float], b: list[float]) -> float:
    """Intersection over union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def intersection_area_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return float(iw * ih)


def ioa_xyxy(a: list[float], b: list[float]) -> float:
    """Intersection over area(b)."""
    bx1, by1, bx2, by2 = b[:4]
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_b <= 0:
        return 0.0
    return intersection_area_xyxy(a, b) / area_b


def person_center_in_chair(person_xyxy: list[float], chair_xyxy: list[float]) -> bool:
    """Heuristic: seated people often have bbox center inside the chair bbox."""
    px1, py1, px2, py2 = person_xyxy[:4]
    cx = 0.5 * (px1 + px2)
    cy = 0.5 * (py1 + py2)
    bx1, by1, bx2, by2 = chair_xyxy[:4]
    return (cx >= bx1) and (cx <= bx2) and (cy >= by1) and (cy <= by2)


def expand_xyxy(xyxy: list[float], frac: float) -> list[float]:
    """Expand box by frac of its width/height (symmetric)."""
    x1, y1, x2, y2 = xyxy[:4]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    dx = w * float(frac)
    dy = h * float(frac)
    return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]


def point_in_xyxy(px: float, py: float, xyxy: list[float]) -> bool:
    x1, y1, x2, y2 = xyxy[:4]
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def person_foot_point_xy(person_xyxy: list[float]) -> tuple[float, float]:
    """Approximate contact point near the bottom of the person bbox (hips/feet region)."""
    x1, y1, x2, y2 = person_xyxy[:4]
    cx = 0.5 * (x1 + x2)
    h = max(1.0, y2 - y1)
    # Slightly above the bottom edge to tolerate bbox jitter.
    cy = y2 - 0.03 * h
    return cx, cy


def person_foot_in_chair(
    person_xyxy: list[float],
    chair_xyxy: list[float],
    chair_expand_frac: float,
) -> bool:
    chair_e = expand_xyxy(chair_xyxy, chair_expand_frac)
    fx, fy = person_foot_point_xy(person_xyxy)
    return point_in_xyxy(fx, fy, chair_e)


def overlap_score(
    metric: str,
    chair_xyxy: list[float],
    person_xyxy: list[float],
) -> float:
    metric = metric.lower().strip()
    if metric == "iou":
        return iou_xyxy(chair_xyxy, person_xyxy)
    if metric == "ioa":
        # How much of the chair bbox is covered by the person bbox (good for seated poses).
        return ioa_xyxy(person_xyxy, chair_xyxy)
    if metric == "blend":
        # Seated people: IoA is usually high; standing walk-by: IoU often low. Take the best of both.
        return max(
            iou_xyxy(chair_xyxy, person_xyxy),
            ioa_xyxy(person_xyxy, chair_xyxy),
        )
    if metric == "center+ioa":
        if not person_center_in_chair(person_xyxy, chair_xyxy):
            return 0.0
        return ioa_xyxy(person_xyxy, chair_xyxy)
    raise ValueError(f"Unknown chair match metric: {metric!r}")


@dataclass(frozen=True)
class ChairMatchConfig:
    metric: str = "blend"
    threshold: float = 0.25
    use_detection_score: bool = False
    require_foot_in_chair: bool = False
    chair_expand_frac: float = 0.12


def chair_occupied_by_overlap(
    chair_xyxy: list[float],
    person_boxes: list[list[float]],
    iou_threshold: float,
    metric: str = "iou",
) -> bool:
    """True if any person box overlaps this chair enough."""
    return any(overlap_score(metric, chair_xyxy, p) >= iou_threshold for p in person_boxes)


def _pair_scores(
    person_dets: list[dict],
    chair_dets: list[dict],
    cfg: ChairMatchConfig,
) -> list[tuple[float, float, int, int]]:
    """
    Return sorted candidates (rank_score, geom_score, person_idx, chair_idx), highest first.

    - geom_score is the geometric overlap used for thresholds/labels (IoU/IoA/etc).
    - rank_score optionally includes detection confidences to break ties / reduce junk matches.
    """
    pairs: list[tuple[float, float, int, int]] = []
    for pi, pd in enumerate(person_dets):
        p = pd["xyxy"]
        pc = float(pd.get("conf", 1.0))
        for ci, cd in enumerate(chair_dets):
            cbox = cd["xyxy"]
            cc = float(cd.get("conf", 1.0))
            if cfg.require_foot_in_chair and not person_foot_in_chair(
                p, cbox, cfg.chair_expand_frac
            ):
                continue
            g = overlap_score(cfg.metric, cbox, p)
            if g < cfg.threshold:
                continue
            rank = g
            if cfg.use_detection_score:
                rank *= pc * cc
            pairs.append((float(rank), float(g), pi, ci))
    pairs.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return pairs


def union_xyxy(a: list[float], b: list[float]) -> list[float]:
    """Axis-aligned union of two xyxy boxes."""
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def occupied_seat_pairs_matched(
    person_dets: list[dict],
    chair_dets: list[dict],
    cfg: ChairMatchConfig,
) -> dict[int, tuple[int, float]]:
    """
    Greedy one-to-one matching: chair_index -> (person_index, geom_score).

    Same pairing as occupied_chair_scores_matched; use for unified "occupied seat" UI.
    """
    pairs: dict[int, tuple[int, float]] = {}
    used_p: set[int] = set()
    used_c: set[int] = set()
    for _rank, geom, pi, ci in _pair_scores(person_dets, chair_dets, cfg):
        if pi in used_p or ci in used_c:
            continue
        used_p.add(pi)
        used_c.add(ci)
        pairs[ci] = (pi, float(geom))
    return pairs


def occupied_chair_scores_matched(
    person_dets: list[dict],
    chair_dets: list[dict],
    cfg: ChairMatchConfig,
) -> dict[int, float]:
    """
    Greedy one-to-one matching: each person can occupy at most one chair, each chair
    at most one person. This reduces false positives when a walking person overlaps
    multiple nearby chair boxes.
    """
    return {ci: geom for ci, (_pi, geom) in occupied_seat_pairs_matched(person_dets, chair_dets, cfg).items()}


def occupied_chair_scores(
    person_boxes: list[list[float]],
    chair_boxes: list[list[float]],
    iou_threshold: float = 0.15,
    metric: str = "iou",
) -> dict[int, float]:
    """
    Return {chair_index: score} for chairs considered occupied.

    score = max overlap_score(metric) across persons (range 0..1 for iou/ioa paths).
    """
    scores: dict[int, float] = {}
    for ci, c in enumerate(chair_boxes):
        best = 0.0
        for p in person_boxes:
            best = max(best, overlap_score(metric, c, p))
        if best >= iou_threshold:
            scores[ci] = best
    return scores


def occupied_chair_indices(
    person_boxes: list[list[float]],
    chair_boxes: list[list[float]],
    iou_threshold: float = 0.15,
    metric: str = "iou",
) -> set[int]:
    """Return indices of chair_boxes that are considered occupied."""
    return set(occupied_chair_scores(person_boxes, chair_boxes, iou_threshold, metric).keys())


def compute_chair_overlap_occupancy(
    person_dets: list[dict],
    chair_dets: list[dict],
    cfg: ChairMatchConfig,
) -> list[dict]:
    """
    One summary zone compatible with the API / dashboard.

    total_seats = number of detected chairs
    occupied = chairs matched to a person above threshold (greedy one-to-one)
    """
    n_chairs = len(chair_dets)
    if n_chairs == 0:
        return [
            {
                "id": "chair_overlap",
                "name": "Seats (chair + person match)",
                "total_seats": 0,
                "occupied": 0,
                "available": 0,
            }
        ]
    occ = occupied_chair_scores_matched(person_dets, chair_dets, cfg)
    occupied = len(occ)
    occupied = min(occupied, n_chairs)
    return [
        {
            "id": "chair_overlap",
            "name": "Seats (chair + person match)",
            "total_seats": n_chairs,
            "occupied": occupied,
            "available": n_chairs - occupied,
        }
    ]
