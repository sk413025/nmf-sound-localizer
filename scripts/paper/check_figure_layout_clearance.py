#!/usr/bin/env python3
"""Audit manuscript/layout clearances from layout metadata sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox(axis: dict[str, Any], key: str = "decorated_bbox_mm") -> dict[str, float] | None:
    return axis.get(key) or axis.get("bbox_mm")


def _panel_and_role(axis: dict[str, Any]) -> tuple[str | None, str | None]:
    gid = axis.get("gid")
    if not gid:
        return None, None
    parts = str(gid).split(".")
    if len(parts) >= 3 and parts[1].startswith("panel_"):
        return parts[1], parts[2]
    return None, None


def _edges(bbox: dict[str, float]) -> tuple[float, float, float, float]:
    left = float(bbox["x0"])
    bottom = float(bbox["y0"])
    right = left + float(bbox["width"])
    top = bottom + float(bbox["height"])
    return left, bottom, right, top


def _x_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    a_left, _a_bottom, a_right, _a_top = _edges(a)
    b_left, _b_bottom, b_right, _b_top = _edges(b)
    return min(a_right, b_right) - max(a_left, b_left)


def _y_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    _a_left, a_bottom, _a_right, a_top = _edges(a)
    _b_left, b_bottom, _b_right, b_top = _edges(b)
    return min(a_top, b_top) - max(a_bottom, b_bottom)


def _vertical_clearance(upper: dict[str, float], lower: dict[str, float]) -> float:
    _u_left, u_bottom, _u_right, _u_top = _edges(upper)
    _l_left, _l_bottom, _l_right, l_top = _edges(lower)
    return u_bottom - l_top


def _horizontal_clearance(left_box: dict[str, float], right_box: dict[str, float]) -> float:
    _l_left, _l_bottom, l_right, _l_top = _edges(left_box)
    r_left, _r_bottom, _r_right, _r_top = _edges(right_box)
    return r_left - l_right


def _union_bbox(boxes: list[dict[str, float]]) -> dict[str, float]:
    left = min(box["x0"] for box in boxes)
    bottom = min(box["y0"] for box in boxes)
    right = max(box["x0"] + box["width"] for box in boxes)
    top = max(box["y0"] + box["height"] for box in boxes)
    return {
        "x0": round(left, 3),
        "y0": round(bottom, 3),
        "width": round(right - left, 3),
        "height": round(top - bottom, 3),
    }


def _center_y(bbox: dict[str, float]) -> float:
    return float(bbox["y0"]) + float(bbox["height"]) / 2.0


def _stack_clearances(payload: dict[str, Any], threshold_mm: float) -> list[dict[str, Any]]:
    panel_groups: dict[str, list[dict[str, Any]]] = {}
    for axis in payload.get("axes", []):
        panel_id, role = _panel_and_role(axis)
        if panel_id is None or role == "colorbar":
            continue
        box = _bbox(axis)
        if box is None:
            continue
        panel_groups.setdefault(panel_id, []).append(axis)

    rows: list[dict[str, Any]] = []
    for panel_id, axes in sorted(panel_groups.items()):
        if len(axes) < 2:
            continue
        axes_sorted = sorted(
            axes,
            key=lambda item: float((_bbox(item) or {"y0": 0.0})["y0"]),
            reverse=True,
        )
        for upper, lower in zip(axes_sorted, axes_sorted[1:]):
            upper_box = _bbox(upper)
            lower_box = _bbox(lower)
            if upper_box is None or lower_box is None:
                continue
            if _x_overlap(upper_box, lower_box) <= 0:
                continue
            clearance = _vertical_clearance(upper_box, lower_box)
            rows.append(
                {
                    "panel_id": panel_id,
                    "upper_gid": upper.get("gid"),
                    "lower_gid": lower.get("gid"),
                    "clearance_mm": round(clearance, 3),
                    "threshold_mm": threshold_mm,
                    "passed": clearance >= threshold_mm,
                }
            )
    return rows


def _panel_clearances(payload: dict[str, Any], threshold_mm: float) -> list[dict[str, Any]]:
    panel_boxes: dict[str, dict[str, float]] = {}
    grouped_boxes: dict[str, list[dict[str, float]]] = {}
    for axis in payload.get("axes", []):
        panel_id, _role = _panel_and_role(axis)
        box = _bbox(axis)
        if panel_id is None or box is None:
            continue
        grouped_boxes.setdefault(panel_id, []).append(box)
    for panel_id, boxes in grouped_boxes.items():
        panel_boxes[panel_id] = _union_bbox(boxes)

    rows: list[dict[str, Any]] = []
    panels = sorted(panel_boxes.items(), key=lambda item: (item[1]["y0"], item[1]["x0"]), reverse=True)
    for idx, (left_id, left_box) in enumerate(panels):
        for right_id, right_box in panels[idx + 1:]:
            if _y_overlap(left_box, right_box) <= 0:
                continue
            same_row = abs(_center_y(left_box) - _center_y(right_box)) <= 0.5 * min(
                float(left_box["height"]),
                float(right_box["height"]),
            )
            if not same_row:
                continue
            l_left, _l_bottom, _l_right, _l_top = _edges(left_box)
            r_left, _r_bottom, _r_right, _r_top = _edges(right_box)
            if l_left <= r_left:
                ordered_left_id, ordered_left_box = left_id, left_box
                ordered_right_id, ordered_right_box = right_id, right_box
            else:
                ordered_left_id, ordered_left_box = right_id, right_box
                ordered_right_id, ordered_right_box = left_id, left_box
            clearance = _horizontal_clearance(ordered_left_box, ordered_right_box)
            rows.append(
                {
                    "left_panel": ordered_left_id,
                    "right_panel": ordered_right_id,
                    "clearance_mm": round(clearance, 3),
                    "threshold_mm": threshold_mm,
                    "passed": clearance >= threshold_mm,
                }
            )
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["left_panel"], row["right_panel"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check layout clearances from layout metadata.")
    parser.add_argument("layout_json", help="Path to the layout metadata JSON file.")
    parser.add_argument("--stack-threshold-mm", type=float, default=1.5)
    parser.add_argument("--panel-threshold-mm", type=float, default=1.0)
    parser.add_argument("--report", help="Optional JSON report output path.")
    args = parser.parse_args()

    layout_path = (_repo_root() / args.layout_json).resolve() if not Path(args.layout_json).is_absolute() else Path(args.layout_json)
    payload = _load_payload(layout_path)

    stack_rows = _stack_clearances(payload, args.stack_threshold_mm)
    panel_rows = _panel_clearances(payload, args.panel_threshold_mm)
    all_rows = stack_rows + panel_rows
    overall_pass = all(row["passed"] for row in all_rows)

    report = {
        "layout_json": str(layout_path),
        "figure_mm": payload.get("figure_mm"),
        "stack_threshold_mm": args.stack_threshold_mm,
        "panel_threshold_mm": args.panel_threshold_mm,
        "stack_clearances": stack_rows,
        "panel_clearances": panel_rows,
        "overall_pass": overall_pass,
        "min_stack_clearance_mm": None if not stack_rows else min(row["clearance_mm"] for row in stack_rows),
        "min_panel_clearance_mm": None if not panel_rows else min(row["clearance_mm"] for row in panel_rows),
    }

    if args.report:
        report_path = (_repo_root() / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
