#!/usr/bin/env python3
"""Convert Unified_Dataset (YOLO detection boxes) into Unified_Dataset_seg
(YOLO-seg polygons) using SAM2 box prompts.

Adapted from generate_new_data_seg.py: same box->mask->polygon conversion
logic, retargeted at Unified_Dataset's train/val/test x good/defect x
images/labels layout and the merged 3-class taxonomy (Hair already folded
into Foreign_Body upstream).

Good images need no SAM call (they have no boxes) -- they are copied
through unchanged with an empty label file, matching the detection-stage
convention.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "Bug", 1: "Foreign_Body", 2: "Stain"}
COLORS = {
    0: (60, 60, 230),    # Bug -- red
    1: (230, 140, 40),   # Foreign_Body -- orange
    2: (60, 200, 60),    # Stain -- green
}
MIN_AREA_PX = 32
BOX_EXPAND = 0.15
SIMPLIFY_EPS = 0.0015
# Foreign_Body is the thin/low-fill class (bounding box mostly background);
# Bug and Stain are blob-like and expected to fill their box more fully.
LOW_FILL_CLASSES = (1,)


def parse_boxes(label_path: Path, width: int, height: int):
    boxes = []
    text = label_path.read_text(encoding="utf-8-sig") if label_path.exists() else ""
    for line_no, line in enumerate(text.splitlines(), 1):
        z = line.split()
        if not z:
            continue
        if len(z) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected 5 fields, got {len(z)}")
        cls_f, xc, yc, bw, bh = map(float, z)
        cls = int(cls_f)
        if cls not in CLASS_NAMES or cls != cls_f:
            raise ValueError(f"{label_path}:{line_no}: invalid class {cls_f}")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, bw, bh)):
            raise ValueError(f"{label_path}:{line_no}: coordinate outside [0,1]")
        x1 = max(0.0, (xc - bw / 2) * width)
        y1 = max(0.0, (yc - bh / 2) * height)
        x2 = min(float(width - 1), (xc + bw / 2) * width)
        y2 = min(float(height - 1), (yc + bh / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{label_path}:{line_no}: degenerate bounding box")
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def rectangle_polygon(box):
    _, x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def clean_sam_mask(mask: np.ndarray, box, width: int, height: int,
                   keep_all=False, close_px=0, simplify_eps=SIMPLIFY_EPS,
                   min_component_px=MIN_AREA_PX):
    """SAM mask -> polygon(s). Returns (polygons, total_area, touches_clip).

    keep_all=False reproduces the original behaviour exactly: only the single
    largest connected component survives. That is what erased every thin
    structure in the corpus -- a mosquito's legs and a hair's wisps are
    separate components from the body once SAM's mask is thresholded, so they
    were discarded at annotation time. Measured over the 2879 training
    instances built this way, ZERO were thinner than 2px at model scale, so
    the segmenter was never shown a filament as a positive example.

    keep_all=True keeps every component above min_component_px; close_px first
    dilates-then-erodes so a leg that merely NEARLY touches the body stays one
    component rather than becoming a second instance.
    """
    cls, x1, y1, x2, y2 = box
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    bw, bh = x2 - x1, y2 - y1
    ex, ey = bw * BOX_EXPAND, bh * BOX_EXPAND
    cx1, cy1 = int(max(0, np.floor(x1 - ex))), int(max(0, np.floor(y1 - ey)))
    cx2, cy2 = int(min(width - 1, np.ceil(x2 + ex))), int(min(height - 1, np.ceil(y2 + ey)))
    clipped = np.zeros_like(mask)
    clipped[cy1:cy2 + 1, cx1:cx2 + 1] = mask[cy1:cy2 + 1, cx1:cx2 + 1]

    if close_px > 0:
        k = int(close_px) | 1  # odd kernel
        clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE,
                                   np.ones((k, k), np.uint8))

    n, cc, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    if n <= 1:
        return [], 0, False

    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    if keep_all:
        chosen = [i for i in order if stats[i, cv2.CC_STAT_AREA] >= min_component_px]
    else:
        chosen = order[:1]
        if stats[chosen[0], cv2.CC_STAT_AREA] < MIN_AREA_PX:
            return [], int(stats[chosen[0], cv2.CC_STAT_AREA]), False
    if not chosen:
        return [], int(stats[order[0], cv2.CC_STAT_AREA]), False

    polygons, total_area, touches_clip = [], 0, False
    for comp in chosen:
        area = int(stats[comp, cv2.CC_STAT_AREA])
        selected = np.where(cc == comp, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        if simplify_eps > 0:
            polygon = cv2.approxPolyDP(contour, max(0.5, simplify_eps * perimeter), True)
            polygon = polygon.reshape(-1, 2).astype(np.float32)
        else:
            polygon = contour.reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3:
            polygon = contour.reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3:
            continue
        polygons.append(polygon)
        total_area += area
        touches_clip = touches_clip or bool(
            selected[cy1:cy1 + 1, cx1:cx2 + 1].any()
            or selected[cy2:cy2 + 1, cx1:cx2 + 1].any()
            or selected[cy1:cy2 + 1, cx1:cx1 + 1].any()
            or selected[cy1:cy2 + 1, cx2:cx2 + 1].any()
        )
    return polygons, total_area, touches_clip


def polygon_line(cls: int, polygon: np.ndarray, width: int, height: int):
    points = polygon.astype(np.float64).copy()
    points[:, 0] = np.clip(points[:, 0] / width, 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / height, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in points.reshape(-1))


def draw_preview(image, boxes, polygons, methods):
    overlay = image.copy()
    for box, polygon in zip(boxes, polygons):
        cls = box[0]
        cv2.fillPoly(overlay, [np.round(polygon).astype(np.int32)], COLORS[cls])
    vis = cv2.addWeighted(overlay, 0.42, image, 0.58, 0)
    for box, polygon, method in zip(boxes, polygons, methods):
        cls, x1, y1, x2, y2 = box
        color = COLORS[cls]
        cv2.rectangle(vis, (round(x1), round(y1)), (round(x2), round(y2)), color, 5)
        cv2.polylines(vis, [np.round(polygon).astype(np.int32)], True, color, 5, cv2.LINE_AA)
        cv2.putText(
            vis, f"{CLASS_NAMES[cls]} [{method}]",
            (round(x1), max(40, round(y1) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA,
        )
    scale = min(1.0, 1600 / vis.shape[1])
    if scale < 1.0:
        vis = cv2.resize(vis, (round(vis.shape[1] * scale), round(vis.shape[0] * scale)))
    return vis


def copy_good(src_split_dir: Path, dst_split_dir: Path, totals: Counter):
    src_img_dir = src_split_dir / "good" / "images"
    src_lbl_dir = src_split_dir / "good" / "labels"
    dst_img_dir = dst_split_dir / "good" / "images"
    dst_lbl_dir = dst_split_dir / "good" / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(src_img_dir.glob("*.jpg")):
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        lbl_src = src_lbl_dir / f"{img_path.stem}.txt"
        lbl_dst = dst_lbl_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")
        totals["good_copied"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("/media/m100/32AACB51AACB1071/industrial_v2/Unified_Dataset"))
    ap.add_argument("--output", type=Path, default=Path("/media/m100/32AACB51AACB1071/industrial_v2/Unified_Dataset_seg"))
    ap.add_argument("--sam-model", type=Path, default=Path("/media/m100/32AACB51AACB1071/task/sam2_b.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="Process at most N defect images per split; 0 means all.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-previews", action="store_true")
    # Thin-structure recovery. Defaults reproduce the original corpus byte for
    # byte so existing runs stay comparable; pass --keep-components all to stop
    # discarding legs, wisps and satellite specks at annotation time.
    ap.add_argument("--keep-components", choices=["largest", "all"], default="largest",
                    help="'largest' keeps only the biggest SAM component (original "
                         "behaviour, erases thin structure); 'all' keeps every "
                         "component above --min-component-px.")
    ap.add_argument("--min-component-px", type=int, default=MIN_AREA_PX,
                    help="Minimum native-pixel area for a kept component.")
    ap.add_argument("--close-frac", type=float, default=0.0,
                    help="Morphological close with kernel = frac * box diagonal "
                         "before component analysis, so a nearly-touching leg "
                         "stays attached to its body. 0 disables.")
    ap.add_argument("--simplify-eps", type=float, default=SIMPLIFY_EPS,
                    help="approxPolyDP epsilon as a fraction of contour perimeter; "
                         "0 keeps the full contour (preserves filaments).")
    args = ap.parse_args()

    from ultralytics import SAM

    if not args.sam_model.is_file():
        raise FileNotFoundError(args.sam_model)
    model = SAM(str(args.sam_model))
    report_rows = []
    totals = Counter()

    for split in args.splits:
        src_split_dir = args.source / split
        dst_split_dir = args.output / split

        # Good images: no SAM, just copy through.
        copy_good(src_split_dir, dst_split_dir, totals)

        # Defect images: SAM box-prompted conversion.
        image_dir = src_split_dir / "defect" / "images"
        label_dir = src_split_dir / "defect" / "labels"
        out_labels = dst_split_dir / "defect" / "labels"
        out_images = dst_split_dir / "defect" / "images"
        out_previews = dst_split_dir / "defect" / "previews"
        out_labels.mkdir(parents=True, exist_ok=True)
        out_images.mkdir(parents=True, exist_ok=True)
        if not args.no_previews:
            out_previews.mkdir(parents=True, exist_ok=True)

        label_files = sorted(label_dir.glob("*.txt"))
        if args.limit:
            label_files = label_files[:args.limit]
        print(f"{split}/defect: {len(label_files)} labeled images")

        for index, label_path in enumerate(label_files, 1):
            out_label = out_labels / label_path.name
            image_path = image_dir / f"{label_path.stem}.jpg"
            out_image = out_images / image_path.name
            if args.resume and out_label.exists() and out_image.exists():
                totals["resumed"] += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read {image_path}")
            height, width = image.shape[:2]
            boxes = parse_boxes(label_path, width, height)
            shutil.copy2(image_path, out_image)
            if not boxes:
                out_label.write_text("", encoding="utf-8")
                continue

            prompts = [[b[1], b[2], b[3], b[4]] for b in boxes]
            results = model(image, bboxes=prompts, device=args.device, verbose=False)
            mask_data = None
            if results and results[0].masks is not None:
                mask_data = results[0].masks.data.cpu().numpy()

            polygons, methods, lines = [], [], []
            poly_boxes = []  # parallel to polygons; a multipart box repeats
            for box_index, box in enumerate(boxes):
                cls, x1, y1, x2, y2 = box
                mask = mask_data[box_index] if mask_data is not None and box_index < len(mask_data) else None
                parts = []
                mask_area = 0
                touches_clip = False
                if mask is not None:
                    close_px = (args.close_frac
                                * np.hypot(x2 - x1, y2 - y1)) if args.close_frac else 0
                    parts, mask_area, touches_clip = clean_sam_mask(
                        mask, box, width, height,
                        keep_all=args.keep_components == "all",
                        close_px=close_px, simplify_eps=args.simplify_eps,
                        min_component_px=args.min_component_px)
                method = "sam2"
                flags = []
                if not parts:
                    parts = [rectangle_polygon(box)]
                    method = "bbox_fallback"
                    mask_area = round((x2 - x1) * (y2 - y1))
                    flags.append("SAM_EMPTY")
                if len(parts) > 1:
                    flags.append(f"MULTIPART_{len(parts)}")
                    totals["multipart_instances"] += 1
                bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
                area_ratio = mask_area / bbox_area
                min_ratio = 0.005 if cls in LOW_FILL_CLASSES else 0.02
                if area_ratio < min_ratio:
                    flags.append("LOW_FILL")
                if area_ratio > 1.15:
                    flags.append("HIGH_FILL")
                if touches_clip:
                    flags.append("TOUCHES_CLIP")

                # One YOLO-seg line per component. The downstream target is a
                # semantic raster (rasterise_classes), so extra lines paint the
                # legs correctly; they do inflate GT instance counts, which the
                # instance evaluator must group by parent box.
                for part in parts:
                    polygons.append(part)
                    poly_boxes.append(box)
                    methods.append(method)
                    lines.append(polygon_line(cls, part, width, height))
                totals[method] += 1
                totals[f"class_{cls}"] += 1
                if flags:
                    totals["flagged_instances"] += 1
                report_rows.append({
                    "split": split, "image": image_path.name, "box_index": box_index,
                    "class_id": cls, "class_name": CLASS_NAMES[cls], "method": method,
                    "bbox_pixels": round(bbox_area), "mask_pixels": mask_area,
                    "mask_bbox_ratio": f"{area_ratio:.6f}", "touches_clip": int(touches_clip),
                    "flags": ";".join(flags),
                })

            out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if not args.no_previews:
                preview = draw_preview(image, poly_boxes, polygons, methods)
                cv2.imwrite(str(out_previews / image_path.name), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            totals["images"] += 1
            if index % 50 == 0 or index == len(label_files):
                print(f"  {index}/{len(label_files)}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classes.txt").write_text(
        "\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n"
    )
    (args.output / "data.yaml").write_text(
        f"path: {args.output}\n"
        f"train:\n  - train/good/images\n  - train/defect/images\n"
        f"val:\n  - val/good/images\n  - val/defect/images\n"
        f"test:\n  - test/good/images\n  - test/defect/images\n\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items()))
    )

    fields = [
        "split", "image", "box_index", "class_id", "class_name", "method",
        "bbox_pixels", "mask_pixels", "mask_bbox_ratio", "touches_clip", "flags",
    ]
    with (args.output / "conversion_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nSummary:")
    for key in sorted(totals):
        print(f"  {key}: {totals[key]}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
