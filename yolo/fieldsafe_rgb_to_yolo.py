#!/usr/bin/env python3
"""
Convert FieldSafePedestrian demo RGB images and text annotations into YOLO.

This exporter merges all source classes into one class: "person".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FieldSafePedestrian RGB data to YOLO with a single 'person' class."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to the FieldSafePedestrian_Demo root containing RGB/ and Annotation/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the YOLO dataset.",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy RGB images to <out>/images/{train,val}. If omitted, only labels and split files are created.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Optional confidence threshold on the source 'score' field.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio in [0,1]. Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val splitting.",
    )
    return parser.parse_args()


def read_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            return img.size[0], img.size[1]
    except Exception as exc:
        raise RuntimeError(
            f"Failed reading image size for {image_path}. Install pillow. Details: {exc}"
        ) from exc


def clamp_bbox_xyxy(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Optional[List[float]]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_xywh_to_yolo(
    x: float, y: float, box_width: float, box_height: float, width: int, height: int
) -> List[float]:
    center_x = (x + box_width / 2.0) / float(width)
    center_y = (y + box_height / 2.0) / float(height)
    norm_w = box_width / float(width)
    norm_h = box_height / float(height)
    return [
        max(0.0, min(1.0, center_x)),
        max(0.0, min(1.0, center_y)),
        max(0.0, min(1.0, norm_w)),
        max(0.0, min(1.0, norm_h)),
    ]


def parse_annotation_line(line: str) -> Optional[Tuple[float, float, float, float, float]]:
    parts = [part for part in re.split(r"[,\s]+", line.strip()) if part]
    if len(parts) < 6:
        return None
    try:
        score = float(parts[0])
        x1 = float(parts[2])
        y1 = float(parts[3])
        x2 = float(parts[4])
        y2 = float(parts[5])
    except ValueError:
        return None
    return score, x1, y1, x2, y2


def write_subset_yolo(
    records: List[Dict[str, object]],
    subset_name: str,
    out_images_root: Path,
    out_labels_root: Path,
    copy_images: bool,
) -> Tuple[int, int]:
    subset_image_dir = out_images_root / subset_name
    subset_label_dir = out_labels_root / subset_name
    subset_label_dir.mkdir(parents=True, exist_ok=True)

    if copy_images:
        subset_image_dir.mkdir(parents=True, exist_ok=True)

    annotation_count = 0
    for rec in records:
        image_name = str(rec["image_name"])

        if copy_images:
            dst = subset_image_dir / image_name
            if not dst.exists():
                shutil.copy2(Path(rec["image_path"]), dst)

        label_lines = []
        for yolo_box in rec["yolo_boxes"]:
            x_c, y_c, box_w, box_h = yolo_box
            label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}")
            annotation_count += 1

        label_path = subset_label_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return len(records), annotation_count


def export(
    dataset_root: Path,
    out_dir: Path,
    copy_images: bool,
    score_threshold: float,
    train_ratio: float,
    seed: int,
) -> None:
    rgb_dir = dataset_root / "RGB"
    ann_dir = dataset_root / "Annotation"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_splits = out_dir / "splits"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)
    if copy_images:
        out_images.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(rgb_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No RGB JPG images found in {rgb_dir}")

    records: List[Dict[str, object]] = []
    missing_annotations = 0
    skipped_low_score = 0
    skipped_bad_boxes = 0

    for image_path in image_paths:
        ann_path = ann_dir / f"{image_path.stem}.txt"
        if not ann_path.exists():
            missing_annotations += 1
            continue

        width, height = read_image_size(image_path)
        yolo_boxes: List[List[float]] = []

        for line in ann_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_annotation_line(line)
            if parsed is None:
                continue

            score, x1, y1, x2, y2 = parsed
            if score < score_threshold:
                skipped_low_score += 1
                continue

            bbox = clamp_bbox_xyxy(x1, y1, x2, y2, width, height)
            if bbox is None:
                skipped_bad_boxes += 1
                continue

            yolo_boxes.append(
                bbox_xywh_to_yolo(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
            )

        records.append(
            {
                "image_name": image_path.name,
                "image_path": image_path,
                "yolo_boxes": yolo_boxes,
            }
        )

    if len(records) < 2:
        raise RuntimeError("Need at least 2 valid image/annotation pairs to split train/val.")

    rng = random.Random(seed)
    rng.shuffle(records)

    train_count = int(round(len(records) * train_ratio))
    train_count = max(1, min(train_count, len(records) - 1))
    train_records = records[:train_count]
    val_records = records[train_count:]

    train_images, train_annotations = write_subset_yolo(
        train_records, "train", out_images, out_labels, copy_images
    )
    val_images, val_annotations = write_subset_yolo(
        val_records, "val", out_images, out_labels, copy_images
    )

    (out_splits / "train.txt").write_text(
        "\n".join(str(rec["image_name"]) for rec in train_records) + "\n",
        encoding="utf-8",
    )
    (out_splits / "val.txt").write_text(
        "\n".join(str(rec["image_name"]) for rec in val_records) + "\n",
        encoding="utf-8",
    )

    (out_dir / "classes.txt").write_text("person\n", encoding="utf-8")
    data_yaml = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: person\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "dataset_root": str(dataset_root),
        "images_total_rgb": len(image_paths),
        "images_exported_total": len(records),
        "annotations_exported_total": train_annotations + val_annotations,
        "train_images": train_images,
        "train_annotations": train_annotations,
        "val_images": val_images,
        "val_annotations": val_annotations,
        "missing_annotation_files": missing_annotations,
        "skipped_low_score": skipped_low_score,
        "skipped_bad_boxes": skipped_bad_boxes,
        "score_threshold": score_threshold,
        "train_ratio": train_ratio,
        "seed": seed,
        "copied_images": copy_images,
        "train_labels_dir": str(out_labels / "train"),
        "val_labels_dir": str(out_labels / "val"),
        "data_yaml": str(out_dir / "data.yaml"),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    export(
        dataset_root=args.dataset_root,
        out_dir=args.out,
        copy_images=args.copy_images,
        score_threshold=args.score_threshold,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
