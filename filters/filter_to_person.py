#!/usr/bin/env python3
"""
Filter images and annotations from COCO or YOLO-format datasets, keeping only
user-specified human-related classes and remapping them to one "person" label.

Output is COCO JSON, with optional image copy or symlink export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter datasets to human classes and merge them into a single 'person' label."
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["coco", "kitti"],
        help="Input dataset format: 'coco' or 'kitti' (YOLO txt labels as used by Ultralytics KITTI zips).",
    )
    parser.add_argument(
        "--ann_file",
        default=None,
        help="[COCO only] Path to the COCO annotation JSON file.",
    )
    parser.add_argument(
        "--label_dir",
        default=None,
        help="[kitti only] Directory containing per-image YOLO txt label files.",
    )
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the filtered COCO JSON file.",
    )
    parser.add_argument(
        "--human_labels",
        nargs="+",
        required=True,
        help=(
            "For COCO: class names to keep, e.g. --human_labels person. "
            "For kitti: class IDs to keep, e.g. --human_labels 3 4 5. "
            "All matches are merged into one 'person' category."
        ),
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy filtered images instead of creating symbolic links.",
    )
    parser.add_argument(
        "--img_out_dir",
        default=None,
        help="Destination folder for filtered images.",
    )
    parser.add_argument(
        "--img_prefix",
        default=None,
        help="Optional path prefix prepended to every image file_name in the output JSON.",
    )

    args = parser.parse_args()
    if args.format == "coco" and not args.ann_file:
        parser.error("--ann_file is required for COCO format.")
    if args.format == "kitti" and not args.label_dir:
        parser.error("--label_dir is required for kitti format.")
    if args.img_out_dir is None and args.copy_images:
        parser.error("--img_out_dir is required when --copy_images is set.")
    return args


def _apply_prefix(filename: str, prefix: Optional[str]) -> str:
    if not prefix:
        return filename
    return f"{prefix.strip('/')}/{filename}"


def make_coco_output() -> Dict[str, object]:
    return {
        "info": {"description": "Filtered dataset with person class only"},
        "licenses": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": [],
        "annotations": [],
    }


def filter_coco(
    ann_file: str, human_labels: List[str], img_prefix: Optional[str]
) -> Dict[str, object]:
    print(f"[COCO] Loading annotations from: {ann_file}")
    with open(ann_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    human_cat_ids = {
        cat["id"] for cat in coco["categories"] if cat["name"] in set(human_labels)
    }
    matched_names = [c["name"] for c in coco["categories"] if c["id"] in human_cat_ids]
    print(f"[COCO] Matched categories: {matched_names} -> merged into 'person'")

    if not human_cat_ids:
        available = [c["name"] for c in coco["categories"]]
        raise ValueError(
            f"None of the requested labels {human_labels} were found. Available: {available}"
        )

    filtered_annotations = [
        ann for ann in coco["annotations"] if ann["category_id"] in human_cat_ids
    ]
    valid_image_ids = {ann["image_id"] for ann in filtered_annotations}
    filtered_images = [img for img in coco["images"] if img["id"] in valid_image_ids]

    print(f"[COCO] Kept {len(filtered_annotations)} / {len(coco['annotations'])} annotations.")
    print(f"[COCO] Kept {len(filtered_images)} / {len(coco['images'])} images.")

    output = make_coco_output()
    output_images = output["images"]
    output_annotations = output["annotations"]

    for img in filtered_images:
        new_img = dict(img)
        new_img["file_name"] = _apply_prefix(img["file_name"], img_prefix)
        output_images.append(new_img)

    for ann in filtered_annotations:
        new_ann = dict(ann)
        new_ann["category_id"] = 1
        output_annotations.append(new_ann)

    return output


def _find_image_file(img_dir: str, stem: str) -> Optional[str]:
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        candidate = Path(img_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def _get_image_dimensions(img_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image as PILImage  # type: ignore

        with PILImage.open(img_path) as image:
            return image.size
    except ImportError:
        pass

    with open(img_path, "rb") as handle:
        header = handle.read(24)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        import struct

        width = struct.unpack(">I", header[16:20])[0]
        height = struct.unpack(">I", header[20:24])[0]
        return width, height

    if header[:2] == b"\xff\xd8":
        import struct

        with open(img_path, "rb") as handle:
            handle.read(2)
            while True:
                marker, length = struct.unpack(">HH", handle.read(4))
                if marker in (0xFFC0, 0xFFC1, 0xFFC2):
                    handle.read(1)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return width, height
                handle.read(length - 2)

    raise ValueError(f"Cannot read dimensions for unsupported image format: {img_path}")


def filter_kitti(
    label_dir: str, img_dir: str, human_labels: List[str], img_prefix: Optional[str]
) -> Dict[str, object]:
    print(f"[KITTI/YOLO] Loading labels from: {label_dir}")
    human_ids: Set[int] = {int(label) for label in human_labels}
    print(f"[KITTI/YOLO] Keeping class IDs: {sorted(human_ids)} -> merged into 'person'")

    output = make_coco_output()
    output_images = output["images"]
    output_annotations = output["annotations"]

    image_id = 0
    annotation_id = 0
    total_annotations = 0
    kept_annotations = 0

    for label_path in sorted(Path(label_dir).glob("*.txt")):
        stem = label_path.stem
        person_boxes: List[Tuple[float, float, float, float]] = []

        with open(label_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                total_annotations += 1
                if class_id not in human_ids:
                    continue

                x_center, y_center, width_norm, height_norm = map(float, parts[1:5])
                person_boxes.append((x_center, y_center, width_norm, height_norm))
                kept_annotations += 1

        if not person_boxes:
            continue

        img_filename = _find_image_file(img_dir, stem)
        if img_filename is None:
            print(f"[WARN] No image file found for label '{stem}', skipping.")
            continue

        img_path = Path(img_dir) / img_filename
        img_width, img_height = _get_image_dimensions(img_path)

        image_id += 1
        output_images.append(
            {
                "id": image_id,
                "file_name": _apply_prefix(img_filename, img_prefix),
                "width": img_width,
                "height": img_height,
            }
        )

        for x_center_norm, y_center_norm, width_norm, height_norm in person_boxes:
            box_width = width_norm * img_width
            box_height = height_norm * img_height
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            x_top_left = x_center - box_width / 2
            y_top_left = y_center - box_height / 2

            annotation_id += 1
            output_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_top_left, y_top_left, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )

    print(
        f"[KITTI/YOLO] Kept {kept_annotations} / {total_annotations} annotations across "
        f"{len(output_images)} images."
    )
    return output


def export_filtered_images(
    output_coco: Dict[str, object], img_dir: str, img_out_dir: str, copy_mode: bool = False
) -> None:
    os.makedirs(img_out_dir, exist_ok=True)
    filenames = {img["file_name"] for img in output_coco["images"]}
    mode = "COPY" if copy_mode else "SYMLINK"
    print(f"[{mode}] Exporting {len(filenames)} images to: {img_out_dir}")

    for filename in filenames:
        bare_name = Path(filename).name
        src = Path(img_dir).resolve() / bare_name
        dst = Path(img_out_dir) / bare_name

        if not src.exists():
            print(f"[WARN] Image not found, skipping: {src}")
            continue
        if dst.exists():
            continue

        try:
            if copy_mode:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
        except OSError as exc:
            print(f"[ERROR] Failed for {filename}: {exc}")

    print(f"[{mode}] Done.")


def main() -> None:
    args = parse_args()

    if args.format == "coco":
        output = filter_coco(
            ann_file=args.ann_file,
            human_labels=args.human_labels,
            img_prefix=args.img_prefix,
        )
    else:
        output = filter_kitti(
            label_dir=args.label_dir,
            img_dir=args.img_dir,
            human_labels=args.human_labels,
            img_prefix=args.img_prefix,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(
        f"[OUTPUT] Saved filtered COCO JSON -> {out_path} "
        f"({len(output['images'])} images, {len(output['annotations'])} annotations)"
    )

    if args.img_out_dir:
        export_filtered_images(
            output_coco=output,
            img_dir=args.img_dir,
            img_out_dir=args.img_out_dir,
            copy_mode=args.copy_images,
        )


if __name__ == "__main__":
    main()
