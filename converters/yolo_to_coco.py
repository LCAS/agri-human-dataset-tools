#!/usr/bin/env python3
"""
Convert a YOLO dataset split into COCO JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO labels into COCO JSON.")
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing source images for one split.",
    )
    parser.add_argument(
        "--labels_dir",
        required=True,
        help="Directory containing YOLO txt label files for the same split.",
    )
    parser.add_argument(
        "--yaml_path",
        required=True,
        help="Path to the YOLO data.yaml file containing class names.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to write the COCO JSON output.",
    )
    parser.add_argument(
        "--category_id_start",
        type=int,
        default=1,
        help="Starting COCO category ID. Default: 1.",
    )
    return parser.parse_args()


def load_class_names(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data["names"]
    if isinstance(names, dict):
        return [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return list(names)
    raise ValueError(f"Unsupported names structure in {yaml_path}: expected list or dict.")


def image_files(images_dir: Path) -> Sequence[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(images_dir.glob(pattern)))
        files.extend(sorted(images_dir.glob(pattern.upper())))
    unique_files = sorted({path.resolve(): path for path in files}.values())
    return unique_files


def yolo_box_to_coco(
    x_center: float, y_center: float, box_width: float, box_height: float, width: int, height: int
) -> List[float]:
    abs_x = (x_center - box_width / 2.0) * width
    abs_y = (y_center - box_height / 2.0) * height
    abs_w = box_width * width
    abs_h = box_height * height
    return [abs_x, abs_y, abs_w, abs_h]


def convert_yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    yaml_path: Path,
    output_path: Path,
    category_id_start: int,
) -> Dict[str, object]:
    classes = load_class_names(yaml_path)
    coco: Dict[str, object] = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(classes, start=category_id_start)
        ],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    coco_images = coco["images"]
    coco_annotations = coco["annotations"]

    for image_id, image_path in enumerate(image_files(images_dir), start=1):
        with Image.open(image_path) as image:
            width, height = image.size

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:5])
                coco_category_id = class_id + category_id_start
                bbox = yolo_box_to_coco(
                    x_center, y_center, box_width, box_height, width, height
                )

                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": coco_category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)

    return coco


def main() -> None:
    args = parse_args()
    coco = convert_yolo_to_coco(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        yaml_path=Path(args.yaml_path),
        output_path=Path(args.output_path),
        category_id_start=args.category_id_start,
    )
    print(
        f"Done: {len(coco['images'])} images, {len(coco['annotations'])} annotations -> "
        f"{args.output_path}"
    )


if __name__ == "__main__":
    main()
