"""
filter_to_person.py
--------------------
Filters images and annotations from COCO or KITTI (Ultralytics YOLO format)
datasets, keeping only user-specified human-related classes and remapping
them all to a single "person" label.

Output: a COCO-format JSON file (+ optionally copies filtered images)
that can then be converted to YOLO format or used directly for Faster-RCNN.

Supported input formats:
  - coco : standard COCO JSON annotation file
  - kitti: YOLO-format .txt label files, as distributed by Ultralytics
           (this is the format you get from the Ultralytics KITTI zip)
           Each line: class_id  x_center  y_center  width  height
           (all values normalized 0–1 relative to image dimensions)

           ⚠️  Note: this is NOT the original raw KITTI format.
           Ultralytics pre-converts KITTI to YOLO format in their zips.
           Check the data.yaml in the zip to see which class_id maps to
           which name (e.g. 0=Car, 1=Van, 2=Truck, 7=Pedestrian, etc.)
           then pass the matching IDs via --human_labels.

Usage examples:
  # Filter COCO, keeping "person" class only
  python filter_to_person.py \\
      --format coco \\
      --ann_file /data/coco/annotations/instances_val2017.json \\
      --img_dir  /data/coco/val2017 \\
      --output   /data/filtered/coco_person.json \\
      --img_out_dir /data/filtered/images \\
      --human_labels person

  # Filter KITTI (Ultralytics YOLO format), keeping class IDs 3, 4, and 5
  # (check your data.yaml to confirm which IDs correspond to pedestrians)
  python filter_to_person.py \\
      --format kitti \\
      --label_dir   /data/kitti/labels/train \\
      --img_dir     /data/kitti/images/train \\
      --output      /data/filtered/kitti_person.json \\
      --img_out_dir /data/filtered/images
      --human_labels 3 4 5 \\

  # Add --img_out_dir /data/filtered/images and --copy_images (optionally) to also copy (or symlink if no --copy_images) the filtered images to a new folder.
  python filter_to_person.py ... --copy_images --img_out_dir /data/filtered/images
"""

import os
import json
import shutil
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter datasets to human classes and unify under 'person' label."
    )

    # Format selector
    parser.add_argument(
        "--format",
        required=True,
        choices=["coco", "kitti"],
        help="Input dataset format: 'coco' or 'kitti'.",
    )

    # COCO-specific
    parser.add_argument(
        "--ann_file",
        default=None,
        help="[COCO only] Path to the COCO annotation JSON file.",
    )

    # KITTI-specific
    parser.add_argument(
        "--label_dir",
        default=None,
        help="[KITTI only] Directory containing per-image KITTI .txt label files.",
    )

    # Shared
    parser.add_argument(
        "--img_dir", required=True, help="Directory containing the dataset images."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output filtered COCO JSON file.",
    )
    parser.add_argument(
        "--human_labels",
        nargs="+",
        required=True,
        help=(
            "For COCO  : class *names* to keep, e.g. --human_labels person. "
            "For KITTI : class *IDs* (integers) to keep, e.g. --human_labels 0 2. "
            "  Check data.yaml in the Ultralytics zip for the ID-to-name mapping. "
            "All matched classes are merged into a single 'person' label."
        ),
    )

    # Image handling mode
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Physically copy images instead of creating symbolic links (default: symlink).",
    )

    parser.add_argument(
        "--img_out_dir",
        default=None,
        help="Destination folder for filtered images (required if copying or linking).",
    )

    args = parser.parse_args()

    # Validate format-specific required args
    if args.format == "coco" and not args.ann_file:
        parser.error("--ann_file is required for COCO format.")
    if args.format == "kitti" and not args.label_dir:
        parser.error("--label_dir is required for KITTI format.")
    if args.img_out_dir is None and args.copy_images:
        parser.error("--img_out_dir is required when --copy_images is set.")

    return args


# ---------------------------------------------------------------------------
# Shared output structure
# ---------------------------------------------------------------------------


def make_coco_output():
    """Initialize an empty COCO-format annotation dictionary."""
    return {
        "info": {"description": "Filtered dataset with person class only"},
        "licenses": [],
        # Single category: person (id=1)
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": [],
        "annotations": [],
    }


# ---------------------------------------------------------------------------
# COCO parser
# ---------------------------------------------------------------------------


def filter_coco(ann_file: str, human_labels: list[str]) -> dict:
    """
    Read a COCO JSON file and return a filtered COCO dict containing
    only images/annotations that belong to the specified human classes.

    Args:
        ann_file:     Path to the original COCO JSON.
        human_labels: List of COCO category names to keep (e.g. ["person"]).

    Returns:
        A new COCO-format dict with a single 'person' category.
    """
    print(f"[COCO] Loading annotations from: {ann_file}")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # --- Step 1: Find the category IDs that match the requested human labels ---
    human_cat_ids = {
        cat["id"] for cat in coco["categories"] if cat["name"] in human_labels
    }
    matched_names = [c["name"] for c in coco["categories"] if c["id"] in human_cat_ids]
    print(f"[COCO] Matched categories: {matched_names} → will be merged into 'person'")

    if not human_cat_ids:
        raise ValueError(
            f"None of the requested labels {human_labels} were found in the dataset. "
            f"Available: {[c['name'] for c in coco['categories']]}"
        )

    # --- Step 2: Filter annotations that belong to those categories ---
    filtered_anns = [
        ann for ann in coco["annotations"] if ann["category_id"] in human_cat_ids
    ]
    print(f"[COCO] Kept {len(filtered_anns)} / {len(coco['annotations'])} annotations.")

    # --- Step 3: Keep only images that have at least one matching annotation ---
    valid_image_ids = {ann["image_id"] for ann in filtered_anns}
    filtered_imgs = [img for img in coco["images"] if img["id"] in valid_image_ids]
    print(f"[COCO] Kept {len(filtered_imgs)} / {len(coco['images'])} images.")

    # --- Step 4: Build the output COCO dict ---
    output = make_coco_output()
    output["images"] = filtered_imgs

    # Remap all annotations to category_id = 1 ("person")
    for ann in filtered_anns:
        new_ann = dict(ann)  # shallow copy to avoid mutating the original
        new_ann["category_id"] = 1  # unified "person" label
        output["annotations"].append(new_ann)

    return output


# ---------------------------------------------------------------------------
# KITTI / Ultralytics YOLO-format parser
# ---------------------------------------------------------------------------
# The KITTI dataset distributed by Ultralytics (kitti.zip) is already
# converted to YOLO format. Each label file is a .txt with one row per object:
#
#   class_id  x_center  y_center  width  height
#
# All bbox values are normalized (0–1) relative to the image dimensions.
# class_id is an integer index into the class list defined in data.yaml.
#
# ⚠️  Always open data.yaml first to confirm which integer maps to which class.
# Example KITTI YOLO class mapping (may vary):
#   0: Car  1: Van  2: Truck  3: Pedestrian  4: Person_sitting
#   5: Cyclist  6: Tram  7: Misc
# ---------------------------------------------------------------------------


def _find_image_file(img_dir: str, stem: str) -> str | None:
    """
    Given a label file stem (e.g. '000042'), find the matching image file
    by trying common extensions. Returns the filename or None if not found.
    """
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        candidate = Path(img_dir) / (stem + ext)
        if candidate.exists():
            return candidate.name
    return None


def _get_image_dimensions(img_path: Path) -> tuple[int, int]:
    """
    Return (width, height) of an image.
    Uses PIL if available; falls back to reading raw file headers.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(img_path) as im:
            return im.size  # (width, height)
    except ImportError:
        pass

    # --- Pure-stdlib fallback: read from PNG/JPEG headers ---
    with open(img_path, "rb") as f:
        header = f.read(24)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG spec: width at bytes 16–19, height at 20–23, big-endian uint32
        import struct

        w = struct.unpack(">I", header[16:20])[0]
        h = struct.unpack(">I", header[20:24])[0]
        return w, h

    if header[:2] == b"\xff\xd8":
        # JPEG: scan markers until we hit a Start-Of-Frame (SOF) segment
        import struct

        with open(img_path, "rb") as f:
            f.read(2)  # skip the SOI (Start Of Image) marker
            while True:
                marker, length = struct.unpack(">HH", f.read(4))
                # SOF0, SOF1, SOF2 all contain image dimensions
                if marker in (0xFFC0, 0xFFC1, 0xFFC2):
                    f.read(1)  # skip precision byte
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                f.read(length - 2)  # skip to next marker

    raise ValueError(f"Cannot read dimensions — unsupported image format: {img_path}")


def filter_kitti(label_dir: str, img_dir: str, human_labels: list[str]) -> dict:
    """
    Parse KITTI labels in Ultralytics YOLO format and return a filtered
    COCO dict with only person annotations.

    YOLO label format (one row per object, values space-separated):
        class_id  x_center  y_center  bbox_width  bbox_height
        (bbox values are normalized 0–1 relative to image W/H)

    The bbox values must be de-normalized using the actual image dimensions
    and then converted to COCO format [x_top_left, y_top_left, w, h].

    Args:
        label_dir:    Directory of YOLO .txt label files.
        img_dir:      Directory of corresponding images.
        human_labels: List of class ID *strings* to keep, e.g. ["0", "3"].
                      These are matched against the first column of each row.
                      (Pass as strings because argparse collects them as strings.)

    Returns:
        A COCO-format dict with a single 'person' category.
    """
    print(f"[KITTI/YOLO] Loading labels from: {label_dir}")

    # Store human class IDs as a set of ints for fast lookup
    human_ids = set(int(lbl) for lbl in human_labels)
    print(f"[KITTI/YOLO] Keeping class IDs: {human_ids} → will be merged into 'person'")

    output = make_coco_output()
    image_id = 0
    ann_id = 0
    total_anns = 0
    kept_anns = 0

    label_files = sorted(Path(label_dir).glob("*.txt"))
    print(f"[KITTI/YOLO] Found {len(label_files)} label files.")

    for label_path in label_files:
        stem = label_path.stem  # e.g. "000042"

        # --- Parse all rows in this label file ---
        person_anns_for_image = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])
                total_anns += 1

                # Skip classes that are not in our human list
                if class_id not in human_ids:
                    continue

                # YOLO bbox: normalized x_center, y_center, width, height
                x_c_norm = float(parts[1])
                y_c_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Store normalized values for now; we'll convert once we have
                # the image dimensions (read below)
                person_anns_for_image.append((x_c_norm, y_c_norm, w_norm, h_norm))
                kept_anns += 1

        # Skip images with no person annotations
        if not person_anns_for_image:
            continue

        # --- Resolve image file ---
        img_filename = _find_image_file(img_dir, stem)
        if img_filename is None:
            print(f"  [WARN] No image file found for label '{stem}', skipping.")
            continue

        # --- Get image dimensions to de-normalize the bbox coordinates ---
        img_path = Path(img_dir) / img_filename
        img_w, img_h = _get_image_dimensions(img_path)

        # --- Add image entry ---
        image_id += 1
        output["images"].append(
            {"id": image_id, "file_name": img_filename, "width": img_w, "height": img_h}
        )

        # --- Convert YOLO bbox → COCO bbox and add annotation entries ---
        for x_c_norm, y_c_norm, w_norm, h_norm in person_anns_for_image:

            # De-normalize: multiply by actual image dimensions
            w_abs = w_norm * img_w
            h_abs = h_norm * img_h
            x_c = x_c_norm * img_w
            y_c = y_c_norm * img_h

            # COCO bbox uses top-left corner [x, y, width, height]
            x_tl = x_c - w_abs / 2
            y_tl = y_c - h_abs / 2

            ann_id += 1
            output["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # unified "person"
                    "bbox": [x_tl, y_tl, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0,
                }
            )

    print(
        f"[KITTI/YOLO] Kept {kept_anns} / {total_anns} annotations across "
        f"{len(output['images'])} images."
    )
    return output


# ---------------------------------------------------------------------------
# Image copying (optional)
# ---------------------------------------------------------------------------


def export_filtered_images(
    output_coco: dict, img_dir: str, img_out_dir: str, copy_mode: bool = False
):
    """
    Export filtered images either by:
        - Creating symbolic links (default)
        - Physically copying files if copy_mode=True
    """

    os.makedirs(img_out_dir, exist_ok=True)
    filenames = {img["file_name"] for img in output_coco["images"]}

    mode = "COPY" if copy_mode else "SYMLINK"
    print(f"\n[{mode}] Exporting {len(filenames)} images to: {img_out_dir}")

    for fname in filenames:
        src = Path(img_dir).resolve() / fname
        dst = Path(img_out_dir) / fname

        if not src.exists():
            print(f"  [WARN] Image not found, skipping: {src}")
            continue

        if dst.exists():
            continue  # avoid overwriting

        try:
            if copy_mode:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
        except OSError as e:
            print(f"  [ERROR] Failed for {fname}: {e}")

    print(f"[{mode}] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- Run the appropriate format-specific filter ---
    if args.format == "coco":
        output = filter_coco(ann_file=args.ann_file, human_labels=args.human_labels)
    elif args.format == "kitti":
        output = filter_kitti(
            label_dir=args.label_dir,
            img_dir=args.img_dir,
            human_labels=args.human_labels,
        )

    # --- Save the filtered COCO JSON ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OUTPUT] Saved filtered COCO JSON → {out_path}")
    print(
        f"         {len(output['images'])} images, {len(output['annotations'])} annotations."
    )

    # --- Optionally copy filtered images ---
    if args.img_out_dir:
        export_filtered_images(
            output_coco=output,
            img_dir=args.img_dir,
            img_out_dir=args.img_out_dir,
            copy_mode=args.copy_images,
        )


if __name__ == "__main__":
    main()
