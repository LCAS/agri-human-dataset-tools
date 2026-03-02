#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Undistort camera images using intrinsics.json.

Usage:
  python undistort_images.py --root D:\\AOC\\datasets\\agri-human-dataset\\labelled_dataset ^
    --intrinsics D:\\AOC\\datasets\\agri-human-dataset\\calibration\\intrinsics.json

By default, writes sibling folders with suffix "_undistorted", e.g.:
  sensor_data/cam_fish_left        -> sensor_data/cam_fish_left_undistorted
  sensor_data/cam_zed_rgb          -> sensor_data/cam_zed_rgb_undistorted
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def load_intrinsics(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Intrinsics JSON must be an object: {path}")
    return data


def k_from_entry(entry: Dict) -> np.ndarray:
    if "camera_matrix" in entry:
        k = np.array(entry["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        return k
    if "k" in entry:
        k = np.array(entry["k"], dtype=np.float64).reshape(3, 3)
        return k
    raise KeyError("Missing camera_matrix/k in intrinsics entry.")


def d_from_entry(entry: Dict) -> np.ndarray:
    if "distortion_coefficients" in entry:
        d = np.array(entry["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
        return d
    if "d" in entry:
        d = np.array(entry["d"], dtype=np.float64).reshape(-1, 1)
        return d
    raise KeyError("Missing distortion_coefficients/d in intrinsics entry.")


def size_from_entry(entry: Dict) -> Tuple[int, int]:
    if "image_width" in entry and "image_height" in entry:
        return int(entry["image_width"]), int(entry["image_height"])
    if "width" in entry and "height" in entry:
        return int(entry["width"]), int(entry["height"])
    raise KeyError("Missing image width/height in intrinsics entry.")


def estimate_fisheye_newk(
    k: np.ndarray, d: np.ndarray, size: Tuple[int, int], balance: float
) -> Tuple[np.ndarray, bool]:
    w, h = size
    r = np.eye(3, dtype=np.float64)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, (w, h), r, balance=balance)
    if not np.isfinite(new_k).all():
        return k, False
    fx, fy = new_k[0, 0], new_k[1, 1]
    cx, cy = new_k[0, 2], new_k[1, 2]
    if fx < 1.0 or fy < 1.0:
        return k, False
    if (cx < -w) or (cx > 2 * w) or (cy < -h) or (cy > 2 * h):
        return k, False
    return new_k, True


def build_fisheye_maps(
    k: np.ndarray, d: np.ndarray, size: Tuple[int, int], balance: float
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = size
    r = np.eye(3, dtype=np.float64)
    new_k, ok = estimate_fisheye_newk(k, d, (w, h), balance)
    if not ok:
        print("  [warn] degenerate newK from estimate; falling back to K")
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, r, new_k, (w, h), cv2.CV_16SC2)
    return map1, map2


def build_pinhole_newk(k: np.ndarray, d: np.ndarray, size: Tuple[int, int], alpha: float) -> np.ndarray:
    w, h = size
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (w, h), alpha)
    return new_k


def scale_k(k: np.ndarray, base_size: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    base_w, base_h = base_size
    w, h = size
    if (w, h) == (base_w, base_h):
        return k
    sx = w / float(base_w)
    sy = h / float(base_h)
    k2 = k.copy()
    k2[0, 0] *= sx  # fx
    k2[1, 1] *= sy  # fy
    k2[0, 2] *= sx  # cx
    k2[1, 2] *= sy  # cy
    k2[0, 1] *= sx  # skew (if any)
    k2[1, 0] *= sy  # skew (if any)
    return k2


def undistort_dir(
    src_dir: Path,
    dst_dir: Path,
    entry: Dict,
    balance: float,
    alpha: float,
    border_mode: int,
    dry_run: bool,
) -> int:
    model = (entry.get("distortion_model") or "").lower()
    k = k_from_entry(entry)
    d = d_from_entry(entry)
    base_w, base_h = size_from_entry(entry)

    # Cache for maps keyed by image size
    fisheye_maps: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    pinhole_newk: Dict[Tuple[int, int], np.ndarray] = {}

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        if dry_run:
            count += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        size = (w, h)

        # If image size differs from nominal, use actual size for maps.
        k_use = scale_k(k, (base_w, base_h), size)
        if model == "equidistant":
            if size not in fisheye_maps:
                fisheye_maps[size] = build_fisheye_maps(k_use, d, size, balance)
            map1, map2 = fisheye_maps[size]
            undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=border_mode)
        else:
            if size not in pinhole_newk:
                pinhole_newk[size] = build_pinhole_newk(k_use, d, size, alpha)
            new_k = pinhole_newk[size]
            undist = cv2.undistort(img, k_use, d, None, new_k)

        out_path = dst_dir / img_path.name
        cv2.imwrite(str(out_path), undist)
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (e.g., labelled_dataset)")
    ap.add_argument("--intrinsics", required=True, help="path to intrinsics.json")
    ap.add_argument("--output_suffix", default="_undistorted", help="suffix for output folders")
    ap.add_argument("--balance", type=float, default=0.0, help="fisheye balance [0..1], 0 crops, 1 keeps more FOV")
    ap.add_argument("--alpha", type=float, default=0.0, help="pinhole alpha [0..1], 0 crops, 1 keeps more FOV")
    ap.add_argument(
        "--border_mode",
        default="constant",
        choices=["constant", "replicate", "reflect"],
        help="border mode for remap (fisheye). 'replicate' avoids black borders.",
    )
    ap.add_argument("--dry_run", action="store_true", help="report without writing images")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    intr_path = Path(args.intrinsics).resolve()
    intr = load_intrinsics(intr_path)

    # Find camera folders under root by camera key
    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
    }
    border_mode = border_map[args.border_mode]

    for cam_key, entry in intr.items():
        # Skip non-cameras
        if not isinstance(entry, dict):
            continue
        # Locate directories named cam_key under root
        cam_dirs = [p for p in root.rglob(cam_key) if p.is_dir() and p.name == cam_key]
        if not cam_dirs:
            continue
        for src_dir in cam_dirs:
            dst_dir = src_dir.with_name(src_dir.name + args.output_suffix)
            n = undistort_dir(
                src_dir, dst_dir, entry, args.balance, args.alpha, border_mode, args.dry_run
            )
            print(f"{cam_key}: {src_dir} -> {dst_dir}  ({n} images)")


if __name__ == "__main__":
    main()
