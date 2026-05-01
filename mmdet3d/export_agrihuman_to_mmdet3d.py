#!/usr/bin/env python3
"""
Export the Agri-Human dataset into an MMDetection3D-friendly LiDAR dataset.

This script consumes the existing:
  - labelled_dataset/manifest_samples.tsv
  - labelled_dataset/splits/default/{train,val,test}.txt
  - *_label/annotations/lidar_ann.json

It writes:
  <out>/
    points/000000.bin
    labels/000000.txt
    ImageSets/{train,val,test,trainval}.txt
    infos/agri_person_infos_{train,val,test,trainval}.pkl
    sample_index.tsv
    export_summary.json

The output annotation PKL files follow the `Det3DDataset` / MMEngine format:
top-level `metainfo` plus `data_list`.

Important assumptions for this dataset:
  - LiDAR boxes are stored in LiDAR coordinates.
  - LiDAR boxes are bottom-centered, matching common outdoor LiDAR datasets.
  - BoundingBoxes in lidar_ann.json use one of:
      [x, y, z, dx, dy, dz, yaw]
      [x, y, z, dx, dy, dz, roll, pitch, yaw]
  - Yaw is usually stored in degrees in this dataset. The exporter auto-detects
    degrees vs radians and always writes radians wrapped to [-pi, pi).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CLASSES = ("person",)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    session_id: str
    lidar_path: Path
    lidar_ann_path: Path
    anchor_ts: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Agri-Human LiDAR data to an MMDetection3D-ready dataset."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset root, e.g. D:\\AOC\\datasets\\agri-human-sensing",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the exported MMDetection3D dataset.",
    )
    parser.add_argument(
        "--labelled_subdir",
        default="labelled_dataset",
        help="Relative folder under dataset_root containing sessions, manifest, and splits.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional override for manifest_samples.tsv. Defaults to <dataset_root>/<labelled_subdir>/manifest_samples.tsv",
    )
    parser.add_argument(
        "--splits_root",
        default=None,
        help="Optional override for the directory that contains splits/default/*.txt. Defaults to <dataset_root>/<labelled_subdir>",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Output classes. For your use-case keep the default single class: person",
    )
    parser.add_argument(
        "--merge_all_to",
        default="person",
        help="Merge every source class into this class name. Default: person",
    )
    parser.add_argument(
        "--human_prefix",
        default="human",
        help="Source labels starting with this prefix are treated as human classes.",
    )
    parser.add_argument(
        "--yaw_unit",
        choices=["auto", "deg", "rad"],
        default="auto",
        help="How to interpret yaw values from lidar_ann.json.",
    )
    parser.add_argument(
        "--rgb_as_intensity",
        action="store_true",
        help="If PCD has rgb but no intensity, derive a grayscale intensity in [0,1]. Default is zeros.",
    )
    parser.add_argument(
        "--drop_empty_samples",
        action="store_true",
        help="Drop frames that end up with no person boxes after class merging.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of exported samples for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .bin/.txt/.pkl outputs.",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path, labelled_root: Path) -> Dict[str, ManifestRow]:
    rows: Dict[str, ManifestRow] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            sample_id = (raw.get("sample_id") or "").strip()
            session_id = (raw.get("session_id") or "").strip()
            lidar_rel = (raw.get("lidar_path") or "").strip()
            ann_rel = (raw.get("lidar_ann_path") or "").strip()
            if not sample_id or not session_id or not lidar_rel:
                continue
            rows[sample_id] = ManifestRow(
                sample_id=sample_id,
                session_id=session_id,
                lidar_path=labelled_root / Path(lidar_rel),
                lidar_ann_path=(labelled_root / Path(ann_rel)) if ann_rel else Path(),
                anchor_ts=(raw.get("anchor_ts") or "").strip(),
            )
    return rows


def read_split_ids(splits_root: Path) -> Dict[str, List[str]]:
    split_ids: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        split_path = splits_root / "splits" / "default" / f"{split_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            sample_id = None
            for token in parts:
                if "_label_" in token and token.split("_label_")[-1].isdigit():
                    sample_id = token
                    break
            if sample_id:
                split_ids[split_name].append(sample_id)
    return split_ids


def map_source_class(raw_name: str, merge_all_to: str, human_prefix: str) -> Optional[str]:
    name = (raw_name or "").strip()
    if not name:
        return None
    lower = name.lower()
    if lower.startswith(human_prefix.lower()):
        return merge_all_to
    return merge_all_to


def infer_yaw_unit(yaw_value: float, requested: str) -> str:
    if requested != "auto":
        return requested
    if abs(yaw_value) > (2.0 * math.pi + 1e-3):
        return "deg"
    return "rad"


def parse_bbox_3d(values: Sequence[float], yaw_unit: str) -> Tuple[List[float], Dict[str, float]]:
    if len(values) == 7:
        x, y, z, dx, dy, dz, yaw_raw = map(float, values)
        roll = 0.0
        pitch = 0.0
    elif len(values) >= 9:
        x, y, z, dx, dy, dz, roll, pitch, yaw_raw = map(float, values[:9])
    else:
        raise ValueError(f"Unsupported lidar bounding box length: {len(values)}")

    unit = infer_yaw_unit(yaw_raw, yaw_unit)
    yaw = math.radians(yaw_raw) if unit == "deg" else yaw_raw
    yaw = wrap_to_pi(yaw)

    bbox = [x, y, z, dx, dy, dz, yaw]
    extras = {
        "roll": roll,
        "pitch": pitch,
        "yaw_raw": yaw_raw,
        "yaw_unit": 1.0 if unit == "deg" else 0.0,
    }
    return bbox, extras


def load_lidar_annotations(
    ann_path: Path,
    classes: Sequence[str],
    merge_all_to: str,
    human_prefix: str,
    yaw_unit: str,
) -> Tuple[Dict[str, List[dict]], Dict[str, float]]:
    if not ann_path.exists():
        return {}, {"frames": 0, "objects": 0, "nonzero_roll": 0, "nonzero_pitch": 0, "deg_yaw": 0, "rad_yaw": 0}

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    out: Dict[str, List[dict]] = {}
    stats = {
        "frames": 0,
        "objects": 0,
        "nonzero_roll": 0,
        "nonzero_pitch": 0,
        "deg_yaw": 0,
        "rad_yaw": 0,
    }

    frames: Iterable[dict]
    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        frames = data["frames"]
    else:
        frames = []

    for frame in frames:
        file_name = (frame.get("File") or frame.get("file") or "").strip()
        if not file_name:
            continue
        stem = Path(file_name).stem.lower()
        stats["frames"] += 1
        instances: List[dict] = []
        for raw_obj in frame.get("Labels", []) or []:
            mapped_class = map_source_class(
                raw_name=str(raw_obj.get("Class") or raw_obj.get("class") or ""),
                merge_all_to=merge_all_to,
                human_prefix=human_prefix,
            )
            if mapped_class is None or mapped_class not in class_to_id:
                continue

            bbox_values = raw_obj.get("BoundingBoxes") or raw_obj.get("bbox") or raw_obj.get("box")
            if not isinstance(bbox_values, list):
                continue
            try:
                bbox_3d, extras = parse_bbox_3d(bbox_values, yaw_unit=yaw_unit)
            except ValueError:
                continue

            if bbox_3d[3] <= 0 or bbox_3d[4] <= 0 or bbox_3d[5] <= 0:
                continue

            instances.append(
                {
                    "bbox_3d": bbox_3d,
                    "bbox_label_3d": class_to_id[mapped_class],
                    "bbox_label": class_to_id[mapped_class],
                }
            )
            stats["objects"] += 1
            if abs(extras["roll"]) > 1e-6:
                stats["nonzero_roll"] += 1
            if abs(extras["pitch"]) > 1e-6:
                stats["nonzero_pitch"] += 1
            if extras["yaw_unit"] == 1.0:
                stats["deg_yaw"] += 1
            else:
                stats["rad_yaw"] += 1

        out[stem] = instances

    return out, stats


def _pcd_numpy_dtype(fields: Sequence[str], sizes: Sequence[int], types: Sequence[str], counts: Sequence[int]) -> np.dtype:
    names: List[str] = []
    formats: List[np.dtype] = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        if typ == "F":
            base = {4: np.float32, 8: np.float64}.get(size)
        elif typ == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        elif typ == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        else:
            base = None
        if base is None:
            raise ValueError(f"Unsupported PCD field type/size: {typ}{size}")
        if count == 1:
            names.append(field)
            formats.append(base)
        else:
            for idx in range(count):
                names.append(f"{field}_{idx}")
                formats.append(base)
    return np.dtype(list(zip(names, formats)))


def read_pcd_xyzi(pcd_path: Path, rgb_as_intensity: bool) -> np.ndarray:
    with pcd_path.open("rb") as handle:
        header_lines: List[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PCD header: {pcd_path}")
            line_text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(line_text)
            if line_text.upper().startswith("DATA "):
                data_mode = line_text.split(None, 1)[1].strip().lower()
                break

        header: Dict[str, str] = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0].upper()] = " ".join(parts[1:])

        fields = header.get("FIELDS", "").split()
        sizes = [int(x) for x in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts = [int(x) for x in header.get("COUNT", " ".join("1" for _ in fields)).split()]
        width = int(header.get("WIDTH", "0"))
        height = int(header.get("HEIGHT", "1"))
        points = int(header.get("POINTS", str(width * height)))

        if not fields or len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
            raise ValueError(f"Malformed PCD header: {pcd_path}")

        if data_mode == "binary_compressed":
            raise ValueError(f"binary_compressed PCD is not supported: {pcd_path}")

        dtype = _pcd_numpy_dtype(fields, sizes, types, counts)

        if data_mode == "ascii":
            arr = np.loadtxt(handle, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] != len(dtype.names or []):
                raise ValueError(f"ASCII PCD column mismatch: {pcd_path}")
            structured = np.zeros(arr.shape[0], dtype=dtype)
            for idx, name in enumerate(dtype.names or []):
                structured[name] = arr[:, idx]
            raw = structured
        elif data_mode == "binary":
            raw = np.fromfile(handle, dtype=dtype, count=points)
        else:
            raise ValueError(f"Unsupported PCD DATA mode '{data_mode}' in {pcd_path}")

    if "x" not in raw.dtype.names or "y" not in raw.dtype.names or "z" not in raw.dtype.names:
        raise ValueError(f"PCD must contain x,y,z fields: {pcd_path}")

    xyz = np.column_stack(
        [
            raw["x"].astype(np.float32, copy=False),
            raw["y"].astype(np.float32, copy=False),
            raw["z"].astype(np.float32, copy=False),
        ]
    )

    if "intensity" in raw.dtype.names:
        intensity = raw["intensity"].astype(np.float32, copy=False)
    elif "reflectivity" in raw.dtype.names:
        intensity = raw["reflectivity"].astype(np.float32, copy=False)
    elif "rgb" in raw.dtype.names and rgb_as_intensity:
        rgb_u32 = raw["rgb"].view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32)
        g = ((rgb_u32 >> 8) & 0xFF).astype(np.float32)
        b = (rgb_u32 & 0xFF).astype(np.float32)
        intensity = ((0.299 * r + 0.587 * g + 0.114 * b) / 255.0).astype(np.float32)
    else:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)

    return np.column_stack([xyz, intensity]).astype(np.float32, copy=False)


def write_pickle(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def build_info_entry(
    sample_idx: str,
    point_rel_path: str,
    num_pts_feats: int,
    instances: List[dict],
    timestamp_ns: Optional[int],
) -> dict:
    entry = {
        "sample_idx": sample_idx,
        "lidar_points": {
            "lidar_path": point_rel_path,
            "num_pts_feats": num_pts_feats,
        },
        "instances": instances,
    }
    if timestamp_ns is not None:
        entry["timestamp"] = timestamp_ns
    return entry


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    labelled_root = dataset_root / args.labelled_subdir
    manifest_path = Path(args.manifest).resolve() if args.manifest else labelled_root / "manifest_samples.tsv"
    splits_root = Path(args.splits_root).resolve() if args.splits_root else labelled_root
    out_root = Path(args.out).resolve()

    ensure_dir(out_root)
    ensure_dir(out_root / "points")
    ensure_dir(out_root / "labels")
    ensure_dir(out_root / "ImageSets")
    ensure_dir(out_root / "infos")

    manifest = read_manifest(manifest_path, labelled_root=labelled_root)
    split_ids = read_split_ids(splits_root=splits_root)
    split_lookup = {sample_id: split_name for split_name, ids in split_ids.items() for sample_id in ids}

    ordered_sample_ids: List[str] = []
    for split_name in SPLIT_NAMES:
        for sample_id in split_ids[split_name]:
            if sample_id in manifest:
                ordered_sample_ids.append(sample_id)

    if args.max_samples is not None:
        ordered_sample_ids = ordered_sample_ids[: args.max_samples]

    classes = tuple(args.classes)
    if args.merge_all_to not in classes:
        raise SystemExit(f"--merge_all_to '{args.merge_all_to}' must be present in --classes")
    ann_cache: Dict[Path, Tuple[Dict[str, List[dict]], Dict[str, float]]] = {}

    info_entries: Dict[str, List[dict]] = {name: [] for name in SPLIT_NAMES}
    split_frame_ids: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    sample_index_rows: List[dict] = []

    class_counts = {name: 0 for name in classes}
    total_instances = 0
    skipped_missing_lidar = 0
    skipped_missing_ann = 0
    exported = 0
    roll_nonzero_total = 0
    pitch_nonzero_total = 0
    yaw_deg_total = 0
    yaw_rad_total = 0

    global_point_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_point_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    global_box_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_box_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    global_box_size_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_box_size_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    sampled_points_xyz: List[np.ndarray] = []

    for frame_idx, sample_id in enumerate(ordered_sample_ids):
        row = manifest.get(sample_id)
        if row is None:
            continue

        split_name = split_lookup.get(sample_id)
        if split_name not in SPLIT_NAMES:
            continue

        if not row.lidar_path.exists():
            skipped_missing_lidar += 1
            continue

        if row.lidar_ann_path and row.lidar_ann_path not in ann_cache:
            ann_cache[row.lidar_ann_path] = load_lidar_annotations(
                ann_path=row.lidar_ann_path,
                classes=classes,
                merge_all_to=args.merge_all_to,
                human_prefix=args.human_prefix,
                yaw_unit=args.yaw_unit,
            )
            stats = ann_cache[row.lidar_ann_path][1]
            roll_nonzero_total += int(stats["nonzero_roll"])
            pitch_nonzero_total += int(stats["nonzero_pitch"])
            yaw_deg_total += int(stats["deg_yaw"])
            yaw_rad_total += int(stats["rad_yaw"])

        ann_index = ann_cache.get(row.lidar_ann_path, ({}, {}))[0]
        lidar_stem = row.lidar_path.stem.lower()
        instances = list(ann_index.get(lidar_stem, []))

        if not instances and args.drop_empty_samples:
            skipped_missing_ann += 1
            continue

        point_cloud = read_pcd_xyzi(row.lidar_path, rgb_as_intensity=args.rgb_as_intensity)
        if point_cloud.size:
            global_point_min = np.minimum(global_point_min, point_cloud[:, :3].min(axis=0))
            global_point_max = np.maximum(global_point_max, point_cloud[:, :3].max(axis=0))
            nonzero_mask = np.linalg.norm(point_cloud[:, :3], axis=1) > 1e-6
            sampled = point_cloud[nonzero_mask, :3]
            if sampled.size:
                stride = max(1, sampled.shape[0] // 512)
                sampled_points_xyz.append(sampled[::stride])

        for inst in instances:
            bbox = np.array(inst["bbox_3d"], dtype=np.float64)
            global_box_min = np.minimum(global_box_min, bbox[:3])
            global_box_max = np.maximum(global_box_max, bbox[:3])
            global_box_size_min = np.minimum(global_box_size_min, bbox[3:6])
            global_box_size_max = np.maximum(global_box_size_max, bbox[3:6])
            class_name = classes[int(inst["bbox_label_3d"])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_instances += 1

        frame_id = f"{frame_idx:06d}"
        point_rel_path = f"points/{frame_id}.bin"
        label_rel_path = f"labels/{frame_id}.txt"

        point_out_path = out_root / point_rel_path
        label_out_path = out_root / label_rel_path

        if args.overwrite or not point_out_path.exists():
            point_cloud.astype(np.float32, copy=False).tofile(point_out_path)

        label_lines = []
        for inst in instances:
            x, y, z, dx, dy, dz, yaw = inst["bbox_3d"]
            class_name = classes[int(inst["bbox_label_3d"])]
            label_lines.append(
                f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {yaw:.6f} {class_name}"
            )
        write_text(label_out_path, "\n".join(label_lines) + ("\n" if label_lines else ""), overwrite=args.overwrite)

        timestamp_ns = None
        try:
            timestamp_ns = int(sample_id.rsplit("_label_", 1)[1])
        except Exception:
            timestamp_ns = None

        info_entry = build_info_entry(
            sample_idx=frame_id,
            point_rel_path=point_rel_path,
            num_pts_feats=4,
            instances=instances,
            timestamp_ns=timestamp_ns,
        )
        info_entries[split_name].append(info_entry)
        split_frame_ids[split_name].append(frame_id)
        sample_index_rows.append(
            {
                "frame_id": frame_id,
                "split": split_name,
                "sample_id": row.sample_id,
                "session_id": row.session_id,
                "timestamp_ns": timestamp_ns if timestamp_ns is not None else "",
                "lidar_src": str(row.lidar_path),
                "label_src": str(row.lidar_ann_path) if row.lidar_ann_path else "",
                "num_instances": len(instances),
            }
        )
        exported += 1

    split_frame_ids["trainval"] = split_frame_ids["train"] + split_frame_ids["val"]
    trainval_entries = info_entries["train"] + info_entries["val"]

    metainfo = {
        "classes": classes,
        "categories": {name: idx for idx, name in enumerate(classes)},
        "dataset": "agrihuman_person",
        "info_version": "1.0",
    }

    for split_name in SPLIT_NAMES:
        write_text(
            out_root / "ImageSets" / f"{split_name}.txt",
            "\n".join(split_frame_ids[split_name]) + ("\n" if split_frame_ids[split_name] else ""),
            overwrite=args.overwrite,
        )
        write_pickle(
            out_root / "infos" / f"agri_person_infos_{split_name}.pkl",
            {"metainfo": metainfo, "data_list": info_entries[split_name]},
            overwrite=args.overwrite,
        )

    write_text(
        out_root / "ImageSets" / "trainval.txt",
        "\n".join(split_frame_ids["trainval"]) + ("\n" if split_frame_ids["trainval"] else ""),
        overwrite=args.overwrite,
    )
    write_pickle(
        out_root / "infos" / "agri_person_infos_trainval.pkl",
        {"metainfo": metainfo, "data_list": trainval_entries},
        overwrite=args.overwrite,
    )

    sample_index_path = out_root / "sample_index.tsv"
    sample_index_headers = [
        "frame_id",
        "split",
        "sample_id",
        "session_id",
        "timestamp_ns",
        "lidar_src",
        "label_src",
        "num_instances",
    ]
    with sample_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_index_headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(sample_index_rows)

    point_range = None
    if np.isfinite(global_point_min).all() and np.isfinite(global_point_max).all():
        point_range = [
            float(global_point_min[0]),
            float(global_point_min[1]),
            float(global_point_min[2]),
            float(global_point_max[0]),
            float(global_point_max[1]),
            float(global_point_max[2]),
        ]

    suggested_point_cloud_range = None
    if point_range is not None:
        suggested_point_cloud_range = [
            math.floor(point_range[0]),
            math.floor(point_range[1]),
            math.floor(point_range[2]),
            math.ceil(point_range[3]),
            math.ceil(point_range[4]),
            math.ceil(point_range[5]),
        ]

    robust_point_percentiles = None
    robust_suggested_point_cloud_range = None
    if sampled_points_xyz:
        sampled_xyz = np.concatenate(sampled_points_xyz, axis=0)
        low = np.percentile(sampled_xyz, 0.5, axis=0)
        high = np.percentile(sampled_xyz, 99.5, axis=0)
        robust_point_percentiles = {
            "p0_5": [float(x) for x in low],
            "p99_5": [float(x) for x in high],
        }
        robust_suggested_point_cloud_range = [
            math.floor(float(low[0])),
            math.floor(float(low[1])),
            math.floor(float(low[2])),
            math.ceil(float(high[0])),
            math.ceil(float(high[1])),
            math.ceil(float(high[2])),
        ]

    summary = {
        "dataset_root": str(dataset_root),
        "labelled_root": str(labelled_root),
        "output_root": str(out_root),
        "exported_samples": exported,
        "split_counts": {name: len(info_entries[name]) for name in SPLIT_NAMES},
        "trainval_count": len(trainval_entries),
        "classes": list(classes),
        "class_instance_counts": class_counts,
        "total_instances": total_instances,
        "skipped_missing_lidar": skipped_missing_lidar,
        "skipped_missing_annotation_match": skipped_missing_ann,
        "point_cloud_xyz_minmax": point_range,
        "suggested_point_cloud_range": suggested_point_cloud_range,
        "robust_point_cloud_percentiles": robust_point_percentiles,
        "robust_suggested_point_cloud_range": robust_suggested_point_cloud_range,
        "bbox_center_min": global_box_min.tolist() if np.isfinite(global_box_min).all() else None,
        "bbox_center_max": global_box_max.tolist() if np.isfinite(global_box_max).all() else None,
        "bbox_size_min": global_box_size_min.tolist() if np.isfinite(global_box_size_min).all() else None,
        "bbox_size_max": global_box_size_max.tolist() if np.isfinite(global_box_size_max).all() else None,
        "yaw_source_counts": {
            "deg": yaw_deg_total,
            "rad": yaw_rad_total,
        },
        "nonzero_roll_count": roll_nonzero_total,
        "nonzero_pitch_count": pitch_nonzero_total,
        "assumptions": {
            "boxes_are_in_lidar_coordinates": True,
            "boxes_are_bottom_centered": True,
            "output_point_features": ["x", "y", "z", "intensity"],
            "intensity_fallback": "rgb_grayscale" if args.rgb_as_intensity else "zeros",
            "all_source_classes_merged_to": args.merge_all_to,
        },
    }
    write_text(
        out_root / "export_summary.json",
        json.dumps(summary, indent=2),
        overwrite=True,
    )

    print("[done]")
    print(f"  output: {out_root}")
    print(f"  samples: {exported}")
    print(
        "  splits: "
        + ", ".join(f"{name}={len(info_entries[name])}" for name in SPLIT_NAMES)
    )
    print(f"  instances: {total_instances}")
    print(f"  classes: {list(classes)}")
    if robust_suggested_point_cloud_range is not None:
        print(f"  robust suggested point_cloud_range: {robust_suggested_point_cloud_range}")
    elif suggested_point_cloud_range is not None:
        print(f"  suggested point_cloud_range: {suggested_point_cloud_range}")


if __name__ == "__main__":
    main()
