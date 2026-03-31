#!/usr/bin/env python3
"""
YOLO export for:
  A) ONE session folder (e.g. footpath1_..._label/)  [--session]
  B) MANY sessions under a root folder               [--root --split_tag]

It uses:
- <session>/sync.json  (produced by sync_and_match.py)
- <session>/annotations/<camera>_ann.json (one per camera)
- split files at: <splits_root>/splits/default/{train,val,test}.txt
  (supports either "<session> <timestamp_ns>" OR file-path lines)

Output:
  <out>/
    images/{train,val,test}/000000.png
    labels/{train,val,test}/000000.txt
    classes.txt
    data.yaml

Recommended usage (session-safe, no leakage):
- Export by split across sessions:
    python yolo_export_session.py --root <dataset_root> --splits_root <dataset_root> --split_tag train --out yolo_out
    python yolo_export_session.py --root <dataset_root> --splits_root <dataset_root> --split_tag val   --out yolo_out
    python yolo_export_session.py --root <dataset_root> --splits_root <dataset_root> --split_tag test  --out yolo_out

Multi-camera note:
- --anchor_camera supports a comma list (or [a,b,c] form). All cameras are exported.
- --camera_folder and --ann_json can be comma lists too (length 1 or same as --anchor_camera).

Windows note:
- symlinks may be blocked; use --link_mode copy (or keep symlink; it auto-fallbacks to copy)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # pyyaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class SampleKey:
    session: str
    timestamp_ns: int


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    Place src into dst using mode.
    If mode=symlink but symlink fails (common on NTFS/exFAT/Windows), auto-fallback to copy.
    """
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, PermissionError):
        shutil.copy2(src, dst)


def read_image_size(img_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]  # (W,H)
    except Exception:
        try:
            import cv2  # type: ignore
            im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("cv2.imread returned None")
            h, w = im.shape[:2]
            return w, h
        except Exception as e:
            raise RuntimeError(f"Cannot read image size for {img_path}: {e}")


# -----------------------------
# Dataset parsing
# -----------------------------
def load_sync_samples(session_dir: Path) -> List[dict]:
    p = session_dir / "sync.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    return []


def parse_ts_ns(sample: dict) -> int:
    if "timestamp_ns" in sample:
        return int(sample["timestamp_ns"])
    if "timestamp" in sample:
        return int(float(sample["timestamp"]) * 1e9)
    raise ValueError("sync sample missing timestamp_ns/timestamp.")


def load_ann_index(ann_path: Path) -> Dict[str, List[dict]]:
    """
    Index annotations by filename stem and exact filename.
    Supports list items like:
      {"File":"173...png","Labels":[{"Class":"human1","BoundingBoxes":[x,y,w,h]}, ...]}
    """
    if not ann_path.exists():
        return {}
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    idx: Dict[str, List[dict]] = {}

    def add(key: str, labels: List[dict]):
        if not key:
            return
        stem = Path(key).stem.lower()
        idx.setdefault(stem, []).extend(labels or [])
        idx.setdefault(key.lower(), []).extend(labels or [])

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            k = item.get("File") or item.get("file") or item.get("filename") or item.get("name")
            labels = item.get("Labels") or item.get("labels") or item.get("objects") or []
            add(k, labels)
    elif isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            for item in data["frames"]:
                k = item.get("File") or item.get("file") or item.get("filename")
                labels = item.get("Labels") or item.get("labels") or []
                add(k, labels)
        else:
            for k, v in data.items():
                if isinstance(v, list):
                    add(k, v)
                elif isinstance(v, dict) and "Labels" in v:
                    add(k, v["Labels"])
    return idx


def get_labels_for_image(ann_idx: Dict[str, List[dict]], filename: str) -> List[dict]:
    if not filename:
        return []
    k1 = filename.lower()
    k2 = Path(filename).stem.lower()
    return ann_idx.get(k1, ann_idx.get(k2, []))


def xywh_to_yolo(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / W, cy / H, w / W, h / H


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize_class_name(name: str, merge_humans_to_person: bool) -> str:
    """
    If merge_humans_to_person=True:
      human1..human5 (case-insensitive) => person
    """
    n = (name or "").strip()
    if not n:
        return n
    low = n.lower()
    if merge_humans_to_person and low.startswith("human"):
        suffix = low[5:]
        if suffix.isdigit():
            k = int(suffix)
            if 1 <= k <= 5:
                return "person"
    return n


# -----------------------------
# Split parsing (supports your split format)
# -----------------------------

_SAMPLE_ID_RE = re.compile(r"^(?P<sess>.+_label)_(?P<tsns>\d{12,})$")

def load_split_files(splits_root: Path):
    """
    Your split lines look like:
      <lidar_path> <fish_left_path> ... <session_id> <scenario> <sample_id> <anchor_modality> <anchor_ts>

    We parse sample_id:
      <session>_label_<timestamp_ns>
    """
    out = {"train": set(), "val": set(), "test": set()}
    base = splits_root / "splits" / "default"

    for tag in ("train", "val", "test"):
        p = base / f"{tag}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t") if "\t" in line else line.split()

            sess = None
            ts_ns = None

            # Best: parse sample_id token "<session>_label_<timestamp_ns>"
            for tok in parts:
                m = _SAMPLE_ID_RE.match(tok)
                if m:
                    sess = m.group("sess")
                    ts_ns = int(m.group("tsns"))
                    break

            if sess is None or ts_ns is None:
                continue

            out[tag].add(SampleKey(sess, ts_ns))

    return out

# -----------------------------
# Output metadata
# -----------------------------
def write_data_yaml(out_dir: Path, names: List[str]) -> None:
    d = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    p = out_dir / "data.yaml"
    if yaml is None:
        lines = [
            f"path: {d['path']}",
            f"train: {d['train']}",
            f"val: {d['val']}",
            f"test: {d['test']}",
            "names:",
        ]
        for i, n in enumerate(names):
            lines.append(f"  {i}: {n}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


# -----------------------------
# Core export logic
# -----------------------------
def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    s = value.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p]


def _expand_list(items: Optional[List[str]], n: int, name: str) -> List[Optional[str]]:
    if items is None:
        return [None] * n
    if len(items) == 1 and n > 1:
        return [items[0]] * n
    if len(items) != n:
        raise ValueError(f"{name} must have length 1 or {n} (got {len(items)})")
    return list(items)


def export_one_session(
    session_dir: Path,
    out_dir: Path,
    split_sets: Optional[Dict[str, set]],
    only_split: Optional[str],
    anchor_cameras: List[str],
    camera_folders: List[Optional[str]],
    ann_jsons: List[Optional[str]],
    link_mode: str,
    split_ratio: str,
    seed: int,
    class_map_json: Optional[str],
    drop_unknown: bool,
    merge_humans_to_person: bool,
    counters: Dict[str, int],
    class_to_id: Dict[str, int],
) -> None:
    session_name = session_dir.name
    samples = load_sync_samples(session_dir)
    if not samples:
        return

    cam_configs = []
    for cam, folder, ann_override in zip(anchor_cameras, camera_folders, ann_jsons):
        cam_folder = folder or cam
        img_root = session_dir / "sensor_data" / cam_folder
        ann_path = Path(ann_override).resolve() if ann_override else (session_dir / "annotations" / f"{cam}_ann.json")
        ann_idx = load_ann_index(ann_path)
        cam_configs.append((cam, img_root, ann_idx))

    class_map = json.loads(class_map_json) if class_map_json else None
    if class_map:
        class_map = {str(k).lower(): str(v) for k, v in class_map.items()}

    def get_class_id(cls_name: str) -> Optional[int]:
        c = cls_name.strip()
        if not c:
            return None
        if class_map is not None:
            key = c.lower()
            if key in class_map:
                c = class_map[key]
            elif drop_unknown:
                return None
        if c not in class_to_id:
            class_to_id[c] = len(class_to_id)
        return class_to_id[c]

    # Build list of (SampleKey, camera, image_file)
    items: List[Tuple[SampleKey, str, str]] = []
    for s in samples:
        ts = parse_ts_ns(s)
        for cam, _img_root, _ann_idx in cam_configs:
            if s.get("anchor_modality") == cam and s.get("anchor_file"):
                img_file = s["anchor_file"]
            else:
                img_file = (s.get("cameras", {}) or {}).get(cam)
            if not img_file or img_file == "null":
                continue
            items.append((SampleKey(session_name, ts), cam, img_file))

    if not items:
        return

    # Determine split assignment (either from split files or ratio split)
    tag_for: Dict[SampleKey, str] = {}

    if split_sets is not None:
        # use global split files
        for k, _cam, _img in items:
            if k in split_sets["train"]:
                tag_for[k] = "train"
            elif k in split_sets["val"]:
                tag_for[k] = "val"
            elif k in split_sets["test"]:
                tag_for[k] = "test"

        # filter only those found in split lists
        items = [(k, cam, img) for (k, cam, img) in items if k in tag_for]

        # if only_split specified, keep only that
        if only_split:
            items = [(k, cam, img) for (k, cam, img) in items if tag_for.get(k) == only_split]
    else:
        # ratio split INSIDE this one session (not recommended for leakage)
        parts = [p.strip() for p in split_ratio.split(",")]
        if len(parts) != 3:
            raise ValueError("--split must be like 0.8,0.1,0.1")
        tr, va, te = map(float, parts)
        total = tr + va + te
        tr, va, te = tr / total, va / total, te / total

        rng = random.Random(seed)
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        train = items[:n_tr]
        val = items[n_tr:n_tr + n_va]
        test = items[n_tr + n_va:]
        for k, _cam, _ in train: tag_for[k] = "train"
        for k, _cam, _ in val: tag_for[k] = "val"
        for k, _cam, _ in test: tag_for[k] = "test"

    # Prepare output dirs
    for tag in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / tag)
        ensure_dir(out_dir / "labels" / tag)

    # Export
    ann_idx_by_cam = {cam: ann_idx for cam, _img_root, ann_idx in cam_configs}
    img_root_by_cam = {cam: img_root for cam, img_root, _ann_idx in cam_configs}

    for k, cam, img_file in items:
        tag = tag_for.get(k)
        if tag not in ("train", "val", "test"):
            continue

        src_img = img_root_by_cam[cam] / img_file
        if not src_img.exists():
            continue

        idx = counters[tag]
        counters[tag] += 1

        out_img = out_dir / "images" / tag / f"{idx:06d}{src_img.suffix.lower()}"
        out_lab = out_dir / "labels" / tag / f"{idx:06d}.txt"

        symlink_or_copy(src_img, out_img, link_mode)

        W, H = read_image_size(src_img)
        objs = get_labels_for_image(ann_idx_by_cam[cam], img_file)

        yolo_lines: List[str] = []
        for obj in objs:
            cls = obj.get("Class") or obj.get("class") or obj.get("type")
            bb = obj.get("BoundingBoxes") or obj.get("bbox") or obj.get("box")
            if cls is None or bb is None:
                continue
            if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                continue

            x, y, w, h = map(float, bb[:4])
            cx, cy, ww, hh = xywh_to_yolo(x, y, w, h, W, H)
            cx, cy, ww, hh = clamp01(cx), clamp01(cy), clamp01(ww), clamp01(hh)

            cls_norm = normalize_class_name(str(cls), merge_humans_to_person)
            cid = get_class_id(cls_norm)
            if cid is None:
                continue

            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        out_lab.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Either session mode OR root mode
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--session", help="Path to one *_label session folder")
    mx.add_argument("--root", help="Path containing many *_label session folders")

    ap.add_argument("--out", required=True, help="Output YOLO dataset folder")

    ap.add_argument("--anchor_camera", default="cam_zed_rgb",
                    help="Camera modality to export as images (comma list supported)")
    ap.add_argument("--camera_folder", default=None,
                    help="Override sensor_data/<camera> folder name (comma list supported)")
    ap.add_argument("--ann_json", default=None,
                    help="Override annotation json path (session mode only, comma list supported)")

    ap.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink",
                    help="symlink is faster; auto-fallback to copy if symlink not permitted")

    # split handling
    ap.add_argument("--splits_root", default=None,
                    help="If set, use <splits_root>/splits/default/{train,val,test}.txt (recommended)")
    ap.add_argument("--split_tag", choices=["train", "val", "test"], default=None,
                    help="If provided in --root mode, export ONLY this split (recommended).")
    ap.add_argument("--split", default="0.8,0.1,0.1",
                    help="Ratio split (ONLY used if --splits_root is NOT provided)")
    ap.add_argument("--seed", type=int, default=42)

    # class mapping
    ap.add_argument("--class_map", default=None, help="JSON map original->new classes")
    ap.add_argument("--drop_unknown", action="store_true")
    ap.add_argument("--merge_humans_to_person", action="store_true",
                    help="Map human1..human5 -> person automatically")

    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    # Load split sets if provided
    split_sets = None
    if args.splits_root:
        split_sets = load_split_files(Path(args.splits_root).resolve())

    # Counters across whole export run (so filenames are unique)
    counters = {"train": 0, "val": 0, "test": 0}
    class_to_id: Dict[str, int] = {}

    anchor_cameras = _parse_csv_list(args.anchor_camera) or []
    if not anchor_cameras:
        raise ValueError("--anchor_camera must include at least one camera")
    camera_folders = _expand_list(_parse_csv_list(args.camera_folder), len(anchor_cameras), "--camera_folder")
    ann_jsons = _expand_list(_parse_csv_list(args.ann_json), len(anchor_cameras), "--ann_json")

    if args.session:
        session_dir = Path(args.session).resolve()
        export_one_session(
            session_dir=session_dir,
            out_dir=out_dir,
            split_sets=split_sets,
            only_split=None,  # session mode exports whichever split each sample belongs to
            anchor_cameras=anchor_cameras,
            camera_folders=camera_folders,
            ann_jsons=ann_jsons,
            link_mode=args.link_mode,
            split_ratio=args.split,
            seed=args.seed,
            class_map_json=args.class_map,
            drop_unknown=args.drop_unknown,
            merge_humans_to_person=args.merge_humans_to_person,
            counters=counters,
            class_to_id=class_to_id,
        )
    else:
        root = Path(args.root).resolve()
        sessions = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_label")]
        if not sessions:
            raise FileNotFoundError(f"No *_label folders found under: {root}")

        # In root mode: if user gives --split_tag, export ONLY that split (recommended)
        only_split = args.split_tag

        for sess in sorted(sessions):
            export_one_session(
                session_dir=sess,
                out_dir=out_dir,
                split_sets=split_sets,
                only_split=only_split,
                anchor_cameras=anchor_cameras,
                camera_folders=camera_folders,
                ann_jsons=[None] * len(anchor_cameras),  # per-session default: annotations/<camera>_ann.json
                link_mode=args.link_mode,
                split_ratio=args.split,
                seed=args.seed,
                class_map_json=args.class_map,
                drop_unknown=args.drop_unknown,
                merge_humans_to_person=args.merge_humans_to_person,
                counters=counters,
                class_to_id=class_to_id,
            )

    # Write classes + data.yaml
    classes = [None] * len(class_to_id)
    for name, i in class_to_id.items():
        classes[i] = name
    classes = [c for c in classes if c is not None]

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    write_data_yaml(out_dir, classes)

    print("[done]")
    print(f"  out: {out_dir}")
    print(f"  counts: train={counters['train']} val={counters['val']} test={counters['test']}")
    print(f"  classes: {classes}")
    print(f"  data.yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
