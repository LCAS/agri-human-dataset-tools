# YOLO Export Toolkit for Agri-Human Dataset

This document explains how to export the **Agri-Human dataset** into **YOLO format**
for training, validation, and testing.

This repository focuses **only on YOLO export**.
(KITTI export and related tooling live in a separate repository.)

The YOLO exporter is designed to:

- Reuse **sensor synchronisation (`sync.json`)**
- Reuse **global train/val/test splits** (session-safe, no leakage)
- Avoid splitting frames from the same recording session
- Support **single- or multi-camera YOLO (RGB)**
- Work on **Linux and Windows**

---

## 0. Recommended Repository Layout

It is recommended to separate **code** and **data**:

```
agri-human-yolo-tools/
├── sync_and_match.py
├── build_manifest_and_splits.py
├── yolo_export_session.py
├── README.md
└── dataset/
    ├── footpath1_..._label/
    ├── footpath2_..._label/
    ├── manifest_samples.tsv
    └── splits/
        └── default/
            ├── train.txt (created by build_manifest_and_splits.py)
            ├── val.txt (created by build_manifest_and_splits.py)
            └── test.txt (created by build_manifest_and_splits.py)
```

All scripts can be executed from any directory as long as paths are correct.

---

## 1. What is YOLO Format?

A YOLO dataset has the following structure:

```
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── classes.txt
└── data.yaml
```

Each image has a corresponding label file:

```
labels/train/000123.txt
```

Each line in a YOLO label file:

```
<class_id> <cx> <cy> <w> <h>
```

All values are **normalized to [0, 1]**.

---

## 2. Input Requirements

Before exporting YOLO, your dataset must already contain:

### 2.1 Session folders

Each recording session lives in a folder ending with `_label`:

```
footpath1_..._label/
```

Each session must contain:

```
sensor_data/
annotations/
metadata/
sync.json
```

`sync.json` is produced by:

```bash
python sync_and_match.py <DATASET_ROOT>
```

---

### 2.2 Global manifest and splits (REQUIRED)

YOLO export **does not split data inside a session**.

Instead, global splits are generated once and reused to prevent overfitting.

Create them with:

```bash
python build_manifest_and_splits.py --root <DATASET_ROOT>
```

This generates:

```
manifest_samples.tsv
splits/default/train.txt
splits/default/val.txt
splits/default/test.txt
```

---

## 3. Full Workflow (Recommended)

### Step 1 — Synchronise sensors

Linux / macOS:
```bash
python sync_and_match.py <DATASET_ROOT>
```

Windows (PowerShell):
```powershell
python sync_and_match.py "D:\Root\Of\Dataset"
```

---

### Step 2 — Build manifest and splits

```bash
python build_manifest_and_splits.py --root <DATASET_ROOT>
```

---

### Step 3 — Export YOLO (train / val / test)

#### Train
```bash
python yolo_export_session.py \
  --root <DATASET_ROOT> \
  --splits_root <DATASET_ROOT> \
  --split_tag train \
  --out yolo_dataset \
  --anchor_camera cam_zed_rgb \
  --merge_humans_to_person \
  --link_mode copy
```

#### Validation
```bash
python yolo_export_session.py \
  --root <DATASET_ROOT> \
  --splits_root <DATASET_ROOT> \
  --split_tag val \
  --out yolo_dataset \
  --anchor_camera cam_zed_rgb \
  --merge_humans_to_person \
  --link_mode copy
```

#### Test
```bash
python yolo_export_session.py \
  --root <DATASET_ROOT> \
  --splits_root <DATASET_ROOT> \
  --split_tag test \
  --out yolo_dataset \
  --anchor_camera cam_zed_rgb \
  --merge_humans_to_person \
  --link_mode copy
```

Multi-camera export (example):
```bash
python yolo_export_session.py \
  --root <DATASET_ROOT> \
  --splits_root <DATASET_ROOT> \
  --split_tag train \
  --out yolo_dataset \
  --anchor_camera cam_fish_front,cam_fish_left,cam_fish_right \
  --merge_humans_to_person \
  --link_mode copy
```

Notes:
- --anchor_camera accepts a comma list or [a,b,c] form.
- --camera_folder and --ann_json can be comma lists too (length 1 or same length as --anchor_camera).

---

## 4. Related Utilities

In the merged branch, non-export helpers live outside `yolo/`:

- `converters/fieldsafe_rgb_to_yolo.py` converts the FieldSafePedestrian RGB demo dataset into a single-class YOLO dataset.
- `converters/yolo_to_coco.py` converts a YOLO image/label split plus `data.yaml` into COCO JSON.
- `filters/filter_to_person.py` filters COCO or Ultralytics-style YOLO/KITTI labels down to human classes and writes COCO JSON.
- `preprocessing/undistort_dataset_images.py` undistorts camera folders using `intrinsics.json`.

Examples:

```bash
python converters/fieldsafe_rgb_to_yolo.py --dataset_root <FIELDSAFE_ROOT> --out <YOLO_OUT> --copy_images
python converters/yolo_to_coco.py --images_dir <IMAGES_DIR> --labels_dir <LABELS_DIR> --yaml_path <DATA_YAML> --output_path <OUT_JSON>
python filters/filter_to_person.py --format coco --ann_file <ANN_JSON> --img_dir <IMG_DIR> --output <OUT_JSON> --human_labels person
python preprocessing/undistort_dataset_images.py --root <DATASET_ROOT> --intrinsics <INTRINSICS_JSON>
```

---

## 5. Linux vs Windows Notes

- Linux / macOS: use forward slashes (`/`)
- Windows: use quoted paths if spaces exist

All scripts are **OS-independent**.

---

## 6. Class Mapping

### Merge all human classes into `person` (recommended)

```bash
--merge_humans_to_person
```

This automatically converts:
```
human1, human2, human3, human4, human5 → person
```

### Custom class mapping

```bash
--class_map '{"human1":"person","worker":"person"}'
```

---

## 7. Output Files Explained

- **images/** – YOLO images
- **labels/** – YOLO label files
- **classes.txt** – ordered class list
- **data.yaml** – YOLO training configuration

---

## 8. Training Example (Ultralytics YOLO)

```bash
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt imgsz=640
```

---

## 9. Guarantees

- No train/val/test leakage
- Session-safe splitting
- Deterministic, reproducible exports
- Same dataset can be re-exported multiple times

---

## 10. Current Limitations

- 2D bounding boxes only
- Requires per-camera annotation json named `<camera>_ann.json` unless `--ann_json` overrides
