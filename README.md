# Agri-Human Dataset Tools

This `merged` branch consolidates the code that previously lived across multiple branches into a single, organized layout while leaving the original branches untouched.

## Layout

```text
.
|-- LICENSE
|-- README.md
|-- shared/
|   |-- README.md
|   |-- build_manifest_and_splits.py
|   `-- sync_and_match.py
|-- kitti/
|   |-- README.md
|   |-- kitti_export.yaml
|   |-- kitti_export_common.py
|   |-- kitti_export_ctl.py
|   |-- kitti_export_custom.py
|   |-- kitti_export_depth.py
|   |-- kitti_export_object.py
|   `-- kitti_export_raw.py
|-- yolo/
|   |-- README.md
|   |-- coco_export_session.py
|   `-- yolo_export_session.py
|-- converters/
|   |-- fieldsafe_rgb_to_yolo.py
|   `-- yolo_to_coco.py
|-- filters/
|   `-- filter_to_person.py
|-- preprocessing/
|   `-- undistort_dataset_images.py
`-- ros2bag/
    |-- README.md
    `-- check_and_make_rosbag2.py
```

## Branch Origins

- `shared/` comes from `v1.0` and contains the common dataset synchronization and manifest tooling.
- `kitti/` comes from `KITTI-converter`.
- `yolo/` comes from `YOLO-converter`.
- `converters/` contains reusable format-transform utilities.
- `filters/` contains dataset filtering utilities.
- `preprocessing/` contains dataset preparation scripts such as image undistortion.
- `ros2bag/` comes from `ROS2bag-converter`.

The original branches remain available as-is:

- `KITTI-converter`
- `origin/KITTI-converter`
- `origin/YOLO-converter`
- `origin/ROS2bag-converter`
- `origin/v1.0`

## How To Use This Branch

You can either run scripts from the repository root with explicit paths:

```powershell
python .\shared\sync_and_match.py <DATASET_ROOT>
python .\shared\build_manifest_and_splits.py --root <DATASET_ROOT>
python .\kitti\kitti_export_object.py --root <DATASET_ROOT> --out <OUT_DIR>
python .\yolo\yolo_export_session.py --root <DATASET_ROOT> --out <OUT_DIR>
python .\converters\yolo_to_coco.py --images_dir <IMAGES_DIR> --labels_dir <LABELS_DIR> --yaml_path <DATA_YAML> --output_path <OUT_JSON>
python .\filters\filter_to_person.py --format coco --ann_file <ANN_JSON> --img_dir <IMG_DIR> --output <OUT_JSON> --human_labels person
python .\preprocessing\undistort_dataset_images.py --root <DATASET_ROOT> --intrinsics <INTRINSICS_JSON>
python .\ros2bag\check_and_make_rosbag2.py --bag-dir <DATASET_BAG_DIR>
```

Or `cd` into a toolkit folder and follow the local `README.md` there.

## Notes

- `shared/build_manifest_and_splits.py` and `shared/sync_and_match.py` are identical across `v1.0`, `KITTI-converter`, and `YOLO-converter`, so they are stored once.
- Each toolkit keeps its own branch-specific `README.md` to preserve the original usage guidance close to the code.
