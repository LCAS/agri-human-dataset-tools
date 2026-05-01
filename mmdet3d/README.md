# Agri-Human -> MMDetection3D Export

This folder adds a LiDAR-only export path for MMDetection3D models such as
PointPillars and SECOND.

## Why this exporter exists

Your current repo already covers:
- `yolo/` for 2D detection export
- `kitti/` for KITTI-style camera/LiDAR export

For LiDAR-only MMDetection3D training, the practical requirements are:
- point clouds in `.bin`
- 3D boxes in LiDAR coordinates
- split files
- annotation PKL files in `Det3DDataset` format

The official MMDetection3D custom-data docs still describe
`tools/create_data.py custom`, but current releases do not implement that path
cleanly. This exporter writes the PKL files directly.

## Export command

```powershell
python .\mmdet3d\export_agrihuman_to_mmdet3d.py `
  --dataset_root "D:\AOC\datasets\agri-human-sensing" `
  --out "D:\AOC\datasets\agri-human-sensing\mmdet3d_person"
```

Optional smoke test:

```powershell
python .\mmdet3d\export_agrihuman_to_mmdet3d.py `
  --dataset_root "D:\AOC\datasets\agri-human-sensing" `
  --out "D:\AOC\datasets\agri-human-sensing\mmdet3d_person_smoke" `
  --max_samples 50 `
  --overwrite
```

## Output layout

```text
<out>/
|-- points/
|-- labels/
|-- ImageSets/
|   |-- train.txt
|   |-- val.txt
|   |-- test.txt
|   `-- trainval.txt
|-- infos/
|   |-- agri_person_infos_train.pkl
|   |-- agri_person_infos_val.pkl
|   |-- agri_person_infos_test.pkl
|   `-- agri_person_infos_trainval.pkl
|-- sample_index.tsv
`-- export_summary.json
```

## Dataset assumptions

- `.pcd` is converted to 4D points `[x, y, z, intensity]`.
- If source PCD has no intensity field, intensity is written as zeros by default.
- All source classes are merged into one class: `person`.
- `lidar_ann.json` is treated as LiDAR-frame 3D boxes.
- 9-value boxes are interpreted as:
  `[x, y, z, dx, dy, dz, roll, pitch, yaw]`
- yaw is auto-detected as degrees vs radians and written to output in radians.
- roll and pitch are ignored by MMDetection3D voxel detectors.

## MMDetection3D integration

Copy [agri_person_dataset.py](./agri_person_dataset.py) into your
MMDetection3D dataset package and register it.

Minimal dataset config:

```python
dataset_type = 'AgriPersonDataset'
data_root = 'data/agri_person/'
class_names = ['person']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/agri_person_infos_train.pkl',
        data_prefix=dict(pts=''),
        metainfo=metainfo,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=False))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/agri_person_infos_val.pkl',
        data_prefix=dict(pts=''),
        metainfo=metainfo,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True))

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/agri_person_infos_test.pkl',
        data_prefix=dict(pts=''),
        metainfo=metainfo,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True))
```

For `LoadPointsFromFile`, keep `load_dim=4` and `use_dim=4`.

Use `export_summary.json` to set:
- `point_cloud_range`
- anchor ranges
- voxel size

Those values depend on your actual point spread, so they should be tuned from
the exported statistics rather than copied blindly from KITTI.
