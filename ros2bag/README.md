# ROS2 Bag Converter (Dataset → ROS 2 rosbag2)

This tool **checks a dataset folder for consistency** (sensor files, annotations, metadata)
and then **converts it into a ROS 2 rosbag2 directory** that can be played in ROS 2 Humble
and visualized in RViz2.

> **Note**
> A ROS 2 bag is a **directory**, not a single file.
> In this README, `<ROS2_BAG_DIR>` refers to the output directory passed to `--rosbag-out`.

---

## Expected dataset structure

```
<DATASET_BAG_DIR>/
├── sensor_data/
│   ├── cam_zed_rgb/        *.png
│   ├── cam_zed_depth/      *.npy
│   ├── cam_fish_front/     *.png
│   ├── cam_fish_left/      *.png
│   ├── cam_fish_right/     *.png
│   └── lidar/              *.pcd
├── annotations/
│   ├── cam_zed_rgb_ann.json
│   ├── cam_fish_front_ann.json
│   ├── cam_fish_left_ann.json
│   ├── cam_fish_right_ann.json
│   └── lidar_ann.json
├── metadata/
│   ├── *.jsonl
│   └── tf/
│       ├── map__to__odom.jsonl
│       └── odom__to__base_link.jsonl
└── sync.json   (optional)
```

---

## What the script does

### 1) Dataset checks (fail fast)

Before any conversion, the script validates:

- Sensor files
  - Counts files per modality
  - Extracts timestamps from filenames
- Annotations (`*_ann.json`)
  - Every `File` entry exists under `sensor_data/`
  - Annotation `Timestamp` matches the filename timestamp within a tolerance (default 5 ms)
- Metadata (`*.jsonl`)
  - Lines parse as valid JSON
  - Reports non-monotonic timestamps (does not fail by default)
- `sync.json`
  - Optional; missing file only produces a warning

If any annotation references are missing or timestamps do not match,
the script **stops immediately** and does **not** create a rosbag.

---

## ROS 2 bag output

### Sensor topics

| Dataset modality | ROS topic | Message type |
|---|---|---|
| cam_fish_front | /dataset/cam_fish_front/image | sensor_msgs/msg/Image |
| cam_fish_left | /dataset/cam_fish_left/image | sensor_msgs/msg/Image |
| cam_fish_right | /dataset/cam_fish_right/image | sensor_msgs/msg/Image |
| cam_zed_rgb | /dataset/cam_zed_rgb/image | sensor_msgs/msg/Image |
| cam_zed_depth | /dataset/cam_zed_depth/image | sensor_msgs/msg/Image |
| lidar | /dataset/lidar/points | sensor_msgs/msg/PointCloud2 |

---

### Label topics (ground truth)

| Source | ROS topic | Message type |
|---|---|---|
| cam_*_ann.json | /dataset/labels/<camera> | vision_msgs/msg/Detection2DArray |
| lidar_ann.json | /dataset/labels/lidar | vision_msgs/msg/Detection3DArray |

---

### Optional visualization

- /dataset/viz/lidar_boxes (visualization_msgs/msg/MarkerArray)

---

### TF handling

- Dynamic /tf (map → lidar)
- Written at every LiDAR timestamp to avoid extrapolation
- /tf_static intentionally not used

---

## Requirements

- ROS 2 Humble
- Python 3
- vision_msgs
- numpy
- Pillow

---

## Usage

### Check only

```bash
python3 check_and_make_rosbag2.py --bag-dir <DATASET_BAG_DIR>
```

### Convert to rosbag2

```bash
rm -rf <ROS2_BAG_DIR>

python3 check_and_make_rosbag2.py \
  --bag-dir <DATASET_BAG_DIR> \
  --make-rosbag \
  --rosbag-out <ROS2_BAG_DIR>
```

### Convert with TF

```bash
rm -rf <ROS2_BAG_DIR>

python3 check_and_make_rosbag2.py \
  --bag-dir <DATASET_BAG_DIR> \
  --make-rosbag \
  --rosbag-out <ROS2_BAG_DIR> \
  --write-tf \
  --tf-parent map \
  --tf-child lidar \
  --tf-xyzrpy 0,0,0,0,0,0
```

### Play

```bash
ros2 bag play <ROS2_BAG_DIR> --clock 
```

Note: Make sure **--clock** is used

---

## RViz2

- Fixed Frame: lidar (no TF) or map (with --write-tf)
- Add PointCloud2: /dataset/lidar/points
- Add MarkerArray: /dataset/viz/lidar_boxes

