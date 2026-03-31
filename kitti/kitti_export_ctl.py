#!/usr/bin/env python3
"""
export_ctl.py

Run one or many export jobs from a YAML file.

YAML example:
-------------
jobs:
  - mode: raw
    root: /data/ds
    out: /out/kitti_raw
    anchor_camera: cam_zed_rgb
    require_image: true
    require_lidar: true
    lidar_frame: front_lidar_link
    camera_optical_frame: front_left_camera_optical_frame
  - mode: object
    root: /data/ds
    out: /out/kitti_object
    anchor_camera: cam_zed_rgb
    ann_source: [cam_zed_rgb]
    manifest_tsv: /data/ds/manifest_samples.tsv
    split_tag: train
"""

import argparse, json, importlib
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML with jobs[]")
    args=ap.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML required. pip install pyyaml")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = cfg.get("jobs", [])
    if not jobs:
        raise SystemExit("No jobs found in YAML.")

    from export_common import run_export
    for j in jobs:
        mode = j["mode"]
        # Coerce/convert paths
        def P(x): return Path(x).resolve() if x else None
        run_export(
            mode=mode,
            root=P(j.get("root")),
            out=P(j.get("out")),
            anchor_camera=j.get("anchor_camera","cam_zed_rgb"),
            require_image=bool(j.get("require_image", False)),
            require_lidar=bool(j.get("require_lidar", False)),
            lidar_frame=j.get("lidar_frame"),
            camera_optical_frame=j.get("camera_optical_frame"),
            scenarios_file=P(j.get("scenarios_file")),
            list_suffix=j.get("list_suffix","_label"),
            manifest_tsv=P(j.get("manifest_tsv")),
            split_tag=j.get("split_tag"),
            split_file=P(j.get("split_file")),
            oxts_fix_jsonl_rel=j.get("oxts_fix_jsonl_rel","metadata/gps_fix.jsonl"),
            oxts_odom_jsonl_rel=j.get("oxts_odom_jsonl_rel","metadata/gps_odom.jsonl"),
            oxts_ts_key=j.get("oxts_ts_key","t"),
            oxts_max_dt_ms=int(j.get("oxts_max_dt_ms",200)),
            oxts_fields=j.get("oxts_fields"),
            ann_source=j.get("ann_source", []),
            fisheyes=j.get("fisheyes", []),
            depth_camera=j.get("depth_camera"),
            depth_write_png16=bool(j.get("depth_write_png16", False))
        )

if __name__ == "__main__":
    main()

