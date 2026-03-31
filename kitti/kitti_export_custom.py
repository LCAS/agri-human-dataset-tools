#!/usr/bin/env python3
import argparse
from pathlib import Path
from kitti_export_common import run_export

def main():
    ap=argparse.ArgumentParser(description="Export CUSTOM: raw + fisheyes + labels per fisheye")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor_camera", default="cam_zed_rgb")
    ap.add_argument("--fisheyes", nargs="*", default=["cam_fish_front","cam_fish_left","cam_fish_right"])
    ap.add_argument("--ann_source", nargs="*", default=["cam_zed_rgb","cam_fish_front","cam_fish_left","cam_fish_right"])
    ap.add_argument("--depth_camera", help="optional, include depth_2 as well")
    ap.add_argument("--depth_write_png16", action="store_true")
    ap.add_argument("--require_image", action="store_true")
    ap.add_argument("--require_lidar", action="store_true")
    ap.add_argument("--lidar_frame")
    ap.add_argument("--camera_optical_frame")
    ap.add_argument("--scenarios_file")
    ap.add_argument("--list_suffix", default="_label")
    # splits
    ap.add_argument("--manifest_tsv")
    ap.add_argument("--split_tag", choices=["train","val","test"])
    ap.add_argument("--split_file")
    # OXTS
    ap.add_argument("--oxts_fix_jsonl_rel", default="metadata/gps_fix.jsonl")
    ap.add_argument("--oxts_odom_jsonl_rel", default="metadata/gps_odom.jsonl")
    ap.add_argument("--oxts_ts_key", default="t")
    ap.add_argument("--oxts_max_dt_ms", type=int, default=200)
    ap.add_argument("--oxts_fields", nargs="*")
    args=ap.parse_args()

    run_export(
        mode="custom",
        root=Path(args.root).resolve(),
        out=Path(args.out).resolve(),
        anchor_camera=args.anchor_camera,
        require_image=args.require_image,
        require_lidar=args.require_lidar,
        lidar_frame=args.lidar_frame,
        camera_optical_frame=args.camera_optical_frame,
        scenarios_file=Path(args.scenarios_file).resolve() if args.scenarios_file else None,
        list_suffix=args.list_suffix,
        manifest_tsv=Path(args.manifest_tsv).resolve() if args.manifest_tsv else None,
        split_tag=args.split_tag,
        split_file=Path(args.split_file).resolve() if args.split_file else None,
        oxts_fix_jsonl_rel=args.oxts_fix_jsonl_rel,
        oxts_odom_jsonl_rel=args.oxts_odom_jsonl_rel,
        oxts_ts_key=args.oxts_ts_key,
        oxts_max_dt_ms=args.oxts_max_dt_ms,
        oxts_fields=args.oxts_fields,
        ann_source=args.ann_source,
        fisheyes=args.fisheyes,
        depth_camera=args.depth_camera,
        depth_write_png16=args.depth_write_png16
    )

if __name__=="__main__":
    main()

