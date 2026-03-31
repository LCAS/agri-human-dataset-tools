#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kitti_export_common.py

Shared helpers + core export for KITTI-style exporters:
  - kitti_export_raw.py
  - kitti_export_object.py
  - kitti_export_depth.py
  - kitti_export_custom.py

Key features:
- calib_root override (so calibration/ can live one level above --root)
- undirected TF graph (fixes path search child→parent→child)
- robust annotations loader matching your schema:
    list of {"File": "...", "Labels":[{"Class": "...", "BoundingBoxes":[x,y,w,h]}, ...]}
  plus KITTI-like dicts keyed by filename
- annotation key modes: auto/stem/filename
- optional debug printing when labels missing
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json, csv, math
import numpy as np


# -----------------------------
# Scenario list helpers
# -----------------------------
def load_list_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"scenarios_file not found: {path}")
    ext = path.suffix.lower()
    if ext in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit("YAML scenarios_file requires PyYAML (pip install pyyaml).")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("YAML scenarios_file must be a list of names.")
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("JSON scenarios_file must be an array of names.")
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        names = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            key = "session_id" if reader.fieldnames and "session_id" in reader.fieldnames else (
                "folder" if reader.fieldnames and "folder" in reader.fieldnames else None
            )
            if key is None:
                raise SystemExit("CSV/TSV scenarios_file needs 'session_id' or 'folder' column.")
            for row in reader:
                v = (row.get(key) or "").strip()
                if v:
                    names.append(v)
        return names
    # txt
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def resolve_names(raw: List[str], root: Path, suffix: str) -> Tuple[List[Path], List[str], List[Tuple[str, str]]]:
    resolved, missing, mapped = [], [], []
    for nm in raw:
        p = root / nm
        if p.is_dir():
            resolved.append(p); continue
        if not nm.endswith(suffix):
            q = root / f"{nm}{suffix}"
            if q.is_dir():
                resolved.append(q); mapped.append((nm, f"{nm}{suffix}")); continue
        missing.append(nm)
    return resolved, missing, mapped


# -----------------------------
# Sync & manifest helpers
# -----------------------------
def load_sync(session_dir: Path) -> List[dict]:
    p = session_dir / "sync.json"
    if not p.exists():
        return []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("samples", obj)  # support list or {"samples":[...]}
    except Exception:
        return []

def ts_ns(sample: dict) -> int:
    if "timestamp_ns" in sample:
        return int(sample["timestamp_ns"])
    if "timestamp" in sample:
        return int(round(float(sample["timestamp"]) * 1e9))
    return 0

def load_manifest_tsv(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        sniffer = csv.Sniffer()
        buf = f.read(4096); f.seek(0)
        try:
            dialect = sniffer.sniff(buf)
            reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_split_file_any(path: Path):
    txt = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return set(), set()
    # naive: try CSV with session_id[,timestamp_ns]
    try:
        reader = csv.DictReader(lines)
        fns = [h.strip().lower() for h in (reader.fieldnames or [])]
        sess, samp = set(), set()
        for row in reader:
            sid = (row.get("session_id") or "").strip()
            if not sid:
                continue
            if "timestamp_ns" in fns:
                tns = int((row.get("timestamp_ns") or "0").strip() or "0")
                samp.add((sid, tns))
            else:
                sess.add(sid)
        return sess, samp
    except Exception:
        pass
    # txt list
    return set(lines), set()


# -----------------------------
# Calibration I/O (with calib_root override)
# -----------------------------
def _as_arr(x, shape=None):
    if x is None: return None
    a = np.array(x, dtype=float)
    if shape is not None:
        a = a.reshape(shape)
    return a

def load_intrinsics(root: Path, calib_root: Optional[Path] = None) -> Optional[dict]:
    base = calib_root if calib_root else root
    p = base / "calibration" / "intrinsics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def load_extrinsics(root: Path, calib_root: Optional[Path] = None) -> Optional[dict]:
    base = calib_root if calib_root else root
    p = base / "calibration" / "extrinsics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def parse_ros_caminfo(cam: dict):
    K = _as_arr(cam.get("k"), (3, 3))
    D = _as_arr(cam.get("d"))
    W = int(cam.get("width", 0)); H = int(cam.get("height", 0))
    frame = (cam.get("header", {}) or {}).get("frame_id")
    return K, D, (W, H), frame

def parse_fisheye(cam: dict):
    cm = cam.get("camera_matrix", {})
    dm = cam.get("distortion_coefficients", {})
    K = _as_arr(cm.get("data"), (3, 3)) if isinstance(cm, dict) else None
    D = _as_arr(dm.get("data")) if isinstance(dm, dict) else None
    W = int(cam.get("image_width", 0)); H = int(cam.get("image_height", 0))
    frame = (cam.get("header") or {}).get("frame_id") if isinstance(cam.get("header"), dict) else None
    return K, D, (W, H), frame

def pick_camera_intrinsics(intr: dict, cam_key: str):
    cam = intr.get(cam_key) if isinstance(intr, dict) and cam_key in intr else intr
    if not isinstance(cam, dict):
        if isinstance(intr, dict):
            for v in intr.values():
                if isinstance(v, dict):
                    cam = v; break
    if not isinstance(cam, dict):
        return None, None, (0, 0), None
    if "k" in cam and "d" in cam:
        return parse_ros_caminfo(cam)
    if "camera_matrix" in cam and "distortion_coefficients" in cam:
        return parse_fisheye(cam)
    K = _as_arr(cam.get("K") or cam.get("camera_matrix") or cam.get("intrinsics"), (3, 3))
    D = _as_arr(cam.get("D") or cam.get("distortion") or cam.get("distortion_coefficients"))
    W = int(cam.get("width", cam.get("image_width", 0)))
    H = int(cam.get("height", cam.get("image_height", 0)))
    frame = cam.get("frame_id") or (cam.get("header", {}) or {}).get("frame_id")
    return K, D, (W, H), frame

def write_cam_to_cam(out_dir: Path, K: np.ndarray, D: Optional[np.ndarray], size):
    W, H = size if size else (0, 0)
    D5 = np.zeros(5) if D is None else np.pad(np.array(D, dtype=float).ravel()[:5], (0, max(0, 5 - len(np.array(D).ravel()))))
    R_rect = np.eye(3)
    P_rect = np.hstack([K, np.zeros((3, 1))]) if K is not None else np.hstack([np.eye(3), np.zeros((3, 1))])
    lines = []
    lines.append(f"S_02: {int(W)} {int(H)}")
    lines.append("K_02: " + " ".join(f"{x:.12e}" for x in (K if K is not None else np.eye(3)).reshape(-1)))
    lines.append("D_02: " + " ".join(f"{x:.12e}" for x in D5))
    lines.append("R_02: " + " ".join(f"{x:.12e}" for x in R_rect.reshape(-1)))
    lines.append("T_02: " + " ".join(f"{0.0:.12e}" for _ in range(3)))
    lines.append("P_rect_02: " + " ".join(f"{x:.12e}" for x in P_rect.reshape(-1)))
    (out_dir / "calib_cam_to_cam.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# TF graph (undirected)
# -----------------------------
def quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)

def edge_to_T(edge: dict):
    t = edge["transform"]["translation"]
    r = edge["transform"]["rotation"]
    R = quat_to_R(r["x"], r["y"], r["z"], r["w"])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T

def invert_T(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -(R.T @ t)
    return Ti

def build_frame_graph(extr: dict):
    edges = extr.get("transforms", []) if isinstance(extr, dict) else []
    adj: Dict[str, List[str]] = {}
    Tmap: Dict[tuple, np.ndarray] = {}
    for e in edges:
        parent = (e.get("header", {}) or {}).get("frame_id")
        child  = e.get("child_frame_id")
        if not parent or not child:
            continue
        # make the graph undirected for BFS
        adj.setdefault(parent, []).append(child)
        adj.setdefault(child, []).append(parent)   # <-- reverse edge
        # transforms (forward + inverse)
        T = edge_to_T(e)
        Tmap[(parent, child)] = T
        Tmap[(child, parent)] = invert_T(T)
    return adj, Tmap

def find_path(adj, start, goal):
    from collections import deque
    if start not in adj or goal not in adj:
        return None
    q = deque([start]); prev = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u; q.append(v)
    if goal not in prev:
        return None
    path = []; cur = goal
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path

def chain_T(Tmap, path):
    T = np.eye(4)
    for a, b in zip(path[:-1], path[1:]):
        T = T @ Tmap[(a, b)]
    return T

def write_velo_to_cam(out_dir: Path, T_cam_lidar: Optional[np.ndarray]):
    if T_cam_lidar is None:
        Tr = np.hstack([np.eye(3), np.zeros((3, 1))])
    else:
        Tr = T_cam_lidar[:3, :4]
    lines = [
        "R: " + " ".join(f"{x:.12e}" for x in Tr[:, :3].reshape(-1)),
        "T: " + " ".join(f"{x:.12e}" for x in Tr[:, 3].reshape(-1)),
        "Tr_velo_to_cam: " + " ".join(f"{x:.12e}" for x in Tr.reshape(-1)),
    ]
    (out_dir / "calib_velo_to_cam.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# OXTS helpers (optional)
# -----------------------------
def index_jsonl_generic(session_dir: Path, rel_path: str, ts_key: str):
    fpath = session_dir / rel_path
    if not fpath or not fpath.exists():
        return [], [], fpath
    times = []; offs = []
    with fpath.open("rb") as f:
        off = 0
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get(ts_key, None)
                if isinstance(t, (int, float)):
                    times.append(int(round(float(t) * 1e9)))
                    offs.append(off)
            except Exception:
                pass
            off += len(line)
    return times, offs, fpath

def nearest_idx(sorted_ns: List[int], t_ns: int):
    import bisect
    if not sorted_ns: return -1, 1 << 60
    i = bisect.bisect_left(sorted_ns, t_ns)
    if i == 0: return 0, abs(sorted_ns[0] - t_ns)
    if i == len(sorted_ns):
        j = len(sorted_ns) - 1; return j, abs(sorted_ns[j] - t_ns)
    before = sorted_ns[i - 1]; after = sorted_ns[i]
    return (i, abs(after - t_ns)) if abs(after - t_ns) < abs(before - t_ns) else (i - 1, abs(before - t_ns))

def read_jsonl_at(path: Path, off: int):
    with path.open("rb") as f:
        f.seek(off); line = f.readline()
    return json.loads(line)

def quat_to_rpy(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw

def enu_to_body(vn, ve, vz, yaw):
    cy, sy = math.cos(yaw), math.sin(yaw)
    vf =  cy * ve + sy * vn
    vl = -sy * ve + cy * vn
    vu = vz
    return vf, vl, vu


# -----------------------------
# PCD → BIN
# -----------------------------
def pcd_to_bin(pcd_path: Path, bin_path: Path):
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit("open3d is required for PCD→BIN. Install: pip install open3d")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        arr = np.zeros((0, 4), dtype=np.float32)
    else:
        pts = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        arr = np.hstack([pts, intensity]).astype(np.float32)
    arr.tofile(str(bin_path))


# -----------------------------
# Robust annotations
# -----------------------------
def _norm_key(s: str) -> str:
    return Path(s).stem.lower()

def _to_kitti_obj(obj):
    # Already KITTI-like?
    if "type" in obj and "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) == 4:
        xmin, ymin, xmax, ymax = obj["bbox"]
        return {"type": str(obj["type"]), "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]}
    # Your schema: Class + BoundingBoxes [x,y,w,h]
    cls = obj.get("Class") or obj.get("class") or obj.get("label")
    bb  = obj.get("BoundingBoxes") or obj.get("bbox") or obj.get("box")
    if cls is not None and isinstance(bb, (list, tuple)) and len(bb) == 4:
        x, y, w, h = [float(v) for v in bb]
        xmin, ymin = x, y
        xmax, ymax = x + w, y + h
        cls_map = {
            "human": "Person", "person": "Person",
            "human1": "Person", "human2": "Person",
        }
        kitti_type = cls_map.get(str(cls).lower(), str(cls))
        return {"type": kitti_type, "bbox": [xmin, ymin, xmax, ymax]}
    return None

def load_ann_for_modality(session_dir: Path, modality: str) -> dict:
    """
    Accepts:
      A) dict: {"<filename>": [objects...], ...}
      B) list: [{"File": "<filename>", "Labels":[objects...]}, ...]
    Returns dict indexed by STEM (lowercased).
    """
    p = session_dir / "annotations" / f"{modality}_ann.json"
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    index: Dict[str, List[dict]] = {}

    def add_one(fname: str, objs_raw):
        if not fname:
            return
        stem = _norm_key(fname)
        objs = []
        for o in (objs_raw or []):
            n = _to_kitti_obj(o)
            if n:
                objs.append(n)
        if objs:
            index.setdefault(stem, []).extend(objs)

    if isinstance(data, dict):
        for fname, objs_raw in data.items():
            if isinstance(objs_raw, dict) and "objects" in objs_raw:
                objs_raw = objs_raw["objects"]
            add_one(fname, objs_raw)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fname = item.get("File") or item.get("file") or item.get("image") or item.get("name")
            objs_raw = item.get("Labels") or item.get("objects") or item.get("annotations") or []
            add_one(fname, objs_raw)
    return index

def lookup_ann(ann_index: dict, image_filename: str, key_mode: str = "auto") -> list:
    if not image_filename:
        return []
    fname = Path(image_filename).name
    stem = _norm_key(fname)
    if key_mode == "filename":
        return ann_index.get(fname.lower(), [])  # only if you indexed by filename
    # we index by stem; auto/stem behave the same here
    return ann_index.get(stem, [])


def kitti_label_line(o: dict) -> str:
    cls = str(o.get("type", "DontCare"))
    xmin, ymin, xmax, ymax = [float(x) for x in o.get("bbox", [0, 0, 0, 0])]
    truncated = float(o.get("trunc", 0.0))
    occluded  = int(o.get("occ", 0))
    alpha     = float(o.get("alpha", -10.0))
    h, w, l = [float(x) for x in o.get("dim", [0, 0, 0])]
    tx, ty, tz = [float(x) for x in o.get("loc", [0, 0, 0])]
    ry = float(o.get("rot_y", 0.0))
    return f"{cls} {truncated:.2f} {occluded:d} {alpha:.2f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} {h:.2f} {w:.2f} {l:.2f} {tx:.2f} {ty:.2f} {tz:.2f} {ry:.2f}"

def write_label_2_for_frame(out_dir: Path, frame: str, objs: List[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [kitti_label_line(o) for o in objs]
    (out_dir / f"{frame}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# -----------------------------
# Depth writers
# -----------------------------
def save_depth_npy_to_npy(depth_src: Path, dst: Path):
    arr = np.load(depth_src)
    np.save(dst, arr)

def save_depth_npy_to_png16(depth_src: Path, dst: Path, meters_to_unit: float = 256.0):
    import imageio.v2 as imageio
    d = np.load(depth_src)
    d = np.clip(d, 0, np.finfo(np.float32).max)
    png = (d * meters_to_unit).astype(np.uint16)  # 0 = invalid
    imageio.imwrite(dst, png)


# -----------------------------
# Core export
# -----------------------------
def export_session(
    mode: str,
    root: Path, session_dir: Path, out_dir: Path,
    anchor_camera: str, require_image: bool, require_lidar: bool,
    lidar_frame: Optional[str], camera_optical_frame: Optional[str],
    oxts_fix_rel: Optional[str], oxts_odom_rel: Optional[str], oxts_ts_key: str, oxts_max_dt_ms: int,
    oxts_order: List[str],
    per_sample_keep: Optional[set],
    ann_modalities: List[str],
    fisheyes: List[str],
    depth_camera: Optional[str],
    depth_write_png16: bool
) -> int:
    """
    Export one session into KITTI-style folders.
    Supports:
      - RAW: image_2, velodyne, oxts, calib
      - OBJECT/CUSTOM: label_2 from image 2D and/or LiDAR 3D (merged)
      - DEPTH/CUSTOM: depth_2 from .npy or .png
    Merging policy for label_2 is controlled by attributes set by run_export():
      _use_lidar_3d (bool), _prefer_lidar_3d (bool), _prefer_2d (bool),
      _ann_key_mode (str), _debug_labels (bool), _calib_root (Path or None)
    """
    samples = load_sync(session_dir)
    if not samples:
        print(f"[warn] {session_dir.name}: no sync.json; skip.")
        return 0

    # standard dirs
    img_dir = out_dir / "image_2"
    vel_dir = out_dir / "velodyne"
    oxts_dir = out_dir / "oxts" / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    if require_lidar: vel_dir.mkdir(parents=True, exist_ok=True)
    if (oxts_fix_rel or oxts_odom_rel): oxts_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timestamps.txt").touch()

    # extra dirs
    if mode in ("object", "custom"): (out_dir / "label_2").mkdir(parents=True, exist_ok=True)
    if mode in ("depth", "custom") and depth_camera: (out_dir / "depth_2").mkdir(parents=True, exist_ok=True)
    if mode == "custom":
        for cam in fisheyes:
            (out_dir / f"image_{cam.replace('cam_', '')}").mkdir(parents=True, exist_ok=True)

    # intrinsics
    intr = load_intrinsics(root, calib_root=getattr(export_session, "_calib_root", None))
    K = D = None; size = (0, 0); cam_frame_inferred = None
    if intr:
        K, D, size, cam_frame_inferred = pick_camera_intrinsics(intr, anchor_camera)
    if K is None:
        print("[warn] intrinsics missing or no K; writing identity.")
        K = np.eye(3)
    write_cam_to_cam(out_dir, K, D, size)

    # extrinsics (LiDAR → camera)
    T_cam_lidar = None
    extr = load_extrinsics(root, calib_root=getattr(export_session, "_calib_root", None))
    if extr and lidar_frame and (camera_optical_frame or cam_frame_inferred):
        adj, Tmap = build_frame_graph(extr)
        target_cam = camera_optical_frame or cam_frame_inferred
        path = find_path(adj, lidar_frame, target_cam)
        if path:
            T_cam_lidar = chain_T(Tmap, path)
        else:
            print(f"[warn] no TF path {lidar_frame} -> {target_cam}; identity Tr.")
    else:
        if require_lidar:
            print("[warn] extrinsics/frames missing while LiDAR required; identity Tr.")
    write_velo_to_cam(out_dir, T_cam_lidar)

    # OXTS indices
    fix_times = fix_offs = fix_path = None
    odom_times = odom_offs = odom_path = None
    if oxts_fix_rel:
        fix_times, fix_offs, fix_path = index_jsonl_generic(session_dir, oxts_fix_rel, oxts_ts_key)
    if oxts_odom_rel:
        odom_times, odom_offs, odom_path = index_jsonl_generic(session_dir, oxts_odom_rel, oxts_ts_key)

    # annotations caches / flags
    ann_maps = {m: load_ann_for_modality(session_dir, m) for m in (ann_modalities or [])}
    ann_key_mode   = getattr(export_session, "_ann_key_mode", "auto")
    debug_labels   = getattr(export_session, "_debug_labels", False)
    use_lidar_3d   = getattr(export_session, "_use_lidar_3d", False)
    prefer_lidar_3d= getattr(export_session, "_prefer_lidar_3d", True)
    prefer_2d      = getattr(export_session, "_prefer_2d", False)  # 🆕 if True, this overrides prefer_lidar_3d

    # lazy per-session LiDAR ann index (only if needed)
    lidar_ann_idx = None
    if (mode in ("object", "custom")) and use_lidar_3d:
        lidar_ann_idx = load_lidar_ann_by_frame(session_dir)

    n = 0
    for s in samples:
        sid = session_dir.name
        t = ts_ns(s)
        if per_sample_keep is not None and (sid, t) not in per_sample_keep:
            continue

        cams = s.get("cameras", {})
        cam_file = s.get("anchor_file") if s.get("anchor_modality") == anchor_camera else cams.get(anchor_camera)
        lidar_file = s.get("lidar")
        if require_image and (not cam_file or cam_file == "null"): continue
        if require_lidar and (not lidar_file or lidar_file == "null"): continue

        cam_src = session_dir / "sensor_data" / anchor_camera / cam_file if cam_file else None
        lid_src = session_dir / "sensor_data" / "lidar" / lidar_file if lidar_file else None

        frame = f"{n:06d}"

        # image_2
        if cam_src is not None and cam_src.exists():
            (img_dir / f"{frame}.png").write_bytes(cam_src.read_bytes())

        # velodyne
        if require_lidar and lid_src is not None and lid_src.exists():
            pcd_to_bin(lid_src, vel_dir / f"{frame}.bin")

        # OXTS (optional)
        if (fix_times or odom_times):
            fix_obj = odom_obj = None
            if fix_times:
                j_fix, dt_fix = nearest_idx(fix_times, t)
                if (dt_fix / 1e6) <= getattr(export_session, "_oxts_max_dt_ms", 200):
                    fix_obj = read_jsonl_at(fix_path, fix_offs[j_fix])
            if odom_times:
                j_odom, dt_odom = nearest_idx(odom_times, t)
                if (dt_odom / 1e6) <= getattr(export_session, "_oxts_max_dt_ms", 200):
                    odom_obj = read_jsonl_at(odom_path, odom_offs[j_odom])
            if (fix_obj is not None) or (odom_obj is not None):
                lat = float(fix_obj.get("lat", fix_obj.get("latitude", 0.0))) if fix_obj else 0.0
                lon = float(fix_obj.get("lon", fix_obj.get("longitude", 0.0))) if fix_obj else 0.0
                alt = float(fix_obj.get("alt", fix_obj.get("altitude", 0.0))) if fix_obj else 0.0
                roll = pitch = yaw = 0.0
                if odom_obj and "q" in odom_obj:
                    q = odom_obj["q"]
                    roll, pitch, yaw = quat_to_rpy(float(q.get("x", 0.0)), float(q.get("y", 0.0)),
                                                   float(q.get("z", 0.0)), float(q.get("w", 1.0)))
                    if (q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0), q.get("w", 1.0)) == (0,0,0,1) and ("yaw" in odom_obj):
                        yaw = float(odom_obj["yaw"])
                vn = ve = vz = 0.0
                if odom_obj and "v" in odom_obj:
                    vn = float(odom_obj["v"].get("x", 0.0))
                    ve = float(odom_obj["v"].get("y", 0.0))
                    vz = float(odom_obj["v"].get("z", 0.0))
                vf, vl, vu = enu_to_body(vn, ve, vz, yaw)
                wx = wy = wz = 0.0
                if odom_obj and "w" in odom_obj:
                    wx = float(odom_obj["w"].get("x", 0.0))
                    wy = float(odom_obj["w"].get("y", 0.0))
                    wz = float(odom_obj["w"].get("z", 0.0))
                ax = ay = az = 0.0
                posacc = 0.0; velacc = 0.0; navstat = 0.0
                if fix_obj:
                    cov = fix_obj.get("cov")
                    if isinstance(cov, list) and len(cov) >= 9:
                        sigma2 = max(float(cov[0]), float(cov[4]), float(cov[8]))
                        posacc = math.sqrt(max(0.0, sigma2))
                    navstat = float(fix_obj.get("status", 0.0))
                values = {
                    "lat": lat, "lon": lon, "alt": alt, "roll": roll, "pitch": pitch, "yaw": yaw,
                    "vn": vn, "ve": ve, "vz": vz, "vf": vf, "vl": vl, "vu": vu,
                    "ax": ax, "ay": ay, "az": az, "wx": wx, "wy": wy, "wz": wz,
                    "posacc": posacc, "velacc": velacc, "navstat": navstat
                }
                (oxts_dir / f"{frame}.txt").write_text(
                    " ".join(f"{float(values[k]):.12e}" for k in oxts_order) + "\n", encoding="utf-8"
                )

        # timestamps
        (out_dir / "timestamps.txt").open("a", encoding="utf-8").write(f"{t/1e9:.9f}\n")

        # -------------------------------
        # label_2 merging policy
        # -------------------------------
        if mode in ("object", "custom"):
            lines: List[str] = []

            # Prepare 2D (image) objects for this frame
            objs2d = []
            if ann_modalities:
                anchor_ann = ann_maps.get(anchor_camera, {})
                objs2d = lookup_ann(anchor_ann, cam_file, ann_key_mode) if anchor_ann else []
                if not objs2d and debug_labels:
                    print(f"[labels] no 2D match for {cam_file} (mod={anchor_camera}) in {session_dir.name}")

            # Prepare LiDAR 3D → camera 3D (+ projected 2D) for this frame
            objs3d_lines = []
            if use_lidar_3d and (T_cam_lidar is not None) and lidar_ann_idx is not None:
                lid_key = Path(lidar_file).stem.lower() if lidar_file else None
                boxes_velo = []
                if lid_key and lid_key in lidar_ann_idx:
                    boxes_velo = lidar_ann_idx[lid_key]
                else:
                    # fallback: timestamp key as seconds string
                    k2 = str(int(t/1e9)) if t else None
                    if k2 and k2 in lidar_ann_idx:
                        boxes_velo = lidar_ann_idx[k2]
                if boxes_velo:
                    boxes_cam = transform_boxes_velo_to_cam(boxes_velo, T_cam_lidar)
                    for b in boxes_cam:
                        xmin=ymin=xmax=ymax=0.0
                        if K is not None and (require_image or cam_src is not None):
                            rect = rect_from_3d(K, b)
                            if rect is not None:
                                xmin,ymin,xmax,ymax = rect
                        o = {
                            "type": b.cls, "trunc": 0.0, "occ": 0, "alpha": -10.0,
                            "bbox": [xmin, ymin, xmax, ymax],
                            "dim": [b.h, b.w, b.l],
                            "loc": [b.x, b.y, b.z],
                            "rot_y": b.ry
                        }
                        objs3d_lines.append(kitti_label_line(o))

            # Emit according to preference
            if prefer_2d:
                # 2D first (zero 3D), then append 3D if any
                for o in objs2d:
                    xmin,ymin,xmax,ymax = o.get("bbox",[0,0,0,0])
                    o2 = {"type": o.get("type","DontCare"), "trunc":0.0, "occ":0, "alpha":-10.0,
                          "bbox":[xmin,ymin,xmax,ymax], "dim":[0,0,0], "loc":[0,0,0], "rot_y":0.0}
                    lines.append(kitti_label_line(o2))
                lines.extend(objs3d_lines)
            else:
                # default: prefer_lidar_3d controls order
                if prefer_lidar_3d and objs3d_lines:
                    lines.extend(objs3d_lines)
                    # append 2D-only as zero-3D
                    for o in objs2d:
                        xmin,ymin,xmax,ymax = o.get("bbox",[0,0,0,0])
                        o2 = {"type": o.get("type","DontCare"), "trunc":0.0, "occ":0, "alpha":-10.0,
                              "bbox":[xmin,ymin,xmax,ymax], "dim":[0,0,0], "loc":[0,0,0], "rot_y":0.0}
                        lines.append(kitti_label_line(o2))
                else:
                    # 2D-only first (if no 3D or not preferred), then add 3D if present
                    for o in objs2d:
                        xmin,ymin,xmax,ymax = o.get("bbox",[0,0,0,0])
                        o2 = {"type": o.get("type","DontCare"), "trunc":0.0, "occ":0, "alpha":-10.0,
                              "bbox":[xmin,ymin,xmax,ymax], "dim":[0,0,0], "loc":[0,0,0], "rot_y":0.0}
                        lines.append(kitti_label_line(o2))
                    lines.extend(objs3d_lines)

            # write label_2
            (out_dir / "label_2" / f"{frame}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # custom fisheyes (images + optional labels per fisheye)
        if mode == "custom":
            for cam in fisheyes:
                ffile = cams.get(cam)
                if ffile and ffile != "null":
                    src = session_dir / "sensor_data" / cam / ffile
                    dst = out_dir / f"image_{cam.replace('cam_', '')}" / f"{frame}.png"
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                ann_map = ann_maps.get(cam, {})
                if ann_map:
                    objs_f = lookup_ann(ann_map, ffile, ann_key_mode)
                    lab_dir = out_dir / f"label_{cam.replace('cam_', '')}"
                    lab_dir.mkdir(parents=True, exist_ok=True)
                    (lab_dir / f"{frame}.txt").write_text("\n".join(kitti_label_line(o) for o in objs_f) + ("\n" if objs_f else ""), encoding="utf-8")

        # depth
        if (mode in ("depth", "custom")) and depth_camera:
            dfile = cams.get(depth_camera)
            if dfile and dfile != "null":
                dsrc = session_dir / "sensor_data" / depth_camera / dfile
                if dsrc.suffix.lower() == ".npy":
                    if depth_write_png16:
                        save_depth_npy_to_png16(dsrc, (out_dir / "depth_2" / f"{frame}.png"))
                    else:
                        save_depth_npy_to_npy(dsrc, (out_dir / "depth_2" / f"{frame}.npy"))
                else:
                    (out_dir / "depth_2" / f"{frame}{dsrc.suffix}").write_bytes(dsrc.read_bytes())

        n += 1

    print(f"[export:{mode}] {session_dir.name}: {n} frames")
    return n


def run_export(
    mode: str,
    root: Path, out: Path,
    anchor_camera: str,
    require_image: bool, require_lidar: bool,
    lidar_frame: Optional[str], camera_optical_frame: Optional[str],
    scenarios_file: Optional[Path], list_suffix: str,
    manifest_tsv: Optional[Path], split_tag: Optional[str], split_file: Optional[Path],
    oxts_fix_jsonl_rel: Optional[str], oxts_odom_jsonl_rel: Optional[str], oxts_ts_key: str, oxts_max_dt_ms: int,
    oxts_fields: Optional[List[str]],
    ann_source: List[str], fisheyes: List[str],
    depth_camera: Optional[str], depth_write_png16: bool,
    # new:
    ann_key_mode: str = "auto",
    debug_labels: bool = False,
    calib_root: Optional[Path] = None,
):
    out.mkdir(parents=True, exist_ok=True)

    # sessions
    if scenarios_file:
        raw = load_list_file(scenarios_file)
        sessions, missing, mapped = resolve_names(raw, root, list_suffix)
        print(f"[sessions] list: {len(raw)} | resolved: {len(sessions)} | missing: {len(missing)}")
        if mapped:
            print("[mapped-with-suffix]"); [print(f"  {a} -> {b}") for a, b in mapped]
        if missing:
            print("[missing-from-list]"); [print(f"  - {nm}") for nm in missing]
    else:
        sessions = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(list_suffix)]
        print(f"[sessions] discovered under root: {len(sessions)}")

    # split filters
    per_session_keep = None; per_sample_keep = None
    if manifest_tsv:
        rows = load_manifest_tsv(manifest_tsv)
        if split_tag:
            rows = [r for r in rows if (r.get("split", "").strip().lower() == split_tag)]
        per_session_keep = set(r["session_id"] for r in rows if r.get("session_id"))
        per_sample_keep = set()
        for r in rows:
            sid = r.get("session_id"); tns = r.get("timestamp_ns")
            if sid and tns:
                try: per_sample_keep.add((sid, int(tns)))
                except: pass
    if split_file:
        sess_set, samp_set = load_split_file_any(split_file)
        if sess_set:
            per_session_keep = sess_set if per_session_keep is None else (per_session_keep & sess_set)
        if samp_set:
            per_sample_keep = samp_set if per_sample_keep is None else (per_sample_keep | samp_set)
    if per_session_keep is not None:
        sessions = [s for s in sessions if s.name in per_session_keep]

    # OXTS order for writing
    oxts_order = oxts_fields or ["lat","lon","alt","roll","pitch","yaw","vn","ve","vz","vf","vl","vu","ax","ay","az","wx","wy","wz","posacc","velacc","navstat"]

    # stash config for export_session to read
    export_session._ann_key_mode = ann_key_mode
    export_session._debug_labels = debug_labels
    export_session._calib_root   = calib_root
    export_session._oxts_max_dt_ms = oxts_max_dt_ms
    export_session._oxts_order     = oxts_order

    total = 0
    for sess in sorted(sessions):
        total += export_session(
            mode=mode,
            root=root, session_dir=sess, out_dir=out,
            anchor_camera=anchor_camera,
            require_image=require_image, require_lidar=require_lidar,
            lidar_frame=lidar_frame, camera_optical_frame=camera_optical_frame,
            oxts_fix_rel=oxts_fix_jsonl_rel, oxts_odom_rel=oxts_odom_jsonl_rel, oxts_ts_key=oxts_ts_key, oxts_max_dt_ms=oxts_max_dt_ms,
            oxts_order=oxts_order,
            per_sample_keep=per_sample_keep,
            ann_modalities=ann_source or [],
            fisheyes=fisheyes or [],
            depth_camera=depth_camera,
            depth_write_png16=depth_write_png16
        )
    print(f"[done:{mode}] exported {total} frames to {out}")
