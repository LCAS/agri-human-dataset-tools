#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a master manifest (and optional splits). Can also run the sync step first.

Session selection priority:
1) If --scenarios_file is given (TXT/CSV/TSV/JSON/YAML), it is AUTHORITATIVE:
   - Names may be exact folder names (e.g., foo_..._label) OR base names (e.g., foo_...).
   - We will try appending --list_suffix (default: "_label") when needed.
   - We print a Missing-from-list report for unresolved names.
2) If no --scenarios_file, process all folders under --root that end with "_label".

Other features:
- YAML/JSON config, CLI overrides (booleans only if explicitly passed).
- Attach JSONL metadata streams (configurable) to each row (inline fields + pointers).
- Optional per-session splits (train/val/test) or `--no_splits`.

See README for examples.
"""
import argparse, csv, json, re, random, subprocess, sys
from pathlib import Path
from bisect import bisect_left
from collections import defaultdict

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODS = {
    "lidar":         ("sensor_data/lidar",          [".pcd"]),
    "cam_fish_left": ("sensor_data/cam_fish_left",  [".png", ".jpg", ".jpeg"]),
    "cam_fish_front":("sensor_data/cam_fish_front", [".png", ".jpg", ".jpeg"]),
    "cam_fish_right":("sensor_data/cam_fish_right", [".png", ".jpg", ".jpeg"]),
    "cam_zed_rgb":   ("sensor_data/cam_zed_rgb",    [".png", ".jpg", ".jpeg"]),
    "cam_zed_depth": ("sensor_data/cam_zed_depth",  [".png", ".jpg", ".jpeg", ".npy"]),
}
DEFAULT_LEGACY_MAP = {
    "fisheye_images_12": "sensor_data/cam_fish_left",
    "fisheye_images_13": "sensor_data/cam_fish_front",
    "fisheye_images_14": "sensor_data/cam_fish_right",
    "output_images":     "sensor_data/cam_zed_rgb",
    "front_depth":       "sensor_data/cam_zed_depth",
    "lidar_points":      "sensor_data/lidar",
}
DEFAULT_TOL_MS = {k: (120 if k == "lidar" else 60) for k in DEFAULT_MODS}

ANN_FILES = {
    "lidar": "lidar_ann.json",
    "cam_fish_left": "cam_fish_left_ann.json",
    "cam_fish_front": "cam_fish_front_ann.json",
    "cam_fish_right": "cam_fish_right_ann.json",
    "cam_zed_rgb": "cam_zed_rgb_ann.json",
    "cam_zed_depth": "cam_zed_depth_ann.json",
}

# ── config loader ────────────────────────────────────────────────────────────
def load_config(path: str | None):
    if not path: return {}
    p=Path(path)
    if not p.exists(): raise SystemExit(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()
    if ext in (".yml",".yaml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit("YAML config requested but PyYAML not installed. pip install pyyaml")
        try:
            data = yaml.safe_load(text)
            return data or {}
        except Exception as e:
            raise SystemExit(f"Failed to parse YAML config '{p}': {e}")
    else:
        try:
            return json.loads(text)
        except Exception as e:
            raise SystemExit(f"Failed to parse JSON config '{p}': {e}")

# ── scenarios helpers ────────────────────────────────────────────────────────
def load_scenarios_list(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"scenarios_file not found: {path}")
    ext = path.suffix.lower()
    if ext in (".yml",".yaml"):
        try: import yaml
        except ImportError: raise SystemExit("YAML scenarios_file requires PyYAML. pip install pyyaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("YAML scenarios_file must be a list.")
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("JSON scenarios_file must be an array.")
    if ext in (".csv",".tsv"):
        sep = "," if ext == ".csv" else "\t"
        names=[]
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            key = "session_id" if "session_id" in reader.fieldnames else ("folder" if "folder" in reader.fieldnames else None)
            if key is None:
                raise SystemExit("CSV/TSV scenarios_file needs a 'session_id' or 'folder' column.")
            for row in reader:
                v=(row.get(key) or "").strip()
                if v: names.append(v)
        return names
    names=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        s=line.strip()
        if s and not s.startswith("#"): names.append(s)
    return names

def resolve_names_against_root(raw_names: list[str], root: Path, list_suffix: str):
    resolved=[]; missing=[]; mapped=[]
    for nm in raw_names:
        p = root / nm
        if p.is_dir():
            resolved.append(p); continue
        if not nm.endswith(list_suffix):
            q = root / f"{nm}{list_suffix}"
            if q.is_dir():
                resolved.append(q); mapped.append((nm, f"{nm}{list_suffix}")); continue
        missing.append(nm)
    return resolved, missing, mapped

# ── misc helpers ─────────────────────────────────────────────────────────────
def resolve(session_dir: Path, canonical_rel: str, legacy_map: dict) -> Path:
    p = session_dir / canonical_rel
    if p.exists(): return p
    for legacy, canon in legacy_map.items():
        if canon == canonical_rel:
            cand = session_dir / legacy
            if cand.exists(): return cand
    return p

def ts_ns_from_name(name: str) -> int|None:
    stem = Path(name).stem
    if "_" not in stem: return None
    a,b = stem.split("_",1)
    if not (a.isdigit() and b.isdigit()): return None
    return int(a)*1_000_000_000 + int(b)

def list_files(dir_path: Path, exts):
    if not dir_path.exists(): return []
    out=[]
    for p in dir_path.iterdir():
        if p.suffix.lower() in [e.lower() for e in exts]:
            ns=ts_ns_from_name(p.name)
            if ns is not None:
                out.append((ns, p.name))
    out.sort(key=lambda x:x[0])
    return out

def parse_scenario(session_id: str) -> str:
    import re
    m = re.match(r"^(.*)_\d\d_\d\d_\d\d\d\d(?:_\d+)?_label$", session_id)
    if m: return m.group(1)
    m2 = re.match(r"^(.*)_\d\d_\d\d_\d\d\d\d(?:_\d+)?$", session_id)
    return m2.group(1) if m2 else session_id

def nearest_idx(arr, t):
    from bisect import bisect_left
    i = bisect_left(arr, t)
    if i==0: return 0
    if i==len(arr): return len(arr)-1
    return i if abs(arr[i]-t)<abs(arr[i-1]-t) else i-1

def write_tsv(rows, out_path: Path):
    if not rows:
        out_path.write_text("", encoding="utf-8"); return
    headers=[]
    for r in rows:
        for k in r.keys():
            if k not in headers: headers.append(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, delimiter="\t", fieldnames=headers)
        w.writeheader(); w.writerows(rows)

def make_splits(samples: list, root: Path, seed: int = 42, ratios=(0.8,0.1,0.1)):
    random.seed(seed)
    sessions=sorted({r["session_id"] for r in samples})
    random.shuffle(sessions)
    n=len(sessions); n_tr=max(1,int(ratios[0]*n)); n_va=max(1,int(ratios[1]*n))
    train=set(sessions[:n_tr]); val=set(sessions[n_tr:n_tr+n_va]); test=set(sessions[n_tr+n_va:]) or {sessions[-1]}
    def bucket(r): return "train" if r["session_id"] in train else ("val" if r["session_id"] in val else "test")
    by_scn=defaultdict(list)
    for r in samples: by_scn[r["scenario"]].append(r)
    parts=list(DEFAULT_MODS.keys())
    def write_lists(rows, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        b={"train":[],"val":[],"test":[]}
        for r in rows:
            line=" ".join([r.get(f"{m}_path","") for m in parts]+[r.get("session_id",""),r.get("scenario",""),r.get("sample_id",""),r.get("anchor_modality",""),r.get("anchor_ts","")])
            b[bucket(r)].append(line)
        for k,v in b.items():
            (out_dir/f"{k}.txt").write_text("\n".join(v), encoding="utf-8")
    write_lists(samples, root/"splits"/"default")
    for scn,rows in by_scn.items():
        write_lists(rows, root/"splits"/f"scenario={scn}")

# ── metadata helpers ─────────────────────────────────────────────────────────
def index_jsonl_times(session_dir: Path, rel_path: str, ts_key: str):
    fpath = session_dir / rel_path
    if not fpath.exists():
        return [], [], fpath
    times = []
    offsets = []
    with fpath.open("rb") as f:
        off = 0
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get(ts_key, None)
                if isinstance(t, (int, float)):
                    t_ns = int(round(float(t) * 1e9))
                    times.append(t_ns)
                    offsets.append(off)
            except Exception:
                pass
            off += len(line)
    return times, offsets, fpath

def find_nearest_idx(sorted_ns: list[int], t_ns: int):
    import bisect
    if not sorted_ns:
        return -1, None
    i = bisect.bisect_left(sorted_ns, t_ns)
    if i == 0:
        return 0, abs(sorted_ns[0] - t_ns)
    if i == len(sorted_ns):
        j = len(sorted_ns) - 1
        return j, abs(sorted_ns[j] - t_ns)
    before = sorted_ns[i - 1]
    after = sorted_ns[i]
    return (i, abs(after - t_ns)) if abs(after - t_ns) < abs(before - t_ns) else (i - 1, abs(before - t_ns))

def read_jsonl_line_by_offset(path: Path, byte_offset: int):
    with path.open("rb") as f:
        f.seek(byte_offset)
        line = f.readline()
    return json.loads(line)

# ── core ─────────────────────────────────────────────────────────────────────
def build_manifest(root: Path, cfg: dict):
    samples=[]; sessions=[]
    mods_map    = cfg["mods_map"]
    legacy_map  = cfg["legacy_map"]
    anchor      = cfg["anchor"]
    tol_ms      = cfg["tolerances_ms"]
    include_orp = cfg["include_orphans"]
    include_mods= cfg.get("include_modalities")
    meta_cfg    = cfg.get("metadata_sync", [])
    sessions_dirs = cfg["sessions"]

    calib_intr = (root/"calibration"/"intrinsics.json")
    calib_extr = (root/"calibration"/"extrinsics.json")
    calib_intr_rel = str(calib_intr.relative_to(root)) if calib_intr.exists() else ""
    calib_extr_rel = str(calib_extr.relative_to(root)) if calib_extr.exists() else ""

    for sess in sorted(sessions_dirs):
        session_id = sess.name
        scenario   = parse_scenario(session_id)
        meta_path_file  = resolve(sess, "metadata/meta.json", legacy_map)  # legacy support
        meta_rel   = str(meta_path_file.relative_to(root)) if meta_path_file.exists() else ""

        # annotations per modality (paths only)
        ann_root = resolve(sess, "annotations", legacy_map)
        ann_paths={}
        for m in ANN_FILES:
            p = ann_root/ANN_FILES[m]
            ann_paths[m] = str(p.relative_to(root)) if p.exists() else ""

        sessions.append({"session_id":session_id,"scenario":scenario,
                         "meta_path":meta_rel,"calib_intrinsics":calib_intr_rel,"calib_extrinsics":calib_extr_rel})

        # Build indices for configured metadata streams
        meta_indices = []
        for m in meta_cfg:
            times, offs, path = index_jsonl_times(sess, m["rel_path"], m.get("ts_key", "t"))
            meta_indices.append({"cfg": m, "times": times, "offs": offs, "path": path})

        sync_path = sess/"sync.json"
        if sync_path.exists():
            data=json.loads(sync_path.read_text())
            for s in data.get("samples",[]):
                row={"sample_id":f"{session_id}_{s['timestamp_ns']}",
                     "session_id":session_id,"scenario":scenario,
                     "anchor_modality": s.get("anchor_modality", anchor),
                     "anchor_ts": f"{s['timestamp_ns']/1e9:.9f}",
                     "meta_path":meta_rel,"calib_intrinsics":calib_intr_rel,"calib_extrinsics":calib_extr_rel}
                # lidar
                if "lidar" in mods_map:
                    if s.get("lidar","null")!="null":
                        sub,_=mods_map["lidar"]
                        row["lidar_path"]=str((resolve(sess, sub, legacy_map)/s["lidar"]).relative_to(root))
                        row["has_lidar"]=1; row["dt_lidar_ms"]=""
                        row["lidar_ann_path"]=ann_paths.get("lidar","")
                    else:
                        row["lidar_path"]=""; row["has_lidar"]=0; row["dt_lidar_ms"]=""
                        row["lidar_ann_path"]=ann_paths.get("lidar","")
                # anchor camera (if applicable)
                if anchor in mods_map and anchor!="lidar":
                    anchor_file=s.get("anchor_file","")
                    sub,_=mods_map[anchor]
                    if anchor_file:
                        row[f"{anchor}_path"]=str((resolve(sess, sub, legacy_map)/anchor_file).relative_to(root))
                        row[f"has_{anchor}"]=1; row[f"dt_{anchor}_ms"]="0"
                        row[f"{anchor}_ann_path"]=ann_paths.get(anchor,"")
                    else:
                        row[f"{anchor}_path"]=""; row[f"has_{anchor}"]=0; row[f"dt_{anchor}_ms"]=""
                        row[f"{anchor}_ann_path"]=ann_paths.get(anchor,"")
                # other cameras
                cams=s.get("cameras",{})
                for cam,(sub,_) in mods_map.items():
                    if cam in ("lidar", anchor): continue
                    if include_mods and cam not in include_mods:  # skip excluded
                        row[f"{cam}_path"]=""; row[f"has_{cam}"]=0; row[f"dt_{cam}_ms"]=""
                        row[f"{cam}_ann_path"]=ann_paths.get(cam,""); continue
                    fname=cams.get(cam,"null")
                    if fname=="null":
                        row[f"{cam}_path"]=""; row[f"has_{cam}"]=0; row[f"dt_{cam}_ms"]=""
                        row[f"{cam}_ann_path"]=ann_paths.get(cam,"")
                    else:
                        row[f"{cam}_path"]=str((resolve(sess, sub, legacy_map)/fname).relative_to(root))
                        row[f"has_{cam}"]=1; row[f"dt_{cam}_ms"]=""
                        row[f"{cam}_ann_path"]=ann_paths.get(cam,"")

                # attach metadata
                try:
                    anchor_ns = int(float(row["anchor_ts"]) * 1e9)
                except Exception:
                    anchor_ns = None
                if anchor_ns is not None:
                    for meta in meta_indices:
                        mcfg = meta["cfg"]
                        name = mcfg["name"]
                        prefix = mcfg.get("output_prefix", name)
                        times = meta["times"]; offs = meta["offs"]; p = meta["path"]
                        max_dt_ms = int(mcfg.get("max_dt_ms", 200))
                        also_ptr = bool(mcfg.get("also_store_pointer", False))
                        store_fields = mcfg.get("store_fields", [])
                        if also_ptr:
                            row[f"{name}_ptr"] = ""
                        for f in store_fields:
                            row[f"{prefix}_{f}"] = ""
                        if times:
                            j, dt = find_nearest_idx(times, anchor_ns)
                            if j != -1 and dt is not None and (dt/1e6) <= max_dt_ms:
                                if also_ptr:
                                    row[f"{name}_ptr"] = f"{str(p.relative_to(root))}::{offs[j]}"
                                if store_fields:
                                    obj = read_jsonl_line_by_offset(p, offs[j])
                                    for f in store_fields:
                                        if f in obj:
                                            row[f"{prefix}_{f}"] = obj[f]

                samples.append(row)
        else:
            # fallback NN
            files={m:list_files(resolve(sess, sub, legacy_map), exts) for m,(sub,exts) in mods_map.items()}
            if anchor not in files: continue
            anchors = files[anchor]
            for ns, fname in anchors:
                row={"sample_id":f"{session_id}_{ns}",
                     "session_id":session_id,"scenario":scenario,
                     "anchor_modality":anchor,"anchor_ts":f"{ns/1e9:.9f}",
                     "meta_path":meta_rel,"calib_intrinsics":calib_intr_rel,"calib_extrinsics":calib_extr_rel}
                # anchor path
                sub_a,_=mods_map[anchor]
                row[f"{anchor}_path"]=str((resolve(sess, sub_a, legacy_map)/fname).relative_to(root))
                row[f"has_{anchor}"]=1; row[f"dt_{anchor}_ms"]="0"
                row[f"{anchor}_ann_path"]=ann_paths.get(anchor,"")
                # others
                for m,(sub,exts) in mods_map.items():
                    if m==anchor: continue
                    if include_mods and m not in include_mods:
                        row[f"{m}_path"]=""; row[f"has_{m}"]=0; row[f"dt_{m}_ms"]=""
                        row[f"{m}_ann_path"]=ann_paths.get(m,""); continue
                    lst=files[m]
                    if not lst:
                        row[f"{m}_path"]=""; row[f"has_{m}"]=0; row[f"dt_{m}_ms"]=""
                        row[f"{m}_ann_path"]=ann_paths.get(m,""); continue
                    ts=[t for t,_ in lst]; idx=nearest_idx(ts, ns); dt_ms=abs(ts[idx]-ns)/1e6
                    if dt_ms<=tol_ms.get(m,60):
                        row[f"{m}_path"]=str((resolve(sess, sub, legacy_map)/lst[idx][1]).relative_to(root))
                        row[f"has_{m}"]=1; row[f"dt_{m}_ms"]=f"{dt_ms:.1f}"
                        row[f"{m}_ann_path"]=ann_paths.get(m,"")
                    else:
                        row[f"{m}_path"]=""; row[f"has_{m}"]=0; row[f"dt_{m}_ms"]=""
                        row[f"{m}_ann_path"]=ann_paths.get(m,"")

                # attach metadata (NN fallback path)
                anchor_ns = ns
                for meta in meta_indices:
                    mcfg = meta["cfg"]
                    name = mcfg["name"]
                    prefix = mcfg.get("output_prefix", name)
                    times = meta["times"]; offs = meta["offs"]; p = meta["path"]
                    max_dt_ms = int(mcfg.get("max_dt_ms", 200))
                    also_ptr = bool(mcfg.get("also_store_pointer", False))
                    store_fields = mcfg.get("store_fields", [])
                    if also_ptr:
                        row[f"{name}_ptr"] = ""
                    for f in store_fields:
                        row[f"{prefix}_{f}"] = ""
                    if times:
                        j, dt = find_nearest_idx(times, anchor_ns)
                        if j != -1 and dt is not None and (dt/1e6) <= max_dt_ms:
                            if also_ptr:
                                row[f"{name}_ptr"] = f"{str(p.relative_to(root))}::{offs[j]}"
                            if store_fields:
                                obj = read_jsonl_line_by_offset(p, offs[j])
                                for f in store_fields:
                                    if f in obj:
                                        row[f"{prefix}_{f}"] = obj[f]

                samples.append(row)

        # include orphans (single-modality rows)
        if include_orp:
            covered={m:set() for m in mods_map}
            for r in samples:
                if r.get("session_id")!=session_id: continue
                for m in mods_map:
                    p=r.get(f"{m}_path","")
                    if p: covered[m].add(Path(p).stem)
            for m,(sub,exts) in mods_map.items():
                for ns,fname in list_files(resolve(sess, sub, legacy_map), exts):
                    stem=Path(fname).stem
                    if stem in covered[m]: continue
                    row={"sample_id":f"{session_id}_{ns}_{m}",
                         "session_id":session_id,"scenario":scenario,
                         "anchor_modality":m,"anchor_ts":f"{ns/1e9:.9f}",
                         "meta_path":meta_rel,"calib_intrinsics":calib_intr_rel,"calib_extrinsics":calib_extr_rel}
                    for mm,(sub2,_) in mods_map.items():
                        if mm==m:
                            row[f"{mm}_path"]=str((resolve(sess, sub2, legacy_map)/fname).relative_to(root))
                            row[f"has_{mm}"]=1; row[f"dt_{mm}_ms"]="0"
                            row[f"{mm}_ann_path"]=ann_paths.get(mm,"")
                        else:
                            row[f"{mm}_path"]="" ; row[f"has_{mm}"]=0 ; row[f"dt_{mm}_ms"]="" ; row[f"{mm}_ann_path"]=ann_paths.get(mm,"")
                    # attach metadata to orphan row
                    anchor_ns = ns
                    for meta in meta_indices:
                        mcfg = meta["cfg"]
                        name = mcfg["name"]
                        prefix = mcfg.get("output_prefix", name)
                        times = meta["times"]; offs = meta["offs"]; p = meta["path"]
                        max_dt_ms = int(mcfg.get("max_dt_ms", 200))
                        also_ptr = bool(mcfg.get("also_store_pointer", False))
                        store_fields = mcfg.get("store_fields", [])
                        if also_ptr:
                            row[f"{name}_ptr"] = ""
                        for f in store_fields:
                            row[f"{prefix}_{f}"] = ""
                        if times:
                            j, dt = find_nearest_idx(times, anchor_ns)
                            if j != -1 and dt is not None and (dt/1e6) <= max_dt_ms:
                                if also_ptr:
                                    row[f"{name}_ptr"] = f"{str(p.relative_to(root))}::{offs[j]}"
                                if store_fields:
                                    obj = read_jsonl_line_by_offset(p, offs[j])
                                    for f in store_fields:
                                        if f in obj:
                                            row[f"{prefix}_{f}"] = obj[f]
                    samples.append(row)

    return samples, sessions

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root directory")
    ap.add_argument("--config", help="path to config file (.json, .yml, .yaml)")
    ap.add_argument("--anchor", default=None, help="override anchor modality")
    ap.add_argument("--include_orphans", action="store_true", default=None,
                    help="also add single-modality rows for unmatched frames")
    ap.add_argument("--no_splits", action="store_true", default=None,
                    help="skip writing train/val/test splits")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run_sync", action="store_true", help="run sync_and_match.py first with same --config/anchor")
    # Scenario selection
    ap.add_argument("--scenarios_file", help="authoritative list of sessions to process (path anywhere)")
    ap.add_argument("--list_suffix", default="_label", help="suffix to append to names from list when needed")
    # Metadata selection
    ap.add_argument("--metadata", nargs="*", default=None,
                    help="names of metadata streams to include (filters config metadata_sync by 'name')")
    ap.add_argument("--no_metadata", action="store_true", default=None,
                    help="disable metadata syncing for this run")
    args=ap.parse_args()

    cfg = load_config(args.config)

    anchor = args.anchor or cfg.get("anchor", "lidar")
    # Build modality map
    mods_cfg = cfg.get("mods", None)
    if mods_cfg:
        mods_map = {k:(v["path"], v["exts"]) for k,v in mods_cfg.items()}
        if "lidar" not in mods_map:
            mods_map = {"lidar": (cfg.get("lidar_path","sensor_data/lidar"), cfg.get("lidar_exts", [".pcd"]))} | mods_map
    else:
        mods_map = DEFAULT_MODS
    legacy_map = cfg.get("legacy_map", DEFAULT_LEGACY_MAP)
    tol_ms = cfg.get("tolerances_ms", DEFAULT_TOL_MS)

    # Boolean merges: config is base; CLI overrides only when provided
    include_orp = cfg.get("include_orphans", False) if args.include_orphans is None else True
    do_splits   = cfg.get("do_splits", True)        if args.no_splits      is None else (not args.no_splits)
    seed        = cfg.get("seed", 42)               if args.seed           is None else args.seed

    # Metadata merge
    meta_cfg = cfg.get("metadata_sync", [])
    if args.no_metadata is True:
        meta_cfg = []
    elif args.metadata is not None:
        wanted = set(args.metadata)
        meta_cfg = [m for m in meta_cfg if m.get("name") in wanted]

    root = Path(args.root).resolve()

    # Discover sessions
    if args.scenarios_file:
        raw = load_scenarios_list(Path(args.scenarios_file).resolve())
        sessions, missing, mapped = resolve_names_against_root(raw, root, args.list_suffix)
        print(f"[sessions] from list: {len(raw)} | resolved: {len(sessions)} | missing: {len(missing)}")
        if mapped:
            print("[mapped-with-suffix]")
            for (r, m) in mapped:
                print(f"  {r}  ->  {m}")
        if missing:
            print("[missing-from-list]")
            for nm in missing:
                print(f"  - {nm}")
    else:
        sessions = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(args.list_suffix)]
        print(f"[sessions] discovered under root: {len(sessions)}")

    if args.run_sync:
        cmd=[sys.executable, str(Path(__file__).with_name("sync_and_match.py")), str(root)]
        if args.config: cmd += ["--config", args.config]
        if args.anchor: cmd += ["--anchor", args.anchor]
        if args.scenarios_file:
            cmd += ["--scenarios_file", args.scenarios_file, "--list_suffix", args.list_suffix]
        print("Running sync step:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"[cfg] anchor={anchor}  include_orphans={include_orp}  do_splits={do_splits}  seed={seed}")
    if meta_cfg:
        print("[cfg] metadata_sync:", [m.get("name") for m in meta_cfg])

    eff = {
        "mods_map": mods_map, "legacy_map": legacy_map, "anchor": anchor,
        "tolerances_ms": tol_ms, "include_orphans": include_orp, "sessions": sessions,
        "include_modalities": cfg.get("include_modalities", None),
        "metadata_sync": meta_cfg
    }

    samples, sess_rows = build_manifest(root, eff)
    write_tsv(samples, root/"manifest_samples.tsv")
    write_tsv(sess_rows, root/"sessions.tsv")
    if do_splits:
        make_splits(samples, root, seed)

if __name__ == "__main__":
    main()
