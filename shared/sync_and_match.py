#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchronise multi-modal streams per session and write <session>/sync.json.

Session selection priority:
1) If --scenarios_file is given (TXT/CSV/TSV/JSON/YAML), it is AUTHORITATIVE:
   - Names may be exact folder names (e.g., foo_..._label) OR base names (e.g., foo_...).
   - We will try appending --list_suffix (default: "_label") when needed.
   - We print a Missing-from-list report for unresolved names.
2) If no --scenarios_file, process all folders under <root> that end with "_label".

USAGE
-----
# Config (YAML or JSON), process all *_label
python sync_and_match.py /abs/path/Include_in_list --config config.yaml

# Limit to a list; the list file can live anywhere
python sync_and_match.py /abs/path/Include_in_list --config config.yaml --scenarios_file /lists/keep.csv

# If your list omits "_label", use --list_suffix (default = "_label")
python sync_and_match.py /data --scenarios_file /lists/keep.txt --list_suffix "_label"

WHAT IT WRITES
--------------
For each session folder under the root (e.g., Include_in_list/<scenario>_label),
writes <session>/sync.json with:
  - chosen_threshold_s
  - clock_offsets_ns (per target stream relative to anchor)
  - stats (completeness, p95, etc.)
  - samples[]: one entry per anchor frame:
        {timestamp_ns, anchor_modality, anchor_file, lidar, cameras{...}}

CONFIG KEYS (JSON or YAML)
--------------------------
anchor: "cam_zed_rgb" | "lidar" | ...
mods:
  cam_fish_left:  {path: sensor_data/cam_fish_left,  exts: [".png",".jpg",".jpeg"]}
  cam_fish_front: {path: sensor_data/cam_fish_front, exts: [".png",".jpg",".jpeg"]}
  cam_fish_right: {path: sensor_data/cam_fish_right, exts: [".png",".jpg",".jpeg"]}
  cam_zed_rgb:    {path: sensor_data/cam_zed_rgb,    exts: [".png",".jpg",".jpeg"]}
  cam_zed_depth:  {path: sensor_data/cam_zed_depth,  exts: [".png",".jpg",".jpeg",".npy"]}
lidar_path: "sensor_data/lidar"
lidar_exts: [".pcd"]
legacy_map: {fisheye_images_12: sensor_data/cam_fish_left, ...}
thresholds: [0.10, 0.12, 0.13, 0.15, 0.20, 0.30, 0.40, 0.50]
p95_limit_s: 0.05
plateau_eps: 0.2
one_to_one: false
enforce_monotonic: true
include_modalities: ["lidar","cam_fish_left","cam_fish_front","cam_fish_right","cam_zed_depth"]
"""
from __future__ import annotations
import json, argparse, csv as _csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODS = {
    "cam_fish_left":  {"path": "sensor_data/cam_fish_left",  "exts": [".png", ".jpg", ".jpeg"]},
    "cam_fish_front": {"path": "sensor_data/cam_fish_front", "exts": [".png", ".jpg", ".jpeg"]},
    "cam_fish_right": {"path": "sensor_data/cam_fish_right", "exts": [".png", ".jpg", ".jpeg"]},
    "cam_zed_rgb":    {"path": "sensor_data/cam_zed_rgb",    "exts": [".png", ".jpg", ".jpeg"]},
    "cam_zed_depth":  {"path": "sensor_data/cam_zed_depth",  "exts": [".png", ".jpg", ".jpeg", ".npy"]},
}
DEFAULT_LIDAR_PATH = "sensor_data/lidar"
DEFAULT_LIDAR_EXTS = [".pcd"]
DEFAULT_LEGACY_MAP = {
    "fisheye_images_12": "sensor_data/cam_fish_left",
    "fisheye_images_13": "sensor_data/cam_fish_front",
    "fisheye_images_14": "sensor_data/cam_fish_right",
    "output_images":     "sensor_data/cam_zed_rgb",
    "front_depth":       "sensor_data/cam_zed_depth",
    "lidar_points":      "sensor_data/lidar",
}
DEFAULT_THRESHOLDS    = [round(x / 100, 2) for x in range(10, 51)]  # 0.10..0.50
DEFAULT_P95_LIMIT_S   = 0.05
DEFAULT_PLATEAU_EPS   = 0.2
DEFAULT_ONE_TO_ONE    = False
DEFAULT_ENF_MONOTONIC = True

# ── datatypes ─────────────────────────────────────────────────────────────────
@dataclass
class TSFile:
    t_ns: int
    name: str

# ── config loader (JSON or YAML) ─────────────────────────────────────────────
def load_config(path: str|None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()
    if ext in (".yml", ".yaml"):
        try:
            import yaml  # pip install pyyaml
        except ImportError:
            raise SystemExit("YAML config requested but PyYAML is not installed. pip install pyyaml")
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

# ── scenarios list loader ────────────────────────────────────────────────────
def load_scenarios_list(path: Path) -> list[str]:
    """Return ordered list of raw names from file (keep ordering)."""
    if not path.exists():
        raise SystemExit(f"scenarios_file not found: {path}")
    ext = path.suffix.lower()
    # YAML
    if ext in (".yml", ".yaml"):
        try: import yaml
        except ImportError: raise SystemExit("YAML scenarios_file requires PyYAML. pip install pyyaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("YAML scenarios_file must be a list.")
    # JSON
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
        raise SystemExit("JSON scenarios_file must be an array.")
    # CSV/TSV
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        names = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f, delimiter=sep)
            key = "session_id" if "session_id" in reader.fieldnames else ("folder" if "folder" in reader.fieldnames else None)
            if key is None:
                raise SystemExit("CSV/TSV scenarios_file needs a 'session_id' or 'folder' column.")
            for row in reader:
                v = (row.get(key) or "").strip()
                if v: names.append(v)
        return names
    # TXT / LST / others -> one per line
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return names

def resolve_names_against_root(raw_names: list[str], root: Path, list_suffix: str) -> tuple[list[Path], list[str], list[tuple[str,str]]]:
    """
    Try to resolve each raw name to an existing folder under root.
    - First try exact match under root.
    - If not found and suffix not already present, try name+suffix.
    Returns: (resolved_paths, missing_names, suffix_mapped_pairs[(raw, mapped)])
    """
    resolved = []
    missing = []
    mapped  = []  # (raw, raw+suffix)
    for nm in raw_names:
        p = root / nm
        if p.is_dir():
            resolved.append(p); continue
        if not nm.endswith(list_suffix):
            q = root / f"{nm}{list_suffix}"
            if q.is_dir():
                resolved.append(q); mapped.append((nm, f"{nm}{list_suffix}")); continue
        # not found
        missing.append(nm)
    return resolved, missing, mapped

# ── util & IO ─────────────────────────────────────────────────────────────────
def resolve_folder(session_dir: Path, canonical_rel: str, legacy_map: dict) -> Path:
    p = session_dir / canonical_rel
    if p.exists():
        return p
    for legacy, canon in legacy_map.items():
        if canon == canonical_rel:
            cand = session_dir / legacy
            if cand.exists():
                return cand
    return p

def ts_from_name_ns(fname: str) -> int | None:
    stem = Path(fname).stem
    if "_" not in stem: return None
    a, b = stem.split("_", 1)
    if not (a.isdigit() and b.isdigit()): return None
    return int(a)*1_000_000_000 + int(b)

@dataclass
class TSFile:  # re-declare for clarity
    t_ns: int
    name: str

def scan(session_dir: Path, rel_path: str, exts: list[str], legacy_map: dict) -> list[TSFile]:
    folder = resolve_folder(session_dir, rel_path, legacy_map)
    out: list[TSFile] = []
    if folder.exists():
        for f in folder.iterdir():
            if f.suffix.lower() in [e.lower() for e in exts]:
                ns = ts_from_name_ns(f.name)
                if ns is not None:
                    out.append(TSFile(ns, f.name))
    out.sort(key=lambda x: x.t_ns)
    return out

def estimate_offsets_ns(anchor_ns, targets):
    offs={}
    for name, items in targets.items():
        if anchor_ns.size == 0 or not items:
            offs[name]=0; continue
        tgt_ns = np.array([it.t_ns for it in items], dtype=np.int64)
        j = np.abs(tgt_ns[:,None] - anchor_ns).argmin(axis=0)
        diffs = tgt_ns[j] - anchor_ns
        offs[name] = int(np.median(diffs)) if diffs.size else 0
    return offs

def nearest_idx(ts: np.ndarray, t: int) -> int:
    i = np.searchsorted(ts, t)
    if i==0: return 0
    if i==ts.size: return ts.size-1
    return i if abs(ts[i]-t)<abs(ts[i-1]-t) else i-1

def g_match(anchor_ns: np.ndarray, target_ns: np.ndarray, thr_ns: int, one_to_one: bool, mono: bool):
    nA, nT = len(anchor_ns), len(target_ns)
    used = np.zeros(nT, bool) if one_to_one else None
    out_idx = [-1]*nA; out_dif=[np.nan]*nA
    if mono:
        j=0
        for i,t in enumerate(anchor_ns):
            while j+1<nT and abs(target_ns[j+1]-t)<abs(target_ns[j]-t): j+=1
            cand=[j-1,j,j+1]
            cand=[k for k in cand if 0<=k<nT and (used is None or not used[k])]
            pick=-1; best=None
            for k in cand:
                d=abs(target_ns[k]-t)
                if best is None or d<best:
                    best=d; pick=k
            if pick!=-1 and best<=thr_ns:
                out_idx[i]=pick; out_dif[i]=best/1e9
                if used is not None: used[pick]=True
            if pick>j: j=pick
    else:
        for i,t in enumerate(anchor_ns):
            k=nearest_idx(target_ns,t)
            if used is not None and used[k]:
                k2 = k-1 if (k>0 and not used[k-1]) else (k+1 if (k+1<nT and not used[k+1]) else -1)
                if k2!=-1: k=k2
            d=abs(target_ns[k]-t)
            if d<=thr_ns and (used is None or not used[k]):
                out_idx[i]=k; out_dif[i]=d/1e9
                if used is not None: used[k]=True
    return out_idx, out_dif

def compute_metrics(anchor_ns, targets, thr_s, one_to_one, enf_monotonic):
    thr_ns=int(round(thr_s*1e9))
    total=len(anchor_ns)
    if total==0: return None
    mnn_hits=mnn_total=order_v=complete=0
    diffs=[]
    prev={k:-1 for k in targets}
    for name,data in targets.items():
        idx,df = g_match(anchor_ns, data["ts"], thr_ns, one_to_one, enf_monotonic)
        data["match_idx"], data["match_diff"]=idx,df
    for i in range(total):
        all_ok=True
        for name,data in targets.items():
            ci=data["match_idx"][i]
            if ci==-1: all_ok=False; continue
            diffs.append(data["match_diff"][i])
            li=nearest_idx(anchor_ns, data["ts"][ci])
            mnn_total+=1
            if li==i: mnn_hits+=1
            if prev[name]>ci: order_v+=1
            prev[name]=ci
        if all_ok: complete+=1
    diffs=np.array(diffs) if diffs else np.array([np.nan])
    return {
        "threshold": thr_s,
        "percent_complete": complete/total*100.0,
        "avg_diff": float(np.nanmean(diffs)),
        "p95": float(np.nanpercentile(diffs,95)) if np.isfinite(diffs).any() else np.nan,
        "p99": float(np.nanpercentile(diffs,99)) if np.isfinite(diffs).any() else np.nan,
        "mnn_rate": (mnn_hits/mnn_total) if mnn_total else 0.0,
        "order_violations": order_v,
    }

def choose_best_threshold(results, p95_limit, plateau_eps):
    def pick(cands):
        if not cands: return None
        maxc=max(r["percent_complete"] for r in cands)
        plateau=[r for r in cands if (maxc - r["percent_complete"]) <= plateau_eps]
        plateau.sort(key=lambda r: (-r["mnn_rate"], r["order_violations"], r["threshold"]))
        return plateau[0]
    ok=[r for r in results if not np.isnan(r["p95"]) and r["p95"]<=p95_limit]
    return pick(ok) or pick(results)

# ── per-session ───────────────────────────────────────────────────────────────
def process_session(session_dir: Path, cfg: dict):
    mods         = cfg["mods"]
    lidar_path   = cfg["lidar_path"]
    lidar_exts   = cfg["lidar_exts"]
    legacy_map   = cfg["legacy_map"]
    anchor       = cfg["anchor"]
    thresholds   = cfg["thresholds"]
    p95_limit    = cfg["p95_limit_s"]
    plateau_eps  = cfg["plateau_eps"]
    one_to_one   = cfg["one_to_one"]
    enf_mono     = cfg["enforce_monotonic"]
    include_list = cfg.get("include_modalities")

    print(f"\n── {session_dir.name}")

    # load streams
    lidar_items = scan(session_dir, lidar_path, lidar_exts, legacy_map)
    cam_items   = {k: scan(session_dir, v["path"], v["exts"], legacy_map) for k,v in mods.items()}
    all_items: dict[str, list[TSFile]] = {"lidar": lidar_items} | cam_items

    # filter targets if include_modalities specified
    if include_list:
        all_items = {k: v for k, v in all_items.items() if (k == anchor or k in include_list)}

    for name in ["cam_fish_left","cam_fish_front","cam_fish_right","cam_zed_rgb","cam_zed_depth","lidar"]:
        if name in all_items:
            print(f"  {name:<13}: {len(all_items[name])}")

    if anchor not in all_items or not all_items[anchor]:
        print(f"  !! Anchor '{anchor}' not present or empty → skip.")
        return None
    anchor_list = all_items[anchor]
    anchor_ns = np.array([it.t_ns for it in anchor_list], dtype=np.int64)
    targets = {k:v for k,v in all_items.items() if k != anchor}

    offs = estimate_offsets_ns(anchor_ns, targets)
    aligned = {k: {"ts": np.array([it.t_ns - offs.get(k,0) for it in v], dtype=np.int64),
                   "files":[it.name for it in v]}
               for k,v in targets.items()}

    results=[]
    for thr in thresholds:
        tmp={k:{"ts":v["ts"].copy(), "files":list(v["files"])} for k,v in aligned.items()}
        results.append(compute_metrics(anchor_ns, tmp, thr, one_to_one, enf_mono))
    best = choose_best_threshold(results, cfg["p95_limit_s"], cfg["plateau_eps"])
    if not best:
        print("  !! No viable threshold chosen."); return None
    print(f"  thr={best['threshold']:.2f}s  p95={best['p95']*1e3:.1f} ms  %Comp={best['percent_complete']:.1f}")

    thr_ns=int(round(best["threshold"]*1e9))
    final_idx={}
    for name,data in aligned.items():
        idx,_=g_match(anchor_ns, data["ts"], thr_ns, one_to_one, enf_mono)
        final_idx[name]=idx

    samples=[]
    for i, anchor_item in enumerate(anchor_list):
        cameras={}
        for cam in DEFAULT_MODS.keys():
            if cam == anchor: continue
            if cam in final_idx:
                ci=final_idx[cam][i]
                cameras[cam]=aligned[cam]["files"][ci] if ci!=-1 else "null"
            else:
                cameras[cam]="null"
        # preserve lidar filename for convenience
        if anchor == "lidar":
            lidar_name = anchor_item.name
        else:
            if "lidar" in final_idx and "lidar" in aligned:
                li=final_idx["lidar"][i]
                lidar_name = aligned["lidar"]["files"][li] if li!=-1 else "null"
            else:
                lidar_name = "null"
        samples.append({
            "timestamp_ns": int(anchor_item.t_ns),
            "anchor_modality": anchor,
            "anchor_file": anchor_item.name,
            "lidar": lidar_name,
            "cameras": cameras
        })

    (session_dir/"sync.json").write_text(json.dumps({
        "session": session_dir.name,
        "chosen_threshold_s": best["threshold"],
        "clock_offsets_ns": offs,
        "stats": best,
        "samples": samples
    }, indent=2), encoding="utf-8")
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root directory")
    ap.add_argument("--config", help="path to config file (.json, .yml, .yaml)")
    ap.add_argument("--anchor", default=None, help="override: anchor modality")
    ap.add_argument("--scenarios_file", help="authoritative list of sessions to process (path anywhere)")
    ap.add_argument("--list_suffix", default="_label", help="suffix to append to names from list when needed")
    args = ap.parse_args()

    cfg_in = load_config(args.config)
    eff = {
        "mods":        cfg_in.get("mods", DEFAULT_MODS),
        "lidar_path":  cfg_in.get("lidar_path", DEFAULT_LIDAR_PATH),
        "lidar_exts":  cfg_in.get("lidar_exts", DEFAULT_LIDAR_EXTS),
        "legacy_map":  cfg_in.get("legacy_map", DEFAULT_LEGACY_MAP),
        "thresholds":  cfg_in.get("thresholds", DEFAULT_THRESHOLDS),
        "p95_limit_s": cfg_in.get("p95_limit_s", DEFAULT_P95_LIMIT_S),
        "plateau_eps": cfg_in.get("plateau_eps", DEFAULT_PLATEAU_EPS),
        "one_to_one":  cfg_in.get("one_to_one", DEFAULT_ONE_TO_ONE),
        "enforce_monotonic": cfg_in.get("enforce_monotonic", DEFAULT_ENF_MONOTONIC),
        "include_modalities": cfg_in.get("include_modalities"),
        "anchor": args.anchor or cfg_in.get("anchor", "lidar"),
    }

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

    if not sessions:
        print("No sessions to process."); return

    summary=[]
    for sess in sorted(sessions):
        r=process_session(sess, eff)
        if r:
            r["session"]=sess.name
            summary.append(r)
    if summary:
        (root/"sync_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
