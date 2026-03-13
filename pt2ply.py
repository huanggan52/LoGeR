"""
Convert a saved .pt inference result to:
  - a colored .ply point cloud (subsampled, confidence-filtered)
  - a TUM-format trajectory file: t px py pz qx qy qz qw
    where t = frame_index * stride  (e.g. frame_0007.jpg + stride=3 -> t=21.000000)

Usage:
  python pt2ply.py \
      --input  results_pi3/scene0000_00_color_1000_0_-1_1.pt \
      --image_dir  data/long_scannet_s3/scene0000_00/color_1000 \
      --stride 3 \
      --conf_threshold 0.1 \
      --subsample 2
"""
import argparse
import os
import re
import glob
import numpy as np
import torch
from natsort import natsorted
from loger.utils.basic import write_ply
from loger.utils.rotation import mat_to_quat

parser = argparse.ArgumentParser()
parser.add_argument("--input",          required=True,  help="Path to .pt file")
parser.add_argument("--image_dir",      default=None,   help="Image folder (used to recover frame filenames for timestamps)")
parser.add_argument("--stride",         type=float, default=1.0,
                    help="Stride/scale factor: timestamp = frame_index * stride")
parser.add_argument("--output_ply",     default=None,   help="Output .ply path (default: <pt_stem>.ply)")
parser.add_argument("--output_traj",    default=None,   help="Output trajectory .txt path (default: <pt_stem>_traj.txt)")
parser.add_argument("--conf_threshold", type=float, default=0.1,
                    help="Confidence threshold [0,1] for point cloud filtering")
parser.add_argument("--subsample",      type=int,   default=1,
                    help="Spatial subsample factor for point cloud (1 = no subsampling)")
args = parser.parse_args()

# ── derive default output paths from input stem ──────────────────────────────
stem = os.path.splitext(args.input)[0]
out_ply  = args.output_ply   or f"{stem}.ply"
out_traj = args.output_traj  or f"{stem}_traj.txt"

# ── load .pt ─────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
data = torch.load(args.input, map_location="cpu", weights_only=False)
for k in list(data.keys()):
    if torch.is_tensor(data[k]):
        data[k] = data[k].float().numpy()

points = data["points"]        # (S, H, W, 3)
images = data["images"]        # (S, H, W, 3), float [0,1]
conf   = data["conf"]          # (S, H, W) or (S, H, W, 1)
poses  = data["camera_poses"]  # (S, 4, 4)  Twc

if conf.ndim == 4:
    conf = conf[..., 0]

S = points.shape[0]

# ── build timestamps from image filenames ────────────────────────────────────
def index_from_filename(name):
    """Extract the leading integer from a filename like frame_0007.jpg -> 7."""
    m = re.search(r'(\d+)', os.path.splitext(os.path.basename(name))[0])
    return int(m.group(1)) if m else None

timestamps = []
if args.image_dir and os.path.isdir(args.image_dir):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.image_dir, e))
    files = natsorted(files)
    if len(files) >= S:
        for f in files[:S]:
            idx = index_from_filename(f)
            timestamps.append(idx * args.stride if idx is not None else len(timestamps) * args.stride)
    else:
        print(f"Warning: found {len(files)} images but need {S}. Falling back to sequential timestamps.")

if len(timestamps) != S:
    timestamps = [i * args.stride for i in range(S)]

# ── save TUM trajectory ───────────────────────────────────────────────────────
# TUM format: t px py pz qx qy qz qw
Twc    = torch.from_numpy(poses)          # (S, 4, 4)
R      = Twc[:, :3, :3]                   # (S, 3, 3)
t      = Twc[:, :3, 3]                    # (S, 3)
qxyzw  = mat_to_quat(R)                   # (S, 4)  XYZW

os.makedirs(os.path.dirname(os.path.abspath(out_traj)), exist_ok=True)
with open(out_traj, "w") as f:
    f.write("# timestamp tx ty tz qx qy qz qw\n")
    for i in range(S):
        ts = timestamps[i]
        tx, ty, tz = t[i].tolist()
        qx, qy, qz, qw = qxyzw[i].tolist()
        f.write(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
print(f"Trajectory saved to {out_traj}  ({S} poses)")

# ── save point cloud ──────────────────────────────────────────────────────────
sub = args.subsample
pts_sub  = points[:, ::sub, ::sub, :]   # (S, H', W', 3)
img_sub  = images[:, ::sub, ::sub, :]
conf_sub = conf[:,   ::sub, ::sub]

mask = conf_sub > args.conf_threshold
xyz  = pts_sub[mask]
rgb  = (img_sub[mask] * 255).clip(0, 255).astype(np.uint8)

print(f"Point cloud: {xyz.shape[0]:,} points after filtering (conf>{args.conf_threshold}, subsample={sub})")
write_ply(xyz, rgb, path=out_ply)
print(f"Point cloud saved to {out_ply}")
