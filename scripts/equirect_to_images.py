"""
Cuts a ring of overlapping virtual pinhole views out of every frame of a 360 equirectangular video
into <out_dir>/images, so it can be reconstructed like any image folder:
    python scripts/equirect_to_images.py -i video.mp4 -o data/my360
    python train.py -s data/my360 --init_focal <printed focal> --fix_focal --downsampling 1
Views are named <frame>_<view>.jpg and turn right around the ring, so consecutive images always overlap,
including the last view of a frame and the first view of the next one.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.append('.')
from utils import get_equirect_maps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="360 equirectangular video to overlapping pinhole images")
    parser.add_argument("-i", "--input", type=str, required=True, help="Equirectangular video (2:1 aspect)")
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=6, help="Views evenly spread around the horizon")
    parser.add_argument("--hfov", type=float, default=100.0,
                        help="Horizontal FoV of each view in degrees. Overlap between neighbours is hfov - 360 / num_views")
    parser.add_argument("--vfov", type=float, default=75.0, help="Vertical FoV of each view in degrees")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw of the first view from the panorama centre, in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Tilts all views up, in degrees")
    parser.add_argument("--every", type=int, default=1, help="Keep one video frame out of every N")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of video frames to keep")
    args = parser.parse_args()
    overlap = args.hfov - 360 / args.num_views
    if overlap < 15:
        print(f"Warning: neighbouring views only overlap by {overlap:.0f} deg. Aim for 20-40 deg, "
              f"e.g. --hfov 100 --num_views 6 or --hfov 80 --num_views 8")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Failed to open {args.input}")
    eq_w, eq_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(eq_w - 2 * eq_h) > 2:
        print(f"Warning: {eq_w}x{eq_h} is not 2:1, this does not look like an equirectangular video. "
              "Dual fisheye recordings need to be stitched (e.g. exported from the camera app) first.")

    step = 360 / args.num_views
    overlap = args.hfov - step
    if overlap <= 0:
        print(f"Warning: views do not overlap, increase --hfov above {step:.0f}")
    yaws = [args.yaw + k * step for k in range(args.num_views)]
    maps = [get_equirect_maps(eq_w, eq_h, args.hfov, args.vfov, yaw, args.pitch) for yaw in yaws]
    focal = maps[0][2]
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    idx, saved = 0, 0
    while args.max_frames < 0 or saved < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.every == 0:
            for k, (map1, map2, _) in enumerate(maps):
                view = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                cv2.imwrite(os.path.join(images_dir, f"{saved:06d}_{k}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1
    cap.release()

    h, w = maps[0][0].shape[:2]
    with open(os.path.join(args.out_dir, "equirect.json"), "w") as f:
        json.dump({"source": args.input, "width": w, "height": h, "focal": focal, "hfov": args.hfov,
                   "vfov": args.vfov, "yaws": yaws, "pitch": args.pitch}, f, indent=2)
    print(f"Saved {saved} frames x {args.num_views} views of {w}x{h} ({overlap:.0f} deg overlap) to {images_dir}")
    print(f"Focal: {focal:.2f} px. If train.py downsamples, divide it by the downsampling ratio, or pass --downsampling 1.")
    print(f"python train.py -s {args.out_dir} --init_focal {focal:.2f} --fix_focal --downsampling 1")
