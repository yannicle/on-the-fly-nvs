#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import queue
import time
from threading import Thread

import cv2
import numpy as np
import torch
from torch import Tensor

from utils import equirect_view_rotation, get_equirect_maps, get_undistort_maps, open_webcam


class StreamDataset:
    def __init__(self, video_url: str, downsampling: float, retry_delay: float = 1.0,
                 undistort: str = "", undistort_fov_scale: float = 1.0, equirect: tuple = None):
        """
        Args:
            video_url (str): video stream URL, or webcam://<index> for a local USB webcam.
            retry_delay (int): Delay in seconds between retries.
            undistort (str): optional fisheye calibration json (scripts/calibrate_camera.py) used to undistort frames.
            undistort_fov_scale (float): < 1 keeps more of the field of view but adds black borders.
            equirect (tuple): (hfov, vfov, num_views, pitch) to cut a 360 equirectangular stream into a ring of num_views
                overlapping virtual pinhole views (angles in degrees). getnext returns view 0, and the other views
                with their rotation relative to view 0 in info["rig_views"].
        """
        self.video_url = video_url
        self.downsampling = downsampling
        self.undistort = undistort
        self.undistort_fov_scale = undistort_fov_scale
        self.equirect = equirect
        self.undistort_maps = None
        self.focal = None

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.retry_delay = retry_delay
        self.cap = None

        # Thread to get frames from the video stream
        self.capture_thd = Thread(target=self._capture_frames, daemon=True)
        self.capture_thd.start()

        self.num_frames = 0
    
    def _connect(self):
        if self.cap is not None:
            return

        if self.video_url.startswith("webcam://"):
            # 360 webcams such as the RICOH THETA output 2:1 equirectangular frames
            size = (3840, 1920) if self.equirect else (1920, 1080)
            cap = open_webcam(int(self.video_url[len("webcam://"):]), *size)
        else:
            cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print(f"Failed to open video stream: {self.video_url}")
            return

        print("Connected to camera stream.")
        self.cap = cap

    def _capture_frames(self) -> None:
        while self.running:
            if self.cap is None:
                self._connect() 
                time.sleep(self.retry_delay)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from stream.")
                self.cap.release()
                self.cap = None
                continue

            if not self.frame_queue.empty():
                self.frame_queue.get()  # Discard the older frame
            self.frame_queue.put(frame)

    def _equirect_views(self, frame):
        """
        Cuts the ring of views out of an equirectangular frame, view 0 looking at the panorama centre.
        Also returns, for each view, the fixed rotation from view 0's camera frame to its own.
        """
        hfov, vfov, num_views, pitch = self.equirect
        yaws = [k * 360 / num_views for k in range(int(num_views))]
        if self.undistort_maps is None:
            h, w = frame.shape[:2]
            print(f"Equirectangular stream {w}x{h}, cutting {len(yaws)} views of {hfov:.0f}x{vfov:.0f} deg")
            self.undistort_maps = [get_equirect_maps(w, h, hfov, vfov, yaw, pitch)[:2] for yaw in yaws]
            self.focal = w / (2 * np.pi)
            if self.downsampling > 0.0:
                self.focal /= self.downsampling
            ref = equirect_view_rotation(yaws[0], pitch)
            self.rig_rotations = [
                torch.from_numpy(equirect_view_rotation(yaw, pitch).T @ ref).float().cuda() for yaw in yaws
            ]
        views = [cv2.remap(frame, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) for maps in self.undistort_maps]
        return views, self.rig_rotations

    def _to_tensor(self, frame):
        if self.downsampling > 0.0 and self.downsampling != 1.0: 
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        return torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0

    def getnext(self) -> tuple[Tensor, dict]:
        frame = self.frame_queue.get(block=True)
        self.num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        info = {"is_test": False}
        if self.equirect:
            # View 0 is tracked, the other views share its centre with a fixed rotation
            views, rotations = self._equirect_views(frame)
            frame = views[0]
            info["rig_views"] = [(self._to_tensor(view), rot) for view, rot in zip(views[1:], rotations[1:])]
        elif self.undistort:
            if self.undistort_maps is None:
                h, w = frame.shape[:2]
                *self.undistort_maps, self.focal = get_undistort_maps(
                    self.undistort, w, h, self.undistort_fov_scale)
                if self.downsampling > 0.0:
                    self.focal /= self.downsampling
                print(f"Undistorting frames, focal after undistortion: {self.focal:.1f} px")
            frame = cv2.remap(frame, *self.undistort_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return self._to_tensor(frame), info

    def get_image_size(self):
        frame = self.getnext()[0]
        self.num_frames -= 1
        return frame.shape[-2], frame.shape[-1]

    def stop(self) -> None:
        self.running = False
        self.cap.release()
        self.capture_thd.join()
    
    def __len__(self):
        # Arbitrary large number as we don't know the length of a stream
        return 100_000_000  

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Dataset")
    parser.add_argument("-s", "--source_path", type=str, help="video stream URL")
    parser.add_argument("--downsampling", type=float, default=1.5)
    parser.add_argument("--undistort", type=str, default="")
    parser.add_argument("--undistort_fov_scale", type=float, default=1.0)
    parser.add_argument("--equirect", type=float, nargs=4, default=None, metavar=("HFOV", "VFOV", "NUM_VIEWS", "PITCH"))
    args = parser.parse_args()

    stream = StreamDataset(args.source_path, args.downsampling, undistort=args.undistort,
                           undistort_fov_scale=args.undistort_fov_scale, equirect=args.equirect)
    try:
        while True:
            image, info = stream.getnext()
            # Example processing
            cv2.imshow("Stream", image.permute(1, 2, 0).cpu().numpy()[..., ::-1])
            cv2.waitKey(1)
    except KeyboardInterrupt:
        stream.stop()
