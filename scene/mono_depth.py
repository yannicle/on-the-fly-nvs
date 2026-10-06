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

import torch
import os
import sys
import urllib.request
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from utils import sample

sys.path.append("submodules/Depth-Anything-V2")
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
from depth_anything_v2.dpt import DepthAnythingV2

# xformers picks its Hopper-only FlashAttention-3 kernels on any GPU >= sm90, which fails on
# Blackwell (e.g. RTX 50xx) with "invalid argument". Fall back to its other kernels there.
try:
    from xformers.ops.fmha import dispatch as xformers_dispatch
    if torch.cuda.get_device_capability()[0] != 9:
        xformers_dispatch._set_use_fa3(False)
except ImportError:
    pass

size = 518
encoder = "vitl"


class MonoDepthInternal(torch.nn.Module):
    def __init__(self, width: int, height: int, preprocessing: str = "square"):
        super(MonoDepthInternal, self).__init__()
        # "square": the original pipeline, frames squashed to size x size without normalisation.
        # "aspect": Depth-Anything-V2's own preprocessing, shortest side at size, both sides multiples of 14,
        # and ImageNet normalisation. Sharper depth maps, but not better renders on StaticHikes/forest1.
        self.preprocessing = preprocessing
        if preprocessing == "aspect":
            scale = size / min(width, height)
            self.input_size = (max(round(height * scale / 14), size // 14) * 14, max(round(width * scale / 14), size // 14) * 14)
        else:
            self.input_size = (size, size)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.half).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.half).view(1, 3, 1, 1))
        model_path = f"models/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = model.to("cuda").half().eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, image: torch.Tensor):
        if self.preprocessing == "aspect":
            img = torch.nn.functional.interpolate(
                image[None].half(), self.input_size, mode="bicubic", align_corners=True
            )
            img = (img - self.mean) / self.std
        else:
            img = torch.nn.functional.interpolate(
                image[None].half(), self.input_size, mode="bilinear", align_corners=True
            )
        depth = self.model(img)[None]
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.cat((grad_x, grad_y), dim=0)

        edges_sq_norm = (edges**2).sum(0, keepdim=True)
        var = 0.2
        confidence = torch.exp(-edges_sq_norm / var)
        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().median()
    return t, s


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    t_tri, s_tri = get_t_s(tri_idepth)
    t_mono, s_mono = get_t_s(mono_idepth)
    scale = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return mono_idepth * scale + offset, scale, offset


def align_depth(
    mono_depth_map: torch.Tensor, desc_kpts: DescribedKeypoints, width: int, height: int
):
    """Aligns the mono depth map with the triangulated depth from keypoints by finding the best scale and offset."""
    mono_idepth = sample(
        mono_depth_map,
        desc_kpts.kpts[desc_kpts.has_pt3d].view(1, 1, -1, 2),
        width,
        height,
    )[0, 0, 0]
    tri_idepth = 1 / desc_kpts.depth[desc_kpts.has_pt3d]

    mono_idepth_aligned, scale, offset = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    valid = err < 5 * err.median()
    mono_idepth_aligned, scale, offset = align_samples(
        tri_idepth[valid], mono_idepth[valid]
    )
    mono_depth_map_aligned = mono_depth_map * scale + offset

    return mono_depth_map_aligned


class MonoDepthEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int, preprocessing: str = "square"):
        self.width = width
        self.height = height
        model = MonoDepthInternal(width, height, preprocessing)

        dummy = torch.zeros(3, height, width).cuda()
        self.model = torch.cuda.make_graphed_callables(model, [dummy])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, conf = self.model(image)
        return depth.clone(), conf.clone()
