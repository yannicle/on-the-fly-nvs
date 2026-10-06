#
# Streams the active anchor's Gaussians to Unreal Engine (RobVR Gaussian Splatting plugin,
# URobVRSplatNetworkSourceComponent) over TCP while reconstruction runs.
#
# Wire format (little-endian, identical to a .rvsplat file, see RobVRSplatTypes.h in the plugin):
#   envelope  16 B : "RVSS", u16 version=1, u16 type=2 (AnchorData), u32 payload_bytes, u32 sequence
#   header    52 B : u32 anchor_id, u32 generation, u32 num_splats, u32 flags,
#                    f32[3] anchor_position, f32[3] bounds_min, f32[3] bounds_max   (Unreal cm)
#   splats  N*32 B : f32[3] position (Unreal cm), u8[4] RGBA (sRGB DC colour, opacity),
#                    f16[6] covariance (xx, xy, xz, yy, yz, zz; Unreal cm^2), u32 padding
#
# Only the active anchor is sent. On-the-fly NVS builds every new anchor from the previous one's
# Gaussians (merging the small, distant ones), so the active anchor always covers the whole scene -
# fine near the camera, coarser further away - and the receiver only ever shows one payload anyway.
#
# Conversion matches the plugin's .ply importer with the COLMAP axis convention: source +Z forward,
# +X right, +Y down becomes Unreal (z, x, -y), metres become centimetres.
#
# Standalone, to stream a saved reconstruction (no training needed):
#   python unreal_stream.py -m results/pxl --tcp 127.0.0.1:47800
#

from __future__ import annotations

import argparse
import os
import socket
import struct
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

ENVELOPE = struct.Struct("<4sHHII")
HEADER = struct.Struct("<IIII3f3f3f")
SPLAT_BYTES = 32
FLAG_FROZEN = 1
SH_C0 = 0.28209479177387814
HALF_MAX = 65504.0

# Rows are where each source axis lands in Unreal space: x -> +Y, y -> -Z, z -> +X.
# As a column-vector matrix: unreal = SOURCE_TO_UNREAL @ source = (z, x, -y).
SOURCE_TO_UNREAL = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) normalised (w, x, y, z) -> (N, 3, 3), columns are the images of the axes."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)


@torch.no_grad()
def pack_gaussians(
    xyz: torch.Tensor,
    f_dc: torch.Tensor,
    log_scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity_logit: torch.Tensor,
    anchor_position: torch.Tensor | None = None,
    unit_scale: float = 100.0,
    min_opacity: float = 0.02,
) -> tuple[bytes, int, tuple[list[float], list[float], list[float]]]:
    """
    Raw training parameters (as stored in gaussian_params) -> packed splat bytes, the splat count, and
    (anchor_position, bounds_min, bounds_max) in Unreal cm. Runs on whatever device the inputs are on.
    """
    device = xyz.device
    basis = SOURCE_TO_UNREAL.to(device)

    opacity = torch.sigmoid(opacity_logit.reshape(-1))
    keep = (opacity >= min_opacity) & torch.isfinite(xyz).all(-1)
    xyz, f_dc, log_scaling, rotation, opacity = (
        xyz[keep], f_dc[keep], log_scaling[keep], rotation[keep], opacity[keep]
    )
    n = xyz.shape[0]

    position = (xyz @ basis.T) * unit_scale

    # Covariance straight in Unreal space: sum_i s_i^2 a_i a_i^T with a_i the rotated axes, each axis
    # taken through the basis change. Same construction as the importer's BuildCovarianceUnreal.
    axes = basis @ quaternion_to_matrix(F.normalize(rotation, dim=-1))  # columns = axes in Unreal
    scaled = axes * (torch.exp(log_scaling) * unit_scale)[:, None, :]
    cov = scaled @ scaled.transpose(1, 2)
    cov6 = torch.stack(
        [cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], dim=-1
    )
    # Half floats top out at 65504 (a ~2.5 m standard deviation in cm^2). Shrink oversized splats
    # uniformly instead of letting them turn into inf - keeps their shape, and they are almost always
    # near-transparent background anyway.
    max_diag = cov6[:, [0, 3, 5]].amax(-1, keepdim=True)
    cov6 = cov6 * torch.clamp(HALF_MAX / max_diag.clamp_min(1e-30), max=1.0)

    rgb = torch.clamp(SH_C0 * f_dc.reshape(n, 3) + 0.5, 0.0, 1.0)
    rgba = torch.cat([rgb, opacity[:, None]], dim=-1)
    rgba = (rgba * 255.0 + 0.5).to(torch.uint8)

    out = torch.zeros((n, SPLAT_BYTES), dtype=torch.uint8, device=device)
    out[:, 0:12] = position.to(torch.float32).contiguous().view(torch.uint8).view(n, 12)
    out[:, 12:16] = rgba
    out[:, 16:28] = cov6.to(torch.float16).contiguous().view(torch.uint8).view(n, 12)

    # Bounds: centres padded by the 99th percentile of the 3-sigma extent, as the importer does, so a
    # few huge background splats do not blow up the culling box.
    if n > 0:
        extent = 3.0 * torch.sqrt(cov6[:, [0, 3, 5]].amax(-1))
        k = min(n, max(1, int(n * 0.99)))
        padding = torch.kthvalue(extent.float().cpu(), k).values.item()
        bmin = (position.amin(0) - padding).tolist()
        bmax = (position.amax(0) + padding).tolist()
    else:
        bmin = bmax = [0.0, 0.0, 0.0]

    if anchor_position is not None:
        anchor = ((anchor_position.to(device).float().reshape(3) @ basis.T) * unit_scale).tolist()
    else:
        anchor = [0.0, 0.0, 0.0]

    return out.cpu().numpy().tobytes(), n, (anchor, bmin, bmax)


def build_message(
    splats: bytes, num_splats: int, anchor_id: int, generation: int, flags: int,
    anchor: list[float], bmin: list[float], bmax: list[float],
) -> bytes:
    header = HEADER.pack(anchor_id, generation, num_splats, flags, *anchor, *bmin, *bmax)
    envelope = ENVELOPE.pack(b"RVSS", 1, 2, HEADER.size + len(splats), generation)
    return envelope + header + splats


def parse_target(target: str) -> tuple[str, int]:
    host, port = target.rsplit(":", 1)
    return host, int(port)


class UnrealStreamer:
    """
    Sends the scene model's active anchor to Unreal every `interval` seconds on a daemon thread.

    Never blocks training: the only work on the caller's side is nothing at all, and on the streaming
    thread the scene lock is held just long enough to clone five tensors. Connection problems are
    retried quietly; if Unreal is not running yet, the streamer keeps trying until it is.
    """

    def __init__(self, scene_model, target: str, interval: float = 1.0, unit_scale: float = 100.0,
                 min_opacity: float = 0.02):
        self.scene_model = scene_model
        self.address = parse_target(target)
        self.target = target
        self.interval = interval
        self.unit_scale = unit_scale
        self.min_opacity = min_opacity
        self.generation = 0
        self.sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="UnrealStreamer", daemon=True)
        self._last_status = ""

    def start(self):
        print(f"Streaming the active anchor to Unreal at {self.target} every {self.interval:g}s")
        self._thread.start()
        return self

    def stop(self, send_final: bool = True):
        """Stops the thread; optionally sends one last frame flagged as frozen."""
        self._stop.set()
        self._thread.join()
        if send_final:
            self.send_once(flags=FLAG_FROZEN, retry_connect=False)
        self._close()

    def _status(self, message: str):
        if message != self._last_status:
            print(f"[unreal_stream] {message}")
            self._last_status = message

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection(self.address, timeout=2.0)
            sock.settimeout(30.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = sock
            self._status(f"connected to {self.target}")
            return True
        except OSError as e:
            self._status(f"waiting for Unreal at {self.target} ({e.strerror or e})")
            return False

    def _close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _snapshot(self):
        model = self.scene_model
        with model.lock:
            anchor = model.active_anchor
            params = anchor.gaussian_params
            if params["xyz"]["val"].shape[0] == 0:
                return None
            tensors = [params[name]["val"].detach().clone()
                       for name in ("xyz", "f_dc", "scaling", "rotation", "opacity")]
            anchor_id = model.anchors.index(anchor) if anchor in model.anchors else len(model.anchors) - 1
            position = anchor.position.detach().clone() if torch.is_tensor(anchor.position) else None
        return tensors, anchor_id, position

    def send_once(self, flags: int = 0, retry_connect: bool = True) -> bool:
        snapshot = self._snapshot()
        if snapshot is None:
            return False
        (xyz, f_dc, scaling, rotation, opacity), anchor_id, position = snapshot
        splats, n, (anchor, bmin, bmax) = pack_gaussians(
            xyz, f_dc, scaling, rotation, opacity, position, self.unit_scale, self.min_opacity)
        if n == 0:
            return False

        if self.sock is None and not (retry_connect and self._connect()):
            return False

        self.generation += 1
        message = build_message(splats, n, anchor_id, self.generation, flags, anchor, bmin, bmax)
        try:
            self.sock.sendall(message)
            return True
        except OSError as e:
            self._status(f"connection to {self.target} lost ({e.strerror or e}), reconnecting")
            self._close()
            return False

    def _run(self):
        while not self._stop.is_set():
            start = time.time()
            try:
                self.send_once()
            except Exception as e:  # never take training down because of the stream
                self._status(f"send failed: {e!r}")
            self._stop.wait(max(0.05, self.interval - (time.time() - start)))


def load_saved_anchor(model_path: str, anchor_index: int = -1):
    """Raw parameters of one saved anchor (the last by default) from a results directory."""
    import json
    from plyfile import PlyData

    with open(os.path.join(model_path, "metadata.json")) as f:
        metadata = json.load(f)
    anchors = metadata["anchors"]
    anchor_index = anchor_index % len(anchors)
    ply = PlyData.read(os.path.join(model_path, "point_clouds", f"anchor_{anchor_index}.ply"))["vertex"]

    def cols(names):
        return torch.tensor(np.stack([np.asarray(ply[n]) for n in names], axis=1), dtype=torch.float32)

    tensors = (
        cols(["x", "y", "z"]),
        cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        cols(["scale_0", "scale_1", "scale_2"]),
        cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
        cols(["opacity"]),
    )
    position = torch.tensor(anchors[anchor_index]["position"], dtype=torch.float32)
    return tensors, anchor_index, position


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream a saved reconstruction to Unreal (RobVR splat plugin)")
    parser.add_argument("-m", "--model_path", required=True, help="results directory with metadata.json and point_clouds/")
    parser.add_argument("--tcp", required=True, metavar="HOST:PORT", help="Unreal network source, e.g. 127.0.0.1:47800")
    parser.add_argument("--anchor", type=int, default=-1, help="which saved anchor to send (default: the last)")
    parser.add_argument("--unit_scale", type=float, default=100.0, help="Unreal cm per scene unit")
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--repeat", type=float, default=0.0,
                        help="resend every N seconds (so Unreal can be started later); 0 sends once")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (xyz, f_dc, scaling, rotation, opacity), anchor_id, position = load_saved_anchor(args.model_path, args.anchor)
    t0 = time.time()
    splats, n, (anchor, bmin, bmax) = pack_gaussians(
        xyz.to(device), f_dc.to(device), scaling.to(device), rotation.to(device), opacity.to(device),
        position, args.unit_scale, args.min_opacity)
    print(f"anchor {anchor_id}: {n} of {xyz.shape[0]} Gaussians kept, {len(splats) / 1e6:.1f} MB, "
          f"packed in {time.time() - t0:.2f}s on {device}")
    print(f"bounds (cm): {[round(v) for v in bmin]} .. {[round(v) for v in bmax]}")

    generation = 0
    sock = None
    while True:
        if sock is None:
            try:
                sock = socket.create_connection(parse_target(args.tcp), timeout=5.0)
                sock.settimeout(None)
            except OSError as e:
                print(f"waiting for Unreal at {args.tcp} ({e.strerror or e})")
                time.sleep(1.0)
                continue
        generation += 1
        try:
            sock.sendall(build_message(splats, n, anchor_id, generation, FLAG_FROZEN, anchor, bmin, bmax))
            print(f"sent generation {generation}")
        except OSError as e:
            print(f"connection lost ({e.strerror or e})")
            sock.close()
            sock = None
            continue
        if args.repeat <= 0:
            break
        time.sleep(args.repeat)
    sock.close()
