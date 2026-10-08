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

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from webviewer.webviewer import WebViewer
from utils import BlurDetector, align_mean_up_fwd, increment_runtime
from unreal_stream import UnrealStreamer

if __name__ == "__main__":
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    args = get_args()

    # Initialize dataloader
    # Live streams train asynchronously while frames keep coming, video files are processed frame by frame
    is_live = "://" in args.source_path
    is_video = os.path.isfile(args.source_path) and args.source_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    if is_live or is_video:
        dataset = StreamDataset(args.source_path, args.downsampling, undistort=args.undistort,
                                undistort_fov_scale=args.undistort_fov_scale, equirect=args.equirect)
        is_stream = True
    else:
        dataset = ImageDataset(args)
        is_stream = False
    height, width = dataset.get_image_size()
    # Undistorted and equirectangular streams have a known focal, so use it instead of estimating one
    if is_stream and dataset.focal is not None and args.init_focal <= 0:
        args.init_focal = dataset.focal
        args.fix_focal = True

    # Initialize other modules
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)
    min_displacement = max(args.min_displacement * width, 30)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, 2 * max_error, args
    )
    focal = pose_initializer.f_init
    dense_extractor = DenseExtractor(width, height)
    depth_estimator = MonoDepthEstimator(width, height, args.depth_preprocessing)
    scene_model = SceneModel(width, height, args, matcher)
    detector = Detector(args.num_kpts, width, height)
    blur_detector = BlurDetector(args.blur_ratio)
    n_blurry_skips = 0

    # Initialize the viewer
    if args.viewer_mode in ["server", "local"]:
        # Imported here so the other viewer modes work without the native imgui libraries
        from gaussianviewer import GaussianViewer
        from graphdecoviewer.types import ViewerMode
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")

        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    # Stream the reconstruction to Unreal while it is built
    unreal_streamer = None
    if args.unreal_stream:
        unreal_streamer = UnrealStreamer(
            scene_model, args.unreal_stream, args.unreal_stream_interval,
            args.unreal_unit_scale, args.unreal_min_opacity,
        ).start()

    # 360 streams: the views of a frame share one centre, view 0 holds the pose and the others a fixed rotation of it.
    # All views are used for tracking, as one direction alone can easily be textureless.
    is_rig = is_stream and args.equirect is not None and args.equirect[2] > 1
    n_mvs_cams = args.num_prev_keyframes_miniba_incr
    # Each 360 frame adds one keyframe per view, so train proportionally more and favour all of the new views
    n_iters_per_frame = args.num_iterations
    if is_rig:
        n_iters_per_frame *= int(args.equirect[2])
        scene_model.num_new_keyframes = int(args.equirect[2])

    def add_rig_views(parent, rig_views, rig_descs, f):
        """Adds the other views of parent's 360 frame as keyframes tied to parent. Returns them."""
        keyframes = []
        for view_id, ((view_image, rig_rot), view_desc_kpts) in enumerate(zip(rig_views, rig_descs), 1):
            keyframe = Keyframe(
                view_image,
                {"is_test": False, "view_id": view_id, **({"name": f"{parent.info['name']}_{view_id}"} if "name" in parent.info else {})},
                view_desc_kpts,
                None,
                len(scene_model.keyframes),
                f,
                dense_extractor,
                depth_estimator,
                triangulator,
                args,
                rig_parent=parent,
                rig_rot=rig_rot,
            )
            scene_model.add_keyframe(keyframe)
            keyframes.append(keyframe)
        return keyframes

    def match_rig_view(keyframe):
        """Matches a rig view with nearby views of other frames, which have the baseline to triangulate its keypoints."""
        for other in scene_model.get_rig_neighbours(keyframe, n_mvs_cams):
            matcher(keyframe.desc_kpts, other.desc_kpts, remove_outliers=True, update_kpts_flag="all",
                    kID=keyframe.index, kID_other=other.index)

    def track_rig(image, desc_kpts, rig_views, rig_descs, is_test):
        """
        Estimates the pose of every view of a 360 frame and keeps the one with the most inliers.
        Returns view 0's pose, or None if no view could be registered.
        """
        base_index = len(scene_model.keyframes)
        views = [(image, desc_kpts, None)] + [(v, d, rot) for (v, rot), d in zip(rig_views, rig_descs)]
        best_Rt, best_inliers = None, 0
        # Select (and re-triangulate) the neighbours of every view before matching any of them, as matches with
        # the not yet added views would make the re-triangulation look up keyframes that do not exist
        all_prev_keyframes = [scene_model.get_prev_keyframes(n_mvs_cams, True, d) for _, d, _ in views]
        for k, ((view_image, view_desc_kpts, rig_rot), prev_keyframes) in enumerate(zip(views, all_prev_keyframes)):
            # Indices match the order in which the views are added as keyframes
            Rt = pose_initializer.initialize_incremental(
                prev_keyframes, view_desc_kpts, base_index + k, is_test, view_image, verbose=False
            )
            if Rt is not None and pose_initializer.last_num_inliers > best_inliers:
                best_inliers = pose_initializer.last_num_inliers
                if rig_rot is not None:
                    # Back to view 0: R_k = rot R_0 and t_k = rot t_0
                    Rt = Rt.clone()
                    Rt[:3, :3] = rig_rot.T @ Rt[:3, :3]
                    Rt[:3, 3] = rig_rot.T @ Rt[:3, 3]
                best_Rt = Rt
        if best_Rt is None:
            print("Too few inliers for pose initialization in every view")
            # The frame is dropped, so its indices will be reused by the next one
            for keyframe in scene_model.keyframes:
                for k in range(len(views)):
                    keyframe.desc_kpts.matches.pop(base_index + k, None)
        return best_Rt

    def add_new_gaussians(keyframe):
        if is_rig:
            scene_model.add_new_gaussians(keyframe.index, scene_model.get_rig_neighbours(keyframe, n_mvs_cams))
        else:
            scene_model.add_new_gaussians(keyframe.index)

    n_active_keyframes = 0
    n_keyframes = 0
    needs_reboot = False
    bootstrap_keyframe_dicts = []
    bootstrap_desc_kpts = []

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}

    ## Scene reconstruction
    print(f"Starting reconstruction for {args.source_path}")
    pbar = tqdm(range(0, len(dataset)))
    reconstruction_start_time = time.time()
    for frameID in pbar:
        start_time = time.time()

        if args.viewer_mode == "web":
            viewer.trainer_state = "running"

            # Paused
            while viewer.state == "stop":
                pbar.set_postfix_str(
                    "\033[31mPaused. Press the Start button in the webviewer\033[0m"
                )
                time.sleep(0.1)
            
            # Finish reconstruction
            if viewer.state == "finish":
                viewer.trainer_state = "finish"
                break
        
        if n_keyframes == 0:
            image, info = dataset.getnext()
            if image is None:
                break
            prev_desc_kpts = detector(image)
            prev_rig_descs = [detector(view) for view, _ in info.get("rig_views", [])]
            info["rig_descs"] = prev_rig_descs
            bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            bootstrap_desc_kpts = [prev_desc_kpts]
            n_keyframes += 1
            continue

        image, info = dataset.getnext()
        if image is None:
            break
        desc_kpts = detector(image)
        # Match features between the previous and current frame
        curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
        # Determine if we should add a keyframe based on the matches
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        n_matches = len(curr_prev_matches.kpts)
        if is_rig:
            # Pool the matches of all views of the 360 frame
            rig_descs = [detector(view) for view, _ in info["rig_views"]]
            info["rig_descs"] = rig_descs
            for view_desc_kpts, prev_view_desc_kpts in zip(rig_descs, prev_rig_descs):
                view_matches = matcher(view_desc_kpts, prev_view_desc_kpts)
                dist = torch.cat([dist, torch.norm(view_matches.kpts - view_matches.kpts_other, dim=-1)])
                n_matches += len(view_matches.kpts)
        should_add_keyframe = (
            len(dist) > 0
            and dist.median() > min_displacement
            and n_matches > args.min_num_inliers
        )
        # Wait for a sharper frame instead of adding a blurry keyframe, but not for too long to keep tracking
        is_blurry = blur_detector(image)
        if should_add_keyframe and is_blurry and n_blurry_skips < args.max_blurry_skips:
            should_add_keyframe = False
            n_blurry_skips += 1
        elif should_add_keyframe:
            n_blurry_skips = 0
        # Always add test frames so we estimate their poses
        should_add_keyframe |= info["is_test"]
        increment_runtime(runtimes["Load"], start_time)

        if should_add_keyframe:
            ## Bootstrap
            # Accumulate keyframes for pose initialization
            if n_keyframes < args.num_keyframes_miniba_bootstrap:
                bootstrap_keyframe_dicts.append({"image": image, "info": info})
                bootstrap_desc_kpts.append(desc_kpts)

            if n_keyframes == args.num_keyframes_miniba_bootstrap - 1:
                start_time = time.time()
                Rts, f, _ = pose_initializer.initialize_bootstrap(bootstrap_desc_kpts)
                focal = f.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)
                bootstrap_rig_views = []
                for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
                    zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                ):
                    start_time = time.time()
                    bootstrap_rig_views.append(
                        (keyframe_dict["info"].pop("rig_views", []), keyframe_dict["info"].pop("rig_descs", []))
                    )
                    if args.use_colmap_poses:
                        Rt = keyframe_dict["info"]["Rt"]
                        f = keyframe_dict["info"]["focal"]
                    keyframe = Keyframe(
                        keyframe_dict["image"],
                        keyframe_dict["info"],
                        desc_kpts,
                        Rt,
                        index,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe, f)
                    increment_runtime(runtimes["Add"], start_time)
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                prev_keyframe = keyframe
                new_keyframes = scene_model.keyframes[: args.num_keyframes_miniba_bootstrap]
                if is_rig:
                    # Add every rig view first so they can be matched with the views of all bootstrap frames
                    start_time = time.time()
                    rig_keyframes = []
                    for parent, (rig_views, rig_descs) in zip(list(new_keyframes), bootstrap_rig_views):
                        rig_keyframes += add_rig_views(parent, rig_views, rig_descs, f)
                    for keyframe in rig_keyframes:
                        match_rig_view(keyframe)
                    new_keyframes = new_keyframes + rig_keyframes
                    increment_runtime(runtimes["Add"], start_time)
                for keyframe in new_keyframes:
                    start_time = time.time()
                    add_new_gaussians(keyframe)
                    increment_runtime(runtimes["Init"], start_time)
                start_time = time.time()
                # Run initial optimization on the bootstrap keyframes
                # If streaming, run async optimization until the next keyframe is added
                if is_live:
                    scene_model.optimize_async(n_iters_per_frame)
                else:
                    scene_model.optimization_loop(n_iters_per_frame)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes

            ## Reboot
            if (
                args.enable_reboot
                and not is_rig
                and scene_model.approx_cam_centres is not None
                and len(scene_model.anchors)
            ):
                # Check if the camera baseline is a lot smaller or larger than expected
                last_centers = scene_model.approx_cam_centres[-20:]
                rel_dist = torch.norm(
                    last_centers[1:] - last_centers[:-1], dim=-1
                ).mean()
                needs_reboot = (
                    rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
                ) and n_keyframes - last_reboot > 50
            if needs_reboot:
                # Reboot: run mini BA on the last 8 keyframes
                bs_kfs = scene_model.keyframes[-8:]
                bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
                in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
                Rts, _, final_residual = pose_initializer.initialize_bootstrap(
                    bootstrap_desc_kpts, rebooting=True
                )
                # Check if the reboot succeeded
                if final_residual < max_error * 0.5:
                    Rts = align_mean_up_fwd(Rts, in_Rts)
                    for Rt, keyframe in zip(Rts, bs_kfs):
                        keyframe.set_Rt(Rt)
                    # Reset the scene model and reinitialize the gaussians
                    scene_model.reset()
                    for i in range(3, 0, -1):
                        scene_model.add_new_gaussians(-i)
                    for _ in range(3 * args.num_iterations):
                        scene_model.optimization_step()
                    needs_reboot = False
                    last_reboot = n_keyframes

            ## Incremental reconstruction
            # Incremental pose initialization
            if n_keyframes >= args.num_keyframes_miniba_bootstrap:
                start_time = time.time()
                if is_rig:
                    Rt = track_rig(image, desc_kpts, info["rig_views"], info["rig_descs"], info["is_test"])
                else:
                    prev_keyframes = scene_model.get_prev_keyframes(
                        args.num_prev_keyframes_miniba_incr, True, desc_kpts
                    )
                    increment_runtime(runtimes["tri"], start_time)
                    start_time = time.time()
                    Rt = pose_initializer.initialize_incremental(
                        prev_keyframes, desc_kpts, len(scene_model.keyframes), info["is_test"], image
                    )
                increment_runtime(runtimes["BAI"], start_time)
                start_time = time.time()
                if Rt is not None:
                    if args.use_colmap_poses:
                        Rt = info["Rt"]
                    rig_views = info.pop("rig_views", [])
                    rig_descs = info.pop("rig_descs", [])
                    keyframe = Keyframe(
                        image,
                        info,
                        desc_kpts,
                        Rt,
                        len(scene_model.keyframes),
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe)
                    prev_keyframe = keyframe
                    new_keyframes = [keyframe]
                    if is_rig:
                        rig_keyframes = add_rig_views(keyframe, rig_views, rig_descs, f)
                        # View 0 lost its matches if it was not the view the frame was registered with
                        for rig_keyframe in [keyframe] + rig_keyframes:
                            match_rig_view(rig_keyframe)
                        new_keyframes += rig_keyframes
                    increment_runtime(runtimes["Add"], start_time)
                    # Gaussian initialization
                    start_time = time.time()
                    for new_keyframe in new_keyframes:
                        add_new_gaussians(new_keyframe)
                    increment_runtime(runtimes["Init"], start_time)
                    start_time = time.time()
                    # If streaming, run async optimization until the next keyframe is added
                    if is_live:
                        scene_model.optimize_async(n_iters_per_frame)
                    else:
                        scene_model.optimization_loop(n_iters_per_frame)
                    increment_runtime(runtimes["Opt"], start_time)
                else:
                    should_add_keyframe = False

        if should_add_keyframe:
            ## Check if anchor creation is needed based on the primitives' size 
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            increment_runtime(runtimes["anc"], start_time)

            n_keyframes += 1
            if not info["is_test"]:
                prev_desc_kpts = desc_kpts
                if is_rig:
                    prev_rig_descs = rig_descs

            ## Intermediate evaluation
            if (
                n_keyframes % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0 or args.eval_poses)
            ):
                metrics = scene_model.evaluate(args.eval_poses)

            ## Save intermediate model
            if (
                frameID % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{frameID:05d}")
                )

            ## Display optimization progress and metrics
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                        ]
            bar_postfix += [
                f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{n_keyframes}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
            ]
            if args.densify_grad_threshold > 0:
                bar_postfix += [f"\033[36mCloned:{scene_model.n_cloned},Split:{scene_model.n_split}\033[0m"]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    reconstruction_time = time.time() - reconstruction_start_time

    # Set to inference mode so that the model can be rendered properly
    scene_model.enable_inference_mode()

    # Send the finished reconstruction once more, flagged as final
    if unreal_streamer is not None:
        unreal_streamer.stop(send_final=True)

    # Save the model and metrics
    print("Saving the reconstruction to:", args.model_path)
    metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
    print(
        ", ".join(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
            for metric, value in metrics.items()
        )
    )

    # Fine tuning after initial reconstruction
    if len(args.save_at_finetune_epoch) > 0:
        finetune_epochs = max(args.save_at_finetune_epoch)
        torch.cuda.empty_cache()
        scene_model.inference_mode = False
        pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
        for epoch in pbar:
            # Run one epoch of fine-tuning
            epoch_start_time = time.time()
            scene_model.finetune_epoch()
            epoch_time = time.time() - epoch_start_time
            reconstruction_time += epoch_time
            # Save the model and metrics
            if epoch + 1 in args.save_at_finetune_epoch:
                torch.cuda.empty_cache()
                scene_model.inference_mode = True
                metrics = scene_model.save(
                    os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                )
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                pbar.set_postfix_str(",".join(bar_postfix))
                scene_model.inference_mode = False
                torch.cuda.empty_cache()
                
        # Set to inference mode so that the model can be rendered properly
        scene_model.inference_mode = True

    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)
