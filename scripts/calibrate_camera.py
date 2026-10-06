import argparse
import ctypes
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.append('.')
from utils import fisheye_distort, open_webcam


def get_monitors():
    """Returns the (x, y, w, h) of every monitor, primary first."""
    if sys.platform != "win32":
        return [(0, 0, 1920, 1080)]
    ctypes.windll.user32.SetProcessDPIAware()
    monitors = []

    class RECT(ctypes.Structure):
        _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]

    def callback(hmon, hdc, rect, data):
        r = rect.contents
        monitors.append((r.l, r.t, r.r - r.l, r.b - r.t))
        return 1

    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_double)
    ctypes.windll.user32.EnumDisplayMonitors(None, None, proto(callback), 0)
    monitors.sort(key=lambda m: (m[0], m[1]) != (0, 0))
    return monitors


def make_board(screen_w, screen_h, squares_x, squares_y):
    """ChArUco board sized to fill the screen. Partial views still give usable corners, which matters at the fisheye edges."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), 1.0, 0.75, dictionary)
    square_px = min((screen_w - 40) // squares_x, (screen_h - 40) // squares_y)
    board_img = board.generateImage((square_px * squares_x, square_px * squares_y))
    canvas = np.full((screen_h, screen_w), 255, np.uint8)
    y0, x0 = (screen_h - board_img.shape[0]) // 2, (screen_w - board_img.shape[1]) // 2
    canvas[y0:y0 + board_img.shape[0], x0:x0 + board_img.shape[1]] = board_img
    return board, canvas


def is_new_view(ids, corners, views, min_shift):
    """A view is kept only if the board moved enough in the image compared to every kept view."""
    for prev_ids, prev_corners in views:
        common, a, b = np.intersect1d(ids.ravel(), prev_ids.ravel(), return_indices=True)
        if len(common) < 0.5 * len(ids):
            continue
        shift = np.linalg.norm(corners[a].reshape(-1, 2) - prev_corners[b].reshape(-1, 2), axis=1).mean()
        if shift < min_shift:
            return False
    return True


def init_pose(obj, img, focal, centre):
    """Pose of one view assuming an equidistant fisheye with the given focal."""
    d = img - centre
    r = np.linalg.norm(d, axis=-1, keepdims=True)
    theta = r / focal
    valid = theta[:, 0] < 1.3
    xy = d / np.maximum(r, 1e-8) * np.tan(np.minimum(theta, 1.3))
    ok, rvec, tvec = cv2.solvePnP(obj[valid], xy[valid], np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    return np.concatenate([rvec.ravel(), tvec.ravel()]) if ok else None


def fit(obj_pts, img_pts, size, focal, max_nfev):
    """Joint least squares of fx, fy, cx, cy, k1..k4 and every view pose (OpenCV fisheye model)."""
    centre = np.array(size, np.float64) / 2
    poses = [init_pose(o, i, focal, centre) for o, i in zip(obj_pts, img_pts)]
    keep = [p is not None for p in poses]
    obj_pts = [o for o, k in zip(obj_pts, keep) if k]
    img_pts = [i for i, k in zip(img_pts, keep) if k]
    x0 = np.concatenate([[focal, focal, *centre, 0, 0, 0, 0]] + [p for p in poses if p is not None])
    counts = [len(o) for o in obj_pts]

    def residuals(x):
        K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
        res = []
        for v, (obj, img) in enumerate(zip(obj_pts, img_pts)):
            pose = x[8 + 6 * v: 14 + 6 * v]
            cam = Rotation.from_rotvec(pose[:3]).apply(obj) + pose[3:]
            z = np.maximum(cam[:, 2:], 1e-6)
            res.append((fisheye_distort(cam[:, :2] / z, K, x[4:8]) - img).ravel())
        return np.concatenate(res)

    result = least_squares(residuals, x0, method="lm", max_nfev=max_nfev)
    x = result.x
    K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
    per_point = np.linalg.norm(residuals(x).reshape(-1, 2), axis=-1)
    view_errors = [np.sqrt(np.mean(e ** 2)) for e in np.split(per_point, np.cumsum(counts)[:-1])]
    rms = np.sqrt(np.mean(per_point ** 2))
    return rms, K, x[4:8], obj_pts, img_pts, np.array(view_errors)


def calibrate(board, views, size):
    obj_pts, img_pts = [], []
    for ids, corners in views:
        obj, img = board.matchImagePoints(corners, ids)
        obj_pts.append(obj.reshape(-1, 3).astype(np.float64))
        img_pts.append(img.reshape(-1, 2).astype(np.float64))

    # The focal of wide lenses is poorly known, so start from the guess that fits best
    focals = [size[0] * s for s in (0.25, 0.35, 0.45, 0.6)]
    focal = min(focals, key=lambda f: fit(obj_pts, img_pts, size, f, 5)[0])

    # Drop the worst views (blur, rolling shutter, bad detections) and refit
    for _ in range(3):
        rms, K, D, obj_pts, img_pts, errors = fit(obj_pts, img_pts, size, focal, 200)
        focal = (K[0, 0] + K[1, 1]) / 2
        keep = errors < max(3 * np.median(errors), 1.0)
        if keep.all():
            break
        print(f"Dropping {np.sum(~keep)} outlier views")
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        img_pts = [i for i, k in zip(img_pts, keep) if k]
    return rms, K, D, len(obj_pts)


if __name__ == '__main__':
    """
    Fisheye calibration of a webcam using a ChArUco board shown fullscreen on a monitor.
    Move the camera in front of the board: tilt it, go close and far, and make sure the board
    reaches every corner and edge of the image. Views are captured automatically.
    Keys (in the camera window): space = force capture, c = calibrate, q = quit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help="Webcam index")
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--screen', type=int, default=1, help="Monitor index to show the board on (0 = primary)")
    parser.add_argument('--squares', type=int, nargs=2, default=[14, 8], help="Board squares in x and y")
    parser.add_argument('--num_views', type=int, default=40, help="Calibrate automatically once this many views are captured")
    parser.add_argument('--output', type=str, default="calib/insta360_one_r_4k_1080p.json")
    args = parser.parse_args()

    monitors = get_monitors()
    print("Monitors:", monitors)
    mx, my, mw, mh = monitors[min(args.screen, len(monitors) - 1)]
    board, board_img = make_board(mw, mh, *args.squares)
    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.moveWindow("board", mx, my)
    cv2.resizeWindow("board", mw, mh)
    cv2.setWindowProperty("board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("board", board_img)

    detector = cv2.aruco.CharucoDetector(board)
    cap = open_webcam(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read from the camera. Is another app using it?")
    h, w = frame.shape[:2]
    size = (w, h)

    views = []
    coverage = np.zeros((h, w), np.uint8)
    min_corners = 12
    last_capture = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        vis = frame.copy()
        vis[coverage > 0] = (0.6 * vis[coverage > 0] + 0.4 * np.array([0, 180, 0])).astype(np.uint8)

        force = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            force = True

        if ids is not None and corners is not None and len(ids) == len(corners) and len(ids) >= min_corners:
            ids, corners = ids.reshape(-1, 1).astype(np.int32), corners.reshape(-1, 1, 2).astype(np.float32)
            for x, y in corners.reshape(-1, 2).astype(int):
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var() > 30
            now = time.time()
            if force or (sharp and now - last_capture > 0.4 and is_new_view(ids, corners, views, 0.04 * w)):
                views.append((ids.copy(), corners.copy()))
                last_capture = now
                hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
                cv2.fillConvexPoly(coverage, hull, 255)

        coverage_pct = 100 * np.count_nonzero(coverage) / coverage.size
        cv2.putText(vis, f"views {len(views)}/{args.num_views}  coverage {coverage_pct:.0f}%  (space: capture, c: calibrate, q: quit)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("camera", cv2.resize(vis, (w * 720 // h, 720)))

        if key == ord('c') or len(views) >= args.num_views:
            if len(views) < 10:
                print("Need at least 10 views")
                continue
            rms, K, D, n_used = calibrate(board, views, size)
            print(f"RMS reprojection error: {rms:.3f} px over {n_used} views")
            print("K =\n", K, "\nD =", D.ravel())
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({
                    "model": "fisheye", "width": w, "height": h,
                    "K": K.tolist(), "D": D.ravel().tolist(),
                    "rms": float(rms), "num_views": n_used,
                }, f, indent=2)
            print("Saved to", args.output)
            break

    cv2.destroyAllWindows()
    cap.release()
