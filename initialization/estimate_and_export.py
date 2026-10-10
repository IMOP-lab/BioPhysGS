# BioPhysGS release: explicit input paths and module-relative imports; solver settings unchanged.
# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'droid_slam'))

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import time
import argparse
import time
import signal

from droid import Droid
import video_calib




def print_intrinsics(_intr_est, size_factor, args):

    # rescale intrinsics
    intr_est = _intr_est.copy()
    intr_est[0:4:2] /= size_factor[0]
    intr_est[1:4:2] /= size_factor[1]

    # recover initial intrinsics
    example_imfile = sorted(os.listdir(args.imagedir))[0]
    image = cv2.imread(os.path.join(args.imagedir, example_imfile))
    h0, w0, _ = image.shape
    intr_initial = np.array([(w0+h0)/2, (w0+h0)/2, w0/2, h0/2])

    # print self-calibration result
    print("#" * 32)
    print("Initial intrinsics:")
    print("fx = {}, fy = {}, ppx = {}, ppy = {}".format(*intr_initial))

    print("Estimated intrinsics:")

    if args.camera_model == "pinhole" or args.camera_model == "focal":
        print("fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}".format(*intr_est))
    else:
        print("fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}, xi = {:.3f}".format(*intr_est))


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)












import numpy as np
import cv2
import torch
import os

def image_stream(imagedir, image_size, calib, stride, num_images,
                 camera_model="pinhole", output_dir=None):
    """ image generator """

    # 如果image_size为None，则自动读取第一张图片的尺寸
    if image_size is None:
        example_imfile = sorted(os.listdir(imagedir))[0]
        example_image = cv2.imread(os.path.join(imagedir, example_imfile))
        if example_image is None:
            raise ValueError(f"Failed to load example image: {example_imfile}")
        h0, w0, _ = example_image.shape
        # 确保尺寸是8的倍数
        wd = w0 - w0 % 8
        ht = h0 - h0 % 8
    else:
        wd, ht = image_size

    if len(calib) > 0:
        calib = np.loadtxt(calib, delimiter=" ")
    else:
        example_imfile = sorted(os.listdir(imagedir))[0]
        image = cv2.imread(os.path.join(imagedir, example_imfile))
        if image is None:
            raise ValueError(f"Failed to load example image: {example_imfile}")
        h0, w0, _ = image.shape
        calib = np.array([(w0+h0)/2, (w0+h0)/2, w0/2, h0/2, 0, 0, 0, 0, 0])

    fx, fy, cx, cy = calib[:4]

    # Ensure camera matrix is float32
    K = np.eye(3, dtype=np.float32)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # Ensure distortion coefficients are float32
    dist_coeffs = np.array(calib[4:], dtype=np.float32) if len(calib) > 4 else np.zeros(5, dtype=np.float32)

    image_list = sorted(os.listdir(imagedir))[:num_images:stride]

    # 创建输出目录
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if image is None:
            print(f"Warning: Failed to load image {imfile}")
            continue

        # Ensure image is uint8 and RGB
        if image.dtype != np.uint8:
            print(f"Warning: Image {imfile} has dtype {image.dtype}, converting to uint8")
            image = image.astype(np.uint8)

        if len(calib) > 4:
            # Create output image with same shape and dtype
            undistorted_image = np.zeros_like(image, dtype=np.uint8)
            # Debug: Print shapes and types
            # print(f"Image {imfile}: shape={image.shape}, dtype={image.dtype}")
            # print(f"K=\n{K}, dist_coeffs={dist_coeffs}")
            # Perform undistortion with newCameraMatrix=K
            cv2.undistort(image, K, dist_coeffs, undistorted_image, K)
            image = undistorted_image

        h0, w0, _ = image.shape

        if image_size is None:
            # 直接裁剪到8的倍数
            image = image[:ht, :wd]
            h1, w1 = ht, wd
        else:
            # 原有的缩放逻辑
            h1 = int(h0 * np.sqrt((ht * wd) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((ht * wd) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            h1, w1 = image.shape[:2]

        # 保存处理后的图片到输出目录
        if output_dir is not None:
            output_path = os.path.join(output_dir, imfile)
            cv2.imwrite(output_path, image)

        image = torch.as_tensor(image).permute(2, 0, 1)

        if camera_model == "pinhole" or camera_model == "focal":
            intrinsics = torch.as_tensor([fx, fy, cx, cy])
        elif camera_model == "mei":
            intrinsics = torch.as_tensor([fx, fy, cx, cy, 0])
        else:
            raise Exception("Camera model not implemented! Choose either \
                             pinhole or mei model.")

        size_factor = [(w1 / w0), (h1 / h0)]
        intrinsics[0::2] *= size_factor[0]
        intrinsics[1::2] *= size_factor[1]

        # Always retain the original filename correspondence. The working
        # image does not need to be written to disk for this metadata.
        yield t, image[None], intrinsics, size_factor, imfile




def get_point_cloud_from_droid(droid):
    """
    从 droid.video 中提取点云数据和 2D-3D 对应关系

    Args:
        droid: Droid 对象，包含重建数据

    Returns:
        points3D: 3D 点列表，格式为 [(point3D_id, x, y, z, r, g, b, error), ...]
        tracks: 2D-3D 对应关系列表，格式为 [(image_id, point2D_idx, point3D_id), ...]
        success: 是否成功提取点云
    """
    try:
        # 检查点云数据
        if not hasattr(droid.video, 'points') or droid.video.points is None or len(droid.video.points) == 0:
            print("错误：点云数据为空或未找到")
            return [], [], False
        point_data = droid.video.points.cpu().numpy()

        # 检查颜色数据
        colors = droid.video.colors.cpu().numpy() if hasattr(droid.video, 'colors') and droid.video.colors is not None else None

        # 检查 2D-3D 对应关系
        tracks = droid.video.tracks.cpu().numpy() if hasattr(droid.video, 'tracks') and droid.video.tracks is not None else None

        # 构建 3D 点
        points3D = [(i + 1, *point[:3], *(colors[i] * 255 if colors is not None else [0, 0, 0]), 0.0) for i, point in enumerate(point_data)]

        # 构建 tracks
        tracks_out = [(int(t[0]), int(t[1]), int(t[2])) for t in tracks] if tracks is not None else []

        print(f"提取 {len(points3D)} 个 3D 点，{len(tracks_out)} 个 tracks")
        return points3D, tracks_out, True

    except Exception as e:
        print(f"提取点云失败：{e}")
        return [], [], False






import os
import numpy as np
from pathlib import Path
import cv2

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def save_reconstruction(droid, reconstruction_path, traj_est, frame_names=None, camera_model="pinhole"):
    """
    Save DROID-SLAM reconstruction results in COLMAP format.

    Args:
        droid: Droid object containing reconstruction data.
        reconstruction_path: Path to save the COLMAP reconstruction.
        camera_model: Camera model used (e.g., "pinhole", "mei").
    """
    # Extract reconstruction data
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    # images = droid.video.images[:t].cpu().numpy()  # Shape: [N, 3, H, W]
    images = droid.video.images.cpu().numpy()  # Shape: [N, 3, H, W]

    disps = droid.video.disps_up[:t].cpu().numpy()  # Shape: [N, H, W]
    # poses = droid.video.poses[:t].cpu().numpy()    # Shape: [N, 7]

    poses = traj_est    # Shape: [N, 7]

    # poses = traj_est    # Shape: [N, 7]
    print("traj_est",traj_est.shape)

    intrinsics = droid.video.intrinsics[:t].cpu().numpy()  # Shape: [N, 4 or 5]

    # Debug: Check shapes
    print(f"Poses shape: {poses.shape}")
    print(f"Poses sample: {poses[:2]}")

    # Create output directory structure for COLMAP
    output_dir = Path(reconstruction_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reproducible GT-pose reprojection payload. tstamp maps each retained
    # keyframe to the input-frame index even when DROID removes frames.
    keyframe_tstamps = droid.video.tstamp[:droid.video.counter.value].cpu().numpy().astype(np.int64)
    np.save(output_dir / "denseflow_keyframe_tstamps.npy", keyframe_tstamps)
    if hasattr(droid.video, "local_points"):
        np.savez_compressed(output_dir / "denseflow_local_points.npz",
                           xyz=droid.video.local_points,
                           rgb=droid.video.colors.detach().cpu().numpy(),
                           keyframe=droid.video.point_keyframes)
    # Preserve the raw DROID trajectory and exact filename correspondence for
    # reproducible Sim(3) alignment to an external GT trajectory.
    np.save(output_dir / "denseflow_trajectory_raw.npy", traj_est)
    if frame_names is not None:
        (output_dir / "denseflow_frame_names.txt").write_text(
            "\n".join(frame_names) + "\n", encoding="utf-8"
        )
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save cameras.txt
    fx, fy, cx, cy = intrinsics[0][:4] * 8.0
    if camera_model == "pinhole" or camera_model == "focal":
        camera_model_colmap = "PINHOLE"
        h, w = images.shape[2], images.shape[3]
        camera_params = f"{fx} {fy} {cx} {cy}"
    elif camera_model == "mei":
        camera_model_colmap = "PINHOLE"
        h, w = images.shape[2], images.shape[3]
        camera_params = f"{fx} {fy} {cx} {cy}"
        print("Warning: Mei camera model not fully supported in COLMAP. Using PINHOLE model.")
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")

    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 {camera_model_colmap} {w} {h} {camera_params}\n")



    # 2. Save images.txt
    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines with data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        all = poses.shape[0]
        print("all", all)
        for i in range(all):
            # 使用原始文件名或序列号
            if frame_names is not None and i < len(frame_names):
                img_name = frame_names[i]
            else:
                img_name = f"{i:06d}.png"

            # Check pose validity
            pose = poses[i]
            if pose.shape != (7,):
                print(f"Warning: Invalid pose at frame {i}, shape: {pose.shape}. Skipping.")
                continue

            # Extract translation and quaternion (rx, ry, rz, rw)
            tx, ty, tz, rx, ry, rz, rw = pose

            # Construct 4x4 world-to-camera transformation matrix
            R = quaternion_to_rotation_matrix([rw, rx, ry, rz])  # Note the order: rw, rx, ry, rz
            t = np.array([tx, ty, tz])
            pose_w2c = np.eye(4)
            pose_w2c[:3, :3] = R
            pose_w2c[:3, 3] = t

            # Convert to camera-to-world
            try:
                pose_c2w = np.linalg.inv(pose_w2c)
            except np.linalg.LinAlgError as e:
                print(f"Error: Failed to invert pose at frame {i}: {e}")
                continue

            # Extract rotation and translation for COLMAP
            R_c2w = pose_c2w[:3, :3]
            t_c2w = pose_c2w[:3, 3]

            # Convert rotation matrix to quaternion
            trace = np.trace(R_c2w)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (R_c2w[2, 1] - R_c2w[1, 2]) / s
                qy = (R_c2w[0, 2] - R_c2w[2, 0]) / s
                qz = (R_c2w[1, 0] - R_c2w[0, 1]) / s
            else:
                if R_c2w[0, 0] > R_c2w[1, 1] and R_c2w[0, 0] > R_c2w[2, 2]:
                    s = np.sqrt(1.0 + R_c2w[0, 0] - R_c2w[1, 1] - R_c2w[2, 2]) * 2
                    qw = (R_c2w[2, 1] - R_c2w[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R_c2w[0, 1] + R_c2w[1, 0]) / s
                    qz = (R_c2w[0, 2] + R_c2w[2, 0]) / s
                elif R_c2w[1, 1] > R_c2w[2, 2]:
                    s = np.sqrt(1.0 + R_c2w[1, 1] - R_c2w[0, 0] - R_c2w[2, 2]) * 2
                    qw = (R_c2w[0, 2] - R_c2w[2, 0]) / s
                    qx = (R_c2w[0, 1] + R_c2w[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R_c2w[1, 2] + R_c2w[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R_c2w[2, 2] - R_c2w[0, 0] - R_c2w[1, 1]) * 2
                    qw = (R_c2w[1, 0] - R_c2w[0, 1]) / s
                    qx = (R_c2w[0, 2] + R_c2w[2, 0]) / s
                    qy = (R_c2w[1, 2] + R_c2w[2, 1]) / s
                    qz = 0.25 * s

            # Write to images.txt
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t_c2w[0]} {t_c2w[1]} {t_c2w[2]} 1 {img_name}\n")
            f.write("\n")  # No 2D-3D correspondences yet


# 2. 尝试提取现有点云
    points3D, tracks, success = get_point_cloud_from_droid(droid)




# 4. 保存 points3D.txt. faster

    # import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm


    def process_chunk(chunk):
        lines = []
        for pid, x, y, z, r, g, b, error in tqdm(chunk, desc="Processing chunk"):
            # 将 pid 和 r, g, b 格式化为整数
            lines.append(f"{int(pid)} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error}\n")
        return lines

    # 分块并行处理
    chunks = np.array_split(points3D, 8)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_chunk, chunks),
                            total=len(chunks), desc="Processing chunks"))

    # 写入文件
    with open(output_dir / "points3D.txt", "w", buffering=8192) as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for lines in tqdm(results, desc="Writing to file"):
            f.writelines(lines)

    print(f"Reconstruction saved successfully to: {output_dir}")
    print(f"Files created: cameras.txt, images.txt, points3D.txt")
    print(f"Total 3D points: {len(points3D)}")

import os
import time
import torch
import threading
import signal
import cv2
import numpy as np
from tqdm import tqdm
from lietorch import SE3
import droid_backends
# from save_reconstruction import save_reconstruction  # 假设 save_reconstruction 在单独文件中
# from visualization import show_image, print_intrinsics, image_stream, video_calib, Droid  # 假设这些函数存在

import open3d as o3d




def signal_handler(sig, frame, droid, reconstruction_path):
    print("程序中断，尝试保存已有数据...")
    if droid is not None and reconstruction_path is not None:
        save_reconstruction(droid, reconstruction_path)
    exit(0)










def generate_point_cloud(video, device="cuda:0", filter_thresh=0.001):
    """Generate point cloud and tracks for all frames from video data."""
    torch.cuda.set_device(device)
    print(f"Starting point cloud generation on device: {device}")

    video.points = None
    video.colors = None
    video.tracks = None

    with torch.no_grad():
        with video.get_lock():
            t = video.counter.value
            # t = 1201

            # 处理所有帧，而不是只处理 dirty_index
            all_frames = torch.arange(t, device="cuda", dtype=torch.long)
            print(f"Debug: Processing all frames, total frames = {t}")
            if t == 0:
                print("Debug: No frames available, skipping point cloud generation.")
                return
            print("video.disps",video.disps.shape)

            poses = video.poses[:t]
            disps = video.disps[:t]
            images = video.images[:t]
            images = images.cpu()[:, [2,1,0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0], video.model_id).cpu()
            # print(dir(droid_backends))

            print(f"Debug: points shape = {points.shape}, disps shape = {disps.shape}")

            thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], all_frames, thresh, video.model_id)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 1) & (disps > 0.1 * disps.mean(dim=[1,2], keepdim=True)))
            # masks = ((disps > 0.1 * disps.mean(dim=[1,2], keepdim=True)))

            print(f"Debug: masks valid points = {[torch.sum(m).item() for m in masks]}")

            all_points = []
            all_colors = []
            all_local_points = []
            all_point_keyframes = []
            # all_tracks = []
            total_points = 0  # 累计点数，用于 point3D_indices

            # 处理所有帧
            for i in range(t):
                ix = all_frames[i].item()
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask.cpu().numpy()].cpu().numpy()
                print(f"Debug: frame {ix}, valid points = {len(pts)}")

                if len(pts) > 0:
                    all_points.append(pts)
                    all_colors.append(clr)
                    local = SE3(poses[i:i+1]) * torch.from_numpy(pts).to(poses.device)
                    all_local_points.append(local.cpu().numpy())
                    all_point_keyframes.append(np.full(len(pts), i, np.int32))
                    h, w = images.shape[1:3]
                    u, v = np.meshgrid(np.arange(w), np.arange(h))
                    u = u.reshape(-1)[mask.cpu().numpy()]
                    v = v.reshape(-1)[mask.cpu().numpy()]
                    point2D_indices = np.arange(len(pts))
                    point3D_indices = point2D_indices + total_points
                    # 生成 tracks，包含 frame_id, point2D_id, point3D_id
                    # tracks = np.stack([np.full_like(point2D_indices, ix + 1), point2D_indices, point3D_indices], axis=1)
                    # all_tracks.append(tracks)
                    total_points += len(pts)
                    # print(f"Debug: frame {ix}, tracks shape = {tracks.shape}, point3D_indices = {point3D_indices[:5]}")

                else:
                    # 即使没有点，也记录空 tracks 以保持帧对齐
                    print(f"Debug: frame {ix}, no valid points, appending empty tracks")
                    # tracks = np.empty((0, 3), dtype=np.int64)  # 空 tracks
                    # all_tracks.append(tracks)



            if all_points:
                video.points = torch.from_numpy(np.concatenate(all_points, axis=0)).to(device)
                video.colors = torch.from_numpy(np.concatenate(all_colors, axis=0)).to(device)
                video.local_points = np.concatenate(all_local_points, axis=0)
                video.point_keyframes = np.concatenate(all_point_keyframes, axis=0)
                # if all_tracks:
                #     video.tracks = torch.from_numpy(np.concatenate(all_tracks, axis=0)).to(device)
                #     print(f"Debug: Stored video.points shape = {video.points.shape}, "
                #           f"video.tracks shape = {video.tracks.shape}")
                # else:
                #     print("Debug: No tracks generated.")
            else:
                print("Debug: No valid points generated.")
                # video.tracks = torch.empty((0, 3), dtype=torch.int64).to(device)  # 空 tracks







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, required=True, help="path to image directory")
    parser.add_argument("--calib", default="", type=str, help="path to calibration file if available")
    parser.add_argument(
        "--opt_intr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="optimize camera intrinsics (use --no-opt_intr to keep calibration fixed)",
    )
    parser.add_argument("--camera_model", default="pinhole", type=str, help="pinhole or mei or focal")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")
    parser.add_argument("--num_images", default=1201, type=int, help="using only the first n images")
    parser.add_argument("--video_calib", action="store_true", help="create video with intrinsics overlay")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "checkpoints/droidcalib.pth"))
    parser.add_argument("--buffer", type=int, default=1202)
    parser.add_argument("--image_size_target", default=None, nargs=2, type=int,
                        metavar=("WIDTH", "HEIGHT"),
                        help="image width and height; reduce to make inference faster. If None, use original image size.")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    # parser.add_argument("--keyframe_thresh", type=float, default=18.0, help="threshold to create a new keyframe")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")

    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", default=None, help="path to saved reconstruction (if None, will be derived from imagedir)")
    args = parser.parse_args()

    # 如果reconstruction_path没有指定，从imagedir自动推导
    if args.reconstruction_path is None:
        import os
        # 从imagedir路径推导reconstruction_path
        # 例如: /path/to/dataset/s1/images/ -> /path/to/dataset/s1/sparse/0
        imagedir_parent = os.path.dirname(args.imagedir.rstrip('/'))
        args.reconstruction_path = os.path.join(imagedir_parent, 'sparse', '0')

    args.stereo = False
    args.disable_vis = not args.visualize
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    start_time = time.time()

    if args.reconstruction_path is not None:
        args.upsample = True
    # args.upsample = True

    # 注册中断信号处理
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, droid, args.reconstruction_path))

    # 创建输出图片目录
    # Never write resized working images back into the source dataset. Doing
    # so silently corrupts the calibrated 3DGS inputs when image_size_target
    # differs from the formal reconstruction resolution.
    output_images_dir = None

    tstamps = []
    frame_names = []  # 存储原始文件名
    for (t, image, intrinsics, sf, imfile) in tqdm(image_stream(args.imagedir, args.image_size_target,
                                                        args.calib, args.stride, args.num_images,
                                                        args.camera_model, output_images_dir)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, intrinsics=intrinsics)
        frame_names.append(imfile)  # 记录原始文件名

        if args.video_calib:
            intr_t = droid.video.intrinsics[0].clone()
            intr_t[:4] *= 8.0
            intr_t = video_calib.rescale_intr(intr_t, size_factor=sf)
            video_calib.save_image(t, image, intr_t, (time.time() - start_time))

    # TrajectoryFiller follows DROID's original four-field stream contract;
    # filenames are only needed by this exporter's first pass.
    filler_stream = (
        item[:4] for item in image_stream(
            args.imagedir, args.image_size_target, args.calib, args.stride,
            args.num_images, args.camera_model
        )
    )
    traj_est, intr_est = droid.terminate(filler_stream)

    # traj_est, intr_est = droid.terminate()

    print_intrinsics(intr_est, sf, args)

    if args.video_calib:
        intr_t = droid.video.intrinsics[0].clone()
        intr_t[:4] *= 8.0
        intr_t = video_calib.rescale_intr(intr_t, size_factor=sf)
        video_calib.save_image(9999, image, intr_t, (time.time() - start_time))
        video_calib.create_video()

    # Generate point cloud directly in main function
    if args.reconstruction_path is not None:
        generate_point_cloud(droid.video, device="cuda:0", filter_thresh=0.001)
        print(f"Debug: After point cloud generation, video.points exists = {hasattr(droid.video, 'points')}, "
              f"shape = {droid.video.points.shape if hasattr(droid.video, 'points') and droid.video.points is not None else 'None'}")
        save_reconstruction(droid, args.reconstruction_path, traj_est, frame_names)
