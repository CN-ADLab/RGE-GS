import os
import cv2
import numpy as np
import open3d as o3d
from typing import NamedTuple
from utils.pose_utils import getLidarPose
from utils.graphics_utils import getWorld2View2
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_name: str
    camera_name: str
    width: int
    height: int
    K: np.array
    img_mask: np.array
    camera_mask: np.array
    normal: np.array
    depth: np.array
    img_mask_inpainting: np.array
    image_inpainting: np.array
    project_image: np.array
    img_mask_wo_conf: np.array


class VoxelInfo(NamedTuple):
    normal_array: np.array
    origin_points: np.array
    voxel_size: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    camera_infos: list
    nerf_normalization: dict
    voxel_info: VoxelInfo


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


def depthToRgb(depth):
    depth *= 3
    red_val, green_val, blue_val = 0, 0, 0
    if depth <= 51:
        blueVal = 255
        green_val = depth * 5
        red_val = 0
    elif depth <= 102:
        depth -= 51
        blueVal = 255 - depth * 5
        greenVal = 255
        red_val = 0
    elif depth <= 153:
        depth -= 102
        blueVal = 0
        greenVal = 255
        red_val = depth * 5
    elif depth <= 204:
        depth -= 153
        blueVal = 0
        greenVal = 255 - int(128 * depth / 51 + 0.5)
        red_val = 255
    else:
        depth -= 204
        blueVal = 0
        greenVal = 127 - int(127 * depth / 51 + 0.5)
        red_val = 255

    return red_val, green_val, blue_val


def pointsToDepth(img_height, img_width, intrinsic, pointcloud_camera, scale=0.2):
    scale_height = int(scale * img_height)
    scale_width = int(scale * img_width)
    scale_intrinsic = intrinsic.copy()
    scale_intrinsic[:2] = scale_intrinsic[:2] * scale

    pts_depth = np.zeros([1, scale_height, scale_width])
    uvz = pointcloud_camera[pointcloud_camera[:, 2] > 0]

    uvz = uvz @ scale_intrinsic.T
    uvz[:, :2] /= uvz[:, 2:]
    
    uvz = uvz[uvz[:, 1] >= 0]
    uvz = uvz[uvz[:, 1] < scale_height]
    uvz = uvz[uvz[:, 0] >= 0]
    uvz = uvz[uvz[:, 0] < scale_width]

    indices_desc_sorted_by_third_col = np.argsort(-uvz[:, 2])
    uvz_sorted = uvz[indices_desc_sorted_by_third_col]
    uv = uvz_sorted[:, :2]
    uv = uv.astype(int)
    pts_depth[0, uv[:, 1], uv[:, 0]] = uvz_sorted[:, 2]

    depth = cv2.resize(
        pts_depth[0], (img_width, img_height), interpolation=cv2.INTER_NEAREST
    )
    return depth


def saveDebugProjection(gs_init_points, cam_infos, save_path):
    cam_trajectory = []
    for cam_info in cam_infos:
        if cam_info.camera_name != cam_infos[0].camera_name : continue
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam_info.R
        world_to_cam[:3, 3] = cam_info.T
        cam_to_world = np.linalg.inv(world_to_cam)
        cam_trajectory.append(cam_to_world[:3, 3])
    cam_trajectory_array = np.vstack(cam_trajectory)
    np.savetxt(save_path + "/cam_trajectory.txt", cam_trajectory_array, delimiter=",", fmt="%.2f")

    save_dir = os.path.join(save_path, "debug_projection")
    os.makedirs(save_dir, exist_ok=True)

    for cam_id in range(len(cam_infos)):
        cv_img = cv2.cvtColor(np.array(cam_infos[cam_id].image), cv2.COLOR_RGB2BGR)
        cv_img_projection = cv_img.copy()
        cv_mask = cam_infos[cam_id].camera_mask
        cv_mask = np.repeat(cv_mask[:, :, np.newaxis], 3, axis=2)
        cv_img_projection = cv_img_projection * cv_mask

        camera_points = cam_infos[cam_id].R @ np.transpose(gs_init_points) + cam_infos[cam_id].T[:, np.newaxis]
        intrinsic_list = cam_infos[cam_id].K

        if cam_infos[cam_id].camera_name == "CAMERA_FRONT_resize":
            intrinsic_array = np.eye(3)
            intrinsic_array[0, 0] = intrinsic_list[0]
            intrinsic_array[0, 2] = intrinsic_list[2]
            intrinsic_array[1, 1] = intrinsic_list[1]
            intrinsic_array[1, 2] = intrinsic_list[3]
            projections = intrinsic_array @ camera_points
            projections = projections[:-1, :] / projections[-1, :]

        else:
            intrinsic_array = np.eye(4)
            intrinsic_array[0, 0] = intrinsic_list[0]
            intrinsic_array[0, 2] = intrinsic_list[2]
            intrinsic_array[1, 1] = intrinsic_list[1]
            intrinsic_array[1, 2] = intrinsic_list[3]

            atn_xz = np.arctan2(camera_points[0, :], camera_points[2, :])
            y_p_dxz = camera_points[1, :] / np.sqrt(camera_points[0, :] * camera_points[0, :] + camera_points[2, :] * camera_points[2, :])
            pts_3d_homo = np.concatenate(
                [atn_xz.reshape(-1, 1), y_p_dxz.reshape(-1, 1), np.ones((camera_points.shape[1], 2), dtype=np.float32)], axis=1)

            pts_2d = np.dot(intrinsic_array, pts_3d_homo.transpose(1, 0))
            projections = pts_2d[:2, :]

        img_rows = cv_img.shape[0]
        img_cols = cv_img.shape[1]
        for row in range(0, projections.shape[1]):
            center = (int(projections[0, row]), int(projections[1, row]))
            if camera_points[2][row] > 0.1:
                if center[0] > 0 and center[0] < img_cols and center[1] > 0 and center[1] < img_rows:
                    red_val, green_val, blue_val = depthToRgb(camera_points[2][row])
                    cv2.circle(
                        cv_img_projection, center, 3, (red_val, green_val, blue_val), -1
                    )
        cv2.imwrite(
            os.path.join(save_dir, str(cam_id) + "_debug_projection.png"),
            np.concatenate((cv_img, cv_img_projection), axis=1),
        )
    return


def getGroundPointsFromRansac(world_points, cam_infos):
    ground_height_from_cam = 0
    mid_cam_info = cam_infos[int(len(cam_infos) / 2)]
    if mid_cam_info.camera_name == "CAMERA_FRONT_resize":
        ground_height_from_cam = 0.5
    else:
        ground_height_from_cam = 0

    camera_points = mid_cam_info.R @ np.transpose(world_points) + mid_cam_info.T[:, None]
    valid_indices = np.where(camera_points[1, :] > ground_height_from_cam)[0]

    ground_points = world_points[valid_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)

    world_points_aug = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    points_dist = world_points_aug @ plane_model[:, None]
    valid_indices = np.where(points_dist[:, 0] < 0.15)[0]
    ground_points = world_points[valid_indices]
    pcd.points = o3d.utility.Vector3dVector(ground_points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)

    ground_points = pcd.select_by_index(inliers)
    return np.asarray(ground_points.points)


def getGroundPointsFromLabel(args, meta_info, pointclouds_dir, pcd_label_dir, lidar_stitch_interval = 5):
    concat_lidar_frame_count = 1
    ground_points_in_world = []

    for info in meta_info["frames"]:
        timestamp = info["log_time_stamp"]
        if timestamp not in args.train_frame_times:
            continue
        lidar_to_world = getLidarPose(info)
        if lidar_to_world is None:
            continue

        if concat_lidar_frame_count % lidar_stitch_interval == 0:
            lidar_info = np.load(os.path.join(pointclouds_dir, str(str(timestamp) + ".npz")))["data"]
            semantic_valid_indices = set(np.squeeze(np.argwhere(lidar_info[:, 5] == 0)))
            ground_label = np.load(os.path.join(pcd_label_dir, str(str(timestamp) + ".npz")))["data"]
            ground_label_indices = set(np.squeeze(np.argwhere(ground_label == 97)))

            valid_indices = np.array(list(semantic_valid_indices.intersection(ground_label_indices)))
            ground_points = lidar_info[:, :3][valid_indices]
            ground_points_in_world.append(
                np.transpose(
                    lidar_to_world[:3, :3].dot(ground_points.transpose())
                    + lidar_to_world[:3, 3, np.newaxis]
                )
            )
        concat_lidar_frame_count += 1
        if concat_lidar_frame_count == args.train_frame_number:
            break

    ground_points_in_world = np.concatenate(ground_points_in_world, axis=0)
    return ground_points_in_world
