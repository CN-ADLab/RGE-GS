import os
import sys
import json
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from utils.pose_utils import disturbPose, getLidarPose

sys.path.append('../')
from scene.gaussian_model import GaussianModel
from scene.scene import Scene, TimePose, GaussianView

class Rect:
    def __init__(self):
        self.min_x = sys.maxsize
        self.max_x = -sys.maxsize
        self.min_y = sys.maxsize
        self.max_y = -sys.maxsize

    def calculateRect(self, points):
        for point in points:
            self.min_x = min(point[0], self.min_x)
            self.max_x = max(point[0], self.max_x)
            self.min_y = min(point[1], self.min_y)
            self.max_y = max(point[1], self.max_y)

    def setRect(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


def checkPoseCollision(model_params, model_id_list, diff_lidar_xyz, diff_lidar_rot):
    meta_file = os.path.join(model_params.source_path, "meta_infos", model_params.cyber_log + ".pkl")
    meta_info = pickle.load(open(meta_file, "rb"))
    for model_id in model_id_list:
        print("model id ", model_id)
        if model_id == 0:
            continue

        model_folder = os.path.join(model_params.source_path, "temp", model_params.cyber_log, "occ/preproc/dynamic/objects", str(model_id))
        model_vertices = np.asarray(o3d.io.read_point_cloud(os.path.join(model_folder, "stitch.pcd")).points)[:, :2]
        model_rect = Rect()
        model_rect.calculateRect(model_vertices)

        info_json_file = os.path.join(model_folder, "info.json")
        with open(info_json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        lidar_time_pose_dict = {}
        object_start_time = sys.maxsize
        object_end_time = 0
        for key, value in json_data.get("tracks").items():
            object_start_time = min(object_start_time, int(key))
            object_end_time = max(object_end_time, int(key))
            lidar_time_pose_dict[int(key)] = np.resize(value["T_b2l"], (4, 4))

        for camera_channel in model_params.camera_channel:
            curr_Rect = Rect()
            if camera_channel == "CAMERA_FRONT_resize":
                curr_Rect.setRect(model_rect.min_x - 1.5, model_rect.max_x + 1.5, model_rect.min_y, model_rect.max_x)
            elif camera_channel == "CAMERA_PANO_FRONT_resize" or camera_channel == "CAMERA_PANO_BACK_resize":
                curr_Rect.setRect(model_rect.min_x - 0.5, model_rect.max_x + 0.5, model_rect.min_y, model_rect.max_x)
            else:
                curr_Rect.setRect(model_rect.min_x, model_rect.max_x, model_rect.min_y - 0.5, model_rect.max_x + 0.5)

            for info in meta_info["frames"]:
                timestamp = info["log_time_stamp"]
                if timestamp not in model_params.train_frame_times:
                    continue
                if timestamp < object_start_time:
                    continue
                if timestamp not in lidar_time_pose_dict:
                    continue
                if timestamp > object_end_time:
                    break

                object_to_lidar = lidar_time_pose_dict[timestamp]
                lidar_to_object = np.linalg.inv(object_to_lidar)
                lidar_to_object = disturbPose(lidar_to_object, diff_lidar_xyz, diff_lidar_rot)
                object_to_lidar = np.linalg.inv(lidar_to_object)
                camera_to_lidar = info["cams_info"][camera_channel]["compensated_camera2lidar"]
                camera_to_object = np.linalg.inv(object_to_lidar) @ camera_to_lidar
                curr_x = camera_to_object[0, 3]
                curr_y = camera_to_object[1, 3]

                if curr_x < curr_Rect.max_x and curr_x > curr_Rect.min_x and curr_y < curr_Rect.max_y and curr_y > curr_Rect.min_y:
                    if timestamp not in model_params.invalid_render_frame_times[camera_channel]:
                        model_params.invalid_render_frame_times[camera_channel].append(timestamp)
    return


def checkPoseInRoad(model_params, diff_lidar_xyz, diff_lidar_rot):
    ground_file_path = os.path.join(model_params.model_path, "0", str(model_params.block_id), "ground_points_from_label.txt")
    ground_points_from_label = np.loadtxt(ground_file_path, dtype=float, delimiter=',')[:, :2]
    kdtree = cKDTree(ground_points_from_label)

    meta_file = os.path.join(model_params.source_path, "meta_infos", model_params.cyber_log + ".pkl")
    meta_info = pickle.load(open(meta_file, "rb"))

    for camera_channel in model_params.camera_channel:
        for info in meta_info["frames"]:
            timestamp = info["log_time_stamp"]
            if timestamp not in model_params.train_frame_times:
                continue
            lidar_to_world = getLidarPose(info)
            if lidar_to_world is None:
                continue

            lidar_to_world = disturbPose(lidar_to_world, diff_lidar_xyz, diff_lidar_rot)
            camera_to_lidar = info["cams_info"][camera_channel]["compensated_camera2lidar"]
            camera_to_world = lidar_to_world @ camera_to_lidar
            target_point = camera_to_world[:2, 3]
            distances, indices = kdtree.query(target_point, k=20)
            near_points = ground_points_from_label[indices]

            patch_rect = Rect()
            patch_rect.calculateRect(near_points)
            if not (target_point[0] < patch_rect.max_x and target_point[0] > patch_rect.min_x and \
                    target_point[1] < patch_rect.max_y and target_point[1] > patch_rect.min_y):
                if timestamp not in model_params.invalid_render_frame_times[camera_channel]:
                    model_params.invalid_render_frame_times[camera_channel].append(timestamp)
    return


def getInvalidRenderTime(model_params, model_id_list, diff_lidar_xyz, diff_lidar_rot):
    for camera_channel in model_params.camera_channel:
        model_params.invalid_render_frame_times[camera_channel] = []
    if diff_lidar_xyz is None and diff_lidar_rot is None:
        return
    if np.all(diff_lidar_xyz == 0) and np.array_equal(diff_lidar_rot, np.identity(4)):
        return

    checkPoseCollision(model_params, model_id_list, diff_lidar_xyz, diff_lidar_rot)
    checkPoseInRoad(model_params, diff_lidar_xyz, diff_lidar_rot)

    for camera_channel in model_params.camera_channel:
        print("Invalid frames number for {0} is {1}".format(camera_channel, len(model_params.invalid_render_frame_times[camera_channel])))
    return


def getModelSceneInfo(model_id_list, model_params, use_cylindrical_projection, diff_lidar_xyz=None, diff_lidar_rot=None, prefix=None):
    # getInvalidRenderTime(model_params, model_id_list, diff_lidar_xyz, diff_lidar_rot)
    model_id_scene_info = {}
    for model_id in model_id_list:
        model_gaussians = GaussianModel(
            model_params.feat_dim,
            model_params.n_offsets,
            model_params.voxel_size,
            model_params.update_depth,
            model_params.update_init_factor,
            model_params.update_hierachy_factor,
            model_params.use_feat_bank,
            model_params.appearance_dim,
            model_params.ratio,
            model_params.add_opacity_dist,
            model_params.add_cov_dist,
            model_params.add_color_dist,
            use_cylindrical_projection,
            model_params.use_pose_correction,
        )
        model_scene = Scene(
            model_params,
            model_gaussians,
            model_id,
            diff_lidar_xyz=diff_lidar_xyz,
            diff_lidar_rot=diff_lidar_rot,
            prefix=prefix
        )

        if model_scene.init_status:
            views = model_scene.getCamerasInfo()
            time_poses = {}
            for view in views:
                camname_timestamp = view.camera_name + "_" + view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = view.R
                camera_pose[:3, 3] = view.T
                time_poses[camname_timestamp] = TimePose(camera_pose=camera_pose, view=view)
            model_id_scene_info[model_id] = GaussianView(
                gaussians=model_gaussians, scene=model_scene, time_poses=time_poses
            )
    return model_id_scene_info
