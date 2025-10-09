import os
import numpy as np
from typing import NamedTuple
from utils.general_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.camera_utils import cameraListFromcamInfos
from init.para_lane_data_loader import paraLaneDataLoader
from init.waymo_data_loader import readWaymoDataInfo


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        model_params,
        gaussians,
        model_id,
        diff_lidar_xyz=None,
        diff_lidar_rot=None,
        prefix=None
    ):
        self.model_path = model_params.model_path
        self.gaussians = gaussians

        self.loaded_iter = 0  
        if model_params.train_from_checkpoint:
            model_pth_path = os.path.join(self.model_path, str(model_id), str(model_params.block_id), "point_cloud")
            if os.path.exists(model_pth_path):
                self.loaded_iter = searchForMaxIteration(model_pth_path)
        print("Loading iteration: ", self.loaded_iter)

        if model_params.data_type == "ParaLane":
            scene_info = paraLaneDataLoader(model_params, diff_lidar_xyz, diff_lidar_rot, prefix)
        elif model_params.data_type == "Waymo":
            scene_info = readWaymoDataInfo(model_params, model_id, diff_lidar_xyz, diff_lidar_rot)
        else:
            print("Invalid data type!")
            exit(-1)
        model_id = str(model_id)

        if scene_info is None:
            self.init_status = False
            return

        self.init_status = True
        self.voxel_info = scene_info.voxel_info
        self.camera_infos = cameraListFromcamInfos(scene_info.camera_infos)
        self.gaussians.initConfidenceNetwork()

        if scene_info.nerf_normalization is not None:
            self.cameras_extent = scene_info.nerf_normalization["radius"]

        if self.loaded_iter != 0:
            anchor_status = self.gaussians.load_ply_sparse_gaussian(
                os.path.join(
                    self.model_path,
                    model_id,
                    str(model_params.block_id),
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
            if not anchor_status:
                self.init_status = False
                return

            self.gaussians.load_mlp_checkpoints(
                os.path.join(
                    self.model_path,
                    model_id,
                    str(model_params.block_id),
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                )
            )
            
        else:
            self.gaussians.set_appearance(len(scene_info.camera_infos))
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration, model_id, block_id):
        if isinstance(model_id, (int, float)):
            model_id = str(model_id)
        point_cloud_path = os.path.join(
            self.model_path,
            model_id,
            str(block_id),
            "point_cloud/iteration_{}".format(iteration),
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getCamerasInfo(self):
        return self.camera_infos

    def getVoxelInfo(self):
        return self.voxel_info


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool
