import os
import math
import copy
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
from utils.general_utils import computeHashIndex


def getLidarPose(info, args):
    if args.data_type == 'Waymo':
        return info["lidar2world"]
    else:
        if "optimized_pose" not in info.keys():
            # lidar_to_world = info["lidar2world"]
            return None
        return info["optimized_pose"]


def disturbTranslation(lidar_to_world, diff_lidar_xyz):
    if diff_lidar_xyz is None:
        return lidar_to_world

    lidar_to_world_rotation = lidar_to_world[:3, :3]
    curr_trans = lidar_to_world_rotation.dot(np.transpose(diff_lidar_xyz))
    lidar_to_world[:3, 3] += np.squeeze(curr_trans)
    return lidar_to_world


def disturbRotation(world_to_lidar, diff_lidar_rot):
    if diff_lidar_rot is not None:
        world_to_lidar = diff_lidar_rot.dot(world_to_lidar)
    return world_to_lidar


def disturbPose(lidar_to_world, diff_lidar_xyz, diff_lidar_rot):
    lidar_to_world = disturbTranslation(lidar_to_world, diff_lidar_xyz)
    world_to_lidar = np.linalg.inv(lidar_to_world)
    world_to_lidar = disturbRotation(world_to_lidar, diff_lidar_rot)
    lidar_to_world = np.linalg.inv(world_to_lidar)
    return lidar_to_world


def readTimestepDictBetweenLiDARAndCamera(initposes_file):
    timestamp_dict = {}
    with open(initposes_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split(" ")
            if (len(line) > 2):
                camera_channel = line[1]
                lidar_timestamp = line[0].split("/")[0]
                camera_timestamp = line[2]
                timestamp_dict[str(lidar_timestamp)+"_"+str(camera_channel)] = camera_timestamp
    return timestamp_dict


def getSfMPose(timestamp, camera_channel, timestamp_dict, sfm_camera_pose):
    first_key = str(timestamp) + "_" + camera_channel[ : -7]
    if first_key in timestamp_dict:
        second_key = timestamp_dict[first_key]
        if second_key in sfm_camera_pose:
            if len(sfm_camera_pose[second_key]) > 0:
                camera_to_world = sfm_camera_pose[second_key].pop(0)
                return camera_to_world
    return None


def getSfMTimeDict(args, meta_info):
    use_sfm_pose = args.use_sfm_pose
    timestamp_dict = None
    sfm_camera_pose = None
    sfm_camera_pose_copy = None
    if use_sfm_pose:
        sfm_file_name = os.path.join(args.source_path, "temp", args.cyber_log, "occ", "sfm", "refined", "refined_image_poses_all.txt")
        if os.path.exists(sfm_file_name):
            print("use sfm pose")
            sfm_camera_pose = readSFMPose(sfm_file_name)
            sfm_camera_pose_copy = copy.deepcopy(sfm_camera_pose)
            initposes_file = os.path.join(args.source_path, "temp", args.cyber_log, "image_init_poses.txt")
            timestamp_dict = readTimestepDictBetweenLiDARAndCamera(initposes_file)

            # check sfm pose
            prev_pose = {}
            for camera_channel in args.camera_channel:
                prev_pose[camera_channel] = None

            for info in meta_info:
                timestamp = info["log_time_stamp"]
                for camera_channel in args.camera_channel:
                    camera_to_world = getSfMPose(timestamp, camera_channel, timestamp_dict, sfm_camera_pose)
                    if camera_to_world is None:
                        prev_pose[camera_channel] = None
                        continue

                    if prev_pose[camera_channel] is not None:
                        diff_xyz = np.abs(prev_pose[camera_channel] - camera_to_world)[:3 ,3]
                        if diff_xyz[0] > 1.7 or diff_xyz[1] > 1.7 or diff_xyz[2] > 0.15:
                            print("sfm pose false, use lidar pose: ", diff_xyz)
                            return False, None, None
                    prev_pose[camera_channel] = camera_to_world

        else:
            print("sfm_pose_file doesn't exist, turn to lidar pose")
            use_sfm_pose = False
    else:
        print("use lidar pose")
    return use_sfm_pose, timestamp_dict, sfm_camera_pose_copy


def getCameraPose(use_sfm_pose, info, timestamp, camera_channel, timestamp_dict, sfm_camera_pose, lidar_to_world):
    camera_to_world = None
    if use_sfm_pose:
        camera_to_world = getSfMPose(timestamp, camera_channel, timestamp_dict, sfm_camera_pose)

    # do not use sfm pose or sfm pose query failure
    if camera_to_world is None:
        camera_to_lidar = info["cams_info"][camera_channel]["compensated_camera2lidar"] if "compensated_camera2lidar" in info["cams_info"][camera_channel] else info["cams_info"][camera_channel]["camera2lidar"]
        camera_to_world = lidar_to_world.dot(camera_to_lidar)
    return camera_to_world


def interpolatePose(start_pose, end_pose, num_poses):
    start_trans = start_pose[:3, 3]
    end_trans = end_pose[:3, 3]
    start_rot = start_pose[:3, :3]
    end_rot = end_pose[:3, :3]

    interpolated_poses = []
    for i in range(num_poses):
        t = i / (num_poses - 1)
        interp_trans = start_trans + (end_trans - start_trans) * t
        slerp = Slerp([0, 1], Rotation.from_matrix([start_rot, end_rot]))
        interp_rot = slerp([t]).as_matrix()[0]

        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rot
        interp_pose[:3, 3] = interp_trans   
        interpolated_poses.append(interp_pose)
    return interpolated_poses


def saveInsertPos(file_path, insert_pos, groud_plane, voxel_info):
    insert_pos_x = insert_pos[0]
    insert_pos_y = insert_pos[1]
    insert_pos[2] = (-groud_plane[3] - groud_plane[0] * insert_pos_x - groud_plane[1] * insert_pos_y) / groud_plane[2]

    voxel_xyz = ((insert_pos - voxel_info.origin_points) / voxel_info.voxel_size).astype(int)
    voxel_val = computeHashIndex(voxel_xyz[None, :])[0]
    col_id = np.where(voxel_info.normal_array[0, :] == voxel_val)[0]
    if col_id:
        voxel_plane = voxel_info.normal_array[:, col_id]
        insert_pos[2] = (-voxel_plane[4] - voxel_plane[1] * insert_pos_x - voxel_plane[2] * insert_pos_y) / voxel_plane[3]

    np.savetxt(file_path, insert_pos, delimiter=",", fmt="%.2f")
    return


def getNoisePose(noise_value):
    diff_lidar_xyz = np.array([[noise_value[0], noise_value[1], noise_value[2]]])

    yaw_theta = noise_value[3] / 180.0
    cos_yaw_theta = math.cos(yaw_theta)
    sin_yaw_theta = math.sin(yaw_theta)
    diff_lidar_yaw = np.array(
        [
            [cos_yaw_theta, sin_yaw_theta, 0, 0],
            [-sin_yaw_theta, cos_yaw_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    pitch_theta = noise_value[4] / 180.0
    cos_pitch_theta = math.cos(pitch_theta)
    sin_pitch_theta = math.sin(pitch_theta)
    diff_lidar_pitch = np.array(
        [
            [cos_pitch_theta, 0, -sin_pitch_theta, 0],
            [0, 1, 0, 0],
            [sin_pitch_theta, 0, cos_pitch_theta, 0],
            [0, 0, 0, 1],
        ]
    )

    diff_lidar_rot = diff_lidar_pitch.dot(diff_lidar_yaw)
    return diff_lidar_xyz, diff_lidar_rot


def obtainCameraPoseFrom3To5():
    camera_diff_pose_info = {
        "CAMERA_FRONT_resize": [0, -0.2, 0, 0, 0],
        "CAMERA_PANO_FRONT_resize": [0, -0.2, 0, 0, 0],
        "CAMERA_PANO_LEFT_resize": [-0.1, -0.2, 0, 0, 0],
        "CAMERA_PANO_RIGHT_resize": [0.1, -0.2, 0, 0, 0],
        "CAMERA_PANO_BACK_resize": [0, 0, -0.5, 0, 0],
    }
    return camera_diff_pose_info


def readSFMPose(sfm_file_name):
    sfm_camera_pose = {}
    with open(sfm_file_name, "r") as file:
        for line in file:
            data = line.strip().split(" ")
            frame_timestep = data[0].split(".")[0]
            quaternion = [
                float(data[2]),
                float(data[3]),
                float(data[4]),
                float(data[1]),
            ]
            rot_matrix = Rotation.from_quat(quaternion).as_matrix()

            trans_matrix = np.eye(4)
            trans_matrix[:3, :3] = rot_matrix
            trans_matrix[0, 3] = data[5]
            trans_matrix[1, 3] = data[6]
            trans_matrix[2, 3] = data[7]

            if frame_timestep in sfm_camera_pose: # resolve timestamp conflict eg.  CAMERA_FRONT and CAMERA_PANO_FRONT have same timestep
                sfm_camera_pose[frame_timestep].append(trans_matrix) # sequential storage : front \ pano front \ pano back \ pano left \ pano right
            else:
                sfm_camera_pose[frame_timestep] = [trans_matrix]
    return sfm_camera_pose


def getObjectToWorldPose(object_pose, model_static=True):
    object_to_world_list = []    
    if object_pose.shape[0] == 50:
        for id in range(50):
            object_to_world_list.append(object_pose[id, :, :])
    else: 
        for id in range(50):
            object_to_world_list.append(object_pose)    
    return object_to_world_list