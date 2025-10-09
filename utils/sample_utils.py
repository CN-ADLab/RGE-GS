import fpsample
import time
import open3d as o3d
import numpy as np


def samplePointsByFPS(point_cloud, total_samples):
    # farthest point sample
    t1 = time.time()
    indices = fpsample.bucket_fps_kdline_sampling(point_cloud, total_samples, h=9)
    t2 = time.time()
    print("fpsample time : ", t2 - t1)
    gs_init_points_in_world = gs_init_points_in_world[indices]
    return gs_init_points_in_world


def samplePointsByVoxelDownsample(point_cloud, voxel_size = 0.8):
    print("points before down_sample: ", point_cloud.shape[0])
    temp = o3d.geometry.PointCloud()
    temp.points = o3d.utility.Vector3dVector(point_cloud)
    gs_init_points_in_world = np.asarray(
        temp.voxel_down_sample(voxel_size=0.8).points
    )
    print("points after down_sample: ", gs_init_points_in_world.shape[0])
    return gs_init_points_in_world


def samplePointsByDistanceWeight(point_cloud, center, total_samples):
    # calculate distance to center
    distances = np.linalg.norm(point_cloud - center, axis=1)
    weights = 1 / (1 + distances)
    
    # weights normalize
    weights = np.exp(weights - np.max(weights))
    weights =  weights / weights.sum(axis=0)
    
    # sample by weights
    sampled_indices = np.random.choice(len(point_cloud), size=total_samples, replace=False, p=weights)
    
    return point_cloud[sampled_indices]


def samplePointsByRadius(point_cloud, cam_infos, radius, total_samples, near_ratio = 0.6):
    near_all_index = None
    # cal index within radius
    for cam in cam_infos:
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam.R
        world_to_cam[:3, 3] = cam.T
        cam_to_world = np.linalg.inv(world_to_cam)
        cam_to_world_t = cam_to_world[:3, 3]
        distances = np.linalg.norm(point_cloud - cam_to_world_t, axis=1)
        temp_index = np.where(distances <= radius)[0]
        if near_all_index is None:
            near_all_index = temp_index
        else:
            near_all_index = np.concatenate((near_all_index, temp_index))
            near_all_index = np.unique(near_all_index)
    
    far_all_index = np.setdiff1d(np.arange(point_cloud.shape[0]), near_all_index)
    
    # cal num 
    near_samples_num = int(near_ratio * total_samples)
    far_samples_num = total_samples - near_samples_num

    # cal select index
    near_indices = np.random.choice(near_all_index, size = near_samples_num, replace = False)

    if far_samples_num < len(far_all_index):
        far_indices = np.random.choice(far_all_index, size = far_samples_num, replace = False)
    else:
        far_indices = far_all_index

    near_points = point_cloud[near_indices]
    far_points = point_cloud[far_indices]
    samples_points = np.concatenate((near_points, far_points), axis = 0)
    return samples_points


def samplePointsByRadiusAndFPS(point_cloud, cam_infos, radius, total_samples, near_ratio = 0.6):
    near_all_index = None
    # cal index within radius
    for cam in cam_infos:
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam.R
        world_to_cam[:3, 3] = cam.T
        cam_to_world = np.linalg.inv(world_to_cam)
        cam_to_world_t = cam_to_world[:3, 3]
        distances = np.linalg.norm(point_cloud - cam_to_world_t, axis=1)
        temp_index = np.where(distances <= radius)[0]
        if near_all_index is None:
            near_all_index = temp_indices
        else:
            near_all_index = np.concatenate((near_all_index, temp_index))
            near_all_index = np.unique(near_all_index)
    
    far_all_index = np.setdiff1d(np.arange(point_cloud.shape[0]), near_all_index)
    far_all_points = point_cloud[far_all_points]
    # cal num 
    near_samples_num = int(near_ratio * total_samples)
    far_samples_num = total_samples - near_samples_num
    
    # cal select index
    near_indices = np.random.choice(near_all_index, size = near_samples_num, replace = False)
    near_points = point_cloud[near_indices]

    far_points = fpsample.bucket_fps_kdline_sampling(far_all_points, far_samples_num, h=9)
    samples_points = np.concatenate((near_points, far_points), axis = 0)
    return samples_points
