import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scene.cameras import Camera


def computeReprojectionError(project_image: torch.Tensor, 
                              diffusion_image: torch.Tensor,
                              distance_type: str = 'manhattan',
                              mask: torch.Tensor = None) -> torch.Tensor:
    """
    param:
        project_image: [B,C,H,W] (0-255)
        diffusion_image: [B,C,H,W] (0-255)
        distance_type: ('euclidean', 'manhattan')
        mask: pred mask [B,1,H,W] or [B,C,H,W]
    
    return:
        error_map [B,H,W]
    """
    assert project_image.shape == diffusion_image.shape
    assert distance_type in ['euclidean', 'manhattan']

    project_float = project_image.to(torch.float32)  # [C,H,W]
    diffusion_float = diffusion_image.to(torch.float32)

    diff = project_float - diffusion_float  # [C,H,W]
    if distance_type == 'euclidean':
        error_map = torch.norm(diff, p=2, dim=0).unsqueeze(0)  # L2
    else:  # manhattan
        error_map = torch.abs(diff).sum(dim=0).unsqueeze(0)  # L1

    project_non_black_mask = (project_float != 0).any(dim=0)  # [H,W]
    # print(error_map.shape, project_non_black_mask.shape)
    error_map = error_map * project_non_black_mask.to(torch.float32)
    effective_mask = project_non_black_mask
    
    if mask is not None:
        if mask.dim() == 3:
            error_map = error_map * mask  # [1,H,W]
        else:
            raise ValueError
        
    valid_pixels = effective_mask.sum()
    total_error = (error_map * effective_mask.to(torch.float32)).sum()
    
    return total_error / valid_pixels.clamp(min=1e-6)

def alignMonoDepthToLidar(points, mono_depth, w2c, K):
    height, width = mono_depth.shape
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    
    points_cam = (w2c @ points_homo.T).T
    points_cam = points_cam[points_cam[:, 2] > 0]
    z_coords = points_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy

    u_rounded = np.round(u).astype(int)
    v_rounded = np.round(v).astype(int)

    valid = (u_rounded >= 0) & (u_rounded < width) & (v_rounded >= 0) & (v_rounded < height)
    u_valid = u_rounded[valid]
    v_valid = v_rounded[valid]
    z_valid = z_coords[valid]

    lidar_depth = np.full((height, width), np.inf)
    indices = np.ravel_multi_index((v_valid, u_valid), (height, width))
    np.minimum.at(lidar_depth.ravel(), indices, z_valid)
    lidar_depth[lidar_depth == np.inf] = 0
    # cut off upper half
    lidar_depth[:height//2, :] = 0
    
    mask = (lidar_depth > 0) & (~np.isnan(lidar_depth))
    x = mono_depth[mask].reshape(-1, 1)
    y = lidar_depth[mask]
    
    model = RANSACRegressor()
    model.fit(x, y)
    scale = model.estimator_.coef_[0]
    shift = model.estimator_.intercept_
    print(f"Inlier ratio: {np.mean(model.inlier_mask_):.2f}")
    print(f"R^2 score: {model.score(x, y):.2f}")
    
    aligned_depth = mono_depth * scale + shift
    return aligned_depth

def warpImage(
    src_image: np.ndarray,
    src_depth: np.ndarray, 
    K: np.ndarray, 
    T_src: np.ndarray,
    T_dst: np.ndarray
) -> np.ndarray:
    """
    Warp image by reprojection
    :param src_image: (H, W, C)
    :param src_mono_depth: (H, W)
    :param K: intrinsics (3x3)
    :param T_src: GT image transform matrix (4x4), w2c
    :param T_dst: diffusion image transform matrix (4x4), w2c
    :return: projected target image
    """
    assert src_image.ndim == 3
    assert src_depth.shape == src_image.shape[:2]
    assert K.shape == (3,3)
    assert T_src.shape == (4,4)
    assert T_dst.shape == (4,4)

    height, width = src_image.shape[:2]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    pixel_matrix = np.zeros((3, height, width), dtype=int) # 3 * 720 * 1280
    for row in range(height):
        for col in range(width):
            pixel_matrix[0, row, col] = col
            pixel_matrix[1, row, col] = row
            pixel_matrix[2, row, col] = 1
    inverse_K = np.linalg.inv(K)
    repeat_depth = np.repeat(src_depth[np.newaxis, :, :], 3, axis = 0)
    points_cam_src = np.einsum('ij,jkl->ikl', inverse_K, pixel_matrix)
    points_cam_src = repeat_depth * points_cam_src
    points_cam_src = points_cam_src.reshape(3, -1)
    
    c2w = np.linalg.inv(T_src)    
    points_cam_src_hom = np.reshape(points_cam_src, (3, -1))
    points_cam_src_hom = np.vstack([points_cam_src_hom, np.ones((1, points_cam_src_hom.shape[1]))])
    points_world = c2w @ points_cam_src_hom  # (4, 4) @ (4, N) = (4, N)
    points_world = points_world[:3, :]
    
    points_cam_dst_hom = T_dst @ np.vstack([points_world, np.ones((1, points_world.shape[1]))])  # (4, N)
    points_cam_dst = points_cam_dst_hom[:3, :]

    points_2d = K @ points_cam_dst             
    z = points_2d[2, :]  
    valid_z = z > 1e-6  
    u = np.where(valid_z, points_2d[0, :] / z, 0)
    v = np.where(valid_z, points_2d[1, :] / z, 0)
    u_round = np.round(u).astype(int)
    v_round = np.round(v).astype(int)
    valid = valid_z & (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)
    
    dst_image = np.zeros_like(src_image)
    dst_depth = np.full((height, width), np.inf)
    
    valid_indices = np.where(valid)[0]
    valid_u = u_round[valid_indices]
    valid_v = v_round[valid_indices]
    src_cols = cols.ravel()[valid_indices]
    src_rows = rows.ravel()[valid_indices]
    current_depth = z[valid_indices]
    
    for idx in range(len(valid_indices)):
        u_idx = valid_u[idx]
        v_idx = valid_v[idx]
        depth = current_depth[idx]
        if depth < dst_depth[v_idx, u_idx]:
            dst_depth[v_idx, u_idx] = depth
            dst_image[v_idx, u_idx] = src_image[src_rows[idx], src_cols[idx]]
    
    return dst_image