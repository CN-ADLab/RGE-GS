import numpy as np
import pickle
import open3d as o3d
import cv2
import os
import torch
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Pointclouds
import imageio
from utils.colmap_data_io_utils import readExtrinsicsText, readIntrinsicsText, qvec2rotmat

def process_case_log(scene_name: str, track_train: str, track_test: str):
    # Load meta information
    # meta_info_path = f"/mnt/GStudio/data/{case}/meta_infos/{log}.pkl"
    # with open(meta_info_path, 'rb') as pickle_file:
    #     data = pickle.load(pickle_file)
    #     frames = data['frames']

    
    # Load diffusion pose information
    cameras_extrinsic_file = f"/mnt/GStudio/data/pack/{scene_name}/{track_test}/sparse/0/images_CAMERA_FRONT.txt"
    cam_extrinsics = readExtrinsicsText(cameras_extrinsic_file)
    cameras_intrinsic_file = f"/mnt/GStudio/data/pack/{scene_name}/{track_test}/sparse/0/cameras_undistortion.txt"
    cam_intrinsics = readIntrinsicsText(cameras_intrinsic_file)
        
    
    # Load point cloud
    colored_point_path = f"/mnt/GStudio/data/pack/{scene_name}/{track_train}/colored_sdf.pcd"
    pcd = o3d.io.read_point_cloud(colored_point_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    points_tensor = torch.tensor(np.asarray(pcd.points), dtype=torch.float).to("cuda")
    colors_tensor = torch.tensor(np.asarray(pcd.colors), dtype=torch.float).to("cuda")
    point_cloud = Pointclouds(points=[points_tensor], features=[colors_tensor])
    
    raster_settings = PointsRasterizationSettings( # 
        image_size=(720, 1280),          
        radius=0.01,    
        points_per_pixel=10
    )

    # Process each frame
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        image_name = extr.name
        timestamp = extr.name.split("/")[0]
        # if int(timestamp) not in model_params.train_frame_times:
        #     continue

        # intrinsics
        intr = cam_intrinsics['CAMERA_FRONT']
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        intrinsic = np.array([intr.params[0], intr.params[1], intr.params[2], intr.params[3]])
        

        # world to camera
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = qvec2rotmat(extr.qvec)
        world_to_camera[:3, 3] = np.array(extr.tvec)

        # Coordinate transformation
        pcd_camera = (np.pad(points[...,:3], ((0, 0), (0, 1)), constant_values=1) @ world_to_camera.T)[:, :3]

        # Projection parameters
        H = 720
        W = 1280
        # H = 576
        # W = 1024
        scale = 1280 / 1910
        scale_intrinsic = np.array([[intrinsic[0], 0, intrinsic[2]],[0, intrinsic[1], intrinsic[3]],[0,0,1]])
        # scale_intrinsic[:2] = scale_intrinsic[:2] * scale
        scale_intrinsic = scale * scale_intrinsic
        scale_intrinsic[2, 2] = 1

        # Initialize buffers
        dense_colored_pcd_projection_R = np.zeros([1, H, W])
        dense_colored_pcd_projection_G = np.zeros([1, H, W])
        dense_colored_pcd_projection_B = np.zeros([1, H, W])
        pts_depth = np.zeros([1, H, W])

        # Projection logic
        uvz = pcd_camera[pcd_camera[:, 2] > 0]
        selected_pcd_color = colors[pcd_camera[:, 2] > 0]
        uvz = uvz @ scale_intrinsic.T
        uvz[:, :2] /= uvz[:, 2:]

        # Filter valid points
        valid_mask = (uvz[:, 1] >= 0) & (uvz[:, 1] < H) & (uvz[:, 0] >= 0) & (uvz[:, 0] < W)
        selected_pcd_color = selected_pcd_color[valid_mask]
        uvz = uvz[valid_mask]

        # Sort by depth
        indices_desc = np.argsort(-uvz[:, 2])
        uvz_sorted = uvz[indices_desc]
        selected_pcd_color_final = selected_pcd_color[indices_desc]

        # Fill pixel buffers
        uv = uvz_sorted[:, :2].astype(int)
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz_sorted[:, 2]
        dense_colored_pcd_projection_R[0, uv[:, 1], uv[:, 0]] = selected_pcd_color_final[:,0]
        dense_colored_pcd_projection_G[0, uv[:, 1], uv[:, 0]] = selected_pcd_color_final[:,1]
        dense_colored_pcd_projection_B[0, uv[:, 1], uv[:, 0]] = selected_pcd_color_final[:,2]

        # Generate output image
        dense_colored_pcd_projection = np.concatenate(
            (dense_colored_pcd_projection_R, dense_colored_pcd_projection_G, dense_colored_pcd_projection_B),
            axis=0
        )
        dense_colored_pcd_projection = (dense_colored_pcd_projection * 255.0).astype(np.uint8).transpose(1, 2, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) 
        dense_colored_pcd_projection = cv2.dilate(
            dense_colored_pcd_projection, 
            kernel, 
            iterations=1  # 膨胀次数，增大可强化效果
        )
        
        s = min(H, W)
        focalx = scale_intrinsic[0, 0]
        focaly = scale_intrinsic[1, 1]
        principalx = scale_intrinsic[0, 2]
        principaly = scale_intrinsic[1, 2]
        f_ndcx = focalx * 2.0 / (s - 1)
        f_ndcy = focaly * 2.0 / (s - 1)
        p_ndcx = - (principalx - (W - 1) / 2.0) * 2.0 / (s - 1) 
        p_ndcy = - (principaly - (H - 1) / 2.0) * 2.0 / (s - 1) 
        f_ndc = torch.Tensor([f_ndcx, f_ndcy]).unsqueeze(0).to('cuda')
        p_ndc = torch.Tensor([p_ndcx, p_ndcy]).unsqueeze(0).to('cuda')
        
        camera_to_world = np.linalg.inv(world_to_camera)
        c2w = torch.Tensor(camera_to_world).to('cuda')
        c2w[:,0] = -c2w[:,0]
        c2w[:,1] = -c2w[:,1]
        w2c = c2w.inverse()
        T = w2c[:3, -1]
        R_row = c2w[:3,:3].unsqueeze(0).to('cuda')
        T = T.unsqueeze(0).to('cuda')

        # 创建相机对象
        cameras = PerspectiveCameras(f_ndc, p_ndc, R_row, T, device='cuda')
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )
        images = renderer(point_cloud)

        # Save output
        output_dir = f"/mnt/GStudio/data/pack/{scene}/{track_test}/colored_pcd_projection/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        imageio.imwrite(os.path.join(output_dir, 'dense_pcd_project_test.png'), dense_colored_pcd_projection)
        
        img = images[0, ..., :3].cpu().numpy()
        img = (img*255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, 'dense_pcd_render_test.png'), img)

# 使用示例
if __name__ == "__main__":
    scene = "scene_3"  # 替换为实际case名
    track_train = "track_0"    # 替换为实际log名
    track_test = "track_1"
    process_case_log(scene, track_train, track_test)
