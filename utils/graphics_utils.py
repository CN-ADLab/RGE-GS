import torch
import math
import numpy as np
from typing import NamedTuple
from utils.general_utils import buildRotation


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geomTransformPoints(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getAsymmetricProjectionMatrix(K,H,W,aspect=1.0,n=0.1,f=1000.0):
    # print("intrinsic shape",K.shape)
    if K.shape == (3,3):
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    elif K.shape == (4,):
        fx,fy,cx,cy = K[0],K[1],K[2],K[3]
    else:
        print("getAsymmetricProjectionMatrix input error, K shape unknown")
        exit(1)
    r = float(W) - cx
    l = -cx
    t =  cy 
    b = -(float(H) - cy)
    z_sign = 1.0
    return torch.tensor(
        [[2*fx/(r-l),     0,      (r+l)/(r-l),          0       ],
         [      0,  2*fy/(t-b),   (t+b)/(t-b),          0       ],
         [      0,        0,      z_sign*(f+n)/(f-n),    -(2*f*n)/(f-n)],
         [      0,        0,        z_sign,                   0      ]],dtype=torch.float32
    )


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def depthsToPoints(view, depthmap, points_buffer=None):
        
    c2w = (view.world_view_transform.T).inverse()

    W, H = view.image_width, view.image_height
    if points_buffer is None:
        points_buffer = torch.empty((H*W, 3), device='cuda', dtype=torch.float32)
    fx = W / (2 * math.tan(view.FoVx / 2.0))
    fy = H / (2 * math.tan(view.FoVy / 2.0))
    intrins = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
                        device='cuda', dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda'), torch.arange(H, device='cuda'), indexing="xy")
    points_buffer[:, 0] = grid_x.flatten()
    points_buffer[:, 1] = grid_y.flatten()
    points_buffer[:, 2] = 1.0
    rays_d = points_buffer @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o

    return points


def depthToNormal(view, depth, points_buffer=None):
    """
    view: view camera
    depth: depthmap
    """
    points = depthsToPoints(view, depth, points_buffer).reshape(*depth.shape[1:], 3)

    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)

    normal_map = torch.nn.functional.normalize(
        torch.cross(dx, dy, dim=-1), dim=-1
    )  # eg: abc三个像素 c-a得到b的梯度 x和y方向的梯度叉乘得到法向量
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def renderNormal(allmap, rot, viewpoint_camera, points_buffer=None):
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5] 
    render_normal = (
        render_normal.permute(1, 2, 0)
        @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (
        render_depth_expected / render_alpha
    )  # 期望的深度是 深度值除透明度
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map 
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    depth_ratio = 0
    surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal, surf_point = depthToNormal(viewpoint_camera, surf_depth, points_buffer)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rotations_mat = buildRotation(rot)
    world_normal = rotations_mat[:, :, 2]
    return world_normal, render_normal, surf_depth, surf_normal