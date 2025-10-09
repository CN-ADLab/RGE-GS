import torch
from einops import repeat
from scipy.spatial.transform import Rotation
import math
import numpy as np

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.graphics_utils import renderNormal
from utils.general_utils import calculateVoxelValidId

def prefilterVoxel(
    viewpoint_camera,
    gaussians,
    background,
    gt_image = None,
    scaling_modifier=1.0,
    depth_threshold=0.0001,
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = gaussians.get_anchor(viewpoint_camera.uid)
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation(viewpoint_camera.uid)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :2],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rad_mask = radii_pure > 0

    return rad_mask


def getConstrainNeuralGaussians(repeat_anchor, total_xyz, voxel_info):
    valid_curr_gaussian_pre_indices, valid_curr_gaussian_total_index = calculateVoxelValidId(repeat_anchor, voxel_info)
    if valid_curr_gaussian_pre_indices is None or valid_curr_gaussian_total_index is None:
        return None, None, None

    # get projection xyz from plane function
    pre_plane_normal = voxel_info.normal_array[1:-1]
    pre_plane_function = voxel_info.normal_array[1:]

    normal_pre = torch.from_numpy(
        np.transpose(pre_plane_normal[:, valid_curr_gaussian_pre_indices])
    ).cuda()
    normal_pre = normal_pre.reshape(-1, 3)

    plane_pre = torch.from_numpy(
        np.transpose(pre_plane_function[:, valid_curr_gaussian_pre_indices])
    ).cuda()
    plane_pre = plane_pre.reshape(-1, 4)

    select_neural_xyz = total_xyz[valid_curr_gaussian_total_index, :]
    select_neural_xyz = select_neural_xyz.reshape(-1, 3)

    dist = (torch.sum(normal_pre * select_neural_xyz, dim=1) + plane_pre[:, -1])
    dist = dist.unsqueeze(1)
    diff_point  = dist * normal_pre
    projected_point = select_neural_xyz - diff_point
    projected_point = projected_point.reshape(-1, 3)

    # get rotation from normal
    world_normal = torch.tensor([0, 0, 1], dtype=normal_pre.dtype, device=normal_pre.device).repeat(normal_pre.shape[0], 1)
    rot_axis = torch.cross(world_normal, normal_pre, dim=-1)
    dot_product = (world_normal * normal_pre).sum(dim=-1)
    rot_angle = torch.acos(dot_product)
    sin_half_angle = torch.sin(0.5 * rot_angle)
    cos_half_angle = torch.cos(0.5 * rot_angle)
    quaternions = torch.zeros(normal_pre.shape[0], 4, device=normal_pre.device, dtype=normal_pre.dtype)
    quaternions[:, 0] = cos_half_angle
    quaternions[:, 1:] = rot_axis * sin_half_angle.unsqueeze(-1) # q x y z

    return projected_point.float(), quaternions.float(), valid_curr_gaussian_total_index


def generateNeuralGaussians(viewpoint_camera, gaussians, voxel_info, visible_mask=None, jitter_trick=False, pre_mask=None):
    if visible_mask is None:
        visible_mask = torch.ones(
            gaussians._anchor.shape[0],
            dtype=torch.bool,
            device=gaussians._anchor.device,
        )

    feat = gaussians._anchor_feat[visible_mask]
    anchor = gaussians.get_anchor(viewpoint_camera.uid)[visible_mask]
    grid_offsets = gaussians.get_offset[visible_mask]
    grid_scaling = gaussians.get_scaling[visible_mask]

    # get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    # 
    if jitter_trick: 
        ob_view = ob_view + torch.normal(mean=0, std=0.01, size=ob_view.shape, device="cuda")

    # view-adaptive feature
    if gaussians.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = gaussians.get_featurebank_mlp(cat_view).unsqueeze(
            dim=1
        )  # [n, 1, 3]

        # multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = (
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
    if gaussians.appearance_dim > 0:
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * viewpoint_camera.uid
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = gaussians.get_appearance(camera_indicies)

    # get offset's opacity
    if gaussians.add_opacity_dist:
        neural_opacity = gaussians.get_opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = gaussians.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)
    if jitter_trick and pre_mask is not None:
        mask = pre_mask
    opacity = neural_opacity[mask]

    # get offset's color
    if gaussians.appearance_dim > 0:
        if gaussians.add_color_dist:
            color = gaussians.get_color_mlp(
                torch.cat([cat_local_view, appearance], dim=1)
            )
        else:
            color = gaussians.get_color_mlp(
                torch.cat([cat_local_view_wodist, appearance], dim=1)
            )
    else:
        if gaussians.add_color_dist:
            color = gaussians.get_color_mlp(cat_local_view)
        else:
            color = gaussians.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * gaussians.n_offsets, 3])  # [mask]

    # get offset's cov
    if gaussians.add_cov_dist:
        scale_rot = gaussians.get_cov_mlp(cat_local_view)
    else:
        scale_rot = gaussians.get_cov_mlp(cat_local_view_wodist)

    scale_rot = scale_rot.reshape([anchor.shape[0] * gaussians.n_offsets, 6])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(
        concatenated, "n (c) -> (n k) (c)", k=gaussians.n_offsets
    )
    concatenated_all = torch.cat(
        [concatenated_repeated, color, scale_rot, offsets], dim=-1
    )
    masked = concatenated_all[mask]

    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
        [5, 3, 3, 6, 3], dim=-1
    )
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :2]
    )
    rot = gaussians.rotation_activation(scale_rot[:, 2:6])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    plane_gaussian_index = None
    if voxel_info is not None:
        projected_point, quaternions, plane_gaussian_index = \
            getConstrainNeuralGaussians(repeat_anchor, xyz, voxel_info)
        if projected_point is not None:
            xyz[plane_gaussian_index, :] = projected_point
            rot[plane_gaussian_index, :] = quaternions

    return xyz, color, opacity, scaling, rot, neural_opacity, mask, plane_gaussian_index


def renderComposite(
    model_params,
    valid_model_info,
    scene_view,
    voxel_info,
    gt_image = None,
    retain_grad=True,
    points_buffer=None,
    depth_threshold=0.0001
):
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    total_opacity_dict = {}
    total_mask_dict = {}
    total_visable_dict = {}
    total_xyz = None
    total_color = None
    total_opacity = None
    total_scaling = None
    total_rot = None
    total_points_with_model_id = None
    total_screenspace_points = None

    total_xyz_grad = None
    total_color_grad = None
    total_opacity_grad = None
    total_scaling_grad = None
    total_rot_grad = None

    init = False
    plane_gaussian_index = None
    for model_info in valid_model_info:
        curr_points_with_model_id = None
        if isinstance(model_info.model_id, str) and model_info.model_id[0:6] == 'insert':
            if not retain_grad:
                model_xyz = model_info.gs_xyz
                model_color = model_info.gs_color
                model_opacity = model_info.gs_opacity
                model_scaling = model_info.gs_scaling
                model_rot = model_info.gs_rotation
                model_opacity = torch.ones(model_info.gs_opacity.shape, dtype=model_info.gs_opacity.dtype, device="cuda")
        else:
            model_visible_mask = prefilterVoxel(
                model_info.model_view,
                model_info.model_gaussians,
                background,
                gt_image,
                depth_threshold,
            )
            
            curr_voxel_info = None
            if model_info.model_id == 0:
                curr_voxel_info = voxel_info
            model_xyz, model_color, model_opacity, model_scaling, model_rot, neural_opacity, mask, curr_plane_gaussian_index = (
                generateNeuralGaussians(
                    model_info.model_view,
                    model_info.model_gaussians,
                    curr_voxel_info,
                    model_visible_mask
                )
            )
            ## jitter trick 
            model_xyz_jitter, model_color_jitter, model_opacity_jitter, model_scaling_jitter, model_rot_jitter, neural_opacity_jitter, mask_jitter, curr_plane_gaussian_index_jitter = (
                generateNeuralGaussians(
                    model_info.model_view,
                    model_info.model_gaussians,
                    curr_voxel_info,
                    model_visible_mask,
                    jitter_trick=True,
                    pre_mask=mask
                )
            )
            xyz_grad = torch.sum(torch.abs(model_xyz_jitter - model_xyz), dim=-1, keepdim=True) / 3
            color_grad = torch.sum(torch.abs(model_color_jitter - model_color), dim=-1, keepdim=True) / 3
            opacity_grad = torch.sum(torch.abs(model_opacity_jitter - model_opacity), dim=-1, keepdim=True) / 3
            scaling_grad = torch.sum(torch.abs(model_scaling_jitter - model_scaling), dim=-1, keepdim=True) / 3
            rot_grad = torch.sum(torch.abs(model_rot_jitter - model_rot), dim=-1, keepdim=True) / 3
            ### 
            if model_info.model_id == 0:
                plane_gaussian_index = curr_plane_gaussian_index

            total_opacity_dict[model_info.model_id] = neural_opacity
            total_mask_dict[model_info.model_id] = mask
            total_visable_dict[model_info.model_id] = model_visible_mask
            curr_points_with_model_id = torch.full((model_xyz.shape[0],), model_info.model_id)

        curr_screenspace_points = torch.zeros((model_xyz.shape[0], 4), dtype=torch.float32, requires_grad=model_info.need_train, device="cuda")
        model_to_world_r = (
            torch.from_numpy(model_info.model_pose[:3, :3]).cuda().to(torch.float32)
        )
        model_to_world_t = (
            torch.from_numpy(model_info.model_pose[:3, 3]).cuda().to(torch.float32)
        )
        model_to_world_q = Rotation.from_matrix(
            torch.from_numpy(model_info.model_pose[:3, :3])
        ).as_quat()  # x y z w
        w1 = model_to_world_q[3]
        x1 = model_to_world_q[0]
        y1 = model_to_world_q[1]
        z1 = model_to_world_q[2]

        w2 = model_rot[:, 0]
        x2 = model_rot[:, 1]
        y2 = model_rot[:, 2]
        z2 = model_rot[:, 3]
        trans_model_rot = torch.zeros(model_rot.shape, dtype = model_rot.dtype, device = model_rot.device)
        trans_model_rot[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        trans_model_rot[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        trans_model_rot[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        trans_model_rot[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        model_xyz = model_to_world_r @ model_xyz.t() + model_to_world_t.reshape(3, 1)

        if not init:
            init = True
            total_xyz = model_xyz.t()
            total_color = model_color
            total_opacity = model_opacity
            total_scaling = model_scaling
            total_rot = trans_model_rot
            total_screenspace_points = curr_screenspace_points
            if retain_grad:
                total_points_with_model_id = curr_points_with_model_id
            total_xyz_grad = xyz_grad
            total_color_grad = color_grad
            total_opacity_grad = opacity_grad
            total_scaling_grad = scaling_grad
            total_rot_grad = rot_grad
        else:
            total_xyz = torch.cat((total_xyz, model_xyz.t()), dim=0)
            total_color = torch.cat((total_color, model_color), dim=0)
            total_opacity = torch.cat((total_opacity, model_opacity), dim=0)
            total_scaling = torch.cat((total_scaling, model_scaling), dim=0)
            total_rot = torch.cat((total_rot, trans_model_rot), dim=0)
            total_screenspace_points = torch.cat((total_screenspace_points, curr_screenspace_points), dim=0)
            if retain_grad:
                total_points_with_model_id = torch.cat((total_points_with_model_id, curr_points_with_model_id), dim = 0)
            total_xyz_grad = torch.cat((total_xyz_grad, xyz_grad), dim=0)
            total_color_grad = torch.cat((total_color_grad, color_grad), dim=0)
            total_opacity_grad = torch.cat((total_opacity_grad, opacity_grad), dim=0)
            total_scaling_grad = torch.cat((total_scaling_grad, scaling_grad), dim=0)
            total_rot_grad = torch.cat((total_rot_grad, rot_grad), dim=0)

    if retain_grad:
        total_screenspace_points.retain_grad()

    tanfovx = math.tan(scene_view.FoVx * 0.5)
    tanfovy = math.tan(scene_view.FoVy * 0.5)
    render_normal = None

    raster_settings = GaussianRasterizationSettings(
        image_height=int(scene_view.image_height),
        image_width=int(scene_view.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        depth_threshold=depth_threshold,
        viewmatrix=scene_view.world_view_transform,
        projmatrix=scene_view.full_proj_transform,
        sh_degree=1,
        campos=scene_view.camera_center,
        prefiltered=False,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, allmap, pixels = rasterizer(
        means3D=total_xyz,
        means2D=total_screenspace_points,
        shs=None,
        colors_precomp=total_color,
        opacities=total_opacity,
        scales=total_scaling,
        rotations=total_rot,
        cov3D_precomp=None,
    )

    if not retain_grad:
        return {"render": rendered_image}

    rets = {
        "render": rendered_image,
        "viewspace_points": total_screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "points_with_model_id": total_points_with_model_id,
        "selection_mask": total_mask_dict,
        "visable_mask": total_visable_dict,
        "neural_opacity": total_opacity_dict,
        "scaling": total_scaling,
        "xyz": total_xyz,
        "plane_gaussian_index": plane_gaussian_index
    }

    world_normal, render_normal, surf_depth, surf_normal =\
        renderNormal(allmap, total_rot, scene_view, points_buffer)
    rets.update(
        {
            "render": rendered_image,
            "world_normal": world_normal,
            "render_normal": render_normal,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
            "total_xyz_grad": total_xyz_grad,
            "total_color_grad": total_color_grad,
            "total_opacity_grad": total_opacity_grad,
            "total_scaling_grad": total_scaling_grad,
            "total_rot_grad": total_rot_grad,
        }
    )
    return rets


def render(
    viewpoint_camera,
    gaussians,
    background,
    voxel_info,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
    depth_threshold=0.0001,
    points_buffer=None
):
    is_training = gaussians.get_color_mlp.training
    xyz, color, opacity, scaling, rot, neural_opacity, mask, plane_gaussian_index = \
            generateNeuralGaussians(viewpoint_camera, gaussians, voxel_info, visible_mask)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_normal = None
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=gaussians._anchor.dtype, requires_grad=True, device="cuda") + 0

    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, allmap, pixels = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)



    if is_training:
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "xyz": xyz,
            "plane_gaussian_index": plane_gaussian_index
        }
    else:
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    world_normal, render_normal, surf_depth, surf_normal =\
        renderNormal(allmap, rot, viewpoint_camera, points_buffer)
    rets.update(
        {
            "render": rendered_image,
            "world_normal": world_normal,
            "render_normal": render_normal,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )
    return rets
