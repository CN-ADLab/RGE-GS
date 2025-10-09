import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        original_image,
        original_image_inpainting,
        image_width,
        image_height,
        img_mask,
        camera_mask,
        img_mask_inpainting,
        img_depth,
        image_name,
        camera_name,
        uid,
        intrinsic,
        project_image,
        img_mask_wo_conf
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.camera_name = camera_name
        self.intrinsic = intrinsic
        self.image_width = image_width
        self.image_height = image_height

        self.img_depth = img_depth
        self.img_mask = img_mask
        self.camera_mask = camera_mask
        self.img_mask_inpainting = img_mask_inpainting
        self.project_image = project_image

        self.img_mask_wo_conf = None
        if img_mask_wo_conf is not None:
            self.img_mask_wo_conf=torch.from_numpy(img_mask_wo_conf).cuda()

        self.zfar = 1000.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        self.original_image = None
        if original_image is not None:
            self.original_image = original_image.clamp(0.0, 1.0)

        self.original_image_inpainting = None
        if original_image_inpainting is not None:
            self.original_image_inpainting = original_image_inpainting.clamp(0.0, 1.0)

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
