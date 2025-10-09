import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from utils.general_utils import calculateVoxelValidId


def l1_loss(network_output, gt):
    diff_value = network_output - gt
    zero_index = torch.all(diff_value.eq(0), dim=0).sum().item()
    valid_number = (3 * (1910 * 1075 - zero_index))
    return torch.abs(diff_value).sum() / valid_number

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt))
    weighted_loss = loss * weight
    return weighted_loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def createWindow(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = createWindow(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculateVoxelLoss(world_normal, static_gaussian_xyz, voxel_info):
    if voxel_info is None:
        voxel_normal_loss = 0
        voxel_plane_loss = 0
    else:
        valid_curr_gaussian_pre_indices, valid_curr_gaussian_total_index = calculateVoxelValidId(static_gaussian_xyz, voxel_info)

        if valid_curr_gaussian_pre_indices is None:
            voxel_normal_loss = 0.001
            voxel_plane_loss = 0.001
        else:
            pre_plane_normal = voxel_info.normal_array[1:-1]
            pre_plane_function = voxel_info.normal_array[1:]

            normal_pre = torch.from_numpy(
                np.transpose(pre_plane_normal[:, valid_curr_gaussian_pre_indices])
            ).cuda()

            if world_normal is None:
                voxel_normal_loss = 0
            else:
                curr_selected_normal = world_normal[valid_curr_gaussian_total_index]
                cos_value = torch.sum(curr_selected_normal * normal_pre, dim=1)
                voxel_normal_loss = (1 - cos_value.abs()).mean()

            plane_pre = torch.from_numpy(
                np.transpose(pre_plane_function[:, valid_curr_gaussian_pre_indices])
            ).cuda()
            curr_selected_xyz = static_gaussian_xyz[valid_curr_gaussian_total_index]

            dist = (
                torch.sum(normal_pre * curr_selected_xyz, dim=1) + plane_pre[:, -1]
            )
            voxel_plane_loss = dist.abs().mean()
    return voxel_normal_loss, voxel_plane_loss


def getInpaintingLoss(iteration, update_until, viewpoint_cam, render_dict, image):
    if iteration > update_until:
        eval_image = image * torch.from_numpy(viewpoint_cam.img_mask).cuda()        # hard mask
        gt_image = viewpoint_cam.original_image.cuda() * torch.from_numpy(viewpoint_cam.img_mask).cuda()
    else:
        eval_image = image * torch.from_numpy(viewpoint_cam.img_mask_inpainting).cuda()      # soft mask
        gt_image = viewpoint_cam.original_image_inpainting.cuda()

    # TODO: loss for color change
    diff_mask = viewpoint_cam.img_mask[0, :, :].astype(bool) ^ viewpoint_cam.img_mask_inpainting[0, :, :].astype(bool)
    diff_mask = diff_mask.astype(np.uint8)

    if iteration > update_until:
        eval_image_previous = render_dict[viewpoint_cam].cuda()
        # compute diff mask
        erode_kernel_size = (iteration - update_until) // 600 + 1
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        diff_mask_erode = cv2.erode(diff_mask, erode_kernel)
        if erode_kernel_size > 0:
            diff_mask_previous = cv2.erode(diff_mask, np.ones((erode_kernel_size - 1, erode_kernel_size - 1), np.uint8))
        else:
            diff_mask_previous = diff_mask
        diff_mask_step_hw = diff_mask_erode.astype(bool) ^ diff_mask_previous.astype(bool)
        diff_mask_step = np.repeat(diff_mask_step_hw[np.newaxis, :, :], 3, axis=0)
        
        # TODO: propogate patch replacement

        # add soft l1_loss weight & weighted l1_loss
        weight = np.ones_like(diff_mask_step_hw, dtype=np.float32)
        weight[viewpoint_cam.img_mask_inpainting[0, :, :] < 1] = 0
        # weight[diff_mask_step_hw == 1] = 0.3
        weight = np.repeat(weight[np.newaxis, :, :], 3, axis=0)
        Ll1 = weighted_l1_loss(eval_image, gt_image, torch.from_numpy(weight).cuda())
        ssim_loss = 1.0 - ssim(eval_image * torch.from_numpy(viewpoint_cam.img_mask_inpainting).cuda(), gt_image * torch.from_numpy(viewpoint_cam.img_mask_inpainting).cuda())

        # add color change loss
        eval_image_current_mask = image.clone() * torch.from_numpy(diff_mask_step).cuda()
        eval_image_previous_mask = eval_image_previous.clone() * torch.from_numpy(diff_mask_step).cuda()
        
        cc_loss = l1_loss(eval_image_current_mask, eval_image_previous_mask)
        render_dict[viewpoint_cam].grad = None
        del render_dict[viewpoint_cam]
        torch.cuda.empty_cache()
    else:
        if viewpoint_cam in render_dict:
            render_dict[viewpoint_cam].grad = None
            del render_dict[viewpoint_cam]
            torch.cuda.empty_cache()
        Ll1 = l1_loss(eval_image, gt_image)
        ssim_loss = 1.0 - ssim(eval_image, gt_image)
        cc_loss = 0
    
    return Ll1, ssim_loss, cc_loss