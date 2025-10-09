import os
import re
import cv2
import copy
import shutil
import torch
import sys
import random
import numpy as np
from datetime import datetime
from errno import EEXIST
from os import makedirs, path
import subprocess


def install_package(package_name):
    pip_command = [sys.executable, "-m", "pip", "install", package_name]
    subprocess.run(
        pip_command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return


def removeEmptyFolder(model_path):
    for model_id in os.listdir(model_path):
        model_folder = os.path.join(model_path, model_id)
        if model_id.isdigit():
            for block_id in os.listdir(model_folder):
                block_folder = os.path.join(model_folder, block_id)
                if not os.listdir(block_folder):
                    shutil.rmtree(block_folder, ignore_errors=True)
            if not os.listdir(model_folder):
                shutil.rmtree(model_folder, ignore_errors=True)


def getFolderNameFromNoiseXYZ(diff_lidar_xyz):
    if diff_lidar_xyz is None:
        return 'x_0_y_0_z_0'
    folder_name = 'x_' + str(diff_lidar_xyz[0, 0]) + '_y_' + str(diff_lidar_xyz[0, 1]) + '_z_' + str(diff_lidar_xyz[0, 2])
    return folder_name

def getDilatedImageMask(image_mask, dilation_radius):
    image_mask = image_mask.transpose(1, 2, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_radius, dilation_radius))
    image_mask = cv2.dilate(image_mask, kernel)
    return image_mask.transpose(2, 0, 1)


def getModelList(args, is_render, block_id = 0):
    if args.data_type == "ParaLane":
        return [0]

    model_dir = os.path.join(
        args.source_path, "temp", args.cyber_log, "occ/preproc/dynamic/objects"
    )
    if not os.path.exists(model_dir):
        model_dir = os.path.join(args.source_path, "recon_related", args.cyber_log, "occ/preproc/dynamic/objects")
    model_id_list = obtainAllModelID(model_dir)
    if 0 not in model_id_list:
        model_id_list.append(0)
    model_id_list.sort()
    if not is_render:
        return model_id_list

    render_model_id_list = []
    for curr_model_id in model_id_list:
        pth_save_path = os.path.join(args.model_path, str(curr_model_id), str(block_id), "point_cloud")
        if os.path.exists(pth_save_path):
            render_model_id_list.append(curr_model_id)
    return render_model_id_list


def obtainAllModelID(model_dir):
    all_folders = os.listdir(model_dir)
    model_id_list = []
    for folder in all_folders:
        if re.fullmatch(r'\d+', folder):
            model_id_list.append(int(folder))

    return model_id_list


def foldModelList(model_id_list, fold_length):
    model_id_list_fold = []
    curr_list = []

    for id in model_id_list:
        curr_list.append(id)
        if len(curr_list) == fold_length:
            model_id_list_fold.append(copy.deepcopy(curr_list))
            curr_list.clear()

    if len(curr_list) > 0:
        model_id_list_fold.append(copy.deepcopy(curr_list))
    return model_id_list_fold


def updateParams(args):
    args.start_stat = int(args.iterations / 40.0)
    args.update_from = int(args.iterations * 0.2)
    args.update_interval = int(args.iterations / 200.0)
    args.update_until = int(args.iterations * 0.5)
    return


def computeHashIndex(voxel_points_id):
    voxel_points_id[:, 0] *= 73856093
    voxel_points_id[:, 1] *= 471943
    voxel_points_id[:, 2] *= 83492791
    points_index = (
        np.bitwise_xor(
            np.bitwise_xor(voxel_points_id[:, 0], voxel_points_id[:, 1]),
            voxel_points_id[:, 2],
        )
        % 10000000
    )
    return points_index


def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def inverseSigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def getExponlrFunc(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def stripLowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def stripSymmetric(sym):
    return stripLowerdiag(sym)


def buildRotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def buildScalingRotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = buildRotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def quaternionRawMultiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def safeState(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n"," [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def calculateVoxelValidId(gaussian_xyz, voxel_info):
    pre_voxel_val = voxel_info.normal_array[0]
    curr_gaussian_voxel_xyz = (
        (gaussian_xyz.cpu().detach().numpy() - voxel_info.origin_points) / voxel_info.voxel_size
    ).astype(int)
    
    curr_gaussian_voxel_val = computeHashIndex(curr_gaussian_voxel_xyz)
    pre_insert_id = np.searchsorted(pre_voxel_val, curr_gaussian_voxel_val)

    pre_voxel_val_aug = np.append(pre_voxel_val, 2**31 - 1)
    valid_indices = np.where(
        pre_voxel_val_aug[pre_insert_id] == curr_gaussian_voxel_val
    )[0]

    if valid_indices.shape[0] == 0:
        valid_curr_gaussian_pre_indices = None
        valid_curr_gaussian_total_index = None
    else:
        curr_gaussian_pre_indices = -1 * np.ones_like(curr_gaussian_voxel_val, dtype=int)
        curr_gaussian_pre_indices[valid_indices] = pre_insert_id[valid_indices]

        # get pre normal and curr normal
        valid_curr_gaussian_pre_indices = curr_gaussian_pre_indices[
            curr_gaussian_pre_indices >= 0
        ]
        curr_gaussian_total_index = np.arange(curr_gaussian_voxel_val.shape[0])
        valid_curr_gaussian_total_index = curr_gaussian_total_index[
            curr_gaussian_pre_indices >= 0
        ]
    return valid_curr_gaussian_pre_indices, valid_curr_gaussian_total_index
