import os
import cv2
import json
import numpy as np


undistort_info = {"CAMERA_PANO_FRONT": {"fov_w" : 180.0, "fov_h" : 56.0, "ratio_cty" : 0.65, "max_extra_height" : 150},
                  "CAMERA_PANO_BACK": {"fov_w" : 180.0, "fov_h" : 56.0, "ratio_cty" : 0.65, "max_extra_height" : 150},
                  "CAMERA_PANO_LEFT": {"fov_w" : 180.0, "fov_h" : 77.4, "ratio_cty" : 0.60, "max_extra_height" : 450},
                  "CAMERA_PANO_RIGHT": {"fov_w" : 180.0, "fov_h" : 77.4, "ratio_cty" : 0.60, "max_extra_height" : 450}}


def obtainPinholeImage(sem_mask, camera_channel, undistort_image_dir, timestamp, info):
    camera_channel = camera_channel[ : -7]
    intrinsic = info["cams_info"][camera_channel]["camera_intrinsics"][:3, :3]
    distortion = np.array(info["cams_info"][camera_channel]['distortion_coefficients'])
    ori_image_path = os.path.join(undistort_image_dir, str(timestamp),  camera_channel + ".png")
    original_img = cv2.imread(ori_image_path)
    sem_mask = cv2.resize(sem_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return original_img, sem_mask, intrinsic
