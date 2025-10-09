import os
import cv2
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
from utils.general_utils import install_package
install_package("diffusers[torch]==0.31.0")

import torch
import numpy as np
from PIL import Image
import torch.utils.checkpoint
from argparse import ArgumentParser
from torchvision import transforms
from diffusers import UniPCMultistepScheduler
from src.pipeline_controlnet import StableDiffusionControlNetPipeline
from arguments import getValidCamera, getValidCameraWaymo
from init.get_image_mask import getCameraMask


def imageInferenceProcess(model_folder, gs_folder, data_type):
    if data_type == "ParaLane":
        all_camera_channels = getValidCamera()
    elif data_type == "Waymo":
        all_camera_channels = getValidCameraWaymo()
    else:
        print("False data type")

    for camera_channel in all_camera_channels:
        diffusion_model_folder = model_folder

        if data_type == "ParaLane":
            image_shape = (1280, 720)
        elif data_type == "Waymo":
            image_shape = (1920, 1280)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_model_folder,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_model_cpu_offload()

        gs_result_folder = os.path.join(gs_folder, camera_channel, 'render_result')
        for folder_name in os.listdir(gs_result_folder):
            image_folder = os.path.join(gs_result_folder, folder_name)
            for image_name in os.listdir(image_folder):
                control_img_path = os.path.join(image_folder, image_name)
                control_img = Image.open(control_img_path)
                conditioning_pixel_values = control_img.convert("RGB")

                conditioning_image_transforms = transforms.Compose([
                    transforms.Resize([720,1280], interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor()])
                conditioning_pixel_values = conditioning_image_transforms(conditioning_pixel_values)
                conditioning_pixel_values = torch.unsqueeze(conditioning_pixel_values, 0).to(torch.float32)

                generator = torch.manual_seed(0)
                image = pipeline("A realistic autonomous driving scene", num_inference_steps=20, generator=generator, image=conditioning_pixel_values).images[0]

                np_image = np.array(image)
                np_image = cv2.resize(np_image, image_shape, interpolation=cv2.INTER_LINEAR)
                mask_image = Image.fromarray(np_image)

                save_folder = os.path.join(gs_folder, camera_channel, 'diffusion_result', folder_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                diffusion_image_path = os.path.join(gs_folder, camera_channel, 'diffusion_result', folder_name, image_name)
                mask_image.save(diffusion_image_path)


if __name__ == "__main__":
    print("Process Image Diffusion")
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--diffusion_model_folder", type=str)
    parser.add_argument("--gs_folder", type=str)
    parser.add_argument("--data_type", type=str)
    args = parser.parse_args()

    imageInferenceProcess(args.diffusion_model_folder, args.gs_folder, args.data_type)
