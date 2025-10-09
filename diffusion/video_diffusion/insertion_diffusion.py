import os
import cv2
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
from utils.general_utils import install_package
install_package("diffusers[torch]==0.28.0")
install_package("huggingface_hub==0.25.2")

import math
import torch
import numpy as np
from PIL import Image
import torch.utils.checkpoint
from argparse import ArgumentParser
from src.pipelines.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipeline_convnb_multiframe
from arguments import getValidCamera
from init.get_image_mask import getCameraMask
from utils.general_utils import mkdir_p


def getAllCameraMask():
    cameras_mask = {}
    valid_camera_channels = getValidCamera()
    for camera_channel in valid_camera_channels:
        camera_mask = getCameraMask(camera_channel, (384, 960))
        cameras_mask[camera_channel] = np.repeat(camera_mask[:, :, np.newaxis], 3, axis=2)
    return cameras_mask


def videoInferenceProcess(model_folder, images_folder):
    max_video_length = 25
    cameras_mask = getAllCameraMask()
    pipe = StableVideoDiffusionPipeline_convnb_multiframe.from_pretrained(
        model_folder, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    for camera_channel, camera_mask in cameras_mask.items():
        print("Diffusion camera channel: ", camera_channel)
        time_list = os.listdir(images_folder)
        time_list.sort()
        all_image_length = len(time_list)

        if "PANO" in camera_channel:
            image_shape = (640, 360)
        else:
            image_shape = (1280, 720)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(camera_channel + "_vdm.mp4"),
            fourcc,
            10,
            image_shape,
        )

        loop_num = math.ceil(all_image_length / max_video_length)
        for curr_loop in range(loop_num):
            pseudo_images = []
            start_id = curr_loop * max_video_length

            for curr_id in range(0, max_video_length):
                image_id = start_id + curr_id
                if image_id >= all_image_length:
                    break

                curr_img_path = os.path.join(images_folder, time_list[image_id], camera_channel + ".png")
                control_img = Image.open(curr_img_path).convert("RGB")
                control_img = control_img.resize((int(960),int(384)))
                pseudo_images.append(control_img)

            generator = torch.manual_seed(42)
            curr_video_length = len(pseudo_images)
            print("pseudo images length: ", curr_video_length)

            frames = pipe(pseudo_images, origin_image=None, width=960, height=384, \
                          num_frames=curr_video_length, num_inference_steps=25, \
                          min_guidance_scale=2.0, max_guidance_scale=2.0, noise_aug_strength=0.02, generator=generator)
            diffusion_outputs = frames.frames[0]

            for curr_id in range(0, curr_video_length):
                image_id = start_id + curr_id
                if image_id >= all_image_length:
                    break

                img = diffusion_outputs[curr_id]
                np_image = np.array(img) * camera_mask
                np_image = cv2.resize(np_image, image_shape, interpolation=cv2.INTER_LINEAR)
                np_image = np_image[:, :, ::-1]
                video.write(np_image)

                diffusion_image_path = os.path.join(images_folder, time_list[image_id], camera_channel + "_vdm.png")
                img = img.resize(image_shape)
                img.save(diffusion_image_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--diffusion_model_folder", type=str)
    parser.add_argument("--images_folder", type=str)
    args = parser.parse_args()

    videoInferenceProcess(args.diffusion_model_folder, args.images_folder)
