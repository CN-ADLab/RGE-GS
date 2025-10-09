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
from arguments import getValidCamera, getValidCameraWaymo
from init.get_image_mask import getCameraMask
from utils.general_utils import mkdir_p


def videoInferenceProcess(model_folder, gs_folder, data_type):
    if data_type == "ParaLane":
        all_camera_channels = getValidCamera()
    elif data_type == "Waymo":
        all_camera_channels = getValidCameraWaymo()
    else:
        print("False data type")

    max_video_length = 25
    pipe = StableVideoDiffusionPipeline_convnb_multiframe.from_pretrained(
        model_folder, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    for camera_channel in all_camera_channels:
        print("Diffusion camera channel: ", camera_channel)
        gs_result_folder = os.path.join(gs_folder, camera_channel, 'render_result')
        if data_type == "ParaLane":
            image_shape = (1280, 720)
        elif data_type == "Waymo":
            image_shape = (1920, 1280)

        for folder_name in os.listdir(gs_result_folder):
            image_folder = os.path.join(gs_result_folder, folder_name)
            image_list = os.listdir(image_folder)
            image_list.sort()
            all_image_length = len(image_list)

            save_folder = os.path.join(gs_folder, camera_channel, 'diffusion_result', folder_name)
            if not os.path.exists(save_folder):
                mkdir_p(save_folder)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                os.path.join(save_folder, "output.mp4"),
                fourcc,
                10,
                image_shape,
            )

            loop_num = math.ceil(all_image_length / max_video_length)
            for curr_loop in range(loop_num):
                start_id = curr_loop * max_video_length
                print("Diffusion start id: ", start_id)

                pseudo_images = []
                for curr_id in range(0, max_video_length):
                    image_id = start_id + curr_id
                    if image_id >= len(image_list):
                        break

                    image_name = image_list[image_id]
                    control_img = Image.open(os.path.join(image_folder, image_name)).convert("RGB")
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
                    if image_id >= len(image_list):
                        break

                    img = diffusion_outputs[curr_id]
                    np_image = np.array(img)
                    np_image = cv2.resize(np_image, image_shape, interpolation=cv2.INTER_LINEAR)
                    np_image = np_image[:, :, ::-1]
                    video.write(np_image)

                    image_name = image_list[image_id]
                    diffusion_image_path = os.path.join(save_folder, image_name)
                    img = img.resize(image_shape)
                    img.save(diffusion_image_path)


if __name__ == "__main__":
    print("Process Video Diffusion")
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--diffusion_model_folder", type=str)
    parser.add_argument("--gs_folder", type=str)
    parser.add_argument("--data_type", type=str)
    args = parser.parse_args()

    videoInferenceProcess(args.diffusion_model_folder, args.gs_folder, args.data_type)
