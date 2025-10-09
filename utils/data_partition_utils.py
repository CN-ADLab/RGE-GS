import os
import sys
import math
import json
import pickle
import numpy as np

sys.path.append('..')
from utils.pose_utils import getLidarPose
from arguments import getValidCamera
from collections import OrderedDict
from .colmap_data_io_utils import readExtrinsicsText, qvec2rotmat

chunk_max_image = 150
chunk_min_image = 30

def getWaymoInfo(args):
    pkl_file_name = args.cyber_log + ".pkl"
    meta_file = os.path.join(args.source_path, "meta_infos", pkl_file_name)
    meta_info = pickle.load(open(meta_file, "rb"))
    print("Total frame number: ", len(meta_info["frames"]))
    print("Start ID: ", args.start_id)
    
    max_x = -sys.maxsize
    max_y = -sys.maxsize
    min_x = sys.maxsize
    min_y = sys.maxsize
    time_with_pose = {}
    for frame_ind, info in enumerate(meta_info["frames"]):
        if frame_ind < args.start_id:
            continue

        lidar_to_world = info["lidar2world"]
        min_x = min(min_x, float(lidar_to_world[0, 3]))
        min_y = min(min_y, float(lidar_to_world[1, 3]))
        max_x = max(max_x, float(lidar_to_world[0, 3]))
        max_y = max(max_y, float(lidar_to_world[1, 3]))
        time_with_pose[str(frame_ind).zfill(3)] = (float(lidar_to_world[0, 3]), float(lidar_to_world[1, 3]))

        if len(time_with_pose) == args.para_lane_single_length - 1:
            break

    min_x = math.floor(min_x)
    min_y = math.floor(min_y)
    max_x = math.ceil(max_x)
    max_y = math.ceil(max_y)
    print("Delta x: ", max_x - min_x)
    print("Delta y: ", max_y - min_y)
    return time_with_pose, min_x, min_y, max_x, max_y


def getParaLaneInfo(args):
    max_x = -sys.maxsize
    max_y = -sys.maxsize
    min_x = sys.maxsize
    min_y = sys.maxsize
    time_with_pose = {}

    init_pose = None
    for track_name in args.para_lane_track_list:
        data_path = os.path.join(args.source_path, track_name)
        cameras_extrinsic_file = os.path.join(data_path, "sparse/0", "images_CAMERA_FRONT.txt")
        cam_extrinsics = readExtrinsicsText(cameras_extrinsic_file)

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            image_name = extr.name
            timestamp = extr.name.split("/")[0]
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = qvec2rotmat(extr.qvec)
            world_to_camera[:3, 3] = np.array(extr.tvec)
            
            if init_pose is None:
                init_pose = world_to_camera
                camera_pose = np.eye(4)
            else:
                camera_to_world = np.linalg.inv(world_to_camera)
                camera_pose = init_pose @ camera_to_world

            time_with_pose[int(timestamp)] = (float(camera_pose[0, 3]), float(camera_pose[2, 3]))
            min_x = min(min_x, float(camera_pose[0, 3]))
            min_y = min(min_y, float(camera_pose[2, 3]))
            max_x = max(max_x, float(camera_pose[0, 3]))
            max_y = max(max_y, float(camera_pose[2, 3]))

            if idx == args.para_lane_single_length - 1:
                break

    min_x = math.floor(min_x)
    min_y = math.floor(min_y)
    max_x = math.ceil(max_x)
    max_y = math.ceil(max_y)
    print("Delta x: ", max_x - min_x)
    print("Delta y: ", max_y - min_y)

    return time_with_pose, min_x, min_y, max_x, max_y, init_pose


def dataPartition(args, root_model_path):
    if args.data_type == "Waymo":
        use_downsample = False
        chunk_size = 50
        expand_size = 10
        time_with_pose, min_x, min_y, max_x, max_y = getWaymoInfo(args)
    elif args.data_type == "ParaLane":
        use_downsample = False
        chunk_size = 25
        expand_size = 5
        time_with_pose, min_x, min_y, max_x, max_y, init_pose = getParaLaneInfo(args)
        np.savetxt(os.path.join(root_model_path, "para_lane_init_pos.txt"), init_pose, delimiter=",", fmt="%.2f")
    else:
        exit(1)

    block_id_with_rect = {}
    block_time_with_extend = {}
    block_time_without_extend = {}
    block_height = 0
    if len(time_with_pose) < chunk_max_image:
        block_time_with_extend[0] = []
        for time, pose in time_with_pose.items():
            block_time_with_extend[0].append(time)
        block_time_without_extend = block_time_with_extend
        block_id_with_rect[0] = [min_x, min_y, max_x, max_y]
    else:
        # obtain nose to tail block id
        '''
        block id example:

        3 | 4 | 9
        ---------
        2 | 5 | 8
        ---------
        1 | 6 | 7
        '''
        block_id_with_extend_rect = {}
        block_height = math.ceil((max_y - min_y) / chunk_size)
        cols = 0
        for curr_min_x in range(min_x, max_x, chunk_size):
            cols += 1
            rows = 0
            for curr_min_y in range(min_y, max_y, chunk_size):
                curr_max_x = curr_min_x + chunk_size
                curr_max_y = curr_min_y + chunk_size

                rows += 1
                if cols % 2 == 0:
                    curr_block_id = cols * block_height - rows + 1
                else:
                    curr_block_id = (cols - 1) * block_height + rows

                block_id_with_rect[curr_block_id] = [curr_min_x, curr_min_y, curr_max_x, curr_max_y]
                block_id_with_extend_rect[curr_block_id] = [
                    curr_min_x - expand_size, curr_min_y - expand_size,
                    curr_max_x + expand_size, curr_max_y + expand_size]

        # correspond time with block id
        block_id_with_timelist = {}
        last_pos = np.array([0, 0])
        for timestamp, pos in time_with_pose.items():
            curr_x = float(pos[0])
            curr_y = float(pos[1])
            curr_pos = np.array([curr_x, curr_y])
            if np.linalg.norm(curr_pos - last_pos) < 0.1:
                continue

            last_pos = curr_pos
            for block_id, rect in block_id_with_extend_rect.items():
                if curr_x > rect[0] and curr_x < rect[2] and curr_y > rect[1] and curr_y < rect[3]:
                    block_id_with_timelist.setdefault(block_id, []).append(timestamp)
        block_id_with_timelist = OrderedDict(sorted(block_id_with_timelist.items()))

        # merge and get train block
        last_key = list(block_id_with_timelist.keys())[-1]
        prev_values = []
        print("Train block info")
        for key, values in block_id_with_timelist.items():
            curr_value = values + prev_values
            if key != last_key:
                if len(curr_value) < chunk_min_image:
                    prev_values += values
                    continue

            if use_downsample and len(curr_value) > chunk_max_image * 1.5:
                curr_value.sort()
                interval = max(int(len(curr_value) / chunk_max_image), 2)
                curr_value = [curr_value[i] for i in range(0, len(curr_value), interval)]

            block_time_with_extend[key] = sorted(list(set(curr_value)))
            prev_values = []

        # remove invalid area
        new_block_id_with_rect = {}
        for block_id in block_id_with_rect.keys():
            if block_id in block_time_with_extend.keys():
                new_block_id_with_rect[block_id] = block_id_with_rect[block_id]
        block_id_with_rect = new_block_id_with_rect

        # get render block
        print("Render block info")
        for timestamp, pos in time_with_pose.items():
            pos_x = float(pos[0])
            pos_y = float(pos[1])
            distance = sys.maxsize
            result_block_id = -1

            for block_id, area in block_id_with_rect.items():
                min_x = float(area[0])
                min_y = float(area[1])
                max_x = float(area[2])
                max_y = float(area[3])
                mid_x = (min_x + max_x) / 2
                mid_y = (min_y + max_y) / 2

                curr_distance = (pos_x - mid_x) * (pos_x - mid_x) + (pos_y - mid_y) * (pos_y - mid_y)
                if curr_distance < distance:
                    distance = curr_distance
                    result_block_id = block_id

            block_time_without_extend.setdefault(result_block_id, []).append(timestamp)
        block_time_without_extend = OrderedDict(sorted(block_time_without_extend.items()))

    # save json files
    block_info = {'block_time_with_extend': block_time_with_extend, 'block_time_without_extend': block_time_without_extend,\
                 'block_id_with_rect': block_id_with_rect, 'block_height': block_height}
    block_info_json = os.path.join(root_model_path, 'block_info.json')
    with open(block_info_json, 'w', encoding='utf-8') as file:
        json.dump(block_info, file, ensure_ascii=False, indent=4)
    return block_time_with_extend, block_time_without_extend


def getBlockInfo(root_model_path):    
    block_info_json = os.path.join(root_model_path, 'block_info.json')
    if os.path.exists(block_info_json):
        with open(block_info_json, "r", encoding="utf-8") as file:
            block_info = json.load(file)

        block_time_with_extend = block_info['block_time_with_extend']
        block_time_without_extend = block_info['block_time_without_extend']
        block_id_with_rect = block_info['block_id_with_rect']
        block_height = int(block_info['block_height'])

    else:
        info_json_file = os.path.join(root_model_path, 'block_time_with_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_time_with_extend = json.load(file)
        info_json_file = os.path.join(root_model_path, 'block_time_without_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_time_without_extend = json.load(file)
        info_json_file = os.path.join(root_model_path, 'block_area_without_extend.json')
        with open(info_json_file, "r", encoding="utf-8") as file:
            block_id_with_rect = json.load(file)
        block_height = 0
        
    return block_time_with_extend, block_time_without_extend, block_id_with_rect, block_height


def addDiffusionDataToBlock(root_model_path):
    # get block info from original data
    block_time_with_extend, block_time_without_extend, _, _ = getBlockInfo(root_model_path)
    
    # add diffusion info
    valid_camera_channels = getValidCamera()
    for camera_channel in valid_camera_channels:
        diffusion_result_dir = os.path.join(root_model_path, camera_channel, 'diffusion_result')
        if not os.path.exists(diffusion_result_dir):
            continue

        diffusion_track_list = os.listdir(diffusion_result_dir)
        for curr_track in diffusion_track_list:
            curr_folder = os.path.join(diffusion_result_dir, curr_track)
            pkl_path = os.path.join(curr_folder, '0_diffusion_pose.pkl')
            pkl_info = pickle.load(open(pkl_path, "rb"))

            for key, value in pkl_info.items():
                timestamp = int(key.split("_")[0])
                for block_id, time_list in block_time_with_extend.items():
                    if timestamp in time_list:
                        time_list.append(key)

                for block_id, time_list in block_time_without_extend.items():
                    if timestamp in time_list:
                        time_list.append(key)
        break
    return block_time_with_extend
