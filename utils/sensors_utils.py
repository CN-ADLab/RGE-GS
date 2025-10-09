import numpy as np
import yaml


def getCameraExtrinsics(sensors_path, camera_channels):
    data = ''.join(open(sensors_path, 'r').readlines())
    car_cfg = loadYamlStr(data)
    sensors_extrinsics = obtainSensorsExtrinsics(car_cfg, camera_channels)
    camera_extrinsics = getBodytoCamera(camera_channels, sensors_extrinsics)
    return camera_extrinsics


def loadYamlStr(param):
    if not param:
        return ''
    param = param.replace('%YAML:1.0', '').replace('---', '').replace('...', '').replace('!!opencv-matrix', '').replace('!<tag:yaml.org,2002:opencv-matrix>', '')
    params = yaml.load(param, Loader=yaml.FullLoader)
    return params


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def obtainSensorsExtrinsics(car_cfg, camera_channels):
    sensors_extrinsics = {}
    sensors_channels = camera_channels.copy()
    sensors_channels.append("ROBO_TOP")
    for curr_channel in sensors_channels:
        original_sensor_channel = curr_channel
        if curr_channel[-7:] == "_resize":
            original_sensor_channel = curr_channel[0:-7]

        xyz = np.array(car_cfg[original_sensor_channel]['initial_transform']['translation'])
        rpy = car_cfg[original_sensor_channel]['initial_transform']['rpy']
        rotation = rotz(rpy[2] * np.pi / 180) @ roty(
            rpy[1] * np.pi / 180) @ rotx(rpy[0] * np.pi / 180)
        trans = np.eye(4)
        trans[:3, :3] = rotation
        trans[:3, 3] = xyz
        sensors_extrinsics[curr_channel] = trans
    return sensors_extrinsics


def getBodytoCamera(camera_channels, sensors_extrinsics):
    camera_extrinsics = {}
    lidar_to_body = sensors_extrinsics["ROBO_TOP"]
    for camera_channel in camera_channels:
        camera_to_lidar = sensors_extrinsics[camera_channel]
        camera_extrinsics[camera_channel] = np.linalg.inv(camera_to_lidar) @ np.linalg.inv(lidar_to_body)
    return camera_extrinsics
