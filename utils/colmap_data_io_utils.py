import os
import numpy as np
import collections
from plyfile import PlyData
from utils.graphics_utils import BasicPointCloud

random_sample_points_number = 100000

BaseImageParaLane = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "name"])
CameraParaLane = collections.namedtuple(
    "Camera", ["model", "width", "height", "params"])

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


class ImageDataParaLane(BaseImageParaLane):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def fetchPly(path):
    plydata = PlyData.read(os.path.join(path, "pcd_rescale/sparse/0/points3D.ply"))
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T

    if positions.shape[0] > random_sample_points_number:
        indices = np.random.choice(
            positions.shape[0],
            random_sample_points_number,
            replace=True,
        )
        positions = positions[indices]

    colors = np.zeros_like(positions)
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchMultiPly(path, track_list, mono_depth=None):
    positions = None
    for track_name in track_list:
        plydata = PlyData.read(os.path.join(path, track_name, "sparse/0/points3D.ply"))
        vertices = plydata["vertex"]
        curr_positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        if positions is None:
            positions = curr_positions
        else:
            positions = np.vstack((curr_positions, positions))

    if positions.shape[0] > random_sample_points_number:
        indices = np.random.choice(
            positions.shape[0],
            random_sample_points_number,
            replace=True,
        )
        positions = positions[indices]
    
    if mono_depth is not None:
        positions = np.vstack((positions, mono_depth))
    print("positions ", positions.shape)

    colors = np.zeros_like(positions)
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def readExtrinsicsText(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                image_name = elems[9]
                images[image_id] = ImageDataParaLane(
                    id=image_id, qvec=qvec, tvec=tvec,
                    name=image_name)
    return images


def readIntrinsicsText(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_channel = str(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_channel] = CameraParaLane(model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def debugProjections(cam_infos, pcd):
    id = 0
    for cam_info in cam_infos:
        world_to_camera_r = cam_info.R
        world_to_camera_t = cam_info.T
        intrinsic = np.eye(3)
        intrinsic[0, 0] = cam_info.K[0]
        intrinsic[1, 1] = cam_info.K[1]
        intrinsic[0, 2] = cam_info.K[2]
        intrinsic[1, 2] = cam_info.K[3]

        projections, _ = cv2.projectPoints(pcd.points, world_to_camera_r, world_to_camera_t, intrinsic, None)
        test_image = cam_info.img_mask

        for point in projections:
            cv2.circle(
                test_image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1
            )
        cv2.imwrite("test_image_" + str(id) + ".png", test_image)
        id += 1
    return    