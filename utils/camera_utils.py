import torch
import numpy as np

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


def loadCam(id, cam_info):
    gt_image = None
    gt_image_inpainting = None
    image_mask_inpainting = None
    if cam_info.image is not None:
        resolution = (cam_info.width, cam_info.height)
        torch_image = PILtoTorch(cam_info.image, resolution)
        gt_image = torch_image[:3, ...]

    image_mask = None
    if cam_info.img_mask is not None:
        image_mask = cam_info.img_mask.astype(bool)
        image_mask = np.repeat(image_mask[np.newaxis, :, :], 3, axis=0)

    img_mask_wo_conf = None
    if cam_info.img_mask_wo_conf is not None:
        img_mask_wo_conf = cam_info.img_mask_wo_conf.astype(bool)
        img_mask_wo_conf = np.repeat(img_mask_wo_conf[np.newaxis, :, :], 3, axis=0)
    
    camera_mask = None
    if cam_info.camera_mask is not None:
        camera_mask = cam_info.camera_mask.astype(bool)
        camera_mask = np.repeat(camera_mask[np.newaxis, :, :], 3, axis=0)

    image_depth = None
    if cam_info.depth is not None:
        image_depth = torch.from_numpy(cam_info.depth[None, ...]).float()

    project_image = None
    if cam_info.project_image is not None:
        project_image = PILtoTorch(cam_info.project_image, resolution)[:3, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        intrinsic=cam_info.K,
        original_image=gt_image,
        original_image_inpainting=gt_image_inpainting,
        img_mask=image_mask,
        camera_mask=camera_mask,
        img_mask_inpainting=image_mask_inpainting,
        img_depth=image_depth,
        image_width=cam_info.width,
        image_height=cam_info.height,
        image_name=cam_info.image_name,
        camera_name=cam_info.camera_name,
        uid=id,
        project_image=project_image,
        img_mask_wo_conf = img_mask_wo_conf
    )


def cameraListFromcamInfos(cam_infos):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(id, c))
    return camera_list


def cameraToJSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    return_pt=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    if return_pt:
        c2w = torch.from_numpy(c2w).to(torch.float32)

    return c2w


def getUniformPoses(num_images, camera_dist, translation, coord="opengl"):
    points = []
    for idx in range(num_images):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append(np.array([x, y, z]))

    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    poses = []

    for p in points:
        pose = get_c2w_from_up_and_look_at(
            up, look_at, p * camera_dist, return_pt=False
        )
        if coord == "opengl":
            pose[..., 1:3] *= -1
        pose[:3, 3] += translation
        poses.append(pose)

    poses = np.stack(poses, axis=0)

    return poses


def getHorizontalPoses(num_images, camera_dist, translation,coord="opengl"):
    points = []
    camera_dist_list=[]
    for _ in range(num_images):
        # front view xz=0 y=-1 theta = 1.5pi
        camera_dist_list.append(camera_dist*np.random.uniform(1.00, 1.00))
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-0.10, 0.4))
        # theta = 1.0 * np.pi
        # phi = np.arccos(0.5)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append(np.array([x, y, z]))

    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    poses = []

    for idx, p in enumerate(points):
        pose = get_c2w_from_up_and_look_at(
            up, look_at, p * camera_dist_list[idx], return_pt=False
        )
        if coord == "opengl":
            pose[..., 1:3] *= -1
        pose[:3, 3] += translation
        poses.append(pose)

    poses = np.stack(poses, axis=0)

    return poses


def getSemanticMaskFromNpz(label_npz, sem_val):
    label_data = np.load(label_npz)["data"]
    sem_mask = label_data == sem_val
    return sem_mask.astype(np.uint8)


def getFlowSamMaskFromNpy(mask_npy):
    sem_mask = np.load(mask_npy) == 0
    return sem_mask.astype(np.uint8)


def obtainInpaintingInfo(label_npz_dir, camera_channel, soft_mask, sem_mask, original_img, camera_mask):
    if not os.path.exists(os.path.join(label_npz_dir, str(camera_channel + ".npy"))):
        print("No flow-guided semantic mask!")
        exit(1)
    # erode semantic_mask
    soft_mask = cv2.resize(soft_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    soft_mask = soft_mask.astype(bool)
    soft_mask = np.repeat(
        soft_mask[np.newaxis, :, :], 3, axis=0
    )
    soft_mask = soft_mask.astype(np.uint8)
    soft_mask = soft_mask.transpose(1, 2, 0)
    morph_radius = int(21 * original_img.shape[0] / 1080.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_radius, morph_radius))
    soft_mask = cv2.erode(soft_mask, kernel)
    soft_mask = soft_mask[:, :, 0]

    sem_mask = sem_mask.astype(bool)
    sem_mask = np.repeat(
        sem_mask[np.newaxis, :, :], 3, axis=0
    )
    sem_mask = sem_mask.astype(np.uint8)
    sem_mask = sem_mask.transpose(1, 2, 0)
    morph_radius = int(15 * original_img.shape[0] / 1080.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_radius, morph_radius))
    sem_mask = cv2.erode(sem_mask, kernel)
    sem_mask = sem_mask[:, :, 0]

    image_mask_inpainting = camera_mask * soft_mask
    scene_image_soft = original_img * np.repeat(
        image_mask_inpainting[:, :, np.newaxis], 3, axis=2
    )
    img_data_inpainting = Image.fromarray(cv2.cvtColor(scene_image_soft, cv2.COLOR_BGR2RGB))

    image_mask = camera_mask * sem_mask
    scene_image = original_img * np.repeat(
        image_mask[:, :, np.newaxis], 3, axis=2
    )
    img_data = Image.fromarray(cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB))
    
    return image_mask_inpainting, img_data_inpainting, img_data, image_mask