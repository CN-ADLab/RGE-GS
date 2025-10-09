import os
from enum import Enum
from argparse import ArgumentParser


class WorkMode(Enum):
    TRAIN_NSG = 1
    INFERENCE = 2
    SIMULATION = 3


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            t = type(value)
            value = value if not fill_none else None

            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])

        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size = 0
        self.update_depth = 2
        self.update_init_factor = 2
        self.update_hierachy_factor = 2

        self.use_feat_bank = True
        self.source_path = ""
        self.model_path = ""
        self.cyber_log = ""

        self.work_mode = WorkMode.TRAIN_NSG.value
        self.block_id = 0
        self.train_frame_number = 0
        self.train_frame_times = []
        self.invalid_render_frame_times = {}

        self.init_voxel_size = 1.0
        self.least_cam_info_length = 5

        self.camera_channel = []
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.ratio = 1
        self.undistorted = False

        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        self.use_sfm_pose = True
        self.use_radius_sample = True

        self.use_pose_correction = False
        self.train_from_checkpoint = False
        self.start_id = 0

        self.data_type = "ParaLane" 
        self.para_lane_track_list = ["track_0", "track_1", "track_2"]
        self.para_lane_single_length = 240
        self.para_lane_use_diffusion = False

        self.diffusion_pipeline = False
        self.use_confidence = False
        self.only_use_GS_0 = False
        self.use_mono_depth_init = False
        self.diffusion_folder = ""

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10000
        self.nsg_max_iterations = 100_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.offset_lr_init = 0.001
        self.offset_lr_final = 0.00001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_normal = 0.01

        self.voxel_loss_ratio = 100
        self.pose_loss_ratio = 0
        self.conf_loss_ratio = 1.0
        self.scale_loss_ratio = 0
        self.depth_loss_ratio = 0

        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15000

        self.min_opacity = 0.005
        self.success_threshold = 0.6
        self.densify_grad_threshold = 0.0002
        self.abs_grad_f = 1.5  # 如果使用abs_grad 2 / 3 / 4 值越大抑制分裂  不使用则设置为1

        super().__init__(parser, "Optimization Parameters")


def getValidCamera():
    valid_camera_channels = [
        "CAMERA_FRONT_resize",
    ]
    return valid_camera_channels

def getValidCameraWaymo():
    valid_camera_channels = [
        "FRONT",
        # "FRONT_LEFT", 
        # "FRONT_RIGHT",
        # "SIDE_LEFT",
        # "SIDE_RIGHT"
    ]
    return valid_camera_channels
