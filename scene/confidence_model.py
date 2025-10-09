import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.confidence_utils import computeReprojectionError
from utils.general_utils import PILtoTorch
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

class ConfidenceNetwork(nn.Module):
    def __init__(self, in_channels=3, init_features=32):
        super().__init__()
        features = init_features
        
        # Encoder Path
        self.encoder1 = ConfidenceNetwork._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConfidenceNetwork._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConfidenceNetwork._block(features*2, features*4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConfidenceNetwork._block(features*4, features*8, name="bottleneck")
        
        # Decoder Path
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = ConfidenceNetwork._block(features*8, features*4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = ConfidenceNetwork._block(features*4, features*2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = ConfidenceNetwork._block(features*2, features, name="dec1")
        
        # Final output
        self.conv = nn.Conv2d(
            in_channels=features, 
            out_channels=1, 
            kernel_size=1,
            padding=0  # No padding for 1x1 conv
        )
        # nn.init.constant_(self.conv.weight, 0)
        # nn.init.constant_(self.conv.bias, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        return self.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True)
        )
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        return block

    def save_state_dict(self, path):
        state_dict = {
            'params': self.state_dict()
        }
        torch.save(state_dict, os.path.join(path, "confidence_net.pt"))

    def load_state_dict(self, path):
        ckpt_path = os.path.join(path, "confidence_net.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            super().load_state_dict(state_dict['params'])

    def training_setup(self, lr_init=5e-4, lr_final=1e-5, lr_delay=1000):
        params_group = [
            {'params': self.encoder1.parameters(), 'weight_decay': 1e-3},
            {'params': self.encoder2.parameters(), 'weight_decay': 1e-3},
            {'params': self.encoder3.parameters(), 'weight_decay': 1e-3},
            {'params': self.bottleneck.parameters(), 'weight_decay': 1e-4},
            {'params': self.decoder3.parameters()},
            {'params': self.decoder2.parameters()},
            {'params': self.decoder1.parameters()},
            {'params': self.conv.parameters()}
        ]

        optimizer_params = []
        for group in params_group:
            base_lr = lr_init * group.get('lr_multi', 1.0)
            optimizer_params.append({
                'params': group['params'],
                'lr': base_lr,
                'initial_lr': base_lr,
                'weight_decay': group.get('weight_decay', 0.0)
            })

        self.optimizer = torch.optim.AdamW(
            optimizer_params, 
            betas=(0.9, 0.99),
            eps=1e-7
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, 
                    start_factor=0.01, 
                    end_factor=1.0, 
                    total_iters=lr_delay
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=20000, 
                    eta_min=lr_final
                )
            ],
            milestones=[lr_delay]
        )

    def update_learning_rate(self, iteration):
        self.scheduler.step()

    def update_optimizer(self):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
    def compute_loss(self, pred_mask, proj_image, gen_image, is_real_img, distance_type='manhattan'):
        total_loss = 0.0
        real_mask_loss = 0.0
        reproj_error = 0.0
        if is_real_img:  
            # real_mask_loss = torch.abs(pred_mask - 1.0).mean()
            # total_loss += real_mask_loss * 1.0
            total_loss += 0.0
        else:  
            if proj_image is not None: 
                reproj_error = computeReprojectionError(
                    proj_image.cuda(), 
                    gen_image.cuda(),
                    distance_type,
                    pred_mask
                )
                total_loss += reproj_error
        
        # binary regularization
        binary_reg = 0.0
        binary_weight = 0.0
        binary_reg = torch.mean(pred_mask * (1 - pred_mask))  

        total_loss += binary_reg * binary_weight
        # print(f"real_image_loss: {real_mask_loss * 1.0}, reproj_error_loss: {reproj_error}, binary_reg_loss: {binary_reg * binary_weight}")
        
        return total_loss

def savePredictedConfidence(source_path: str,
                            model_path: str,
                            ckpt_path: str):
    """
    e.g.
    source_path = /home/administrator/workspace/data/regroup_92171_all/pack/scene_3/track_1/
    model_path = /home/administrator/workspace/GStudio/eval_output/cross_lane_split_train/
    ckpt_path = /home/administrator/workspace/GStudio/eval_output/cross_lane_split_train/CAMERA_FRONT_resize/0/0/point_cloud/iteration_30000
    """
    save_path = os.path.join(source_path, "pred_confidence")
    os.makedirs(save_path, exist_ok=True)
    diffusion_image_dir = os.path.join(model_path, "CAMERA_FRONT_resize", "diffusion_result", "x_0_y_0_z_0")
    
    diffusion_image_name = sorted(os.listdir(diffusion_image_dir))
    diffusion_image_list = [os.path.join(diffusion_image_dir, i) for i in diffusion_image_name if "render" in i]
    
    conf_net = ConfidenceNetwork()
    conf_net.load_state_dict(ckpt_path)
    conf_net.to("cuda")
    conf_net.eval()
    
    for diffusion_image_path in tqdm(diffusion_image_list):
        diffusion_image = Image.open(diffusion_image_path)
        diffusion_image_torch = PILtoTorch(diffusion_image, (1910,1075))
        diffusion_image_torch_resize = PILtoTorch(diffusion_image, (1280,720))
        conf_mask = conf_net(diffusion_image_torch_resize.unsqueeze(0).cuda()).detach().cpu().squeeze(0)
        conf_mask_resize = F.interpolate(conf_mask.unsqueeze(0), size=(1075, 1910), mode='bilinear', align_corners=False).squeeze(0)
        mask_img = conf_mask_resize.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]
        cv2.imwrite(f"{save_path}/{diffusion_image_path.split('/')[-1]}", (mask_img * 255.0).astype(np.uint8))