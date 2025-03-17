import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.warp_utils import warping_loss

class SCALE(nn.Module):
    def __init__(self, length = 1):
        super(SCALE, self).__init__()
        dtype = torch.FloatTensor
        alpha = 1. 
        beta = 0.
        self.scales = torch.nn.Parameter(torch.Tensor([alpha, beta]).type(dtype).repeat(length, 1), requires_grad=True)
        l = [
            {'params': [self.scales],
             "name": "scale"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=1e-2)

    def forward(self, depth, index):
        # depth (B, H, W)
        depth = self.scales[index, 0] * depth + self.scales[index, 1]
        return depth
    
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "scale/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'scale.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "scale"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "scale/iteration_{}/scale.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

def optimize_depth(model, index, depth, gt_image, next_gt_image, camera_pose, next_camera_pose, motion_mask, opt, vis):
    for iter in range(500):
        depth_gt = model(depth, index)
        loss = warping_loss(depth_gt, gt_image, next_gt_image, camera_pose, next_camera_pose, motion_mask, opt, vis)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
    return depth_gt.detach()

