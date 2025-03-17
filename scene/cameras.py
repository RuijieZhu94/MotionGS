#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getIntrinsicMatrix, getExtrinsicMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, 
                 depth=None, motion_mask=None, meta_only=False, resolution=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.meta_only = meta_only
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)

        if not self.meta_only:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)    
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1] 
            self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None
            self.motion_mask = torch.Tensor(motion_mask).to(self.data_device) if motion_mask is not None else None
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.original_image = image # path
            self.image_width = self.resolution[0]
            self.image_height = self.resolution[1]
            self.depth = None # TODO:add depth path
            self.motion_mask = None

        # pose finetune(from Mono-GS)
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True).cuda()
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True).cuda()
        )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(
            self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = getIntrinsicMatrix(width=self.image_width, height=self.image_height, fovX=self.FoVx,
                                                     fovY=self.FoVy).to(self.data_device)
        # COLMAP axis (Y down, Z forward)
        self.extrinsic = torch.tensor(getExtrinsicMatrix(R, T)).to(self.data_device)

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.intrinsic = getIntrinsicMatrix(width=self.image_width, height=self.image_height, fovX=self.FoVx,
                                                     fovY=self.FoVy).to(self.data_device)
        self.extrinsic = torch.tensor(getExtrinsicMatrix(R, T)).to(self.data_device)
    
    def update_pose(self, converged_threshold=1e-4):
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)

        T_w2c = torch.eye(4, device=tau.device)
        T_w2c[0:3, 0:3] = torch.tensor(self.R.T).cuda()  # R_c2w -> R_w2c
        T_w2c[0:3, 3] = torch.tensor(self.T).cuda()

        new_w2c = SE3_exp(tau) @ T_w2c

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        converged = tau.norm() < converged_threshold
        self.reset_extrinsic(new_R.cpu().numpy().T, new_T.cpu().numpy())    # R_c2w, T_w2c

        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)
        return converged 

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

#############################################
# Functions of SE3 operations(from Mono-GS) #
#############################################
def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm