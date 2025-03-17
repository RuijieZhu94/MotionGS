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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return torch.abs((network_output - gt).mean(0, keepdim=True) * mask).sum() / mask.sum()
    else:
        return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is not None:
            return (ssim_map.mean(0, keepdim=True) * mask).sum() / mask.sum()
        else:    
            return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def scale_and_shift_depth_loss(depth_pred, depth_gt):
    norm_depth_pred = (depth_pred - torch.min(depth_pred)) / (torch.max(depth_pred) - torch.min(depth_pred) + 1e-7)
    norm_depth_gt = (depth_gt - torch.min(depth_gt)) / (torch.max(depth_gt) - torch.min(depth_gt) + 1e-7)
    return l1_loss(norm_depth_pred, norm_depth_gt)


def flow_loss(flow_pred, flow_gt, height, width):
    flow_pred[0] /= height
    flow_pred[1] /= width
    flow_pred = flow_pred.clamp(-1, 1)
    flow_gt[0] /= height
    flow_gt[1] /= width
    flow_gt = flow_gt.clamp(-1, 1)
    return l1_loss(flow_pred, flow_gt)
