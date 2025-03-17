import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
# from utils.warp_utils import warping_image
from utils.loss_utils import l1_loss, ssim
import cv2

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # Points: B 4 HW 
        # K：B 4 4
        # T: B 4 4
        P = torch.matmul(K, T)[:, :3, :]  # B 3 4
        cam_points = torch.matmul(P, points)  # B 4 HW 
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)  # B 2 HW
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width) # B 2 H W
        pix_coords = pix_coords.permute(0, 2, 3, 1) # B H W 2
        # normalize
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords
    

def reproject(depth, cam, coords_3d, ones):
    inv_cam = torch.pinverse(cam)
    cam_points = depth * coords_3d
    cam_points = torch.cat([cam_points, ones], 0)
    cam_points = torch.einsum("ab,bhw->ahw", [inv_cam, cam_points])
    return cam_points

def project(points, cam): 
    cam_points = torch.einsum("ab,bhw->ahw", [cam, points])
    # cam_points = torch.matmul(cam, points)
    pix_coords = cam_points[:2] / (cam_points[2:3] + 1e-7)
    return pix_coords

def calculate_camera_flow(depth1, cam1, cam2):
    H, W = depth1.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(cam1.intrinsic.cuda())[None] # B 4 4
    K2 = cam2.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(cam2.extrinsic.cuda()), 
                     cam1.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth1, inv_K1) # B 4 HW
    _, pixel_coords = project3d(points_3d, K2, T12) # B H W 2
    pixel_coords = pixel_coords.permute(0, 3, 1, 2) # B 2 H W
    ori_coords = backprojdepth.pix_coords.view(1, 3, H, W)[:, :2] # B 2 H W
    camere_flow = pixel_coords - ori_coords # B 2 H W
    return camere_flow[0] # 2 H W


def warping_image(depth_gt, next_gt_image, camera_pose, next_camera_pose):
    H, W = depth_gt.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(camera_pose.intrinsic.cuda())[None] # B 4 4
    K2 = next_camera_pose.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(next_camera_pose.extrinsic.cuda()), 
                     camera_pose.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth_gt, inv_K1) # B 4 HW
    pixel_coords_norm, _ = project3d(points_3d, K2, T12) # B H W 2
    pred_image = F.grid_sample(next_gt_image.unsqueeze(0), pixel_coords_norm, padding_mode="border", align_corners=True) # B 3 H W
    return pred_image.squeeze(0)

def warping_gs_flow(depth_gt, gs_flow, camera_pose, next_camera_pose):
    H, W = depth_gt.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(camera_pose.intrinsic.cuda())[None] # B 4 4
    K2 = next_camera_pose.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(next_camera_pose.extrinsic.cuda()), 
                     camera_pose.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth_gt, inv_K1) # B 4 HW
    pixel_coords_norm, _ = project3d(points_3d, K2, T12) # B H W 2
    gs_flow= F.grid_sample(gs_flow.unsqueeze(0), pixel_coords_norm, padding_mode="border", align_corners=True) # B 3 H W
    return gs_flow.squeeze(0)


def warping_loss(depth_gt, gt_image, next_gt_image, camera_pose, next_camera_pose, mask, opt, vis=False):
    pred_image = warping_image(depth_gt, next_gt_image, camera_pose, next_camera_pose)
    if vis:
        cv2.imwrite("tmp_image1_warp.png", pred_image.permute(1,2,0).detach().cpu().numpy()[..., ::-1]*255)
    if mask is not None:
        Ldepth = (1.0 - opt.lambda_dssim) * l1_loss(pred_image, gt_image.detach(), mask=mask) \
                + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image.detach(), mask=mask))
    else:
        Ldepth = (1.0 - opt.lambda_dssim) * l1_loss(pred_image, gt_image.detach()) \
                + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image.detach()))
    return Ldepth

def flow_warping(gs_flow, gt_image):
    height, width = gt_image.shape[-2:]
    flow_x, flow_y = gs_flow[0,...], gs_flow[1,...]
    coord_x, coord_y = torch.meshgrid(torch.arange(height), torch.arange(width))

    pos_x = coord_x.reshape(height, width).type(torch.float32).cuda() + flow_x
    pos_y = coord_y.reshape(height, width).type(torch.float32).cuda() + flow_y
    pos_x = (pos_x - (height-1)/2) / ((height-1)/2)
    pos_y = (pos_y - (width-1)/2) / ((width-1)/2)

    pos = torch.stack((pos_y, pos_x), 2).type(torch.float32)
    pred_image = F.grid_sample(gt_image.unsqueeze(0), pos.unsqueeze(0), padding_mode="border", align_corners=True) # B 3 H W
    return pred_image.squeeze(0)