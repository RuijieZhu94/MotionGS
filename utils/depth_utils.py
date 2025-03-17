import numpy as np


def compute_scale_and_shift(depth, depth_gt):
    valid = depth_gt > 0
    depth = depth[valid]
    depth_gt = depth_gt[valid]
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    depth_gt_min = np.min(depth_gt)
    depth_gt_max = np.max(depth_gt)
    scale = (depth_gt_max - depth_gt_min).mean() / (depth_max - depth_min).mean()
    shift = depth_gt.mean() - (depth * scale).mean()
    return scale, shift