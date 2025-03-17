# Copyright (C) 2024, Ruijie Zhu
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui, render_w_pose
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import DataLoader

from core_flow.things_eval import get_cfg as get_things_cfg
from core_flow.FlowFormer import build_flowformer
# from core_flow.GMFlow import GMFlow
from gmflow.config import get_cfg as get_gmflow_cfg
from gmflow.gmflow import build_gmflow
from MDFlow.models.FastFlowNet import FastFlowNet
from core_depth.MiDaS import MidasNet
import torch.nn.functional as F

from core_flow.utils_former.flow_viz import flow_to_image
from utils.vis_utils import colorize
import cv2

from scene.scale_model import SCALE, optimize_depth
from utils.warp_utils import calculate_camera_flow, warping_image, warping_gs_flow, flow_warping
from utils.flow_utils import calculate_gs_flow
from utils.loss_utils import scale_and_shift_depth_loss, flow_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import wandb
wandb.init(project="Motion-GS")


# loss for pose optimization
def get_loss_tracking_rgb(image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = 0.01
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()

def get_loss_tracking_l1(image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1 = torch.abs(image  - gt_image)
    return l1.mean()


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof) 
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if dataset.use_depth_and_flow:
        ##### FlowFormer
        # cfg = get_things_cfg()
        # flownet = torch.nn.DataParallel(build_flowformer(cfg))
        # flownet.load_state_dict(torch.load(cfg.model))
        # flownet = flownet.cuda()
        # flownet.eval()

        ##### GMFlow
        cfg = get_gmflow_cfg()
        flownet = torch.nn.DataParallel(build_gmflow(cfg)) 
        flownet = flownet.module
        checkpoint = torch.load(cfg.model, map_location = 'cpu')
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        flownet.load_state_dict(weights)
        flownet = flownet.cuda()
        flownet.eval()

        ##### MDFlow
        # flownet = FastFlowNet().cuda().eval()
        # flownet.load_state_dict(torch.load('MDFlow/checkpoints/fastflownet_sintel.pth'))

        ##### Unimatch
        # cfg = get_unimatch_cfg()
        # flownet = torch.nn.DataParallel(build_unimatch(cfg)) 
        # flownet = flownet.module
        # checkpoint = torch.load(cfg.model, map_location = 'cpu')
        # weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # flownet.load_state_dict(weights)
        # flownet = flownet.cuda()
        # flownet.eval()

        # if not dataset.load_depth:
        #     ##### Midas
            # midas_pretrain_path = 'pretrained_weights/midas_cpkt.pt'
            # depthnet = MidasNet(midas_pretrain_path, non_negative=True, resize=[256, 512], normalize_input=True)
            # depthnet = depthnet.cuda()
            # depthnet = depthnet.eval()

        flow_2d_gt_list = []
        # depth_gt_list = []
        
        # if dataset.optimize_depth:
        #     scale_shift = SCALE(len(scene.getTrainCameras())).cuda()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    if dataset.dataloader:
        training_dataset = scene.getTrainCameraDataset()   
    
    next_gt_image = None

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        elif len(viewpoint_stack) <= 1:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam1 = viewpoint_stack.pop(0)
        viewpoint_cam2 = viewpoint_stack[0]

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam1.load2device()
            viewpoint_cam2.load2device()

        fid1 = viewpoint_cam1.fid
        fid2 = viewpoint_cam2.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            d_xyz_2, d_rotation_2, d_scaling_2 = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input_1 = fid1.unsqueeze(0).expand(N, -1)
            time_input_2 = fid2.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input_1 + ast_noise)
            d_xyz_2, d_rotation_2, d_scaling_2 = deform.step(gaussians.get_xyz.detach(), time_input_2 + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam1, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        depth = render_pkg_re["depth"].detach()
        # alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = render_pkg_re[
        #     "alpha"], render_pkg_re["proj_2D"], render_pkg_re["conic_2D"], render_pkg_re["conic_2D_inv"
        #     ], render_pkg_re["gs_per_pixel"], render_pkg_re["weight_per_gs_pixel"], render_pkg_re["x_mu"]
        
        render_pkg_re_2_1 = render(viewpoint_cam2, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = render_pkg_re_2_1[
            "alpha"], render_pkg_re_2_1["proj_2D"], render_pkg_re_2_1["conic_2D"], render_pkg_re_2_1["conic_2D_inv"
            ], render_pkg_re_2_1["gs_per_pixel"], render_pkg_re_2_1["weight_per_gs_pixel"], render_pkg_re_2_1["x_mu"]         

        render_pkg_re_2 = render(viewpoint_cam2, gaussians, pipe, background, d_xyz_2, d_rotation_2, d_scaling_2, dataset.is_6dof)
        next_proj_2D, next_conic_2D = render_pkg_re_2["proj_2D"], render_pkg_re_2["conic_2D"] 

        # warp gs_flow to match motion flow
        gs_flow = calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu)
        gs_flow = warping_gs_flow(depth, gs_flow, viewpoint_cam1, viewpoint_cam2)

        # Loss
        if not dataset.dataloader:
            gt_image = viewpoint_cam1.original_image.cuda() if next_gt_image is None else next_gt_image            
            next_gt_image = viewpoint_cam2.original_image.cuda()
        else:
            gt_image = training_dataset.load_image(viewpoint_cam1.original_image).cuda() if next_gt_image is None else next_gt_image
            next_gt_image = training_dataset.load_image(viewpoint_cam2.original_image).cuda()

        vis = True if iteration % 100 == 0 else False 
        if vis:
            wandb.log({"tmp_image1": [wandb.Image(gt_image.permute(1,2,0).cpu().numpy()*255, caption="Iteration: {}".format(iteration))]})
            wandb.log({"tmp_image2": [wandb.Image(next_gt_image.permute(1,2,0).cpu().numpy()*255, caption="Iteration: {}".format(iteration))]})
            wandb.log({"tmp_image_render": [wandb.Image(image.permute(1,2,0).detach().cpu().numpy()*255, caption="Iteration: {}".format(iteration))]})
            gs_flow_img = flow_to_image(gs_flow.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
            wandb.log({"tmp_gs_flow": [wandb.Image(gs_flow_img, caption="Iteration: {}".format(iteration))]})

        H, W = gt_image.shape[-2:]

        if dataset.load_depth:
            if dataset.dataloader:
                depth_gt = training_dataset.load_depth(viewpoint_cam1.depth).cuda() # TODO
                motion_mask = None # TODO
            else:
                depth_gt = viewpoint_cam1.depth.cuda()
                motion_mask = viewpoint_cam1.motion_mask.cuda() if viewpoint_cam1.motion_mask is not None else None
        else:
            motion_mask = viewpoint_cam1.motion_mask.cuda() if viewpoint_cam1.motion_mask is not None else None           

        if dataset.use_depth_and_flow:
            if iteration < len(scene.getTrainCameras()):
                with torch.no_grad():
                    # flow_forward_gt, former_backward_gt = process_optical_flow(flownet, next_gt_image, gt_image, H, W, H, W)
                    flow_2d_gt = flownet(gt_image[None]*255, next_gt_image[None]*255) # return flow_predictions, feat_s, feat_t
                    # orig_size = (int(H), int(W))
                    # div_size = 64
                    # if H % div_size != 0 or W % div_size != 0:
                    #     input_size = (
                    #         (div_size * int(H / div_size)), 
                    #         (div_size * int(W / div_size))
                    #     )
                    #     img1 = F.interpolate(gt_image[None], size=input_size, mode='bilinear', align_corners=False)
                    #     img2 = F.interpolate(next_gt_image[None], size=input_size, mode='bilinear', align_corners=False)
                    # else:
                    #     input_size = orig_size
                    
                    # output_dict = flownet(torch.cat([img1, img2], 1).cuda())
                    # flow_2d_gt = output_dict.data
                    H_flow, W_flow = flow_2d_gt[0].shape[-2:]
                    if W_flow == W and H_flow == H:
                        flow_2d_gt = flow_2d_gt[0].squeeze() # 2 H W
                    else: 
                        flow_2d_gt = torch.nn.functional.interpolate(flow_2d_gt[0], size=(H, W), mode="bilinear").squeeze()
                        flow_2d_gt[0] *= W / W_flow
                        flow_2d_gt[1] *= H / H_flow

                    # if not dataset.load_depth:
                    #     depth_gt = depthnet(gt_image[None])[0]

                # if dataset.optimize_depth:
                #     depth_gt = optimize_depth(scale_shift, viewpoint_cam1.uid, depth_gt, gt_image, next_gt_image, viewpoint_cam1, viewpoint_cam2, motion_mask, opt, vis)
                # if iteration == 1:
                #     viewpoint_cam1.reset_extrinsic() # initialize first camera pose
                # viewpoint_cam2.reset_extrinsic(estimate_camera_pose(viewpoint_cam1, viewpoint_cam2, depth_gt, gt_image, next_gt_image)) 
                
                flow_2d_gt_list.append(flow_2d_gt)
                # depth_gt_list.append(depth_gt)
            else:
                flow_2d_gt = flow_2d_gt_list[viewpoint_cam1.uid]
                # depth_gt = depth_gt_list[viewpoint_cam1.uid]

            with torch.no_grad():
                camera_flow = calculate_camera_flow(depth, viewpoint_cam1, viewpoint_cam2)
                motion_flow = flow_2d_gt - camera_flow
                motion_flow = motion_flow * (1 - motion_mask) if motion_mask is not None else motion_flow
        
            if vis:
                flow_img = flow_to_image(flow_2d_gt.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
                depth_img = colorize(depth.squeeze().detach().cpu().numpy())
                camera_flow_img = flow_to_image(camera_flow.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
                motion_flow_img = flow_to_image(motion_flow.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
            
                wandb.log({"tmp_flow_gt": [wandb.Image(flow_img, caption="Iteration: {}".format(iteration))]})
                wandb.log({"tmp_depth_gt": [wandb.Image(depth_img, caption="Iteration: {}".format(iteration))]})
                wandb.log({"tmp_camera_flow": [wandb.Image(camera_flow_img, caption="Iteration: {}".format(iteration))]})
                wandb.log({"tmp_motion_flow": [wandb.Image(motion_flow_img, caption="Iteration: {}".format(iteration))]})
                
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if dataset.use_depth_and_flow and iteration >= opt.warm_up:
            # Lflow = flow_loss(gs_flow, motion_flow, H, W)
            Lflow = flow_loss(gs_flow, motion_flow.detach(), H, W)
            loss += opt.flow_loss_weight * Lflow
        # elif iteration >= opt.warm_up:
        #     pred_next_image = flow_warping(gs_flow, gt_image)
        #     Lflow = (1.0 - opt.lambda_dssim) * l1_loss(pred_next_image, next_gt_image) + opt.lambda_dssim * (1.0 - ssim(pred_next_image, next_gt_image))
        else:
            Lflow = None
        # pred_image = warping_image(depth_gt, next_gt_image, viewpoint_cam1, viewpoint_cam2)
        # if vis:
        #     cv2.imwrite("tmp_image1_warp.png", pred_image.permute(1,2,0).detach().cpu().numpy()*255)
        # Ldepth = (1.0 - opt.lambda_dssim) * l1_loss(pred_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image))
        # loss += opt.depth_loss_weight * Ldepth

        # if opt.use_depth_loss:
        #     Ldepth = scale_and_shift_depth_loss(depth, depth_gt.detach())
        #     loss += opt.depth_loss_weight * Ldepth

        #     if vis:
        #         depth_img = colorize(depth.squeeze().detach().cpu().numpy())
        #         depth_gt_img = colorize(depth_gt.squeeze().detach().cpu().numpy())
        #         wandb.log({"tmp_depth_pred.png": [wandb.Image(depth_img, caption="Iteration: {}".format(iteration))]})
        #         wandb.log({"tmp_depth_gt.png": [wandb.Image(depth_gt_img, caption="Iteration: {}".format(iteration))]})
        # else:
        #     Ldepth = None

        log_loss = {"Limage": Ll1.item(), 
                    "Ltotal": loss.item(), 
                    "Lflow": Lflow.item() if Lflow is not None else 0.0, 
                    # "Ldepth": Ldepth.item() if Ldepth is not None else 0.0
                    }
        wandb.log(log_loss)
        
        loss.backward()
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam1.load2device('cpu')
            viewpoint_cam2.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, log_loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, dataset.dataloader)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                # scale_shift.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
                # scale
                # scale_shift.optimizer.step()
                # scale_shift.optimizer.zero_grad()
        
        #  Alternate pose optimization
        if dataset.optimize_pose and iteration >= opt.warm_up:
            opt_params = []
            opt_params.append(
                {
                    "params": [viewpoint_cam1.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam1.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint_cam1.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam1.uid),
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)
            # use updated deformation field (and gaussians)
            if iteration < opt.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                N = gaussians.get_xyz.shape[0]
                time_input_1 = fid1.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input_1 + ast_noise)
            
            render_pkg_re = render_w_pose(viewpoint_cam1, gaussians, pipe, background, d_xyz.detach(), d_rotation.detach(), d_scaling.detach(), False)
            image, depth, opacity = (
                render_pkg_re["render"],
                render_pkg_re["depth"],
                render_pkg_re["opacity"],
            )

            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking_l1(
                image, depth, opacity, viewpoint_cam1
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = viewpoint_cam1.update_pose()

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, log_loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, dataloader=False):
    if tb_writer:
        for key, value in log_loss.items():
            tb_writer.add_scalar('train_loss_patches/' + key, value, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        if dataloader:
            test_dataset = scene.getTestCameraDataset()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    if dataloader:
                        gt_image = torch.clamp(test_dataset.load_image(viewpoint.original_image).to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 20001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
