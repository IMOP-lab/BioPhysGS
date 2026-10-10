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

import os
import json
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, anisotropic_total_variation_loss, depth_aware_smooth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip "
            f"install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    with open(os.path.join(dataset.model_path, "experiment_config.json"), "w") as config_file:
        json.dump({"optimization": vars(opt), "pipeline": vars(pipe)}, config_file, indent=2)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    train_num = len(scene.getTrainCameras())

    gaussians.embeddings = torch.nn.Sequential(
        torch.nn.Linear(108, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, opt.embedding_dims),
    ).cuda()
    gaussians.set_physical_parameter_initial_values(
        light_distance_factor=opt.light_distance_factor_init,
        tissue_scattering_length=opt.tissue_scattering_length_init,
        absorption_coefficient=opt.absorption_coefficient_init,
    )
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    gaussians.train_cam_idx = {uid: camera.colmap_id for uid, camera in enumerate(viewpoint_stack)}
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()



        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        torch.autograd.set_detect_anomaly(True)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]



        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # Clamp images to prevent extreme values that could cause NaN in SSIM
        image = torch.clamp(image, 0.0, 1.0)
        gt_image = torch.clamp(gt_image, 0.0, 1.0)

        # Check for NaN in rendered image before loss calculation
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"Warning: NaN detected in rendered image at iteration {iteration}")
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)


        # Depth regularization. For E2/E3 comparability the depth scheduler is
        # still only an enable check; its numeric value is not a loss weight.
        Ll1depth_pure = 0.0
        Ll1depth = 0
        depth_reliable = viewpoint_cam.depth_reliable
        invDepth = render_pkg["depth"]
        depth_mask = viewpoint_cam.depth_mask.cuda() if depth_reliable else None

        if (not opt.disable_depth_alignment and depth_l1_weight(iteration) > 0
                and depth_reliable and iteration > opt.depth_from_iter):
            gt_depth_m = viewpoint_cam.invdepthmap.cuda()
            rendered_depth = 1.0 / (invDepth + 1e-6)
            mask = (gt_depth_m > 0.005) & (rendered_depth > 0) & (depth_mask > 0)
            if not torch.any(mask):
                Ll1depth_pure = torch.tensor(0.0, device=rendered_depth.device)
            else:
                rendered_masked = torch.masked_select(rendered_depth, mask)
                gt_masked = torch.masked_select(gt_depth_m, mask)
                scale_factor = (torch.median(gt_masked) / torch.median(rendered_masked)).detach()
                rendered_depth_aligned = rendered_depth * scale_factor
                Ll1depth_pure = torch.nn.functional.l1_loss(
                    rendered_depth_aligned[mask], gt_depth_m[mask]
                )
            depth_factor = opt.depth_loss_multiplier
            if opt.apply_depth_scheduler:
                depth_factor *= depth_l1_weight(iteration)
            loss += depth_factor * Ll1depth_pure
            Ll1depth = Ll1depth_pure.item()

        if (not opt.disable_smooth and depth_reliable
                and iteration > opt.smooth_from_iter):
            progress = min(max(iteration / opt.iterations, 0.0), 1.0)
            if opt.smooth_weight_init > 0 and opt.smooth_weight_final > 0:
                current_smooth_weight = opt.smooth_weight_init * (
                    opt.smooth_weight_final / opt.smooth_weight_init
                ) ** progress
            else:
                current_smooth_weight = (
                    opt.smooth_weight_init
                    + (opt.smooth_weight_final - opt.smooth_weight_init) * progress
                )
            loss += current_smooth_weight * depth_aware_smooth_loss(
                invDepth, render_pkg["render"], depth_mask, gamma=0.001
            )

        if (not opt.disable_tv and opt.lambda_tv and iteration > opt.tv_from_iter
                and iteration < opt.tv_until_iter and depth_reliable):
            loss += opt.lambda_tv * anisotropic_total_variation_loss(invDepth.squeeze())

        # scale loss



        if (not opt.disable_scale and iteration > opt.scale_from_iter
                and visibility_filter.sum() > 0):
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
                        # 建议修改
            threshold = 0.0001
            min_scale_loss = torch.clamp(threshold - sorted_scale[..., 0], min=0)
            loss += opt.scale_loss_weight * min_scale_loss.mean()


        # Backprop

        torch.autograd.set_detect_anomaly(True)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Densification
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # ``disable_pruning`` must not disable clone/split
                    # densification.  The previous guard skipped the entire
                    # routine and left the model at its sparse initialization
                    # size for all 30k iterations.
                    disable_pruning = getattr(opt, 'disable_pruning', False)
                    size_threshold = None if disable_pruning else (20 if iteration > opt.opacity_reset_interval else None)
                    min_opacity = 0.0 if disable_pruning else 0.005
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_grad_abs_threshold,
                        min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                    )

                if iteration % opt.opacity_reduce_interval == 0 and opt.use_reduce and not getattr(opt, 'disable_reduce', False):
                    gaussians.reduce_opacity()

                if (not getattr(opt, 'disable_opacity_reset', False)) and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if network_gui.conn == None:
            network_gui.try_connect(dataset.render_items)
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                if custom_cam != None:
                    render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                    net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                metrics_dict = {
                    "iteration": iteration,
                    "#": gaussians.get_opacity.shape[0],
                    "loss": ema_loss_for_log,
                    "depth_loss": Ll1depth,
                    # Add more metrics as needed
                }
                # Send the data
                network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                # raise e
                network_gui.conn = None

    physical_parameters = {
        "light_distance_factor": gaussians.bs_model.light_distance_factor.detach().cpu().tolist(),
        "tissue_scattering_length": gaussians.da_model.tissue_scattering_length.detach().cpu().tolist(),
        "absorption_coefficient": gaussians.da_model.absorption_coefficient.detach().cpu().tolist(),
    }
    with open(os.path.join(dataset.model_path, "physical_parameters_final.json"), "w") as parameter_file:
        json.dump(physical_parameters, parameter_file, indent=2)


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


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 10000,20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 7000, 10000,20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--warn_until_iter", type=int, default=3_000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
