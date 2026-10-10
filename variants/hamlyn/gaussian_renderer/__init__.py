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
from torch.nn import functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.endoscope_model import EndoscopeRenderingModel
from utils.sh_utils import eval_sh
from utils.white_balance import simple_color_balance


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, separate_sh=False,
           override_color=None, use_trained_exp=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True,
                                     device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    features_dc = pc.get_features_dc
    features_rest = pc.get_features_rest


    use_avg_appearance = True

    R = viewpoint_camera.world_view_transform[:3, :3].flatten()  # 旋转矩阵 (3, 3)
    T = viewpoint_camera.world_view_transform[3, :3]

    cam_pose = torch.cat((R, T), dim=-1).cuda()
    pose_features = pc.position_encoding(cam_pose)

    pose_emd = pc.embeddings(pose_features)


    appearance_embed = pose_emd
    appearance_embeddings = pose_emd.repeat(means3D.shape[0], 1)

    # 基于物理原理的内窥镜机理增强外观处理
    dir_pp_normalized = F.normalize(pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1),
                                    dim=1)

    # 计算距离和视角相关参数
    distances = torch.norm(pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1), dim=1, keepdim=True)

    # === 1. 基于Beer-Lambert定律的光传输建模 ===
    embeddings = pc._get_fourier_features(pc.get_xyz, num_features=4)

    # 内窥镜点光源的距离平方反比衰减（物理准确）
    inverse_square_attenuation = 1.0 / (1.0 + distances.pow(2) / pc.bs_model.light_distance_factor.pow(2))

    # 基于Beer-Lambert定律的介质衰减
    # I = I0 * exp(-μ * d)，其中μ是衰减系数
    # 对RGB三通道的吸收系数取平均，得到标量衰减系数
    avg_absorption_coeff = pc.da_model.absorption_coefficient.mean()
    medium_attenuation = torch.exp(-avg_absorption_coeff * distances)

    # 组合光传输效应
    total_attenuation = inverse_square_attenuation * medium_attenuation

    # 物理噪声建模：基于泊松噪声模型
    signal_dependent_noise = torch.sqrt(torch.clamp(total_attenuation, min=1e-6)) * 0.001
    # 确保噪声张量与embeddings维度匹配
    signal_dependent_noise = signal_dependent_noise.expand_as(embeddings)
    embeddings.add_(torch.randn_like(embeddings) * signal_dependent_noise)

    # === 2. 基于Fresnel方程的视角相关反射建模 ===
    position = pc.position_encoding(dir_pp_normalized)

    # 计算视角与表面法线的夹角（简化为基于距离的近似）
    # 在内窥镜中，近距离区域通常视角更垂直，远距离区域视角更倾斜
    viewing_angle = torch.atan2(distances, pc.da_model.tissue_scattering_length)

    # 基于Fresnel方程的反射率计算（简化版）
    # 组织的折射率约为1.4，空气为1.0
    n1, n2 = 1.0, 1.4  # 空气到组织
    cos_theta = torch.cos(viewing_angle)
    sin_theta = torch.sin(viewing_angle)

    # Fresnel反射率（s偏振近似）
    fresnel_reflectance = ((n1 * cos_theta - n2 * torch.sqrt(1 - (n1/n2 * sin_theta).pow(2))) /
                          (n1 * cos_theta + n2 * torch.sqrt(1 - (n1/n2 * sin_theta).pow(2)))).pow(2)
    fresnel_reflectance = torch.clamp(fresnel_reflectance, 0.0, 1.0)

    # 视角相关的位置编码调制
    position = position * (1.0 + fresnel_reflectance * 0.3)
    embed = torch.cat((appearance_embeddings, position), dim=-1)

    app_features = pc.appearance_mlp(features_dc.squeeze(1), embed, embeddings).clamp_max(1.0)

    # === 3. 基于Mie散射理论的血红蛋白光谱吸收建模 ===
    # 血红蛋白在不同波长的吸收系数（基于实际光谱数据）
    # 红光(~650nm): 高吸收, 绿光(~550nm): 中等吸收, 蓝光(~450nm): 低吸收
    wavelength_absorption = pc.appearance_mlp.hemoglobin_absorption  # [3] 对应RGB

    # 基于Beer-Lambert定律的光谱吸收
    # I(λ) = I0(λ) * exp(-ε(λ) * c * l)
    # 其中ε是摩尔吸光系数，c是浓度，l是光程长度
    spectral_attenuation = torch.exp(-wavelength_absorption.unsqueeze(0) * distances)

    # 考虑散射效应：血红蛋白颗粒的Mie散射
    # 散射强度与波长的-4次方成反比（瑞利散射近似）
    wavelength_factors = torch.tensor([1.0, 0.8, 0.6], device=app_features.device).view(1, 3)  # RGB相对散射
    scattering_enhancement = 1.0 + (1.0 - spectral_attenuation) * wavelength_factors * 0.2

    # 应用光谱和散射效应
    app_features = app_features * spectral_attenuation * scattering_enhancement
    # app_features = pc.appearance_mlp(features_dc.squeeze(1), appearance_embeddings).clamp_max(1.0)

    # === 4. 基于双向反射分布函数(BRDF)的组织反照率建模 ===
    colors_toned = torch.cat((app_features[:, None, :], features_rest), dim=1)

    # 组织的BRDF建模：Lambert漫反射 + 镜面反射分量
    tissue_albedo = pc.appearance_mlp.tissue_albedo_factor

    # Lambert漫反射分量（与视角无关）
    diffuse_component = colors_toned * tissue_albedo

    # 镜面反射分量（与视角相关）
    # 使用之前计算的Fresnel反射率，确保数值稳定性
    fresnel_reflectance_safe = torch.clamp(fresnel_reflectance, 0.0, 1.0)

    # 检查并处理NaN/Inf值
    if torch.isnan(fresnel_reflectance_safe).any() or torch.isinf(fresnel_reflectance_safe).any():
        fresnel_reflectance_safe = torch.nan_to_num(fresnel_reflectance_safe, nan=0.0, posinf=1.0, neginf=0.0)

    # 安全的张量广播操作
    fresnel_expanded = fresnel_reflectance_safe.view(-1, 1, 1).expand_as(colors_toned)
    specular_component = colors_toned * fresnel_expanded * 0.1

    # 组合BRDF模型
    colors_toned = diffuse_component + specular_component

    # 内窥镜光源的色温校正（约6500K）
    # 模拟LED光源的光谱特性
    color_temperature_correction = torch.tensor([1.0, 0.95, 0.9], device=colors_toned.device)
    colors_toned = colors_toned * color_temperature_correction.view(1, 1, 3)

    # shdim = (self.config.sh_degree + 1) ** 2
    shdim = (3 + 1) ** 2
    colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
    colors_toned = eval_sh(3, colors_toned, dir_pp_normalized)
    colors_toned = torch.clamp_min(colors_toned + 0.5, 0.0)












    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
                # shs = colors_toned
    else:
        colors_precomp = override_color

    # 注释掉强制使用colors_toned的部分，恢复原版3DGS逻辑
    shs = None
    colors_precomp = colors_toned

    # Render the scene
    if separate_sh:
        rendered_J, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_J, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_J = torch.matmul(rendered_J.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.


    depth_z = 1 / (depth_image + 1e-5)

    # 使用内窥镜渲染模型替代水下渲染
    rendered_I, specular_component, sss_attenuation = EndoscopeRenderingModel.compute_endoscope_rendering(
        rendered_J, depth_z, appearance_embed, pc.bs_model, pc.da_model
    )

    rendered_J = simple_color_balance(rendered_J, scale=0.01, alpha=0.1)

    rendered_J = rendered_J.clamp(0, 1)
    rendered_I = rendered_I.clamp(0, 1)

    out = {
        "render": rendered_I,
        "restore_scene": rendered_J,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "attenuation": sss_attenuation,  # 现在是次表面散射衰减
        "backscatter": specular_component,  # 现在是高光反射分量

    }

    return out
