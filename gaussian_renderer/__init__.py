
import math

import torch
from torch.nn import functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.endoscope_model import EndoscopeRenderingModel
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.white_balance import simple_color_balance


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
):
    """Render the scene. Background tensor must already be on GPU."""

    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

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
        antialiasing=pipe.antialiasing,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    features_dc = pc.get_features_dc
    features_rest = pc.get_features_rest

    rotation = viewpoint_camera.world_view_transform[:3, :3].flatten()
    translation = viewpoint_camera.world_view_transform[3, :3]
    cam_pose = torch.cat((rotation, translation), dim=-1).cuda()
    pose_features = pc.position_encoding(cam_pose)
    appearance_embed = pc.embeddings(pose_features)
    appearance_embeddings = appearance_embed.repeat(means3D.shape[0], 1)

    dir_pp_normalized = F.normalize(
        pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1),
        dim=1,
    )
    distances = torch.norm(
        pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1),
        dim=1,
        keepdim=True,
    )

    # 1. Distance-aware attenuation and mild signal-dependent noise.
    embeddings = pc._get_fourier_features(pc.get_xyz, num_features=4)
    inverse_square_attenuation = 1.0 / (
        1.0 + distances.pow(2) / pc.bs_model.light_distance_factor.pow(2)
    )
    avg_absorption_coeff = pc.da_model.absorption_coefficient.mean()
    medium_attenuation = torch.exp(-avg_absorption_coeff * distances)
    total_attenuation = inverse_square_attenuation * medium_attenuation
    signal_dependent_noise = torch.sqrt(torch.clamp(total_attenuation, min=1e-6)) * 0.001
    embeddings.add_(torch.randn_like(embeddings) * signal_dependent_noise.expand_as(embeddings))

    # 2. View-dependent Fresnel-style modulation.
    position = pc.position_encoding(dir_pp_normalized)
    viewing_angle = torch.atan2(distances, pc.da_model.tissue_scattering_length)
    n1, n2 = 1.0, 1.4
    cos_theta = torch.cos(viewing_angle)
    sin_theta = torch.sin(viewing_angle)
    sqrt_term = torch.clamp(1 - (n1 / n2 * sin_theta).pow(2), min=1e-6)
    fresnel_reflectance = (
        (n1 * cos_theta - n2 * torch.sqrt(sqrt_term))
        / (n1 * cos_theta + n2 * torch.sqrt(sqrt_term))
    ).pow(2)
    fresnel_reflectance = torch.clamp(fresnel_reflectance, 0.0, 1.0)
    position = position * (1.0 + fresnel_reflectance * 0.3)

    app_features = pc.appearance_mlp(
        features_dc.squeeze(1), appearance_embeddings, embeddings
    ).clamp_max(1.0)

    # 3. Channel-dependent attenuation and scattering.
    wavelength_absorption = pc.appearance_mlp.hemoglobin_absorption
    spectral_attenuation = torch.exp(-wavelength_absorption.unsqueeze(0) * distances)
    wavelength_factors = torch.tensor([1.0, 0.8, 0.6], device=app_features.device).view(1, 3)
    scattering_enhancement = 1.0 + (1.0 - spectral_attenuation) * wavelength_factors * 0.2
    app_features = app_features * spectral_attenuation * scattering_enhancement

    # 4. Simple diffuse + specular mixture.
    colors_toned = torch.cat((app_features[:, None, :], features_rest), dim=1)
    diffuse_component = colors_toned * pc.appearance_mlp.tissue_albedo_factor

    fresnel_reflectance_safe = torch.clamp(fresnel_reflectance, 0.0, 1.0)
    if torch.isnan(fresnel_reflectance_safe).any() or torch.isinf(fresnel_reflectance_safe).any():
        fresnel_reflectance_safe = torch.nan_to_num(
            fresnel_reflectance_safe, nan=0.0, posinf=1.0, neginf=0.0
        )
    fresnel_expanded = fresnel_reflectance_safe.view(-1, 1, 1).expand_as(colors_toned)
    specular_component = colors_toned * fresnel_expanded * 0.1
    colors_toned = diffuse_component + specular_component

    color_temperature_correction = torch.tensor([1.0, 0.95, 0.9], device=colors_toned.device)
    colors_toned = colors_toned * color_temperature_correction.view(1, 1, 3)

    shdim = (3 + 1) ** 2
    colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
    colors_toned = eval_sh(3, colors_toned, dir_pp_normalized)
    colors_toned = torch.clamp_min(colors_toned + 0.5, 0.0)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    shs = None
    colors_precomp = colors_toned

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
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_J, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_J = torch.matmul(rendered_J.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    depth_z = 1 / (depth_image + 1e-5)
    rendered_I, specular_component, sss_attenuation = EndoscopeRenderingModel.compute_endoscope_rendering(
        rendered_J, depth_z, appearance_embed, pc.bs_model, pc.da_model
    )

    rendered_J = simple_color_balance(rendered_J, scale=0.01, alpha=0.1)
    rendered_J = rendered_J.clamp(0, 1)
    rendered_I = rendered_I.clamp(0, 1)

    return {
        "render": rendered_I,
        "restore_scene": rendered_J,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "attenuation": sss_attenuation,
        "backscatter": specular_component,
    }
