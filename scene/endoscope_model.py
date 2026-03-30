import torch
from torch import nn
import tinycudann as tcnn


class EndoscopeEmbeddingModel(nn.Module):
    """Appearance model for endoscopic rendering."""

    def __init__(
        self,
        appearance_n_fourier_freqs: int = 4,
        sh_degree: int = 3,
        appearance_embedding_dim: int = 16,
        appearance_model_sh: bool = True,
        use_tcnn: bool = True,
    ):
        super().__init__()

        self.appearance_model_sh = appearance_model_sh
        self.use_tcnn = use_tcnn

        feat_in = 3
        if self.appearance_model_sh:
            feat_in = ((sh_degree + 1) ** 2) * 3
        mlp_input_dim = appearance_embedding_dim + feat_in + 6 * appearance_n_fourier_freqs

        if self.use_tcnn:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
            }
            self.mlp = tcnn.Network(
                n_input_dims=mlp_input_dim,
                n_output_dims=feat_in * 2,
                network_config=network_config,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, feat_in * 2),
            )

        self.tissue_albedo_factor = nn.Parameter(torch.ones(1) * 0.8)
        self.hemoglobin_absorption = nn.Parameter(torch.ones(3) * 0.1)

    def forward(self, color, image_embed, feature_embed):
        """Predict endoscopic appearance features from SH/color and embeddings."""
        c0 = 0.28209479177387814
        input_color = color
        if not self.appearance_model_sh:
            color = color[..., :3]
        inp = torch.cat((color, image_embed, feature_embed), dim=-1)

        output = self.mlp(inp) * 0.01
        offset, mul = torch.split(output, [color.shape[-1], color.shape[-1]], dim=-1)

        offset = torch.cat((offset / c0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])

        tissue_adjusted_color = input_color * self.tissue_albedo_factor
        hemoglobin_effect = torch.ones_like(input_color)
        if input_color.shape[-1] >= 3:
            hemoglobin_effect[..., :3] = 1.0 - self.hemoglobin_absorption.unsqueeze(0).expand_as(input_color[..., :3])
            hemoglobin_effect = torch.clamp(hemoglobin_effect, 0.1, 1.0)

        return tissue_adjusted_color * hemoglobin_effect * mul + offset


class SpecularReflectionNet(nn.Module):
    """Depth-conditioned specular reflection model."""

    def __init__(self):
        super().__init__()
        self.specular_conv = nn.Conv2d(1, 3, 1, bias=False)
        self.roughness_conv = nn.Conv2d(1, 3, 1, bias=False)

        nn.init.uniform_(self.specular_conv.weight, 0, 5.0)
        nn.init.uniform_(self.roughness_conv.weight, 0, 5.0)

        self.specular_intensity = nn.Parameter(torch.rand(3, 1, 1) * 0.8)
        self.surface_roughness = nn.Parameter(torch.rand(3, 1, 1) * 0.2)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.light_distance_factor = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, depth, appearance_embed=None):
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print("Warning: NaN/Inf detected in SpecularReflectionNet depth input")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        specular_coeff = self.softplus(self.specular_conv(depth))
        roughness_coeff = self.softplus(self.roughness_conv(depth))
        light_attenuation = torch.exp(-torch.clamp(depth / self.light_distance_factor, 0.0, 10.0))

        specular_reflection = (
            self.specular_intensity * light_attenuation * torch.exp(-torch.clamp(specular_coeff, 0.0, 10.0)) +
            self.surface_roughness * (1 - torch.exp(-torch.clamp(roughness_coeff, 0.0, 10.0)))
        )

        if torch.isnan(specular_reflection).any() or torch.isinf(specular_reflection).any():
            print("Warning: NaN/Inf detected in SpecularReflectionNet output")
            specular_reflection = torch.nan_to_num(specular_reflection, nan=0.0, posinf=1.0, neginf=0.0)

        return specular_reflection


class SubsurfaceScatterNet(nn.Module):
    """Depth-conditioned attenuation model for subsurface scattering."""

    def __init__(self):
        super().__init__()
        self.scattering_conv = nn.Conv2d(1 + 16, 6, 1, bias=False)
        nn.init.uniform_(self.scattering_conv.weight, 0, 5.0)

        self.scattering_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.tissue_scattering_length = nn.Parameter(torch.ones(1) * 1.5)
        self.absorption_coefficient = nn.Parameter(torch.ones(3) * 0.1)

    def forward(self, depth, appearance_embed=None):
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print("Warning: NaN/Inf detected in SubsurfaceScatterNet depth input")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        depth = depth.unsqueeze(0)
        appearance_embed = appearance_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        appearance_embed = appearance_embed.repeat(depth.shape[0], 1, depth.shape[2], depth.shape[3])
        depth_embed = torch.cat((depth, appearance_embed), dim=1)

        scatter_conv = torch.exp(-self.sigmoid(self.scattering_conv(depth_embed)))
        if torch.isnan(scatter_conv).any() or torch.isinf(scatter_conv).any():
            print("Warning: NaN/Inf detected in scatter_conv")
            scatter_conv = torch.nan_to_num(scatter_conv, nan=1.0, posinf=1.0, neginf=0.0)

        beta_s = torch.stack(
            tuple(
                torch.sum(scatter_conv[:, i:i + 2, :, :] * self.sigmoid(self.scattering_coef[i:i + 2]), dim=1)
                for i in range(0, 6, 2)
            ),
            dim=1,
        )

        if torch.isnan(beta_s).any() or torch.isinf(beta_s).any():
            print("Warning: NaN/Inf detected in beta_s")
            beta_s = torch.nan_to_num(beta_s, nan=0.0, posinf=1.0, neginf=0.0)

        effective_depth = depth / torch.clamp(self.tissue_scattering_length, min=0.1)
        sss_attenuation = torch.exp(-torch.clamp(beta_s, 0.0, 10.0) * effective_depth).squeeze(0)

        if torch.isnan(sss_attenuation).any() or torch.isinf(sss_attenuation).any():
            print("Warning: NaN/Inf detected in sss_attenuation")
            sss_attenuation = torch.nan_to_num(sss_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        absorption_effect = torch.exp(
            -torch.clamp(self.absorption_coefficient, min=0.0, max=1.0).unsqueeze(-1).unsqueeze(-1) * depth.squeeze(0)
        )
        if torch.isnan(absorption_effect).any() or torch.isinf(absorption_effect).any():
            print("Warning: NaN/Inf detected in absorption_effect")
            absorption_effect = torch.nan_to_num(absorption_effect, nan=1.0, posinf=1.0, neginf=0.0)

        final_attenuation = self.sigmoid(sss_attenuation * absorption_effect)
        if torch.isnan(final_attenuation).any() or torch.isinf(final_attenuation).any():
            print("Warning: NaN/Inf detected in final_attenuation")
            final_attenuation = torch.nan_to_num(final_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        return final_attenuation


class EndoscopeRenderingModel:
    """Combine the endoscopic rendering submodules into a single rendering equation."""

    @staticmethod
    def compute_endoscope_rendering(rendered_J, depth, appearance_embed, specular_net, sss_net):
        """Combine clean rendering, specular reflection and attenuation."""
        if torch.isnan(rendered_J).any() or torch.isinf(rendered_J).any():
            print("Warning: NaN/Inf detected in rendered_J")
            rendered_J = torch.nan_to_num(rendered_J, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print("Warning: NaN/Inf detected in depth")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        specular_component = specular_net(depth, appearance_embed)
        if torch.isnan(specular_component).any() or torch.isinf(specular_component).any():
            print("Warning: NaN/Inf detected in specular_component")
            specular_component = torch.nan_to_num(specular_component, nan=0.0, posinf=1.0, neginf=0.0)

        sss_attenuation = sss_net(depth, appearance_embed)
        if torch.isnan(sss_attenuation).any() or torch.isinf(sss_attenuation).any():
            print("Warning: NaN/Inf detected in sss_attenuation")
            sss_attenuation = torch.nan_to_num(sss_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        diffuse_component = rendered_J - specular_component
        if torch.isnan(diffuse_component).any() or torch.isinf(diffuse_component).any():
            print("Warning: NaN/Inf detected in diffuse_component")
            diffuse_component = torch.nan_to_num(diffuse_component, nan=0.0, posinf=1.0, neginf=0.0)

        scattered_diffuse = diffuse_component * sss_attenuation
        if torch.isnan(scattered_diffuse).any() or torch.isinf(scattered_diffuse).any():
            print("Warning: NaN/Inf detected in scattered_diffuse")
            scattered_diffuse = torch.nan_to_num(scattered_diffuse, nan=0.0, posinf=1.0, neginf=0.0)

        final_rendering = scattered_diffuse + specular_component
        if torch.isnan(final_rendering).any() or torch.isinf(final_rendering).any():
            print("Warning: NaN/Inf detected in final_rendering")
            final_rendering = torch.nan_to_num(final_rendering, nan=0.0, posinf=1.0, neginf=0.0)

        return final_rendering, specular_component, sss_attenuation
