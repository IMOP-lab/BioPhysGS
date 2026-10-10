import torch
from torch import Tensor, nn
import torch.nn.functional as F
import tinycudann as tcnn
import math


class EndoscopeEmbeddingModel(nn.Module):
    """
    内窥镜外观建模 - 参考EmbeddingModel结构
    基于内窥镜成像机理，处理组织外观特征和光照条件
    """
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

        # 定义输入特征维度 - 与原始EmbeddingModel保持一致
        feat_in = 3
        if self.appearance_model_sh:
            feat_in = ((sh_degree + 1) ** 2) * 3
        mlp_input_dim = appearance_embedding_dim + feat_in + 6 * appearance_n_fourier_freqs

        # 根据use_tcnn参数初始化不同类型的MLP - 保持原有结构
        if self.use_tcnn:
            # 使用TCNN MLP
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
            # 使用PyTorch MLP
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, feat_in * 2),
            )

        # 内窥镜特有的组织光学特性参数
        self.tissue_albedo_factor = nn.Parameter(torch.ones(1) * 0.8)  # 组织反照率因子
        self.hemoglobin_absorption = nn.Parameter(torch.ones(3) * 0.1)  # 血红蛋白吸收系数

    def forward(self, color, image_embed, feature_embed):
        """
        前向传播 - 保持与原始EmbeddingModel相同的接口
        但加入内窥镜特有的组织光学特性处理
        """
        C0 = 0.28209479177387814
        input_color = color
        if not self.appearance_model_sh:
            color = color[..., :3]
        inp = torch.cat((color, image_embed, feature_embed), dim=-1)

        # 使用MLP（无论是TCNN还是PyTorch实现）
        output = self.mlp(inp) * 0.01
        offset, mul = torch.split(output, [color.shape[-1], color.shape[-1]], dim=-1)

        # 处理offset和mul
        offset = torch.cat((offset / C0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])

        # 内窥镜特有的组织光学特性调整
        # 应用组织反照率因子
        tissue_adjusted_color = input_color * self.tissue_albedo_factor

        # 应用血红蛋白吸收效应（主要影响红色通道）
        hemoglobin_effect = torch.ones_like(input_color)
        if input_color.shape[-1] >= 3:
            hemoglobin_effect[..., :3] = 1.0 - self.hemoglobin_absorption.unsqueeze(0).expand_as(input_color[..., :3])
            hemoglobin_effect = torch.clamp(hemoglobin_effect, 0.1, 1.0)

        # 最终输出结合原有逻辑和内窥镜特性
        return tissue_adjusted_color * hemoglobin_effect * mul + offset


class SpecularReflectionNet(nn.Module):
    """
    高光反射建模 - 参考BackscatterNet结构
    基于Blinn-Phong模型，建模内窥镜光源产生的高光反射
    """
    def __init__(self):
        super().__init__()
        # 参考BackscatterNet的卷积结构
        self.specular_conv = nn.Conv2d(1, 3, 1, bias=False)
        self.roughness_conv = nn.Conv2d(1, 3, 1, bias=False)

        # 初始化权重 - 保持与BackscatterNet相同的初始化策略
        nn.init.uniform_(self.specular_conv.weight, 0, 5.0)
        nn.init.uniform_(self.roughness_conv.weight, 0, 5.0)

        # 内窥镜特有的高光参数 - 对应BackscatterNet的B_inf和J_prime
        self.specular_intensity = nn.Parameter(torch.rand(3, 1, 1) * 0.8)  # 高光强度
        self.surface_roughness = nn.Parameter(torch.rand(3, 1, 1) * 0.2)   # 表面粗糙度

        # 激活函数 - 保持与原始模型一致
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        # 内窥镜光源参数
        self.light_distance_factor = nn.Parameter(torch.ones(1) * 2.0)  # 光源距离因子

    def forward(self, depth, appearance_embed=None):
        """
        前向传播 - 保持与BackscatterNet相同的接口
        基于深度和外观嵌入计算高光反射
        """
        # 输入数值稳定性检查
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print(f"Warning: NaN/Inf detected in SpecularReflectionNet depth input")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        # 使用与BackscatterNet相同的卷积处理方式
        specular_coeff = self.softplus(self.specular_conv(depth))
        roughness_coeff = self.softplus(self.roughness_conv(depth))

        # 内窥镜高光反射模型
        # 基于Blinn-Phong模型，考虑光源距离和表面特性
        light_attenuation = torch.exp(-torch.clamp(depth / self.light_distance_factor, 0.0, 10.0))

        # 高光反射计算 - 参考BackscatterNet的公式结构
        specular_reflection = (
            self.specular_intensity * light_attenuation * torch.exp(-torch.clamp(specular_coeff, 0.0, 10.0)) +
            self.surface_roughness * (1 - torch.exp(-torch.clamp(roughness_coeff, 0.0, 10.0)))
        )

        # 输出数值稳定性检查
        if torch.isnan(specular_reflection).any() or torch.isinf(specular_reflection).any():
            print(f"Warning: NaN/Inf detected in SpecularReflectionNet output")
            specular_reflection = torch.nan_to_num(specular_reflection, nan=0.0, posinf=1.0, neginf=0.0)

        return specular_reflection


class SubsurfaceScatterNet(nn.Module):
    """
    次表面散射衰减建模 - 参考DeattenuateNet结构
    基于Beer-Lambert定律，建模光线在组织内的散射和衰减
    """
    def __init__(self):
        super().__init__()
        # 参考DeattenuateNet的卷积结构 - 输入维度保持一致
        self.scattering_conv = nn.Conv2d(1 + 16, 6, 1, bias=False)
        nn.init.uniform_(self.scattering_conv.weight, 0, 5.0)

        # 参考DeattenuateNet的系数参数
        self.scattering_coef = nn.Parameter(torch.rand(6, 1, 1))

        # 激活函数 - 保持与原始模型一致
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        # 内窥镜组织特有的光学参数
        self.tissue_scattering_length = nn.Parameter(torch.ones(1) * 1.5)  # 组织散射长度
        self.absorption_coefficient = nn.Parameter(torch.ones(3) * 0.1)    # 吸收系数

    def forward(self, depth, appearance_embed=None):
        """
        前向传播 - 保持与DeattenuateNet相同的接口和处理流程
        """
        # 输入数值稳定性检查
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print(f"Warning: NaN/Inf detected in SubsurfaceScatterNet depth input")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        # 与DeattenuateNet完全相同的输入处理
        depth = depth.unsqueeze(0)
        appearance_embed = appearance_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        appearance_embed = appearance_embed.repeat(depth.shape[0], 1, depth.shape[2], depth.shape[3])
        depth_emd = torch.cat((depth, appearance_embed), dim=1)

        # 使用与DeattenuateNet相同的卷积处理
        scatter_conv = torch.exp(-self.sigmoid(self.scattering_conv(depth_emd)))

        # 检查卷积结果
        if torch.isnan(scatter_conv).any() or torch.isinf(scatter_conv).any():
            print(f"Warning: NaN/Inf detected in scatter_conv")
            scatter_conv = torch.nan_to_num(scatter_conv, nan=1.0, posinf=1.0, neginf=0.0)

        beta_s = torch.stack(tuple(
            torch.sum(scatter_conv[:, i:i + 2, :, :] * self.sigmoid(self.scattering_coef[i:i + 2]), dim=1)
            for i in range(0, 6, 2)
        ), dim=1)

        # 检查beta_s
        if torch.isnan(beta_s).any() or torch.isinf(beta_s).any():
            print(f"Warning: NaN/Inf detected in beta_s")
            beta_s = torch.nan_to_num(beta_s, nan=0.0, posinf=1.0, neginf=0.0)

        # 内窥镜次表面散射模型
        # 基于Beer-Lambert定律，考虑组织的散射和吸收特性
        effective_depth = depth / torch.clamp(self.tissue_scattering_length, min=0.1)

        # 计算次表面散射衰减 - 保持与DeattenuateNet相同的计算结构
        sss_attenuation = torch.exp(-1.0 * torch.clamp(beta_s, 0.0, 10.0) * effective_depth).squeeze(0)

        # 检查sss_attenuation
        if torch.isnan(sss_attenuation).any() or torch.isinf(sss_attenuation).any():
            print(f"Warning: NaN/Inf detected in sss_attenuation")
            sss_attenuation = torch.nan_to_num(sss_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        # 应用组织吸收效应
        absorption_effect = torch.exp(-torch.clamp(self.absorption_coefficient, min=0.0, max=1.0).unsqueeze(-1).unsqueeze(-1) * depth.squeeze(0))

        # 检查absorption_effect
        if torch.isnan(absorption_effect).any() or torch.isinf(absorption_effect).any():
            print(f"Warning: NaN/Inf detected in absorption_effect")
            absorption_effect = torch.nan_to_num(absorption_effect, nan=1.0, posinf=1.0, neginf=0.0)

        # 最终衰减图 - 结合散射和吸收
        final_attenuation = sss_attenuation * absorption_effect
        final_attenuation = self.sigmoid(final_attenuation)

        # 最终输出检查
        if torch.isnan(final_attenuation).any() or torch.isinf(final_attenuation).any():
            print(f"Warning: NaN/Inf detected in final_attenuation")
            final_attenuation = torch.nan_to_num(final_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        return final_attenuation


class EndoscopeRenderingModel:
    """
    内窥镜渲染模型整合类
    将三个子模块整合，提供统一的渲染接口
    """
    @staticmethod
    def compute_endoscope_rendering(rendered_J, depth, appearance_embed,
                                  specular_net, sss_net):
        """
        内窥镜渲染计算
        基于物理模型：rendered_I = (rendered_J - specular) * sss_attenuation + specular

        Args:
            rendered_J: 基础渲染结果
            depth: 深度图
            appearance_embed: 外观嵌入
            specular_net: 高光反射网络
            sss_net: 次表面散射网络
        """
        # 输入数值稳定性检查
        if torch.isnan(rendered_J).any() or torch.isinf(rendered_J).any():
            print(f"Warning: NaN/Inf detected in rendered_J")
            rendered_J = torch.nan_to_num(rendered_J, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(depth).any() or torch.isinf(depth).any():
            print(f"Warning: NaN/Inf detected in depth")
            depth = torch.nan_to_num(depth, nan=1.0, posinf=10.0, neginf=0.1)

        # 计算高光反射
        specular_component = specular_net(depth, appearance_embed)

        # 检查高光反射结果
        if torch.isnan(specular_component).any() or torch.isinf(specular_component).any():
            print(f"Warning: NaN/Inf detected in specular_component")
            specular_component = torch.nan_to_num(specular_component, nan=0.0, posinf=1.0, neginf=0.0)

        # 计算次表面散射衰减
        sss_attenuation = sss_net(depth, appearance_embed)

        # 检查次表面散射结果
        if torch.isnan(sss_attenuation).any() or torch.isinf(sss_attenuation).any():
            print(f"Warning: NaN/Inf detected in sss_attenuation")
            sss_attenuation = torch.nan_to_num(sss_attenuation, nan=1.0, posinf=1.0, neginf=0.0)

        # 内窥镜渲染公式
        # 先分离高光，再应用次表面散射，最后重新添加高光
        diffuse_component = rendered_J - specular_component

        # 检查漫反射分量
        if torch.isnan(diffuse_component).any() or torch.isinf(diffuse_component).any():
            print(f"Warning: NaN/Inf detected in diffuse_component")
            diffuse_component = torch.nan_to_num(diffuse_component, nan=0.0, posinf=1.0, neginf=0.0)

        # 安全的乘法运算
        scattered_diffuse = diffuse_component * sss_attenuation

        # 检查散射漫反射结果
        if torch.isnan(scattered_diffuse).any() or torch.isinf(scattered_diffuse).any():
            print(f"Warning: NaN/Inf detected in scattered_diffuse")
            scattered_diffuse = torch.nan_to_num(scattered_diffuse, nan=0.0, posinf=1.0, neginf=0.0)

        final_rendering = scattered_diffuse + specular_component

        # 最终结果检查
        if torch.isnan(final_rendering).any() or torch.isinf(final_rendering).any():
            print(f"Warning: NaN/Inf detected in final_rendering")
            final_rendering = torch.nan_to_num(final_rendering, nan=0.0, posinf=1.0, neginf=0.0)

        return final_rendering, specular_component, sss_attenuation
