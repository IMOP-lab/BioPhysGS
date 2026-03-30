import lpips
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, floor

import numpy as np
import cv2

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2



class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def reg_l1_loss(network_output, gt):
    return torch.abs((network_output - gt) / (network_output.detach() + 1e-3)).mean()


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


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # Check for NaN or inf values in input images
    if torch.isnan(img1).any() or torch.isinf(img1).any() or torch.isnan(img2).any() or torch.isinf(img2).any():
        return torch.tensor(0.0, device=img1.device, requires_grad=True)
    
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

    # Check for NaN values in intermediate calculations
    if (torch.isnan(mu1_sq).any() or torch.isnan(mu2_sq).any() or 
        torch.isnan(sigma1_sq).any() or torch.isnan(sigma2_sq).any() or 
        torch.isnan(sigma12).any()):
        return torch.tensor(0.0, device=img1.device, requires_grad=True)

    # More stable SSIM calculation with epsilon to prevent division by zero
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Add small epsilon to prevent division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    ssim_map = numerator / denominator
    
    # Final NaN check
    if torch.isnan(ssim_map).any() or torch.isinf(ssim_map).any():
        return torch.tensor(0.0, device=img1.device, requires_grad=True)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def ms_ssim(x, y, scales=3, window_size=11):
    ssim_loss = 0
    for i in range(scales):
        ssim_loss += 1 / scales * ssim(x, y, window_size)
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)

    return ssim_loss


def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(
        cv2.Canny((image.detach().cpu().numpy() * 255.).astype(np.uint8), thres1, thres2) / 255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


with torch.no_grad():
    kernelsize = 3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize // 2))
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).reshape(1, 1, kernelsize, kernelsize)
    conv.weight.data = kernel  # torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None, None])
    cnt_map = conv((cnt_map * mask)[None, None])
    nearMean_map = (nearMean_map / (cnt_map + 1e-8)).squeeze()

    return nearMean_map


def anisotropic_total_variation_loss(img):
    d_w = torch.abs(img[:, :-1] - img[:, 1:])
    d_h = torch.abs(img[:-1, :] - img[1:, :])
    w_variation = torch.mean(d_w)
    h_variance = torch.mean(d_h)
    return h_variance + w_variation



def depth_aware_smooth_loss(invDepth, rgb_image, depth_mask=None, gamma=0.001):
    """
    Smoothness loss considering depth influence (simplified version, processes 3D tensors)
    Args:
        invDepth: Inverse depth map [1, H, W]
        rgb_image: RGB image [3, H, W]
        depth_mask: Depth mask [1, H, W]
        gamma: Parameter to avoid division by zero
    """
    # Check for NaN or inf values in input
    if torch.isnan(invDepth).any() or torch.isinf(invDepth).any():
        return torch.tensor(0.0, device=invDepth.device, requires_grad=True)
    
    # More conservative clamping for numerical stability
    invDepth_clamped = torch.clamp(invDepth, min=0.01, max=100.0)
    
    # Recover depth from inverse depth (for weight calculation)
    depth = (1.0 / invDepth_clamped)
    
    # Check for NaN after depth calculation
    if torch.isnan(depth).any() or torch.isinf(depth).any():
        return torch.tensor(0.0, device=invDepth.device, requires_grad=True)

    # More stable depth weight calculation
    depth_mean = torch.clamp(depth.mean(), min=0.1, max=100.0)
    depth_weights = torch.exp(-torch.clamp(depth / depth_mean, min=0.0, max=10.0))

    # Calculate gradients
    invdepth_dx = invDepth_clamped[:, :, 1:] - invDepth_clamped[:, :, :-1]  # [1, H, W-1]
    invdepth_dy = invDepth_clamped[:, 1:, :] - invDepth_clamped[:, :-1, :]  # [1, H-1, W]

    # Propagate weights
    weight_dx = (depth_weights[:, :, 1:] + depth_weights[:, :, :-1]) / 2
    weight_dy = (depth_weights[:, 1:, :] + depth_weights[:, :-1, :]) / 2

    # RGB gradients
    rgb_dx = rgb_image[:, :, 1:] - rgb_image[:, :, :-1]  # [3, H, W-1]
    rgb_dy = rgb_image[:, 1:, :] - rgb_image[:, :-1, :]  # [3, H-1, W]

    # RGB gradient norms with better numerical stability
    rgb_dx_norm = torch.sqrt(torch.sum(rgb_dx ** 2, dim=0, keepdim=True) + 1e-6)  # [1, H, W-1]
    rgb_dy_norm = torch.sqrt(torch.sum(rgb_dy ** 2, dim=0, keepdim=True) + 1e-6)  # [1, H-1, W]

    # More conservative clamping
    rgb_dx_norm = torch.clamp(rgb_dx_norm, min=max(gamma, 1e-4))
    rgb_dy_norm = torch.clamp(rgb_dy_norm, min=max(gamma, 1e-4))

    # Weighted smoothness terms
    smooth_x = weight_dx * torch.abs(invdepth_dx) / rgb_dx_norm
    smooth_y = weight_dy * torch.abs(invdepth_dy) / rgb_dy_norm
    
    # Check for NaN in smoothness terms
    if torch.isnan(smooth_x).any() or torch.isnan(smooth_y).any():
        return torch.tensor(0.0, device=invDepth.device, requires_grad=True)

    if depth_mask is not None:
        # Create gradient mask
        mask_dx = depth_mask[:, :, 1:] * depth_mask[:, :, :-1]  # [1, H, W-1]
        mask_dy = depth_mask[:, 1:, :] * depth_mask[:, :-1, :]  # [1, H-1, W]

        smooth_x = smooth_x * mask_dx
        smooth_y = smooth_y * mask_dy

        # More stable division
        mask_dx_sum = torch.clamp(mask_dx.sum(), min=1e-6)
        mask_dy_sum = torch.clamp(mask_dy.sum(), min=1e-6)
        
        loss = (smooth_x.sum() / mask_dx_sum + smooth_y.sum() / mask_dy_sum)
    else:
        loss = smooth_x.mean() + smooth_y.mean()
    
    # Final NaN check
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=invDepth.device, requires_grad=True)

    return loss

def rgb_saturation_loss(rgb, saturation_val: float):
    """
    Calculate RGB saturation loss
    :param rgb: RGB image
    :param saturation_val: Saturation value
    :return: RGB saturation loss
    """
    # Calculate saturation
    max_rgb = torch.max(rgb, dim=1)[0]
    min_rgb = torch.min(rgb, dim=1)[0]
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    # Calculate loss
    loss = torch.mean(torch.abs(saturation - saturation_val))
    return loss
