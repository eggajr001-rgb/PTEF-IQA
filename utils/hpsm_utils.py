import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 提RGB亮度
def get_luminance(x):
    return (0.2126 * x[:, 0:1, :, :] + 0.7152 * x[:, 1:2, :, :] + 0.0722 * x[:, 2:3, :, :])



def gaussian_blur(x, kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_size += 1

    channels = x.shape[1]

    # 生成高斯核
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * 3.1415926535 * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(x.device)

    return F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=channels)



def difference_of_gaussians(x_lum, sigmas):

    g1 = gaussian_blur(x_lum, kernel_size=5, sigma=sigmas[0])
    g2 = gaussian_blur(x_lum, kernel_size=9, sigma=sigmas[1])
    g3 = gaussian_blur(x_lum, kernel_size=17, sigma=sigmas[2])

    dog1 = torch.abs(g1 - g2)
    dog2 = torch.abs(g2 - g3)
    return (dog1 + dog2) / 2.0


# 算局部RMS对比度
def local_rms_contrast(x_lum, radius):
    kernel_size = 2 * radius + 1

    local_mean = F.avg_pool2d(x_lum, kernel_size=kernel_size, stride=1, padding=radius)
    local_mean_sq = F.avg_pool2d(x_lum.pow(2), kernel_size=kernel_size, stride=1, padding=radius)

    # E[x^2] - (E[x])^2
    local_var = local_mean_sq - local_mean.pow(2)

    # 套个relu加eps，免得底下出现负数或者0导致sqrt报nan
    local_rms = torch.sqrt(F.relu(local_var) + 1e-6)
    return local_rms


# 计算综合掩蔽因子 M_map
def get_masking_factor_M(x_input, rms_radius, dog_sigmas):
    x_lum = get_luminance(x_input)
    rms_contrast = local_rms_contrast(x_lum, rms_radius)
    dog_response = difference_of_gaussians(x_lum, dog_sigmas)

    M_map = rms_contrast + dog_response

    # 简单归一化到0-1
    M_map_min = M_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    M_map_max = M_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    M_map_normalized = (M_map - M_map_min) / (M_map_max - M_map_min + 1e-6)

    return M_map_normalized


# 算特征图的梯度响应 G(x,y)
def get_gradient_response_G(features, input_size):
    B, N, C = features.shape
    H = W = input_size

    # sequence切回二维
    features_2d = rearrange(features, 'b (h w) c -> b c h w', h=H, w=W)

    # 索贝尔算子提边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=features.dtype, device=features.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=features.dtype, device=features.device)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    grad_x = F.conv2d(features_2d, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(features_2d, sobel_y, padding=1, groups=C)

    grad_norm = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    # 按通道求均值拿结构能量
    G_map = torch.mean(grad_norm, dim=1, keepdim=True)

    G_map_min = G_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    G_map_max = G_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    G_map_normalized = (G_map - G_map_min) / (G_map_max - G_map_min + 1e-6)

    return G_map_normalized


# 算最后的敏感度图 S_map
def get_sensitivity_map_S(G_map, M_map_downsampled, hpsm_lambda):
    # 用掩蔽因子压制高梯度区域的权重
    S_map = G_map / (1.0 + hpsm_lambda * M_map_downsampled)

    S_map_min = S_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    S_map_max = S_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    S_map_normalized = (S_map - S_map_min) / (S_map_max - S_map_min + 1e-6)

    return S_map_normalized