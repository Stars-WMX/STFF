import skimage.metrics as skm


def calculate_psnr(img0, img1, data_range=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum possible values). By default, this is estimated from the image data-type.
    Return:
        psnr (float)
    """
    psnr = skm.peak_signal_noise_ratio(img0, img1, data_range=data_range) 
    return psnr

def calculate_ssim(img0, img1, data_range=None):
    """Calculate SSIM (Structural SIMilarity).
    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum possible values). By default, this is estimated from the image data-type.
    Return:
        ssim (float)
    """
    ssim = skm.structural_similarity(img0, img1, data_range=data_range)
    return ssim

def calculate_mse(img0, img1):
    """Calculate MSE (Mean Square Error).
    Args:
        img0 (ndarray)
        img1 (ndarray)
    Return:
        mse (float)
    """
    mse = skm.mean_squared_error(img0, img1)
    return mse


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PSNR, self).__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, X, Y):
        c = X.shape[0]
        if c == 1:  # Y 通道 时
            mse = self.mse_func(X, Y)
            psnr = 10. * torch.log10(torch.reciprocal(mse))
        elif c == 3:  # 3 通道 时
            all_psnr = 0
            for i in range(c):
                mse = self.mse_func(X[i], Y[i])
                psnr = 10. * torch.log10(torch.reciprocal(mse))
                all_psnr += psnr
            psnr = all_psnr / c
        else:
            raise NotImplementedError('PSNR for %d channels is not implemented' % c)
        return psnr


def calculate_ssim_pt(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    Ref:
    对于三通道图像，先计算每个通道的 SSIM，然后求平均值。
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): 图像各边缘的裁剪像素。这些像素不参与计算。
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.  只计算 Y 通道的SSIM值
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if img.shape[1] != 1:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)  # 调用函数在下面
    return ssim


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    它由 func:`calculate_ssim_pt` 调用
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def rgb2ycbcr_pt(img, y_only=False):
    """将 RGB 图像转换为 YCbCr 图像（PyTorch 版本）。
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): 是否只返回 Y 通道。 Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
