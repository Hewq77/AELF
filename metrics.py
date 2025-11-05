import torch
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import sewar as sewar_api
from skimage.metrics import structural_similarity
import cv2


def calc_ergas(img_tgt, img_fus, scale_ratio):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/scale_ratio*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam

def calc_cc_cuda(H_ref, H_fuse):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/torch.sqrt(torch.sum((H_fuse_reshaped- mean_fuse)**2, 1)*torch.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = torch.mean(CC)
    return CC

def calc_cc(H_ref, H_fuse):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.reshape(N_spectral, -1)
    H_ref_reshaped = H_ref.reshape(N_spectral, -1)

    # Calculating mean value
    mean_fuse = np.mean(H_fuse_reshaped, 1)[:, np.newaxis]
    mean_ref = np.mean(H_ref_reshaped, 1)[:, np.newaxis]

    CC = np.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/np.sqrt(np.sum((H_fuse_reshaped- mean_fuse)**2, 1)*np.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = np.mean(CC)
    return CC

def calc_uiqi(img_tgt, img_fus):
    """
    计算 UIQI（Universal Image Quality Index）
    输入：
        img_tgt: 原图，形状可为 (H, W) 或 (C, H, W)
        img_fus: 对比图（重建/融合图）
    输出：
        uiqi: 单个数值, 两个图像的整体UIQI评分
    """
    img_tgt = np.squeeze(img_tgt).astype(np.float64)
    img_fus = np.squeeze(img_fus).astype(np.float64)

    # 如果是多通道图像，转换为 (C, H*W)
    if img_tgt.ndim == 3:
        C, H, W = img_tgt.shape
        img_tgt = img_tgt.reshape(C, -1)
        img_fus = img_fus.reshape(C, -1)
    else:
        img_tgt = img_tgt.reshape(1, -1)
        img_fus = img_fus.reshape(1, -1)

    uiqi_all = []

    for i in range(img_tgt.shape[0]):
        x = img_tgt[i]
        y = img_fus[i]

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.mean((x - mean_x) * (y - mean_y))

        numerator = 4 * cov_xy * mean_x * mean_y
        denominator = (var_x + var_y) * (mean_x**2 + mean_y**2) + 1e-8  # 避免除零
        uiqi = numerator / denominator
        uiqi_all.append(uiqi)

    return np.mean(uiqi_all)
#### SSIM ####
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window
#
# def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
#     mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
#     mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2
#
#     sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2
#
#     C1 = 0.01**2
#     C2 = 0.03**2
#
#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
# def calc_ssim(img1, img2, window_size = 11, size_average = True):
#     # (_, channel, _, _) = img1.size()
#     (_, channel, _, _) = img1.shape
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)


#### version 2 ####
# def calc_ssim2(x_true, x_pred, data_range=255, sewar=False):
#     r"""
#     Args:
#         x_true (np.ndarray): target image, shape like [H, W, C]
#         x_pred (np.ndarray): predict image, shape like [H, W, C]
#         data_range (int): max_value of the image
#         此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
#         sewar (bool): use the api from sewar, Default: False
#     Returns:
#         float: SSIM value
#     """
#     if sewar:
#         return sewar_api.ssim(x_true, x_pred, MAX=data_range)[0]
#
#     return structural_similarity(x_true, x_pred, data_range=data_range, multichannel=True)

#### version 3 ####
def normalize_img(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

def to_uint8(img):
    img = np.squeeze(img)  # 如果你图像有多余维度，先 squeeze 一下
    img_min = img.min()
    img_max = img.max()
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)  # 归一化到 [0, 1]
    img_uint8 = (img_norm * 255).round().astype(np.uint8)   # 映射到 [0, 255]
    return img_uint8

def calc_ssim(img_tgt, img_fus, win_size=11 ):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = np.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = np.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    # img_tgt = img_tgt.cpu().numpy()
    # img_fus = img_fus.cpu().numpy()
    
    img_tgt = to_uint8(img_tgt)
    img_fus = to_uint8(img_fus)
    ssim = structural_similarity(img_tgt, img_fus,win_size=win_size)

    return ssim