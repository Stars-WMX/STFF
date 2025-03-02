"""Toolbox for python code.
To-do:
    yuv2rgb:   8 bit yuv 420p
    yuv2ycbcr: 8 bit yuv 420p
    ycbcr2yuv: 8 bit yuv 420p
    imread
Ref:
    scikit-image: https://scikit-image.org/docs/
    mmcv: https://mmcv.readthedocs.io/en/latest/
    opencv-python
    BasicSR: https://github.com/xinntao/BasicSR
"""
from .file_io import import_yuv, write_ycbcr, FileClient, dict2str, CPUPrefetcher
from .conversion import img2float32, ndarray2img, rgb2ycbcr, ycbcr2rgb, rgb2gray, gray2rgb, bgr2rgb, rgb2bgr, paired_random_crop, augment, totensor, yuv2rgb
from .metrics import calculate_psnr, calculate_ssim, calculate_mse
from .deep_learning import set_random_seed, init_dist, get_dist_info, DistSampler, create_dataloader, CharbonnierLoss, FFTCharbonnierLoss, PSNR, CosineAnnealingRestartLR
from .system import mkdir, get_timestr, Timer, Counter
from .lmdb import make_lmdb_from_imgs, make_y_lmdb_from_yuv

from .loss import FFTLoss, CharbonnierLoss, CVQE_Loss, CVQE_Loss1, CVQE_Loss2


__all__ = [
    'import_yuv', 'write_ycbcr', 'FileClient', 'dict2str', 'CPUPrefetcher',

    'img2float32', 'ndarray2img', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2gray', 'gray2rgb', 'bgr2rgb', 'rgb2bgr', 'paired_random_crop', 'augment', 'totensor', 'yuv2rgb',

    'calculate_psnr', 'calculate_ssim', 'calculate_mse',

    'set_random_seed', 'init_dist', 'get_dist_info', 'DistSampler', 'create_dataloader', 'CharbonnierLoss', 'FFTCharbonnierLoss', 'PSNR', 'CosineAnnealingRestartLR',

    'mkdir', 'get_timestr', 'Timer', 'Counter',

    'make_lmdb_from_imgs', 'make_y_lmdb_from_yuv',
    'FFTLoss', 'CharbonnierLoss', 'CVQE_Loss', 'CVQE_Loss1', 'CVQE_Loss2',

    ]
