import random
import torch
import skimage.color as skc
import numpy as np
import cv2


def img2float32(img):
    """Convert the type and range of the input image into np.float32 and [0, 1].
    Args:
        img (img in ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type, with unknown range.

    Return:
        img (ndarray): The converted image with type of np.float32 and 
        range of [0, 1].
    """
    img_type = img.dtype
    assert img_type in (np.uint8, np.float32), f'The image type should be np.float32 or np.uint8, but got {img_type}'
    
    if img_type == np.uint8:  # the range must be [0, 255]
        img = img.astype(np.float32)
        img /= 255.
    else:  # np.float32, may excess the range [0, 1]
        img = img.clip(0, 1)

    return img


def ndarray2img(ndarray):
    """Convert the type and range of the input ndarray into np.uint8 and 
    [0, 255].
    Args:
        ndarray (ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type with unknown range.

    Return:
        img (img in ndarray): The converted image with type of np.uint8 and 
        range of [0, 255].

    对float32类型分情况讨论: 
        1. 如果最大值超过阈值, 则视为较黑的图像, 直接clip处理；
        2. 否则, 视为[0, 1]图像处理后的结果, 乘以255.再clip.
    不能直接astype, 该操作会删除小数, 不精确. 应先round, 再clip, 再转换格式.
    image -> img2float32 -> ndarray2img 应能准确还原.
    """
    data_type = ndarray.dtype
    assert data_type in (np.uint8, np.float32), (
        f'The data type should be np.float32 or np.uint8, but got {data_type}')

    if data_type == np.float32:
        detection_threshold = 2
        if (ndarray < detection_threshold).all():  # just excess [0, 1] slightly
            ndarray *= 255.
        else:  # almost a black picture
            pass
        img = ndarray.round()  # first round. directly astype will cut decimals
        img = img.clip(0, 255)  # or, -1 -> 255, -2 -> 254!
        img = img.astype(np.uint8)
    else:
        img = ndarray

    return img


def rgb2ycbcr(rgb_img):
    """RGB to YCbCr color space conversion.
    Args:
        rgb_img (img in ndarray): (..., 3) format.

    Return:
        ycbcr_img (img in ndarray): (..., 3) format.
    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    ycbcr_img = skc.rgb2ycbcr(rgb_img)
    return ycbcr_img



def ycbcr2rgb(ycbcr_img):
    """YCbCr to RGB color space conversion.
    Args:
        ycbcr_img (img in ndarray): (..., 3) format.

    Return:
        rgb_img (img in ndarray): (..., 3) format.

    Error:
        ycbcr_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `ycbcr2rgb` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    rgb_img = skc.ycbcr2rgb(ycbcr_img)
    return rgb_img


def rgb2gray(rgb_img):
    """Compute luminance of an RGB image.
    Args:
        rgb_img (img in ndarray): (..., 3) format.
    Return:
        gray_img (single channel img in array)

    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    alpha通道会被忽略.
    """
    gray_img = skc.rgb2gray(rgb_img)
    return gray_img


def gray2rgb(gray_img):
    """Create an RGB representation of a gray-level image.
    Args:
        gray_img (img in ndarray): (..., 1) or (... , ) format.
    Return:
        rgb_img (img in ndarray)
    Input image, not float array!
    其实还有一个alpha通道参数, 但不常用. 参见: 
    https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.gray2rgb
    """
    rgb_img = skc.gray2rgb(gray_img, alpha = None)
    return rgb_img


def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img


def rgb2bgr(img):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    img = cv2.cvtColor(img, code)
    return img


def yuv2rgb(Y,U,V,h,w):
    bgr_data = np.zeros((h, w, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    c = (Y-np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data


# ==========
# Data augmentation 数据增强操作
# ==========

def paired_random_crop(img_gts, img_lqs, gt_patch_size, gt_path, scale=1):
    """成对的随机裁剪。Paired random crop
        它使用相应的位置裁剪GT和LQ图像列表。
        参数：
            img_gts（list [ndarray] | ndarray）：GT图像。请注意，所有图像应具有相同的形状。如果输入是ndarray，则将其转换为包含自身的列表。
            img_lqs（list [ndarray] | ndarray）：LQ图像。请注意，所有图像应具有相同的形状。如果输入是ndarray，则将其转换为包含自身的列表。
            gt_patch_size（int）：GT块大小。
            scale（int）：缩放因子。
            gt_path（str）：Ground-truth的路径。

        返回：
            list [ndarray] | ndarray：GT图像和LQ图像。如果返回结果只有一个元素，则只返回ndarray。
    """
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None):
    """Augment: horizontal flips or rotate (0, 90, 180, 270 degrees).数据增强：水平翻转或旋转（0、90、180、270度）。
        使用垂直翻转和转置进行旋转实现。
        列表中的所有图像使用相同的增强。
        参数：
            imgs（list [ndarray]）：要增强的图像列表。
            hflip（bool）：水平翻转。默认值：True。
            rotation（bool）：是否旋转。默认值：True。
            flows（list [ndarray]）：要增强的流列表。维度为（h，w，2）。默认值：None。
        返回：
            list [ndarray] | ndarray：增强的图像和流。如果返回结果只有一个元素，则只返回ndarray。
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _imflip_(img, direction='horizontal'):
        """Inplace flip an image horizontally or vertically.
        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or "vertical" or "diagonal".
        Returns:
            ndarray: The flipped image (inplace 这个代表原地操作).
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return cv2.flip(img, 1, img)
        elif direction == 'vertical':
            return cv2.flip(img, 0, img)
        else:
            return cv2.flip(img, -1, img)

    def _augment(img):
        if hflip:
            _imflip_(img, 'horizontal')
        if vflip:
            _imflip_(img, 'vertical')
        if rot90:  # for (H W 3) image, H <-> W
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    def _augment_flow(flow):
        if hflip:
            _imflip_(flow, 'horizontal')
            flow[:, :, 0] *= -1
        if vflip:
            _imflip_(flow, 'vertical')
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        return imgs


def totensor(imgs, opt_bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        opt_bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have one element, just return tensor.
    """
    def _totensor(img, opt_bgr2rgb, float32):
        if img.shape[2] == 3 and opt_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1)) # 维度位置调换，将图像的维度顺序调整为(通道数C, 高度H, 宽度W)
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, opt_bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, opt_bgr2rgb, float32)
