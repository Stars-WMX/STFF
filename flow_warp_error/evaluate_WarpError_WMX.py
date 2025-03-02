import os
import glob
import numpy as np
import torch
from networks.resample2d_package.resample2d import Resample2d
import utils

class Opts:
    cuda = torch.cuda.is_available()
    data_dir = "/tdx/WMX/Flow_Error/result/STDF"
opts = Opts()
torch.cuda.set_device(3)

device = torch.device("cuda" if opts.cuda else "cpu")
flow_warping = Resample2d().to(device)
video_list = sorted(os.listdir(os.path.join(opts.data_dir, "fw_flow")))

err_all = []

for v, video in enumerate(video_list):
    print(f"Processing video: {video}")
    frame_dir = os.path.join(opts.data_dir, "fw_flow_rgb", video)
    occ_dir = os.path.join(opts.data_dir, "fw_occlusion", video)
    flow_dir = os.path.join(opts.data_dir, "fw_flow", video)

    frame_list = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    err = 0


    for t in range(1, len(frame_list)):
        print(f"Evaluate Warping Error on: video {v + 1}/{len(video_list)}, frame {t}/{len(frame_list) - 1}")


        img1_path = os.path.join(frame_dir, f"{t - 1:05d}.png")
        img2_path = os.path.join(frame_dir, f"{t:05d}.png")
        img1 = utils.read_img(img1_path)  # 第一帧
        img2 = utils.read_img(img2_path)  # 第二帧


        flow_file = os.path.join(flow_dir, f"{t - 1:05d}.flo")
        flow = utils.read_flo(flow_file)


        occ_file = os.path.join(occ_dir, f"{t - 1:05d}.png")
        occ_mask = utils.read_img(occ_file)
        noc_mask = 1 - occ_mask

        with torch.no_grad():
            img2_tensor = utils.img2tensor(img2).to(device)  # 第二帧
            flow_tensor = utils.img2tensor(flow).to(device)  # 光流张量

            warp_img2 = flow_warping(img2_tensor, flow_tensor.contiguous())  # 变形后的图像
            warp_img2 = utils.tensor2img(warp_img2)  # 转为 numpy 格式

        assert warp_img2.shape == img1.shape, f"Shape mismatch: warp_img2 {warp_img2.shape}, img1 {img1.shape}"


        diff = np.multiply(warp_img2 - img1, noc_mask)  # 使用非遮挡掩码计算差异
        N = np.sum(noc_mask)
        if N == 0:
            N = diff.shape[0] * diff.shape[1] * diff.shape[2]  # 总像素数

        err += np.sum(np.square(diff)) / N

    video_err = err / (len(frame_list) - 1)
    err_all.append((video, video_err))
    print(f"Video: {video}, Warping Error: {video_err:.6f}")

average_err = np.mean([err[1] for err in err_all])
print(f"\nAverage Warping Error = {average_err}\n")

metric_filename = os.path.join(opts.data_dir, "STDF_WarpError.txt")
os.makedirs(os.path.dirname(metric_filename), exist_ok=True)

with open(metric_filename, "w") as f:
    for video, video_err in err_all:
        f.write(f"{video}: {video_err:.6f}\n")
    f.write(f"\nAverage Warping Error = {average_err:.6f}\n")

print(f"Results saved to {metric_filename}")
