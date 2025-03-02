import os
import glob
import math
import cv2
import utils
import torch
from models import FlowNet2


torch.cuda.set_device(3)

class Options:
    rgb_max = 1.0
    fp16 = False
opts = Options()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlowNet2(opts).to(device)


pretrained_path = "/tdx/WMX/Flow_Error/pretrained_models/FlowNet2_checkpoint.pth.tar"  # 替换为权重文件的实际路径

checkpoint = torch.load(pretrained_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Model loaded successfully.")


data_dir = "/tdx/WMX/STFF/duibi/2020_AAAI_STDF/wmx_STDF_result/MFQE/QP37_png"
output_dir = "/tdx/WMX/Flow_Error/result/STDF"
video_list = sorted(os.listdir(data_dir))


for video in video_list:
    frame_dir = os.path.join(data_dir, video)
    fw_flow_dir = os.path.join(output_dir, "fw_flow", video)
    fw_occ_dir = os.path.join(output_dir, "fw_occlusion", video)
    fw_rgb_dir = os.path.join(output_dir, "fw_flow_rgb", video)

    os.makedirs(fw_flow_dir, exist_ok=True)
    os.makedirs(fw_occ_dir, exist_ok=True)
    os.makedirs(fw_rgb_dir, exist_ok=True)

    frame_list = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    for t in range(len(frame_list) - 1):
        print(f"Compute flow for {video} frame {t} and {t + 1}")

        img1 = utils.read_img(frame_list[t])
        img2 = utils.read_img(frame_list[t + 1])

        size_multiplier = 64
        H_orig = img1.shape[0]
        W_orig = img1.shape[1]

        H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)

        img1 = cv2.resize(img1, (W_sc, H_sc))
        img2 = cv2.resize(img2, (W_sc, H_sc))

        with torch.no_grad():

            img1_tensor = utils.img2tensor(img1).to(device)
            img2_tensor = utils.img2tensor(img2).to(device)

            fw_flow = model(img1_tensor, img2_tensor)
            fw_flow = utils.tensor2img(fw_flow)

            bw_flow = model(img2_tensor, img1_tensor)
            bw_flow = utils.tensor2img(bw_flow)

        fw_flow = utils.resize_flow(fw_flow, W_out=W_orig, H_out=H_orig)
        bw_flow = utils.resize_flow(bw_flow, W_out=W_orig, H_out=H_orig)

        fw_occ = utils.detect_occlusion(bw_flow, fw_flow)

        output_flow_filename = os.path.join(fw_flow_dir, f"{t:05d}.flo")
        if not os.path.exists(output_flow_filename):
            utils.save_flo(fw_flow, output_flow_filename)


        output_occ_filename = os.path.join(fw_occ_dir, f"{t:05d}.png")
        if not os.path.exists(output_occ_filename):
            utils.save_img(fw_occ, output_occ_filename)

        output_rgb_filename = os.path.join(fw_rgb_dir, f"{t:05d}.png")
        if not os.path.exists(output_rgb_filename):
            flow_rgb = utils.flow_to_rgb(fw_flow)
            utils.save_img(flow_rgb, output_rgb_filename)

print("Flow computation completed.")
