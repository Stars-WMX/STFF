import torch
import numpy as np
from collections import OrderedDict
import math
import utils
import re
from tqdm import tqdm
import glob
import os.path as op
import pandas as pd
import time
from model.STFF_S import STFF_S

def get_resolution(filename):
    match = re.search(r'_(\d+)x(\d+)_', op.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return 0, 0

gt_dir = '/tdx/WMX/data/MFQEV2/test_18/raw'
lq_dir = '/tdx/WMX/data/MFQEV2/test_18/HM16.5_LDP/QP37'

ckp_paths = [
    '/tdx/WMX/STFF/exp/MFQE_STFF_S_QP37_CVQE_Loss_7_600000/ckp_600000.pth',
]
result_dir = '/tdx/WMX/STFF/result/STFF-S/QP37/'
gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')), key=lambda x: int(x.split('_')[-2].split('x')[0]))
lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')), key=lambda x: int(x.split('_')[-2].split('x')[0]))
torch.cuda.set_device(3)

resolution_to_divide_block = {
    '2560x1600': [150],
    '1920x1080': [500, 240],
    '1280x720': [105],
    '832x480': [105, 300, 600],
    '416x240': [105, 300, 500],
}

def get_divide_block(wxh, nfs):
    if wxh in resolution_to_divide_block:
        divide_block_list = resolution_to_divide_block[wxh]
    else:
        divide_block_list = nfs

    for block_size in divide_block_list:
        if nfs == block_size:
            return block_size
    return divide_block_list[0]  # Returns the first value in the list as the default

def init_log_and_excel(weight_name):
    weight_name = weight_name.replace('ckp_', '')
    log_path = op.join(result_dir, f"STFF_S_QP37_CVQE_Loss_7_{weight_name}_FPS.log")
    excel_path = op.join(result_dir, f"STFF_S_QP37_CVQE_Loss_7_{weight_name}_FPS.xlsx")
    log_fp = open(log_path, 'w')
    return log_fp, excel_path

def save_results_to_excel(results, excel_path):
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print("Results saved to Excel file.")

def process_video(ckp_path):
    model = STFF_S()

    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path, map_location='cpu')
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    results = []
    for cdx in range(len(gt_video_list)):
        raw_yuv_path = gt_video_list[cdx]
        lq_yuv_path = lq_video_list[cdx]
        vname = raw_yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        divide_bolck = get_divide_block(wxh, nfs)
        divide = math.ceil(nfs / divide_bolck)
        add_frame = 0

        msg = f'loading raw and low-quality yuv: {vname}'
        print(msg)
        raw_y = utils.import_yuv(seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)
        lq_y = lq_y.astype(np.float32) / 255.

        msg = '> yuv loaded.'
        print(msg)

        unit = 'dB'
        pbar = tqdm(total=nfs, ncols=80)
        ori_psnr_counter = utils.Counter()
        enh_psnr_counter = utils.Counter()
        ori_ssim_counter = utils.Counter()
        enh_ssim_counter = utils.Counter()

        lq_y = torch.from_numpy(lq_y)
        lq_y = torch.unsqueeze(lq_y, 0).cuda()
        enhanced = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))

        start_time = time.time()
        processed_frames = nfs

        with torch.no_grad():
            if h <= 720:
                for ccc in range(divide):
                    if ccc == 0:
                        enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :, :].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:, :divide_bolck, :,:, :]
                    elif ccc == divide - 1:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous())
                        enhanced[:, ccc * divide_bolck:, :, :, :] = enc_all[:, add_frame:, :, :, :]
                    else:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous())
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:, add_frame:divide_bolck + add_frame, :, :, :]
            else:
                add_h_w = 4
                for bbb in range(2):
                    if bbb == 0:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,:divide_bolck,:, :,:int(w / 2)]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, :int(w / 2)] = enc_all[:, add_frame:, :, :, :int(w / 2)]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:, add_frame:divide_bolck + add_frame, :, :, :]
                    else:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:,:divide_bolck,:, :, add_h_w:]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, int(w / 2):w] = enc_all[:, add_frame:, :, :, add_h_w:]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:, add_frame:divide_bolck + add_frame, :, :, add_h_w:]

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        FPS = processed_frames / elapsed_time

        # Convert enhanced and lq_y to numpy
        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
        for idx in range(nfs):
            batch_ori = utils.calculate_psnr(lq_y[0, idx, ...], raw_y[idx], data_range=1.0)
            batch_perf = utils.calculate_psnr(enhanced[0, idx, 0, :, :], raw_y[idx], data_range=1.0)
            ssim_ori = utils.calculate_ssim(lq_y[0, idx, ...], raw_y[idx], data_range=1.0)
            ssim_perf = utils.calculate_ssim(enhanced[0, idx, 0, :, :], raw_y[idx], data_range=1.0)

            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            ori_ssim_counter.accum(volume=ssim_ori)
            enh_ssim_counter.accum(volume=ssim_perf)

            # display
            pbar.set_description(
                "[{:.4f}] {:s} -> [{:.4f}] {:s}"
                .format(batch_ori, unit, batch_perf, unit)
            )
            pbar.update()

        pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        msg = "VideoName {:s} ave: ori [{:.4f}] {:s}, enh [{:.4f}] {:s}, delta [{:.4f}] {:s}  ave ori_ssim [{:.5f}], enh_ssim [{:.5f}], delta_ssim [{:.5f}], FPS {:.4f}".format(
            vname, ori_, unit, enh_, unit, (enh_ - ori_), unit, ori_ssim, enh_ssim, (enh_ssim - ori_ssim) * 100, FPS
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

        # Append results to the list
        results.append({
            "VideoName": vname,
            "ori_psnr": ori_,
            "enh_psnr": enh_,
            "delta_psnr": enh_ - ori_,
            "ori_ssim": ori_ssim,
            "enh_ssim": enh_ssim,
            "delta_ssim": (enh_ssim - ori_ssim) * 100,
            "FPS": FPS  # Add FPS to results
        })
    return results

# 主循环
for ckp_path in ckp_paths:
    weight_name = op.basename(ckp_path).split('.')[0]
    log_fp, excel_path = init_log_and_excel(weight_name)
    results = process_video(ckp_path)
    log_fp.close()
    save_results_to_excel(results, excel_path)
