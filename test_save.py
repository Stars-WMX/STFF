import torch
import numpy as np
import math
import utils
import glob
import os.path as op
from collections import OrderedDict
from utils.conversion import yuv2rgb
from tqdm import tqdm
import cv2
import os
import re
from model.STFF_L import STFF_L

ckp_path = './exp/STFF_QP37_CVQE_Loss_7_600000.pth'
gt_dir = './data/MFQEV2/test_18/raw'
lq_dir = './data/MFQEV2/test_18/HM16.5_LDP/QP37'
log_fp = open('./results/STFF-L/QP37/STFF_QP37_600000.log', 'w')
gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')), key=lambda x: int(x.split('_')[-2].split('x')[0]))
lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')), key=lambda x: int(x.split('_')[-2].split('x')[0]))
torch.cuda.set_device(3)

def get_resolution(filename):
    match = re.search(r'_(\d+)x(\d+)_', op.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return 0, 0

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

def main():
    model = STFF_L()
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

    # ==========
    # Load entire video
    # ==========
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
        raw_y = utils.import_yuv(seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True) # (nfs, h, w)
        raw_y = raw_y.astype(np.float32) / 255.

        lq_y, lq_u, lq_v = utils.import_yuv(seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False) # 拿到 y u v 数据
        lq_y = lq_y.astype(np.float32) / 255. # (nfs, h, w)

        msg = '> yuv loaded.'
        print(msg)

        # ==========
        # Test
        # ==========
        unit = 'dB'
        pbar = tqdm(total=nfs, ncols=80)
        ori_psnr_counter = utils.Counter()
        enh_psnr_counter = utils.Counter()

        ori_ssim_counter = utils.Counter()
        enh_ssim_counter = utils.Counter()

        lq_y = torch.from_numpy(lq_y)
        lq_y = torch.unsqueeze(lq_y, 0).cuda()
        enhanced = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))

        with torch.no_grad():
            if h <= 720:
                for ccc in range(divide):
                    if ccc == 0:
                        enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous()) # （1, divide_bolck, 1, h, w ）
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,:divide_bolck,:, :,:]
                    elif ccc == divide - 1:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, :].contiguous()) # （1, nfs - divide * divide_bolck, 1, h, w ）
                        enhanced[:, ccc * divide_bolck:, :, :, :] = enc_all[:, add_frame:, :, :, :]
                    else:
                        enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:].contiguous()) # （1, divide_bolck, 1, h, w ）
                        enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :] = enc_all[:,add_frame:divide_bolck + add_frame, :, :,:]
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
                                enhanced[:, ccc * divide_bolck:, :, :, :int(w / 2)] = enc_all[:, add_frame:, :, :,:int(w / 2)]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,:int(w / 2) + add_h_w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, :int(w / 2)] = enc_all[:,add_frame:divide_bolck + add_frame,:, :,:int(w / 2)]
                    else:
                        for ccc in range(divide):
                            if ccc == 0:
                                enc_all = model(lq_y[:, ccc * divide_bolck:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):   w] = enc_all[:,:divide_bolck,:, :, add_h_w:]
                            elif ccc == divide - 1:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:, :, int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:, :, :, int(w / 2):w] = enc_all[:, add_frame:, :, :, add_h_w:]
                            else:
                                enc_all = model(lq_y[:, ccc * divide_bolck - add_frame:(ccc + 1) * divide_bolck + add_frame, :,int(w / 2) - add_h_w:w].contiguous())
                                enhanced[:, ccc * divide_bolck:(ccc + 1) * divide_bolck, :, :, int(w / 2):w] = enc_all[:, add_frame:divide_bolck + add_frame,:, :, add_h_w:]

        enhanced = np.float32(enhanced.cpu())
        lq_y = np.float32(lq_y.cpu())
        for idx in range(nfs):
            batch_ori = utils.calculate_psnr(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            batch_perf = utils.calculate_psnr(enhanced[0, idx, 0,:,:], raw_y[idx],data_range=1.0)
            ssim_ori = utils.calculate_ssim(lq_y[0, idx,...], raw_y[idx],data_range=1.0)
            ssim_perf = utils.calculate_ssim(enhanced[0, idx, 0,:,:], raw_y[idx], data_range=1.0)

            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            ori_ssim_counter.accum(volume=ssim_ori)
            enh_ssim_counter.accum(volume=ssim_perf)

            #display
            pbar.set_description(
                "[{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(batch_ori, unit, batch_perf, unit)
                )
            pbar.update()

            frame_msg = "VideoName: {:s}, Frame: {:d}, ori: [{:.4f}], enh: [{:.4f}], delta: [{:.4f}], ori_ssim: [{:.5f}], enh_ssim: [{:.5f}], delta_ssim: [{:.4f}]".format(
                vname, idx, batch_ori, batch_perf, (batch_perf - batch_ori), ssim_ori, ssim_perf,
                (ssim_perf - ssim_ori)
            )
            log_fp.write(frame_msg + '\n')
            log_fp.flush()

        pbar.close()
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        ori_ssim = ori_ssim_counter.get_ave()
        enh_ssim = enh_ssim_counter.get_ave()
        msg = "VideoName: {:s}, ori_psnr: [{:.4f}], enh_psnr: [{:.4f}], delta_psnr: [{:.4f}], ori_ssim: [{:.5f}], enh_ssim: [{:.5f}], delta_ssim: [{:.4f}]".format(
            vname,ori_, enh_, (enh_ - ori_), ori_ssim, enh_ssim, (enh_ssim - ori_ssim) * 100
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

        enhanced = np.squeeze(enhanced)
        enhanced = (enhanced * 255).astype(np.uint8)
        for idx in range(nfs):
            y_frame = enhanced[idx]
            u_frame = lq_u[idx]
            v_frame = lq_v[idx]

            directory = f'./result/STFF-L/QP37/{vname}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = f'{directory}/f{idx+1:03}.png'
            rgb_frame = yuv2rgb(y_frame, u_frame, v_frame, h, w)

            cv2.imwrite(filename, rgb_frame)
        print("保存完成。")
    log_fp.close()
if __name__ == '__main__':
    main()
