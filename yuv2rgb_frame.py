import os.path as op
import cv2
import os
import numpy as np
import glob
from tqdm import tqdm

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


def import_yuv(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=True):
    global u_seq, v_seq
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size

    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:  # 是否只提取 Y 通道
        return y_seq
    else:
        return y_seq, u_seq, v_seq

data_dir = '/home/tdx/桌面/Project/WMX/data/FPEA/test/QP37'
video_list = sorted(glob.glob(op.join(data_dir, '*.yuv')))

def main():
    # ==========
    # Load entire video
    # ==========
    for cdx in range(len(video_list)):
        yuv_path = video_list[cdx]
        vname = yuv_path.split("/")[-1].split('.')[0]
        _, wxh, nfs = vname.split('_')
        nfs = int(nfs)
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])
        y, u, v = import_yuv(seq_path=yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False)  # (nfs, h, w)

        with tqdm(total=nfs, desc=f'Processing {vname}', unit='frame') as pbar:
            for idx in range(1, nfs+1): # 确保生成的 png 图片是从 f001.png 开始的
                y_frame = y[idx-1]
                u_frame = u[idx-1]
                v_frame = v[idx-1]

                # 构造目录路径
                directory = f'/home/tdx/桌面/Project/WMX/data/FPEA_png/test/QP37/{vname}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = f'{directory}/f{idx:03d}.png'
                rgb_frame = yuv2rgb(y_frame, u_frame, v_frame, h, w)

                cv2.imwrite(filename, rgb_frame)
                pbar.update(1)
        print(f"视频 {vname} 处理完成。")

if __name__ == '__main__':
    main()


