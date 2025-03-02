import lmdb
import os.path as op
import cv2
from tqdm import tqdm
from multiprocessing import Pool

def make_lmdb_from_imgs(img_dir, lmdb_path, img_path_list, keys, batch=5000, compress_level=1, multiprocessing_read=False, map_size=None):
    """Make lmdb from images.
    参数:
        img_dir (str): 图像根目录。
        lmdb_path (str): LMDB保存路径。
        img_path_list (str): 图像目录下的图像子路径。
        keys (str): LMDB键。
        batch (int): 处理批量图像后，LMDB提交。
        compress_level (int): 编码图像时的压缩级别。范围从0到9，其中0表示无压缩。
        multiprocessing_read (bool): 是否使用多进程将所有图像读入内存。如果为True，它将使用多进程将所有图像读入内存。因此，您的服务器需要有足够的内存。
        map_size (int | None): LMDB环境的映射大小。如果为None，则使用来自图像的估计大小。默认值: None

    Usage instance: see STDF-PyTorch.

    用法示例：参见STDF-PyTorch。
            LMDB的内容。文件结构如下:
            example.lmdb
            ├── data.mdb
            ├── lock.mdb
            └── meta_info.txt
            data.mdb和lock.mdb是标准的lmdb文件。有关更多详细信息，请参阅
            https://lmdb.readthedocs.io/en/release/。

            meta_info.txt是一个指定的txt文件，用于记录我们数据集的元信息。
            在使用我们提供的数据集工具准备数据集时，它将自动创建。
            txt文件中的每一行记录:
                1)图像名称（带扩展名），
                2)图像形状，
                3)压缩级别，
            由一个空格分隔。
            例如，00001/0001/im1.png (256,448,3) 1
                图像路径: 00001/0001/im1.png
                图像形状（HWC）: (256,448,3)
                压缩级别: 1
                键: 00001/0001/im1
    """

    # check
    assert len(img_path_list) == len(keys), 'img_path_list and keys should have the same length, 'f'but got {len(img_path_list)} and {len(keys)}'
    assert lmdb_path.endswith('.lmdb'), "lmdb_path must end with '.lmdb'."
    assert not op.exists(lmdb_path), f'Folder {lmdb_path} already exists. Exit.'

    # display info
    num_img = len(img_path_list)

    # read all the images to memory by multiprocessing
    if multiprocessing_read:
        def _callback(arg):
            """Register imgs and shapes into the dict & update pbar."""
            key, img_byte, img_shape = arg
            dataset[key], shapes[key] = img_byte, img_shape
            pbar.set_description(f'Read {key}')
            pbar.update(1)

        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        pbar = tqdm(total=num_img, ncols=80)
        pool = Pool()  # default: cpu core num
        # read an image, and record its byte and shape into the dict
        for path, key in zip(img_path_list, keys):
            pool.apply_async(_read_img_worker, args=(op.join(img_dir, path), key, compress_level), callback=_callback)
        pool.close()
        pool.join()
        pbar.close()

    all_image_list = concat_image(img_dir, img_path_list)
    if map_size is None:
        img = cv2.imread(all_image_list[0], cv2.IMREAD_UNCHANGED)

        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])

        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10   # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(op.join(lmdb_path, 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)

    for idx, (path, key) in enumerate(zip(img_path_list, keys)):

        pbar.set_description(f'Write {key}')
        pbar.update(1)

        # load image bytes
        if multiprocessing_read:
            img_byte = dataset[key]  # read from prepared dict
            h, w, c = shapes[key]
        else:
            # concat_image
            # 0 4 001/001/im4.png
            _, img_byte, img_shape = _read_img_worker(all_image_list[idx], key, compress_level)  # use _read function

            h, w, c = img_shape
        # write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)

        # write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # commit per batch
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()


def concat_image(img_dir, img_path_list):
    length = len(img_dir)
    all_list = []
    for idx in range(length):
        all_list.append(img_dir[idx] + '/' + "%08d" % img_path_list[idx] + '.png')

    return all_list


def _read_img_worker(path, key, compress_level):
    """Read image worker.
    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.
    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    不要把该函数放到主函数里，否则无法并行。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return key, img_byte, (h, w, c)


from utils.file_io import import_yuv
import numpy as np

def _read_y_from_yuv_worker(video_path, yuv_type, h, w, index_frame, key, compress_level):
    """不要把该函数放到主函数里，否则无法并行。"""
    if h is None:
        w, h = [int(k) for k in op.basename(video_path).split('_')[1].split('x')]  # 数据集中的 yuv 命名方式 分辨率是  wxh
    img_y = import_yuv(
                            seq_path=video_path,
                            yuv_type=yuv_type,
                            h=h,
                            w=w,
                            tot_frm=1,
                            start_frm=index_frame,
                            only_y=True,
                      )
    img = np.squeeze(img_y)
    c = 1
    _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return key, img_byte, (h, w, c)  # 维度 （H, W, C）


def make_y_lmdb_from_yuv(video_path_list, index_frame_list, key_list, lmdb_path, yuv_type='420p', h=None, w=None,
                         batch=7000, compress_level=1, multiprocessing_read=False, map_size=None):
    # check
    assert lmdb_path.endswith('.lmdb'), "lmdb_path must end with '.lmdb'."
    assert not op.exists(lmdb_path), f'Folder {lmdb_path} already exists.'

    num_img = len(key_list)

    # read all the images to memory by multiprocessing
    assert multiprocessing_read, "Not implemented."

    def _callback(arg):
        """Register imgs and shapes into the dict & update pbar."""
        key, img_byte, img_shape = arg
        dataset[key], shapes[key] = img_byte, img_shape
        pbar.set_description(f'Reading {key}')
        pbar.update(1)

    dataset = {}  # use dict to keep the order for multiprocessing
    shapes = {}
    pbar = tqdm(total=num_img, ncols=80)
    pool = Pool()  # default: cpu core num

    # read an image, and record its byte and shape into the dict
    for iter_frm in range(num_img):
        pool.apply_async(
                            _read_y_from_yuv_worker,
                            args=(
                                    video_path_list[iter_frm],
                                    yuv_type,
                                    h,
                                    w,
                                    index_frame_list[iter_frm],
                                    key_list[iter_frm],
                                    compress_level,
                                 ),
                            callback=_callback
                        )

    pool.close()
    pool.join()
    pbar.close()

    # estimate map size if map_size is None
    if map_size is None:
        # find the first biggest frame
        biggest_index = 0
        biggest_size = 0
        for iter_img in range(num_img):
            vid_path = video_path_list[iter_img]
            if w is None:
                w, h = map(int, vid_path.split('.')[-2].split('_')[-2].split('x'))
            img_size = w * h
            if img_size > biggest_size:
                biggest_size = img_size
                biggest_index = iter_img
        # obtain data size of one image
        _, img_byte, _ = _read_y_from_yuv_worker(
                                                    video_path_list[biggest_index],
                                                    yuv_type,
                                                    h,
                                                    w,
                                                    index_frame_list[biggest_index],
                                                    key_list[biggest_index],
                                                    compress_level,
                                                )
        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * num_img
        map_size = data_size * 10  # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(op.join(lmdb_path, 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)
    for idx, key in enumerate(key_list):
        pbar.set_description(f'Writing {key}')
        pbar.update(1)

        # load image bytes
        img_byte = dataset[key]  # read from prepared dict
        h, w, c = shapes[key]

        # write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)

        # write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # commit per batch
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)

    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()