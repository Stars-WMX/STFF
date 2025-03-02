import os
import time
import os.path as op


def mkdir(dir_path):
    """创建文件夹. 如果文件夹已存在, 在名称后依次添加 _1, _2 等后缀.
    Args:
        dir_path (str): 目录路径
    """
    original_dir_path = dir_path
    suffix = 1

    # 循环检查目录是否存在，如果存在则添加后缀
    while op.exists(dir_path):
        dir_path = f"{original_dir_path}_{suffix}"
        suffix += 1

    os.makedirs(dir_path)
    return dir_path  # 返回最终创建的目录路径

# ==========
# Time
# ==========
def get_timestr():
    """获取本地时间 str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

# 计时器类 Timer 的实现，用来记录时间
class Timer:
    def __init__(self):
        self.reset()

    # 重置计时器，将开始时间和累积时间都设置为当前时间。
    def reset(self):
        self.start_time = time.time()
        self.accum_time = 0

    # 重新开始计时，将开始时间设置为当前时间
    def restart(self):
        self.start_time = time.time()

    # 累积时间，将累积时间增加从上次开始计时到当前时间的时间间隔。
    def accum(self):
        self.accum_time += time.time() - self.start_time
    # 获取当前时间
    def get_time(self):
        return time.time()
    # 获取从上次开始计时到当前时间的时间间隔
    def get_interval(self):
        return time.time() - self.start_time
    # 获取累积时间
    def get_accum(self):
        return self.accum_time

# 计数器类 Counter 的实现，用于统计数量和计算平均值
class Counter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        self.accum_volume = 0

    def accum(self, volume):
        self.time += 1
        self.accum_volume += volume

    def get_ave(self):
        return self.accum_volume / self.time
