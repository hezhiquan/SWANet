import os
import shutil
from glob import glob

from natsort import natsorted


def build_checkpoint_path(mode, weights_path):
    if mode == "psnr":
        return weights_path + "/models/model_bestPSNR.pth"
    elif mode == "latest":
        return weights_path + "/models/model_latest.pth"
    elif mode == "ssim":
        return weights_path + "/models/model_bestSSIM.pth"
    else:
        raise Exception("mode 错误")


def get_paths(path):
    return natsorted(glob(os.path.join(path, '*.jpg')) +
                     glob(os.path.join(path, '*.JPG'))
                     + glob(os.path.join(path, '*.png'))
                     + glob(os.path.join(path, '*.PNG')))


def clear_path(path):
    shutil.rmtree(path, True)


def clear_test_path():
    del_dir = "/home/hzq/Code/LLIE/temp/SWANet/checkpoints/"
    del_list = os.listdir(del_dir)
    for item in del_list:
        if item.startswith("test_"):
            print(item)
    res = input("是否删除上述目录:(y/n) ")
    if res == "y":
        for item in del_list:
            if item.startswith("test_"):
                clear_path(del_dir+item)


if __name__ == "__main__":
    clear_test_path()
    # clear_path("/home/hzq/Code/LLIE/temp/SWANet/checkpoints/test_correction_net_26_01")
