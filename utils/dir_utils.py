import os
import shutil

from natsort import natsorted
from glob import glob


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def ve_lol_dir(type):
    prefix = "/home/hzq/Code/LLIE/datasets/VE_LOL/" + type + "/"
    os.makedirs(prefix + "test", exist_ok=True)
    # os.makedirs(prefix + "test/high", exist_ok=True)
    # os.makedirs(prefix + "train/low", exist_ok=True)
    # os.makedirs(prefix + "train/high", exist_ok=True)


def ve_lol_merge_test():
    # cp -r
    combine_test = "/home/hzq/Code/LLIE/datasets/VE_LOL/combine/test/"
    real_test = "/home/hzq/Code/LLIE/datasets/VE_LOL/real/test/"
    syn_test = "/home/hzq/Code/LLIE/datasets/VE_LOL/syn/test/"
    os.makedirs(combine_test + "high/", exist_ok=True)
    os.makedirs(combine_test + "low/", exist_ok=True)

    os.system("cp -r {} {}".format(real_test + "high/*", combine_test + "high/"))
    os.system("cp -r {} {}".format(real_test + "low/*", combine_test + "low/"))


    os.system("cp -r {} {}".format(syn_test + "high/*", combine_test + "high/"))
    os.system("cp -r {} {}".format(syn_test + "low/*", combine_test + "low/"))
    print("high 合并后的数据:{}".format(len(os.listdir(combine_test + "high/"))))
    print("low 合并后的数据:{}".format(len(os.listdir(combine_test + "low/"))))

    # shutil.move("/home/hzq/Code/LLIE/datasets/VE_LOL/real/test/high/", combine_high)
    # shutil.move("/home/hzq/Code/LLIE/datasets/VE_LOL/syn/test/high/", combine_high)
    # print("high 合并后的数据:{}".format(len(os.listdir(combine_high))))
    # combine_low = "/home/hzq/Code/LLIE/datasets/VE_LOL/combine/test/"
    # os.makedirs(combine_low, exist_ok=True)
    # shutil.move("/home/hzq/Code/LLIE/datasets/VE_LOL/real/test/low/", combine_low)
    # shutil.move("/home/hzq/Code/LLIE/datasets/VE_LOL/syn/test/low/", combine_low)
    # print("low 合并后的数据:{}".format(len(os.listdir(combine_low))))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_last_path(path, session):
    x = natsorted(glob(os.path.join(path, '*%s' % session)))[-1]
    return x


if __name__ == "__main__":
    # ve_lol_dir("syn")
    ve_lol_merge_test()
