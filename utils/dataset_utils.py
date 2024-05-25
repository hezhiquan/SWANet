import os
from glob import glob

import torch

import shutil

from natsort import natsorted


def copy(filenames, dst):
    for name in filenames:
        shutil.copy(name, dst)
        # print(name, " | ", dst)


def split_mit_dataset():
    """
    将mit数据集前1500张归入训练集，后500张设为测试集
    """
    print("[split_mit_dataset]")
    file_path = '/home/hzq/Code/LLIE/datasets/MIT_Adobe_fivek/'
    train_low_path = '/home/hzq/Code/LLIE/datasets/MIT_Adobe_fivek_split/train/low/'
    test_low_path = '/home/hzq/Code/LLIE/datasets/MIT_Adobe_fivek_split/test/low/'
    train_high_path = "/home/hzq/Code/LLIE/datasets/MIT_Adobe_fivek_split/train/high/"
    test_high_path = "/home/hzq/Code/LLIE/datasets/MIT_Adobe_fivek_split/test/high/"
    os.makedirs(train_low_path, exist_ok=True)
    os.makedirs(test_low_path, exist_ok=True)
    os.makedirs(train_high_path, exist_ok=True)
    os.makedirs(test_high_path, exist_ok=True)
    # low
    filenames = natsorted(glob(file_path + 'low/*.jpg'))
    print("len {} | first {} | 487 {} last {}".format(len(filenames), filenames[0], filenames[486], filenames[-1]))
    train_low_filenames, test_low_filenames = filenames[:4500], filenames[4500:]
    copy(train_low_filenames, train_low_path)
    copy(test_low_filenames, test_low_path)

    # high
    high_filenames = natsorted(glob(file_path + 'high/*.jpg'))
    print("len {} | first {} | 487 {} last {}".format(len(high_filenames), high_filenames[0], high_filenames[486],
                                                      high_filenames[-1]))
    train_high_filenames, test_high_filenames = high_filenames[:4500], high_filenames[4500:]
    copy(train_high_filenames, train_high_path)
    copy(test_high_filenames, test_high_path)


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


if __name__ == "__main__":
    split_mit_dataset()
