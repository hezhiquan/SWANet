import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted


def get_list_average(arr):
    """
    输入数组，获取数组的平均值
    :param arr:
    :return:
    """
    return sum(arr) / len(arr)


def get_balance_scale(loss1, loss2):
    """
    输入2个loss，根据第一个loss的大小，返回可以将loss1和loss2的数值大小相等的scale
    :param loss1:
    :param loss2:
    :return:
    """
    return round(loss1 / loss2, 2)


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 255.0


def imread2tensor(img_path, rgb=False):
    """Read image to tensor.

    Args:
        img_path (str): path of image
        rgb: convert input to RGB if true
    """
    img = Image.open(img_path)
    if rgb:
        img = img.convert('RGB')
    img_tensor = TF.to_tensor(img)
    return img_tensor


def niqe(imgA):
    """ import pyiqa"""
    pass
    # imgA=np.transpose(imgA, [2, 0, 1])
    # R, G, B = imgA
    # print("R: {} ; imgA: {}".format(R.shape, imgA.shape))
    # Y = 0.299 * R + 0.587 * G + 0.114 * B
    # Cr = (R - Y) * 0.713 + 0.5
    # Cb = (B - Y) * 0.564 + 0.5
    # imgA = [Y, Cr, Cb]
    # imgA=np.asarray(imgA)
    # print(imgA.shape)
    # img_new = torch.Tensor(np.expand_dims(imgA, axis=0))/255

    """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_metric = pyiqa.create_metric('niqe')

    print(niqe_metric(imgA))"""
    # print(niqe_metric(img_png.unsqueeze(0)))

    # print("list_models : ", pyiqa.list_models())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # iqa_metric = pyiqa.create_metric("niqe", test_y_channel=True)
    # # check if lower better or higher better
    # print("iqa_metric.lower_better : ", iqa_metric.lower_better)
    # score_nr = iqa_metric(t(imgA))
    # print("score_nr: ", score_nr)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    filelist = natsorted(os.listdir(folder))
    return [os.path.join(folder, x) for x in filelist]


if __name__ == "__main__":
    # img = imread("/home/hzq/Code/LLIE/temp/SWANet/I03.bmp")
    import pyiqa

    img_tensor = imread2tensor("/home/hzq/Code/LLIE/temp/SWANet/I03.bmp").unsqueeze(0)
    img_2 = imread2tensor("/home/hzq/Code/LLIE/temp/SWANet/I03.bmp").unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_metric = pyiqa.create_metric('niqe').to(device)

    print(niqe_metric(img_2))
    # niqe(img_tensor)
