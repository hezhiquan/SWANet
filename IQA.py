import glob
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import argparse

from natsort import natsort
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import lpips
from tqdm import tqdm
from utils import tensor2img, load_img, paths_from_folder


def ssim_img(low_img, high_img, gray_scale=False):
    """
    可处理多种取值范围的图像
    :param low_img:
    :param high_img:
    :param gray_scale:
    :return:
    """
    if gray_scale:
        score, diff = structural_similarity(cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY),
                                            cv2.cvtColor(high_img, cv2.COLOR_RGB2GRAY),
                                            full=True,
                                            multichannel=True)
    # multichannel: If True, treat the last dimension of the array as channels.
    # Similarity calculations are done independently for each channel then averaged.
    else:
        score = structural_similarity(low_img, high_img, channel_axis=-1, multichannel=True)
    return score


def psnr_img(low_img, high_img):
    """
    可处理多种取值范围的图像
    :param low_img:
    :param high_img:
    :return:
    """
    psnr_val = peak_signal_noise_ratio(low_img, high_img)
    return psnr_val


def ssim_tensor(x, y, h_w, gray_scale=False):
    x1 = tensor2img(x, h_w)
    y1 = tensor2img(y, h_w)
    score = ssim_img(x1, y1, gray_scale)
    return score


def psnr_tensor(x, y, h_w):
    x1 = tensor2img(x, h_w)
    y1 = tensor2img(y, h_w)
    val = psnr_img(x1, y1)
    return val


class IQAMetric:
    """
    图像质量评估(IQA)
    输入的图像范围[0,255], tensor[0,1]
    提供基础的求平均功能
    """

    def __init__(self, net='alex', use_gpu=False, metric_type=["psnr", "ssim"]):
        """
        :param net: 算lpips时的感知网络
        :param use_gpu: 算lpips时是否使用gpu
        :param metric_type: 要计算的图像指标
        """
        self.device = 'cuda' if use_gpu else 'cpu'
        self.metric_dict = {"psnr": psnr_img, "ssim": ssim_img, "lpips": self.lpips}
        self.metric_methods = []
        self.metric_names = []
        for metric in metric_type:
            func = self.metric_dict.get(metric)
            if metric == "lpips":
                self.model = lpips.LPIPS(net=net)
                self.model.to(self.device)
            if func:
                self.metric_methods.append(func)
                self.metric_names.append(metric)
            else:
                raise Exception("未实现{}指标".format(metric))
        # 提供基础的求平均功能
        self.result_list = []

    def measure_from_path(self, low_img_path, high_img_path, add_record=True):
        """
        根据成对的低光照、正常光照图像的路径算 IQA
        """
        low_img, high_img = load_img(low_img_path), load_img(high_img_path)
        return self.measure_from_img(low_img, high_img, add_record)

    def measure_from_tensor(self, low_tensor, high_tensor, h_w=None, add_record=True):
        """
        根据成对的低光照、正常光照图像的 tensor 算 IQA
        """
        low_img = tensor2img(low_tensor, h_w)
        high_img = tensor2img(high_tensor, h_w)
        return self.measure_from_img(low_img, high_img, add_record)

    def measure_from_img(self, low_img, high_img, add_record=True):
        """
        根据成对的低光照、正常光照图像的 img 算 IQA
        """

        result = [func(low_img, high_img) for func in self.metric_methods]

        if add_record:
            self.result_list.append(result)
        return result

    def get_avg(self):
        """
        获取所有指标的平均值
        """
        return np.average(np.array(self.result_list), axis=0)

    def get_named_avg(self):
        """
        获取所有指标的名字和平均值
        """
        return zip(self.metric_names, self.get_avg())

    def reset(self):
        self.result_list = []

    def lpips(self, low_img, high_img):
        """
        img取值范围[0, 255)
        :param low_img:
        :param high_img:
        :return:
        """
        low = convert(low_img).to(self.device)
        high = convert(high_img).to(self.device)
        dist = self.model.forward(low, high).item()
        return dist

    def test_dir(self, low_path, high_path):
        low_files, high_files = paths_from_folder(low_path), paths_from_folder(high_path)
        assert len(low_files) > 0 and len(low_files) == len(high_files), "The number of files is 0 or not equal"
        self.reset()
        for i in range(len(high_files)):
            self.measure_from_path(low_files[i], high_files[i], add_record=True)
        avg_result = self.get_avg()
        print("avg: ", " ".join(["{}:{}".format(self.metric_names[i], avg_result[i]) for i in range(len(avg_result))]))


def convert(img):
    """
    将[H,W,C]的图片转换为[1,C,H,W]的tensor,取值范围[-1,1]
    :param img: 形状[H,W,C],取值范围[0,255]
    :return:
    """

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

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f}, {lpips:0.3f}'


def measure_dirs(low_path, high_path, metric_type="fr", use_gpu=True, verbose=True):
    """
    计算dirA,dirB目录下的
    :param low_path:
    :param high_path:
    :param use_gpu:
    :param metric_type: 无参考(nr)还是有参考 (fr)
    :param verbose:
    :return:
    """
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None

    t_init = time.time()

    paths_a = fiFindByWildcard(os.path.join(low_path, f'*.*'))
    paths_b = fiFindByWildcard(os.path.join(high_path, f'*.*'))

    vprint("Comparing: ")
    vprint(low_path)
    vprint(high_path)

    if metric_type == "fr":
        check_path(paths_a, paths_b)
        metric = IQAMetric(use_gpu=use_gpu, metric_type=["psnr", "ssim", "lpips"])

        count = 0
        for pathA, pathB in tqdm(list(zip(paths_a, paths_b))):

            # 展示数据集的前3张图片
            count += 1
            if count <= 3:
                plt.Figure()
                plt.subplot(1, 2, 1)
                plt.imshow(load_img(pathA))
                plt.subplot(1, 2, 2)
                plt.imshow(load_img(pathB))
                plt.show()

            metric.measure_from_path(pathA, pathB, add_record=True)

        vprint(f"Final Result: {format_result(*metric.get_avg())}, {time.time() - t_init:0.1f}s")

    elif metric_type == "nr":
        assert len(paths_a) > 0, "没有图片"
        from utils.niqe_metric.niqe import calculate_niqe
        res = []
        for pathA in tqdm(paths_a):
            img = cv2.imread(pathA)
            niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
            # print("{}:{}".format(pathA.split("\\")[-1],niqe_result))
            res.append(niqe_result)
        # print("[{} niqe: {:.3f}]".format(low_path.split("/")[-2], np.average(res)))
        print("[{} niqe: {:.3f}]".format(low_path, np.average(res)))
    else:
        print("[error:指标类型 {} 不属于nr 或 fr]".format(metric_type))
        exit(1)
    print("图片个数 {}".format(len(paths_a)))


def check_path(paths_a, paths_b):
    """
    判断两个路径下的图片是否成对
    :param paths_a: 路径列表
    :param paths_b: 路径列表
    :return:
    """
    assert len(paths_b) == len(paths_a) and len(paths_b) > 0
    for i in range(len(paths_b)):
        name_a = paths_a[i].split("/")[-1]
        name_b = paths_b[i].split("/")[-1]
        if name_b != name_a:
            print("i:{} |  B {} | A {}".format(i, paths_b[i], paths_a[i]))
            raise Exception("A,B不成对")
    print("[两个目录下的图片名相同]")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default=r'/home/hzq/Code/LLIE/datasets/LOL/eval15/low',
                        type=str)
    parser.add_argument('-dirB', default=r'/home/hzq/Code/LLIE/datasets/LOL/eval15/high',
                        type=str)
    parser.add_argument('--metric_type', default='fr')
    parser.add_argument('--use_gpu', default=False)

    args = parser.parse_args()
    dirA = args.dirA
    dirB = args.dirB
    IQA = IQAMetric(use_gpu=args.use_gpu, metric_type=["psnr", "ssim", "lpips"])
    IQA.test_dir(dirA, dirB)
