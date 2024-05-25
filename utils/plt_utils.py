import math
import numpy as np
import torch
from skimage import img_as_ubyte
from matplotlib import pyplot as plt


def plot_results(images, titles, figure_size=(12, 12), col_num=4, title="", hwc=True, file_path=None):
    """
    一行画多个图
    images:(1,m)
    col num:有几列
    """
    row_num = math.ceil(len(images) / col_num)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.figure(figsize=figure_size)
    for i in range(row_num):
        for j in range(col_num):
            if i * col_num + j >= len(images):
                # 读完后,结束绘制
                break
            if i == 0:
                plt.subplot(row_num, col_num, (i * col_num) + j + 1).set_title(titles[j])
            else:
                plt.subplot(row_num, col_num, (i * col_num) + j + 1)
            # 如果排列为CHW，则转变为HWC
            if hwc:
                plt.imshow(images[i * col_num + j])
            else:
                plt.imshow(images[i * col_num + j].transpose(1, 2, 0))
            plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def tensor2img(in_tensor, h_w=None):
    if h_w:
        in_tensor = in_tensor[:, :, :h_w[0], :h_w[1]]
    restored_img = torch.clamp(in_tensor, 0, 1)
    restored_img = restored_img.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_img = img_as_ubyte(restored_img[0])
    return restored_img


def clip(tensor_list, h, w):
    return [item[:, :, :h, :w] for item in tensor_list]


def plot_line(images, titles, figure_size=(12, 12), col_num=4, title="", file_path=None, gray_index=None):
    """
    一行画多个图
    images:(1,m)
    col num:有几列
    """
    row_num = math.ceil(len(images) / col_num)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.figure(figsize=figure_size)
    for i in range(row_num):
        for j in range(col_num):
            idx = i * col_num + j
            if idx >= len(images):
                # 读完后,结束绘制
                break

            # 如果标题数量与图片数量相等，全都设标题，否则只有第一行有标题
            if len(titles) == len(images) or i == 0:
                plt.subplot(row_num, col_num, idx + 1).set_title(titles[idx])
            else:
                plt.subplot(row_num, col_num, idx + 1)

            # 判断是否要画灰度图
            if gray_index and idx in gray_index:
                plt.imshow(images[idx], cmap="gray")
            else:
                plt.imshow(images[idx])
            plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def plot_tensor(tensors, titles, figure_size=(12, 12), col_num=4, title="", file_path=None, gray_index=None):
    img_list = [tensor2img(item) for item in tensors]
    plot_line(img_list, titles, figure_size, col_num, title, file_path, gray_index)


def plot_multi_results(images, titles, figure_size=(12, 12), file_path=None, hwc=True):
    """
    多行画多个图
    images:(n,m)
    """
    plt.figure(figsize=figure_size)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    row_num = len(images)
    col_num = len(images[0])
    for i in range(row_num):
        for j in range(col_num):
            if i == 0:
                plt.subplot(row_num, col_num, (i * col_num) + j + 1).set_title(titles[j])
            else:
                plt.subplot(row_num, col_num, (i * col_num) + j + 1)

            plt.imshow(images[i][j])
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.0)

    plt.show()


def plot_multi_tensors(tensors, titles, figure_size=(12, 12), file_path=None, hwc=True, h_w=None):
    images = [[tensor2img(item, h_w) for item in item_list] for item_list in tensors]
    plot_multi_results(images, titles, figure_size, file_path, hwc)


def get_multi_hist(tensors,path=None):
    images = [[tensor2img(item) for item in tensor] for tensor in tensors]
    hists = get_hist(images)
    plot_multi_hist(hists,path)


def get_hist(images):
    """
    获取m行n列的图像的直方图
    :param images: [m,n],images[i,j]=[h,w]
    :return: [m,n,3],返回m行n列图像的3通道直方图
    """
    res = []
    for i in range(len(images)):
        rows = []
        for j in range(len(images[i])):
            img = images[i][j]
            count = get_img_count_num(img)
            rows.append(count)
        res.append(rows)
    return res


def get_img_count_num(img, show_detail=False):
    img_count_num = []
    h, w, c = img.shape
    for i in range(c):
        img_2d = img[:, :, i]
        count_num = np.zeros(256)
        for j in range(h):
            for k in range(w):
                count_num[img_2d[j][k]] += 1
        img_count_num.append(count_num)
        if show_detail:
            count_num_dict = {idx: count_num[idx] for idx in range(len(count_num)) if count_num[idx] > 0}
            print("{}: {}".format(i, str(count_num_dict)))
    return img_count_num


def plot_multi_hist(histograms,path):
    """

    :param histograms:直方图列表,每一行包含多张图片 e.g. shape:[4,2,3],有4行2列图像要画,3是通道数
    :return:
    """
    label = ['B', 'G', 'R']
    color = ["dodgerblue", "seagreen", "r"]
    plt.figure(figsize=(12,6))
    for i in range(len(histograms)):
        for j in range(len(histograms[i])):
            plt.subplot(len(histograms), len(histograms[i]), (i * len(histograms[i])) + j + 1)
            for c in range(3):
                plt.plot(range(256), histograms[i][j][c], label=label[c], color=color[c])
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_histogram(img, path=None, title=None):
    """
    绘制图片的直方图(像素分布直方图)
    :param path:
    :param img:
    :return:
    """
    _, _, colorChannel = img.shape
    label = ['B', 'G', 'R']
    color = ["dodgerblue", "seagreen", "r"]
    plt.figure()
    for i in reversed(range(colorChannel)):
        # print(i)
        # hist_img, _ = np.histogram(img[:, :, i], 256)
        # print(Counter(hist_img))
        img_2d = img[:, :, i]
        count_num = np.zeros(256)
        for j in range(len(img_2d)):
            for k in range(len(img_2d[i])):
                count_num[img_2d[j][k]] += 1
        count_num_dict = {idx: count_num[idx] for idx in range(len(count_num)) if count_num[idx] > 0}
        print("{}: {}".format(label[i], str(count_num_dict)))
        plt.plot(range(256), count_num, label=label[i], color=color[i])
        # plt.scatter(range(30), count_num[:30], label=label[i], color=color[i])
        if title:
            plt.title(title)
        plt.xlabel("pixel value")
        plt.ylabel("pixel number")
        plt.legend(loc='best')

    if path:
        plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    print()
