import sys

import numpy as np
import torch
import torch.nn.functional as F


def pad_img_v3(model,img_tensor, mul=16):
    """
    非重叠裁剪图片时，padding图片，将图片转换为16的倍数
    :param img_tensor:
    :return:
    """
    # print("[padding: {}]".format(sys._getframe().f_code.co_name))
    _, _, ori_h, ori_w = img_tensor.size()

    new_h, new_w = ((ori_h + mul) // mul) * mul, ((ori_w + mul) // mul) * mul
    pad_h = new_h - ori_h if ori_h % mul != 0 else 0
    pad_w = new_w - ori_w if ori_w % mul != 0 else 0
    margin = 4 * mul
    print("margin:{}".format(margin))
    pad_h += margin
    pad_w += margin
    image = F.pad(img_tensor, (margin, pad_w, margin, pad_h), 'reflect')
    output = model(image)
    return output[:, :, margin:margin + ori_h, margin:margin + ori_w]


def scale_process_non_overlap(model,image, crop_h, crop_w):
    b, c, ori_h, ori_w = image.size()
    stride_h = crop_h  # 步长
    stride_w = crop_w
    # 可以走几步
    grid_h = int(np.ceil(float(ori_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(ori_w - crop_w) / stride_w) + 1)

    output = torch.zeros((b, c, ori_h, ori_w)).to(image.device)
    # 计算每个像素被计算的次数
    count_crop = torch.zeros((b, c, ori_h, ori_w)).to(image.device)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            # 计算每个patch对应的起点和终点
            start_h = index_h * stride_h
            end_h = min(start_h + crop_h, ori_h)
            start_h = end_h - crop_h

            start_w = index_w * stride_w
            end_w = min(start_w + crop_w, ori_w)
            start_w = end_w - crop_w

            image_crop = image[:, :, start_h:end_h, start_w:end_w].clone().detach()
            count_crop[:, :, start_h:end_h, start_w:end_w] += 1
            # result = pad_img(image_crop)
            # result = pad_img_v2(image_crop)
            # result = pad_img_v3(model,image_crop)
            result = model(image_crop)
            output[:, :, start_h:end_h, start_w:end_w] += result

    return output / count_crop
