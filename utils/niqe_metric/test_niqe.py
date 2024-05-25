import cv2
import warnings

from niqe import calculate_niqe

# 文件来源: https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/__init__.py
def main():
    img_path = 'C:\\Users\zqh\Desktop\\baboon.png'
    img = cv2.imread(img_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
    print(niqe_result)


if __name__ == '__main__':
    main()