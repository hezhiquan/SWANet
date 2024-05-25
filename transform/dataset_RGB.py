import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from utils.image_utils import load_img
import torch.nn.functional as F


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, low_train_paths, normal_train_paths, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.inp_filenames = low_train_paths
        self.tar_filenames = normal_train_paths

        # change
        self.inp_imgs = []
        for item in self.inp_filenames:
            self.inp_imgs.append(Image.open(item).convert('RGB'))
        self.tar_imgs = []
        for item in self.tar_filenames:
            self.tar_imgs.append(Image.open(item).convert('RGB'))

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        """
        If there is an amplification factor, amplify the data set size
        :return:
        """
        # return self.sizex * 10
        return self.sizex if not self.img_options.get('factor') else self.img_options.get('factor') * self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        """inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')"""
        # cache: avoid loading images from disk every time
        inp_img = self.inp_imgs[index_]
        tar_img = self.tar_imgs[index_]

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTrain2Stage(Dataset):
    def __init__(self, low_train_paths, normal_train_paths, stage1_low_train_paths, img_options=None):
        super(DataLoaderTrain2Stage, self).__init__()

        self.inp_filenames = low_train_paths
        self.tar_filenames = normal_train_paths
        self.stage1_low_filenames = stage1_low_train_paths

        # change
        self.inp_imgs = []
        for item in self.inp_filenames:
            self.inp_imgs.append(Image.open(item).convert('RGB'))
        self.tar_imgs = []
        for item in self.tar_filenames:
            self.tar_imgs.append(Image.open(item).convert('RGB'))
        self.stage1_low_imgs = []
        for item in self.stage1_low_filenames:
            self.stage1_low_imgs.append(Image.open(item).convert('RGB'))

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        # return self.sizex * 10
        return self.sizex if not self.img_options.get('factor') else self.img_options.get('factor') * self.sizex

    def __getitem__(self, index):
        # index_ = index % self.sizex
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        """inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')"""
        inp_img = self.inp_imgs[index_]
        tar_img = self.tar_imgs[index_]
        stage1_low_img = self.stage1_low_imgs[index_]

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
            stage1_low_img = TF.pad(stage1_low_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        stage1_low_img = TF.to_tensor(stage1_low_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        stage1_low_img = stage1_low_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            stage1_low_img = stage1_low_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            stage1_low_img = stage1_low_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            stage1_low_img = torch.rot90(stage1_low_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            stage1_low_img = torch.rot90(stage1_low_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            stage1_low_img = torch.rot90(stage1_low_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            stage1_low_img = torch.rot90(stage1_low_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            stage1_low_img = torch.rot90(stage1_low_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename, stage1_low_img


def preprocess_image(filename):
    img = Image.open(filename).convert('RGB')
    input_ = TF.to_tensor(img)
    mul = 16
    # Pad the input if not_multiple_of 8
    h, w = input_.shape[1], input_.shape[2]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = np.pad(input_, ((0, 0), (0, padh), (0, padw)), 'reflect')
    return input_, h, w


class DataLoaderVal(Dataset):
    def __init__(self, low_val_paths: list, normal_val_paths: list, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        self.inp_filenames = low_val_paths
        self.tar_filenames = normal_val_paths
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        # The width and height of low and high are the same, so just pass one.
        # width_low = width_high, height_low = height_high
        inp_img, low_old_h, low_old_w = preprocess_image(inp_path)
        tar_img, _, _ = preprocess_image(tar_path)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename, (low_old_h, low_old_w)


class DataLoaderVal2Stage(Dataset):
    def __init__(self, low_val_paths: list, normal_val_paths: list, stage1_low_paths, img_options=None, rgb_dir2=None):
        super(DataLoaderVal2Stage, self).__init__()

        self.inp_filenames = low_val_paths
        self.tar_filenames = normal_val_paths
        self.low_filenames = stage1_low_paths
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        low_path = self.low_filenames[index_]
        inp_img, low_old_h, low_old_w = preprocess_image(inp_path)
        tar_img, _, _ = preprocess_image(tar_path)
        low_img, _, _ = preprocess_image(low_path)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename, (low_old_h, low_old_w), low_img


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename


class DataLoaderTest_(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest_, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.clean_filenames = [os.path.join(rgb_dir, 'target', x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_image_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename
