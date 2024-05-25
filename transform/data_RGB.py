import os
from glob import glob

from transform.dataset_RGB import *
from natsort import natsorted


def get_mit_train_data(path, img_options):
    print("[load data: get_mit_train_data]")
    low_data_names = natsorted(glob(path + '/low/*.jpg'))
    train_low_data_names = low_data_names[:4500]
    high_data_names = natsorted(glob(path + '/high/*.jpg'))
    train_high_data_names = high_data_names[:4500]
    return DataLoaderTrain(train_low_data_names, train_high_data_names, img_options)


def get_mit_test_data(path, img_options):
    print("[load data: get_mit_test_data]")
    low_data_names = natsorted(glob(path + '/low/*.jpg'))
    test_low_data_names = low_data_names[4500:]
    high_data_names = natsorted(glob(path + '/high/*.jpg'))
    test_high_data_names = high_data_names[4500:]
    return DataLoaderVal(test_low_data_names, test_high_data_names, img_options)


def get_training_data(rgb_dir, img_options):
    print("[load data: get_training_data]")
    assert os.path.exists(rgb_dir)
    inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
    tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

    inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]
    return DataLoaderTrain(inp_filenames, tar_filenames, img_options)


def get_training_data_v2(low_dir, high_dir, img_options):
    print("[load data: get_training_data_v2]")
    assert os.path.exists(low_dir) and os.path.exists(high_dir)
    inp_files = sorted(os.listdir(low_dir))
    tar_files = sorted(os.listdir(high_dir))

    inp_filenames = [os.path.join(low_dir, x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(high_dir, x) for x in tar_files if is_image_file(x)]
    return DataLoaderTrain(inp_filenames, tar_filenames, img_options)


def get_validation_data_v2(low_dir, high_dir, img_options=None):
    print("[load data: get_validation_data_v2]")
    assert os.path.exists(low_dir) and os.path.exists(high_dir)
    inp_files = sorted(os.listdir(low_dir))
    tar_files = sorted(os.listdir(high_dir))

    inp_filenames = [os.path.join(low_dir, x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(high_dir, x) for x in tar_files if is_image_file(x)]

    return DataLoaderVal(inp_filenames, tar_filenames, img_options)


def get_training_data_2stage(rgb_dir, stage1_low_dir, img_options):
    print("[load data: get_training_data_2stage]")
    assert os.path.exists(rgb_dir)
    inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
    tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))
    low_files = sorted(os.listdir(stage1_low_dir))

    inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]
    low_filenames = [os.path.join(stage1_low_dir, x) for x in low_files if is_image_file(x)]
    return DataLoaderTrain2Stage(inp_filenames, tar_filenames, low_filenames, img_options)


def get_validation_data_2stage(rgb_dir, stage1_low_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    low_filenames = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
    normal_filenames = sorted(os.listdir(os.path.join(rgb_dir, 'high')))
    stage1_files = sorted(os.listdir(stage1_low_dir))

    low_paths = [os.path.join(rgb_dir, 'low', x) for x in low_filenames if is_image_file(x)]
    normal_paths = [os.path.join(rgb_dir, 'high', x) for x in normal_filenames if is_image_file(x)]
    stage1_paths = [os.path.join(stage1_low_dir, x) for x in stage1_files if is_image_file(x)]

    return DataLoaderVal2Stage(low_paths, normal_paths, stage1_paths, img_options)


def get_validation_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    low_filenames = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
    normal_filenames = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

    low_paths = [os.path.join(rgb_dir, 'low', x) for x in low_filenames if is_image_file(x)]
    normal_paths = [os.path.join(rgb_dir, 'high', x) for x in normal_filenames if is_image_file(x)]

    return DataLoaderVal(low_paths, normal_paths, img_options)


def get_test_data_noisy(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest_(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)


def get_train_val_dataset(path: str, train_img_option, val_img_option, factor: float = 0.8, ):
    """
    According to the path, divide the validation set and data set proportionally
    :param factor: Proportion of training set
    """
    print("[load data: get_train_val_dataset]")
    assert os.path.exists(path)
    assert 0 < factor <= 1, "The factor is not in the range (0,1]"
    # get filenames
    low_light_filenames = sorted(os.listdir(os.path.join(path, 'low')))
    normal_light_filenames = sorted(os.listdir(os.path.join(path, 'high')))
    assert len(low_light_filenames) == len(normal_light_filenames) and len(
        low_light_filenames) != 0, "The number of image pairs is not equal or the image does not exist"

    # get absolute path
    low_light_paths = [os.path.join(path, 'low', x) for x in low_light_filenames if is_image_file(x)]
    normal_light_paths = [os.path.join(path, 'high', x) for x in normal_light_filenames if is_image_file(x)]

    # divide training set and validation set according to proportion
    train_num = int(len(low_light_paths) * factor)
    low_train_img_paths, low_val_img_paths = low_light_paths[:train_num], low_light_paths[train_num:]
    normal_train_img_paths, normal_val_img_paths = normal_light_paths[:train_num], normal_light_paths[train_num:]

    # load dataset
    train_dataset = DataLoaderTrain(low_train_img_paths, normal_train_img_paths, train_img_option)
    val_dataset = DataLoaderVal(low_val_img_paths, normal_val_img_paths, val_img_option)
    return train_dataset, val_dataset
