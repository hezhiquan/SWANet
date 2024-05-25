import os
import shutil

from natsort import natsorted


def get_low(input_path: str, output_path: str, start_index=1):
    index = start_index
    dir_list = natsorted(os.listdir(input_path))
    print("begin index:", index)
    for dir_item in dir_list:
        if dir_item.isdigit():
            new_path = os.path.join(input_path, dir_item)
            filenames = natsorted(os.listdir(new_path))
            # save_file_index = len(filenames) // 2 - 1 if len(filenames) == 7 or len(filenames) == 9 else 0
            save_file_index = 0
            arr = filenames[save_file_index].split(".")
            assert len(arr) == 2
            new_filename = str(index) + "." + arr[1]
            file_path = os.path.join(input_path, dir_item, filenames[save_file_index])
            new_file_path = os.path.join(output_path, new_filename)
            print("len:{} old:{}; new:{}".format(filenames, file_path, new_file_path))
            index += 1
            shutil.copyfile(file_path, new_file_path)
        else:
            print("dir item {} is not digit".format(dir_item))
    print("end index:", index)


def get_high(input_path: str, output_path, start_index=1):
    index = start_index
    dir_list = natsorted(os.listdir(input_path))
    print("begin index:", index)
    for dir_item in dir_list:
        file_path = os.path.join(input_path, dir_item)
        arr = dir_item.split(".")
        assert len(arr) == 2
        new_filename = str(index) + "." + arr[1]
        new_file_path = os.path.join(output_path, new_filename)
        print("old:{}; new:{}".format(file_path, new_file_path))
        index += 1
        shutil.copyfile(file_path, new_file_path)

    print("end index:", index)


def build_sice():
    get_low(r"E:\Dataset\SICE\Dataset_Part1", r"E:\Dataset\SICE_pair\total\low", start_index=1)
    get_low(r"E:\Dataset\SICE\Dataset_Part2(1)", r"E:\Dataset\SICE_pair\total\low", start_index=361)
    # get_high(r"E:\Dataset\SICE\Dataset_Part1\Label", r"E:\Dataset\SICE_pair\total\high", start_index=1)
    # get_high(r"E:\Dataset\SICE\Dataset_Part2(1)\Label", r"E:\Dataset\SICE_pair\total\high", start_index=361)


if __name__ == "__main__":
    build_sice()
