import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import argparse
from tqdm import tqdm
import utils
from IQA import IQAMetric
from model.SWANet import SWANet, load_stage1_net, load_stage2_net


def get_paths(path):
    return natsorted(glob(os.path.join(path, '*.*')))


def init_model(weight1_path, weight2_path):
    # Load corresponding models architecture and weights
    net = SWANet(weight1_path, weight2_path)
    net.to(device)
    net.eval()
    return net


# load MIANet
def init_one_stage_model(stage, weight_path):
    if stage == 1:
        net = load_stage1_net(weight_path)
    elif stage == 2:
        net = load_stage2_net(weight_path)
    else:
        raise Exception("stage num not in (1,2)")
    net.to(device)
    net.eval()
    return net


def load_and_pad_img(filename, pad_base=16):
    img = Image.open(filename).convert('RGB')
    in_img = TF.to_tensor(img).unsqueeze(0).to(device)

    # Pad the input if not_multiple_of pad_base=16
    h, w = in_img.shape[2], in_img.shape[3]
    H, W = ((h + pad_base) // pad_base) * pad_base, ((w + pad_base) // pad_base) * pad_base
    pad_h = H - h if h % pad_base != 0 else 0
    pad_w = W - w if w % pad_base != 0 else 0
    in_img = F.pad(in_img, (0, pad_w, 0, pad_h), 'reflect')
    return in_img, (h, w)


def get_model(model_type):
    if model_type == 1:
        # load stage1 net
        model = init_one_stage_model(1, args.weights_1)
    elif model_type == 2:
        # load stage2 net
        model = init_one_stage_model(2, args.weights_2)
    else:
        # load stage1 + stage2 net
        model = init_model(args.weights_1, args.weights_2)
    return model


def test():
    inp_dir = args.input_dir
    low_files = get_paths(inp_dir)
    if len(low_files) == 0:
        raise Exception(f"No files found at {inp_dir}")

    # output dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model = get_model(args.model_type)

    print('=== begin test ===')
    for i in tqdm(range(len(low_files))):
        print("{}/{}".format(i, len(low_files)))
        in_img, (h, w) = load_and_pad_img(low_files[i])

        with torch.no_grad():
            restored = model(in_img)

        # tensor to img
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        # save enhanced img
        filename, suffix = os.path.splitext(os.path.split(low_files[i])[-1])
        utils.save_img((os.path.join(out_dir, filename + suffix)), restored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Script of SWANet')
    parser.add_argument('--input_dir', default="/home/hzq/Code/LLIE/datasets/LOL/eval15/low",
                        type=str, help='Input images')
    parser.add_argument('--output_dir', default='output/lol/', type=str, help='Directory for results')
    parser.add_argument('--high_dir', default='/home/hzq/Code/LLIE/datasets/LOL/eval15/high',
                        type=str, help='Directory for GT')
    parser.add_argument('--weights_1', type=str, help='weight of stage1', default="pretrained_model/lol/stage1.pth")
    parser.add_argument('--weights_2', type=str, help='weight of stage2', default="pretrained_model/lol/stage2.pth")
    parser.add_argument('--gpu', type=str, default="0", help='gpu id, -1 means do not use gpu')
    parser.add_argument('--model_type', type=int, default=0, help='1=MIANet, 2=WNENet, others=SWANet')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu != "-1" else ""
    device = torch.device(f"cuda:{args.gpu}" if args.gpu != "-1" else "cpu")
    test()

    if args.high_dir:
        IQA = IQAMetric(use_gpu=True, metric_type=["psnr", "ssim", "lpips"])
        IQA.test_dir(args.output_dir, args.high_dir)
