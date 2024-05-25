import torch
import os
from collections import OrderedDict


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)


def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir, "model_epoch_{}_{}.pth".format(epoch, session))
    torch.save(state, model_out_path)


def load_checkpoint(model, weights, map_location=None):
    checkpoint = torch.load(weights, map_location=map_location)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_woE(model, weights, map_location=None):
    checkpoint = torch.load(weights, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr


def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params


def get_rgb_mask(high_img, stage1_img):
    """
    从ground truth、增强图像中算出二者之间的差距，用于当做先验
    :param high_img:
    :param stage1_img:
    :return:
    """
    high_mean = torch.mean(high_img, dim=[2, 3], keepdim=True)
    stage1_mean = torch.mean(stage1_img, dim=[2, 3], keepdim=True)
    mask = high_img / high_mean - stage1_img / stage1_mean
    torch.tanh_(mask)
    # mask_max, mask_min = mask.max(), mask.min()
    #
    # print("[mask range:[{} : {}], high mean:{}; stage1 mean:{}]".format(mask_max.cpu().detach().numpy(),
    #                                                                            mask_min.cpu().detach().numpy(),
    #                                                                            high_mean.cpu().detach().numpy(),
    #                                                                            stage1_mean.cpu().detach().numpy()))

    return mask
