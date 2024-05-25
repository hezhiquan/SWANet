import os
import torch
import yaml
import numpy as np
import random

# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
from utils import network_parameters, losses, torchSSIM, torchPSNR, AverageMeter
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from transform.data_RGB import get_validation_data, get_training_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import utils.losses
import utils

from MIANet import MIANet

# Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
print(yaml.dump(opt, default_flow_style=False))
Train = opt['TRAINING']
OPT = opt['OPTIM']
# GPU
device = torch.device("cuda:" + str(opt["GPU"]) if int(opt["GPU"]) >= 0 else "cpu")
# Build Model
print('==> Build the model')

model_restored = MIANet(in_chn=3, wf=Train['WF'], depth=4)
p_number = network_parameters(model_restored)
model_restored.to(device)

# Training model path direction
mode = opt['MODEL']['MODE']
print("==> mode: ", mode)

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
test_dir = Train['TEST_DIR']

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.AdamW(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

# Scheduler (Strategy)
warmup_epochs = 3

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

optimizer.zero_grad()
optimizer.step()
scheduler.step()

# Resume (Continue training by a pretrained model)
if Train['RESUME']:
    # path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    path_chk_rest = Train["checkpoint_path"]
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:{} ;  start epoch:{}".format(new_lr, start_epoch))
    print('------------------------------------------------------------------')

# Loss
Charloss = losses.CharbonnierLoss()

# DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS'], 'factor': Train['factor']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(test_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    WF:                 {str(Train['WF'])}
    factor:             {str(Train['factor'])}
    loss factor         {str(Train['loss_factor'])}
    ''')
print("==> model")
print(model_restored)
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

# Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)

train_writer = SummaryWriter(log_dir=log_dir + "/train", filename_suffix=f'_{mode}')
val_writer = SummaryWriter(log_dir=log_dir + "/val", filename_suffix=f'_{mode}')

ssim_loss = losses.SSIMLoss()


def validate_test(epoch):
    global best_psnr, best_epoch_psnr, best_ssim, best_epoch_ssim
    model_restored.eval()
    # validate
    psnr_val_rgb = []
    ssim_val_rgb = []
    val_loss = 0
    loss1_total = []
    loss2_total = []
    for ii, data_val in enumerate(val_loader, 0):
        target = data_val[0].to(device)
        input_ = data_val[1].to(device)
        with torch.no_grad():
            restored = model_restored(input_)

            # Compute loss
            loss1 = Charloss(restored, target)

            ssim_loss_mean = torch.mean(ssim_loss(restored, target))
            loss2 = float(Train['loss_factor']) * ssim_loss_mean
            loss1_total.append(loss1.item())
            loss2_total.append(loss2.item())
            loss = loss1 + loss2

            val_loss += loss.item()

        h, w = data_val[3][0].item(), data_val[3][1].item()
        restored = restored[:, :, :h, :w]
        target = target[:, :, :h, :w]
        for res, tar in zip(restored, target):
            psnr_val_rgb.append(utils.torchPSNR(res, tar))
            ssim_val_rgb.append(utils.torchSSIM(restored, target))

    print("[char loss : {}, ssim loss : {}]".format(utils.get_list_average(loss1_total),
                                                    utils.get_list_average(loss2_total)))

    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

    val_loss_show = val_loss / len(val_dataset)

    # Save the best PSNR model of validation
    if psnr_val_rgb > best_psnr:
        best_psnr = psnr_val_rgb
        best_epoch_psnr = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_bestPSNR.pth"))
    print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f Loss %.4f, val_loader len : %d]" % (
        epoch, psnr_val_rgb, best_epoch_psnr, best_psnr, val_loss_show, len(val_dataset)))
    # Save the best SSIM model of validation
    if ssim_val_rgb > best_ssim:
        best_ssim = ssim_val_rgb
        best_epoch_ssim = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_bestSSIM.pth"))

    print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
        epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

    val_writer.add_scalar('metric/PSNR', psnr_val_rgb, epoch)
    val_writer.add_scalar('metric/SSIM', ssim_val_rgb, epoch)
    val_writer.add_scalar("param/loss", val_loss_show, epoch)


dataset_size = len(train_dataset)

psnr_meter = AverageMeter()
ssim_meter = AverageMeter()
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    # train
    loss1_total = []
    loss2_total = []
    psnr_meter.reset()
    ssim_meter.reset()
    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        B = data[0].shape[0]


        for param in model_restored.parameters():
            param.grad = None
        target = data[0].to(device)
        input_ = data[1].to(device)
        restored = model_restored(input_)

        # Compute loss
        loss1 = Charloss(restored, target)
        ssim_loss_mean = torch.mean(ssim_loss(restored, target))
        loss2 = float(Train['loss_factor']) * ssim_loss_mean
        loss1_total.append(loss1.item() * B)
        loss2_total.append(loss2.item() * B)

        loss = loss1 + loss2

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * B

        with torch.no_grad():
            for a, b in zip(restored, target):
                a1 = a.unsqueeze(0)
                b1 = b.unsqueeze(0)
                psnr_meter.update(torchPSNR(a1, b1))
                ssim_meter.update(torchSSIM(a1, b1))
    #
    train_writer.add_scalar('train/PSNR', psnr_meter.avg, epoch)
    train_writer.add_scalar('train/SSIM', ssim_meter.avg, epoch)


    print("[char loss : {}, ssim loss : {}]".format(sum(loss1_total) / dataset_size,
                                                    sum(loss2_total) / dataset_size))

    # Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        validate_test(epoch)

    scheduler.step()
    # loss of each epoch
    epoch_loss_show = epoch_loss / dataset_size
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}\ttrain loader len: {}".format(epoch,
                                                                                                    time.time() - epoch_start_time,
                                                                                                    epoch_loss_show,
                                                                                                    scheduler.get_lr()[
                                                                                                        0],
                                                                                                    len(train_dataset)))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    train_writer.add_scalar('param/loss', epoch_loss_show, epoch)
    train_writer.add_scalar('param/lr', scheduler.get_lr()[0], epoch)

val_writer.close()
train_writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
