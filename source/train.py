import os
import argparse

# not sure if this will work in SM...
gpus = ','.join([str(i) for i in [0,1,2,3]])
gpus = '0'  # comment out to use 4 gpus. Switch instance type to ml.p3.8xlarge.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from MPRNet import MPRNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx


start_epoch = 1
mode = 'Deblurring'
session = 'MPRNet'

parser = argparse.ArgumentParser()
# SageMaker parameters, like the directories for training data and saving models; set automatically
# Do not need to change
parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

# Training Parameters, given
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 3000)')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default:  2e-4)')
parser.add_argument('--lr-min', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default:  1e-6)')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ps', type=int, default=256, metavar='PS',
                    help='patch size for training (defautlt: 256)')
parser.add_argument('--val-freq', type=int, default=20, metavar='VF',
                    help='validate after N epochs (default: 20)')
parser.add_argument('--resume', type=bool, default=False, metavar='R',
                    help='to resume training (default: False)')
args = parser.parse_args()
                    
                    
######### Set Seeds ###########
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
    
result_dir = args.model_dir
model_dir  = args.model_dir
                    
utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = args.data_dir
val_dir   = args.test_dir

######### Model ###########
model_restoration = MPRNet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-warmup_epochs, eta_min=args.lr_min)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if args.resume:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':args.ps})
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size':args.ps})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,args.epochs + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, args.epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored = model_restoration(input_)
 
        # Compute loss at each stage
        loss_char = np.sum([criterion_char(restored[j],target) for j in range(len(restored))])
        loss_edge = np.sum([criterion_edge(restored[j],target) for j in range(len(restored))])
        loss = (loss_char) + (0.05*loss_edge)
       
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

    #### Evaluation ####
    if epoch%args.val_freq == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]

            for res,tar in zip(restored,target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

