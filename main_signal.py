import os
import sys
import argparse
import time, re
import builtins
import numpy as np
import random 
import pickle 
import socket 
import math 
from tqdm import tqdm 
# from backbone.select_backbone import select_backbone

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
# from torchvision import transforms
# import torchvision.utils as vutils

# import utils.augmentation as A
# import utils.transforms as T
import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils.utils import AverageMeter, ProgressMeter, calc_topk_accuracy, save_checkpoint, lowpass_shift
from mydatasets.get_dataset import get_dataset

import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
# from model.pretrain import InfoNCE, UberNCE
# from dataset.lmdb_dataset import *
from network.contrastive_network import ContrastiveSignalNet
from torch.utils.data import random_split, DataLoader



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='tinyfeature', type=str) # r18-all
    parser.add_argument('--model', default='infonce', type=str)
    parser.add_argument('--dataset', default='edf39_signal', type=str)
    parser.add_argument('--data_path', default='../sleep-cassette/npz', type=str)
    parser.add_argument('--seq_len', default=16, type=int, help='epochs of signals sequence')
    # parser.add_argument('--num_seq', default=2, type=int, help='number of video blocks')
    # parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--prefix', default='pretrain', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)


    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    

    parser.add_argument('--print', default=True, type=bool)

    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0
    args.gpu = gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### model ###
    # model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    model = ContrastiveSignalNet(base_encoder=args.net, dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t)
    model.to(device)

    args.img_path, args.model_path, args.exp_path = set_path(args)

    ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})

    # print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.requires_grad)
    # print('=================================\n')

    # 这里把梯度false的加入在了params，不知道有没有影响
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().to(device)
    args.iteration = 1

    # 启用 cuDNN 自动寻找最适合当前硬件的卷积实现的一个设置
    torch.backends.cudnn.benchmark = True

    ### data ### 
    dataset = get_dataset(data_name=args.dataset, data_path=args.data_path, seq_len=args.seq_len, is_train=True, is_contrastive=True, random_shuffle=args.seed)
    real_train_size = int(len(dataset)*0.8)
    val_size = len(dataset) - real_train_size
    real_train_data, _ = random_split(dataset, [real_train_size, val_size], generator=torch.Generator().manual_seed(42))
    dataloader = DataLoader(real_train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    ### restart training ### 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            model.load_state_dict(state_dict)
            try: optimizer.load_state_dict(checkpoint['optimizer'])
            except: print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
    
    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
                
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
            model.load_state_dict(state_dict)
        else:
            print("=> no checkpoint found at '{}', use random init".format(args.pretrain))
    
    else:
        print("=> train from scratch")

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.train_plotter = TB.PlotterThread(writer_train)

    ### main loop ###    
    for epoch in range(args.start_epoch, args.epochs):
        # np.random.seed(epoch)
        # random.seed(epoch)
 
        adjust_learning_rate(optimizer, epoch, args)

        _, train_acc = train_one_epoch(dataloader, model, criterion, optimizer, epoch, device, args)
        

        # save model
        if epoch>10 and ((epoch % args.save_freq == 0) or (epoch == args.epochs - 1)): 

            is_best = train_acc > best_acc
            best_acc = max(train_acc, best_acc)
            state_dict = model.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best,
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch))
    
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))

    model.train() 

    tic = time.time()
    end = time.time()

    for idx, (input_seq, label) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
        data_time.update(time.time() - end)
        # input_seq = input_seq.reshape(-1, input_seq.size(2))
        # aug_seq = lowpass_shift(input_seq.numpy(), cutoff_freq=0.3, shift_value=random.randint(10, 30))

        #按一定概率调整匹配的正样本（在同一seq内）
        if random.random()<0.5:
            input_seq = input_seq[:, torch.randperm(input_seq.size(1)), ...]

        if args.dataset.split('_')[-1] == 'timefreq':
            input_seq = input_seq.reshape(-1, 2, input_seq.size(2), input_seq.size(3), input_seq.size(4))
        elif args.dataset.split('_')[-1] == 'stft':
            input_seq = input_seq.reshape(-1, 2, input_seq.size(2), input_seq.size(3), input_seq.size(4))
        elif args.dataset == 'isruc':
            input_seq = input_seq.reshape(-1, 2, 1, input_seq.size(2))
        else:
            input_seq = input_seq.reshape(-1, 2, 1, input_seq.size(2))

        x1 = input_seq[:, 0, ...].to(device, dtype=torch.float32)
        x2 = input_seq[:, 1, ...].to(device, dtype=torch.float32)

        B = input_seq.size(0)

        # infonce 'target' is the index of self
        output, target = model(x1, x2)
        loss = criterion(output, target)
        top1, top5 = calc_topk_accuracy(output, target, (1,5))

        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.display(idx)

        if idx % args.print_freq == 0:
            if args.print:
                args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
                args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
        
        args.iteration += 1

    print('Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))
    
    if args.print:
        args.train_plotter.add_data('global/loss', losses.avg, epoch)
        args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
        args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)

    return losses.avg, top1_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_path(args):
    print(args)
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-{args.prefix}/{args.name_prefix}{args.model}_k{args.moco_k}_{args.dataset}-dim{args.moco_dim}_{args.net}_\
bs{args.batch_size}_seq{args.seq_len}_lr{args.lr}{0}'.format(
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    return img_path, model_path, exp_path


if __name__ == '__main__':
    '''
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank
       
    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    '''
    args = parse_args()
    main(args)