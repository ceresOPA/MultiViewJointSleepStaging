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

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils.utils import AverageMeter, ProgressMeter, calc_topk_accuracy, save_checkpoint, calc_mask_accuracy
from mydatasets.get_dataset import get_co_dataset
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from cotrain_network import CoTrainNet, PureCoTrainNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net1', default='tinysleepnet_cnn', type=str)
    parser.add_argument('--net2', default='resnet18', type=str)
    parser.add_argument('--model', default='coclr', type=str)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dataset', default='ucf101-2stream-2clip', type=str)
    parser.add_argument('--data_path1', default='', type=str)
    parser.add_argument('--data_path2', default='', type=str)
    parser.add_argument('--seq_len', default=16, type=int, help='number of frames in each video block')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default=['random', 'random'], nargs=2, type=str, help='path of pretrained model: rgb, flow')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--prefix', default='copretrain', type=str)
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
    parser.add_argument('--cos', action='store_true', 
                        help='use cosine lr schedule')
    
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

    args.gpu == 0

    ### model ###
    print("=> creating {} model with '{} {}' backbone".format(args.model, args.net1, args.net2))
    if args.model == 'pure_coclr':
        model = PureCoTrainNet(args.net1, args.net2, args.seq_len, dim=args.moco_dim, K=args.moco_k, 
                           m=args.moco_m, T=args.moco_t, topk=args.topk, reverse=args.reverse)
        model.to(device)
        if args.reverse:
            print('[Warning] using Signal-Mining to help Freq')
        else:
            print('[Warning] using Freq-Mining to help Signal')
    elif args.model == 'coclr':
        model = CoTrainNet(args.net1, args.net2, args.seq_len, dim=args.moco_dim, K=args.moco_k, 
                           m=args.moco_m, T=args.moco_t, topk=args.topk, reverse=args.reverse)
        model.to(device)
        if args.reverse:
            print('[Warning] using Signal-Mining to help Freq')
        else:
            print('[Warning] using Freq-Mining to help Signal')
    else:
        raise NotImplementedError
    args.num_seq = 2
    print('Re-write num_seq to %d' % args.num_seq)
        
    args.img_path, args.model_path, args.exp_path = set_path(args)

    # print(model)

    ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###  
    dataset = get_co_dataset(data_name=args.dataset, data_path1=args.data_path1, data_path2=args.data_path2, seq_len=args.seq_len, random_shuffle=args.seed)
    real_train_size = int(len(dataset)*0.8)
    val_size = len(dataset) - real_train_size
    train_data, _ = random_split(dataset, [real_train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
    
    print('===================================')

    lr_scheduler = None

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
            except: print('[WARNING] Not loading optimizer states')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    elif args.pretrain != ['random', 'random']:
        # first path: weights to be trained
        # second path: weights as the oracle, not trained
        if os.path.isfile(args.pretrain[1]): # second network --> load as sampler
            checkpoint = torch.load(args.pretrain[1], map_location=torch.device('cpu'))
            second_dict = checkpoint['state_dict']
            new_dict = {}
            for k,v in second_dict.items(): # only take the encoder_q
                if 'encoder_q.' in k:
                    k = k.replace('encoder_q.', 'sampler.0.')
                    new_dict[k] = v
                elif 'projection_layer' in k:
                    k = k.replace('projection_layer.', 'sampler.1.')
                    new_dict[k] = v
            second_dict = new_dict

            new_dict = {} # remove queue, queue_ptr
            for k, v in second_dict.items():
                if 'queue' not in k:
                    new_dict[k] = v 
            second_dict = new_dict
            print("=> Use Oracle checkpoint '{}' (epoch {})".format(args.pretrain[1], checkpoint['epoch']))
        else:
            print("=> NO Oracle checkpoint found at '{}', use random init".format(args.pretrain[1]))
            second_dict = {}

        if os.path.isfile(args.pretrain[0]): # first network --> load both encoder q & k
            checkpoint = torch.load(args.pretrain[0], map_location=torch.device('cpu'))
            first_dict = checkpoint['state_dict']

            new_dict = {} # remove queue, queue_ptr
            for k, v in first_dict.items():
                if 'queue' not in k:
                    new_dict[k] = v 
            first_dict = new_dict

            # update both q and k with q
            new_dict = {}
            for k,v in first_dict.items(): # only take the encoder_q
                if 'encoder_q.' in k:
                    new_dict[k] = v
                    k = k.replace('encoder_q.', 'encoder_k.')
                    new_dict[k] = v
                elif 'projection_layer' in k:
                    new_dict[k] = v
            first_dict = new_dict
            
            print("=> Use Training checkpoint '{}' (epoch {})".format(args.pretrain[0], checkpoint['epoch']))
        else:
            print("=> NO Training checkpoint found at '{}', use random init".format(args.pretrain[0]))
            first_dict = {}

        state_dict = {**first_dict, **second_dict}
        try:
            del state_dict['queue_label'] # always re-fill the queue
        except:
            pass 

        for k,v in state_dict.items():
            print("kk", k)

        msg = model.load_state_dict(state_dict, strict=False)
        assert len(msg) == 2

    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.train_plotter = TB.PlotterThread(writer_train)
    
    ### main loop ###    
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        _, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        # 队满之后再保存模型
        if epoch>10 and model.queue_is_full and ((epoch % args.save_freq == 0) or (epoch == args.epochs - 1)):
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
    top1_self_meter = AverageMeter('Self-acc@1', ':.4f')
    top5_self_meter = AverageMeter('Self-acc@5', ':.4f')
    sacc_meter = AverageMeter('Sampling-Acc@%d' % args.topk, ':.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter, top1_self_meter, top5_self_meter, sacc_meter],
        prefix='Epoch:[{}]'.format(epoch))

    model.train() 
    model.sampler.eval() # the sampler is always fixed

    tic = time.time()
    end = time.time()

    for idx, (input_seq, y) in enumerate(data_loader):
        data_time.update(time.time() - end)
        
        # 保证两个不同的view的样本是一致的
        assert torch.all(y[0]==y[1])

        # X_input = input_seq[0].reshape(-1, 2, 1, input_seq[0].size(2)).to(device, dtype=torch.float32)
        # F_input = input_seq[1].reshape(-1, 2, input_seq[1].size(2), input_seq[1].size(3), input_seq[1].size(4)).to(device, dtype=torch.float32)

        X_input = input_seq[0].reshape(-1, 1, input_seq[0].size(2)).to(device, dtype=torch.float32)
        F_input = input_seq[1].reshape(-1, input_seq[1].size(2), input_seq[1].size(3), input_seq[1].size(4)).to(device, dtype=torch.float32)
        B = X_input.size(0)
        output, mask = model(X_input, F_input)
        mask_sum = mask.sum(1)

        mask_clone = mask.clone()
        mask_clone[mask_sum!=1, 0] = 0 # mask out self-similarity
        loss = criterion(output, mask_clone)

        """
        if random.random() < 0.5:
            # because model has been pretrained with infoNCE, 
            # in this stage, self-similarity is already very high,
            # randomly mask out the self-similarity for optimization efficiency,
            mask_clone = mask.clone()
            mask_clone[mask_sum!=1, 0] = 0 # mask out self-similarity
            loss = criterion(output, mask_clone)
        else:
            loss = criterion(output, mask)
        """

        top1, top5 = calc_mask_accuracy(output, mask, (1,5))
        top1_self, top5_self = calc_topk_accuracy(output, torch.zeros(B, dtype=torch.long).cuda(), (1,5))

        del output

        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        top1_self_meter.update(top1_self.item(), B)
        top5_self_meter.update(top5_self.item(), B)
        
        # 队列满后再进行梯度更新，没满之前队列是随机初始化的
        if model.queue_is_full:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        del loss 
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5', top5_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/self-top1', top1_self_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/self-top5', top5_self_meter.local_avg, args.iteration)

        args.iteration += 1
        
    print('Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))
    
    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)
    args.train_plotter.add_data('global/self-top1', top1_self_meter.avg, epoch)
    args.train_plotter.add_data('global/self-top5', top5_self_meter.avg, epoch)

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
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-{args.prefix}/{args.name_prefix}{args.model}-top{args.topk}{0}_k{args.moco_k}_{args.dataset}_{args.net1}_{args.net2}\
bs{args.batch_size}_lr{args.lr}_len{args.seq_len}_ds{args.ds}'.format(
                    '-R' if args.reverse else '', \
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
    Three ways to run:
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch\
       --nproc_per_node=2 main_coclr.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank
       
    2. CUDA_VISIBLE_DEVICES=0,1 python main_coclr.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. SLURM scheduler
    '''
    args = parse_args()
    main(args)