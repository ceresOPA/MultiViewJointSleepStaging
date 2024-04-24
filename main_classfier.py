import os
import sys
import argparse
import time
import re
import numpy as np
import random 
import pickle 
from tqdm import tqdm 
from PIL import Image
import json 
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F 


from utils.utils import AverageMeter, save_checkpoint, \
calc_topk_accuracy, Logger, ProgressMeter, lowpass_shift, compute_kl_loss

import utils.tensorboard_utils as TB

from mydatasets.get_dataset import get_dataset
from torch.utils.data import random_split, DataLoader
from network.classification_network import ContrativeSignalClassify

import sklearn.metrics as skmetrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='tinyfeature', type=str)
    parser.add_argument('--dataset', default='edf_sleep', type=str)
    parser.add_argument('--data_path', default='../sleep-cassette/npz', type=str)
    parser.add_argument('--seq_len', default=16, type=int, help='number of epochs in a sequence')
    parser.add_argument('--num_fc', default=1, type=int, help='number of fc')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.9, type=float, help='dropout')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--train_what', default='last', type=str)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    # parser.add_argument('--kl_coefficient', default=0.5, type=float)
    
    parser.add_argument('--prefix', default='linclr', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--retrieval', action='store_true', help='path of model to ucf retrieval')

    parser.add_argument('--dirname', default=None, type=str, help='dirname for feature')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--five_crop', action='store_true')
    parser.add_argument('--ten_crop', action='store_true')
    
    args = parser.parse_args()
    return args


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # num_gpu = len(str(args.gpu).split(','))
    # args.batch_size = num_gpu * args.batch_size
    # print('=> Effective BatchSize = %d' % args.batch_size)
    args.img_path, args.model_path, args.exp_path = set_path(args)
    
    # ### classifier model ###
    if args.train_what == 'last': # for linear probe
        args.final_bn = True 
        args.final_norm = True 
        args.use_dropout = False
    else: # for training the entire network
        args.final_bn = False 
        args.final_norm = False 
        args.use_dropout = True


    # model = TinySleepNet(sampling_rate=100, seq_len=args.seq_len, hidden_size=128, num_classes=5, 
    #                      use_final_bn=args.final_bn, use_l2_norm=args.final_norm)

    model = ContrativeSignalClassify(backbone=args.net, seq_len=args.seq_len, use_dropout=args.use_dropout, use_final_bn=args.final_bn, use_l2_norm=args.final_norm)
    model.to(device)

    ### optimizer ###
    if args.train_what == 'last':
        print('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else: 
                params.append({'params': param})
    
    elif args.train_what == 'ft':
        print('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr/10})
            else:
                params.append({'params': param})
    
    else: # train all
        params = []
        print('=> [optimizer] train all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})

    if args.train_what == 'last':
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    if args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        raise NotImplementedError
    
    ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 1, 1])).to(device)
    # ce_loss = FocalLoss(gamma=3)
    args.iteration = 1

    ### test: higher priority ### 
    test_data = get_dataset(data_name=args.dataset, data_path=args.data_path, seq_len=args.seq_len, is_train=False, is_contrastive=False, random_shuffle=args.seed)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            print("test epoch:", epoch)
            state_dict = checkpoint['state_dict']
            
            model.load_state_dict(state_dict)

        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0
            print("[Warning] if test random init weights, press c to continue")
            import ipdb; ipdb.set_trace()

        args.logger = Logger(path=os.path.dirname(args.test))
        args.logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
        
        test(test_loader, model, ce_loss, device)

        sys.exit(0)

    ### data ###
    all_data = get_dataset(data_name=args.dataset, data_path=args.data_path, seq_len=args.seq_len, is_train=True, random_shuffle=args.seed)
    # all_data = ISRUCDataset("../ISRUC-Sleep", is_train=True)

    # 先拆分训练集和验证集
    real_train_size = int(len(all_data)*0.8)
    val_size = len(all_data) - real_train_size
    real_train_data, val_dataset = random_split(all_data, [real_train_size, val_size], generator=torch.Generator().manual_seed(42))
    # 再选择要实际用于训练的样本量
    train_size = int(len(real_train_data)*1.0)
    print("train_size", train_size)
    remain_size = len(real_train_data) - train_size
    train_data, _ = random_split(real_train_data, [train_size,remain_size], generator=torch.Generator().manual_seed(42))
    #使用DataLoader加载数据集，转换为迭代器
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True) #用于训练中验证模型效果，进而可以动态调整超参数，控制训练
    print(f"train_dataset: {len(train_loader)}, validate_dataset: {len(val_loader)}")
    
    print('===================================')

    ### restart training ### 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            model.load_state_dict(state_dict)
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
    
    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k,v in state_dict.items():
                if 'encoder_q' in k:
                    k = k.replace('encoder_q.', 'backbone.')
                    new_dict[k] = v
            state_dict = new_dict

            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            assert len(msg) == 2
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.pretrain))
            raise NotImplementedError
    
    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # plot tools
    writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.val_plotter = TB.PlotterThread(writer_val)
    args.train_plotter = TB.PlotterThread(writer_train)

    args.logger = Logger(path=args.img_path)
    args.logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    
    # main loop 
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train_one_epoch(train_loader, model, ce_loss, optimizer, device, epoch, args)

        if epoch % args.eval_freq == 0:
            _, val_acc = validate(val_loader, model, ce_loss, device, epoch, args)

            # save check_point
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
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


def train_one_epoch(data_loader, model, criterion, optimizer, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))

    if args.train_what == 'last':
        model.eval() # totally freeze BN in backbone
    else:
        model.train()

    if args.final_bn:
        model.final_bn.train()

    end = time.time()
    tic = time.time()

    for idx, (input_seq, target) in enumerate(data_loader):
        data_time.update(time.time() - end)

        if args.dataset.split('_')[-1] == 'timefreq':
            input_seq = input_seq.reshape(-1, input_seq.size(2), input_seq.size(3), input_seq.size(4))
        elif args.dataset.split('_')[-1] == 'stft':
            input_seq = input_seq.reshape(-1, input_seq.size(2), input_seq.size(3), input_seq.size(4))
        else:
            input_seq = input_seq.reshape(-1, 1, input_seq.shape[-1])

        B = input_seq.size(0)

        input_seq = input_seq.to(device, non_blocking=True, dtype=torch.float32)
        target = target.reshape(-1,).to(device, non_blocking=True, dtype=torch.long)
        
        # R-Drop
        logit = model(input_seq)
        # loss = criterion(logit, target)

        logit2 = model(input_seq)

        ce_loss = 0.5*(criterion(logit, target)+criterion(logit2, target))
        kl_loss = compute_kl_loss(logit,logit2)

        loss = ce_loss + 2*kl_loss
        
        top1, top5 = calc_topk_accuracy(logit, target, (1,5))
        
        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)

            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5', top5_meter.local_avg, args.iteration)

        args.iteration += 1

    print('Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))

    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)
    
    args.logger.log('train Epoch: [{0}][{1}/{2}]\t'
                    'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))

    return losses.avg, top1_meter.avg


def validate(data_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            
            if args.dataset.split('_')[-1] == 'timefreq':
                input_seq = input_seq.reshape(-1, input_seq.size(2), input_seq.size(3), input_seq.size(4))
            elif args.dataset.split('_')[-1] == 'stft':
                input_seq = input_seq.reshape(-1, input_seq.size(2), input_seq.size(3), input_seq.size(4))
            else:
                input_seq = input_seq.reshape(-1, 1, input_seq.shape[-1])

            B = input_seq.size(0)

            input_seq = input_seq.to(device, non_blocking=True, dtype=torch.float32)
            target = target.reshape(-1,).to(device, non_blocking=True, dtype=torch.long)
            logit = model(input_seq)
            loss = criterion(logit, target)
            top1, top5 = calc_topk_accuracy(logit, target, (1,5))

            losses.update(loss.item(), B)
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()
            
    print('Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
          .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    args.val_plotter.add_data('global/loss', losses.avg, epoch)
    args.val_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.val_plotter.add_data('global/top5', top5_meter.avg, epoch)

    args.logger.log('val Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    return losses.avg, top1_meter.avg


def test(data_loader, model, criterion, device):
    test_loss = []
    test_trues = []
    test_preds = []
    model.eval()
    with torch.no_grad(): #不计算梯度，加快运算速度
        for batch_idx, (X, y) in enumerate(data_loader):
            if args.dataset.split('_')[-1] == 'timefreq':
                X = X.reshape(-1, X.size(2), X.size(3), X.size(4))
            elif args.dataset.split('_')[-1] == 'stft':
                X = X.reshape(-1, X.size(2), X.size(3), X.size(4))
            else:
                X = X.reshape(-1, 1, X.shape[-1])

            X = X.to(device, dtype=torch.float32)
            y = y.reshape(-1,).to(device, dtype=torch.long)
            pred = model(X)
            loss = criterion(pred, y)
            test_trues.append(y.cpu())
            test_preds.append(pred.argmax(dim=1).cpu())
            test_loss.append(loss.item())

    test_trues = np.hstack(test_trues)
    test_preds = np.hstack(test_preds)
    test_acc = skmetrics.accuracy_score(y_true=test_trues, y_pred=test_preds)
    test_f1_score = skmetrics.f1_score(test_trues, test_preds, average="macro")
    report = skmetrics.classification_report(y_true=test_trues, y_pred=test_preds)

    kappa = skmetrics.cohen_kappa_score(test_trues, test_preds)

    print(f"test_loss:{np.sum(test_loss):6.2f} || test_acc:{test_acc*100:5.2f}% || test_mf1:{test_f1_score:4.2f} || kappa:{kappa:4.2f}" , flush=True)
    print(report)


def summarize_probability(prob_dict, action_to_idx, title):
    acc = [AverageMeter(),AverageMeter()]
    stat = {}
    for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        try:
            action_name = vname.split('/')[-3]
        except:
            action_name = vname.split('/')[-2]
        target = action_to_idx(action_name)
        mean_prob = torch.stack(item['mean_prob'], 0).mean(0)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob, torch.LongTensor([target]).cuda(), (1,5))
        stat[vname] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

    print('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
          .format(acc=acc))

    with open(os.path.join(os.path.dirname(args.test), 
        '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def adjust_learning_rate(optimizer, epoch, args):
    '''Decay the learning rate based on schedule'''
    # stepwise lr schedule
    ratio = 0.1 if epoch in args.schedule else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * ratio


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-eval-{args.prefix}/{args.dataset}_{args.net}\
{1}_bs{args.batch_size}_lr{args.lr}_dp{args.dropout}_wd{args.wd}_seqlen{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{0}'.format(
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    '_SGD' if args.optim=='sgd' else '_Adam', \
                    args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path


if __name__ == '__main__':
    args = parse_args()
    main(args)