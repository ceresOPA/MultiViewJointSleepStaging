import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from mydatasets.get_dataset import get_dataset
import sklearn.metrics as skmetrics
import argparse
from network.classification_network import ContrativeSignalClassify


parser = argparse.ArgumentParser()
parser.add_argument('--net1', default='tinysleepnet_cnn', type=str)
parser.add_argument('--net2', default='stftnet', type=str)
parser.add_argument('--dataset1', default='edf39_signal', type=str)
parser.add_argument('--dataset2', default='edf39_stft', type=str)
parser.add_argument('--data_path1', default='../sleep-cassette/npz', type=str)
parser.add_argument('--data_path2', default='../sleep-cassette/freq', type=str)
parser.add_argument('--seq_len', default=16, type=int, help='number of epochs in a sequence')
parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
parser.add_argument('--test1', default='', type=str, help='path of model to load and pause')
parser.add_argument('--test2', default='', type=str, help='path of model to load and pause')
parser.add_argument('--train_what', default='last', type=str)

parser.add_argument('--seed', default=0, type=int)


args = parser.parse_args()

test_data1 = get_dataset(data_name=args.dataset1, data_path=args.data_path1, seq_len=args.seq_len, is_train=False, is_contrastive=False, random_shuffle=args.seed)
test_loader1 = DataLoader(test_data1, batch_size=args.batch_size, shuffle=False)

test_data2 = get_dataset(data_name=args.dataset2, data_path=args.data_path2, seq_len=args.seq_len, is_train=False, is_contrastive=False, random_shuffle=args.seed)
test_loader2 = DataLoader(test_data2, batch_size=args.batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(data_loader1, data_loader2, model1, model2, device):
    test_trues = []
    test_preds = []
    model1.eval()
    model2.eval()
    with torch.no_grad(): #不计算梯度，加快运算速度
        for batch_idx, ((X1, y1), (X2, y2)) in enumerate(zip(data_loader1, data_loader2)):
            print(X1.shape, X2.shape)
            assert torch.all(y1==y2)
            X1 = X1.reshape(-1, 1, X1.shape[-1]).to(device, dtype=torch.float32)
            X2 = X2.reshape(-1, X2.size(2), X2.size(3), X2.size(4)).to(device, dtype=torch.float32)
            y1 = y1.reshape(-1,).to(device, dtype=torch.long)
            y2 = y2.reshape(-1,).to(device, dtype=torch.long)
            pred1 = model1(X1)
            pred2 = model2(X2)
            print(pred1.shape, pred2.shape)
            pred = (pred1+pred2)/2.0
            test_trues.append(y1.cpu())
            test_preds.append(pred.argmax(dim=1).cpu())

    test_trues = np.hstack(test_trues)
    test_preds = np.hstack(test_preds)
    test_acc = skmetrics.accuracy_score(y_true=test_trues, y_pred=test_preds)
    test_f1_score = skmetrics.f1_score(test_trues, test_preds, average="macro")
    report = skmetrics.classification_report(y_true=test_trues, y_pred=test_preds)

    kappa = skmetrics.cohen_kappa_score(test_trues, test_preds)

    print(f"test_acc:{test_acc*100:5.2f}% || test_mf1:{test_f1_score:4.2f} || kappa:{kappa:4.2f}" , flush=True)
    print(report)


import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ### classifier model ###
if args.train_what == 'last': # for linear probe
    args.final_bn = True 
    args.final_norm = True 
    args.use_dropout = False
else: # for training the entire network
    args.final_bn = False 
    args.final_norm = False 
    args.use_dropout = True

model1 = ContrativeSignalClassify(backbone=args.net1, seq_len=args.seq_len, use_dropout=args.use_dropout, use_final_bn=args.final_bn, use_l2_norm=args.final_norm)
model1.to(device)

model2 = ContrativeSignalClassify(backbone=args.net2, seq_len=args.seq_len, use_dropout=args.use_dropout, use_final_bn=args.final_bn, use_l2_norm=args.final_norm)
model2.to(device)

import os

if os.path.isfile(args.test1):
    print("=> loading testing checkpoint '{}'".format(args.test1))
    checkpoint = torch.load(args.test1, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    print("test epoch:", epoch)
    state_dict = checkpoint['state_dict']
    
    model1.load_state_dict(state_dict)
else:
    print("[Warning] no checkpoint found at '{}'".format(args.test1))
    epoch = 0
    print("[Warning] if test random init weights, press c to continue")
    import ipdb; ipdb.set_trace()


if os.path.isfile(args.test2):
    print("=> loading testing checkpoint '{}'".format(args.test2))
    checkpoint = torch.load(args.test2, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    print("test epoch:", epoch)
    state_dict = checkpoint['state_dict']
    
    model2.load_state_dict(state_dict)
else:
    print("[Warning] no checkpoint found at '{}'".format(args.test2))
    epoch = 0
    print("[Warning] if test random init weights, press c to continue")
    import ipdb; ipdb.set_trace()



test(test_loader1, test_loader2, model1, model2, device)