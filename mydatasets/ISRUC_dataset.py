import torch
from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np
import scipy.signal as sci_signal

from tqdm import tqdm

# Butterworth低通滤波器
def butter_lowpass_filter(data,order,cutoff,fs):
    wn = 2*cutoff/fs
    b, a = sci_signal.butter(order, wn, 'lowpass', analog = False)
    output = sci_signal.filtfilt(b, a, data, axis=1)

    return output

class ISRUCDataset(Dataset):
    def __init__(self, data_path, signal="F3_A2", is_train=True, shuffle_seed=43, choice_idx=[], seq_len=16):
        super(ISRUCDataset,self).__init__()
        # 两个不同的EEG的通道 "F3_A2", "C3_A2"
        signal_idx = 1 if signal == "F3_A2" else 0
        # 加载数据
        files = glob.glob(os.path.join(data_path, "*.npz"))
        files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[-2].split('_')[-1]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(files))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        if len(choice_idx)!=0:
            print(f"choice_idx: ", choice_idx)
            nfiles = [f for i,f in enumerate(files) if i in choice_idx]
        else:
            train_idx = shuffle_idx[:int(len(files)*0.8)]
            test_idx = shuffle_idx[int(len(files)*0.8):]
            print(f"train_idx: {train_idx} \n \
                  test_idx: {test_idx}", flush=True)
            if is_train:
                nfiles = [f for i,f in enumerate(files) if i in train_idx]
            else:
                nfiles = [f for i,f in enumerate(files) if i in test_idx]
        
        # 读取数据并转换为tensor
        X_data = []
        y = []
        for fi in nfiles:
            data = np.load(fi)
            data_X = data['x'][:, signal_idx, ::2]
            for seq_idx in tqdm(range(len(data_X)//seq_len)):
                item_x = data_X[seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x = butter_lowpass_filter(item_x, 8, 30, 100)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x = torch.from_numpy(item_x.copy())
                item_y = torch.from_numpy(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
        
        assert len(X_data) == len(y)

        print(X_data[0].shape, y[0].shape)
        print("the count of samples: ", len(y), flush=True)

        # X_data = (X_data - 0.008943356)/22.07068
        self.X_data = X_data
        self.y = y

        # print(np.unique(y, return_counts=True))

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)
    

class ISRUCSTFTDataset(Dataset):
    """
    STFT数据集
    """
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42, choice_idx=[]):
        super(ISRUCSTFTDataset,self).__init__()
        # 加载数据
        files = glob.glob(os.path.join(data_path, "*.npz"))
        files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[-2][-4:]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(files))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        train_idx = shuffle_idx[:int(len(files)*0.8)]
        test_idx = choice_idx if len(choice_idx)>0 else shuffle_idx[int(len(files)*0.8):]
        print(f"train_idx: {train_idx} \n \
            test_idx: {test_idx}", flush=True)
        if is_train:
            nfiles = [f for i,f in enumerate(files) if i in train_idx]
        else:
            nfiles = [f for i,f in enumerate(files) if i in test_idx]
        
        print([ n.split('/')[-1] for n in nfiles ])

        # 读取数据并转换为tensor
        X_data = []
        y = []
        for fi in nfiles:
            data = np.load(fi)
            for seq_idx in tqdm(range(len(data['x'])//seq_len)):
                item_x = data['x'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x = torch.from_numpy(item_x.copy())
                item_y = torch.from_numpy(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
        
        assert len(X_data) == len(y)
        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)



class ISRUCCoTrainFreqDataset(Dataset):
    """
    ISRUCCoTrainFreqDataset
    contain signals and timefreq spectra
    """
    def __init__(self, data_path1, data_path2, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], signal = "F3_A2"):
        super(ISRUCCoTrainFreqDataset, self).__init__()
        # 加载Npz数据
        signal_idx = 1 if signal == "F3_A2" else 0
        files = glob.glob(os.path.join(data_path1, "*.npz"))
        files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[-2][-4:]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(files))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        train_idx = shuffle_idx[:int(len(files)*0.8)]
        test_idx = choice_idx if len(choice_idx)>0 else shuffle_idx[int(len(files)*0.8):]
        print(f"train_idx: {train_idx} \n \
            test_idx: {test_idx}", flush=True)
        if is_train:
            nfiles = [f for i,f in enumerate(files) if i in train_idx]
        else:
            nfiles = [f for i,f in enumerate(files) if i in test_idx]
        
        print([ n.split('/')[-1] for n in nfiles ])
        # 读取数据并转换为tensor
        X_data1, X_data2 = [], []
        y1 ,y2 = [], []
        for fi in nfiles:
            data1 = np.load(fi)
            data1_X = data1['x'][:, signal_idx, ::2]
            data2 = np.load(fi.replace('/npz','/freq'))
            for seq_idx in tqdm(range(len(data1_X)//seq_len)):
                # npz
                item_x1 = data1_X[seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x1 = butter_lowpass_filter(item_x1, 8, 30, 100)
                item_y1 = data1['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x1 = torch.from_numpy(item_x1.copy())
                item_y1 = torch.from_numpy(item_y1)
                assert item_x1.shape[0] == item_y1.shape[0]
                X_data1.append(item_x1.unsqueeze(0))
                y1.append(item_y1.unsqueeze(0))
                # freq
                item_x2 = data2['x'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_y2 = data2['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                item_x2 = torch.from_numpy(item_x2.copy())
                item_y2 = torch.from_numpy(item_y2)
                assert item_x2.shape[0] == item_y2.shape[0]
                
                assert torch.all(item_y1==item_y2)
                X_data2.append(item_x2.unsqueeze(0))
                y2.append(item_y2.unsqueeze(0))

        X_data1 = torch.vstack(X_data1)
        y1 = torch.vstack(y1)
        X_data2 = torch.vstack(X_data2)
        y2 = torch.vstack(y2)

        assert X_data1.size(0) == y1.size(0) == X_data2.size(0) == y2.size(0), f"{X_data1.shape}, {y1.shape}, {X_data2.shape}, {y2.shape}"

        print("the count of samples: ", y1.size(0), flush=True)

        self.X_data1 = X_data1
        self.y1 = y1
        self.X_data2 = X_data2
        self.y2 = y2

        print(np.unique(y1, return_counts=True))

    def __getitem__(self, idx):
        return (self.X_data1[idx], self.X_data2[idx]), (self.y1[idx], self.y2[idx])

    def __len__(self):
        return len(self.y1)


if __name__ == "__main__":
    data_path = "../ISRUC-Sleep/npz"

    edf_dataset = ISRUCDataset(data_path, is_train=True)
    edf_dataloader = DataLoader(edf_dataset, batch_size=16, shuffle=False)

    count = 0
    for X, y in edf_dataloader:
        print(X.shape, y.shape)
        count += X.shape[0]
        
        # print(X.shape, y.shape)
        # print(y)
    print("count: ", count)