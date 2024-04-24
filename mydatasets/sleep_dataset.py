import torch
from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np

from tqdm import tqdm

import scipy.signal as sci_signal

# Butterworth低通滤波器
def butter_lowpass_filter(data,order,cutoff,fs):
    wn = 2*cutoff/fs
    b, a = sci_signal.butter(order, wn, 'lowpass', analog = False)
    output = sci_signal.filtfilt(b, a, data, axis=1)

    return output

class EdfDataset(Dataset):
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42, is_EDF39=False, choice_idx=[]):
        super(EdfDataset,self).__init__()
        # 加载数据
        if is_EDF39:
            files = [ item for item in glob.glob(os.path.join(data_path, "*.npz")) if int(item.split('/')[-1][3:5])<20 ]
        else:
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
                item_x = butter_lowpass_filter(item_x, 8, 30, 100)
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
    

from torchvision import transforms
from PIL import Image

Image_Processor_Contrastive = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.GaussianBlur(kernel_size=5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

Image_Processor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

class EdfAugFreqTimeDataset(Dataset):
    """
    EDFSleep频谱图数据集
    """
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], is_EDF39=False):
        super(EdfAugFreqTimeDataset,self).__init__()
        # 加载数据
        if is_EDF39:
            subjects = [ name for name in os.listdir(data_path) if int(name[3:5])<20 ]
        else:
            subjects = os.listdir(data_path)
        # 为保证signal和timefreq的数据在使用相同random_seed时，保持一致的数据样本
        subjects = sorted(subjects, key=lambda s: int(s[-4:]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(subjects))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        if len(choice_idx)!=0:
            print(f"choice_idx: ", choice_idx)
            nfiles = [f for i,f in enumerate(subjects) if i in choice_idx]
        else:
            train_idx = shuffle_idx[:int(len(subjects)*0.8)]
            test_idx = shuffle_idx[int(len(subjects)*0.8):]
            print(f"train_idx: {train_idx} \n \
                  test_idx: {test_idx}", flush=True)
            if is_train:
                nfiles = [f for i,f in enumerate(subjects) if i in train_idx]
            else:
                nfiles = [f for i,f in enumerate(subjects) if i in test_idx]
        
        print(nfiles)
        
        # 读取数据并转换为tensor
        X_data = []
        y = []
        for fi in nfiles:
            subject_epochs = os.listdir(os.path.join(data_path, fi))
            # os.listdir返回的片段并不是有序的，需重新排序，用于seq_len的处理
            subject_epochs = sorted(subject_epochs, key=lambda fname: int(fname.split('_')[0]))
            subject_X = []
            subject_y = []
            for image_name in tqdm(subject_epochs):
                item_y = int(image_name.split('.')[0].split('_')[-1])
                # print(os.path.join(data_path, fi, image_name))
                image_x = Image.open(os.path.join(data_path, fi, image_name)).convert("RGB")
                item_x1 = Image_Processor(image_x)
                item_x2 = Image_Processor_Contrastive(image_x)
                # print(item_x1.shape, item_x2.shape)
                subject_X.append(item_x1.unsqueeze(0))
                subject_X.append(item_x2.unsqueeze(0))
                subject_y.append(item_y)
            
            subject_X = torch.vstack(subject_X)
            subject_y = torch.asarray(subject_y)
            X_data.append(subject_X.reshape(-1, 2, subject_X.size(1), subject_X.size(2), subject_X.size(3)))
            y.extend(subject_y)
            
        X_data = torch.vstack(X_data)
        y = torch.asarray(y)

        assert X_data.size(0) == y.size(0)

        print("the count of samples: ", y.size(0), flush=True)

        self.X_data = X_data
        self.y = torch.Tensor(y)

        print(np.unique(y, return_counts=True))

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

class EdfFreqTimeDataset(Dataset):
    """
    EDFSleep频谱图数据集
    """
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], is_EDF39=False, is_contrastive=True):
        super(EdfFreqTimeDataset,self).__init__()
        # 加载数据
        if is_EDF39:
            subjects = [ name for name in os.listdir(data_path) if int(name[3:5])<20 ]
        else:
            subjects = os.listdir(data_path)
        # 为保证signal和timefreq的数据在使用相同random_seed时，保持一致的数据样本
        subjects = sorted(subjects, key=lambda s: int(s[-4:]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(subjects))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        if len(choice_idx)!=0:
            print(f"choice_idx: ", choice_idx)
            nfiles = [f for i,f in enumerate(subjects) if i in choice_idx]
        else:
            train_idx = shuffle_idx[:int(len(subjects)*0.8)]
            test_idx = shuffle_idx[int(len(subjects)*0.8):]
            print(f"train_idx: {train_idx} \n \
                  test_idx: {test_idx}", flush=True)
            if is_train:
                nfiles = [f for i,f in enumerate(subjects) if i in train_idx]
            else:
                nfiles = [f for i,f in enumerate(subjects) if i in test_idx]
        
        print(nfiles)
        
        # 读取数据并转换为tensor
        X_data = []
        y = []
        for fi in nfiles:
            subject_epochs = os.listdir(os.path.join(data_path, fi))
            # os.listdir返回的片段并不是有序的，需重新排序，用于seq_len的处理
            subject_epochs = sorted(subject_epochs, key=lambda fname: int(fname.split('_')[0]))
            subject_X = []
            subject_y = []
            for image_name in tqdm(subject_epochs):
                item_y = int(image_name.split('.')[0].split('_')[-1])
                # print(os.path.join(data_path, fi, image_name))
                image_x = Image.open(os.path.join(data_path, fi, image_name)).convert("RGB")
                if is_contrastive:
                    item_x = Image_Processor_Contrastive(image_x)
                else:
                    item_x = Image_Processor(image_x)
                subject_X.append(item_x.unsqueeze(0))
                subject_y.append(item_y)
            
            cut_len = len(subject_X)//seq_len*seq_len
            subject_X = torch.vstack(subject_X[:cut_len])
            subject_y = torch.asarray(subject_y[:cut_len])
            X_data.append(subject_X.reshape(-1, seq_len, subject_X.size(1), subject_X.size(2), subject_X.size(3)))
            y.append(subject_y.reshape(-1, seq_len))
            
        X_data = torch.vstack(X_data)
        y = torch.vstack(y)

        assert X_data.size(0) == y.size(0)

        print("the count of samples: ", y.size(0), flush=True)

        self.X_data = X_data
        self.y = torch.Tensor(y)

        print(np.unique(y, return_counts=True))

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)
    

class EdfSTFTDataset(Dataset):
    """
    STFT数据集
    """
    def __init__(self, data_path, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], is_EDF39=False):
        super(EdfSTFTDataset,self).__init__()
        # 加载数据
        if is_EDF39:
            files = [ item for item in glob.glob(os.path.join(data_path, "*.npz")) if int(item.split('/')[-1][3:5])<20 ]
        else:
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


class EdfCoTrainDataset(Dataset):
    """
    EDFCoTrainDataset
    contain signals and timefreq images
    """
    def __init__(self, data_path1, data_path2, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], is_EDF39=False):
        super(EdfCoTrainDataset,self).__init__()
        # 加载时频图数据
        if is_EDF39:
            timefreq_subjects = [ name for name in os.listdir(data_path2) if int(name[3:5])<20 ]
        else:
            timefreq_subjects = os.listdir(data_path2)
        timefreq_subjects = sorted(timefreq_subjects, key=lambda s: int(s[-4:]))
        # 随机抽选训练集和测试集样本
        np.random.seed(shuffle_seed)
        shuffle_idx = np.arange(len(timefreq_subjects))
        np.random.shuffle(shuffle_idx)
        # 划分训练集和测试集
        if len(choice_idx)!=0:
            print(f"choice_idx: ", choice_idx)
            nfiles = [f for i,f in enumerate(timefreq_subjects) if i in choice_idx]
        else:
            train_idx = shuffle_idx[:int(len(timefreq_subjects)*0.8)]
            test_idx = shuffle_idx[int(len(timefreq_subjects)*0.8):]
            print(f"train_idx: {train_idx} \n \
                  test_idx: {test_idx}", flush=True)
            if is_train:
                nfiles = [f for i,f in enumerate(timefreq_subjects) if i in train_idx]
            else:
                nfiles = [f for i,f in enumerate(timefreq_subjects) if i in test_idx]
        # 读取数据并转换为tensor
        X_data1, X_data2 = [], []
        y1 ,y2 = [], []
        for fi in nfiles:
            subject_epochs = os.listdir(os.path.join(data_path2, fi))
            # os.listdir返回的片段并不是有序的，需重新排序，用于seq_len的处理
            subject_epochs = sorted(subject_epochs, key=lambda fname: int(fname.split('_')[0]))
            subject_X = []
            subject_y = []
            for image_name in tqdm(subject_epochs):
                item_y = int(image_name.split('.')[0].split('_')[-1])
                # print(os.path.join(data_path, fi, image_name))
                image_x = Image.open(os.path.join(data_path2, fi, image_name)).convert("RGB")
                item_x = Image_Processor_Contrastive(image_x)
                # item_x = Image_Processor(image_x)
                subject_X.append(item_x.unsqueeze(0))
                subject_y.append(item_y)
            
            cut_len = len(subject_X)//seq_len*seq_len
            subject_X = torch.vstack(subject_X[:cut_len])
            subject_X = subject_X.reshape(-1, seq_len, subject_X.size(1), subject_X.size(2), subject_X.size(3))
            subject_y = torch.asarray(subject_y[:cut_len]).reshape(-1, seq_len)
            
            # signal process
            signal_X = []
            signal_y = []
            signal_data = np.load(os.path.join(data_path1, f"{fi}.npz"))
            for seq_idx in tqdm(range(len(signal_data['x'])//seq_len)):
                signal_item_x = signal_data['x'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                signal_item_x = butter_lowpass_filter(signal_item_x, 8, 30, 100)
                siganl_item_y = signal_data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
                signal_item_x = torch.from_numpy(signal_item_x.copy())
                signal_item_y = torch.from_numpy(siganl_item_y)
                assert signal_item_x.shape[0] == signal_item_y.shape[0]

                signal_X.append(signal_item_x.unsqueeze(0))
                signal_y.append(signal_item_y.unsqueeze(0))
            
            signal_X = torch.vstack(signal_X)
            signal_y = torch.vstack(signal_y)
            
            # print(fi)
            # print(subject_y)
            # print(signal_y)
            assert torch.all(subject_y==signal_y)

            X_data1.append(signal_X)
            y1.append(signal_y)
            X_data2.append(subject_X)
            y2.append(subject_y)
            
        X_data1 = torch.vstack(X_data1)
        y1 = torch.vstack(y1)
        X_data2 = torch.vstack(X_data2)
        y2 = torch.vstack(y2)

        assert X_data1.size(0) == y1.size(0) == X_data2.size(0) == y2.size(0)

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
    


class EdfCoTrainFreqDataset(Dataset):
    """
    EDFCoTrainFreqDataset
    contain signals and timefreq images
    """
    def __init__(self, data_path1, data_path2, seq_len, is_train=True, shuffle_seed=42, choice_idx=[], is_EDF39=False):
        super(EdfCoTrainFreqDataset,self).__init__()
        # 加载Npz数据
        if is_EDF39:
            files = [ item for item in glob.glob(os.path.join(data_path1, "*.npz")) if int(item.split('/')[-1][3:5])<20 ]
        else:
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
            data2 = np.load(fi.replace('/npz','/freq'))
            for seq_idx in tqdm(range(len(data1['x'])//seq_len)):
                # npz
                item_x1 = data1['x'][seq_idx*seq_len:(seq_idx+1)*seq_len,...]
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
    data_path1 = "../sleep-cassette/npz"
    data_path2 = "../../sleep-cassette/timefreq"
    data_path3 = "../sleep-cassette/freq"

    # dataset = EdfDataset(data_path1, seq_len=16, is_train=False, is_EDF39=True, shuffle_seed=42)
    # dataset = EdfAugFreqTimeDataset(data_path2, seq_len=16, is_train=True, is_EDF39=True)
    # dataset = EdfCoTrainDataset(data_path1, data_path2, seq_len=16, is_EDF39=True)
    # dataset = EdfSTFTDataset(data_path=data_path3, seq_len=16, is_train=False, is_EDF39=True, shuffle_seed=42)
    dataset = EdfCoTrainFreqDataset(data_path1=data_path1, data_path2=data_path3, seq_len=16, is_train=False, shuffle_seed=42, is_EDF39=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for X, y in dataloader:
        assert torch.all(y[0]==y[1])
        print(X[0].shape, y[0].shape)
        # print(X.shape, y.shape)