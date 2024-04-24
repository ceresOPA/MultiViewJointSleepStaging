import torch
import torch.nn as nn
from collections import OrderedDict

class TinyFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=100, seq_len=16, hidden_size=128):
        super(TinyFeatureExtractor, self).__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        first_filter_size = int(sampling_rate / 2.0)  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(sampling_rate / 16.0)  # todo 与论文不同，论文给出的stride是100/4=25
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, dropout=0.5)
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1)
        self.rnn = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
        # self.fc = nn.Linear(hidden_size, 5)


    def forward(self, x):
        x = self.cnn(x)
        # input of LSTM must be shape(seq_len, batch, input_size)
        x = x.view(-1, self.seq_len, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        x, _ = self.rnn(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.rnn_dropout(x)
        # x = self.fc(x)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.padding_edf = {  
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        first_filter_size = int(100 / 2.0)  
        first_filter_stride = int(100 / 16.0) 
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x
    

class STFTNet(nn.Module):
    def __init__(self):
        super(STFTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), padding='same', stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding='same', stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16)
        )
        self.max_pooling1_dropout = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same', stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same', stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32)
        )
        self.max_pooling2_flatten = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5440, 2048),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pooling1_dropout(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pooling2_flatten(x)
        x = self.linear(x)

        return x
    

from torchsummaryX import summary

if __name__ == "__main__":
    model = STFTNet()
    x = torch.randn(3, 2, 41, 68)
    summary(model, x)