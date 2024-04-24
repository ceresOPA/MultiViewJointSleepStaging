import torch
import torch.nn as nn
from collections import OrderedDict
from utils.select_backbone import select_backbone
from network.backbone import TinyFeatureExtractor

class TinySleepNet(nn.Module):
    def __init__(self, sampling_rate=100, seq_len=16, hidden_size=128, num_classes=5,
                 use_final_bn=False, 
                 use_l2_norm=False):
        super(TinySleepNet, self).__init__()
        self.use_final_bn = use_final_bn
        self.use_l2_norm = use_l2_norm
        self.backbone = TinyFeatureExtractor(sampling_rate, seq_len, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(hidden_size)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
    
    def forward(self, x):
        x = self.backbone(x)
        if self.use_l2_norm:
            x = nn.functional.normalize(x, dim=1)
        
        if self.use_final_bn:
            logit = self.fc(self.final_bn(x))
        else:
            logit = self.fc(x)

        return logit
    
    

class ContrativeSignalClassify(nn.Module):
    def __init__(self, backbone, seq_len, dropout=0.5, use_dropout=False, use_final_bn=False, use_l2_norm=False):
        super(ContrativeSignalClassify, self).__init__()
        
        self.use_final_bn = use_final_bn
        self.use_l2_norm = use_l2_norm

        self.backbone = select_backbone(network=backbone, seq_len=seq_len)

        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()

        if use_dropout:
            self.final_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2048, 5))
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(2048, 5))
            
        self._initialize_weights(self.final_fc)

    def forward(self, x):
        x = self.backbone(x)
        if self.use_l2_norm:
            x = nn.functional.normalize(x, dim=1)

        if self.use_final_bn:
            logit = self.final_fc(self.final_bn(x))
        else:
            logit = self.final_fc(x)

        return logit
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


if __name__ == '__main__':
    from torchsummaryX import summary

    # model = TinySleepNet()
    # state = (torch.zeros(size=(1, 15, 128)),
    #          torch.zeros(size=(1, 15, 128)))

    # summary(model, torch.randn(size=(16*16, 1, 3000)))

    model = ContrativeSignalClassify(backbone='efficientnet_v2_s', seq_len=16)
    summary(model, x=torch.randn(16, 3, 224, 224))
