from network.backbone import ConvNet, TinyFeatureExtractor, STFTNet
from torchvision.models import resnet18, resnet34, efficientnet_b0

def select_backbone(network, seq_len=16, feature_size=2048):
    if network == 'resnet18':
        return resnet18(num_classes=feature_size)
    elif network == 'resnet34':
        return resnet34(num_classes=feature_size)
    elif network == 'efficientnet_v2_s':
        return efficientnet_b0(num_classes=feature_size)
    elif network == 'tinysleepnet_cnn':
        return ConvNet()
    elif network == 'tinysleepnet_rnn':
        return TinyFeatureExtractor(seq_len=seq_len, hidden_size=feature_size)
    elif network == 'stftnet':
        return STFTNet()
    else:
        raise NotImplementedError

# from torchsummaryX import summary
# import torch

# model = efficientnet_v2_s()
# summary(model, x=torch.randn(32, 3, 224, 224))
