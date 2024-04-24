import torch
from network.classification_network import TinySleepNet
from network.classification_network import ContrativeSignalClassify

checkpoint = torch.load("/workspace/leyu/sleepstage/TimeSeriesCoCLR/log-pretrain/infonce_k2048_edf39_stft-dim128_stftnet_bs32_seq16_lr0.001/model/model_best.pth.tar",
                         map_location='cpu')

epoch = checkpoint['epoch']
state_dict = checkpoint['state_dict']

print("epoch", epoch)

# new_dict = {}
# for k,v in state_dict.items():
#     if 'encoder_q' in k:
#         k = k.replace('encoder_q.', 'backbone.')
#         new_dict[k] = v
# state_dict = new_dict

# for k,v in state_dict.items():
#     print(k)

# model = ContrativeSignalClassify(backbone="resnet18", seq_len=16)

# print("==="*8)

# for name, param in model.named_parameters():
#     print(name)

# model.load_state_dict(state_dict)