import torch
import torch.nn as nn
from utils.select_backbone import select_backbone

class ContrastiveSignalNet(nn.Module):
    """
    modified from MoCo
    moco code: https://github.com/facebookresearch/moco
    """

    def __init__(self, base_encoder, seq_len=16, prev_dim=2048, dim=128, K=2048, m=0.999, T=0.07, mlp=False):
        """
        seq_len: epochs of signals in a sample (default: 16)
        prev_dim: the last dim size of outputs in base_encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ContrastiveSignalNet, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)

        self.encoder_q = select_backbone(network=base_encoder, seq_len=seq_len, feature_size=prev_dim)
        self.encoder_k = select_backbone(network=base_encoder, seq_len=seq_len, feature_size=prev_dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.projection_layer = nn.Linear(prev_dim, dim)

        # create the queue
        # self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query signals
            im_k: a batch of key signals
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.projection_layer(q)
        # 对第1维进行标准化，即对特定维的元素逐个除以L2范数
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.projection_layer(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # 因为每一个样本对应一个正样本，所以直接是对应位相乘
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()]) # 多个负样本，进行矩阵相乘，初始的为随机初始化的负样本
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        # T越大会对负样本的区分度降低，分布越平滑，越小，则分布越尖锐，区分度加大
        logits /= self.T

        # labels: positive key indicators
        # logits的索引为0的概率就是正样本的预测概率
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SignalFeaExtractor(nn.Module):
    def __init__(self, num_classes=128):
        super(SignalFeaExtractor, self).__init__()
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ContrastiveSignalNet(base_encoder='tinysleepnet_cnn')

    # for name, param in resnet18(num_classes=2048).named_parameters():
        # print(name, param.shape)

    # model = MultiViewTimeFreqNet(base_encoder=resnet18)
    x1 = torch.randn((32, 1, 3000))
    x2 = torch.randn((32, 1, 3000))
    logits, labels = model(x1, x2)
    print(logits.shape, labels.shape)