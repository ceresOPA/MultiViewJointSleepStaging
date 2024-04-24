import torch
import torch.nn as nn
from network.contrastive_network import ContrastiveSignalNet
from utils.select_backbone import select_backbone

class PureCoTrainNet(ContrastiveSignalNet):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network1, network2, seq_len=16, feature_size=2048, dim=128, K=2048, m=0.999, T=0.07, topk=5, reverse=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(PureCoTrainNet, self).__init__(network1, seq_len, feature_size, dim, K, m, T)

        self.topk = topk

        # create another encoder, for the second view of the data 
        backbone = select_backbone(network2)
        self.sampler = nn.Sequential(
                        backbone,
                        nn.Linear(feature_size, dim)) # output layer

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)
        
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)
        
        self.queue_is_full = False
        self.reverse = reverse 

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # keys_second = concat_all_gather(keys_second)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_label[ptr:ptr + batch_size] = torch.ones(batch_size)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block1, block2):
        '''Output: logits, targets'''
        (B, N, *_) = block1.shape # B,N,C,T,H,W
        assert N == 2
        x1 = block1[:,0,...]
        x2 = block1[:,1,...]
        f1 = block2[:,0,...]
        f2 = block2[:,1,...]

        if self.reverse:
            x1, f1 = f1, x1
            x2, f2 = f2, x2 

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = self.projection_layer(q)
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = self.projection_layer(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.view(B, self.dim)

            # compute key feature for second view
            kf = self.sampler(f2) # keys: B,C,1,1,1
            kf = nn.functional.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        mask = torch.zeros(l_neg.shape, dtype=torch.float32).cuda()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)
            if self.queue_is_full: print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)
        
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf)

        return logits, mask.detach()


class CoTrainNet(ContrastiveSignalNet):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network1, network2, seq_len=16, feature_size=2048, dim=128, K=2048, m=0.999, T=0.07, topk=5, reverse=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoTrainNet, self).__init__(network1, seq_len, feature_size, dim, K, m, T)

        self.topk = topk

        # create another encoder, for the second view of the data 
        backbone = select_backbone(network2)
        self.sampler = nn.Sequential(
                        backbone,
                        nn.Linear(feature_size, dim)) # output layer

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)
        
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)
        
        self.queue_is_full = False
        self.reverse = reverse 

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # keys_second = concat_all_gather(keys_second)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_label[ptr:ptr + batch_size] = torch.ones(batch_size)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block1, block2):
        '''Output: logits, targets'''
        # (B, N, *_) = block1.shape # B,N,C,T,H,W
        # assert N == 2
        x1 = block1
        f1 = block2

        B = x1.shape[0]

        if self.reverse:
            x1, f1 = f1, x1

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = self.projection_layer(q)
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x1)  # keys: B,C,1,1,1
            k = self.projection_layer(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.view(B, self.dim)

            # compute key feature for second view
            kf = self.sampler(f1) # keys: B,C,1,1,1
            kf = nn.functional.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        mask = torch.zeros(l_neg.shape, dtype=torch.float32).cuda()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)
            if self.queue_is_full: print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            sim_score, topkidx = torch.topk(mask_sim, self.topk)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)
        
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf)

        return logits, mask.detach()

if __name__ == "__main__":
    model = CoTrainNet(network1='tinysleepnet_cnn', network2='resnet18')
    for i in range(20):
        x = torch.randn(128,  1, 3000)
        f = torch.randn(128,  3, 308, 610)
        logits, mask = model(x, f)
        print(logits.shape, mask.shape)
        