# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pdb import set_trace

import torch
import torch.nn as nn
from utils.utils import nt_xent_debiased

class MoCo_Sep(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, simclr_version=False, return_features=False, return_representation=False):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo_Sep, self).__init__()

        self.T = T
        self.simclr_version = simclr_version

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        if not self.simclr_version:
            self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        if not simclr_version:
            for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        self.return_features = return_features
        self.return_representation = return_representation

        if self.return_representation:
            self.base_encoder.head = nn.Identity()
            self.momentum_encoder.head = nn.Identity()
            self.predictor = nn.Identity()

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def simclr_loss(self, q1, q2):
        # normalize
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        q1 = concat_all_gather_wGrad(q1)
        q2 = concat_all_gather_wGrad(q2)

        return nt_xent_debiased(q1, features2=q2, t=self.T) * torch.distributed.get_world_size()

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        # print("q1 = self.predictor(self.base_encoder(x1))")

        # print("x1.shape is {}".format(x1.shape))

        self.base_encoder.set_block_path(0)
        q1 = self.predictor(self.base_encoder(x1))
        self.base_encoder.set_block_path(1)
        q2 = self.predictor(self.base_encoder(x2))

        if self.simclr_version:
            assert not self.return_features
            return self.simclr_loss(q1, q2)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            self.momentum_encoder.set_block_path(0)
            k1 = self.momentum_encoder(x1)
            self.momentum_encoder.set_block_path(1)
            k2 = self.momentum_encoder(x2)

        if self.return_features:
            q1 = concat_all_gather_wGrad(q1.contiguous())
            q2 = concat_all_gather_wGrad(q2.contiguous())
            k1 = concat_all_gather_wGrad(k1.contiguous())
            k2 = concat_all_gather_wGrad(k2.contiguous())
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), q1, q2, k1, k2
        else:
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo_Sep):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo_Sep):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head
        if not self.simclr_version:
            del self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        if not self.simclr_version:
            self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def concat_all_gather_wGrad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    tensors_gather[torch.distributed.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output
