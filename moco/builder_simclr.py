# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from utils.utils import nt_xent_debiased
from utils.utils_orth import similarity, rank_loss, rank_loss_self

from pdb import set_trace

class SimCLR(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, rank_alpha=0.5, return_features=False):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimCLR, self).__init__()

        self.rank_alpha = rank_alpha

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        self.return_features = return_features

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
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)


    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum, placeholder here
        Output:
            loss
        """

        # compute features
        # print("q1 = self.predictor(self.base_encoder(x1))")

        # print("x1.shape is {}".format(x1.shape))

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)

        set_trace()
        q1 = concat_all_gather_wGrad(q1)
        q2 = concat_all_gather_wGrad(q2)

        # rank_l =  rank_loss(q1, q2)
        rank_l =  rank_loss_self(q1, q2)
        similarity_l = -similarity(q1, q2)
        loss = self.rank_alpha * rank_l + (1 - self.rank_alpha) * similarity_l
        if not self.return_features:
            return loss, rank_l, similarity_l
        else:
            return rank_l, similarity_l, q1, q2


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
