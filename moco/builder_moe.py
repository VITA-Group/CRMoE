# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import nt_xent_debiased
from utils.moe_utils import collect_moe_activation
from utils.bbox_calculation import calculateAreaLossWmiou
from utils.wasserteinLoss import wassertein_distance
from pdb import set_trace

from models.gate_funs.noisy_gate import NoisyGate
import numpy as np
import random

def calculate_gate_num(model):
    out_channels = []
    for m in model.modules():
        if isinstance(m, NoisyGate):
            out_channels.append(m.tot_expert)

    return np.sum(out_channels)


def calculate_gate_channel(model):
    out_channels = []
    for m in model.modules():
        if isinstance(m, NoisyGate):
            out_channels.append(m.tot_expert)

    for out in out_channels:
        assert out == out_channels[0]

    return out_channels[0]

def calculate_gate_info(model):
    out_channels = []
    for m in model.modules():
        if isinstance(m, NoisyGate):
            out_channels.append(m.tot_expert)

    return out_channels

class MoCo_MoE(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, simclr_version=False, gate_model=None,
                 gate_dim=192, contrastive_gate_w=-1, contrastive_gate_proj_layers=-1,
                 contrastive_wassertein_gate=False,
                 wassertein_neg_w=0,
                 wassertein_gate_no_cls=False,
                 wassertein_gate_no_cls_w=1.0,
                 esvit_gate=False,
                 iou_gate=False,
                 iou_gate_similarity_mode=False,
                 iou_gate_threshold=0.2,
                 iou_gate_alpha=0.5,
                 cls_token_gate=False,
                 ):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo_MoE, self).__init__()

        self.T = T
        self.simclr_version = simclr_version

        self.contrastive_wassertein_gate = contrastive_wassertein_gate
        self.wassertein_neg_w = wassertein_neg_w
        self.wassertein_gate_no_cls = wassertein_gate_no_cls
        self.wassertein_gate_no_cls_w = wassertein_gate_no_cls_w

        self.esvit_gate = esvit_gate
        self.cls_token_gate = cls_token_gate

        self.iou_gate = iou_gate
        self.iou_gate_similarity_mode = iou_gate_similarity_mode
        self.iou_gate_threshold = iou_gate_threshold
        self.iou_gate_alpha = iou_gate_alpha

        assert int(self.esvit_gate) + int(self.contrastive_wassertein_gate) + int(self.cls_token_gate) + int(self.iou_gate) <= 1

        # build gate_model
        if gate_model is not None:
            # gate fun that outputs feature
            self.gate_model = gate_model(num_classes=0)
        else:
            self.gate_model = None
            gate_dim = -1

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim, gate_dim=gate_dim)
        if not self.simclr_version:
            self.momentum_encoder = base_encoder(num_classes=mlp_dim, gate_dim=gate_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        self.contrastive_gate_w = contrastive_gate_w
        # contrastive_gate_proj_layers {-1: no concat, no proj, 0: concat, 1: one proj layer, 2: two proj layers}
        self.contrastive_gate_proj_layers = contrastive_gate_proj_layers
        if self.contrastive_gate_proj_layers == 0:
            self.base_encoder.gate_proj_layer = nn.Identity()
            self.momentum_encoder.gate_proj_layer = nn.Identity()
        elif self.contrastive_gate_proj_layers > 0:
            gate_proj_head_dim = calculate_gate_num(self.base_encoder)
            self.base_encoder.gate_proj_layer = self._build_mlp(self.contrastive_gate_proj_layers, gate_proj_head_dim, gate_proj_head_dim, dim)
            self.momentum_encoder.gate_proj_layer = self._build_mlp(self.contrastive_gate_proj_layers, gate_proj_head_dim, gate_proj_head_dim, dim)

        if self.wassertein_gate_no_cls:
            assert self.contrastive_gate_proj_layers < 0

        if not simclr_version:
            for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


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

        # the gate function is the same
        if self.gate_model is not None:
            for (name_b, param_b), (name_m, param_m) in zip(self.base_encoder.named_parameters(), self.momentum_encoder.named_parameters()):
                assert name_b == name_m
                # TODO: ensure the name format
                if "gate" in name_b:
                    param_m.data = param_b.data


    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        # print("logits mean is {}".format(logits.mean()))
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # print("labels is {}".format(labels))
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def cos_sim_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        cos_sim = (q * k).sum(-1).mean()

        # print("labels is {}".format(labels))
        return - cos_sim

    def cos_sim_loss_batch(self, q, k, mask):
        # normalize
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        cos_sim = ((q * k).sum(-1) * mask).sum() / mask.sum().clamp(min=1)

        # print("labels is {}".format(labels))
        return - cos_sim

    def simclr_loss(self, q1, q2):
        # normalize
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        q1 = concat_all_gather_wGrad(q1)
        q2 = concat_all_gather_wGrad(q2)

        return nt_xent_debiased(q1, features2=q2, t=self.T) * torch.distributed.get_world_size()

    def forward(self, x1, x2, m, x3=None, x_local=None):
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

        if x3 is None:
            assert self.gate_model is None
            gate_inp = None
        else:
            gate_inp = self.gate_model(x3)

        q1 = self.predictor(self.base_encoder(x1, gate_inp))

        if self.contrastive_wassertein_gate or self.esvit_gate or self.cls_token_gate or self.iou_gate:
            suppress = "origin"
        else:
            suppress = "pool"

        if self.cls_token_gate:
            assert self.contrastive_gate_proj_layers < 0

        # print("activation_suppress is {}".format(suppress))
        q1_gate = collect_moe_activation(self.base_encoder, q1.shape[0], activation_suppress=suppress)
        if self.contrastive_gate_w < 0:
            q1_gate = None

        q2 = self.predictor(self.base_encoder(x2, gate_inp))
        q2_gate = collect_moe_activation(self.base_encoder, q2.shape[0], activation_suppress=suppress)
        if self.contrastive_gate_w < 0:
            q2_gate = None

        if self.simclr_version:
            assert self.gate_model is None, "Simclr verison is not implemented right now"
            return self.simclr_loss(q1, q2)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1, gate_inp)
            k1_gate = collect_moe_activation(self.momentum_encoder, k1.shape[0], activation_suppress=suppress)
            if self.contrastive_gate_w < 0:
                k1_gate = None

            k2 = self.momentum_encoder(x2, gate_inp)
            k2_gate = collect_moe_activation(self.momentum_encoder, k2.shape[0], activation_suppress=suppress)
            if self.contrastive_gate_w < 0:
                k2_gate = None

        loss_cl = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

        if x_local is not None:
            len_x_local = len(x_local)
            x_local = torch.cat(x_local)
            local_features = self.predictor(self.base_encoder(x_local, gate_inp))
            collect_moe_activation(self.base_encoder, x_local.shape[0], activation_suppress=suppress)
            local_features = local_features.chunk(len_x_local)
            for local_feature in local_features:
                loss_cl += self.contrastive_loss(local_feature, k1) + self.contrastive_loss(local_feature, k2)

        if self.contrastive_gate_w > 0:
            loss_gate = 0
            if self.contrastive_gate_proj_layers < 0:
                return loss_cl, q1_gate, q2_gate, k1_gate, k2_gate
                # for q1_g, q2_g, k1_g, k2_g in zip(q1_gate, q2_gate, k1_gate, k2_gate):
                #     loss_gate = self.contrastive_gate_w * self.gate_contrastive_loss(q1_g, q2_g, k1_g, k2_g, wassertein_distance_metric)
            elif self.contrastive_gate_proj_layers >= 0:
                q1_g, q2_g, k1_g, k2_g = [torch.cat(x, dim=-1) for x in [q1_gate, q2_gate, k1_gate, k2_gate]]
                q1_g, q2_g = self.base_encoder.gate_proj_layer(q1_g), self.base_encoder.gate_proj_layer(q2_g)
                k1_g, k2_g = self.momentum_encoder.gate_proj_layer(k1_g), self.momentum_encoder.gate_proj_layer(k2_g)
                # loss_gate = self.contrastive_gate_w * self.gate_contrastive_loss(q1_g, q2_g, k1_g, k2_g, wassertein_distance_metric)
                return loss_cl, q1_g, q2_g, k1_g, k2_g
        else:
            loss_gate = 0

        return loss_cl + loss_gate

    def gate_contrastive_loss(self, q1, q2, k1, k2, wassertein_distance_metric=None, wassertein_distance_metric_neg=None, bboxes=None):
        if self.contrastive_wassertein_gate:
            if self.wassertein_gate_no_cls:
                cls_loss = 0
                for q1_s, q2_s, k1_s, k2_s in zip(q1, q2, k1, k2):
                    cls_loss += self.contrastive_loss(q1_s[:, 0], k2_s[:, 0]) + self.contrastive_loss(q2_s[:, 0], k1_s[:, 0])
                wassertein_distance_loss = self.contrastive_wassertein_loss(q1, k2, wassertein_distance_metric, wassertein_distance_metric_neg)
                loss = cls_loss + self.wassertein_gate_no_cls_w * wassertein_distance_loss
                # print("cls_loss + self.wassertein_gate_no_cls_w * wassertein_distance_loss, self.wassertein_gate_no_cls_w is {}".format(self.wassertein_gate_no_cls_w))
            else:
                loss = self.contrastive_wassertein_loss(q1, k2, wassertein_distance_metric, wassertein_distance_metric_neg)
        elif self.esvit_gate:
            cls_loss = self.contrastive_loss(q1[:, 0], k2[:, 0]) + self.contrastive_loss(q2[:, 0], k1[:, 0])
            esvit_loss = self.contrastive_loss_batch_esvit(q1[:, 1:], k2[:, 1:]) + self.contrastive_loss_batch_esvit(q2[:, 1:], k1[:, 1:])
            loss = cls_loss + self.wassertein_gate_no_cls_w * esvit_loss
            # print("esvit weight is {}".format(self.wassertein_gate_no_cls_w))
        elif self.cls_token_gate:
            # only contrasting the cls token
            # print("q1[:, 0] is {}".format(q1[:, 0].shape))
            loss = self.contrastive_loss(q1[:, 0], k2[:, 0]) + self.contrastive_loss(q2[:, 0], k1[:, 0])
        elif self.iou_gate:
            q1_noCls, q2_noCls, k1_noCls, k2_noCls = q1[:, 1:], q2[:, 1:], k1[:, 1:], k2[:, 1:]
            contrast_pair1, contrast_pair2 = calculateAreaLossWmiou(q1_noCls, q2_noCls, k1_noCls, k2_noCls, bboxes, self.iou_gate_threshold,
                                                                    patch_num_width = int(np.sqrt(q1_noCls.shape[1])),
                                                                    patch_num_height = int(np.sqrt(q1_noCls.shape[1])))
            mask1, q1_iou, k2_iou = contrast_pair1
            mask2, q2_iou, k1_iou = contrast_pair2

            if not self.iou_gate_similarity_mode:
                iou_loss = self.contrastive_loss_batch(q1_iou, k2_iou, mask1) + \
                           self.contrastive_loss_batch(q2_iou, k1_iou, mask2)

                cls_loss = self.contrastive_loss(q1[:, 0], k2[:, 0]) + self.contrastive_loss(q2[:, 0], k1[:, 0])
            else:
                iou_loss = self.cos_sim_loss_batch(q1_iou, k2_iou, mask1) + \
                           self.cos_sim_loss_batch(q2_iou, k1_iou, mask2)

                cls_loss = self.cos_sim_loss(q1[:, 0], k2[:, 0]) + self.cos_sim_loss(q2[:, 0], k1[:, 0])

            loss = (1 - self.iou_gate_alpha) * cls_loss + self.iou_gate_alpha * iou_loss
            # print("iou_gate_alpha is {}".format(self.iou_gate_alpha))
        else:
            loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

        return loss


    def contrastive_wassertein_loss(self, q, k, wassertein_distance_metric, wassertein_distance_metric_neg=None):
        if isinstance(q, list):
            if self.wassertein_gate_no_cls:
                q = [item[:, 1:] for item in q]
                k = [item[:, 1:] for item in k]
            # print("q len is {}, each item shape is {}".format(len(q), q[0].shape))
            k_for_neg = k
            q = torch.cat(q, dim=0)
            k = torch.cat(k, dim=0)
        else:
            assert not self.wassertein_gate_no_cls

            k_for_neg = k

        # 1. calculate the positive pairs wassertein distance
        # print("q.shape is {}".format(q.shape))
        distant_pos = wassertein_distance_metric(q, k)

        if self.wassertein_neg_w < 1e-6:
            return distant_pos.mean(0)

        # 2. calculate the negtive pairs wassertein distance
        k_neg = rand_gene_negative_pair(k_for_neg)
        distant_neg = wassertein_distance_metric_neg(q, k_neg)

        return distant_pos.mean(0) - self.wassertein_neg_w * distant_neg.mean(0)


    def contrastive_loss_batch_esvit(self, q, k):
        '''
        :param q: n, patches, c
        :param k: n, patches, c
        :return:
        '''
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        similarity = torch.bmm(q_norm, k_norm.permute(0, 2, 1))
        max_sim_idx = similarity.max(dim=2)[1]

        k_max_sim = torch.gather(k_norm, 1, max_sim_idx.unsqueeze(2).expand(-1, -1, k.size(2)))

        return self.contrastive_loss_batch(q, k_max_sim)


    def contrastive_loss_batch(self, q, k, mask=None):
        # normalize
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        # print("q shape is {}".format(q.shape))
        # print("k shape is {}".format(k.shape))

        logits = torch.bmm(q.permute(1,0,2), k.permute(1,2,0)).permute(1,0,2) / self.T
        # logits = torch.einsum('nbc,mbc->nbm', [q, k]) / self.T
        # print("logits mean is {}".format(logits.mean()))
        N, B, _ = logits.shape  # batch size per GPU
        logits = logits.permute(0, 2, 1)
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda().unsqueeze(-1).expand(N, B)
        if mask is not None:
            mask = mask.to(labels.device)
            labels = mask * labels + (1 - mask.float()) * -1
            labels = labels.long()

        # print("labels is {}".format(labels))
        return nn.CrossEntropyLoss(ignore_index=-1)(logits, labels) * (2 * self.T)


def rand_gene_negative_pair(k):
    if isinstance(k, list):
        return torch.cat([rand_gene_negative_pair(x) for x in k], dim=0)

    b = k.shape[0]
    assert b >= 2
    rand_values = []
    for i in range(b):
        while True:
            rand_value = random.randint(0, b-1)
            if i != rand_value:
                break
        rand_values.append(rand_value)
    return k[rand_values]


class MoCo_MoE_ResNet(MoCo_MoE):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_MoE_ViT(MoCo_MoE):
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
