# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import math
import os
import random
import shutil
import datetime
import numpy as np
import time
import warnings
import fmoe
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as torchvision_models
import torch.distributed
from torch.distributed import Backend
import json

from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.utils.clip_grad import dispatch_clip_grad

from deit_utils.datasets import build_dataset
from deit_utils.engine import train_one_epoch, evaluate
from deit_utils.losses import DistillationLoss
from deit_utils.samplers import RASampler
from models import vits_gate, vits_moe, vits
import utils
from utils.lr_sched import adjust_learning_rate

from utils.utils import logger, accuracy, sync_weights
from utils.moe_utils import read_specific_group_experts, collect_moe_model_state_dict, save_checkpoint, \
    collect_noisy_gating_loss, prune_moe_experts
from utils.init_datasets import init_datasets

from models.vits_gate import VisionTransformerMoCoWithGate

from pdb import set_trace

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    parser.add_argument('experiment', type=str)
    parser.add_argument('--save_dir', type=str, default="checkpoints_moco")
    parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')
    parser.add_argument('--finetune_in_train', action='store_true', help='if employ finetune for training')

    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--arch', default='resnet50', type=str, help='model architecture')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # moe parameters
    parser.add_argument('--moe-experts', default=4, type=int, help='moe experts number')
    parser.add_argument('--moe-mlp-ratio', default=-1, type=int, help='moe dim')
    parser.add_argument('--moe-top-k', default=2, type=int, help='moe top k')
    parser.add_argument('--moe-noisy-gate-loss-weight', default=0.01, type=float)
    parser.add_argument('--moe-gate-type', default="noisy", type=str)
    parser.add_argument('--vmoe-noisy-std', default=0, type=float)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--kaiming-sched', action="store_true",
                        help='if use the lr schedule kaiming used')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data', metavar='DIR', default="", help='path to dataset')
    parser.add_argument('--dataset', default="imagenet", help='dataset')
    parser.add_argument('--customSplit', type=str, default='')
    parser.add_argument('--tuneFromFirstFC', action="store_true", help="if tune from the first fc")
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--test-interval', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # only use this for cls token for simplicity
    parser.add_argument('--moe-contrastive-weight', default=-1, type=float)
    parser.add_argument('--moe-contrastive-supervised', action="store_true")

    # distributed training parameters
    parser.add_argument('--local_rank', default=-1, type=int, help='GPU id to use.')
    parser.add_argument('--moe-data-distributed', action="store_true", help='if use moe-data-distributed')
    return parser


def create_model(args, nb_classes, log):
    log.info("=> creating model '{}'".format(args.arch))
    args.moe_use_gate = False
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'head'
    elif args.arch.startswith('moe_vit'):
        if args.moe_data_distributed:
            moe_world_size = 1
        else:
            moe_world_size = torch.distributed.get_world_size()
        if args.moe_experts % moe_world_size != 0:
            print("experts number of {} is not divisible by world size of {}".format(args.moe_experts, moe_world_size))
        args.moe_experts = args.moe_experts // moe_world_size

        if args.moe_use_gate:
            gate_model = vits_gate.__dict__[args.moe_gate_arch](num_classes=0)
            model = vits_moe.__dict__[args.arch](moe_mlp_ratio=args.moe_mlp_ratio, moe_experts=args.moe_experts, moe_top_k=args.moe_top_k,
                                                 world_size=moe_world_size, gate_dim=gate_model.num_features,
                                                 num_classes=nb_classes,
                                                 moe_gate_type=args.moe_gate_type, vmoe_noisy_std=args.vmoe_noisy_std)
            model = VisionTransformerMoCoWithGate(model, gate_model)
        else:
            model = vits_moe.__dict__[args.arch](moe_experts=args.moe_experts, moe_top_k=args.moe_top_k,
                                                 world_size=moe_world_size,
                                                 moe_mlp_ratio=args.moe_mlp_ratio,
                                                 num_classes=nb_classes,
                                                 moe_gate_type=args.moe_gate_type, vmoe_noisy_std=args.vmoe_noisy_std)
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'fc'
        assert not args.tuneFromFirstFC

    if args.tuneFromFirstFC:
        middle_dim = 4096
        ch = model.head.in_features
        num_class = model.head.out_features
        model.head = nn.Sequential(nn.Linear(ch, middle_dim, bias=False), nn.Linear(middle_dim, num_class))

    return model, linear_keyword


def cvt_state_dict(state_dict, model, args, linear_keyword, moe_dir_mode=False, tuneFromFirstFC=False):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]

        if k.startswith('module.base_encoder.%s.0' % linear_keyword) and tuneFromFirstFC:
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]

    if "moe" in args.arch and (not moe_dir_mode) and (not args.moe_data_distributed):
        # if args.local_rank == 0:
        #     print("state dict block1 h4toh avg is {}".format(state_dict["blocks.1.mlp.experts.h4toh.weight"].mean(-1).mean(-1)))
        state_dict = read_specific_group_experts(state_dict, torch.distributed.get_rank(), args.moe_experts)

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)

    if tuneFromFirstFC:
        print(msg.missing_keys)
        assert set(msg.missing_keys) == {"%s.1.weight" % linear_keyword, "%s.1.bias" % linear_keyword}
    else:
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    return model


class NativeScalerForMoe:
    state_dict_key = "amp_scaler"

    def __init__(self, model):
        self.model = model
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self.model.allreduce_params()
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def main(args):
    torch.distributed.init_process_group(backend=Backend.NCCL, init_method="env://")
    torch.distributed.barrier()
    torch.cuda.set_device(args.local_rank)

    best_acc1 = -1

    logName = "log.txt"
    output_dir = args.output_dir
    log = logger(path=output_dir, log_name=logName, local_rank=torch.distributed.get_rank())

    log.info(str(args))

    # global batchsize to batchsize of each gpu
    if args.batch_size % torch.distributed.get_world_size() != 0:
        raise ValueError("Batch size of {} is not divisible by world size of {}".format(args.batch_size, torch.distributed.get_world_size()))
    args.batch_size = int(args.batch_size / torch.distributed.get_world_size())

    if args.distillation_type != 'none' and args.pretrained and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)
    print("device is {}".format(device))

    # fix the seed for reproducibility
    seed = args.seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val, dataset_test = build_dataset(args=args)

    if True:  # args.distributed:
        num_tasks = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        # if args.dist_eval:
        # if len(dataset_val) % num_tasks != 0:
        #     log.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #           'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #           'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # sampler_test = torch.utils.data.DistributedSampler(
        #     dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # else:
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    args.nb_classes = len(np.unique(dataset_train.targets))

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    log.info(f"Creating model: {args.arch}")
    # TODO: change the way for defining model
    model, linear_keyword = create_model(args, args.nb_classes, log)

    if args.pretrained:
        if args.pretrained.startswith('https'):
            assert False
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            if os.path.isfile(args.pretrained) or os.path.isdir(args.pretrained):
                log.info("=> loading checkpoint '{}'".format(args.pretrained))
                if os.path.isfile(args.pretrained):
                    checkpoint = torch.load(args.pretrained, map_location="cpu")
                    moe_dir_read = False
                elif os.path.isdir(args.pretrained):
                    checkpoint = torch.load(os.path.join(args.pretrained, "0.pth".format(torch.distributed.get_rank())),
                                            map_location="cpu")
                    len_save = len([f for f in os.listdir(args.pretrained) if "pth" in f])
                    if args.moe_data_distributed:
                        moe_world_size = 1
                        response_cnt = [i for i in range(len_save)]
                    else:
                        moe_world_size = torch.distributed.get_world_size()
                        assert len_save % moe_world_size == 0
                        response_cnt = [i for i in range(
                            torch.distributed.get_rank() * (len_save // moe_world_size),
                            (torch.distributed.get_rank() + 1) * (len_save // moe_world_size))]
                    # merge all ckpts
                    for cnt, cnt_model in enumerate(response_cnt):
                        if cnt_model != 0:
                            checkpoint_specific = torch.load(os.path.join(args.pretrained, "{}.pth".format(cnt_model)),
                                                             map_location="cpu")
                            if cnt != 0:
                                for key, item in checkpoint_specific["state_dict"].items():
                                    checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                              dim=0)
                            else:
                                checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                        moe_dir_read = True
                else:
                    raise ValueError("Model {} do not exist".format(args.pretrained))

                if "mae" in args.pretrained and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    args.start_epoch = 0
                    msg = model.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                elif args.moe_use_gate:
                    assert not args.tuneFromFirstFC
                    state_dict = checkpoint['state_dict']
                    model = cvt_state_dict_moe_gate(state_dict, model, args, linear_keyword)
                else:
                    state_dict = checkpoint['state_dict']
                    model = cvt_state_dict(state_dict, model, args, linear_keyword, moe_dir_read, args.tuneFromFirstFC)

                log.info("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                raise ValueError("no such check point")

        # state_dict = model.state_dict()
        # # interpolate position embedding
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        # embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = model.patch_embed.num_patches
        # num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # # class_token and dist_token are kept unchanged
        # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # # only the position tokens are interpolated
        # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # pos_tokens = torch.nn.functional.interpolate(
        #     pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        # checkpoint_model['pos_embed'] = new_pos_embed

        # msg = model.load_state_dict(checkpoint_model, strict=False)
        # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    model.to(device)

    model_ema = None
    if args.model_ema:
        log.info("Employing EMA")
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model

    if "moe" in args.arch and (not args.moe_data_distributed):
        model = fmoe.DistributedGroupedDataParallel(model)
        sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info('number of params: {}'.format(n_parameters))

    linear_scaled_lr = args.lr * args.batch_size * torch.distributed.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    if "moe" in args.arch and (not args.moe_data_distributed):
        loss_scaler = NativeScalerForMoe(model)
    else:
        loss_scaler = NativeScaler()

    if not args.kaiming_sched:
        lr_scheduler, _ = create_scheduler(args, optimizer)
    else:
        lr_scheduler = None

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        log.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model, _ = create_model(args, args.nb_classes, log)
        sync_weights(teacher_model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif os.path.isdir(args.resume):
            checkpoint = torch.load(os.path.join(args.resume, "0.pth".format(torch.distributed.get_rank())),
                                    map_location="cpu")
            len_save = len([f for f in os.listdir(args.resume) if "pth" in f])
            assert len_save % torch.distributed.get_world_size() == 0
            response_cnt = [i for i in range(
                torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.resume, "{}.pth".format(cnt_model)),
                                                     map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                      dim=0)
                    else:
                        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                moe_dir_read = True
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_test, model, device)
        log.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    log.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode= args.pretrained == '' or args.finetune_in_train,  # keep in eval mode during finetuning,
            args=args, log=log
        )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        if epoch % args.test_interval == 0:
            # evaluate on validation set
            test_stats = evaluate(data_loader_val, model, device, log)
            log.info(f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.1f}%")

            # remember best acc@1 and save checkpoint
            acc1 = test_stats['acc1']
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_state_dict = model.state_dict()

            if args.arch.startswith('moe_vit') and (not args.moe_data_distributed):
                # hacking for fast saving (saving less times), might miss the best model when it shows in the early stage
                fast_saving_epoch = int(0.8 * args.epochs)
                if epoch % 10 == 0 and epoch < fast_saving_epoch:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': save_state_dict,
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, False, save_dir=log.path,
                        moe_save=args.arch.startswith('moe_vit'))
                elif epoch == fast_saving_epoch:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': save_state_dict,
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, True, save_dir=log.path, only_best=True,
                        moe_save=args.arch.startswith('moe_vit'))
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': save_state_dict,
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, save_dir=log.path, only_best=True,
                        moe_save=args.arch.startswith('moe_vit'))
            else:
                if torch.distributed.get_rank() == 0: # only the first GPU saves checkpoint
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': save_state_dict,
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, save_dir=log.path,
                        moe_save=args.arch.startswith('moe_vit') and (not args.moe_data_distributed))

            log.info(f'Max accuracy: {best_acc1:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}'.format(total_time_str))

    torch.cuda.empty_cache()
    time.sleep(5) # sleep 5 secs for models to finish the saving
    # load best model for testing
    if "moe" in args.arch and (not args.moe_data_distributed):
        # state_dict = read_specific_group_experts(checkpoint['state_dict'], args.local_rank, args.moe_experts)
        checkpoint_specific = torch.load(os.path.join(log.path, "model_best.pth.tar", "{}.pth".format(torch.distributed.get_rank())), map_location="cpu")
        checkpoint = torch.load(os.path.join(log.path, "model_best.pth.tar", "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
        state_dict = checkpoint["state_dict"]
    else:
        checkpoint = torch.load(os.path.join(log.path, 'model_best.pth.tar'), map_location="cpu")
        state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    test_stats = evaluate(data_loader_test, model, device, log)
    log.info("Top 1 acc for best model is {}".format(test_stats['acc1']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.save_dir, args.experiment)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)