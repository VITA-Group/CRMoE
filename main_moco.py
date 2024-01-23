#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import fmoe
from pdb import set_trace

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.builder_moe
import moco.builder_simclr
import moco.builder_sep
import moco.loader
import moco.optimizer

from models import vits_gate, vits_moe, vits, vits_sep
from models.moe_sync_batch_norm import SyncBatchNorm as moeSyncBatchNorm

from utils.utils import logger, sync_weights
from utils.init_datasets import init_datasets
from utils.moe_utils import read_specific_group_experts, collect_moe_model_state_dict, save_checkpoint, collect_noisy_gating_loss
from utils.pretrain import pretrain_transform
from utils.transform_w_pos import pretrain_transform_w_pos

from utils.moe_utils import get_parameter_group

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
model_names += ['moe_vit_small', 'moe_vit_base']
model_names += ['sep_vit_small']

gate_names = ['', 'vit_gate_small', 'vit_gate_base', 'vit_gate_large']

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')

parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', type=str, default="checkpoints_moco")

parser.add_argument('--data', default="", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--customSplit', type=str, default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--experts_lr_ratio', default=1, type=float,
                    help='the lr ratio of moe_experts ratio to base_lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_freq', default=2000, type=int, help='The freq for saving model.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--no_save_last', action='store_true')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# simclr version
parser.add_argument('--simclr_version', action='store_true', help='simclr version')

# moe configs:
parser.add_argument('--moe-data-distributed', action='store_true', help='if employ moe data distributed')
parser.add_argument('--moe-experts', default=4, type=int, help='moe experts number')
parser.add_argument('--moe-mlp-ratio', default=-1, type=int, help='moe dim')
parser.add_argument('--moe-top-k', default=2, type=int, help='moe top k')
parser.add_argument('--moe-noisy-gate-loss-weight', default=0.01, type=float)
parser.add_argument('--moe-gate-arch', default="", choices=gate_names, type=str)
parser.add_argument('--moe-contrastive-weight', default=-1, type=float)
parser.add_argument('--moe-contrastive-gate-proj-layers', default=-1, type=float)
parser.add_argument('--moe-gate-type', default="noisy", type=str)
parser.add_argument('--moe-same-for-all', action='store_true', help='if using the same moe for all layers')
parser.add_argument('--vmoe-noisy-std', default=0, type=float)

# moe wassertein loss configs:
parser.add_argument('--moe-wassertein-gate', action='store_true', help='control wassertein gate')
parser.add_argument('--moe-wassertein-gate-steps', default=50, type=int, help='optimization steps for moe wassertein gate')
parser.add_argument('--moe-wassertein-gate-lr', default=3e-4, type=float, help='moe gate optimization lr')
parser.add_argument('--moe-wassertein-neg-w', default=0, type=float, help='moe gate neg term weight')
parser.add_argument('--moe-wassertein-gate-layers', default=2, type=int, help='moe gate neg term layers number')
parser.add_argument('--moe-wassertein-gate-gp-w', default=1000, type=float, help='moe gate gradient penalty term')
parser.add_argument('--moe-wassertein-gate-no-cls', action='store_true', help='exclude cls token for calculating w loss')
parser.add_argument('--moe-wassertein-gate-no-cls-w', default=1.0, type=float, help='the weight compared to cls part')

# moe esvit gate
parser.add_argument('--moe-esvit-gate', action='store_true', help='if employing esvit-gate')
parser.add_argument('--moe-cls-token-gate', action='store_true', help='if only apply contrasting on cls token')

# moe location based gate
parser.add_argument('--moe-iou-gate', action='store_true', help='patch match with according to miou between patches')
parser.add_argument('--moe-iou-gate-threshold', default=0.2, type=float, help='miou gate threshold')
parser.add_argument('--moe-iou-gate-alpha', default=0.5, type=float, help='miou gate alpha')
parser.add_argument('--moe-gate-return-decoupled-activation', action='store_true', help='if return decoupled-activation')
parser.add_argument('--moe-gate-return-gated-activation', action='store_true', help='if return gated-activation')
parser.add_argument('--moe-iou-gate-similarity-mode', action='store_true', help='if employ gate similarity mode')

# simRank
parser.add_argument('--sim_rank', action='store_true', help='if employ moe data distributed')
parser.add_argument('--sim_rank_alpha', default=0.5, type=float, help="The weight of rank loss")

parser.add_argument('--save_gpu_dif_file', action='store_true', help='if save dif files on dif gpus')
parser.add_argument('--load_gpu_dif_file', action='store_true', help='if save dif files on dif gpus')

# evaluate pretrain performance
parser.add_argument('--evaluate_pretrain', action='store_true', help='if evaluate pretrain performance')
parser.add_argument('--evaluate_pretrain_representation', action='store_true', help='if evaluate pretrain representation')
parser.add_argument('--evaluate_moe_gate_selection', action='store_true', help='if evaluate consistency of moe gate selection')
parser.add_argument('--evaluate_moe_gate_selection_fix_trans', action='store_true', help='if evaluate consistency of moe gate selection')

# save local crops
parser.add_argument('--local_crops_number', default=0, type=int, help='the local crops number')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.local_rank == -1:
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    logName = "log.txt"
    save_dir = os.path.join(args.save_dir, args.experiment)
    if not os.path.exists(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir, log_name=logName)

    # Simply call main_worker function
    main_worker(args.local_rank, args, log)


def main_worker(local_rank, args, log):
    args.local_rank = local_rank

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and args.local_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        log.local_rank = 1

    log.info(str(args))

    if args.local_rank is not None:
        log.info("Use GPU: {} for training".format(args.local_rank))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        torch.distributed.barrier()

    args.rank = torch.distributed.get_rank()
    log.local_rank = args.rank
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    args.moe_use_gate = (args.moe_gate_arch != "")
    if args.sim_rank:
        model = moco.builder_simclr.SimCLR(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.sim_rank_alpha, return_features=args.evaluate_pretrain)
    elif args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, return_features=args.evaluate_pretrain,
            return_representation=args.evaluate_pretrain_representation)
    elif args.arch.startswith('sep_vit'):
        model = moco.builder_sep.MoCo_ViT(
            partial(vits_sep.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, return_features=args.evaluate_pretrain,
            return_representation=args.evaluate_pretrain_representation)
    elif args.arch.startswith('moe_vit'):
        if args.moe_data_distributed:
            moe_world_size = 1
        else:
            moe_world_size = torch.distributed.get_world_size()
            if args.moe_experts % moe_world_size != 0:
                print("experts number of {} is not divisible by world size of {}".format(args.moe_experts, moe_world_size))
            args.moe_experts = args.moe_experts // moe_world_size

        if args.moe_use_gate:
            gate_model = partial(vits_gate.__dict__[args.moe_gate_arch], stop_grad_conv1=args.stop_grad_conv1)
        else:
            gate_model = None
        model = moco.builder_moe.MoCo_MoE_ViT(
            partial(vits_moe.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1,
                    moe_mlp_ratio=args.moe_mlp_ratio, moe_experts=args.moe_experts, moe_top_k=args.moe_top_k,
                    world_size=moe_world_size, gate_return_decoupled_activation=args.moe_gate_return_decoupled_activation,
                    gate_return_gated_activation=args.moe_gate_return_gated_activation,
                    moe_gate_type=args.moe_gate_type, vmoe_noisy_std=args.vmoe_noisy_std, moe_same_for_all=args.moe_same_for_all),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, simclr_version=args.simclr_version,
            gate_model=gate_model, contrastive_gate_w=args.moe_contrastive_weight,
            contrastive_gate_proj_layers=args.moe_contrastive_gate_proj_layers,
            contrastive_wassertein_gate=args.moe_wassertein_gate,
            wassertein_neg_w=args.moe_wassertein_neg_w,
            wassertein_gate_no_cls=args.moe_wassertein_gate_no_cls,
            wassertein_gate_no_cls_w=args.moe_wassertein_gate_no_cls_w,
            iou_gate=args.moe_iou_gate,
            iou_gate_similarity_mode=args.moe_iou_gate_similarity_mode,
            iou_gate_threshold=args.moe_iou_gate_threshold,
            iou_gate_alpha=args.moe_iou_gate_alpha,
            esvit_gate=args.moe_esvit_gate,
            cls_token_gate=args.moe_cls_token_gate,
        )
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        if "moe" in args.arch and (not args.moe_data_distributed):
            # TODO: replace the sync  batch norm here
            model = moeSyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / torch.distributed.get_world_size())
            args.workers = args.workers
            if "moe" in args.arch and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model, device_ids=[args.local_rank])
                sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if "moe" in args.arch and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    parameters = get_parameter_group(args, model)
    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(parameters, args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr,
                                      weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume) or args.load_gpu_dif_file:
            log.info("=> loading checkpoint '{}'".format(args.resume))
            if args.local_rank is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.local_rank)
                if not args.load_gpu_dif_file:
                    checkpoint = torch.load(args.resume, map_location=loc)
                else:
                    checkpoint = torch.load(args.resume.replace("[gpu]", "{}".format(args.local_rank)), map_location=loc)
            args.start_epoch = checkpoint['epoch']

            if args.arch.startswith('moe_vit') and (not args.moe_data_distributed):
                state_dict = read_specific_group_experts(checkpoint['state_dict'], args.local_rank, args.moe_experts)
                model.load_state_dict(state_dict)
                print("no read optimizer for moe archs")
            else:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint = torch.load(os.path.join(args.resume, "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
            len_save = len([f for f in os.listdir(args.resume) if "pth" in f])

            moe_world_size = 1 if args.moe_data_distributed else torch.distributed.get_world_size()
            moe_rank = 0 if args.moe_data_distributed else torch.distributed.get_rank()

            assert len_save % moe_world_size == 0
            response_cnt = [i for i in range(moe_rank * (len_save // moe_world_size),
                                             (moe_rank + 1) * (len_save // moe_world_size))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.resume, "{}.pth".format(cnt_model)), map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item], dim=0)
                    else:
                        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                moe_dir_read = True

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded moe checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.moe_iou_gate:
        train_transform = pretrain_transform_w_pos(args.crop_min, with_gate_aug=args.moe_use_gate,
                                                   local_crops_number=args.local_crops_number)
    else:
        train_transform = pretrain_transform(args.crop_min, with_gate_aug=args.moe_use_gate,
                                             local_crops_number=args.local_crops_number)

    train_dataset, _, _ = init_datasets(args, train_transform, None)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.evaluate_pretrain:
        from utils.pretrain import evaluate_pretrain, evaluate_pretrain_simRank
        if args.sim_rank:
            evaluate_pretrain_simRank(train_loader, model, args, log)
        else:
            evaluate_pretrain(train_loader, model, args, log)
        return

    if args.evaluate_moe_gate_selection:
        from visualization.visualize import visualize
        train_dataset, val_dataset, _ = init_datasets(args, train_transform, train_transform)
        visualize(val_dataset, model, fix_visualization=args.evaluate_moe_gate_selection_fix_trans, args=args, log=log)
        return 

    if args.moe_wassertein_gate:
        from moco.builder_moe import calculate_gate_channel, calculate_gate_num, calculate_gate_info
        from utils.wasserteinLoss import wassertein_distance

        wassertein_gate_steps = args.moe_wassertein_gate_steps
        wassertein_gate_lr = args.moe_wassertein_gate_lr
        wassertein_gp_w = args.moe_wassertein_gate_gp_w

        if model.module.contrastive_gate_proj_layers >= 0:
            in_dim = calculate_gate_num(model.module.base_encoder)
            wassertein_gate_batch_size = args.batch_size
        else:
            gate_info = calculate_gate_info(model.module.base_encoder)
            gate_num = len(gate_info)
            gate_channel_num = gate_info[0]
            wassertein_gate_batch_size = args.batch_size * gate_num
            in_dim = gate_channel_num

        proj_net_args = {"in_dim": in_dim, "hidden_dim": in_dim, "batch_size": wassertein_gate_batch_size,
                         "layers": args.moe_wassertein_gate_layers}
        print("proj_net_args is {}".format(proj_net_args))
        wassertein_distance_metric = wassertein_distance(proj_net_args, wassertein_gate_lr, wassertein_gate_steps, wassertein_gp_w, log)
        wassertein_distance_metric.distribute(args)
        wassertein_distance_metric = wassertein_distance_metric.cuda()
        print(wassertein_distance_metric.wassertein_proj)

        wassertein_distance_metric_neg = wassertein_distance(proj_net_args, wassertein_gate_lr, wassertein_gate_steps, wassertein_gp_w)
        wassertein_distance_metric_neg.distribute(args)
        wassertein_distance_metric_neg = wassertein_distance_metric_neg.cuda()
    else:
        wassertein_distance_metric = None
        wassertein_distance_metric_neg = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, epoch, args, log,
              wassertein_distance_metric=wassertein_distance_metric,
              wassertein_distance_metric_neg=wassertein_distance_metric_neg)

        save_state_dict = model.state_dict()

        moe_save = (args.arch.startswith('moe_vit') and (not args.moe_data_distributed))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0) or moe_save: # only the first GPU saves checkpoint

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename='checkpoint_%04d.pth.tar' % (epoch+1),
                save_dir=log.path, moe_save=moe_save)
            if not args.no_save_last:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename='checkpoint_last.pth.tar',
                save_dir=log.path, moe_save=moe_save)

        if args.multiprocessing_distributed:
            if args.save_gpu_dif_file:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename='checkpoint_last_rank{}.pth.tar'.format(args.local_rank),
                save_dir=log.path, moe_save=(args.arch.startswith('moe_vit') and (not args.moe_data_distributed)))

        if args.multiprocessing_distributed:
            torch.distributed.barrier()

    moe_save = (args.arch.startswith('moe_vit') and (not args.moe_data_distributed))
    if not args.multiprocessing_distributed or moe_save or \
            (args.multiprocessing_distributed and args.rank == 0):  # only the first GPU saves checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': save_state_dict,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, is_best=False, filename='checkpoint_final.pth.tar',
        save_dir=log.path, moe_save=moe_save)

    if args.multiprocessing_distributed:
        torch.distributed.barrier()


def train(train_loader, model, optimizer, scaler, epoch, args, log, wassertein_distance_metric=None,
          wassertein_distance_metric_neg=None):
    for param_group in optimizer.param_groups:
        if "name" in param_group:
            log.info("group {}, current lr is {}".format(param_group["name"], param_group["lr"]))
        else:
            log.info("current lr is {}".format(param_group["lr"]))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    meters = [batch_time, data_time, learning_rates, losses]

    if args.moe_contrastive_weight > 0 and args.moe_wassertein_gate:
        wassertein_distances = AverageMeter('w_dist', ':.4e')
        wassertein_distances_gp = AverageMeter('w_dist_gp', ':.4e')
        wassertein_distances_neg = AverageMeter('w_dist_neg', ':.4e')
        wassertein_distances_gp_neg = AverageMeter('w_dist_neg_gp', ':.4e')
        meters += [wassertein_distances, wassertein_distances_gp, wassertein_distances_neg, wassertein_distances_gp_neg]
    elif args.sim_rank:
        sim_losses = AverageMeter('sim_l', ':.4e')
        rank_losses = AverageMeter('rank_l', ':.4e')
        meters = [batch_time, data_time, learning_rates, losses, sim_losses, rank_losses]
    else:
        meters = [batch_time, data_time, learning_rates, losses]

    if args.moe_contrastive_weight > 0:
        losses_gate = AverageMeter('Loss_gate', ':.4e')
        meters.append(losses_gate)

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Epoch: [{}]".format(epoch),
        log=log)

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (sample, _) in enumerate(train_loader):
        if args.moe_iou_gate:
            images, bboxs = sample
        else:
            images = sample
            bboxs = None

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        assert args.local_rank is not None

        images[0] = images[0].cuda(args.local_rank, non_blocking=True)
        images[1] = images[1].cuda(args.local_rank, non_blocking=True)

        assert not args.moe_use_gate
        x3 = None

        if args.local_crops_number > 0:
            x_local = [img.cuda(args.local_rank, non_blocking=True) for img in images[2]]
        else:
            x_local = None

        with torch.cuda.amp.autocast(True):
            if args.moe_use_gate:
                loss = model(images[0], images[1], moco_m, x3=x3)
            else:
                if args.moe_contrastive_weight > 0:
                    loss, q1_gate, q2_gate, k1_gate, k2_gate = model(images[0], images[1], moco_m, x_local=x_local)
                else:
                    assert x_local is None
                    loss = model(images[0], images[1], moco_m)

        if args.sim_rank:
            loss, rank_l, similarity_l = loss
            sim_losses.update(similarity_l.item(), images[0].size(0))
            rank_losses.update(rank_l.item(), images[0].size(0))

        if args.arch.startswith('moe_vit'):
            loss += collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.moe_contrastive_weight > 0:
            if isinstance(q1_gate, list) and (wassertein_distance_metric is None):
                loss_gate = 0
                for q1_g, q2_g, k1_g, k2_g in zip(q1_gate, q2_gate, k1_gate, k2_gate):
                    bboxs = bboxs[:2] if bboxs is not None else None
                    loss_gate += args.moe_contrastive_weight * model.module.gate_contrastive_loss(q1_g, q2_g, k1_g, k2_g,
                                                                                                  wassertein_distance_metric,
                                                                                                  wassertein_distance_metric_neg,
                                                                                                  bboxes=bboxs)
            else:
                assert not args.moe_iou_gate
                loss_gate = args.moe_contrastive_weight * model.module.gate_contrastive_loss(q1_gate, q2_gate, k1_gate,
                                                                                             k2_gate,
                                                                                             wassertein_distance_metric,
                                                                                             wassertein_distance_metric_neg)
            losses_gate.update(loss_gate.item(), images[0].size(0))
            loss = loss + loss_gate

            if args.moe_wassertein_gate:
                wassertein_distances.update(wassertein_distance_metric.last_target)
                wassertein_distances_gp.update(wassertein_distance_metric.last_gp)
                wassertein_distances_neg.update(wassertein_distance_metric_neg.last_target)
                wassertein_distances_gp_neg.update(wassertein_distance_metric_neg.last_gp)

        scaler.scale(loss).backward()

        if args.arch.startswith('moe_vit') and (not args.moe_data_distributed):
            model.allreduce_params()

        if args.moe_contrastive_weight:
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log = log

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    if "moe" in args.arch:
        for param_group in optimizer.param_groups:
            if param_group["name"] == "moe_mlps":
                param_group['lr'] = lr
            elif param_group["name"] == "other":
                param_group['lr'] = lr * args.experts_lr_ratio
            else:
                raise ValueError("Unknown parameter group name of {}".format(param_group["name"]))
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
