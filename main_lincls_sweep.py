#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
import shutil
import time
import warnings
import fmoe
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as torchvision_models

from models import vits_gate, vits_moe, vits, vits_sep

from models.vits_gate import VisionTransformerMoCoWithGate

from utils.utils import logger, accuracy, sync_weights
from utils.moe_utils import read_specific_group_experts, collect_moe_model_state_dict, save_checkpoint, \
    collect_noisy_gating_loss, prune_moe_experts, set_moe_layer_train_mode
from utils.init_datasets import init_datasets
from utils.utils import DistillCrossEntropy
from utils.speed_test import speed_test

from functools import partial
from pdb import set_trace

from thop_modified import profile
from models.vision_transformer import matmul, count_matmul
from utils.thop_moe import THOP_DICT

from utils_mae.misc import MetricLogger
import itertools


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
model_names += ['moe_vit_small', 'moe_vit_base']
model_names += ['sep_vit_small']

gate_names = ['', 'vit_gate_small', 'vit_gate_base', 'vit_gate_large']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', type=str, default="checkpoints_moco")

parser.add_argument('--data', metavar='DIR', default="", help='path to dataset')
parser.add_argument('--dataset', default="imagenet", help='dataset')
parser.add_argument('--customSplit', type=str, default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')

# moe configs:
parser.add_argument('--moe-data-distributed', action='store_true', help='if employ moe data distributed')
parser.add_argument('--moe-experts', default=4, type=int, help='moe experts number')
parser.add_argument('--moe-mlp-ratio', default=-1, type=int, help='moe dim')
parser.add_argument('--moe-top-k', default=2, type=int, help='moe top k')
parser.add_argument('--moe-noisy-gate-loss-weight', default=0.01, type=float)
parser.add_argument('--moe-gate-arch', default="", choices=gate_names, type=str)
parser.add_argument('--moe-experts-prune', default=-1, type=int, help="if n > 0, prune the experts until there is only n experts")
parser.add_argument('--moe-noisy-train', action='store_true', help="if employ noise for linear evaluating moe")
parser.add_argument('--moe-gate-type', default="noisy", type=str)
parser.add_argument('--vmoe-noisy-std', default=0, type=float)
parser.add_argument('--moe-same-for-all', action='store_true', help="if employ the same gate for all")

# vit sep configs:
parser.add_argument('--sep-path', default=0, type=int, help="select sep patch")

parser.add_argument('--fine-tune', action='store_true', help='moe experts number')
parser.add_argument('--test-interval', type=int, default=1, help='moe experts number')

# options for distillation
parser.add_argument('--distillation', action='store_true', help='if use distillation')
parser.add_argument('--distillation_checkpoint', default="", type=str)
parser.add_argument('--distillation_temp', default=0.1, type=float)

# options for speed testing
parser.add_argument('--speed_test', action='store_true', help='if test the speed')
parser.add_argument('--profile_model', action='store_true', help='if profile the model')

best_acc1 = 0


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


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    logName = "log.txt"
    save_dir = os.path.join(args.save_dir, args.experiment)
    if not os.path.exists(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir, log_name=logName)

    main_worker(args.local_rank, args, log)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def main_worker(local_rank, args, log):
    global best_acc1
    args.local_rank = local_rank

    # suppress printing if not master
    if args.multiprocessing_distributed and args.local_rank != 0:
        # def print_pass(*args):
        #     pass
        # builtins.print = print_pass
        log.local_rank = 1

    log.info(str(args))

    if args.local_rank is not None:
        log.info("Use GPU: {} for training".format(args.local_rank))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        torch.distributed.barrier()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_datasets, val_datasets, test_datasets = init_datasets(args, transform_train, transform_test)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    else:
        train_sampler = None

    # print("args.batch_size is {}".format(args.batch_size))
    batch_size = int(args.batch_size / torch.distributed.get_world_size())
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # Enabling distributed evaluation with an eval dataset not divisible by process number. '
    # 'This will slightly alter validation results as extra duplicate entries are added to achieve '
    # 'equal num of samples per-process, so we only do this for validation
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_datasets)
    val_loader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nb_classes = len(np.unique(train_datasets.targets))
    # print("nb_classes is {}".format(nb_classes))

    args.rank = torch.distributed.get_rank()
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    args.moe_use_gate = (args.moe_gate_arch != "")
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'head'
    elif args.arch.startswith('sep_vit'):
        model = vits_sep.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'head'
        model.set_block_path(args.sep_path)
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
                                                 num_classes=nb_classes)
            model = VisionTransformerMoCoWithGate(model, gate_model)
        else:
            model = vits_moe.__dict__[args.arch](moe_mlp_ratio=args.moe_mlp_ratio, moe_experts=args.moe_experts, moe_top_k=args.moe_top_k,
                                                 world_size=moe_world_size,
                                                 num_classes=nb_classes,
                                                 moe_gate_type=args.moe_gate_type, vmoe_noisy_std=args.vmoe_noisy_std,
                                                 moe_same_for_all=args.moe_same_for_all)
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'fc'

    assert not args.distillation
    model_teacher = None
    if args.distillation:
        model_teacher = copy.deepcopy(model)

    assert not args.fine_tune
    log.info("Conduct linear evaluation")
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    print(model)

    # summary the flops, params
    if args.profile_model:
        if "moe" in args.arch:
            assert args.moe_data_distributed
        thop_dict = {matmul: count_matmul}
        thop_dict.update(THOP_DICT)
        flops, params = profile(model.cuda(), inputs=(torch.randn(1, 3, 224, 224, device="cuda"),),
                                custom_ops=thop_dict, verbose=False)
        flops /= 10 ** 9
        params /= 10 ** 6
        log.info("base encoder flops: {:.04}G, params {:.04}M".format(flops, params))
        return

    feat_dim = model.head.in_features
    model.head = torch.nn.Identity()

    args.lrs = [base * scale for scale in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10] for base in [1, 3, 5, 7, 9]]
    # args.lrs = [base * scale for scale in [ 1e-1, ] for base in [1,]]
    args.wds = [0, 1e-6]
    args.optims = ['sgd']

    args.permutes = list(itertools.product(args.lrs, args.wds, args.optims))


    linear_classifiers = nn.ModuleList()
    optimizers = []
    schedulers = []
    # print("barrier")
    # torch.distributed.barrier()
    # print("after barrier")
    # print("args.local_rank is {}".format(args.local_rank))
    for pm in args.permutes:
        lr, wd, _ = pm
        linear_classifier = LinearClassifier(feat_dim, num_labels=nb_classes)
        linear_classifier = linear_classifier.to(args.local_rank)
        linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.local_rank])
        linear_classifiers.append(linear_classifier)

        # set optimizer
        parameters = linear_classifier.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            lr * args.batch_size * torch.distributed.get_world_size() / 256.,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        optimizers.append(optimizer)
        schedulers.append(scheduler)

    print("set linear classifiers")

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained) or os.path.isdir(args.pretrained):
            log.info("=> loading checkpoint '{}'".format(args.pretrained))
            if os.path.isfile(args.pretrained):
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                moe_dir_read = False
            elif os.path.isdir(args.pretrained):
                checkpoint = torch.load(os.path.join(args.pretrained, "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
                len_save = len([f for f in os.listdir(args.pretrained) if "pth" in f])
                if args.moe_data_distributed:
                    response_cnt = [i for i in range(len_save)]
                else:
                    assert len_save % torch.distributed.get_world_size() == 0
                    response_cnt = [i for i in range(torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                                                     (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
                # merge all ckpts
                # print("rank {}, response_cnt is {}".format(local_rank, response_cnt))
                for cnt, cnt_model in enumerate(response_cnt):
                    if cnt_model != 0:
                        checkpoint_specific = torch.load(os.path.join(args.pretrained, "{}.pth".format(cnt_model)), map_location="cpu")
                        if cnt != 0:
                            for key, item in checkpoint_specific["state_dict"].items():
                                checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item], dim=0)
                        else:
                            checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                    moe_dir_read = True
            else:
                raise ValueError("Model {} do not exist".format(args.pretrained))

            if "mae" in args.pretrained and "model" in checkpoint:
                state_dict = checkpoint["model"]
                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                print(msg)
                # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            elif args.moe_use_gate:
                state_dict = checkpoint['state_dict']
                model = cvt_state_dict_moe_gate(state_dict, model, args, linear_keyword)
            else:
                state_dict = checkpoint['state_dict']
                model = cvt_state_dict(state_dict, model, args, linear_keyword, moe_dir_read)

            log.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.pretrained))


    if not torch.cuda.is_available():
        raise NotImplementedError()
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            if model_teacher is not None:
                model_teacher.cuda(args.local_rank)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / torch.distributed.get_world_size())
            args.workers = args.workers
            assert args.moe_data_distributed

            if "moe" in args.arch and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model, device_ids=[args.local_rank])
                sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
                if model_teacher is not None:
                    model_teacher = fmoe.DistributedGroupedDataParallel(model_teacher, device_ids=[args.local_rank])
                    sync_weights(model_teacher, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])

            # else:
            #     linear_classifiers = torch.nn.parallel.DistributedDataParallel(linear_classifiers, device_ids=[args.local_rank])
            #     if model_teacher is not None:
            #         model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.local_rank],
            #                                                                   find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if "moe" in args.arch and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)

            if model_teacher is not None:
                model_teacher.cuda()
                if "moe" in args.arch and (not args.moe_data_distributed):
                    model_teacher = fmoe.DistributedGroupedDataParallel(model_teacher)
                else:
                    model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher)
    elif args.local_rank is not None:
        raise NotImplementedError()
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
    else:
        raise NotImplementedError()
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # if "moe" in args.arch:
    #     print("after distributed, rank {}, block1 h4toh avg is {}".format(args.local_rank,
    #                                                    model.module.blocks[1].mlp.experts.h4toh.weight.data.mean(-1).mean(-1)))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)

    # parameters = list(filter(lambda p: p.requires_grad, linear_classifiers[0].parameters()))
    # if not args.fine_tune:
    #     # optimize only the linear classifier
    #     assert len(parameters) == 2  # weight, bias

    # optimizer = torch.optim.SGD(parameters, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    assert not args.resume
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         log.info("=> loading checkpoint '{}'".format(args.resume))
    #         if args.local_rank is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single local_rank.
    #             loc = 'cuda:{}'.format(args.local_rank)
    #             checkpoint = torch.load(args.resume, map_location="cpu")
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.local_rank is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.local_rank)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         log.info("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         log.info("=> no checkpoint found at '{}'".format(args.resume))

    if model_teacher is not None:
        assert args.distillation_checkpoint != ""
        checkpoint_distill = torch.load(args.distillation_checkpoint, map_location="cpu")

        if args.moe_use_gate:
            state_dict = read_specific_group_experts(checkpoint_distill['state_dict'], args.local_rank, args.moe_experts)
            model_teacher.load_state_dict(state_dict)
        else:
            model_teacher.load_state_dict(checkpoint_distill["state_dict"])
        model_teacher = model_teacher.cuda()
        for param in model_teacher.parameters():
            param.requires_grad = False

    if args.moe_experts_prune > 0:
        assert args.pretrained
        prune_moe_experts(model, train_loader, log, args.moe_experts_prune)

    if args.evaluate:
        # load best model for testing
        checkpoint = torch.load(os.path.join(log.path, 'model_best.pth.tar'), map_location="cpu")
        if "moe" in args.arch and (not args.moe_data_distributed):
            state_dict = read_specific_group_experts(checkpoint['state_dict'], args.local_rank, args.moe_experts)
        else:
            state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        top_1_avg = validate(test_loader, model, criterion, args, log, prefix="Test: ")
        log.info("Top 1 acc for best model is {}".format(top_1_avg))
        return

    if model_teacher is not None:
        top_1_avg = validate(test_loader, model_teacher, criterion, args, log, prefix="Test Teacher: ")
        log.info("Top 1 acc for teacher model is {}".format(top_1_avg))

    torch.cuda.empty_cache()

    if args.speed_test:
        speed_test(train_loader, model, args, log)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)
        # log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        # train for one epoch
        train(train_loader, model, linear_classifiers, criterion, optimizers, epoch, args, log, model_teacher=model_teacher)

        for scheduler in schedulers:
            scheduler.step()

        if epoch % args.test_interval == 0:
            # evaluate on validation set
            test_stats = validate(val_loader, model, linear_classifiers, args.permutes, criterion, args, log)
            group_best_acc = 0
            group_best_acc_hidx = 0
            group_best_acc_sweep_lr_only = 0
            for group, pm in enumerate(args.permutes):
                if group % (len(args.wds) * len(args.optims)) == 0:
                    group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
                # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
                if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                    group_best_acc_hidx = group
                    group_best_acc = test_stats['acc{}@1'.format(group)]

            log.info(f"Accuracy of the network on the {len(val_datasets)} val images: {group_best_acc:.1f}%")

            # remember best acc@1 and save checkpoint
            is_best = group_best_acc > best_acc1
            best_acc1 = max(group_best_acc, best_acc1)

            moe_save = False
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0) or moe_save: # only the first GPU saves checkpoint
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                    "best_acc": group_best_acc,
                    'best_acc_hidx': group_best_acc_hidx,
                    "best_acc_sweep_lr_only": group_best_acc_sweep_lr_only,
                }
                save_checkpoint(save_dict, is_best, save_dir=log.path, moe_save=moe_save)
                # if epoch == args.start_epoch and (not args.fine_tune) and (not args.moe_use_gate) and (not "mae" in args.pretrained) and (not moe_save):
                #     sanity_check(save_state_dict, args.pretrained, linear_keyword, log)

        torch.distributed.barrier()

    torch.cuda.empty_cache()
    # load best model for testing
    checkpoint = torch.load(os.path.join(log.path, 'model_best.pth.tar'), map_location="cpu")
    state_dict = checkpoint['state_dict']
    linear_classifiers.load_state_dict(state_dict)
    test_stats = validate(test_loader, model, linear_classifiers, args.permutes, criterion, args, log)

    group_best_acc = 0
    group_best_acc_hidx = 0
    group_best_acc_sweep_lr_only = 0
    for group, pm in enumerate(args.permutes):
        if group % (len(args.wds) * len(args.optims)) == 0:
            group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
        # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
        if test_stats['acc{}@1'.format(group)] >= group_best_acc:
            group_best_acc_hidx = group
            group_best_acc = test_stats['acc{}@1'.format(group)]
    log.info(f"Final Test Accuracy of the network on the {len(test_datasets)} test images: {group_best_acc:.1f}%")
    log.info("Best test acc is in lr {} wd {} optim {}".format(*args.permutes[group_best_acc_hidx]))


def train(train_loader, model, linear_classifiers, criterion, optimizers, epoch, args, log, model_teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    metric_logger = MetricLogger(delimiter="  ", log=log)
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch),
    #     log=log)

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if not args.fine_tune:
        model.eval()
        if args.moe_noisy_train:
            set_moe_layer_train_mode(model)
    else:
        model.train()

    end = time.time()
    header = 'Epoch: [{}]'.format(epoch)
    for i, (images, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        if model_teacher is not None:
            with torch.no_grad():
                teacher_logits = model_teacher.eval()(images)

        # compute output
        features = model(images)

        losses = []
        for linear_classifier, optimizer in zip(linear_classifiers, optimizers):
            pred = linear_classifier(features)
            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(pred, target)
            optimizer.zero_grad()
            loss.backward()
            # step
            optimizer.step()
            losses.append(loss.item())

        torch.cuda.synchronize()
        # for group, (loss, optimizer) in enumerate(zip(losses, optimizers)):
        #     metric_logger.update(**{'loss{}'.format(group): loss})
        #     metric_logger.update(**{'lr{}'.format(group): optimizer.param_groups[0]["lr"]})

        metric_logger.update(loss_min=np.min(losses))

        # if model_teacher is None:
        #     loss = criterion(output, target)
        # else:
        #     loss = DistillCrossEntropy(T=args.distillation_temp)(output, teacher_logits)

        if args.arch.startswith('moe_vit'):
            collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()

        if args.arch.startswith('moe_vit') and (not args.moe_data_distributed):
            model.allreduce_params()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info("Averaged stats: {}".format(metric_logger))


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

def validate(val_loader, model, linear_classifiers, permutes, criterion, args, log, prefix="Validation: "):
    # switch to evaluate mode
    model.eval()
    linear_classifiers.eval()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(metric_logger.log_every(val_loader, 50, header)):
            if args.local_rank is not None:
                images = images.cuda(args.local_rank, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            output = model(images)
            losses = []
            acc1s = []
            acc5s = []
            for group, linear_classifier in enumerate(linear_classifiers):

                pred = linear_classifier(output)
                loss = nn.CrossEntropyLoss()(pred, target)
                losses.append(loss)

                acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                acc1s.append(acc1)
                acc5s.append(acc5)

                batch_size = images.shape[0]
                metric_logger.update(**{'loss{}'.format(group): loss.item()})
                metric_logger.meters['acc{}@1'.format(group)].update(acc1.item(), n=batch_size)
                if linear_classifier.module.num_labels >= 5:
                    metric_logger.meters['acc{}@5'.format(group)].update(acc5.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        log_msg = ""
        for group, (pm, linear_classifier) in enumerate(zip(permutes, linear_classifiers)):
            lr, wd, optim = pm
            if linear_classifier.module.num_labels >= 5:
                log_msg += '* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}\n'\
                           .format(lr=lr, wd=wd, optim=optim,
                        top1=metric_logger.meters['acc{}@1'.format(group)],
                        top5=metric_logger.meters['acc{}@5'.format(group)],
                        losses=metric_logger.meters['loss{}'.format(group)])
            else:
                log_msg += '* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}\n'\
                           .format(lr=lr, wd=wd, optim=optim,
                                top1=metric_logger.meters['acc{}@1'.format(group)],
                                losses=metric_logger.meters['loss{}'.format(group)])

        log.info(log_msg)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sanity_check(state_dict, pretrained_weights, linear_keyword, log):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    log.info("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    log.info("=> sanity check passed.")


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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cvt_state_dict(state_dict, model, args, linear_keyword, moe_dir_mode=False):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            if "_aux" in k:
                # print("skip k is {}".format(k))
                continue
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if "moe" in args.arch and (not args.moe_data_distributed) and (not moe_dir_mode):
        # if args.local_rank == 0:
        #     print("state dict block1 h4toh avg is {}".format(state_dict["blocks.1.mlp.experts.h4toh.weight"].mean(-1).mean(-1)))
        state_dict = read_specific_group_experts(state_dict, args.rank, args.moe_experts)

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    return model


def cvt_state_dict_moe_gate(state_dict, model, args, linear_keyword):
    # rename moco pre-trained keys
    from collections import OrderedDict
    feature_state_dict = OrderedDict()
    gate_state_dict = OrderedDict()
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            feature_state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        if k.startswith('module.gate_model'):
            gate_state_dict[k[len("module.gate_model."):]] = state_dict[k]

    feature_state_dict = read_specific_group_experts(feature_state_dict, args.rank, args.moe_experts)

    args.start_epoch = 0
    model.vit_feature.load_state_dict(feature_state_dict, strict=True)
    model.vit_gate.load_state_dict(gate_state_dict, strict=True)

    return model

if __name__ == '__main__':
    main()
