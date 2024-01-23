import torchvision.transforms as transforms
import moco.loader

import os
import torch
from torch import nn

from moco.builder_simclr import SimCLR

from .utils import AverageMeter


def pretrain_transform(crop_min, with_gate_aug=False, local_crops_number=0):
    if local_crops_number > 0:
        raise ValueError("local_crops_number > 0 is not supported for standard transform")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if not with_gate_aug:
        train_transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                                        transforms.Compose(augmentation2))
        return train_transform
    else:
        transform_gate = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_transform = moco.loader.ThreeCropsTransform(transforms.Compose(augmentation1),
                                                          transforms.Compose(augmentation2),
                                                          transform_gate)
        return train_transform


def evaluate_pretrain(train_loader, model, args, log):
    losses = AverageMeter()
    rank_calculate = RankCalculator()

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if args.local_rank is not None:
                images[0] = images[0].cuda(args.local_rank, non_blocking=True)
                images[1] = images[1].cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                loss, q1, q2, k1, k2 = model(images[0], images[1], moco_m)

            losses.update(loss.item(), images[0].size(0))
            rank_calculate.update(q1, q2)
            print("q1 shape is {}".format(q1.shape))

            if i % 10 == 0:
                msg = "{}/{}, loss avg is {:.3f}".format(i, iters_per_epoch, losses.avg)
                log.info(msg)

    rank_calculate.cal_rank_save(log)


def evaluate_pretrain_simRank(train_loader, model, args, log):
    rank_losses = AverageMeter()
    similarity_losses = AverageMeter()
    rank_calculate = RankCalculator()

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if args.local_rank is not None:
                images[0] = images[0].cuda(args.local_rank, non_blocking=True)
                images[1] = images[1].cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                rank_l, similarity_l, q1, q2 = model(images[0], images[1], moco_m)

            rank_losses.update(rank_l.item(), images[0].size(0))
            similarity_losses.update(similarity_l.item(), images[0].size(0))
            rank_calculate.update(q1, q2)

            if i % 10 == 0:
                msg = "{}/{}, rank loss avg is {:.3f}, similarity loss avg is {:.3f}".format(i, iters_per_epoch, rank_losses.avg, similarity_losses.avg)
                log.info(msg)

    rank_calculate.cal_rank_save(log)


class RankCalculator(object):
    def __init__(self):
        self.q_comb = None
        self.update_cnt = 0
        pass

    def update(self, q1, q2):
        q_comb = torch.mm(q1.permute(1,0), q2)
        if self.q_comb is None:
            self.q_comb = q_comb / q1.shape[0]
            self.update_cnt += q1.shape[0]
        else:
            self.q_comb = self.q_comb * (self.update_cnt / (self.update_cnt + q1.shape[0])) + \
                          q_comb      * (q1.shape[0] / (self.update_cnt + q1.shape[0]))
            self.update_cnt += q1.shape[0]


    def cal_rank_save(self, log):

        tol_thresholds = [0, 1e-5, 1e-4, 1e-3, 0.001, 0.01, 0.1, 0.2]

        log.info("q_comb norm is {}".format(torch.norm(self.q_comb)))
        for tol_thre in tol_thresholds:
            log.info("For tol_thre of {}, q_comb rank is {}".format(tol_thre, torch.linalg.matrix_rank(self.q_comb.float(), tol=tol_thre)))

        if torch.distributed.get_rank() == 0:
            torch.save(self.q_comb, os.path.join(log.path, "q_comb.pth"))
