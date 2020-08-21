# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.networks import ResNetBuilder, IDE, Resnet, BFE, StrongBaseline
from trainers.evaluator import ResNetEvaluator
from trainers.trainer import cls_tripletTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin, MyCenterLoss
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from torchvision.models.resnet import resnet18

def train(**kwargs):
    opt._parse(kwargs)     ##设置程序的所有参数

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

    pin_memory = True if use_gpu else False


    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )
    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    if opt.model_name == 'softmax' or opt.model_name == 'softmax_triplet':
        model = ResNetBuilder(dataset.num_train_pids, 1, True)
    elif opt.model_name == 'triplet':
        model = ResNetBuilder(None, 1, True)
    elif opt.model_name == 'bfe':
        if opt.datatype == "person":
            model = BFE(dataset.num_train_pids, 1.0, 0.33)
        else:
            model = BFE(dataset.num_train_pids, 0.5, 0.5)
    elif opt.model_name == 'ide':
        model = IDE(dataset.num_train_pids)
    elif opt.model_name == 'resnet':
        model = Resnet(dataset.num_train_pids)
    elif opt.model_name=='strongBaseline':
        model=StrongBaseline(dataset.num_train_pids)

    optim_policy = model.get_optim_policy()

    # update model
    model=resnet18(True)
    model.fc.out_features=10


    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)


    #xent_criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyLabelSmooth(10)
    epochs=100
    best=0.0
    b_e=0
    for e in range(epochs):
        model.train()
        for i, inputs in enumerate(trainloader):
            imgs, pid, _=inputs
            imgs, pid=imgs.cuda(), pid.cuda()
            outputs=model(imgs)
            loss=criterion(outputs, pid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch=%s\tbatch loss=%s' ,e, loss)
        # val
        rank1=test(model, queryloader)
        print('epoch=%s \t rank1=%s')%(e, loss.item())
        if rank1>best:
            # save best
            b_e=e
            state_dict = model.module.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': e + 1},
                is_best=True, save_dir=opt.save_dir,
                filename='best' + '.pth.tar')
    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best, b_e))

def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu() 
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1 

if __name__ == '__main__':
   train()
