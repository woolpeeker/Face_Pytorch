#!/usr/bin/env python
# encoding: utf-8
'''
forked from train.py
'''
import sys
sys.path.append('./qqquantize')
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from backbone.mobilefacenet import MobileFaceNet
from backbone.cbam import CBAMResNet
from backbone.attention import ResidualAttentionNet_56, ResidualAttentionNet_92
from backbone.smallVGG import SmallVGG, CRB
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.InnerProduct import InnerProduct
from utils.logging import init_log
from dataset.casia_webface import CASIAWebFace
from dataset.lfw import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
from eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from utils.model_summary import calc_flops
import copy
import qqquantize
from qqquantize import utils as qutils
import qqquantize
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
from qqquantize.quantize import ModelConverter
from qqquantize.observers.histogramobserver import HistObserver, swap_minmax_to_hist
from qqquantize.observers.minmaxchannelsobserver import MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver
from qqquantize.observers.minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver
from qqquantize.observers.fake_quantize import (
    FakeQuantize,
    Fake_quantize_per_channel,
    Fake_quantize_per_tensor,
    enable_fake_quant,
    disable_fake_quant,
    enable_observer,
    disable_observer,
    enable_calc_qparams,
    disable_calc_qparams,
    calc_qparams,
)
from easydict import EasyDict as edict

def fuse_bn(model, inplace=False):
    fuse_conv_bn = qutils.fuse_conv_bn
    if not inplace:
        model = copy.deepcopy(model)
    for mod in model.modules():
        if isinstance(mod, CRB):
            mod.conv = fuse_conv_bn(mod.conv, mod.bn)
            mod.bn = nn.Identity()
    return model

def inference_for_quant(net, margin, device, trainset, bs=128, num_workers=8):
    loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                        shuffle=True, num_workers=num_workers)
    net.eval()
    correct = 0
    total = 0
    for i, data in enumerate(loader):
        if i > 1000:
            break
        img, label = data[0].to(device), data[1].to(device)
        raw_logits = net(img)
        output = margin(raw_logits, label)
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
    print("inference_for_quant, train_accuracy: {:.4f}, ".format(correct/total))
    net.train()

def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    assert not multi_gpus, "not sure qqquantize works well when multi gpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(args.save_dir, args.model_pre + args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    if not args.use_gray:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])
        
    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, transform=transform, use_gray=args.use_gray)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    # test dataset
    lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, transform=transform, use_gray=args.use_gray)
    lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=128,
                                             shuffle=False, num_workers=4, drop_last=False)

    # define backbone and margin layer
    in_channels = 1 if args.use_gray else 3
    if args.backbone == 'MobileFace':
        net = MobileFaceNet()
    elif args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Attention_56':
        net = ResidualAttentionNet_56(feature_dim=args.feature_dim)
    elif args.backbone == 'Attention_92':
        net = ResidualAttentionNet_92(feature_dim=args.feature_dim)
    elif args.backbone == 'SmallVGG':
        net = SmallVGG(in_channels, args.feature_dim, alpha=0.5)
    else:
        print(args.backbone, ' is not available!')
        exit(-1)
    calc_flops(net, in_channels, 112, 112)

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'MultiMargin':
        margin = MultiMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'Softmax':
        margin = InnerProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'SphereFace':
        pass
    else:
        print(args.margin_type, 'is not available!')
        exit(-1)

    assert args.resume, 'train_quant must resume'
    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])
        lr = 0.001
    else:
        lr = 0.1
    
    fuse_bn(net, True)
    qconfig = edict({
        'activation': FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=8
        ),
        'weight': FakeQuantize.with_args(
            observer=MinMaxObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=8
        ),
        'bias': FakeQuantize.with_args(bits=8),
    })
    # model = model.fuse()
    qconverter = ModelConverter(qconfig)
    net = qconverter(net)

    
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=lr, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 11, 16], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)
    
    net.eval()
    disable_fake_quant(net)
    disable_calc_qparams(net)
    enable_observer(net)
    inference_for_quant(net, margin, device, trainset)
    calc_qparams(net)

    enable_observer(net)
    enable_fake_quant(net)
    inference_for_quant(net, margin, device, trainset)
    enable_calc_qparams(net)


    print(net)

    best_lfw_acc = 0.0
    best_lfw_iters = 0
    total_iters = 0
    for epoch in range(1, args.total_epoch + 1):
        if epoch > 1:
            exp_lr_scheduler.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        since = time.time()
        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()

            raw_logits = net(img)
            output = margin(raw_logits, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_last_lr()[0]))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:

                # test model on lfw
                net.eval()
                getFeatureFromTorch('./result/cur_lfw_result.mat', net, device, lfwdataset, lfwloader)
                lfw_accs = evaluation_10_fold('./result/cur_lfw_result.mat')
                _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(lfw_accs) * 100))
                if best_lfw_acc <= np.mean(lfw_accs) * 100:
                    best_lfw_acc = np.mean(lfw_accs) * 100
                    best_lfw_iters = total_iters

                net.train()

    _print('Finally Best Accuracy: LFW: {:.4f} in iters: {}'.format(
        best_lfw_acc, best_lfw_iters))
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--use-gray', action='store_true', help='use gray image')
    parser.add_argument('--train_root', type=str, default='/media/HD1/datasets/webface_align_112/webface_align_112', help='train image root')
    parser.add_argument('--train_file_list', type=str, default='/media/HD1/datasets/webface_align_112/labels.txt', help='train list')
    parser.add_argument('--lfw_test_root', type=str, default='/media/HD1/datasets/faces_emore/lfw', help='lfw image root')
    parser.add_argument('--lfw_file_list', type=str, default='/media/HD1/datasets/faces_emore/lfw_pair.txt', help='lfw pair file list')

    parser.add_argument('--backbone', type=str, default='SmallVGG', help='MobileFace, Res50_IR, SERes50_IR, Res100_IR, SERes100_IR, Attention_56, Attention_92')
    parser.add_argument('--margin_type', type=str, default='Softmax', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=256, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=20, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=1000, help='test frequency')
    parser.add_argument('--resume', action='store_true', help='resume model')
    parser.add_argument('--net_path', type=str, default='model/SMALLVGG/Iter_033000_net.ckpt', help='resume model')
    parser.add_argument('--margin_path', type=str, default='model/SMALLVGG/Iter_033000_margin.ckpt', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='quant_', help='model prefix')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    train(args)


