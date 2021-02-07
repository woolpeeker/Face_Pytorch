#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: lfw.py.py
@time: 2018/12/22 10:00
@desc: lfw dataset loader
'''

import numpy as np
import cv2
import os
import torch.utils.data as data

import torch
import torchvision.transforms as transforms

def img_loader(path, use_gray):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if use_gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, -1)
            return img
    except IOError:
        print('Cannot load image ' + path)

class LFW(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader, use_gray=False):

        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.use_gray = use_gray
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            if '\t' in p:
                p = p.split('\t')
            else:
                p = p.split()
            if len(p) == 3:
                nameL = p[0]
                nameR = p[1]
                fold = i // 600
                flag = int(p[2])
            elif len(p) == 4:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                fold = i // 600
                flag = -1
            else:
                raise Exception('unkown data type for %s' % p)
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]), use_gray=self.use_gray)
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]), use_gray=self.use_gray)
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)


if __name__ == '__main__':
    root = 'D:/data/lfw_align_112'
    file_list = 'D:/data/pairs.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = LFW(root, file_list, transform=transform)
    #dataset = LFW(root, file_list)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        for d in data:
            print(d[0].shape)