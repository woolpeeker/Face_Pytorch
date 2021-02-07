
import sys
sys.path.append('./tool_deploy/')
sys.path.append('./')
sys.path.append('../')
import face_model as face_model
import argparse
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from backbone.smallVGG import SmallVGG

CKPT_PATH = '../model/SMALLVGG/Iter_033000_net.ckpt'
DATA_ROOT = '/media/HD1/home/luojiapeng/Projects/Face_Pytorch/data/AI_IMAGES'
DET_GPU = -1

def get_data():
    cls_img_dict = {}
    data = np.loadtxt(DATA_ROOT+'.txt', dtype=str).tolist()
    for img, c in data:
        cls_img_dict.setdefault(c, []).append(Path(DATA_ROOT) / img)
    return cls_img_dict

if __name__ == '__main__':
    rec_model = SmallVGG(1, 256, alpha=0.5)
    rec_model.load_state_dict(torch.load(CKPT_PATH)['net_state_dict'])

    model = face_model.FaceModel(DET_GPU, rec_model, rec_model_in_channels=1)
    data = get_data()
    img_num = sum([len(v) for v in data.values()])
    M = np.zeros([img_num, img_num])
    rows = sum([v for v in data.values()], [])
    r, c = -1, -1
    dists = []
    labels = []
    for c0, imgs0 in data.items():
        for img0 in imgs0:
            r += 1
            print(img0)
            c = -1
            for c1, imgs1 in data.items():
                for img1 in imgs1:
                    c += 1
                    im0 = cv2.imread(str(img0))
                    im0 = model.get_input(im0)
                    f0 = model.get_feature(im0)

                    im1 = cv2.imread(str(img1))
                    im1 = model.get_input(im1)
                    f1 = model.get_feature(im1)
                    d = np.sum(np.square(f0 - f1))
                    dists.append(d)
                    labels.append(1 if c0==c1 else 0)
    dists = np.array(dists)
    min_d = dists.min()
    max_d = dists.max()
    score = np.exp( - (dists - min_d) / (max_d - min_d))
    fpr,tpr,threshold = roc_curve(labels, score)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')