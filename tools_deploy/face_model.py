from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import insightface
from insightface.utils import face_align
import torch


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])

class FaceModel:
    def __init__(self, ctx_id, model, use_large_detector=False, rec_model_in_channels=3):
        # image_size: (height, width, channels)
        if use_large_detector:
            self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        else:
            self.detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
        self.detector.prepare(ctx_id=ctx_id)
        if ctx_id>=0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        self.model = model.eval()
        self.rec_model_in_channels = rec_model_in_channels

    def get_input(self, face_img):
        bbox, pts5 = self.detector.detect(face_img, threshold=0.1)
        if bbox.shape[0]==0:
            return None
        bbox = bbox[0, 0:4]
        pts5 = pts5[0, :]
        nimg = face_align.norm_crop(face_img, pts5)
        return nimg

    def get_feature(self, aligned):
        a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        if self.rec_model_in_channels == 1:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            a = np.expand_dims(a, -1)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        input_blob = torch.tensor(input_blob).float() / 255
        emb = self.model(input_blob)[0].detach().cpu().numpy()
        norm = np.sqrt(np.sum(emb*emb)+0.00001)
        emb /= norm
        return emb

