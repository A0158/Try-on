import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np

import functools


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel,scale, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        if scale == 'same':
            self.scale = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv2d(in_channel, out_channel, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.res = nn.Sequential(
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias)
        )

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.res(residual))



    

class InceptionV3(nn.Module):
    def __init__(self, requires_grad=False):
        super(InceptionV3, self).__init__()
        inception_pretrained_features = models.inception_v3(pretrained=True).features
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        inception_pretrained_features = InceptionV3(include_top=False, weights=weights,
                                     input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), inception_pretrained_features[x])
        for x in range(2, 6):
            self.slice2.add_module(str(x), inception_pretrained_features[x])
        for x in range(6, 12):
            self.slice3.add_module(str(x), inception_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), inception_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), inception_pretrained_features[x])
        for x in range(30, 37):
            self.slice5.add_module(str(x), inception_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6]
        return out
    

class InceptionV3_Loss(nn.Module):
    def __init__(self, layids = None):
        super(InceptionV3, self).__init__()
        self.inception = InceptionV3()
        self.inception.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/64 ,1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_incep, y_incep = self.inception(x), self.inception(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_incep)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_incep[i], y_incep[i].detach())
        return loss

    
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops import array_ops

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()


    def modified_generator_loss(
        discriminator_gen_outputs,
        label_smoothing=0.0,
        weights=1.0,
        scope=None,
        loss_collection=ops.GraphKeys.LOSSES,
        reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,add_summaries=False):


        with ops.name_scope(scope, 'generator_modified_loss',
                            [discriminator_gen_outputs]) as scope:
        loss = losses.sigmoid_cross_entropy(
            array_ops.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs, weights, label_smoothing, scope,
            loss_collection, reduction)


        return loss

