# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : encoder.py
# 将各种用于神经网络的`编码方式`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class encoder(nn.Module):
    """
    四种编码方式集合
    参考文献：Spike timing reshapes robustness against attacks in spiking neural networks
    Args:
        :param schemes: 选着编码方式
            1、直流编码  2、泊松编码  3、Rate-Syn coding  4、TTFS coding (Time-to-first-spike coding)
        :param time_window: 静息电位
    return:
        数据维度: (N, C, H, W, T)
    """
    def __init__(self, schemes=1, time_window=5):
        super().__init__()
        self.time_window = time_window
        if schemes==1:  # 直流编码
            self.encoding = self.current_coding

        if schemes==2:  # 泊松编码
            self.encoding = self.Poisson_coding

        if schemes==3:  # Rate-Syn coding
            self.encoding = self.Rate_Syn

        if schemes==4:  # TTFS coding (Time-to-first-spike coding)
            self.encoding = self.TTFS
 
    def forward(self, x):
        return self.encoding(x) # (N, C, H, W, T)

    def current_coding(self, x):
        # necessary for general dataset: broadcast input
        x, _ = torch.broadcast_tensors(x, torch.zeros( (self.time_window,) + x.shape, device=x.device))  # 函数无法在最后一个维度上进行广播
        return x.permute(1, 2, 3, 4, 0).detach()

    def Poisson_coding(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        for step in range(self.time_window):
            x_[...,step] = x > torch.rand(x.shape, device=x.device)
        return x_.detach()
    
    def Rate_Syn(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        t = ((1-x)*self.time_window).int()
        for step in range(self.time_window):
            x_step = torch.where(step>=t, 1., 0.)
            x_[...,step] = x_step
        return x_.detach()

    def TTFS(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        t = ((1-x)*self.time_window).int()
        for step in range(self.time_window):
            x_step = torch.where(step==t, 1., 0.)
            x_[...,step] = x_step
        return x_.detach()
    

