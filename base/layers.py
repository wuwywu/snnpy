# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : layers.py
# 将各种用于神经网络的`各个层`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class addTLayer(nn.Module):
    """
    将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。
    前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
    Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        :param layer (nn.Module): 需要转换的层。
                The layer needs to convert.
        :param time_window (int): 输入时间窗口。
    """
    def __init__(self, layer, time_window=5):
        super().__init__()
        self.layer = layer
        self.time_window = time_window

    def forward(self, x):
        # 输入数据(B, *) --> (B, *, T)
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (self.time_window,), device=x.device) # 
        for step in range(self.time_window):
            x_[..., step] = self.layer(x[..., step])
        return x_
    

class FRLayer(nn.Module):
    """
    计算放电率

    Args:
        :param if_T (bool): 数据流是否包含时间维度。
        :param if_grad (bool): 是否保留梯度。
        :param time_window (int): 输入时间窗口。
    """
    def __init__(self, if_T=False, if_grad=True, time_window=5):
        super().__init__()
        self.if_T = if_T                      # 是否添加时间维度
        self.if_grad = if_grad                # 是否保留梯度
        self.time_window = time_window        # 时间窗口
        self.count = 0                        # 时间循环计数
        self.out = None

    def forward(self, x):
        if not self.if_grad:
            x = x.clone().detch().cpu()
            
        if self.if_T:
            # 输入数据 (B, N, T)
            self.out = torch.sum(x, dim=-1)/self.time_window  
        else:
            # 输入数据 (B, -1)
            if self.count == 0:
                self.out = torch.zeros_like(x)
                self.out += x/self.time_window
                self.count += 1
            elif self.count >= self.time_window:
                self.count = 0
                self.out = None
            else:
                self.out += x/self.time_window
                self.count += 1
        if not self.if_grad:
            self.out = self.out.clone().detch().cpu()

        return self.out



    


    
