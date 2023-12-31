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
        if schemes == 1:  # 直流编码
            self.encoding = self.current_coding

        if schemes == 2:  # 泊松编码
            self.encoding = self.Poisson_coding

        if schemes == 3:  # Rate-Syn coding
            self.encoding = self.Rate_Syn

        if schemes == 4:  # TTFS coding (Time-to-first-spike coding)
            self.encoding = self.TTFS

    def forward(self, x):
        return self.encoding(x)  # (N, C, H, W, T)

    @torch.no_grad()
    def current_coding(self, x):
        # necessary for general dataset: broadcast input
        x, _ = torch.broadcast_tensors(x,
                                       torch.zeros((self.time_window,) + x.shape, device=x.device))  # 函数无法在最后一个维度上进行广播
        return x.permute(1, 2, 3, 4, 0).detach()

    @torch.no_grad()
    def Poisson_coding(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        for step in range(self.time_window):
            x_[..., step] = x > torch.rand(x.shape, device=x.device)
        return x_.detach()

    @torch.no_grad()
    def Rate_Syn(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        t = ((1 - x) * self.time_window).round()
        for step in range(self.time_window):
            x_step = torch.where(step >= t, 1., 0.)
            x_[..., step] = x_step
        return x_.detach()

    @torch.no_grad()
    def TTFS(self, x):
        x_ = torch.zeros(x.shape + (self.time_window,), device=x.device)
        t = ((1 - x) * self.time_window).round()
        for step in range(self.time_window):
            x_step = torch.where(step == t, 1., 0.)
            x_[..., step] = x_step
        return x_.detach()


@torch.no_grad()
def rate(x, time_window=5):
    """
    reference: https://doi.org/10.1113/jphysiol.1926.sp002308.
    Rate Coding(Poisson_coding)
    :param x: 输入张量
    :return:
    描述：
        速率编码主要基于尖峰计数，以保证时间窗口内发出的尖峰数量与真实值相对应。
        泊松分布可以描述单位时间内发生的随机事件的数量，它对应于放电率。
    """
    shape = x.shape + (time_window,)
    x, _ = torch.broadcast_tensors(x,
                                   torch.zeros((time_window,) + x.shape, device=x.device))
    num_dims = len(shape)
    dim_order = list(range(1, num_dims)) + [0]
    x = x.permute(*dim_order)
    # print(x)
    return (x > torch.rand(shape, device=x.device)).float()


@torch.no_grad()
def Poisson(x, dt=0.1):
    """
    以 ms 为单位步进, 输入的值为刺激强度（泊松过程的平均放电率, HZ），
    args:
        x: 输入的图像值（放电强度，如255）
        dt: 积分步长
    reference: doi: 10.3389/fncom.2015.00099
    """
    return (torch.rand(x.shape, device=x.device)<x*dt/1000).float()


if __name__ == "__main__":
    # encoder = encoder(3)
    # print(encoder(torch.tensor([.1, .2, .3])))
    # print(rate(torch.tensor([1, .5, .3])))
    print(Poisson(torch.tensor([1, 1000, .3]), dt=1))
