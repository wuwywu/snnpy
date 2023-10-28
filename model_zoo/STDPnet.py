# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/26
# User      : WuY
# File      : STDPnet.py
# paper     : Yiting Dong, An unsupervised STDP-based spiking neural network inspired by biologically plausible learning rules and connections
# doi       : https://doi.org/10.1016/j.neunet.2023.06.019
"""
文章中的网络结构
    卷积层(neuron) --> 2x2最大池化层 --> 尖峰归一化层 --> 全连接层(neuron)
    convolutional layer --> 2*2 max pooling layer --> spiking
normalization layer -->  fully connected layer
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse

from base.nodes import LIFSTDP
from base.layers import addTLayer, FRLayer, VotingLayer, WTALayer, LateralInhibition
from base.encoder import encoder
from datasets.datasets import mnist, fashion_MNIST, cifar10
from base.utils.utils import lr_scheduler, calc_correct_total, setup_seed
from base_module import BaseModule

from base.nodes import LIFbackEI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 固定随机种子
setup_seed(110)

parser = argparse.ArgumentParser(description="STDP框架研究")

parser.add_argument('--batch', type=int, default=100, help='批次大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--epoch', type=int, default=100, help='学习周期')
parser.add_argument('--time_window', type=int, default=10, help='LIF时间窗口')

args = parser.parse_args()

# STDP的卷积层+neuron
class STDPConv(nn.Module):
    """
    STDP更新权重的卷积层
    网络结构:
        1、卷积; 2、LIF(spiking neuron);
    网络中的构造：
        1、赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
        2、适应性阈值平衡(Adaptive threshold balance, ATB)
        3、适应性突触滤波器(Adaptive synaptic filter, ASF)
        ASF根据阈值计算，所以必须跟ATB一起进行
    args:
        :params
        in_planes: 卷积层输入特征
        out_planes: 卷积层输出特征
        kernel_size: 卷积核大小
        stride: 卷积的步长
        padding: 卷积的填充
        groups: 批次分组
        decay: LIF的衰减因子
        decay_trace: STDP计算trace时的衰减因子
        offset: 计算梯度时与trace相减的偏置
        inh: 侧抑制的抑制率(mode="threshold", 自适应性阈值)
        time_window: 时间窗口
    """
    def __init__(self, in_planes, out_planes,
                 kernel_size, stride, padding, groups,
                 decay=0.2, decay_trace=0.99, offset=0.3,
                 inh=1.625, time_window=10):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.lif = LIFSTDP(decay=decay, mem_detach=True)
        self.WTA = WTALayer(k=1)    # 赢者通吃
        # 侧抑制
        self.lateralinh = LateralInhibition(self.lif, inh, mode="threshold")
        # STDP参数
        self.trace = None
        self.decay_trace = decay_trace
        self.offset = offset

        self.time_window = time_window
        self.dw = 0 # STDP的改变的权重变化量（/batch*T）

    def forward(self, x):
        """
        args:
            x: 输入脉冲(B, C, H, W)
        return:
            :spikes: 是脉冲 (B,C,H,W)
        """
        spikes, dw = self.STDP(x)
        self.dw += dw/self.time_window

        return spikes

    def STDP(self, x):
        """
        利用STDP获得权重的变化量
        所有的结构都会在这个过程中利用
        args:
            :x : [B,C,H,W] -- 突触前峰(若包含时间就将其降维到B中)
        return:
            :s是脉冲 (B,C,H,W)
            :dw更新量 (out_planes,in_planes,H,W)
        """
        x = x.clone().detach()  # 突触前的峰
        i = self.conv(x)  # 输入电流(经过卷积后)
        with torch.no_grad():
            thre_max = self.getthresh(i.detach())  # 自适应性阈值
            i_ASF = self.ASFilter(i, thre_max)  # 通过适应阈值对电流进行滤波
            s = self.mem_update(i_ASF)  # 输出脉冲
            trace = self.cal_trace(x)  # 通过突触前峰更新trace
            x.data += trace - x.data - self.offset  # x变为trace(求导得出的值)
        dw = torch.autograd.grad(outputs=i, inputs=self.conv.weight, grad_outputs=s)[0]
        # print(x.size(0))
        dw /= x.size(0)  # 批次维度在求导时相加，除去
        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        x : [B,C,W,H] -- 突触前峰
        """
        if self.trace is None:
            self.trace = nn.Parameter(x.clone().detach(), requires_grad=False)
        else:
            self.trace *= self.decay_trace
            self.trace += x
        return self.trace.detach()

    def mem_update(self, x):
        """
        LIF的更新:(经过赢着通吃)
        赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
        args:
            x: 通过卷积核后的输入电流
        return:
            spiking: 输出的脉冲0/1
        """
        x = self.lif(x)  # 通过LIF后的脉冲
        if x.max() > 0:  # 判断有没有脉冲产生
            x = self.WTA(x)      # 赢者通吃(winner take all)
            self.lateralinh(x)   # 抑制不放电神经元的膜电位大小
        return x

    def getthresh(self, current):
        """
        适应性阈值平衡(Adaptive threshold balance, ATB)
        args:
            current: 卷积后的电流(B,C,H,W)
        retuen:
            维度C上的最大电流，阈值（ATB 确保不会因电流过大而丢失信息。）
        """
        thre_max = current.max(0, True)[0].max(2, True)[0].max(3, True)[0]+0.0001
        self.lif.threshold.data = thre_max # 更改LIF的阈值
        return thre_max

    def ASFilter(self, current, thre):
        """
        适应性突触滤波器(Adaptive synaptic filter, ASF)
        args:
            current: 卷积后的电流(B,C,H,W)
            thre: 适应性阈值平衡机制调节调节后的阈值
        return：
            current_ASF: 滤波后的电流
        """
        current = current.clamp(min=0)
        current_ASF = thre / (1 + torch.exp(-(current - 4 * thre / 10) * (8 / thre)))
        return current_ASF

    def normgrad(self):
        """
        将STDP带来的权重变化量放入权重梯度中，然后使用优化器更新权重
        """
        self.conv.weight.grad.data = -self.dw

    def normweight(self, clip=False):
        """
        权重在更新后标准化，防止它们发散或移动
        self.conv.weight --> (N,C,H,W)
        args:
            clip: 是否裁剪权重
        """
        if clip:
            self.conv.weight.data = torch. \
                clamp(self.conv.weight.data, min=-3, max=1.0)
        else:
            N, C, H, W = self.conv.weight.data.shape

            avg = self.conv.weight.data.mean(1, True).mean(2, True).mean(3, True)   # 每个批次的均值不一样
            self.conv.weight.data -= avg
            # 将除了批次维度的其他所有维度全部集中在第2维上，然后可以求出批次上的标准差
            tmp = self.conv.weight.data.reshape(N, 1, -1, 1)
            self.conv.weight.data /= tmp.std(2, unbiased=False, keepdim=True)   # 不使用无偏标准差

    def reset(self):
        """
        重置: 1、LIF的膜电位和spiking; 2、STDP的trace
        """
        self.lif.n_reset()
        self.trace = None
        self.dw = 0    # 将权重变化量清0


class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            STDPConv(in_planes=1, out_planes=12, kernel_size=3,
                     stride=1, padding=1, groups=1, decay=0.2,
                     decay_trace=0.99, offset=0.3, inh=1.625)
        ])


if __name__ == "__main__":
    snn = STDPConv(in_planes=1, out_planes=12,
                 kernel_size=3, stride=1, padding=1, groups=1)
    # snn.STDP(torch.ones((3, 1, 3, 3)))
    # print(snn.STDP(torch.ones((3,1,3,3))))
