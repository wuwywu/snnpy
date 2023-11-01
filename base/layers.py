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
            else:
                self.out += x/self.time_window
                self.count += 1
            if self.count>=self.time_window :
                self.count = 0

        if not self.if_grad:
            self.out = self.out.clone().detch().cpu()

        return self.out


class VotingLayer(nn.Module):
    """
    投票层可用于最后没有标签（无监督学习），判断最终的正确率。
    用于SNNs的输出层, 几个神经元投票选出最终的类
    根据每个神经元在训练集的一次呈现中对十类数字的最高响应为每个神经元分配一个类别。
    这是无监督学习使用投票层时，唯一使用标签的步骤
    reference : Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, 9, 99.
    """
    def __init__(self, label_shape: int=10, alpha=0.1):
        super().__init__()
        self.n_labels = label_shape  # 标签类别数(如MNIST : 10)
        self.assignments = None
        self.alpha = alpha          # 放电率移动平均旧数据的衰减因子
        self.rates = None           # 通过这个放电率判断每个神经元的vote[n_neurons, n_label]

    def forward(self, firingRate: torch.Tensor):
        """
        先使用 assign_votes 得到最后一层神经元各自的票（类别）
        get_label : 根据最后一层的spike 计算得到label
        根据最后一层的spike 计算得到测试的label
        args:
            firingRate --> [N, in_size] 输入放电率
        return:
            每个批次的测试标签
        """
        Nbatch = firingRate.size(0)
        rates = torch.zeros(Nbatch, self.n_labels, device=firingRate.device)
        for i in range(self.n_labels):
            n_assigns = torch.sum(self.assignments == i).float()  # 共有多少个该类别节点
            if n_assigns > 0:
                indices = torch.nonzero(self.assignments == i).view(-1)  # 找到该类别节点位置
                rates[:, i] = torch.sum(firingRate[:, indices], 1) / n_assigns  # 该类别平均所有该类别节点发放脉冲数

        return torch.sort(rates, dim=1, descending=True)[1][:, 0]

    def assign_votes(self, firingRate, labels):
        """
        根据数据的标签，给投票的神经元(voters)分配标签
        args:
            firingRate --> [N, in_size] 输入放电率
            labels --> [N] 批次的标签(如MNIST有数据0-9)
        return:
            None
        """
        # print(firingRate.size())
        n_neurons = firingRate.size(1)  # 获取最后一层的神经元数量(voters)
        if self.rates is None:
            self.rates = torch.zeros(n_neurons, self.n_labels, device=firingRate.device)
        for i in range(self.n_labels):
            n_labeled = torch.sum(labels == i).float()
            if n_labeled > 0:
                indices = torch.nonzero(labels == i).view(-1)
                tmp = torch.sum(firingRate[indices], 0) / n_labeled  # 平均脉冲数
                self.rates[:, i] = self.alpha*self.rates[:, i]+(1-self.alpha)*tmp

        self.assignments = torch.max(self.rates, 1)[1]      # assignments表示voters的支持标签
    

class WTALayer(nn.Module):
    """
    winner take all用于SNNs的每层后，将随机选取一个或者多个输出(在通道维度上处理)
    :param k: X选取的输出数目 k默认等于1
    """
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        # x.shape = [N,C,W,H]
        # ret.shape = [N,C,W,H]
        pos = x * torch.rand(x.shape, device=x.device)
        if self.k > 1:
            x = x * (pos >= pos.topk(self.k, dim=1)[0][:, -1:]).float()
        else:
            x = x * (pos >= pos.max(1, True)[0]).float()

        return x

    
class LateralInhibition(nn.Module):
    """
    侧抑制 用于发放脉冲的神经元抑制其他同层神经元 在膜电位上作用
    """
    def __init__(self, node, inh, mode="constant"):
        super().__init__()
        self.inh = inh
        self.node = node
        self.mode = mode

    def forward(self, x: torch.Tensor, xori=None):
        # x.shape = [N, C, H, W]
        # ret.shape = [N, C, H, W]
        # linear shape [N, C]
        if self.mode == "constant":
            self.node.mem = self.node.mem - self.inh * (x.max(1, True)[0] - x)
        elif self.mode == "max":
            self.node.mem = self.node.mem - self.inh * xori.max(1, True)[0].detach() * (x.max(1, True)[0] - x)
        elif self.mode == "threshold":
            """
            x.max(1, True)[0]: 经过赢者通吃后，c维度只有一个放电，或者不放电
            self.node.threshold: 经过适应性阈值
            """
            self.node.mem = self.node.mem - self.inh * self.node.threshold * (x.max(1, True)[0] - x)
        return x
