# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/01
# User      : WuY
# File      : learningrule.py
# 将各种用于神经网络的`各种学习规则`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class STDP(nn.Module):
    """
    STDP learning rule
    reference: https://doi.org/10.1016/j.neunet.2023.06.019
    """

    def __init__(self, node, connection, decay_trace=0.99, offset=0.3):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例 里面只能有一个操作
        decay_trace: STDP计算trace时的衰减因子
        offset: 计算梯度时与trace相减的偏置
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace = None
        self.decay_trace = decay_trace
        self.offset = offset

    def forward(self, x):
        """
        计算前向传播过程
        :return:s是脉冲 dw更新量
        """
        x = x.clone().detach()
        i = self.connection(x)
        with torch.no_grad():
            s = self.node(i)   # 输出脉冲
            if self.training:  # 是否训练
                trace = self.cal_trace(x)
                x.data += trace - x.data - self.offset  # x变为trace(求导得出的值)
        if self.training:  # 是否训练
            dw = torch.autograd.grad(outputs=i, inputs=self.connection.weight, grad_outputs=s)[0]
            dw /= x.size(0)  # 批次维度在求导时相加，除去
        else:
            dw = 0.
        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        """
        if self.trace is None:
            self.trace = Parameter(x.clone().detach(), requires_grad=False)
        else:
            self.trace *= self.decay_trace
            self.trace += x
        return self.trace.detach()

    def reset(self):
        """
        重置
        """
        self.trace = None
