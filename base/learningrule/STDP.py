# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/01
# User      : WuY
# File      : learningrule.py
# 将各种用于神经网络的`各种学习规则`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(r"../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from nodes import *

class spikesgen:
    """
    验证STPD的峰产生器
    """
    def __init__(self, delta_t=0, pair_t=50, n_pre=None, n_post=None):
        if n_pre is not None:
            self.n_pre = n_pre
            self.n_post = n_post
        else:
            self.n_pre = []
            self.n_post = []
            Tstart_pre  = int(100+delta_t)
            Tstart_post = int(100)
            for i in range(20):
                self.n_pre.append(Tstart_pre)
                self.n_post.append(Tstart_post)
                Tstart_pre += int(pair_t)
                Tstart_post += int(pair_t)

    def __call__(self, n):
        if n in self.n_pre:
            spike_pre = 1
        else: spike_pre = 0

        if n in self.n_post:
            spike_post = 1
        else: spike_post = 0

        return spike_pre, spike_post


class STDP(nn.Module):
    """
    STDP learning rule(只包含pre-post这种权重增大的情况)
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


class FullSTDP(nn.Module):
    """
    STDP learning rule(这是个完整的规则，包含了pre-pose增强和post-pre减小，这两种情况)
    """
    def __init__(self, node, connection, decay=0.99, decay2=0.99):
        """
        :params
        node: node神经元类型实例如IFNode LIFNode
        connection: 连接 类的实例 是一个可迭代对象，包含了多个输入层（一个也行）
        decay: STDP计算trace时的衰减因子 pre-pose增强
        decay2: STDP计算trace2时的衰减因子 post-pre减小
        """
        super().__init__()
        self.node = node
        self.connection = connection
        self.trace = [None for i in self.connection]  # 有多个输入
        self.trace2 = None    # 只有一个输出
        self.decay = decay      # pre-pose增强
        self.decay2 = decay2    # post-pre减小

    def forward(self, *x):
        """
        计算STDP前向传播过程
        args:
            *x: 可迭代的输入，对应 connection 的输入
        return:
            s是脉冲, dw: pre-pose增强更新量，dw2 post-pre减小更新量
        """
        i = 0
        x = [xi.clone().detach() for xi in x]   # 对应 connection
        # batch = x[0].size(0)        # 获得批次
        for xi, coni in zip(x, self.connection):
            i += coni(xi)   # 计算总的输入电流

        with torch.no_grad():
            s = self.node(i)    # 计算输出脉冲
            if self.training:
                # 计算trace2(post-pre减小)
                trace2 = self.cal_trace2(s)
        if self.training:
            dw2 = torch.autograd.grad(outputs=i, inputs=[con.weight for con in self.connection],
                                      retain_graph=True, grad_outputs=trace2)   # post-pre减小更新量
        if self.training:
            # 计算trace(pre-pose增强)
            with torch.no_grad():
                trace = self.cal_trace(x)       # 对应 connection
                for xi, ti in zip(x, trace):
                    xi.data += ti - xi.data
            dw = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection],
                                     grad_outputs=s)  # pre-pose增强更新量
        else:
            dw = 0.
            dw2 = 0.
        return s, dw, dw2

    def cal_trace(self, x):
        """
        arg:
            x: 突触前输入脉冲, 对应 connection
        return:
            trace: pre-pose增强 trace, 对应 connection
        """
        for i in range(len(x)):
            if self.trace[i] is None:
                self.trace[i] = nn.Parameter(x[i].clone().detach(), requires_grad=False)
            else:
                self.trace[i] *= self.decay
                self.trace[i] += x[i].detach()
        return self.trace

    def cal_trace2(self, s):
        """
        arg:
            s: 突触前输出脉冲
        return:
            trace2: post-pre减小 trace2
        """
        if self.trace2 is None:
            self.trace2 = nn.Parameter(torch.zeros_like(s), requires_grad=False)
        else:
            self.trace2 *= self.decay2
        trace2 = self.trace2.clone().detach()
        self.trace2 += s
        return trace2

    def reset(self):
        """
        重置
        """
        self.trace = [None for i in self.connection]  # 有多个输入
        self.trace2 = None  # 只有一个输出


if __name__=="__main__":
    sg = spikesgen(delta_t=10)
    # print(sg.n_pre)
    # print(sg.n_post)
    lif = LIFNode()
    conn = nn.Linear(1,1, bias=False)
    conn.weight.data = torch.tensor([1.], requires_grad=True)
    stdp = FullSTDP(lif, [conn])
    for i in range(10):
        print(stdp(torch.tensor([1.])))

