# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/23
# User      : WuY
# File      : synapses.py

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse


class BaseSynapses(nn.Module):
    """
    这是一个突触的基类, 输入前一个神经元传递过来的突触电流，
    通过突触前神经元pre, 连接conn, 以及突触后神经元post的膜电位
    计算突触前神经元pre给突触后神经元post的突触电流
    args:
        pre: 突触前神经元
        conn: 连接
        post: 突触后神经元
    """
    def __init__(self, pre, conn, post):
        super().__init__()
        self.pre = pre
        self.conn = conn
        self.post = post
        self.g = None   # 电导

    def forward(self):
        pass

    def spikepre(self, I):
        """ 计算突触前神经元的放电 """
        return self.pre(I)

    def n_reset(self):
        """
        突触电导重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.g = None


class synchem(BaseSynapses):
    """
    这个定义一个化学突触
    math:
    $$\tau _ {g} \frac { d g } { d t } = - g +
    \sum _ { i = 1 } ^ { N } \sum _ { k } w _ { i } \delta ( t - t _ { i } ^ { k } )$$
    refernce: https://doi.org/10.1016/j.neunet.2019.09.007

    args:
        pre: 突触前神经元
        conn: 连接
        post: 突触后神经元
        E: 化学突触的平衡电位
        tau: 化学突触更新电导的时间常数
    """
    def __init__(self, pre, conn, post, E=0, tau=1, dt=.1):
        super().__init__(pre, conn, post)
        self.E = E      # 平衡电位，兴奋突触/抑制突触
        self.tau = tau  # 时间常数
        self.dt = dt    # 积分步长

    def forward(self, x):
        """
        args:
            x: 输入到突触前神经元的突触电流值
        return:
            I: 输出到突触后神经元的突触电流值
        """
        s = self.spikepre(x)
        dg = self.conn(s)   # 电导的增量
        if self.g is None:
            self.g = torch.zeros_like(dg, device=dg.device)
            I = torch.zeros_like(dg, device=dg.device)
        else:
            I = self.g*(self.E-self.post.mem)
        self.updata_g(dg)

        return I, s

    def updata_g(self, dg):
        """
        使用突触前计算出来的spike,更新电导
        arg:
            dg: 电导的更新
        """
        self.g += (dg-self.g)*self.dt/self.tau

    def i_reset(self):
        """
        输入的维度一致
        在需要频繁重置时,开辟内存的消耗太大
        """
        if self.g is not None:
            self.g.fill_(0)

