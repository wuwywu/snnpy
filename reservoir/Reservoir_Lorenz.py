# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/29
# User      : WuY
# File      : Reservoir_Lorenz.py
# 使用库网络（储蓄池计算，Reservior）预测洛伦兹系统

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中\
sys.path.append(r"../")

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base.utils.utils import setup_seed

# 全局禁用梯度计算
torch.set_grad_enabled(False)
# 固定随机种子
setup_seed(0)


# 构建回声状态网络(Echo State Network, ESN)
class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size,
                 spectral_radius=0.9):
        super().__init__()
        self.input_size = input_size            # 输入数量
        self.reservoir_size = reservoir_size    # 储蓄池数量
        self.output_size = output_size          # 输出数量
        self.spectral_radius = spectral_radius  # 谱半径（特征值的最大值，大于1就一定不具有回声状态）

        self.h = torch.zeros(reservoir_size)    # 初始化储蓄池状态

        # input-->reservoir
        self.InRs = nn.Linear(input_size, reservoir_size, bias=False)

        # reservoir->reservoir
        self.RsRs = nn.Linear(reservoir_size, reservoir_size, bias=False)

        # reservoir-->output
        self.readout = nn.Linear(reservoir_size, output_size, bias=False)
        self.init_weight()

    def forward(self, x):
        """
        input-->reservoir(->reservoir)-->output
        args:
            x: 外部输入的值
        """
        h = self.reseroir(x)
        out = self.readout(h)
        return out

    def reseroir(self, input):
        """
        创建一个储蓄池（RNN）
        """
        self.h = F.tanh(self.InRs(input)+self.RsRs(self.h))
        return self.h

    def init_weight(self):
        # 初始化输入权重
        nn.init.uniform_(self.InRs.weight.data, a=-.1, b=.1)

        # 初始化回声状态层权重
        nn.init.normal_(self.RsRs.weight.data, std=.1)

        # 调整回声状态层的权重的谱半径
        rho = torch.max(torch.abs(torch.linalg.eigvals(self.RsRs.weight.data)))
        self.RsRs.weight.data *= self.spectral_radius / rho

        # 初始化输出权重
        nn.init.normal_(self.readout.weight.data)

    def i_reset(self):
        """
        输入的维度一致
        在需要频繁重置时,开辟内存的消耗太大
        :return: None
        """
        self.h.fill_(0.)


# Lorenz Chaotic System



# 训练方法（岭回归）



# 画出结果

if __name__ == "__main__":
    input_size = 1
    reservoir_size = 500
    output_size = 1
    model = ESN(input_size, reservoir_size, output_size)



