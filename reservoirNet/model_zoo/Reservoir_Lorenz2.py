# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/9
# User      : WuY
# File      : Reservoir_Lorenz2.py
# 使用库网络（储蓄池计算，Reservior）预测洛伦兹系统

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中\
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_seed
from learningrule import Ridge  # 导入岭回归算法

# 全局禁用梯度计算
torch.set_grad_enabled(False)
# 使用64位数据，精度高
torch.set_default_dtype(torch.float64)
# 固定随机种子
setup_seed(0)

# Lorenz Chaotic System
def lorenz_system(state, sigma, rho, beta):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)

def simulate_lorenz(initial_state, num_steps, dt, sigma=10, rho=28, beta=8./3.):
    states = [initial_state]
    for _ in range(num_steps):
        state = states[-1]
        derivative = lorenz_system(state, sigma, rho, beta)
        next_state = state + derivative * dt
        states.append(next_state)
    return torch.stack(states)


# 构建回声状态网络(Echo State Network, ESN)
class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size,
                 spectral_radius=0.9, alpha=0.3):
        """
        args:
            input_size:      输入节点数
            reservoir_size:  水库中的节点数
            spectral_radius: 谱半径
            alpha:           衰减率
        """
        super().__init__()
        self.input_size = input_size            # 输入数量
        self.reservoir_size = reservoir_size    # 储蓄池数量
        self.output_size = input_size           # 输出数量
        self.spectral_radius = spectral_radius  # 谱半径（特征值的最大值，大于1就一定不具有回声状态）
        self.alpha = alpha                      # 池状态的衰减率

        self.h = torch.zeros(reservoir_size)    # 初始化储蓄池状态

        # input-->reservoir
        self.InRs = nn.Linear(1+input_size, reservoir_size, bias=False)   # 加一个偏置量的维度
        #
        # # reservoir->reservoir
        self.RsRs = nn.Linear(reservoir_size, reservoir_size, bias=False)

        # reservoir-->output
        # self.readout = nn.Linear(input_size+reservoir_size, output_size, bias=False)
        self.readout = nn.Linear(1+input_size+reservoir_size, input_size, bias=False)   # 维度：偏置+输入+库
        self.init_weight()

    def forward(self, x):
        """
        input-->reservoir(->reservoir)-->output
        args:
            x: 外部输入的值（1+input_size）
        """
        h = self.reseroir(x)
        out = self.readout(torch.cat([x, h]))
        # out = self.readout(h)
        return out

    def train_readout(self, x):
        h = self.reseroir(x)
        return h
        # return torch.cat([x, h])

    def reseroir(self, input):
        """
        创建一个储蓄池（RNN）
        """
        # input = torch.hstack([torch.tensor([1]),input])
        self.h = self.h*(1-self.alpha)+self.alpha*F.tanh(self.InRs(input)+self.RsRs(self.h))
        return self.h

    def init_weight(self):
        # 初始化输入权重
        nn.init.uniform_(self.InRs.weight.data, a=-.1, b=.1)
        self.InRs.weight.data *= (torch.rand(self.InRs.weight.data.size()) < 0.05).float()
        # nn.init.normal_(self.InRs.weight.data)
        # self.InRs.weight.data *= .1
        # self.InRs.weight.data = torch.rand(self.InRs.weight.data.size())*2-1

        # 初始化回声状态层权重
        nn.init.normal_(self.RsRs.weight.data, std=.1)
        self.RsRs.weight.data *= (torch.rand(self.RsRs.weight.data.size()) < 0.05).float()
        # nn.init.uniform_(self.RsRs.weight.data, a=-1., b=1.)
        # nn.init.normal_(self.RsRs.weight.data)
        # self.RsRs.weight.data *= .5
        # w = torch.rand(self.RsRs.weight.data.size())
        # w[w>3/self.reservoir_size] = 0
        # self.RsRs.weight.data = w

        # 调整回声状态层的权重的谱半径
        rho = torch.max(torch.abs(torch.linalg.eigvals(self.RsRs.weight.data)))
        self.RsRs.weight.data *= self.spectral_radius/rho

        # 初始化输出权重
        nn.init.normal_(self.readout.weight.data)

    def i_reset(self):
        """
        输入的维度一致
        在需要频繁重置时,开辟内存的消耗太大
        :return: None
        """
        self.h.fill_(0.)


