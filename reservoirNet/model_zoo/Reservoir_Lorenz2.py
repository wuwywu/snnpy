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

