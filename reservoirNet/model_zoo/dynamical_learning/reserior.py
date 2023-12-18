# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/13
# User      : WuY
# File      : reserior.py
# 库网络
# reference: 10.1103/PhysRevLett.125.088103

from settings import *
import os
import sys
sys.path.append(snnpy)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rls import RLS

class rnn(nn.Module):
    def __init__(self, N=1, N_z=1, N_c=1):
        """
        args:
            N: 储蓄池中节点的数量
            N_z: 输出信号的节点数
            N_c: 环境信号的节点数
        """
        super().__init__()
        # 给定参数
        self._params()
        self.N = N
        self.N_z = N_z
        self.N_c = N_c
        self.rls = RLS(N, self.alpha).to(device=device)               # 初始化最小二乘法
        # 储蓄池
        self.r_r = nn.Linear(N, N, bias=False, device=device)      # r --> r
        self.z_r = nn.Linear(N_z, N, bias=False, device=device)    # z --> r
        self.c_r = nn.Linear(N_c, N, bias=False, device=device)    # c --> r
        self.eps_r = nn.Linear(N_z, N, bias=False, device=device)  # error_z --> r
        # 输出(这两个权重需要学习)
        self.r_z = nn.Linear(N, N_z, bias=False, device=device)    # r --> z
        self.r_c = nn.Linear(N, N_c, bias=False, device=device)    # r --> c
        # 储蓄池状态更新偏置
        # self.b = torch.empty((N))
        self.b = torch.empty(N, device=device)
        # 初始化
        self.init_weight()
        self.init_vars()
        
    def _params(self):
        self.tau = 1.            # Time constant
        self.tau_forget = 50      # Context averaging time constant
        # 递归最小二乘法 --> Recursive least squares algorithm
        self.alpha = 1           # learning rate parameter
        self.RLS_prob = 70 # 2        # Mean number of weight updates per unit time (probability of weight updates)
        # 训练时长
        self.t_stay = 500        # 环境误差输入切换时间(200-1_000)
        self.t_fb = 200          # 在切换时间内，误差输入到网络的时间

    def init_vars(self):
        self.x = torch.randn(self.N, device=device)       # activation variable
        self.eps = torch.zeros(self.N_z, device=device)   # error input
        self.r = F.tanh(self.x+self.b)      # r = tanh[x(t)+b]
        self.z = self.r_z(self.r)           # 输出信号
        self.c = self.r_c(self.r)           # 环境信号
        self.t = 0                          # 模拟时间

    def init_weight(self):
        # 循环权重参数
        g = 1.5             # Scaling of recurrent weights
        p = .1              # Connection probability
        # 权重范围
        w_tilde = 1.        # Scaling of external weights
        b_tilde = .2        # Scaling of biases
        # 储蓄池
        nn.init.normal_(self.r_r.weight.data, mean=0, std=g/np.sqrt(p*self.N))
        self.r_r.weight.data *= torch.empty(self.N, self.N, device=device).bernoulli_(p)
        nn.init.uniform_(self.z_r.weight.data, a=-w_tilde, b=w_tilde)
        nn.init.uniform_(self.c_r.weight.data, a=-w_tilde, b=w_tilde)
        nn.init.uniform_(self.eps_r.weight.data, a=-w_tilde, b=w_tilde)
        nn.init.uniform_(self.b, a=-b_tilde, b=b_tilde)
        # 需要学习的权重初始为零
        nn.init.zeros_(self.r_z.weight.data)
        nn.init.zeros_(self.r_c.weight.data)
    
    def forward(self, z_tilde, c_tilde, times, phase="pretraining"):
        if phase=="pretraining":
            self.pretraining(z_tilde, c_tilde, times)
        elif phase=="learn":
            self.learn(z_tilde, c_tilde, times)
        elif phase=="test":
            self.test(z_tilde, c_tilde, times)
        else:
            print("输入正确的阶段：(pretraining, learn, test)")

    def reservior(self, z_tilde):
        """
        循环网络（库网络）计算
        args:
            z_tilde: 目标输出信号
        """
        self.x += dt * (- self.x + self.r_r(self.r) + self.eps_r(self.eps) \
                       + self.z_r(self.z) + self.c_r(self.c)) / self.tau
        self.r = F.tanh(self.x + self.b)
        self.z = self.r_z(self.r)
        self.c = self.r_c(self.r)
        self.eps = self.z - z_tilde
        self.t += dt
    
    def pretraining(self, z_tilde, c_tilde, times):
        """
        预训练块
        arg:
            z_tilde: 目标输出信号
            c_tilde: 目标环境信号
            times:   理论运行次数
        """
        self.reservior(z_tilde)

        # Pretraining specific computations(泊松过程的方式进行训练)
        if torch.rand(1, device=device) < self.RLS_prob * dt:    # Update weights
            self.rls(self.r_z.weight.data, self.r_c.weight.data, self.r, self.eps, self.c-c_tilde)

        if (self.t-dt)%self.t_stay >= self.t_fb:  # 每一段训练中误差反馈给网络的时间
            # Zero error input
            self.eps.fill_(0.)
            # Fix context signal
            self.c = c_tilde.clone()       # 不反馈误差的时间中固定c卫常数目标

    def learn(self, z_tilde, c_tilde, times):
        """
        学习块
        arg:
            z_tilde: 目标输出信号
            c_tilde: 目标环境信号
            times:   理论运行时间
        """
        if times == 0:
            self.c_fix = self.c.clone()  # 学习阶段使用c_fix记录c的平均值(移动平均，tau_forget)
        self.reservior(z_tilde)
        self.c_fix += dt/self.tau_forget * (-self.c_fix + self.c)

    def test(self, z_tilde, c_tilde, times):
        """
        学习块
        arg:
            z_tilde: 目标输出信号
            c_tilde: 目标环境信号
            times:   理论运行时间
        """
        self.reservior(z_tilde)
        # 测试过程 --> 误差为0，
        self.eps.fill_(0.)               # Zero error input
        self.c = self.c_fix.clone()      # 测试阶段使用指数移动平均


if __name__ == "__main__":
    net = rnn(10)
    # print(net.r_z.weight.data)
    # print(net.b)
    # print(net.r_r.weight.data)
    print(net.r)

