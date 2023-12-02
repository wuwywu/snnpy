# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/2
# User      : WuY
# File      : rls.py
# 递归最小二乘法 --> Recursive least squares algorithm

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


class RLS(nn.Module):
    def __init__(self, N=1, local=[0,], alpha=1.0):
        """
        Recursive least squares algorithm
        args:
            N: 输入节点个数
            local: 学习的输出节点位置，是一个列表或者数组
            alpha: 学习率
        """
        super(RLS, self).__init__()
        self.num = N                        # 输入的数量
        self.local = torch.tensor(local)    # 学习的输出节点位置
        self.alpha = alpha                  # 学习率
        self.P = torch.stack([self.alpha * torch.eye(N) for i in local])

    def forward(self, w, input, error):
        """
        使用rls更新权重
        N: input_num: 输入节点的数量
        args:
            w: 需要更新的权重，shape: (output, N)
            input: 输入值，shape: (output, N)
            error: 输出值与对比值的误差, shape: (output, )
        """
        # input 的形状是 (len(self.local), N)
        input = input[self.local]
        # error 的形状是 (len(self.local),)
        error = error[self.local]

        # 执行矩阵-向量乘法
        Prs = torch.einsum('ijk,ik->ij', self.P, input)  # 形状为 (len(self.local), N)

        # 计算 a 的向量化版本
        as_ = 1.0 / (1.0 + torch.einsum('ij,ij->i', input, Prs))  # 形状为 (len(self.local),)

        # 更新 Ps
        P_updates = torch.einsum('i,ij,ik->ijk', as_, Prs, Prs)
        self.P -= P_updates

        # 更新权重 w
        w_updates = torch.einsum('i,ij->ij', as_ * error, Prs)
        w[self.local] -= w_updates

    def n_reset(self):
        """
        重置递归参数
        """
        self.P = torch.stack([self.alpha * torch.eye(self.num) for i in self.local])


class RLS_numpy:
    """
    最初的numpy版本
    """
    def __init__(self, N=1, local=[1, 2]):
        self.params()
        self.num = N        # 神经元的数量
        self.local = local  # 神经元学习的位置
        # Initialization of P matrix
        self.P = np.array([1 / self.alpha * np.eye(N) for i in local])

    def params(self):
        self.alpha = 1  # Inverse of learning rate parameter

    def forward(self, w, input, error):
        # 假设 input 的形状是 (len(self.local), N)
        input = input[self.local]
        # 假设 error 的形状是 (len(self.local),)
        # error = error[self.local]

        # 执行矩阵-向量乘法
        Prs = np.einsum('ijk,ik->ij', self.P, input)  # 形状为 (len(self.local), N)

        # 计算 a 的向量化版本
        as_ = 1.0 / (1.0 + np.einsum('ij,ij->i', input, Prs))  # 形状为 (len(self.local),)

        # 更新 Ps
        P_updates = np.einsum('i,ij,ik->ijk', as_, Prs, Prs)
        self.P -= P_updates

        # 更新权重 w
        w[self.local] -= np.einsum('i,ij->ij', as_ * error, Prs)

    def reset(self):
        self.P = np.array([1 / self.alpha * np.eye(self.num) for i in self.local])


class RLS_base:
    def __init__(self, N=1, local=[1, 2]):
        self.params()
        self.num = N  # 神经元的数量
        self.local = local  # 神经元学习的位置
        # Initialization of P matrix
        self.P = [1 / self.alpha * np.eye(N) for i in local]

    def params(self):
        self.alpha = 1  # Inverse of learning rate parameter

    # Recursive least squares algorithm
    def forward(self, w, input, error):
        # Changes non-local variables in-place
        for i, N in enumerate(self.local):
            # Update P
            Pr = self.P[i] @ input[N]
            a = 1 / (1 + np.dot(input[N], Pr))
            self.P[i] -= np.outer(Pr, Pr) * a

            # Update weights
            w[N] -= (a * Pr) * error[i]

    def reset(self):
        self.P = [1 / self.alpha * np.eye(self.num) for i in self.local]


if __name__ == "__main__":
    rls = RLS(N=10, local=[0])
    w = torch.randn(3, 10)     # 假设 w 是一个 (10, 2) 形状的张量 (output, input)
    print(w)
    input = torch.randn(3, 10) # 输入 (output, input)
    error = torch.randn(3)
    rls(w, input, error)
    print(w)
