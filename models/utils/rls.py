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
    def __init__(self, in_num=1, out_num=1, alpha=1.0):
        """
        Recursive least squares algorithm
        args:
            N: 输入节点个数
            alpha: 学习率
        """
        super(RLS, self).__init__()
        self.in_num = in_num            # 输入的数量
        self.out_num = out_num          # 学习的输出节点位置
        self.alpha = alpha              # 学习率
        # (out_num, in_num, in_num)
        self.P = torch.stack([self.alpha * torch.eye(in_num) for i in range(out_num)])

    def forward(self, w, input, error):
        """
        使用rls更新权重
        N: input_num: 输入节点的数量
        args:
            w: 需要更新的权重，shape: (output, N)
            input: 输入值，shape: (in_num, )
            error: 输出值与对比值的误差, shape: (out_num, )
        """
        # 执行矩阵-向量乘法
        Prs = torch.einsum('ijk,k->ij', self.P, input)  # 形状为 (out_num, in_num)

        # 计算 a 的向量化版本
        as_ = 1.0 / (1.0 + torch.einsum('j,ij->i', input, Prs))  # 形状为 (out_num,)

        # 更新 Ps
        P_updates = torch.einsum('i,ij,ik->ijk', as_, Prs, Prs)
        self.P -= P_updates

        # 更新权重 w
        w_updates = torch.einsum('i,ij->ij', as_ * error, Prs)
        w -= w_updates

    def n_reset(self):
        """
        重置递归参数
        """
        self.P = torch.stack([self.alpha * torch.eye(self.in_num) for i in range(out_num)])

# ================================== dynamic learning of synchronization (DLS) algorithm =============
# 最新版的 DLS
class DLS:
    """
    dynamic learning of synchronization (DLS) algorithm
    这个版本解决的问题:
    1. 解决了并行运行的问题
    2. 解决了学习参量过多时,内存过大的bug

    args:
        N : 需要学习的参量数
        local : 需要调整的状态变量的位置
        alpha : 使用 DLS 的学习率参数
    """
    def __init__(self, N=1, local=[1, 2], alpha=0.1):
        self.num = N        # 需要学习的参量数
        self.local = local  # 需要调整的状态变量的位置
        self.alpha = alpha  # 使用 DLS 的学习率参数
        # 存储每个local元素对应的单位矩阵对角线乘以alpha
        self.P = np.full((len(local), N), self.alpha)

    def forward(self, w, input, error):
        local_input = input[self.local]     # 形状是 (len(self.local), N)
        local_error = error[self.local]     # 形状是 (len(self.local),)

        # 计算 Prs（仅需要对角线与输入相乘）
        Prs = self.P * local_input  # 直接逐元素相乘

        # 计算 a 的向量化版本
        as_ = 1.0 / (1.0 + np.sum(local_input * Prs, axis=1))

        # 更新 Ps，只更新对角线部分
        P_updates = as_[:, np.newaxis] * (Prs ** 2)
        self.P -= P_updates

        # 更新权重 w，使用高级索引和广播去除for循环
        delta_w = (as_ * local_error)[:, np.newaxis] * Prs
        np.add.at(w, self.local, -delta_w)

    def train(self, re_factor, factor, input_mem, self_y=None, dt=0.01):
        """
        用来训练你想要修正的值,使得状态变量同步
        args:
            rev_factor : 需要更新的参数：如权重或设定的需要修正的值 (N_状态变量 , self.num)
            factor : 与rev_factor相乘的量    (N_状态变量 , self.num)
            input_mem : 时刻 t+1 的状态变量, 在给出其他量后   (N_状态变量 ,)
            self_y : 自定义输入的值，与这个值同步 (N_状态变量,)
            dt  :   积分步长
        """
        # 外部因素的输入值(self.num, self.Inum)
        input = factor * dt  # (N_状态变量 , self.num)

        if self_y is not None:
            yMean = self_y  # 监督学习
        else:
            yMean = input_mem[self.local].mean()

        # 最小二乘法差值(self.num,)
        error_y = input_mem - yMean

        self.forward(re_factor, input, error_y)

    def reset(self):
        self.P = np.full((len(self.local), self.num), self.alpha)


# DLS+ADMM
class DLS_ADMM:
    """
    dynamic learning of synchronization (DLS) algorithm
    这个版本解决的问题:
    1. 解决了并行运行的问题
    2. 解决了学习参量过多时,内存过大的bug
    3. 加入了交替方向乘子法, 限定调节范围 (alternating direction method of multipliers (ADMM))

    args:
        N : 需要学习的参量数
        local : 需要调整的状态变量的位置
        alpha : 使用 DLS 的学习率参数
        rho : ADMM的惩罚参数
        use_admm : 使用ADMM的开关
        w_min : 权重的最小值约束
        w_max : 权重的最大值约束
    """
    def __init__(self, N=1, local=[1, 2], alpha=0.1, rho=0.1, use_admm=True, w_min=None, w_max=None):
        self.num = N  # 需要学习的参量数
        self.local = local  # 需要调整的状态变量的位置
        self.alpha = alpha  # 使用 DLS 的学习率参数
        self.rho = rho  # ADMM的惩罚参数
        self.w_min = w_min  # 权重的最小值约束
        self.w_max = w_max  # 权重的最大值约束
        self.use_admm = use_admm  # 是否使用ADMM

        # 存储每个local元素对应的单位矩阵对角线乘以alpha
        self.P = np.full((len(local), N), self.alpha)

        # 初始化ADMM的z和mu
        self.z = np.zeros((len(local), N))
        self.mu = np.zeros((len(local), N))

    def forward(self, w, input, error):
        local_input = input[self.local]  # 形状是 (len(self.local), N)
        local_error = error[self.local]  # 形状是 (len(self.local),)

        # 计算 Prs（仅需要对角线与输入相乘）
        Prs = self.P * local_input  # 直接逐元素相乘

        # 计算 a 的向量化版本
        as_ = 1.0 / (1.0 + np.sum(local_input * Prs, axis=1))

        # 更新 Ps，只更新对角线部分
        P_updates = as_[:, np.newaxis] * (Prs ** 2)
        self.P -= P_updates

        # 计算RLS部分的权重更新
        delta_w_rls = (as_ * local_error)[:, np.newaxis] * Prs
        np.add.at(w, self.local, -delta_w_rls)

        # 进行ADMM更新（如果使用ADMM）
        if self.use_admm:
            self.update_admm(w)

    def update_admm(self, w):
        # 使用向量化操作进行ADMM更新

        # 计算delta_w_admm
        delta_w_admm = self.rho * (self.z - w[self.local]) + self.mu / self.rho

        # 添加轻微的L2正则化项，防止权重过度调整
        delta_w_admm += 1e-5 * w[self.local]

        # 更新权重w
        np.add.at(w, self.local, -delta_w_admm)

        # 更新辅助变量z
        z_new = w[self.local] + self.mu / self.rho

        # 应用权重约束
        if self.w_min is not None:
            z_new = np.maximum(z_new, self.w_min)
        if self.w_max is not None:
            z_new = np.minimum(z_new, self.w_max)

        self.z = z_new  # 更新z

        # 更新拉格朗日乘子mu
        self.mu += self.rho * (w[self.local] - self.z)

    def train(self, re_factor, factor, input_mem, self_y=None, dt=0.01):
        """
        用来训练你想要修正的值,使得状态变量同步
        args:
            rev_factor : 需要更新的参数：如权重或设定的需要修正的值 (N_状态变量 , self.num)
            factor : 与rev_factor相乘的量    (N_状态变量 , self.num)
            input_mem : 时刻 t+1 的状态变量, 在给出其他量后   (N_状态变量 ,)
            self_y : 自定义输入的值，与这个值同步 (N_状态变量,)
            dt  :   积分步长
        """
        # 外部因素的输入值(self.num, self.Inum)
        input = factor * dt  # (N_状态变量 , self.num)

        if self_y is not None:
            yMean = self_y  # 监督学习
        else:
            yMean = input_mem[self.local].mean()

        # 最小二乘法差值(self.num,)
        error_y = input_mem - yMean

        self.forward(re_factor, input, error_y)

    def reset(self):
        self.P = np.full((len(self.local), self.num), self.alpha)
        self.z = np.zeros((len(self.local), self.num))
        self.mu = np.zeros((len(self.local), self.num))


# ================================== DLS 旧版 ==================================
# dynamic learning of synchronization (DLS) algorithm
class RLS_complex(nn.Module):
    def __init__(self, N=1, local=[0,], alpha=1.0):
        """
        Recursive least squares algorithm
        args:
            N: 输入节点个数
            local: 学习的输出节点位置，是一个列表或者数组
            alpha: 学习率
        """
        super().__init__()
        self.num = N                        # 输入的数量
        self.local = local                  # 学习的输出节点位置
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
    in_num = 10
    out_num = 1
    rls = RLS(in_num=in_num, out_num=out_num)
    w = torch.randn(out_num, in_num)     # 假设 w 是一个 (10, 2) 形状的张量 (output, input)
    print(w)
    input = torch.randn(in_num,) # 输入 (input, )
    error = torch.randn(out_num,) # 输入 (ouput, )
    rls(w, input, error)
    print(w)
