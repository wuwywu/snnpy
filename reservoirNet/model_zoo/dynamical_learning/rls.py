# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/13
# User      : WuY
# File      : rls.py
# 重写RLS算法
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

"""
使用方式(RLS)：
    1、创建类：
        in_num -- 输入节点的数量
        alpha -- 算法的学习率，
    2、forward类(直接使用)：
        o_z -- 库网络到输出信号的权重
        o_c -- 库网络到环境信号的权重
        r   -- 库网络的输出
        error_z -- 输出信号与目标输出信号的差值
        error_c -- 环境信号与目标环境型号的差值
"""

class RLS(nn.Module):
    """
    Recursive least squares algorithm
    args:
        in_num: 输入节点个数
        alpha: 学习率
    """
    def __init__(self, in_num=1, alpha=1.0):
        super().__init__()
        self.in_num = in_num        # 输入的数量
        self.alpha = alpha          # 学习率
        # Initialization of P matrix
        self.P = self.alpha * torch.eye(in_num, device=device)

    def forward(self, o_z, o_c, r, error_z, error_c):
        """
        Recursive least squares algorithm
        args:
            o_z: 库网络到输出信号的权重
            o_c: 库网络到环境信号的权重
            r:   库网络的输出
            error_z: 输出信号与目标输出信号的差值
            error_c: 环境信号与目标环境信号的差值
        """
        # Update P
        Pr = self.P @ r
        a = 1 / (1 + torch.dot(r, Pr))
        self.P -= torch.outer(Pr, Pr) * a

        # Update output weights
        o_z -= a * torch.outer(error_z, Pr)
        o_c -= a * torch.outer(error_c, Pr)
