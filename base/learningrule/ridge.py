# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/2
# User      : WuY
# File      : ridge.py
# 岭回归算法--> ridge regression

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

# offline
def Ridge(X, Y, alpha=1e-6):
    """
    实现多输出岭回归的闭式解。
    权重矩阵的形状为 (n_outputs, n_features)。

    args:
        X: 输入特征张量, 形状为 (n_samples, n_features)
        Y: 输出张量, 形状为 (n_samples, n_outputs)
        alpha: 正则化系数
    return:
        权重张量 (n_outputs, n_features)
    """
    n_features = X.size(1)
    I = torch.eye(n_features, dtype=X.dtype, device=X.device)
    X_transpose = torch.transpose(X, 0, 1)

    # 计算权重的闭式解
    W = torch.inverse(X_transpose @ X + alpha * I) @ X_transpose @ Y
    return W.transpose(0, 1)  # 调整为 (n_outputs, n_features)


if __name__ == "__main__":
    in_num = 95
    out_num = 1
    n_samples = 100
    input = torch.randn((n_samples, in_num)) # 输入 (n_samples, n_features)
    # output1 = torch.randn((n_samples, out_num)) # 输入 (n_samples, n_outputs)
    output = torch.sin(input)
    print(output.T)
    w = Ridge(input, output)
    print(w@input.T)
    # plt.scatter(np.arange(n_samples), output.T.numpy()[0])
    plt.scatter(np.arange(n_samples), output.T.numpy()[0])
    plt.plot(np.arange(n_samples), (w@input.T).numpy()[0])
    # plt.scatter(np.arange(n_samples), output.T.numpy()[1])
    # plt.plot(np.arange(n_samples), (w @ input.T).numpy()[1])

    plt.show()


