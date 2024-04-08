# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/22
# User      : WuY
# File      : neuron_jacobian.py
# 各种动力学系统，和其jacobian矩阵

import numpy as np

# Hindmarsh-Rose(HR) 模型
def HR(x, t):
    res = np.zeros_like(x)
    # 常数
    a, b = 1, 3
    Iex = 3.2
    c, d = 1, 5
    r, s, x_rest = .006, 4, -1.6
    # 输出函数变量
    res[0] = x[1] - a*x[0]**3 + b*x[0]**2 - x[2] + Iex
    res[1] = c - d*x[0]**2 - x[1]
    res[2] = r*(s*(x[0]-x_rest) - x[2])
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 常数
    a, b = 1, 3
    Iex = 3.2
    c, d = 1, 5
    r, s, x_rest = .006, 4, -1.6
    # 输出函数变量
    res[0, 0], res[0, 1], res[0, 2] = -3*a*x[0]**2 + 2*b*x[0], 1, -1
    res[1, 0], res[1, 1], res[1, 2] = -2*d*x[0], -1, 0
    res[2, 0], res[2, 1], res[2, 2] = r*s, 0, -r
    return res


# FitzHugh-Nagumo(FHN) 模型
def FHN(x, t):
    res = np.zeros_like(x)
    # 常数
    ξ = 0.175
    a = 0.7
    b = 0.8
    c = 0.1
    # 输出函数变量
    res[0] = x[0] * (1 - ξ) - np.power(x[0], 3) / 3 - x[1]
    res[1] = c * (x[0] + a - b * x[1])
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 常数
    ξ = 0.175
    a = 0.7
    b = 0.8
    c = 0.1
    # 输出函数变量
    res[0, 0], res[0, 1] = (1-ξ)-x[0]*x[0], -1
    res[1, 0], res[1, 1] = c, -c*b
    return res

