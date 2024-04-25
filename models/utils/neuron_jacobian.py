# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/22
# User      : WuY
# File      : neuron_jacobian.py
# 各种动力学系统，和其jacobian矩阵

import numpy as np

# ====================== neuron ======================
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

# Hindmarsh-Rose(HR) ncluding the magnetic flux variable 模型
def HR_mag(x, t):
    res = np.zeros_like(x)
    # 常数
    a, b = 1, 3
    Iex = 3.2
    c, d = 1, 5
    r, s, x_rest = .006, 4, -1.6
    k0, k1, k2 = 1, 1, 0.5
    # 输出函数变量
    res[0] = x[1] - a*x[0]**3 + b*x[0]**2 - x[2] + Iex - k0*(0.01+0.06*x[3]**2)*x[0]
    res[1] = c - d*x[0]**2 - x[1]
    res[2] = r*(s*(x[0]-x_rest) - x[2])
    res[3] = k1*x[0] - k2*x[3]
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 常数
    a, b = 1, 3
    Iex = 3.2
    c, d = 1, 5
    r, s, x_rest = .006, 4, -1.6
    k0, k1, k2 = 1, 1, 0.5
    # 输出函数变量
    res[0, 0], res[0, 1], res[0, 2], res[0, 3] = -3*a*x[0]**2 + 2*b*x[0] - k0*(0.01+0.06*x[3]**2) , 1, -1, k0*2*0.06*x[3]*x[0]
    res[1, 0], res[1, 1], res[1, 2], res[1, 3] = -2*d*x[0], -1, 0, 0
    res[2, 0], res[2, 1], res[2, 2], res[2, 3] = r*s, 0, -r, 0
    res[3, 0], res[3, 1], res[3, 2], res[3, 3] = k1, 0, 0, -k2
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

def FHN2(x, t):
    res = np.zeros_like(x)
    # 常数
    a = 0.5
    b = 0.05
    # 输出函数变量
    res[0] = (x[0] - np.power(x[0], 3) / 3 - x[1])/b
    res[1] = x[0] + a
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 常数
    a = 0.5
    b = 0.05
    # 输出函数变量
    res[0, 0], res[0, 1] = (1-x[0]*x[0])/b, -1/b
    res[1, 0], res[1, 1] = 1, 0
    return res


# ====================== chaos ======================
# lorenz系统
def lorenz(x, t):
    res = np.zeros_like(x)
    # 常数
    SIGMA = 10
    R = 28
    BETA = 8 / 3
    # 输出函数变量
    res[0] = SIGMA * (x[1] - x[0])
    res[1] = R * x[0] - x[1] - x[0] * x[2]
    res[2] = x[0] * x[1] - BETA * x[2]
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 常数
    SIGMA = 10
    R = 28
    BETA = 8 / 3
    # 输出函数变量
    res[0, 0], res[0, 1], res[0, 2] = -SIGMA, SIGMA, 0
    res[1, 0], res[1, 1], res[1, 2] = R - x[2], -1, -x[0]
    res[2, 0], res[2, 1], res[2, 2] = x[1], x[0], -BETA
    return res


# ====================== map models ======================
def Chialvo(x, t):
    res = np.zeros_like(x)
    # 模型常数
    a = 0.89
    b = 0.6  # 0.6 to 0.18 oscillations to aperiodic burst
    c = 0.28
    k = 0.03
    res[0]= (res[0]** 2) * np.exp(res[1] - res[0]) + k
    res[1] = a * res[1] - b * res[0] + c
    return res

def jac(x, t):
    res = np.zeros((x.shape[0], x.shape[0]))
    # 模型常数
    a = 0.89
    b = 0.6  # 0.6 to 0.18 oscillations to aperiodic burst
    c = 0.28
    k = 0.03
    res[0, 0] = (2 * x[0]) * np.exp(x[1] - x[0]) - (x[0]** 2) * np.exp(x[1] - x[0])
    res[0, 1] = (x[0]** 2) * np.exp(x[1] - x[0])
    res[1, 0], res[1, 1] = -b, a
    return res

