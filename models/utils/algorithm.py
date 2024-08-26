# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/8/20
# User      : WuY
# File      : algorithm.py
# 给出演算法计算节点变化, 使用numba加速运行，可以做并行运算


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

@njit
def f(x, t, *args):
    """
    args:
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : 状态变量矩阵
    """
    res = np.zeros_like(x)
    return res


# 四阶 龙格库塔(RK4)
@njit
def rk4(f, x, t, dt, *args):
    '''
    使用 RK4 方法计算一个时间步后系统的状态。
    args:
        f : 函数
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : 状态变量矩阵
    '''
    k1 = f(x, t, *args)
    k2 = f(x + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = f(x + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = f(x + dt * k3, t + dt, *args)
    x = x + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

@njit
def euler(f, x, t, dt, *args):
    '''
    使用 euler 方法计算一个时间步后系统的状态。
    args:
        f : 函数
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : 状态变量矩阵
    '''
    dx_dt = f(x, t, *args)
    x = x + dt * dx_dt

    return x

