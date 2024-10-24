# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/10/24
# User      : WuY
# File      : Lyapunov_jit.py
# 将各种用于动力学系统的李亚普诺夫指数(Lyapunov exponents)
# 使用的结果/算法摘自
# P. Kuptsov's paper on covariant Lyapunov vectors(https://arxiv.org/abs/1105.5228).
# 使用 numba 加快速度

import copy
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from numba import njit, prange
import os
os.environ['NUMBA_NUM_THREADS'] = '4'


# ==================================== 用于 numba 并行运算的函数代码 ====================================
# 注意：
#     *args : 为 f 和 jac 需要修改的量
@njit
def f(x, t, *args):
    res = np.zeros_like(x)
    return res

@njit
def jac(x, t, *args):
    res = np.zeros((x.shape[0], x.shape[0]))
    return res

@njit
def rk4_step(x, t, dt, f, *args):
    k1 = f(x, t, *args)
    k2 = f(x + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = f(x + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = f(x + dt * k3, t + dt, *args)
    return x + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

# ============= 主要函数 =============
@njit
def mLCE_jit(x0, f, jac, n_forward, n_compute, dt, *args):
    """
    Parameters:
        x0 (numpy.ndarray)：初始条件。
        f（function）: ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
        jac（function）: f 相对于 x 的雅可比行列式。
        n_forward (int): Number of steps before starting the mLCE computation.
        n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
        dt（float）: 两个时间步之间的时间间隔。
        *args :  f 和 jac 需要修改的量
    """
    t = 0
    x = x0
    dim = len(x0)
    # 初始化
    for _ in range(n_forward):
        x = rk4_step(x, t, dt, f, *args)
        t += dt

    # Compute the mLCE
    mLCE = 0.
    W = np.random.rand(dim)
    W = W / np.linalg.norm(W)

    for _ in range(n_compute):
        # w = system.next_LTM(w)
        jacobian = jac(x, t, *args)
        k1 = jacobian @ W
        k2 = jacobian @ (W + (dt / 2.) * k1)
        k3 = jacobian @ (W + (dt / 2.) * k2)
        k4 = jacobian @ (W + dt * k3)
        W = W + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        # system.forward(1, False)
        x = rk4_step(x, t, dt, f, *args)
        t += dt

        mLCE += np.log(np.linalg.norm(W))
        W = W / np.linalg.norm(W)

    mLCE = mLCE / (n_compute * dt)

    return mLCE


@njit
def LCE_jit(x0, f, jac, n_forward, n_compute, dt, p=None, *args):
    """
    Parameters:
        x0 (numpy.ndarray)：初始条件。
        f（function）: ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
        jac（function）: f 相对于 x 的雅可比行列式。
        n_forward (int): Number of steps before starting the mLCE computation.
        n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
        dt（float）: 两个时间步之间的时间间隔。
        p (int): Number of LCE to compute.
        *args :  f 和 jac 需要修改的量
    """
    t = 0
    # x = x0
    x = np.ascontiguousarray(x0)
    dim = len(x0)
    if p is None: p = dim
    # 初始化
    for _ in range(n_forward):
        x = rk4_step(x, t, dt, f, *args)
        t += dt

    # Compute the mLCE
    W = np.eye(dim)[:, :p]
    LCE = np.zeros(int(p))

    for _ in range(n_compute):
        # w = system.next_LTM(w)
        jacobian = jac(x, t, *args)
        jacobian = np.ascontiguousarray(jacobian)
        W = np.ascontiguousarray(W)
        k1 = jacobian @ W
        k2 = jacobian @ (W + (dt / 2.) * k1)
        k3 = jacobian @ (W + (dt / 2.) * k2)
        k4 = jacobian @ (W + dt * k3)
        W = W + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        # system.forward(1, False)
        x = rk4_step(x, t, dt, f, *args)
        t += dt

        W, R = np.linalg.qr(W)
        for j in range(p):
            LCE[j] += np.log(np.abs(R[j, j]))

    LCE = LCE / (n_compute * dt)

    return LCE


@njit
def mLCE_jit_discrete(x0, f, jac, n_forward, n_compute, *args):
    """
    Parameters:
        x0 (numpy.ndarray)：初始条件。
        f（function）: ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
        jac（function）: f 相对于 x 的雅可比行列式。
        n_forward (int): Number of steps before starting the mLCE computation.
        n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
        dt（float）: 两个时间步之间的时间间隔。
        *args :  f 和 jac 需要修改的量
    """
    t = 0
    dt = 1
    x = np.ascontiguousarray(x0)
    dim = int(len(x0))
    # 初始化
    for _ in range(n_forward):
        x = f(x, t, *args)
        t += dt

    # Compute the mLCE
    mLCE = 0.
    W = np.random.rand(dim)
    W = W / np.linalg.norm(W)

    for _ in range(n_compute):
        # w = system.next_LTM(w)
        jacobian = jac(x, t, *args)
        if dim == 1:
            jacobian = jacobian.reshape(-1, 1)
        jacobian = np.ascontiguousarray(jacobian)
        W = np.ascontiguousarray(W)
        W = jacobian @ W

        # system.forward(1, False)
        x = f(x, t, *args)
        t += dt

        mLCE += np.log(np.linalg.norm(W))
        W = W / np.linalg.norm(W)

    mLCE = mLCE / (n_compute * dt)
    return mLCE


@njit
def LCE_jit_discrete(x0, f, jac, n_forward, n_compute, p, *args):
    """
    Parameters:
        x0 (numpy.ndarray)：初始条件。
        f（function）: ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
        jac（function）: f 相对于 x 的雅可比行列式。
        n_forward (int): Number of steps before starting the mLCE computation.
        n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
        dt（float）: 两个时间步之间的时间间隔。
        p : 函数f的维度
        *args :  f 和 jac 需要修改的量
    """

    t = 0
    dt = 1
    x = np.ascontiguousarray(x0)
    dim = int(len(x0))

    if p is None: p = dim
    # Forward the system before the computation of LCE
    for _ in range(n_forward):
        x = f(x, t, *args)
        t += dt

    # Computation of LCE
    W = np.eye(dim)[:, :p]
    LCE = np.zeros(int(p))

    for _ in range(n_compute):
        # w = system.next_LTM(w)
        jacobian = jac(x, t, *args)
        if dim == 1:
            jacobian = jacobian.reshape(-1, 1)
        jacobian = np.ascontiguousarray(jacobian)
        W = np.ascontiguousarray(W)
        W = jacobian @ W

        # system.forward(1, False)
        x = f(x, t, *args)
        t += dt

        W, R = np.linalg.qr(W)
        for j in range(p):
            LCE[j] += np.log(np.abs(R[j, j]))

    LCE = LCE / (n_compute * dt)
    return LCE


if __name__ == "__main__":
    # 连续动力系统的定义，此处为 Lorenz63
    sigma = 10.
    rho = 28.
    beta = 8. / 3.
    x0 = np.array([1.5, -1.5, 20.])
    t0 = 0.
    dt = 1e-2
    T_init = int(1e6)
    T_cal = int(1e6)

    @njit
    def f(x, t, sigma, rho, beta):
        res = np.zeros_like(x)
        res[0] = sigma * (x[1] - x[0])
        res[1] = x[0] * (rho - x[2]) - x[1]
        res[2] = x[0] * x[1] - beta * x[2]
        return res

    @njit
    def jac(x, t, sigma, rho, beta):
        res = np.zeros((x.shape[0], x.shape[0]))
        res[0, 0], res[0, 1] = -sigma, sigma
        res[1, 0], res[1, 1], res[1, 2] = rho - x[2], -1., -x[0]
        res[2, 0], res[2, 1], res[2, 2] = x[1], x[0], -beta
        return res


    sigma_list = np.arange(0, 10, .01)
    @njit(parallel=True)
    def parallel_mLCE(sigma_list, x0, f, jac, T_init, T_cal, dt, *args):
        n = len(sigma_list)
        mLCE_values = np.zeros(n)
        for i in prange(n):
            sigma = sigma_list[i]
            mLCE_values[i] = mLCE_jit(x0, f, jac, T_init, T_cal, dt, sigma, *args)

        # print(mLCE_values)
        return mLCE_values

    # @njit(parallel=True)
    # def parallel_LCE(sigma_list, x0, f, jac, T_init, T_cal, dt, *args):
    #     p = None
    #     n = len(sigma_list)
    #     LCE_values = np.zeros((n, 3))
    #     for i in prange(n):
    #         sigma = sigma_list[i]
    #         LCE_values[i] = LCE_jit(x0, f, jac, T_init, T_cal, dt, p, sigma, *args)
    #
    #     # print(mLCE_values)
    #     return LCE_values


    # 测试并行运算

    # mLCE = mLCE_jit(x0, f, jac, T_init, T_cal, dt, sigma, rho, beta)
    # LCE = LCE_jit(x0, f, jac, T_init, T_cal, dt, None, sigma, rho, beta)
    # print(LCE)

    mLCE_values = parallel_mLCE(sigma_list, x0, f, jac, T_init, T_cal, dt, rho, beta)
    # LCE_values = parallel_LCE(sigma_list, x0, f, jac, T_init, T_cal, dt, rho, beta)
    #
    # Plot of LCE
    plt.figure(figsize=(6, 4))
    plt.plot(sigma_list, mLCE_values)
    plt.ylabel("LCE")
    plt.xlabel("$\sigma$")
    plt.show()
