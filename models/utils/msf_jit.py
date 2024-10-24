# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/10/24
# User      : WuY
# File      : msf_jit.py
# 用于研究的主稳点函数 Master stability function
# 使用 numba 加快速度

import copy
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from numba import njit, prange
import os
os.environ["OMP_NUM_THREADS"] = "4"  # 将4替换为你希望使用的线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# ==================================== 用于 numba 并行运算的函数代码 ====================================
# ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
# 同步状态方程演化
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

# MSE相关的矩阵
@njit
def jac(x, t, gamma, *args):
    """
    args:
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : MSF的雅可比矩阵
    """
    # gamma = 1  # 耦合强度与 Laplacian 矩阵的特征值的乘积(自行在外部设定)

    # f 相对于 x 的雅可比行列式。
    Df = np.zeros((x.shape[0], x.shape[0]))

    # 耦合函数 H 相对于 x 的雅可比行列式。
    DH = np.zeros((x.shape[0], x.shape[0]))

    res = Df - gamma * DH
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
def msf_mLCE_jit(x0, f, jac, n_forward, n_compute, dt, gamma, *args):
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
        jacobian = jac(x, t, gamma, *args)
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

# 计算 gamma 为复数时的 MSF
@njit
def msf_mLCE_jit_complex(x0, f, jac, n_forward, n_compute, dt, gamma, *args):
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
    W = np.random.rand(dim) + 1j * np.random.rand(dim)
    W = W / np.linalg.norm(W)

    for _ in range(n_compute):
        # w = system.next_LTM(w)
        jacobian = jac(x, t, gamma, *args)
        jacobian = jacobian.astype(np.complex128)

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
def msf_LCE_jit(x0, f, jac, n_forward, n_compute, dt, gamma, p=None, *args):
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
        jacobian = jac(x, t, gamma, *args)
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
def msf_mLCE_jit_discrete(x0, f, jac, n_forward, n_compute, gamma, *args):
    """
    Parameters:
        x0 (numpy.ndarray)：初始条件。
        f（function）: ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
        jac（function）: f 相对于 x 的雅可比行列式。
        n_forward (int): Number of steps before starting the mLCE computation.
        n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
        *args :  f 和 jac 需要修改的量
    """
    t = 0
    dt = 1
    x = x0
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
        jacobian = jac(x, t, gamma, *args)
        if dim == 1:
            jacobian = jacobian.reshape(-1, 1)
        W = jacobian @ W

        # system.forward(1, False)
        x = f(x, t, *args)
        t += dt

        mLCE += np.log(np.linalg.norm(W))
        W = W / np.linalg.norm(W)

    mLCE = mLCE / (n_compute * dt)

    return mLCE


if __name__ == "__main__":
    # 连续动力系统的定义，此处为 Lorenz63
    sigma = 10.
    rho = 28.
    beta = 2
    x0 = np.array([1.5, -1.5, 20.])
    t0 = 0.
    dt = 1e-2
    T_init = int(5e4)
    T_cal = int(1e6)
    gamma = 10


    # ====================== 并行运算方法 ======================
    @njit
    def f(x, t, sigma, rho, beta):
        res = np.zeros_like(x)
        res[0] = sigma * (x[1] - x[0])
        res[1] = x[0] * (rho - x[2]) - x[1]
        res[2] = x[0] * x[1] - beta * x[2]
        return res


    @njit
    def jac(x, t, gamma, sigma, rho, beta):
        Df = np.zeros((x.shape[0], x.shape[0]))
        Df[0, 0], Df[0, 1] = -sigma, sigma
        Df[1, 0], Df[1, 1], Df[1, 2] = rho - x[2], -1., -x[0]
        Df[2, 0], Df[2, 1], Df[2, 2] = x[1], x[0], -beta

        DH = np.zeros((x.shape[0], x.shape[0]))
        # DH[0, 0] = 1   # 1-->1
        # DH[1, 0] = 1   # 1-->2
        DH[0, 1] = 1  # 2-->1
        # DH[2, 2] = 1  # 3-->3

        res = Df - gamma * DH
        return res

    # gamma = 1
    # mLCE = msf_mLCE_jit(x0, f, jac, T_init, T_cal, dt, gamma, sigma, rho, beta)
    # LCE = msf_LCE_jit(x0, f, jac, T_init, T_cal, dt, gamma, None, sigma, rho, beta)
    # print(LCE)

    gamma_list = np.arange(0.01, 100, .1)

    @njit(parallel=True)
    def parallel_msf_mLCE(gamma_list, x0, f, jac, T_init, T_cal, dt, *args):
        n = len(gamma_list)
        mLCE_values = np.zeros(n)
        for i in prange(n):
            gamma = gamma_list[i]
            mLCE_values[i] = msf_mLCE_jit(x0, f, jac, T_init, T_cal, dt, gamma, sigma, rho, beta)

        return mLCE_values


    LCE_values = parallel_msf_mLCE(gamma_list, x0, f, jac, T_init, T_cal, dt, sigma, rho, beta)
    # Plot of LCE
    plt.figure(figsize=(6, 4))
    plt.plot(gamma_list, LCE_values)
    plt.ylabel("LCE")
    plt.xlabel("$\gamma$")
    plt.show()