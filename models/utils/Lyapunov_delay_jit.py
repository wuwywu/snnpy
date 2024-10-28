# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/10/24
# User      : WuY
# File      : Lyapunov_delay_jit.py
# 将各种用于包含时滞得动力学系统的李亚普诺夫指数(Lyapunov exponents)
# 使用的结果/算法摘自
# J. Doyne Farmer, Chaotic attractors of an infinite-dimensional dynamical system, Physica D: Nonlinear Phenomena 4 (1982) 366–393. https://doi.org/10.1016/0167-2789(82)90042-2.

import copy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

@njit
def f_delay(x, x_tau, t, *args):
    """
    args:
        x (numpy.ndarray) : 当前状态变量
        x_tau (numpy.ndarray) : 状态变量的延时值
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : 状态变量矩阵
    """
    res = np.zeros_like(x)
    return res

@njit
def jac_delay(x, x_tau, t, *args):
    """
    args:
        x (numpy.ndarray) : 当前状态变量
        x_tau (numpy.ndarray) : 状态变量的延时值
        t (float) : 运行时间
    return:
        df_dx (numpy.ndarray) : 雅可比矩阵, 
        df_dx_tau (numpy.ndarray) : 延迟变量得导数矩阵

        jac = [dF/dx, dF/dx_tau]
    """
    df_dx = np.zeros((x.shape[0], x.shape[0]))
    df_dx_tau = np.zeros((x.shape[0], x.shape[0]))
    return df_dx, df_dx_tau

@njit
def compute_mle_jit(x0, f_delay, jac_delay, dt, tau, n_forward, n_compute, *args):
    t = 0

    x = x0.copy()

    dim = len(x0)         # 变量维度
    N = int(tau / dt + 1) # 拓展维度（历史缓冲区大小）

    # 初始化历史缓冲区
    x_hist = np.broadcast_to(x[:, np.newaxis], (dim, N)).copy()
    delta_hist = np.zeros((dim, N))  # 微扰历史缓冲区

    # 初始化插入位置索引
    current_index = 0

    # 在延迟区间内随机初始化微扰，并归一化
    for i in range(dim):
        delta_hist[i, :] = np.random.randn(N)
    delta_hist /= np.linalg.norm(delta_hist)

    ltot = 0.0  # 累积的李雅普诺夫指数

    # 预积分阶段，用于稳定系统
    for _ in range(n_forward):
        # 获取延迟位置索引
        delayed_index = (current_index - int(tau / dt)) % N

        # 计算当前值和延迟值的雅可比矩阵
        x_new = rk4_step(x_hist[:, current_index], x_hist[:, delayed_index], dt, t, f_delay, *args)

        # 更新 current_index，插入 x_new
        current_index = (current_index + 1) % N
        x_hist[:, current_index] = x_new
        t += dt

    # 主循环，计算 MLE
    for _ in range(n_compute):
        # 计算延迟位置索引
        delayed_index = (current_index - int(tau / dt)) % N

        # 计算雅可比矩阵，传入未更新的 current_index 和 delayed_index 数据
        df_dx, df_dx_tau = jac_delay(x_hist[:, current_index], x_hist[:, delayed_index], t, *args)
        
        # 获取当前和延迟的微扰向量
        delta_t = np.ascontiguousarray(delta_hist[:, current_index])
        delta_tau = np.ascontiguousarray(delta_hist[:, delayed_index])
        delta_new = rk4_step_delta(delta_t, delta_tau, dt, df_dx, df_dx_tau)

        # 更新 delta_hist 缓冲区，插入新微扰
        delta_current_index = (current_index + 1) % N
        delta_hist[:, delta_current_index] = delta_new

        # 对微扰历史进行归一化
        delta_norm = np.linalg.norm(delta_hist)
        delta_hist /= delta_norm

        # 累积范数增长的对数
        ltot += np.log(np.maximum(delta_norm, 1e-10))

        # 计算新的 x 值，使用 f_delay 更新 x_hist
        x_new = rk4_step(x_hist[:, current_index], x_hist[:, delayed_index], dt, t, f_delay, *args)
        
        # 更新 x_hist 缓冲区，插入新 x 值
        current_index = (current_index + 1) % N
        x_hist[:, current_index] = x_new

        t += dt

    # 计算 MLE
    mle = ltot / (n_compute * dt)

    return mle

@njit
def compute_mle_discrete(x0, f_delay, jac_delay, tau, n_forward, n_compute, *args):
    t = 0
    dt = 1

    x = x0.copy()

    dim = len(x0)      # 变量维度
    N = int(tau + 1)   # 拓展维度（历史缓冲区大小）

    # 初始化历史缓冲区
    x_hist = np.broadcast_to(x[:, np.newaxis], (dim, N)).copy()
    delta_hist = np.zeros((dim, N))  # 微扰历史

    # 初始化插入位置索引
    current_index = 0

    # 随机初始化微扰并归一化
    for i in range(dim):
        delta_hist[i, :] = np.random.randn(N)
    delta_hist /= np.linalg.norm(delta_hist)

    ltot = 0.0  # 累积的李雅普诺夫指数

    # 预积分阶段，用于稳定系统
    for _ in range(n_forward):
        # 计算延迟位置索引
        delayed_index = (current_index - int(tau)) % N

        # 调用 f_delay 计算 x_new，但不立即更新 x_hist
        x_new = f_delay(x_hist[:, current_index], x_hist[:, delayed_index], t, *args)
        
        # 更新 current_index 后再写入 x_new
        current_index = (current_index + 1) % N
        x_hist[:, current_index] = x_new
        t += dt

    # 主循环，计算 MLE
    for _ in range(n_compute):
        # 计算延迟位置索引
        delayed_index = (current_index - int(tau)) % N

        # 先计算雅可比矩阵，确保传递的是未更新的 current_index 和 delayed_index 数据
        df_dx, df_dx_tau = jac_delay(x_hist[:, current_index], x_hist[:, delayed_index], t, *args)
        
        # 获取当前和延迟的微扰向量
        delta_t = np.ascontiguousarray(delta_hist[:, current_index])
        delta_tau = np.ascontiguousarray(delta_hist[:, delayed_index])
        delta_new = df_dx @ delta_t + df_dx_tau @ delta_tau

        # 将 delta_new 插入 delta_hist
        delta_current_index = (current_index + 1) % N
        delta_hist[:, delta_current_index] = delta_new

        # 对微扰历史进行归一化
        delta_norm = np.linalg.norm(delta_hist)
        delta_hist /= delta_norm

        # 累积范数增长的对数
        ltot += np.log(np.maximum(delta_norm, 1e-10))

        # 计算 f_delay 以获得新的 x 值
        x_new = f_delay(x_hist[:, current_index], x_hist[:, delayed_index], t, *args)

        # 更新 x_hist
        current_index = (current_index + 1) % N
        x_hist[:, current_index] = x_new

        t += dt

    # 计算 MLE
    mle = ltot / (n_compute * dt)

    return mle

@njit
def rk4_step(x, x_tau, dt, t, f_delay, *args):
    # 计算四个斜率
    k1 = f_delay(x, x_tau, t, *args)
    k2 = f_delay(x + 0.5 * dt * k1, x_tau, t + 0.5 * dt, *args)
    k3 = f_delay(x + 0.5 * dt * k2, x_tau, t + 0.5 * dt, *args)
    k4 = f_delay(x + dt * k3, x_tau, t + dt, *args)
    # 计算下一步的值
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

@njit
def rk4_step_delta(delta_x, delta_x_tau, dt, df_dx, df_dx_tau):
    # 同样计算四个斜率
    k1 = df_dx @ delta_x + df_dx_tau @ delta_x_tau
    k2 = df_dx @ (delta_x + 0.5 * dt * k1) + df_dx_tau @ delta_x_tau
    k3 = df_dx @ (delta_x + 0.5 * dt * k2) + df_dx_tau @ delta_x_tau
    k4 = df_dx @ (delta_x + dt * k3) + df_dx_tau @ delta_x_tau
    # 计算下一步的值
    return delta_x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ==================== 下面使用的正交分解的方式，运行时间过长，不推荐使用=======================
@njit
def mLCE_delay_jit_discrete(x0, f_delay, jac_delay, tau, n_forward, n_compute, *args):
    t = 0
    dt = 1

    x = x0.copy()

    dim = int(len(x0))      # 变量维度
    N = int(tau+1)          # 拓展维度

    x = np.broadcast_to(x[:, np.newaxis], (x.shape[0], N)).copy()  # 将变量拓展到(dim, N)
    x_new = np.zeros_like(x[:, 0])                                 # 新得到当前变量(演化后)

    # 初始化
    for _ in range(n_forward):
        x_new[:] = f_delay(x[:, 0], x[:, -1], t, *args)
        # 更新变量
        x[:, 1:] = x[:, :-1]    # 旧的数据在: 1: -1
        x[:, 0] = x_new[:]      # 新的数据在: 0

        t += dt
    
    # Compute the mLCE
    ltot = np.zeros((dim*N))                        # 所有的李指数

    # 微扰变量
    delta_x = np.zeros((dim, N, dim * N))           # (dim, t, dim * N)
    delta_x[:, 0, :] = 1                   
    delta_x_new = np.zeros_like(delta_x[:, 0, :])   # 给出每个维度上所有的微扰方向

    for i in range(n_compute):
        x_new[:] = f_delay(x[:, 0], x[:, -1], t, *args)

        df_dx, df_dx_tau = jac_delay(x[:, 0], x[:, -1], t, *args)
        delta_x_0 = np.ascontiguousarray(delta_x[:, 0, :])      # 确保连续存储
        delta_x_tau = np.ascontiguousarray(delta_x[:, -1, :])   # 确保连续存储

        delta_x_new[:, :] = df_dx @ delta_x_0 + df_dx_tau @ delta_x_tau

        # 更新变量
        x[:, 1:] = x[:, :-1]    # 旧的数据在: 1: -1
        x[:, 0] = x_new[:]      # 新的数据在: 0

        # 更新微扰
        delta_x[:, 1:, :] = delta_x[:, :-1, :]
        delta_x[:, 0, :] = delta_x_new[:, :]

        # delta_x, norms = gram_schmidt_jit(delta_x)
        # delta_x, norms = householder_qr(delta_x)

        # 使用 NumPy 的 QR 分解
        delta_x_reshaped = delta_x.reshape(dim * N, dim * N)
        Q_new, R = np.linalg.qr(delta_x_reshaped)
        Q_new = np.ascontiguousarray(Q_new)
        delta_x = Q_new.reshape(dim, N, dim * N)

        # 提取 R 矩阵的对角线元素
        norms = np.abs(np.diag(R))

        ltot += np.log(np.maximum(norms, 1e-10))  # 避免对非正数取对数

        t += dt

    LEs = ltot / (n_compute * dt)
    mle = np.max(LEs)

    return mle

## gs 正交分解微扰矩阵
@njit(parallel=True)
def gram_schmidt_jit(X):
    p, m, n = X.shape
    X_new = X.reshape(p*m, n)  # 将输入数据转换为连续数组
    Q = np.zeros_like(X_new)
    Znorm = np.zeros(n)

    for j in prange(n):
        v = np.ascontiguousarray(X_new[:, j])  # 确保v连续
        for i in range(j):
            Q_i = np.ascontiguousarray(Q[:, i])  # 确保每个Q_i连续
            proj = np.dot(Q_i, v)
            v -= proj * Q[:, i]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Q[:, j] = v / norm
        else:
            Q[:, j].fill(0)  # 使用.fill(0)来避免创建新的零数组
        Znorm[j] = norm

    return Q.reshape(p, m, n), Znorm

@njit
def householder_qr(X):
    """
    使用 Householder 反射进行 QR 分解，返回与 gram_schmidt_jit 相同的输出。

    X: (dim, N, dim * N) 的三维数组
    返回: Q (正交化后的矩阵)，Znorm (对角线元素的绝对值)
    """
    p, m, n = X.shape
    X_new = X.reshape(p * m, n)  # 转换为二维矩阵 (p * m, n)
    
    # 确保 X_new 是连续的
    X_new = np.ascontiguousarray(X_new)
    
    # 初始化
    R = X_new.copy()
    Q = np.eye(p * m)
    
    Znorm = np.zeros(n)
    
    for k in range(n):
        x = R[k:, k]
        
        # 确保 x 是连续的
        x = np.ascontiguousarray(x)
        
        # 计算 Householder 向量
        alpha = np.linalg.norm(x)
        if alpha == 0:
            Znorm[k] = 0.0
            continue
        else:
            e1 = np.zeros_like(x)
            e1[0] = 1.0
            v = x + np.sign(x[0]) * alpha * e1
            v /= np.linalg.norm(v)
        
        # 确保 v 是连续的
        v = np.ascontiguousarray(v)
        
        # 更新 R 矩阵
        R_k = R[k:, k:]
        # 确保 R_k 是连续的
        R_k = np.ascontiguousarray(R_k)
        R[k:, k:] = R_k - 2.0 * np.outer(v, np.dot(v, R_k))
        
        # 更新 Q 矩阵
        Q_k = Q[:, k:]
        # 确保 Q_k 是连续的
        Q_k = np.ascontiguousarray(Q_k)
        Q[:, k:] = Q_k - 2.0 * np.outer(Q_k @ v, v)
        
        # 存储对角线元素的绝对值，用于计算李雅普诺夫指数
        Znorm[k] = np.abs(R[k, k])
    
    # 将 Q_new（即 Q）重新 reshape 回三维
    Q_new = Q[:, :n]
    Q_new = np.ascontiguousarray(Q_new)

    Q = Q_new.reshape(p, m, n)
    
    return Q, Znorm


if __name__ == "__main__":

    # 模型参数
    @njit
    def f_Mackey_Glass(x, x_tau, t):
        a = .2
        b = .1
        c = 10

        res = np.zeros_like(x)
        res[0] = a * (x_tau[0]/(1+x_tau[0] ** c)) - b * x[0]

        return res
    
    @njit
    def jac_delay1(x, x_tau, t):
        a = .2
        b = .1
        c = 10

        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))
        df_dx[0, 0] = -b
        df_dx_tau[0, 0] = a*(1/(1+x_tau[0] ** c) - c*(x_tau[0]**c)/((1+x_tau[0] ** c)**2))

        return df_dx, df_dx_tau
    
    x0 = np.array([0.9])
    T_init = int(1e5)
    T_cal = int(5e6)
    tau = 23
    f = f_Mackey_Glass
    jac = jac_delay1
    dt = 0.01
    
    mle = compute_mle_jit(x0, f, jac, dt, tau, T_init, T_cal)
    print(mle)  # 0.00946463480301904
    
    @njit
    def f_Ikeda1(x, x_tau, t):
        a = 20
        b = 1
        c = np.pi / 4
        
        res = np.zeros_like(x)
        res[0] = a * (np.sin(x_tau[0] - c) ** 2) - b * x[0]

        return res
    
    @njit
    def jac_delay2(x, x_tau, t):
        # 线性化方程 (利用雅可比矩阵计算)
        """
        jac = [dF/dx, dF/dx_tau]
        jac = [-b, 2*a*sin(x_tau - c)*cos(x_tau - c)]
        """
        a = 20
        b = 1
        c = np.pi / 4

        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))

        df_dx[0, 0] = -b
        df_dx_tau[0, 0] = 2*a*np.sin(x_tau[0] - c)*np.cos(x_tau[0]- c)

        return df_dx, df_dx_tau

    x0 = np.array([0.9])
    T_init = int(1e5)
    T_cal = int(5e6)
    tau = 5
    f = f_Ikeda1
    jac = jac_delay2
    dt = 0.01
    
    mle = compute_mle_jit(x0, f, jac, dt, tau, T_init, T_cal)
    print(mle)  # 0.206293604132223

    tau = 2*np.pi
   
    @njit
    def f_Ikeda2(x, x_tau, t):
        res = np.zeros_like(x)

        res[0] = np.sin(x_tau[0])

        return res
    
    @njit
    def jac_delay3(x, x_tau, t):
        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))

        df_dx_tau[0, 0] = np.cos(x_tau[0])
        return df_dx, df_dx_tau
    
    x0 = np.array([0.9])
    T_init = int(1e5)
    T_cal = int(5e6)
    tau = 2*np.pi
    f = f_Ikeda2
    jac = jac_delay3
    dt = 0.01
    
    mle = compute_mle_jit(x0, f, jac, dt, tau, T_init, T_cal)
    print(mle)  # 0.10722666519935511
    