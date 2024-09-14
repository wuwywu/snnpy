# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/22
# User      : WuY
# File      : msf.py
# 用于研究的主稳点函数 Master stability function

import copy
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from numba import njit, prange
import os
os.environ["OMP_NUM_THREADS"] = "4"  # 将4替换为你希望使用的线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
# 同步状态方程演化
def f(x, t):
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
def jac(x, t):
    """
    args:
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : MSF的雅可比矩阵
    """
    gamma = 1   # 耦合强度与 Laplacian 矩阵的特征值的乘积(自行在外部设定)
    # f 相对于 x 的雅可比行列式。
    def Df(x, t):
        """
        args:
            x (numpy.ndarray) : 状态变量
            t (float) : 运行时间
        return:
            res (numpy.ndarray) : 雅可比矩阵
        """
        res = np.zeros((x.shape[0], x.shape[0]))
        return res
    
    # 耦合函数 H 相对于 x 的雅可比行列式。
    def DH(x, t):
        """
        args:
            x (numpy.ndarray) : 状态变量
            t (float) : 运行时间
        return:
            res (numpy.ndarray) : 雅可比矩阵
        """
        res = np.zeros((x.shape[0], x.shape[0]))
        return res
    
    res = Df(x, t) - gamma * DH(x, t)
    return res

# 动力系统抽象类
class DynamicalSystem(ABC):
    """
    动力系统的实例化。
        Parameters:
            x0 (numpy.ndarray)：初始条件。
            f（function）：ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
            jac（function）：f 相对于 x 的雅可比行列式。
            dt（float）：两个时间步之间的时间间隔。
            t0（float）：初始时间。
            kwargs (dict)：f 和 jac 的参数字典。
    """
    def __init__(self, x0, f, jac, dt=1e-2, t0=0, **kwargs):
        self.x0 = x0
        self.t0 = t0
        self.x = x0
        self.t = t0
        self.dim = len(x0)
        self.f = f
        self.jac = jac
        self.dt = dt
        self.kwargs = kwargs

    @abstractmethod
    def next(self):
        '''
        计算一个时间步后系统的状态。
        '''
        pass

    @abstractmethod
    def next_LTM(self, W, gamma):
        '''
        计算一个时间步后偏差向量的状态。
            Parameters:
                W (numpy.ndarray): 偏差向量数组。
                gamma (float) : 耦合强度与 Laplacian 矩阵的特征值的乘积
            Returns:
                res (numpy.ndarray): 下一个时间步的偏差向量数组。
        '''
        pass

    def forward(self, n_steps, keep_traj=False):
        '''
        将系统转发 n_steps。
            Parameters:
                n_steps (int): 要做的模拟步骤数。
                keep_traj (bool): 返回或不返回系统轨迹。
            Returns:
                traj (numpy.ndarray): 维度系统的轨迹 (n_steps + 1,self.dim) if keep_traj.
        '''
        if (keep_traj):
            traj = np.zeros((n_steps + 1, self.dim))
            traj[0, :] = self.x
            for i in range(1, n_steps + 1):
                self.next()
                traj[i, :] = self.x
            return traj
        else:
            for _ in range(n_steps):
                self.next()


# 连续动力系统
class ContinuousDS(DynamicalSystem):
    '''
    连续动力系统的实例化。
        Parameters:
            x0 (numpy.ndarray)：初始条件。
            f（function）：ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
            jac（function）：f 相对于 x 的雅可比行列式。
            dt（float）：两个时间步之间的时间间隔。
            t0（float）：初始时间。
            kwargs (dict)：f 和 jac 的参数字典。
    '''
    def __init__(self, x0, f, jac, dt=1e-2, t0=0, **kwargs):
        super().__init__(x0, f, jac, dt, t0, **kwargs)

    def next(self):
        '''
        使用 RK4 方法计算一个时间步后系统的状态。
        '''
        k1 = self.f(self.x, self.t, **self.kwargs)
        k2 = self.f(self.x + (self.dt / 2.) * k1, self.t + (self.dt / 2.), **self.kwargs)
        k3 = self.f(self.x + (self.dt / 2.) * k2, self.t + (self.dt / 2.), **self.kwargs)
        k4 = self.f(self.x + self.dt * k3, self.t + self.dt, **self.kwargs)
        self.x = self.x + (self.dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += self.dt

    def next_LTM(self, W):
        '''
        使用 RK4 方法计算一个时间步后偏差向量的状态。
            Parameters:
                W (numpy.ndarray): 偏差向量数组。
            Returns:
                res (numpy.ndarray): 下一个时间步的偏差向量数组
        '''
        jacobian = self.jac(self.x, self.t, **self.kwargs)
        k1 = jacobian @ W
        k2 = jacobian @ (W + (self.dt / 2.) * k1)
        k3 = jacobian @ (W + (self.dt / 2.) * k2)
        k4 = jacobian @ (W + self.dt * k3)
        res = W + (self.dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        return res


# 离散动力系统
class DiscreteDS(DynamicalSystem):
    '''
    离散动力系统的实例化。
        Parameters:
            x0 (numpy.ndarray)：初始条件。
            f（function）：ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
            jac（function）：f 相对于 x 的雅可比行列式。
            dt（float）：两个时间步之间的时间间隔。
            t0（float）：初始时间。
            kwargs (dict)：f 和 jac 的参数字典。
    '''
    def __init__(self, x0, f, jac, dt=1, t0=0, **kwargs):
        super().__init__(x0, f, jac, dt, t0, **kwargs)

    def next(self):
        '''
        使用 RK4 方法计算一个时间步后系统的状态。
        '''
        self.x = self.f(self.x, self.t, **self.kwargs)
        self.t += self.dt

    def next_LTM(self, W):
        '''
        使用 RK4 方法计算一个时间步后偏差向量的状态。
            Parameters:
                W (numpy.ndarray): 偏差向量数组。
            Returns:
                res (numpy.ndarray): 下一个时间步的偏差向量数组
        '''
        jacobian = self.jac(self.x, self.t, **self.kwargs)
        res = jacobian @ W
        if (self.dim == 1):
            return np.array([res])
        else:
            return res


# 最大 1-Lyapunov characteristic exponents (LCE)
def msf_mLCE(system: DynamicalSystem, n_forward: int, n_compute: int, keep:bool=False):
    '''
    Compute the maximal 1-LCE.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the mLCE.
            n_forward (int): Number of steps before starting the mLCE computation.
            n_compute (int): Number of steps to compute the mLCE, can be adjusted using keep_evolution.
            keep (bool): If True return a numpy array of dimension (n_compute,) containing the evolution of mLCE.
        Returns:
            mLCE (float): Maximum 1-LCE.
            history (numpy.ndarray): Evolution of mLCE during the computation.
    '''
    # Forward the system before the computation of mLCE
    system.forward(n_forward, False)

    # Compute the mLCE
    mLCE = 0.
    w = np.random.rand(system.dim)
    w = w / np.linalg.norm(w)
    if keep:
        history = np.zeros(n_compute)
        for i in range(1, n_compute + 1):
            w = system.next_LTM(w)
            system.forward(1, False)
            mLCE += np.log(np.linalg.norm(w))
            history[i - 1] = mLCE / (i * system.dt)
            w = w / np.linalg.norm(w)
        mLCE = mLCE / (n_compute * system.dt)
        return mLCE, history
    else:
        for _ in range(n_compute):
            w = system.next_LTM(w)
            system.forward(1, False)
            mLCE += np.log(np.linalg.norm(w))
            w = w / np.linalg.norm(w)
        mLCE = mLCE / (n_compute * system.dt)
        return mLCE


# Lyapunov characteristic exponents (LCE)
def msf_LCE(system : DynamicalSystem, n_forward : int, n_compute : int, p:int=None, keep:bool=False):
    '''
    Compute LCE.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the LCE.
            n_forward (int): Number of steps before starting the LCE computation.
            n_compute (int): Number of steps to compute the LCE, can be adjusted using keep_evolution.
            p (int): Number of LCE to compute.
            keep (bool): If True return a numpy array of dimension (n_compute,p) containing the evolution of LCE.
        Returns:
            LCE (numpy.ndarray): Lyapunov Charateristic Exponents.
            history (numpy.ndarray): Evolution of LCE during the computation.
    '''
    if p is None: p = system.dim
    # Forward the system before the computation of LCE
    system.forward(n_forward, False)

    # Computation of LCE
    W = np.eye(system.dim)[:,:p]
    LCE = np.zeros(p)
    if keep:
        history = np.zeros((n_compute, p))
        for i in range(1, n_compute + 1):
            W = system.next_LTM(W)
            system.forward(1, False)
            W, R = np.linalg.qr(W)
            for j in range(p):
                LCE[j] += np.log(np.abs(R[j,j]))
                history[i-1,j] = LCE[j] / (i * system.dt)
        LCE = LCE / (n_compute * system.dt)
        return LCE, history
    else:
        for _ in range(n_compute):
            W = system.next_LTM(W)
            system.forward(1, False)
            W, R = np.linalg.qr(W)
            for j in range(p):
                LCE[j] += np.log(np.abs(R[j,j]))
        LCE = LCE / (n_compute * system.dt)
        return LCE


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

    # Lorenz = ContinuousDS(x0, f, jac, dt)

    # 计算LCE
    # LCE = msf_LCE(Lorenz, gamma, T_init, T_cal, keep=False)
    # LCE = msf_mLCE(Lorenz, gamma, T_init, T_cal, keep=False)
    # print(LCE)

    """
    LCE_list = []
    gamma_list = np.arange(0.01, 100, .1)
    # 计算3->3
    for gamma in gamma_list:
        def f(x, t):
            res = np.zeros_like(x)
            res[0] = sigma * (x[1] - x[0])
            res[1] = x[0] * (rho - x[2]) - x[1]
            res[2] = x[0] * x[1] - beta * x[2]
            return res

        def jac(x, t):
            def Df(x, t):
                res = np.zeros((x.shape[0], x.shape[0]))
                res[0, 0], res[0, 1] = -sigma, sigma
                res[1, 0], res[1, 1], res[1, 2] = rho - x[2], -1., -x[0]
                res[2, 0], res[2, 1], res[2, 2] = x[1], x[0], -beta
                return res

            def DH(x, t):
                res = np.zeros((x.shape[0], x.shape[0]))
                # res[0, 0] = 1   # 1-->1
                # res[1, 0] = 1   # 1-->2
                # res[0, 1] = 1   # 2-->1
                res[2, 2] = 1  # 3-->3
                return res

            res = Df(x, t) - gamma * DH(x, t)
            return res

        Lorenz = ContinuousDS(x0, f, jac, dt)
        LCE = msf_mLCE(Lorenz, T_init, T_cal, keep=False)
        LCE_list.append(LCE)


    # Plot of LCE_list
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_list, LCE_list)
    plt.xlabel("gamma")
    plt.ylabel("LCE")
    plt.title("the LCE")
    plt.show()
    """


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
        DH[0, 1] = 1   # 2-->1
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
