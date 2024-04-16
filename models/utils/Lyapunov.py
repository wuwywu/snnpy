# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/22
# User      : WuY
# File      : Lyapunov.py
# 将各种用于动力学系统的李亚普诺夫指数(Lyapunov exponents)
# 使用的结果/算法摘自
# P. Kuptsov's paper on covariant Lyapunov vectors(https://arxiv.org/abs/1105.5228).

import copy
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# ẋ = f(x, t) 或 x_(n 1) = f(x_n) 的函数 f。
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

# f 相对于 x 的雅可比行列式。
def jac(x,t):
    """
    args:
        x (numpy.ndarray) : 状态变量
        t (float) : 运行时间
    return:
        res (numpy.ndarray) : 雅可比矩阵
    """
    res = np.zeros((x.shape[0], x.shape[0]))
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
    def next_LTM(self, W):
        '''
        计算一个时间步后偏差向量的状态。
            Parameters:
                W (numpy.ndarray): 偏差向量数组。
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
def mLCE(system: DynamicalSystem, n_forward: int, n_compute: int, keep:bool=False):
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
def LCE(system : DynamicalSystem, n_forward : int, n_compute : int, p:int=None, keep:bool=False):
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


# Covariant Lyapunov vectors (CLV)
def CLV(system: DynamicalSystem, n_forward: int, n_A: int, n_B: int, n_C: int, traj: bool, p:int=None, check=False):
    '''
    Compute CLV.
        Parameters:
            system (DynamicalSystem): Dynamical system for which we want to compute the mLCE.
            n_forward (int): Number of steps before starting the CLV computation.
            n_A (int): Number of steps for the orthogonal matrice Q to converge to BLV.
            n_B (int): Number of time steps for which Phi and R matrices are stored and for which CLV are computed.
            n_C (int): Number of steps for which R matrices are stored in order to converge A to A-.
            traj (bool): If True return a numpy array of dimension (n_B,system.dim) containing system's trajectory at the times CLV are computed.
            p (int): Number of CLV to compute.
        Returns:
            CLV (List): List of numpy.array containing CLV computed during n_B time steps.
            history (numpy.ndarray): Trajectory of the system during the computation of CLV.
    '''
    if p is None: p = system.dim
    # Forward the system before the computation of CLV
    system.forward(n_forward, False)

    # Make W converge to Phi
    W = np.eye(system.dim)[:, :p]
    for _ in range(n_A):
        W = system.next_LTM(W)
        W, _ = np.linalg.qr(W)
        system.forward(1, False)

    # We continue but now Q and R are stored to compute CLV later
    Phi_list, R_list1 = [W], []
    if traj:
        history = np.zeros((n_B + 1, system.dim))
        history[0, :] = system.x
    if check:
        copy = system.copy()
    for i in range(n_B):
        W = system.next_LTM(W)
        W, R = np.linalg.qr(W)
        Phi_list.append(W)
        R_list1.append(R)
        system.forward(1, False)
        if traj:
            history[i + 1, :] = system.x

    # Now we only store R to compute A- later
    R_list2 = []
    for _ in range(n_C):
        W = system.next_LTM(W)
        W, R = np.linalg.qr(W)
        R_list2.append(R)
        system.forward(1, False)

    # Generate A make it converge to A-
    A = np.triu(np.random.rand(p, p))
    for R in reversed(R_list2):
        C = np.diag(1. / np.linalg.norm(A, axis=0))
        B = A @ C
        A = np.linalg.solve(R, B)
    del R_list2

    # Compute CLV
    CLV = [Phi_list[-1] @ A]
    for Q, R in zip(reversed(Phi_list[:-1]), reversed(R_list1)):
        C = np.diag(1. / np.linalg.norm(A, axis=0))
        B = A @ C
        A = np.linalg.solve(R, B)
        CLV_t = Q @ A
        CLV.append(CLV_t / np.linalg.norm(CLV_t, axis=0))
    del R_list1
    del Phi_list
    CLV.reverse()

    if traj:
        if check:
            return CLV, history, copy
        else:
            return CLV, history
    else:
        if check:
            return CLV, copy
        else:
            return CLV


# Adjoint covariant vectors (ADJ)
def ADJ(CLV : list):
    '''
    Compute adjoints vectors of CLV.
        Parameters:
            CLV (list): List of np.ndarray containing CLV at each time step: [CLV(t1), ...,CLV(tn)].
        Returns:
            ADJ (List): List of numpy.array containing adjoints of CLV at each time step (each column corresponds to an adjoint).
    '''
    ADJ = []
    for n in range(len(CLV)):
        try:
            ADJ_t = np.linalg.solve(np.transpose(CLV[n]), np.eye(CLV[n].shape[0]))
            ADJ.append(ADJ_t / np.linalg.norm(ADJ_t, axis = 0))
        except:
            ADJ_t = np.zeros_like(CLV[n])
            for j in range(ADJ_t.shape[1]):
                columns = [i for i in range(ADJ_t.shape[1])]
                columns.remove(j)
                A = np.transpose(CLV[n][:,columns])
                _, _, Vh = np.linalg.svd(A)
                theta_j = Vh[-1] / np.linalg.norm(Vh[-1])
                ADJ_t[:,j] = theta_j
            ADJ.append(ADJ_t)
    return ADJ


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

    def f(x, t):
        res = np.zeros_like(x)
        res[0] = sigma * (x[1] - x[0])
        res[1] = x[0] * (rho - x[2]) - x[1]
        res[2] = x[0] * x[1] - beta * x[2]
        return res

    def jac(x, t):
        res = np.zeros((x.shape[0], x.shape[0]))
        res[0, 0], res[0, 1] = -sigma, sigma
        res[1, 0], res[1, 1], res[1, 2] = rho - x[2], -1., -x[0]
        res[2, 0], res[2, 1], res[2, 2] = x[1], x[0], -beta
        return res

    Lorenz63 = ContinuousDS(x0, f, jac, dt)

    # 计算LCE
    LCE, history = LCE(Lorenz63, T_init, T_cal, keep=True)

    # Plot of LCE
    plt.figure(figsize=(10, 6))
    plt.plot(history[:5000])
    plt.xlabel("Number of time steps")
    plt.ylabel("LCE")
    plt.title("Evolution of the LCE for the first 5000 time steps")
    plt.show()
