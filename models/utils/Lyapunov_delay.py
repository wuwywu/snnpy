# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/4/15
# User      : WuY
# File      : Lyapunov_delay.py
# 将各种用于包含时滞得动力学系统的李亚普诺夫指数(Lyapunov exponents)
# 使用的结果/算法摘自
# J. Doyne Farmer, Chaotic attractors of an infinite-dimensional dynamical system, Physica D: Nonlinear Phenomena 4 (1982) 366–393. https://doi.org/10.1016/0167-2789(82)90042-2.


import copy
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

# ẋ = f(x, x_tau, t) 或 x_(n 1) = f(x_n, x_tau) 的函数 f。
@jit(nopython=True) # 如果 ContinuousDS系统中使用了jit，请加入装饰器
def f_delay(x, x_tau, t):
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

def jac_delay(x, x_tau, t):
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

"""
注 : 若延迟时间很大的时候，可以使用龙格库塔rk4(打开jit才能使用)，这样选取的维度N不必太大，可以减少计算的开销
"""
# 连续动力系统
class ContinuousDS:
    '''
    连续动力系统的实例化。
        Parameters:
            x0 (numpy.ndarray)：初始条件 (系统)。
            f_delay (function)：ẋ = f(x, x_tau, t) 或 x_(n 1) = f(x_n, x_tau) 的函数 f。
            jac_delay (function): f 相对于 x 的雅可比行列式。
            tau (float): 延迟时间。
            N (int) : 将函数拓展到 N 维上 (dt : tau/N)
            t0 (float): 初始时间。
            jit : 是否用numba编译（使用编译时，函数f_delay加上装饰器@jit(nopython=True)）
            rk : 是否使用四阶龙格-库塔
            kwargs (dict): f 和 jac 的参数字典。
    '''
    def __init__(self, x0, f_delay, jac_delay, tau, N=100, t0=0, jit=True, rk=False, **kwargs):
        # 保留初始值
        self.x0 = x0
        self.t0 = t0

        self.dim = len(x0)  # 变量维度
        self.N = N          # 拓展维度

        # 变量从初始值开始（变量初始值：dimxN; 微扰初始值：dimxNxN）
        self.x = np.tile(x0[:, np.newaxis], (1, self.N))# (dim, N)
        self.xnew = np.zeros_like(self.x[:, 0])
        

        # 微扰变量
        self.delta_x = np.zeros((self.dim, N, self.dim*N))   # (dim, t, N)
        self.delta_x[:, 0, :] = 1
        # np.fill_diagonal(self.delta_x[:, 0, :], 1)  # 对于每个维度，设置初始微扰在第0个时间点的对角线为1
        self.delta_xnew = np.zeros_like(self.delta_x[:, 0, :])

        self.t = t0

        self.f_delay = f_delay
        self.jac_delay = jac_delay

        self.dt = tau/N

        self.jit = jit
        self.rk4 = rk

        self.kwargs = kwargs

    def next(self):
        '''
        使用 Euler/rk4 方法计算一个时间步后系统的状态。
        '''
        x_0 = self.x[:, 0]
        x_tau = self.x[:, -1]
        if self.jit:
            if self.rk4:
                self.xnew[:] = rk4_step(x_0, x_tau, self.dt, self.t, self.f_delay)
            else:
                self.xnew[:] = update_state(x_0, x_tau, self.dt, self.t, self.f_delay)
        else:
            self.xnew[:] = x_0 + self.dt*self.f_delay(x_0, x_tau, self.t)  # 更新当前状态变量 (dim, )
   
    def next_delta_x(self):
        '''
        使用 Euler/rk4 方法计算一个时间步后偏差的状态。
        '''
        x_0 = self.x[:, 0]
        x_tau = self.x[:, -1]
        delta_x_0 = self.delta_x[:, 0, :]
        delta_x_tau = self.delta_x[:, -1, :]

        df_dx, df_dx_tau = self.jac_delay(x_0, x_tau, self.t)
        if self.jit:
            if self.rk4:
                self.delta_xnew[:, :] = rk4_step_delta(delta_x_0, delta_x_tau, self.dt, df_dx, df_dx_tau)
            else:
                self.delta_xnew[:, :] = update_delta_x(delta_x_0, delta_x_tau, self.dt, df_dx, df_dx_tau)
        else:
            ddelta_x_dt = df_dx@delta_x_0 + df_dx_tau@delta_x_tau
            self.delta_xnew[:, :] =  delta_x_0 + self.dt*ddelta_x_dt  # 更新当前微扰变量 (dim, N)

    def update_tau(self):
        self.x[:, 1:] = self.x[:, :-1]
        self.delta_x[:, 1:, :] = self.delta_x[:, :-1, :]

        self.x[:, 0] = self.xnew[:]
        self.delta_x[:, 0, :] = self.delta_xnew[:, :]

        self.t += self.dt


# 离散动力系统
class DiscreteDS:
    '''
    连续动力系统的实例化。
        Parameters:
            x0 (numpy.ndarray)：初始条件 (系统)。
            f_delay (function)：ẋ = f(x, x_tau, t) 或 x_(n 1) = f(x_n, x_tau) 的函数 f。
            jac_delay (function): f 相对于 x 的雅可比行列式。
            tau (int): 延迟时间 (离散系统，延时为多少，拓展到tau+1维)。
            t0 (float): 初始时间。
            kwargs (dict): f 和 jac 的参数字典。
    '''
    def __init__(self, x0, f_delay, jac_delay, tau, t0=0, **kwargs):
        # 保留初始值
        self.x0 = x0
        self.t0 = t0

        self.dim = len(x0)  # 变量维度
        self.N = int(tau+1)   # 拓展维度

        # 变量从初始值开始（变量初始值：dimxN; 微扰初始值：dimxNxN）
        self.x = np.tile(x0[:, np.newaxis], (1, self.N))  # (dim, N)
        self.xnew = np.zeros_like(self.x[:, 0])

        # 微扰变量
        self.delta_x = np.zeros((self.dim, self.N, self.dim * self.N))  # (dim, t, N)
        self.delta_x[:, 0, :] = 1
        # np.fill_diagonal(self.delta_x[:, 0, :], 1)  # 对于每个维度，设置初始微扰在第0个时间点的对角线为1
        self.delta_xnew = np.zeros_like(self.delta_x[:, 0, :])

        self.t = t0

        self.f_delay = f_delay
        self.jac_delay = jac_delay

        self.dt = 1

        self.kwargs = kwargs

    def next(self):
        '''
        计算一个时间步后系统的状态。
        '''
        x_0 = self.x[:, 0]
        x_tau = self.x[:, -1]

        self.xnew[:] = self.f_delay(x_0, x_tau, self.t)  # 更新当前状态变量 (dim, )

    def next_delta_x(self):
        '''
        计算一个时间步后偏差的状态。
        '''
        x_0 = self.x[:, 0]
        x_tau = self.x[:, -1]
        delta_x_0 = self.delta_x[:, 0, :]
        delta_x_tau = self.delta_x[:, -1, :]

        df_dx, df_dx_tau = self.jac_delay(x_0, x_tau, self.t)
        self.delta_xnew[:, :] = df_dx @ delta_x_0 + df_dx_tau @ delta_x_tau

    def update_tau(self):
        self.x[:, 1:] = self.x[:, :-1]
        self.delta_x[:, 1:, :] = self.delta_x[:, :-1, :]

        self.x[:, 0] = self.xnew[:]
        self.delta_x[:, 0, :] = self.delta_xnew[:, :]

        self.t += self.dt


# Lyapunov characteristic exponents (LCE)
def LCE(system, n_forward : int, n_compute : int, jit=True):
    """
    计算李雅普诺夫指数 (Lyapunov Characteristic Exponents)
    Params:
        system (DynamicalSystem): 要计算的动力系统
        n_forward (int): 在开始计算LCE之前前进的步数
        n_compute (int): 用于计算LCE的步数
    Returns:
        numpy.ndarray: 计算得到的LCE
    """
    p = system.dim        # 变量维度
    N = system.N          # 拓展维度
    _dt = system.dt

    # 初始化
    for i in range(n_forward):
        system.next()
        system.x[:, 1:] = system.x[:, :-1]
        system.x[:, 0] = system.xnew[:]

    # 开始计算 LCE
    ltot = np.zeros((p*N))
    for i in range(n_compute):
        system.next()               # 下一个状态
        system.next_delta_x()       # 下一个状态差值
        system.update_tau()         # 更新延迟数组

        # Gram-Schmidt 正交化 (对 delta_x 使用)
        if jit:
            system.delta_x, norms = gram_schmidt_jit(system.delta_x)
        else:
            system.delta_x, norms = gram_schmidt(system.delta_x)
        ltot += np.log(np.maximum(norms, 1e-10))  # 避免对非正数取对数

        # LEs = ltot / ((i+1)*_dt)

        # if i % (n_compute // 10) == 0 and i != 0:  # 定期打印信息
        #     # print(f"Step {i}: Lyapunov Exponents = {LEs}")
        #     print(f"Step {i}: Max Lyapunov Exponent = {np.max(LEs)}")
        #     # print(LEs)
    LEs = ltot / (n_compute * _dt)
    mle = np.max(LEs)

    return mle


# ============================================= 下面都是方法 =============================================
def gram_schmidt(X):
    p, m, n = X.shape
    X_new = X.reshape(p*m, n)
    Q = np.zeros_like(X_new)
    Znorm = np.zeros(n)

    # 处理第一列，初始化
    Znorm[0] = np.linalg.norm(X_new[:, 0])
    Q[:, 0] = X_new[:, 0] / Znorm[0] if Znorm[0] > 1e-10 else 0

    # 处理其他列
    for j in range(1, n):
        v = X_new[:, j].copy()
        # 使用NumPy矩阵操作计算所有之前列的投影并一次性减去
        projections = Q[:, :j].T @ v
        v -= Q[:, :j] @ projections
        norm = np.linalg.norm(v)
        Q[:, j] = v / norm if norm > 1e-10 else 0
        Znorm[j] = norm
    
    Q = Q.reshape(p, m, n)

    return Q, Znorm

@jit(nopython=True, parallel=True)
def gram_schmidt_jit(X):
    p, m, n = X.shape
    X_new = X.reshape(p*m, n)  # 将输入数据转换为连续数组
    Q = np.zeros_like(X_new)
    Znorm = np.zeros(n)

    for j in prange(n):
        v = np.ascontiguousarray(X_new[:, j])  # 确保v连续
        for i in prange(j):
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

# ===================== 在numba中使用Euler ===================== 
@jit(nopython=True)
def update_state(x, x_tau, dt, t, f_delay):
    return x + dt * f_delay(x, x_tau, t)

@jit(nopython=True)
def update_delta_x(delta_x, delta_x_tau, dt, df_dx, df_dx_tau):
    # 确保输入数组是连续的
    delta_x = np.ascontiguousarray(delta_x)
    delta_x_tau = np.ascontiguousarray(delta_x_tau)
    df_dx = np.ascontiguousarray(df_dx)
    df_dx_tau = np.ascontiguousarray(df_dx_tau)
    return delta_x + dt * (df_dx @ delta_x + df_dx_tau @ delta_x_tau)

# ===================== 在numba中使用rk4 ===================== 
@jit(nopython=True)
def rk4_step(x, x_tau, dt, t, f_delay):
    # 计算四个斜率
    k1 = f_delay(x, x_tau, t)
    k2 = f_delay(x + 0.5 * dt * k1, x_tau, t + 0.5 * dt)
    k3 = f_delay(x + 0.5 * dt * k2, x_tau, t + 0.5 * dt)
    k4 = f_delay(x + dt * k3, x_tau, t + dt)
    # 计算下一步的值
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

@jit(nopython=True)
def rk4_step_delta(delta_x, delta_x_tau, dt, df_dx, df_dx_tau):
    # 同样计算四个斜率
    k1 = df_dx @ delta_x + df_dx_tau @ delta_x_tau
    k2 = df_dx @ (delta_x + 0.5 * dt * k1) + df_dx_tau @ delta_x_tau
    k3 = df_dx @ (delta_x + 0.5 * dt * k2) + df_dx_tau @ delta_x_tau
    k4 = df_dx @ (delta_x + dt * k3) + df_dx_tau @ delta_x_tau
    # 计算下一步的值
    return delta_x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    '''
    # 模型参数
    a = 20
    b = 1
    c = np.pi / 4
    tau = 5
    def f_Ikeda(x, x_tau, t):   
        res = np.zeros_like(x)
        res[0] = a * (np.sin(x_tau[0] - c) ** 2) - b * x[0]

        return res
        
    def jac_delay(x, x_tau, t):
        # 线性化方程 (利用雅可比矩阵计算)
        """
        jac = [dF/dx, dF/dx_tau]
        jac = [-b, 2*a*sin(x_tau - c)*cos(x_tau - c)]
        """
        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))

        df_dx[0, 0] = -b
        df_dx_tau[0, 0] = 2*a*np.sin(x_tau[0] - c)*np.cos(x_tau[0]- c)

        return df_dx, df_dx_tau
    
    f_delay = f_Ikeda
    '''

    # 模型参数
    a = .2
    b = .1
    c = 10
    tau = 23
    
    @jit(nopython=True)
    def f_Mackey_Glass(x, x_tau, t):
        res = np.zeros_like(x)
        res[0] = a * (x_tau[0]/(1+x_tau[0] ** c)) - b * x[0]

        return res
    
    def jac_delay(x, x_tau, t):
        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))
        df_dx[0, 0] = -b
        df_dx_tau[0, 0] = a*(1/(1+x_tau[0] ** c) - c*(x_tau[0]**c)/((1+x_tau[0] ** c)**2))
        return df_dx, df_dx_tau

    f_delay = f_Mackey_Glass

    '''
    # 模型参数
    tau = 2*np.pi
   
    def f_Ikeda(x, x_tau, t):
        res = np.zeros_like(x)
        res[0] = np.sin(x_tau)

        return res
    
    def jac_delay(x, x_tau, t):
        df_dx = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau = np.zeros((x.shape[0], x.shape[0]))
        df_dx_tau[0, 0] = np.cos(x_tau)
        return df_dx, df_dx_tau

    f_delay = f_Ikeda
    '''

    x0 = np.array([0.9])
    T_init = int(1e3)
    T_cal = int(1e5)

    system = ContinuousDS(x0, f_delay, jac_delay, tau, N=200, jit=True, rk=True)
    LCE(system, T_init, T_cal, jit=True)

    