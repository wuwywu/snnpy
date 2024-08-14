# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/24
# User      : WuY
# File      : FHN.py
# FitzHugh-Nagumo(FHN) 模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Neurons
from utils.utils_f import spikevent

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

class FHN(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）这个模型最好使用 "rk4"
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="rk4", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self.a = 0.08
        self.b = 0.8
        self.c = 0.7
        self.Iex = 1  # 恒定的外部激励
        self.th_up = 1  # 放电阈值
        self.th_down = 1  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        self.N_vars = 2 # 变量的数量
        # self.t = 0  # 运行时间

    def _fhn(self, I):
        dmem_dt = self.mem - self.mem ** 3 / 3 - self.y + I[0]
        dy_dt = self.a * (self.mem + self.c - self.b * self.y) + I[1]
        return dmem_dt, dy_dt

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        I = np.zeros((self.N_vars, self.num))
        I[0, :] = self.Iex        # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._fhn, I, self.mem, self.y)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进

    def retuen_vars(self):
        """
        用于输出所有状态变量
        """
        return [self.mem, self.y]

    def set_vars_vals(self, vars_vals=[0, 0, 0]):
        """
        用于自定义所有状态变量的值
        """
        self.mem = vars_vals[0]*np.ones(self.num)
        self.y = vars_vals[1]*np.ones(self.num)


class FHN2(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self.a = 0.5
        self.b = 0.05
        self.Iex = 0  # 恒定的外部激励
        self.th_up = 1  # 放电阈值
        self.th_down = 1  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = np.zeros(self.num)
        self.y = np.zeros(self.num)
        self.N_vars = 2 # 变量的数量
        # self.t = 0  # 运行时间

    def _fhn(self, I):
        dmem_dt = (self.mem - self.mem ** 3 / 3 - self.y + I[0])/self.b
        dy_dt = self.mem + self.a + I[1]
        return dmem_dt, dy_dt

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        I = np.zeros((self.N_vars, self.num))
        I[0, :] = self.Iex        # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._fhn, I, self.mem, self.y)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


class FHN2_mag(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem

    包含 magnetic flux 项的 FHN 模型
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self.a = 0.5
        self.b = 0.05
        self.Iex = 0  # 恒定的外部激励
        self.th_up = 1  # 放电阈值
        self.th_down = 1  # 停止放电阈值
        # 磁通量项的常数
        self.k = .1    # 反馈增益，其中k桥接了磁通或磁场对膜电位的耦合和调制
        self.k1 = .1    # K1计算了离子在细胞中传输引起的电磁感应效应
        self.k2 = 1.    # K2描述了极化度和磁化强度，也被认为是调节磁通饱和的物理参数
        self.alpha = 0.1
        self.beta = 0.1

    def _vars(self):
        # 模型变量
        self.mem = np.zeros(self.num)
        self.y = np.zeros(self.num)
        self.phi = np.zeros(self.num)
        self.N_vars = 3 # 变量的数量
        # self.t = 0  # 运行时间

    def _fhn(self, I):
        rho = self.alpha + 3*self.beta*(self.phi**2)
        dmem_dt = (self.mem - self.mem ** 3 / 3 - self.y + I[0]
                   - self.k*rho*self.mem)/self.b
        dy_dt = self.mem + self.a + I[1]
        dphi_dt = self.k1*self.mem - self.k2*self.phi + I[2]
        return dmem_dt, dy_dt, dphi_dt

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        I = np.zeros((self.N_vars, self.num))
        I[0, :] = self.Iex        # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._fhn, I, self.mem, self.y, self.phi)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


# ================================= memristor-based FHN (mFHN) =================================
# references: Y. Xie, Z. Ye, X. Li, X. Wang, Y. Jia, A novel memristive neuron model and its energy characteristics,
# Cogn. Neurodynamics. https://doi.org/10.1007/s11571-024-10065-5
class mFHN(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self.a = 0.3
        self.b = 1.0
        self.c = 0.1
        # 电磁的参数
        self.k1 = 1.0
        self.k2 = 1.8
        self.alpha = 0.1
        self.beta = 0.3
        # 外部信号源参数
        self.A = 0.8
        self.omega = 0.9

        self.Iex = 0  # 恒定的外部激励
        self.th_up = 1  # 放电阈值
        self.th_down = 1  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = 0.2*np.ones(self.num)
        self.y = 0.1*np.ones(self.num)
        self.phi = 0.05*np.ones(self.num)
        self.N_vars = 3  # 变量的数量
        # self.t = 0  # 运行时间

    def _mfhn(self, I):
        us = 0.2 + self.A * np.cos(self.omega * self.t) # 外部信号源
        rho = self.alpha + self.beta * (self.phi ** 2)
        dmem_dt = -rho * (self.mem - us) + self.mem - self.mem ** 3 / 3 - self.y + I[0]
        dy_dt = self.c * (self.mem + self.a - self.b * self.y) + I[1]
        dphi_dt = -self.k1 * (self.mem - us) - self.k2 * self.phi + I[2]

        return dmem_dt, dy_dt, dphi_dt

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        I = np.zeros((self.N_vars, self.num))
        I[0, :] = self.Iex        # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._mfhn, I, self.mem, self.y, self.phi)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    method = "rk4"  # "rk4", "euler"
    # models = FHN(N=N, method=method)
    # models = FHN2_mag(N=N, method=method)
    # models.k = -10.
    models = mFHN(N=N, method=method)
    models.omega = 0.01

    time = []
    mem = []
    y = []
    se = spikevent(N)

    for i in range(50000):
        models()

    for i in range(200000):
        # I = np.random.rand(2, N)*0.01
        # I = np.zeros((2, N))
        models()
        time.append(models.t)
        mem.append(models.mem.copy())
        y.append(models.y.copy())
        se(models.t, models.flaglaunch)

    plt.figure(figsize=(6, 12))
    ax1 = plt.subplot(311)
    plt.plot(time, mem)
    plt.subplot(312, sharex=ax1)
    se.pltspikes()
    ax3 = plt.subplot(313)
    plt.plot(mem, y)
    # print(se.Tspike_list)

    plt.show()

