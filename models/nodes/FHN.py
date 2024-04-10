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
from utils.utils import spikevent

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


if __name__ == "__main__":
    N = 2
    method = "euler"  # "rk4", "euler"
    models = FHN(N=N, method=method)

    time = []
    mem = []
    se = spikevent(N)

    for i in range(10000):
        # I = np.random.rand(2, N)*0.01
        # I = np.zeros((2, N))
        models(Io=1)
        time.append(models.t)
        mem.append(models.mem.copy())
        se(models.t, models.flaglaunch)

    ax1 = plt.subplot(211)
    plt.plot(time, mem)
    plt.subplot(212, sharex=ax1)
    se.pltspikes()
    # print(se.Tspike_list)

    plt.show()