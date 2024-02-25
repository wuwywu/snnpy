# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/24
# User      : WuY
# File      : HH.py
# FitzHugh-Nagumo(FHN) 模型

from base import Neurons
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)                # 给numpy设置随机种子

class FHN(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"eluer", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="eluer", dt=0.01):
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
        self.th_dowm = 1  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        # self.t = 0  # 运行时间

    def _fhn(self, I):
        dmem_dt = self.mem - self.mem ** 3 / 3 - self.y + I
        dy_dt = self.a * (self.mem + self.c - self.b * self.y)
        return dmem_dt, dy_dt

    def __call__(self):
        I = self.Iex        # 恒定的外部激励
        self.method(self._fhn, I, self.mem, self.y)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 10
    method = "eluer"  # "rk4", "eluer"
    models = FHN(N=N, method=method)

    time = []
    mem = []

    for i in range(10000):
        models()
        time.append(models.t)
        mem.append(models.mem[0])

    plt.plot(time, mem)

    plt.show()
