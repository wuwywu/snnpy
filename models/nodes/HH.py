# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/24
# User      : WuY
# File      : HH.py
# Hodgkin-Huxley(HH)模型

from base import Neurons
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

class HH(Neurons):
    """
    N : 建立神经元的数量
    """
    def __init__(self, N=1, method="eluer", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        # HH模型
        self._g_Na = 120  # 钠离子通道的最大电导(mS/cm2)
        self._g_K = 36  # 钾离子通道的最大电导(mS/cm2)
        self._g_L = 0.3  # 漏离子电导(mS/cm2)
        self._E_Na = 50  # 钠离子的平衡电位(mV)
        self._E_K = -77  # 钾离子的平衡电位(mV)
        self._E_L = -54.4  # 漏离子的平衡电位(mV)
        self._Cm = 1.0  # 比膜电容(uF/cm2)
        self.Iex = 10   # 恒定的外部激励
        # self._q10 = 1
        # self.th_up = 0  # 放电阈值
        # self.th_dowm = -10  # 放电阈下值
        # 电磁
        # self.a1 = 0.4
        # self.b1 = 0.02
        # self.k1 = 0.0  # 0.06
        # self.k2 = 0.01
        # self.k3 = 0.2

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = np.random.uniform(-.3, .3, self.num)
        self.m = 0.5 * np.random.rand(self.num)
        self.n = 1 * np.random.rand(self.num)
        self.h = 0.6 * np.random.rand(self.num)
        # 电磁
        # self.f = 0.1 * np.random.rand(self.num)

    def _HH(self, I):
        # _g_Na = self._g_Na; _g_K = self._g_K; _g_L = self._g_L
        # _E_Na = self._E_Na; _E_K = self._E_K; _E_L = self._E_L
        # _Cm = self._Cm
        #电磁
        # I_f = -self.k1*(self.a1+3*self.b1*self.f*self.f)*self.x
        # HH模型
        dmem_dt = (-self._g_Na * np.power(self.m, 3) * self.h * (self.mem - self._E_Na) \
                 - self._g_K * np.power(self.n, 4) * (self.mem - self._E_K) \
                 - self._g_L * (self.mem - self._E_L) + I) / self._Cm
        dm_dt = 0.1 * (self.mem + 40.0) / (1.0 - np.exp(-(self.mem + 40) / 10.0)) * (1.0 - self.m) \
                - 4.0 * np.exp(-(self.mem + 65.0) / 18.0) * self.m
        dn_dt = 0.01 * (self.mem + 55.0) / (1 - np.exp(-(self.mem + 55) / 10)) * (1 - self.n) \
                - 0.125 * np.exp(-(self.mem + 65.0) / 80) * self.n
        dh_dt = 0.07 * np.exp(-(self.mem + 65.0) / 20.0) * (1.0 - self.h) \
                - 1.0 / (1.0 + np.exp(-(self.mem + 35.0) / 10.0)) * self.h

        return dmem_dt, dm_dt, dn_dt, dh_dt

    def __call__(self):
        I = self.Iex        # 恒定的外部激励
        self.method(self._HH, I, self.mem, self.m, self.n, self.h)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 10
    method = "eluer" # "rk4", "eluer"
    models = HH(N=N, method=method)

    time = []
    mem = []

    for i in range(10000):
        models()
        time.append(models.t)
        mem.append(models.mem[0])

    plt.plot(time, mem)

    plt.show()
