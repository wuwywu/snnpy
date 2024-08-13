# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/24
# User      : WuY
# File      : HH.py
# Hodgkin-Huxley(HH) 模型

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

class HH(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）
    dt ： 计算步长
    temperature: 温度(℃)
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="euler", dt=0.01, temperature=None):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        # 温度因子
        self.temperature = temperature      # 标准温度(℃) 实验温度为6.3
        if temperature is not None:
            self.phi = 3.0 ** ((temperature - 6.3) / 10)  # 温度系数
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
        self.th_up = 0  # 放电阈值
        self.th_down = -10  # 放电阈下值

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
        self.N_vars = 4  # 变量的数量
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
                 - self._g_L * (self.mem - self._E_L) + I[0]) / self._Cm
        dm_dt = 0.1 * (self.mem + 40.0) / (1.0 - np.exp(-(self.mem + 40) / 10.0)) * (1.0 - self.m) \
                - 4.0 * np.exp(-(self.mem + 65.0) / 18.0) * self.m + I[1]
        dn_dt = 0.01 * (self.mem + 55.0) / (1 - np.exp(-(self.mem + 55) / 10)) * (1 - self.n) \
                - 0.125 * np.exp(-(self.mem + 65.0) / 80) * self.n + I[2]
        dh_dt = 0.07 * np.exp(-(self.mem + 65.0) / 20.0) * (1.0 - self.h) \
                - 1.0 / (1.0 + np.exp(-(self.mem + 35.0) / 10.0)) * self.h + I[3]
        if self.temperature is not None:
            dm_dt *= self.phi
            dn_dt *= self.phi
            dh_dt *= self.phi

        return dmem_dt, dm_dt, dn_dt, dh_dt

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
        I[0, :] = self.Iex  # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._HH, I, self.mem, self.m, self.n, self.h)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进

    def retuen_vars(self):
        """
        用于输出所有状态变量
        """
        return [self.mem, self.m, self.n, self.h]

    def set_vars_vals(self, vars_vals=[0, 0, 0, 0]):
        """
        用于自定义所有状态变量的值
        """
        self.mem = vars_vals[0]*np.ones(self.num)
        self.m = vars_vals[1]*np.ones(self.num)
        self.n = vars_vals[2]*np.ones(self.num)
        self.h = vars_vals[3]*np.ones(self.num)


if __name__ == "__main__":
    N = 2
    method = "euler" # "rk4", "euler"
    models = HH(N=N, method=method, temperature=6.3)

    time = []
    mem = []
    se = spikevent(N)

    for i in range(10000):
        models()
        time.append(models.t)
        mem.append(models.mem.copy())
        se(models.t, models.flaglaunch)

    ax1 = plt.subplot(211)
    plt.plot(time, mem)
    plt.subplot(212, sharex=ax1)
    se.pltspikes()
    # print(se.Tspike_list)

    plt.show()
