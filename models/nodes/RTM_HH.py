# encoding: utf-8
# Author    : YeZQ<phy_yzq@mails.ccnu.edu.com>
# Datetime  : 2024/3/3
# User      : YeZQ
# File      : RTM_HH.py
# Reduced Traub-Miles Hodgkin-Huxley(HH) 模型
# Description: 这是对 Ermentrout 和 Kopell 提出的模型的轻微修改,
# 又是 Traub 和 Miles 提出的大鼠海马锥体兴奋细胞模型的简化
# https://doi.org/10.1073/pnas.95.3.1259
# https://doi.org/10.1017/CBO9780511895401

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Neurons
from utils.utils import spikevent

class RTM_HH(Neurons):
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
        # Reduced Traub-Miles HH模型
        self._g_Na = 100  # 钠离子通道的最大电导(mS/cm2)
        self._g_K = 80  # 钾离子通道的最大电导(mS/cm2)
        self._g_L = 0.1  # 漏离子电导(mS/cm2)
        self._E_Na = 50  # 钠离子的平衡电位(mV)
        self._E_K = -100  # 钾离子的平衡电位(mV)
        self._E_L = -67  # 漏离子的平衡电位(mV)
        self._Cm = 1.0  # 比膜电容(uF/cm2)
        self.Iex = 1.5   # 恒定的外部激励
        self.th_up = 0  # 放电阈值
        self.th_down = -10  # 放电阈下值        

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = -70*np.random.rand(self.num)
        # self.m = np.random.rand(self.num)
        self.n = np.random.rand(self.num)
        self.h = np.random.rand(self.num)

    def _HH(self, I):
        alpha_n = 0.032 * (52 + self.mem) / (1 - np.exp(-(52 + self.mem) / 5))
        beta_n = 0.5 * np.exp(-(self.mem + 57) / 40)
        alpha_h = 0.128 * np.exp(-(self.mem + 50) / 18)
        beta_h = 4/(np.exp(-(27 + self.mem)/10) + 1)

        alpha_m = 0.32 * (self.mem + 54) / (1 - np.exp(-(54 + self.mem) / 4))
        beta_m = 0.28 * (self.mem + 27) / (np.exp((self.mem + 27) / 5) - 1)
        self.m = alpha_m/(alpha_m + beta_m)

        dmem_dt = (-self._g_Na * np.power(self.m, 3) * self.h * (self.mem - self._E_Na) \
                 - self._g_K * np.power(self.n, 4) * (self.mem - self._E_K) \
                 - self._g_L * (self.mem - self._E_L) + I) / self._Cm
        dn_dt = alpha_n*(1-self.n) - beta_n*self.n
        dh_dt = alpha_h*(1-self.h) - beta_h*self.h

        return dmem_dt, dn_dt, dh_dt

    def __call__(self, Io=0):
        I = self.Iex+Io        # 恒定的外部激励
        self.method(self._HH, I, self.mem, self.n, self.h)
        self._spikes_eval(self.mem)  # 放电测算 
        
        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    method = "euler"     # "rk4", "euler"
    models = RTM_HH(N=N, method=method)
    models.Iex = 0.75

    t_final = 100
    dt = 0.01
    m_steps = int(t_final / dt)

    time = []
    mem = []
    se = spikevent(N)

    for i in range(m_steps):
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
