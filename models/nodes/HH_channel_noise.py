# encoding: utf-8
# Author    : YeZQ<phy_yzq@mails.ccnu.edu.com>
# Datetime  : 2024/3/10
# User      : YeZQ
# File      : HH_channel_noise.py
# Hodgkin-Huxley(HH) 模型加入与离子通道开关相关的通道噪声

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base import Neurons
from utils.utils import spikevent


class HH_cn(Neurons):
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

        # 通道噪声参数
        self.rhoNa = 60.0   # Na+通道密度(/μm2)
        self.rhok = 18.0    # K+通道密度(/μm2)
        self.S = 10.0       # 某个神经元的膜片区的面积(μm2)

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = -70*np.random.rand(self.num)
        self.m = 0.5 * np.random.rand(self.num)
        self.n = 1 * np.random.rand(self.num)
        self.h = 0.6 * np.random.rand(self.num)

    def _HH(self, I):
        alpha_m = (self.mem + 40)/((1-np.exp(-(self.mem + 40)/10))*10)
        beta_m = 4*np.exp(-(self.mem + 65)/18)
        alpha_n = (55 + self.mem)/(1-np.exp(-(55 + self.mem)/10))/100
        beta_n = 0.125*np.exp(-(self.mem + 65)/80)
        alpha_h = 0.07*np.exp(-(self.mem + 65)/20)
        beta_h = 1/(np.exp(-(35 + self.mem)/10) + 1)

        D_m = (alpha_m * beta_m) / (self.rhoNa * self.S * (alpha_m + beta_m))
        D_h = (alpha_h * beta_h) / (self.rhoNa * self.S * (alpha_h + beta_h))
        D_n = (alpha_n * beta_n) / (self.rhok * self.S * (alpha_n + beta_n))

        dmem_dt = (-self._g_Na * np.power(self.m, 3) * self.h * (self.mem - self._E_Na) \
                 - self._g_K * np.power(self.n, 4) * (self.mem - self._E_K) \
                 - self._g_L * (self.mem - self._E_L) + I) / self._Cm
        dm_dt = alpha_m * (1-self.m) - beta_m * self.m + self._noise(D_m)
        dn_dt = alpha_n * (1-self.n) - beta_n * self.n + self._noise(D_n)
        dh_dt = alpha_h * (1-self.h) - beta_h * self.h + self._noise(D_h)

        return dmem_dt, dm_dt, dn_dt, dh_dt

    def __call__(self, Io=0):
        I = self.Iex+Io        # 恒定的外部激励
        self.method(self._HH, I, self.mem, self.m, self.n, self.h)  
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进  

    def _noise(self, D):
        # Box-Muller
        noise = np.sqrt(-4*D*self.dt*np.log(np.random.rand(self.num)))*np.cos(2*np.pi*np.random.rand(self.num))
        # noise = np.random.normal(loc=0., scale=np.sqrt(2 * D * self.dt), size=self.num)
        return noise
        

if __name__ == "__main__":
    N = 10
    method = "euler"
    models = HH_cn(N=N, method=method, dt=0.01)
    models.S = .01
    models.Iex = 6

    time = []
    mem = []
    se = spikevent(N)

    for i in range(100000):
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
