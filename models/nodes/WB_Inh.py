# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/5/22
# User      : WuY
# File      : WB_Inh.py
# Wang-Buzs´aki Model of an Inhibitory Interneuron in Rat Hippocampus
# Description: Wang 和 Buzs´aki 提出了大鼠海马抑制性篮状细胞模型。
# # 篮状细胞之所以得名，是因为其轴突轴的分支形成篮状结构，围绕着其他细胞的细胞体。
# reference: X.-J. Wang and G. Buzs´aki, Gamma oscillation by synaptic inhibition
# in a hippocampal interneuronal network model,
# J. Neurosci., 16 (1996),pp. 6402–6413.

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

class WB_Inh(Neurons):
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
        # Wang-Buzs´aki Model HH模型
        self._g_Na = 35  # 钠离子通道的最大电导(mS/cm2)
        self._g_K = 9  # 钾离子通道的最大电导(mS/cm2)
        self._g_L = 0.1  # 漏离子电导(mS/cm2)
        self._E_Na = 55  # 钠离子的平衡电位(mV)
        self._E_K = -90  # 钾离子的平衡电位(mV)
        self._E_L = -65  # 漏离子的平衡电位(mV)
        self._Cm = 1.0  # 比膜电容(uF/cm2)
        self.Iex = 1.5  # 恒定的外部激励
        self.th_up = 0  # 放电阈值
        self.th_down = -10  # 放电阈下值

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = -70 * np.random.rand(self.num)
        # self.m = np.random.rand(self.num)
        self.n = np.random.rand(self.num)
        self.h = np.random.rand(self.num)
        self.N_vars = 3  # 变量的数量

    def _wb(self, I):
        alpha_n = 0.05 * (34 + self.mem) / (1 - np.exp(-0.1 * (34 + self.mem)))
        beta_n = 0.625 * np.exp(-(self.mem + 44) / 80)

        alpha_h = 0.35 * np.exp(-(self.mem + 58) / 20)
        beta_h = 5 / (np.exp(-0.1 * (28 + self.mem)) + 1)

        alpha_m = 0.1 * (self.mem + 35) / (1 - np.exp(-(35 + self.mem) / 10))
        beta_m = 4 * np.exp(-(self.mem + 60) / 18)
        self.m = alpha_m / (alpha_m + beta_m)

        dmem_dt = (-self._g_Na * np.power(self.m, 3) * self.h * (self.mem - self._E_Na) \
                   - self._g_K * np.power(self.n, 4) * (self.mem - self._E_K) \
                   - self._g_L * (self.mem - self._E_L) + I[0]) / self._Cm
        dn_dt = alpha_n * (1 - self.n) - beta_n * self.n + I[1]
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h + I[2]

        return dmem_dt, dn_dt, dh_dt

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

        self.method(self._wb, I, self.mem, self.n, self.h)
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    method = "euler"     # "rk4", "euler"
    models = WB_Inh(N=N, method=method)
    # models = RTM_HH(N=N, method=method)
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
