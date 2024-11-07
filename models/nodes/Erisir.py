# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/07
# User      : HuangW
# File      : Erisir.py
# Erisir models
# A. Erisir, D. Lau, B. Rudy, and C. S. Leonard, Function of specific  K(+) channels in sustained high-frequency firing of fast-spiking neocortical interneurons,
# J. Neurophysiol., 82 (1999), pp. 2476–89.


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Neurons
from utils.utils_f import spikevent


class Erisir(Neurons):
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
        self._g_Na = 112  # 钠离子通道的最大电导(mS/cm2)
        self._g_K = 224  # 钾离子通道的最大电导(mS/cm2)
        self._g_L = 0.5  # 漏离子电导(mS/cm2)
        self._E_Na = 60  # 钠离子的平衡电位(mV)
        self._E_K = -90  # 钾离子的平衡电位(mV)
        self._E_L = -70  # 漏离子的平衡电位(mV)
        self._Cm = 1.0  # 比膜电容(uF/cm2)
        self.Iex = 0.   # 恒定的外部激励
        
        self.th_up = 0  # 放电阈值
        self.th_down = -10  # 放电阈下值        

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = -5*np.random.rand(self.num)
        # self.m = np.random.rand(self.num)
        self.n = np.random.rand(self.num)
        self.h = np.random.rand(self.num)
        self.N_vars = 3  # 变量的数量

    def _Erisir(self, I):
        alpha_n = (95- self.mem) / (np.exp((95-self.mem)/11.8)-1)
        beta_n = 0.025*np.exp(-self.mem/22.222)
        
        alpha_h = 0.0035*np.exp(-self.mem/24.186)
        beta_h = -0.017*(self.mem+51.25)/(np.exp(-(self.mem+51.25)/5.2) -1)   

        alpha_m = (40*(75.5 -self.mem))/(np.exp((75.5 -self.mem)/13.5) -1)
        beta_m  = 1.2262 *np.exp(-self.mem/42.248)
        self.m = alpha_m/(alpha_m + beta_m)

        dmem_dt = (-self._g_Na * np.power(self.m, 3) * self.h * (self.mem - self._E_Na) \
                 - self._g_K * np.power(self.n, 2) * (self.mem - self._E_K) \
                 - self._g_L * (self.mem - self._E_L) + I[0]) / self._Cm
        dn_dt = alpha_n*(1-self.n) - beta_n*self.n + I[1]
        dh_dt = alpha_h*(1-self.h) - beta_h*self.h + I[2]

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

        self.method(self._Erisir, I, self.mem, self.n, self.h)
        self._spikes_eval(self.mem)  # 放电测算 
        
        self.t += self.dt  # 时间前进

    def retuen_vars(self):
        """
        用于输出所有状态变量
        """
        return [self.mem, self.n, self.h]

    def set_vars_vals(self, vars_vals=[0, 0, 0]):
        """
        用于自定义所有状态变量的值
        """
        self.mem = vars_vals[0]*np.ones(self.num)
        self.n = vars_vals[1]*np.ones(self.num)
        self.h = vars_vals[2]*np.ones(self.num)

    


if __name__ == "__main__":
    N = 2
    method = "euler"     # "rk4", "euler"
    models = Erisir(N=N, method=method)
    # models = RTM_HH(N=N, method=method)
    models.Iex = 7

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