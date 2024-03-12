# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/07
# User      : WuY
# File      : Iz.py
# Izhikevich(Iz) 模型

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

class Iz(Neurons):
    """
    Izhikevich 脉冲神经元
    reference： E.M. Izhikevich, Simple model of spiking neurons, IEEE Transactions on neural networks, 14(6), 1569-1572 (2003).
    v' = 0.04v^2 + 5v + 140 -u + I
    u' = a(bv-u)
    下面是将Izh离散化的写法
    if v>= thresh:
        v = c
        u = u + d

    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"euler", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        self._params()
        self._vars()

    def _params(self):
        """
        excitatory neurons: a=0.02, b=0.2, c=−65, d=8
        inhibitory neurons: a=0.02, b=0.25, c=−65, d=2.
        """
        self.a = .02
        self.b = .2
        self.c = -65.
        self.d = 8.
        self.Iex = 10  # 恒定的外部激励
        self.threshold = 30.    # 神经元发放脉冲需要达到的阈值

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = np.random.uniform(-.10, .10, self.num)
        self.u = np.random.rand(self.num)

    def _Iz(self, I):
        dmem_dt = 0.04 * self.mem * self.mem + 5 * self.mem - self.u + 140 + I
        du_dt = self.a * (self.b * self.mem - self.u)
        return dmem_dt, du_dt

    def __call__(self, Io=0):
        I = self.Iex + Io  # 恒定的外部激励
        self.method(self._Iz, I, self.mem, self.u)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进

    def _spikes_eval(self, mem):
        self.flaglaunch[:] = 0  # 重置放电开启标志
        firing_StartPlace = np.where(mem > self.threshold)
        self.flaglaunch[firing_StartPlace] = 1  # 放电开启标志
        self.firingTime[firing_StartPlace] = self.t  # 记录放电时间

        self.mem = self.mem * (1 - self.flaglaunch) + self.flaglaunch * self.c
        self.u += self.flaglaunch * self.d


if __name__ == "__main__":
    N = 2
    dt = .01
    method = "euler"  # "rk4", "euler"
    models = Iz(N=N, method=method, dt=dt)

    time = []
    mem = []
    se = spikevent(N)

    for i in range(20000):
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
