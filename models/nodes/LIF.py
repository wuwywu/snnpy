# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/19
# User      : WuY
# File      : LIF.py
# leaky integrate-and-fire 模型
# description: 一个包含了不应期的LIF模型

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

class LIF(Neurons):
    """
    leaky integrate-and-fire 脉冲神经元
    tau * v' = -(V - V_rest) + RI
    其中：tau = RC
    下面是将LIF离散化的写法
    if v>= thresh:
        v = V_reset last t_ref

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
        # 神经元模型参数
        self.V_rest = 0.     # 神经元静息电位
        self.tau = 10.      # 时间常数 tau = RC
        self.R = 1.         # 细胞膜电阻
        self.Iex = 22.       # 恒定的外部激励
        # 尖峰设置参数
        self.threshold = 20.  # 神经元发放脉冲需要达到的阈值
        self.V_reset = -5.  # 发放spike后，膜电位重设
        self.t_ref = 5.     # 绝对不应期持续时间

    def _vars(self):
        self.t = 0  # 运行时间
        self.mem = np.random.uniform(self.V_reset, self.threshold, self.num)
        self.N_vars = 1  # 变量的数量

    def _lif(self, I):
        dmem_dt = (-self.mem + self.V_rest + self.R * I[0]) / self.tau
        # 注意只有一个变量的时候，返回必须为 dvar_dt, “,"是必须的
        return dmem_dt,

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

        self.method(self._lif, I, self.mem)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进

    def _spikes_eval(self, mem):
        self.flaglaunch[:] = 0  # 重置放电开启标志
        self.flag[:] = 0        # 重置处于放电标志
        firing_StartPlace = np.where(mem > self.threshold)
        self.flaglaunch[firing_StartPlace] = 1  # 放电开启标志
        self.flag[firing_StartPlace] = 1        # 处于放电标志
        self.firingTime[firing_StartPlace] = self.t  # 记录放电时间

        self.mem[firing_StartPlace] = self.V_reset

        # 不应期
        refractory = (self.t - self.firingTime) <= self.t_ref
        self.mem[refractory] = self.V_reset


if __name__ == "__main__":
    N = 2
    dt = .01
    method = "euler"  # "rk4", "euler"
    models = LIF(N=N, method=method, dt=dt)

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


