# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/5/6
# User      : WuY
# File      : aEIF.py
# adaptive exponential integrate-and-fire(AdEx) 模型
# refences: STDP rule outlined in Clopath et al., Nat. Neurosci. 13(3), 344-352 (2010)

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

class aEIF(Neurons):
    """
    N : 建立神经元的数量
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, dt=0.25):
        super().__init__(N, dt=dt)
        self._params()
        self._vars()

    def _params(self):
        # 神经元模型参数
        self.C = 281                            # [pF] membrane capacitance
        self.g_L = 30                           # [nS] membrane conductance
        self.E_L = -70.6                        # [mV] resting voltage
        self.VT_rest = -50.4                    # [mV] resetting voltage
        self.Delta_T = 2                        # [mV] exponential parameters
        self.tau_w = 144                        # [ms] time constant for adaptation variable w
        self.a = 4                              # [nS] adaptation coupling constant
        self.b = 0.0805                         # spike triggered adaptation
        self.w_jump = 400                       # spike after depolarisation
        self.tau_wtail = 40                     # [ms] time constant for spike after depolarisation
        self.tau_VT = 50                        # [ms] time constant for VT
        self.VT_jump = 20                       # adaptive threshold

        # 尖峰设置参数
        self.th_up = 20                            # [mV] spike threshold

        self.t = 0          # 运行时间
        self.Iex = 0        # 恒定的外部激励 [pA]
        self.t_spike = 2    # 尖峰次序时间

    def _vars(self):
        self.mem = self.E_L * np.ones(self.num)  # membrane potential of output layer  变量u
        self.w = np.zeros(self.num)  # initial values
        self.w_tail = np.zeros(self.num)  # initial values
        self.V_T = -50.4 * np.ones(self.num)  # initial values
        self.N_vars = 4  # 变量的数量

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
        # I = np.zeros((self.N_vars, self.num))
        # I[0, :] = self.Iex  # 恒定的外部激励
        # I[axis, :] += Io
        I = Io + self.Iex

        self.updateVar(I)
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # 时间前进

    def updateVar(self, I):
        dmem_dt = 1 / self.C * (-self.g_L * (self.mem - self.E_L) + self.g_L * self.Delta_T     \
                             * np.exp((self.mem - self.V_T) / self.Delta_T) - self.w + self.w_tail + I)
        self.dw_dt = 1 / self.tau_w * (self.a * (self.mem - self.E_L) - self.w)

        self.mem += dmem_dt * self.dt  # 膜电位u
        self.w += self.dw_dt * self.dt  # hyperpolarizing adaptation current w_ad
        self.w_tail -= self.w_tail * self.dt / self.tau_wtail  # additional current z
        self.V_T = (self.VT_rest * self.dt / self.tau_VT    \
                    + (1 - self.dt / self.tau_VT) * self.V_T)  # the adaptive threshold V_T

    def _spikes_eval(self, mem):
        self.flaglaunch[:] = 0  # 重置放电开启标志

        # 处理尖峰
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))
        self.mem[firing_StartPlace] = 24.4  # 记录放电开始时间
        self.flag[firing_StartPlace] = 1

        # 模拟尖峰持续 (t_spike) ms
        conut = self.t_spike / self.dt - 1
        active_spikes = np.where((self.dt * self.flag < self.t_spike) & (self.flag > 0))
        self.mem[active_spikes] = 32.862 - 8.462 * np.abs(round(conut/2) - self.flag[active_spikes]) / conut
        self.w[active_spikes] -= self.dw_dt[active_spikes] * self.dt        # 保持w不变
        self.flag[active_spikes] += 1

        # 结束尖峰
        firing_endPlace = np.where(self.dt * self.flag >= self.t_spike)
        self.mem[firing_endPlace] = self.E_L + 15 + 6.0984  # about -49.5 mV
        self.w[firing_endPlace] += self.b
        self.w_tail[firing_endPlace] = self.w_jump
        self.V_T[firing_endPlace] = self.VT_jump + self.VT_rest
        self.flag[firing_endPlace] = 0

        self.flaglaunch[firing_StartPlace] = 1  # 放电开启标志
        self.firingTime[firing_StartPlace] = self.t  # 记录放电时间


if __name__ == "__main__":
    n = 2
    I_tot = 1000

    dt = 0.01  # [ms] 时间步长

    nodes = aEIF(n, dt=dt)
    nodes.t_spike = 2

    Tshop = 150  # 理论运行时间
    Tn = int(Tshop / dt)  # 循环次数
    # 记录理论运行时间

    time = []
    mem = []
    se = spikevent(n)

    for i in range(Tn):
        nodes(I_tot)
        time.append(nodes.t)
        mem.append(nodes.mem.copy())
        se(nodes.t, nodes.flaglaunch)

    ax1 = plt.subplot(211)
    plt.plot(time, mem)
    plt.subplot(212, sharex=ax1)
    se.pltspikes()
    # print(se.Tspike_list)

    plt.show()