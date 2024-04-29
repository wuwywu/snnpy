# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/4/29
# User      : WuY
# File      : Rulkov.py
# Rulkov map 模型
# refernce : N.F. Rulkov, Regularization of Synchronized Chaotic Bursts. Phys. Rev. Lett. 86(1), 183-186 (2001).

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import DiscreteDS
from utils.utils import spikevent

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

class Rulkov(DiscreteDS):
    """
    N : 建立节点(神经元)的数量

    第一维的状态变量(神经元的膜电位)都写为：mem
    """
    def __init__(self, N=1):
        super().__init__(N)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self._alpha = 4.3
        self._beta = 0.001
        self._sigma = 0.001

        self.Iex = 0  # 恒定的外部激励

        self.th_up = .5  # 放电阈值
        self.th_down = 5.  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        self.N_vars = 2 # 变量的数量
        # self.t = 0  # 运行时间

    def _rulkov(self, I):
        mem_new = self._alpha/(1+self.mem**2) + self.y + I[0]
        y_new = self.y - self._sigma*self.mem - self._beta + I[1]
        return mem_new, y_new

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
        I[0, :] = self.Iex        # 恒定的外部激励
        I[axis, :] += Io

        self.method(self._rulkov, I, self.mem, self.y)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += 1  # 时间前进


if __name__ == "__main__":
    N = 2
    models = Rulkov(N=N)

    time = []
    mem = []
    se = spikevent(N)

    for i in range(2000):
        models()

    for i in range(2000):
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
