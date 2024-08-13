# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/4/24
# User      : WuY
# File      : Chialvo.py
# Chialvo map 模型
# 描述 : 二维映射模型，有类似于神经元的动力学性质，有周期解与混沌解
# reference : D.R. Chialvo, Generic excitable dynamics on a two-dimensional map, Chaos Solit. Fract. 5(3-4), 461-479 (1995).
import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import DiscreteDS
from utils.utils_f import spikevent

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

class Chialvo(DiscreteDS):
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
        # 模型常数
        self.a = 0.89
        self.b = 0.6    # 0.6 to 0.18 oscillations to aperiodic burst
        self.c = 0.28
        self.k = 0.03

        self.Iex = 0  # 恒定的外部激励

        self.th_up = 1  # 放电阈值
        self.th_down = 1 # 放电阈下值

    def _vars(self):
        self.t = 0  # 运行时间
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        self.N_vars = 2  # 变量的数量

    def _Chialvo(self, I):
        mem_new = (self.mem**2)*np.exp(self.y-self.mem)+self.k + I[0]
        y_new = self.a*self.y-self.b*self.mem+self.c + I[1]
        return mem_new, y_new

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到模型的外部激励，
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

        self.method(self._Chialvo, I, self.mem, self.y)  #
        self._spikes_eval(self.mem)  # 放电测算

        self.t += 1  # 时间前进


if __name__ == "__main__":
    N = 2
    models = Chialvo(N=N)
    models.b = 0.18

    time = []
    mem = []
    se = spikevent(N)
    for i in range(500):
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
