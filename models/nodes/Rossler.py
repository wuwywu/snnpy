# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/4/28
# User      : WuY
# File      : Rossler.py
# Rössler system 模型
# refernce : O.E. Rössler, An equation for continuous chaos, Phys. Lett. A 57(5), 397-398 (1976).


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Nodes

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

class Rossler(Nodes):
    """
    N: 创建节点的数量
    method ： 计算非线性微分方程的方法，（"eluer", "rk4"）
    dt ： 计算步长

    第一个状态变量都写为：mem
    运行时间：t; 时间步长：dt
    节点数量：num
    """
    def __init__(self, N, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        self._alpha = 0.2
        self._beta = 0.2
        self._gamma = 9.

        self.Iex = 0   # 恒定的外部激励

    def _vars(self):
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        self.z = np.random.rand(self.num)

        self.N_vars = 3

    def _Rossler(self, I):
        dmem_dt = -self.y - self.z + I[0]
        dy_dt = self.mem + self._alpha * self.y + I[1]
        dz_dt = self._beta + (self.mem - self._gamma)*self.z + I[2]

        return dmem_dt, dy_dt, dz_dt

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

        self.method(self._Rossler, I, self.mem, self.y, self.z)

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler"
    models = Rossler(N=N, method=method, dt=dt)

    time = []
    mem = []

    for i in range(100000):
        models()
        time.append(models.t)
        mem.append(models.mem.copy())

    plt.plot(time, mem)

    plt.show()
