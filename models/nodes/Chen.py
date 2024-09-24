# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/9/24
# User      : WuY
# File      : Chen.py
# Chen’s system 模型
# refernce : G. Chen and T. Ueta, Int. J. Bifurcation Chaos Appl. Sci. Eng. 9, 1465 (1999).


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

class Chen(Nodes):
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
        self._a = 35.
        self._c = 28.
        self._beta = 8/3.

        self.Iex = 0   # 恒定的外部激励

    def _vars(self):
        self.mem = np.random.rand(self.num)
        self.y = np.random.rand(self.num)
        self.z = np.random.rand(self.num)

        self.N_vars = 3

    def _Chen(self, I):
        dmem_dt = self._a * (self.y - self.mem) + I[0]
        dy_dt = (self._c - self._a - self.z) * self.mem + self._c * self.y + I[1]
        dz_dt = self.mem * self.y - self._beta*self.z + I[2]

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

        self.method(self._Chen, I, self.mem, self.y, self.z)

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler"
    models = Chen(N=N, method=method, dt=dt)

    time = []
    mem = []

    for i in range(1000):
        models()
        time.append(models.t)
        mem.append(models.mem.copy())

    plt.plot(time, mem)

    plt.show()
