# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/25
# User      : WuY
# File      : HR.py
# Hindmarsh-Rose(HR) 模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base import Neurons
from utils.utils import spikevent

seed = 0
np.random.seed(seed)                # 给numpy设置随机种子

class HR(Neurons):
    """
    N : 建立神经元的数量
    method ： 计算非线性微分方程的方法，（"eluer", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
        self._params()
        self._vars()

    def _params(self):
        # Parameters for the Hindmarsh-Rose model
        # 混沌簇放电 (a =1.;b =3.;c =1.;d =5.;s =4;r =0.006;xR =-1.6;Iex =3.)
        self.a = 1.
        self.b = 3.
        self.c = 1.
        self.d = 5.
        self.r = 0.006
        self.s = 4.
        self.xR = -1.6
        self.Iex = 1.6  # 恒定的外部激励
        self.th_up = 1.  # 放电阈值
        self.th_down = 1.  # 停止放电阈值

    def _vars(self):
        # 模型变量
        self.mem = np.random.rand(self.num) - 1.5  # Membrane potential variable, initialized randomly
        self.y = np.random.rand(self.num) - 10.  # Recovery variable
        self.z = np.random.rand(self.num) - 0.5  # Adaptation variable
        # self.t = 0  # 运行时间

    def _hr(self, I):
        dx_dt = self.y - self.a * self.mem ** 3 + self.b * self.mem ** 2 - self.z + I
        dy_dt = self.c - self.d * self.mem ** 2 - self.y
        dz_dt = self.r*(self.s * (self.mem - self.xR) - self.z)
        return dx_dt, dy_dt, dz_dt

    def __call__(self):
        I = self.Iex  # External stimulus
        # Update the variables using the chosen numerical method
        self.method(self._hr, I, self.mem, self.y, self.z)
        # Evaluation of spikes
        self._spikes_eval(self.mem)  # 放电测算

        self.t += self.dt  # Time step forward


if __name__ == "__main__":
    N = 2
    method = "eluer"  # "rk4", "eluer"
    models = HR(N=N, method=method, dt=0.001)
    models.Iex = 3.
    time = []
    mem = []
    se = spikevent(N)

    for i in range(1000000):
        models()
        time.append(models.t)
        mem.append(models.mem.copy())
        se(models.t, models.flaglaunch)

    plt.plot(time, mem)
    plt.figure()
    se.pltspikes()
    # print(se.Tspike_list)

    plt.show()
