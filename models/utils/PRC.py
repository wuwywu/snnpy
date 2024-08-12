# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/8/11
# User      : WuY
# File      : PRC.py
# 相位响应曲线 (Phase Response Curves, PRCs)
# refs : C. Börgers, An Introduction to Modeling Neuronal Dynamics,
# Springer International Publishing, Cham, 2017.
# https://doi.org/10.1007/978-3-319-51171-9.
# 描述 : 这个代码用于测量神经元的 "相位漂移" 和 "相位响应曲线"

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

# 定义 PRC 所使用的突触形式
class syn_chem:
    """
    AMPA-like synaptic input pulse
    这是参考文献中给的一种化学突触形式，这个化学突触作为 PRCs 的输入脉冲使用
    args:
        tau_peak : 突触门控变量到达峰值的时间，
            通过这个值控制 与突触前相关的 q 变量的时间参数 tau_d_q
            (使用的方法是二分法)
        dt : 算法的时间步长
        method : 计算非线性微分方程的方法，（"euler", "rk4")
    """
    def __init__(self, tau_peak=0.5, dt=0.01, method="euler"):
        method_options = ["euler", "rk4"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = self._euler
        if method == "rk4":   self.method = self._rk4
        self.tau_peak = tau_peak
        self.dt = dt
        self.params()
        self.vars()
        self.tau_d_q_function()

    def params(self):
        self.e = 0.         # 化学突触的平衡电位
        self.tau_r = 0.5    # 突触上升时间常数
        self.tau_d = 2.0    # 突触衰减时间常数
        self.g_syn = 0.1    # 突触的最大电导

    def vars(self):
        self.q = np.zeros(1)
        self.s = np.zeros(1)

    def reset(self):
        self.vars()

    def __call__(self, t, ts):
        if np.abs(t - ts) < (0.5 * dt):
            self.q[:] = 1
            print(".")
        self.method(self.syn_model, self.q, self.s)

        return self.q, self.s

    def syn_model(self):
        dq_dt = - self.q / self.tau_d_q
        ds_dt = self.q * (1 - self.s) / self.tau_r - self.s / self.tau_d

        return dq_dt, ds_dt

    def tau_peak_function(self, tau_d_q):
        """
        通过 tau_d_q 给出 s 的峰值时间
        """
        # 参数
        dt = self.dt
        dt05 = 0.5 * self.dt
        tau_r = self.tau_r
        tau_d = self.tau_d

        s = 0
        t = 0
        ds_dt = np.exp(-t / tau_d_q) * (1.0 - s) / tau_r - s * tau_d
        while ds_dt > 0:
            t_old = t
            ds_dt_old = ds_dt
            s_tmp = s + dt05 * ds_dt
            ds_dt_tmp = np.exp(-(t + dt05) / tau_d_q) * \
                        (1.0 - s_tmp) / tau_r - s_tmp / tau_d
            s = s + dt * ds_dt_tmp
            t = t + dt
            ds_dt = np.exp(-t / tau_d_q) * (1.0 - s) / tau_r - s / tau_d

        tau_peak_new = (t_old * (-ds_dt) + t * ds_dt_old) / (ds_dt_old - ds_dt)  # 线性插值法

        return tau_peak_new

    def tau_d_q_function(self):
        dt = self.dt
        tau_r = self.tau_r
        tau_d = self.tau_d
        tau_peak = self.tau_peak

        # 给 tau_d_q 设置一个区间
        tau_d_q_left = 1.0
        while self.tau_peak_function(tau_d_q_left) > tau_peak:
            tau_d_q_left *= 0.5

        tau_d_q_right = tau_r
        while self.tau_peak_function(tau_d_q_right) < tau_peak:
            tau_d_q_right *= 2.0

        # 使用二分法 (bisection method) 求出与 tau_peak 对应的 tau_d_q
        while tau_d_q_right - tau_d_q_left > 1e-12:
            tau_d_q_mid = 0.5 * (tau_d_q_left + tau_d_q_right)
            if (self.tau_peak_function(tau_d_q_mid) <= tau_peak):
                tau_d_q_left = tau_d_q_mid
            else:
                tau_d_q_right = tau_d_q_mid

        self.tau_d_q = 0.5 * (tau_d_q_left + tau_d_q_right)

    def _euler(self, models, *args):
        """
        使用 euler 算法计算非线性微分方程
        arg:
            models: 突触模型函数，返回所有dvars_dt
            *args： 输入所有变量，与dvars_dt一一对应
            # 注意只有一个变量的时候，返回必须为 dvar_dt, “,"是必须的
        """
        vars = list(args)  # 所有的变量
        dvars_dt = models()  # 所有变量的的微分方程
        lens = len(dvars_dt)  # 变量的数量
        for i in range(lens):  # 变量更新
            vars[i] += dvars_dt[i] * self.dt
        print(vars[0])

    def _rk4(self, models, *args):
        """
        使用 fourth-order Runge-Kutta(rk4) 算法计算非线性微分方程
        arg:
            models: 突触模型函数，返回所有dvars_dt
            *args： 输入所有变量，与dvars_dt一一对应
            # 注意只有一个变量的时候，返回必须为 dvar_dt, “,"是必须的
        """
        vars = list(args)  # 所有的变量
        original_vars = copy.deepcopy(vars)  # 原始状态备份
        lens = len(vars)  # 变量的数量
        dt = self.dt  # 时间步长
        # 计算k1
        k1 = models()
        # 计算k2
        for i in range(lens):
            vars[i] += original_vars[i] + 0.5 * dt * k1[i] - vars[i]
        k2 = models()
        # 计算k3
        for i in range(lens):
            vars[i] += original_vars[i] + 0.5 * dt * k2[i] - vars[i]
        k3 = models()
        # 计算k4
        for i in range(lens):
            vars[i] += original_vars[i] + dt * k3[i] - vars[i]
        k4 = models()

        # 最终更新vars
        for i in range(lens):
            vars[i] += original_vars[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 - vars[i]


if __name__ == "__main__":
    dt = 0.01
    syn = syn_chem(dt=dt)
    # print(syn.tau_d_q)
    s_list = []

    n = 700
    t_list = np.arange(0, n*dt, 0.01)
    for i in range(n):
        t = i * dt
        syn(t, ts=0.5)
        s_list.append(syn.s[0])

    plt.plot(t_list, s_list)
    plt.show()






