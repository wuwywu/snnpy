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
    def __init__(self, N=1, tau_peak=0.5, dt=0.01, method="euler"):
        self.num = N  # 输入突触数量
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
        self.q = np.zeros(self.num)
        self.s = np.zeros(self.num)

    def reset(self):
        self.vars()

    def __call__(self, t, ts):
        """
        args:
            ts : list/numpy 所有峰放电的时间
        """
        ts = np.array(ts)
        # if np.abs(t - ts) < (0.5 * self.dt):
        self.q[np.abs(t - ts) < (0.5 * self.dt)] = 1
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


class phase_shift:
    """
    这个代码，给出神经元的相位漂移
    args:
        node : （类）节点类
        phase : list 刺激相位 range(0, 1)

    输入脉冲被设置在第5个到第6个峰之间，代码定位到：_node_init
        self.ts_list = self.in_phase*self.T + self.T_spike_list[4]
    里面的重要参数：
        T_spike_list  : 没有输入脉冲时，spikes的时间
        T_spike_act_list : 有输入脉冲时，spikes的时间
        ts_list :   给输入脉冲的时间
        in_phase :  给输入脉冲的的相位
        mem_no_in  :   没有输入脉冲时，膜电位的变化
        mem_in :       有输入脉冲时，膜电位的变化
    """
    def __init__(self, nodes, phase=[0.5], method="euler"):
        self.nodes = nodes                            # 输入节点
        self.dt = self.nodes.dt
        self.num = len(phase)
        self.in_phase = np.array(phase)
        self.syn_in = syn_chem(self.num, dt=self.dt, method=method)       # 实例化输入突触
        self._params()
        self._vars()
        self._node_init()

    def _params(self):
        self.T_init =10000      # 初始化节点时间
        self.e = 0.             # 化学突触的平衡电位
        self.g_syn = 0.1        # 突触的最大电导

        self.th_up = self.nodes.th_up  # 放电阈值
        self.th_down = self.nodes.th_down  # 放电阈下值

    def _vars(self):
        # 放电变量
        self.max_init = -np.inf                     # 初始最大值
        self.max = -np.inf + np.zeros(self.num)     # 初始变化最大值
        self.nn = np.zeros(self.num, dtype=int)            # 记录每个节点的ISI的个数
        self.flag = np.zeros(self.num, dtype=int)          # 放电标志
        self.T_pre = np.zeros(self.num)         # 前峰时间
        self.T_post = np.zeros(self.num)        # 后峰时间

    def __call__(self):
        mem = self.nodes.mem
        self.T_spike_act_list = np.zeros((self.num, 10))  # 记录刺激后的峰

        t_final = 5000  # 最大初始时间
        self.mem_in = []
        self.I_in = []
        while self.nn.min() < 8 and self.nodes.t < t_final / self.dt:
            t = self.nodes.t
            q, s = self.syn_in(t, self.ts_list)
            I = self.g_syn * s * (self.e - mem)
            self.I_in.append(I.copy())
            self.nodes(I)
            self._spikes_eval(self.nodes.mem)  # 放电测算

            self.mem_in.append(self.nodes.mem.copy())

        self.mem_in= np.array(self.mem_in)

    def _node_init(self):
        """
        这个函数的作用是：
            1、重置节点初始值（所有节点初始值都设定为一样）
            2、重置节点数量
            3、重置节点运行时间
            4、给出节点的振荡周期
            5、通过相位给出输入脉冲刺激的时间
        """
        # 初始化节点
        for i in range(self.T_init):
            self.nodes()
        # 记录初始值，重置时间
        self.nodes.t = 0.       # 初始化运行时间
        vars_init = np.array(self.nodes.retuen_vars())[:, 0]  # 设定变量初始值

        # ================================== 记录没有输入脉冲时峰值时间 ==================================
        mem = self.nodes.mem
        th_up = self.th_up        # 放电阈值
        th_down = self.th_down    # 放电阈下值
        max_init = -np.inf              # 初始最大值（负无穷大）
        max = -np.inf + np.zeros(1)     # 初始变化最大值（负无穷大）
        nn = np.zeros(1)                # 记录每个节点的ISI的个数
        flag = np.zeros(1)              # 放电标志
        T_pre = np.zeros(1)             # 前峰时间
        T_post = np.zeros(1)            # 后峰时间

        self.T_spike_list = []               # 记录峰的时间

        t_final = 5000                  # 最大初始时间
        ISI_list = []
        self.mem_no_in = []
        while nn[0]<10 and self.nodes.t<t_final/self.dt:
            # 运行节点
            self.nodes()
            self.mem_no_in.append(self.nodes.mem[0])

            t = self.nodes.t
            # -------------------- 放电开始 --------------------
            firing_StartPlace = np.where((mem > th_up) & (flag == 0))  # 放电开始的位置
            flag[firing_StartPlace] = 1  # 放电标志改为放电
            # -------------------- 放电期间 --------------------
            firing_Place = np.where((mem > max) & (flag == 1))         # 放电期间并且还没有到达峰值
            max[firing_Place] = mem[firing_Place]
            T_post[firing_Place] = t                                   # 存储前面峰的时间
            # -------------------- 放电结束 --------------------
            firing_endPlace = np.where((mem < th_down) & (flag == 1))  # 放电结束的位置
            firing_endPlace2 = np.where((mem < th_down) & (flag == 1) & (nn > 2))  # 放电结束的位置2
            flag[firing_endPlace] = 0  # 放电标志改为放电
            nn[firing_endPlace] += 1  # 结束放电ISI数量+1

            ISI = T_post[firing_endPlace2] - T_pre[firing_endPlace2]        # ISI（峰峰间隔，周期）
            ISI_list.extend(ISI)

            T_pre[firing_endPlace] = T_post[firing_endPlace]
            self.T_spike_list.extend(T_post[firing_endPlace])

            max[firing_endPlace] = max_init

        # 初始化 `节点初始值`，`初始时间` 和 `节点数量`；给出振荡周期
        self.nodes.num = self.num
        self.nodes._fvars()
        self.nodes.set_vars_vals(vars_vals=vars_init)
        self.nodes.t = 0.  # 初始化运行时间

        self.T = ISI_list[-1]

        # 通过相位给给出 `输入脉冲` 时间(第5个峰后添加，可以修改)
        self.ts_list = self.in_phase*self.T + self.T_spike_list[4]

    def _spikes_eval(self, mem):
        """
        测试放电
        """
        # -------------------- 放电开始 --------------------
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))  # 放电开始的位置
        self.flag[firing_StartPlace] = 1  # 放电标志改为放电
        # -------------------- 放电期间 --------------------
        firing_Place = np.where((mem > self.max) & (self.flag == 1))  # 放电期间并且还没有到达峰值
        self.max[firing_Place] = mem[firing_Place]
        self.T_post[firing_Place] = self.nodes.t
        #  -------------------- 放电结束 -------------------
        firing_endPlace = np.where((mem < self.th_down) & (self.flag == 1))  # 放电结束的位置
        firing_endPlace2 = np.where((mem < self.th_down) & (self.flag == 1) & (self.nn > 2))  # 放电结束的位置2
        self.flag[firing_endPlace] = 0  # 放电标志改为放电
        self.nn[firing_endPlace] += 1  # 结束放电ISI数量+1

        self.T_pre[firing_endPlace] = self.T_post[firing_endPlace]
        if firing_endPlace[0].size != 0:
            # 给出放电的坐标
            coordinates = np.stack((firing_endPlace[0], self.nn[firing_endPlace]-1), axis=-1)
            # print(firing_endPlace[0])
            self.T_spike_act_list[coordinates[:, 0], coordinates[:, 1]] = self.T_post[firing_endPlace]

        self.max[firing_endPlace] = self.max_init

    def plot_phase_shift(self):
        x_l = self.mem_in.shape[0]
        x_ = np.arange(x_l)*self.dt
        fig, axs = plt.subplots(self.num, sharex="all", layout='constrained')
        for i in range(self.num):
            axs[i].plot(x_, self.mem_in[:, i])
            axs[i].plot(x_, self.mem_no_in[:x_l], color="r")
            axs[i].axvline(self.ts_list[i], color='k', linestyle='--', lw=2)

        plt.xlim(self.T_spike_list[3], self.T_spike_list[6])




if __name__ == "__main__":
    # ================== 测试 "相位漂移" 和 "相位响应曲线" 所使用的输入脉冲 ==================
    dt = 0.01
    syn_test = syn_chem(2, dt=dt)
    # print(syn.tau_d_q)
    s_list = []
    s1_list = []

    n = 700
    t_list = np.arange(0, n * dt, 0.01)
    for i in range(n):
        t = i * dt
        syn_test(t, ts=[0.5, 1.])
        s_list.append(syn_test.s[0])
        s1_list.append(syn_test.s[1])

    fig, axs = plt.subplots(2, sharex="all", layout='constrained')
    axs[0].plot(t_list, s_list)
    axs[1].plot(t_list, s1_list)

    # ================== 测试 "相位漂移" ==================
    from nodes.HH import HH
    node = HH()
    phi_shift = phase_shift(node, phase=[0.4, 0.6])
    phi_shift.g_syn = 0.2
    phi_shift()
    phi_shift.plot_phase_shift()
    plt.show()






