# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/5/1
# User      : WuY
# File      : multicompartment.py
# Reduced multicompartmental model
# refernce : Bono, J., Clopath, C., 2017. Modeling somatic and dendritic spike mediated plasticity
# at the single neuron and network level. Nat Commun 8, 706. https://doi.org/10.1038/s41467-017-00740-z

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import spikevent


class multicompartment:
    """
    N : 建立神经元的数量
    N_D : 树突的数量
    dt ： 计算步长
    神经元的膜电位都写为：mem
    """
    def __init__(self, N=1, N_D=1, dt=0.025):
        self.num = N  # 神经元数量
        self.N_D = N_D # 树突的数量
        self.dt = dt  # 积分步长
        self._params()
        self._vars()

    def _params(self):
        self.Iex = 0*np.zeros(self.num)           # 胞体的外部激励
        # 模型参数
        self.E_L = -69          # 静息电位 (mV)
        self.g_L = 40           # leak conductance (nS)
        self.C_m = 281          # 莫电容 (pF)
        self.spineFactor = 1.5  # 突触脊状结构，泄漏电导和膜电容增加了1.5倍
        self.v_reset = -55      # reset voltage after spike [mV]
        self.thresh = 20            # spike threshold [mV]
        # 适应性阈值电位(指数电流)相关的参数
        self.VT_rest = -50.4    # Adaptive threshold rest value (mV)
        self.delta_T = 2        # Slope factor (mV) 描述钠电流激活的指数项的斜率因子
        self.tau_VT = 50       # Adaptive threshold 时间常数
        self.VT_jump = 20       # adaptive threshold 重置
        # 尖峰计算相关
        self.spike_width_soma = 1        # spike width soma [ms]
        self.spike_width_dend = 1        # spike width dendrite [ms]
        self.spike_latency = 0.3         # latency of the spike in dendrites [ms]
        self.refrac_time = 0             # absolute refractory time (total abs. refractory time = spike_width_dend+spike_latency+refrac_time) [ms]
        # 噪声相关
        self.Noise_tau = 20                   # time constant for the noise [ms]
        self.Noise_avg = 0                    # mean of noise [pA] 35
        self.Noise_std = 0                    # std of noise [pA] 3.5

    def _vars(self):
        self.t = 0  # 运行时间
        self.flag = np.zeros(self.num, dtype=int)        # 模型放电标志(>0, 放电)
        self.flaglaunch = np.zeros(self.num, dtype=int)  # 模型开始放电标志(==1, 放电刚刚开始)
        self.firingTime = np.zeros(self.num)             # 模型放电时间(上次放电)
        # 与尖峰宽度相关的延迟
        delay_time = self.spike_width_dend + self.spike_latency
        self.mem_prox_delay = self.E_L*np.ones((self.num, self.N_D, int(np.round(delay_time / self.dt))))
        self.mem_dist_delay = self.E_L * np.ones((self.num, self.N_D, int(np.round(delay_time / self.dt))))

        self.I_noise = np.zeros(self.num)       # 噪声

        self.mem_soma = self.E_L*np.ones(self.num)              # 初始化胞体膜电位值
        # 指数电流模拟了钠在胞体中的快速激活，而在树突室中不存在
        self.V_T_soma = self.E_L*np.ones(self.num)              # 初始化 Adaptive threshold
        self.mem_prox = self.E_L*np.ones((self.num, self.N_D))    # 初始化树突近端膜电位值
        self.mem_dist = self.E_L*np.ones((self.num, self.N_D))    # 初始化树突远端膜电位值

    def _multicompartment(self, I):
        I_soma_syn, I_prox_syn, I_dist_syn = I  # 胞体，树突近端，远端受到的输入电流(如：突触电流)
        # leak current
        I_L_soma = self.g_L * (self.E_L - self.mem_soma)
        g_L_spine = self.spineFactor * self.g_L
        I_L_prox = g_L_spine * (self.E_L - self.mem_prox)
        I_L_dist = g_L_spine * (self.E_L - self.mem_dist)

        # Exponential current (fast Na activation in soma) ** 只有胞体有 **
        I_exp = self.g_L * self.delta_T * np.exp((self.mem_soma - self.V_T_soma) / self.delta_T)

        # =================== 轴向电流 axial currents ===================
        # 胞体与近端之间的轴向电流变化(胞体没有尖峰的时候)
        backpr_som_prox = 0.02      # 0.04
        backpr_prox_dist = 0.15     # 15  #0.09
        V_fact_prox = 25
        I_soma_prox = 2500 * (self.mem_soma[:, None] - self.mem_prox) * (self.flag[:, None] == 0)
        I_prox_dist = 1500 * (self.mem_prox - self.mem_dist)

        tval21 = I_soma_prox * (I_soma_prox > 0) + V_fact_prox * backpr_som_prox * I_soma_prox * (I_soma_prox < 0)
        tval22 = I_prox_dist * (I_prox_dist > 0) + backpr_prox_dist * I_prox_dist * (I_prox_dist < 0)

        tval31 = backpr_prox_dist * I_prox_dist * (I_prox_dist < 0)
        tval32 = I_prox_dist * (I_prox_dist > 0)

        I_axial_soma = -(backpr_som_prox * I_soma_prox * (I_soma_prox < 0)).sum(1)  # (num, )
        I_axial_prox = tval21 - tval22  # (num, N_D)
        I_axial_dist = tval31 + tval32  # (num, N_D)
        # print(I_axial_soma, I_axial_prox, I_axial_dist)

        # =================== 胞体膜电位演化 ===================
        # I_noise = 0
        mem_soma_new = self.mem_soma + (I_L_soma + I_exp + I_axial_soma + self.Iex + self.I_noise + I_soma_syn) * self.dt / self.C_m
        self.noise_color()  # 更新噪声
        V_T_soma_new = self.V_T_soma + (self.VT_rest - self.V_T_soma) * self.dt / self.tau_VT

        # =================== 树突膜电位演变 ===================
        C_matr = self.spineFactor * self.C_m    # 树突的膜电容
        mem_prox_new = self.mem_prox + (I_L_prox + I_axial_prox + I_prox_syn) * self.dt / C_matr
        mem_dist_new = self.mem_dist + (I_L_dist + I_axial_dist + I_dist_syn) * self.dt / C_matr

        # =================== 存储树突的延迟值 ===================
        self.mem_prox_delay[:, :, :-1] = self.mem_prox_delay[:, :, 1:]
        self.mem_prox_delay[:, :, -1] = self.mem_prox
        self.mem_dist_delay[:, :, :-1] = self.mem_dist_delay[:, :, 1:]
        self.mem_dist_delay[:, :, -1] = self.mem_dist

        return mem_soma_new, V_T_soma_new, mem_prox_new, mem_dist_new

    def __call__(self, I_soma_syn=0, I_prox_syn=0, I_dist_syn=0):
        """
        I_soma_syn : 胞体受到的外界电流 [num, ]
        I_prox_syn : 近端受到的外界电流 [num, N_D] or float
        I_dist_syn : 远端受到的外界电流 [num, N_D] or float
        """
        I = [I_soma_syn, I_prox_syn, I_dist_syn]
        mem_soma_new, V_T_soma_new, mem_prox_new, mem_dist_new = self._multicompartment(I)
        self._spikes_eval(mem_soma_new, V_T_soma_new, mem_prox_new, mem_dist_new)

        self.t += self.dt  # 时间前进

    def _spikes_eval(self, mem_soma_new, V_T_soma_new, mem_prox_new, mem_dist_new):
        self.flaglaunch[:] = 0      # 重置放电开启标志

        spike_Height_soma  = 30     # 在产生一个尖峰后(>20 mV)，胞体电压维持1ms 为 30 mV
        spike_Height_prox = 10      # 在产生一个尖峰后(>20 mV)，树突近端电压维持 1ms 为 10 mV
        spike_Height_dist = -3      # 在产生一个尖峰后(>20 mV)，树突近端电压维持 1ms 为 -3 mV

        spike_width_soma = self.spike_width_soma        # spike width soma [ms]
        spike_width_dend = self.spike_width_dend        # spike width dendrite [ms]
        diff_width = spike_width_dend - spike_width_soma    # 尖峰宽度差
        spike_latency = self.spike_latency              # latency of the spike in dendrites [ms]
        refrac_time = self.refrac_time                  # absolute refractory time (total abs. refractory time = spike_width_dend+spike_latency+refrac_time) [ms]

        # =================== 根据放电标志，更新电压 ===================
        spike_cont = self.flag*self.dt      # 计算处于放电状态时间
        # 更新胞体电压
        mem_soma1 = mem_soma_new * (spike_cont < spike_width_soma) + self.v_reset * (spike_cont >= spike_width_soma)

        self.mem_soma[:] = spike_Height_soma * (spike_cont < spike_width_soma) * (mem_soma_new > self.thresh)  \
                     + mem_soma1 * ((spike_cont >= spike_width_soma) | (mem_soma_new <= self.thresh))

        # 更新树突电压
        mem_prox_delay = self.mem_prox_delay[:, :, 0]
        mem_dist_delay = self.mem_dist_delay[:, :, 0]

        spike_cont_d = spike_cont[:, None]

        dend_size = np.ones_like(mem_prox_new)      # 树突的形状 (num, N_D)
        dend_width = spike_width_soma+diff_width+spike_latency

        dv1 = (spike_cont_d >= dend_width) | (self.flag[:, None] == 0)
        dv2 = (spike_cont_d < spike_latency) * (spike_cont_d > 0)
        dv3 = (spike_cont_d >= spike_latency) & (spike_cont_d < dend_width) & (spike_cont_d > 0)

        mem_prox1 = mem_prox_new * (spike_cont_d < dend_width) + mem_prox_delay * (spike_cont_d >= dend_width)
        self.mem_prox[:, :] = mem_prox1 * dv1 + mem_prox1 * dv2 + np.maximum(mem_prox_delay, (spike_Height_prox * dend_size)) * dv3

        # mem_dist1 = mem_dist_new * (spike_cont < dend_width) + (spike_cont >= dend_width)
        mem_dist1 = mem_dist_new * (spike_cont_d < dend_width) + mem_dist_delay * (spike_cont_d >= dend_width)
        self.mem_dist[:, :] = mem_dist1 * dv1 + mem_dist1 * dv2 + np.maximum(mem_dist_delay, (spike_Height_dist * dend_size)) * dv3

        # =================== 根据放电标志，更新适应性阈值电压 V_T ===================
        self.V_T_soma[:] = V_T_soma_new * (np.abs(spike_cont - spike_width_soma) >= self.dt)  \
                        + (self.VT_jump + self.VT_rest) * (np.abs(spike_cont - spike_width_soma) < self.dt)

        # =================== 更新放电标志 ===================
        dv4 = (self.flag == 0) * (mem_soma_new>self.thresh)     # 放电开启
        self.flag = (spike_cont <= (spike_width_soma + diff_width + spike_latency + refrac_time))   \
                * ((self.flag+1)*(self.flag != 0) + dv4)

        firing_StartPlace = np.where(self.flag == 1)
        self.flaglaunch[firing_StartPlace] = 1  # 放电开启标志
        self.firingTime[firing_StartPlace] = self.t  # 记录放电时间
        # print(spike_cont)

    def noise_color(self):
        epi = np.random.normal(size=self.num)     # 标准白噪
        self.I_noise += self.dt / self.Noise_tau * ((self.Noise_avg - self.I_noise)  + epi * self.Noise_std * np.sqrt(2 * self.Noise_tau))


if __name__ == "__main__":
    N = 30
    N_D = 3 # 树突的数量

    models = multicompartment(N, N_D)
    models.Iex = 1200.0     # pA
    models.Noise_avg = 35   # pA
    models.Noise_std = 50   # pA

    for i in range(10000):
        models()

    time = []
    mem_soma = []
    mem_prox = []
    mem_dist = []
    for i in range(10000):
        models()
        time.append(models.t)
        mem_soma.append(models.mem_soma.copy())
        mem_prox.append(models.mem_prox.copy())
        mem_dist.append(models.mem_dist.copy())

    mem_prox1 = np.array(mem_prox)
    mem_dist1 = np.array(mem_dist)
    # print(mem_prox1.shape)
    # print(models.mem_soma)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(time, mem_soma)
    axs[0].set_title("soma")

    axs[1].plot(time, mem_prox1[:, :, 0])
    axs[1].set_title("proximal")

    axs[2].plot(time, mem_dist1[:, :, 0])
    axs[2].set_title("distal")

    fig.tight_layout()

    plt.show()

