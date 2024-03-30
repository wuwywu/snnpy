# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/29
# User      : WuY
# File      : synapse_exp.py
# 这个文件集成了STDP的写法，主要是改变突触类种的self.w
# STDP1的写法：用e指数表示
# STDP2的写法：分别保存前后神经元的脉冲发放所引起的突触权重变化的迹(trace)


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse


class synbase_STDP(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem"]
    method: 计算非线性微分方程的方法，（"eluer", "rk4"）
    """
    def __init__(self, pre, post, conn=None, synType="electr", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        self._params()
        self._vars()

    def _params(self):
        # STDP1/2系数
        self.A_P = 0.1  # 突触权重最大变化量（增强）
        self.A_D = 0.095  # 突触权重最大变化量（抑制）
        self.tau_P = 20  # 突触权重变化量随时间差变化的时间常数（增强）
        self.tau_D = 20  # 突触权重变化量随时间差变化的时间常数（抑制）
        self.w_min = 0.  # 设定权重最小值
        self.w_max = 1.  # 设定权重最大值

    def _vars(self):
        # 0维度--post，1维度--pre
        self.Apre = np.zeros(self.pre.num)  # 突触前神经元突触权重变化的迹，STDP2
        self.Apost = np.zeros(self.post.num)  # 突触后神经元突触权重变化的迹，STDP2
        self.w = .5 * np.ones((self.post.num, self.pre.num))  # 设定连接权重，STDP1/2
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的

        # STDP规则改变权重self.w
        self._STDP1()

        I_post = self.syn()  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def _STDP1(self):
        # 获取所有突触前和突触后神经元的放电时间
        pre_fire_t = self.pre.firingTime[np.newaxis, :]   # 突触前放电时间
        post_fire_t = self.post.firingTime[:, np.newaxis]   # 突触后放电时间
        delta_t = post_fire_t - pre_fire_t

        # 获取所有突触前和突触后神经元的放电标志
        pre_spikes = self.pre.flaglaunch[np.newaxis, :]    # 突触前放电
        post_spikes = self.post.flaglaunch[:, np.newaxis]  # 突触后放电

        # 检查哪些连接是活跃的（即突触前或突触后神经元放电）
        active_connections = np.logical_or(pre_spikes, post_spikes) * self.conn

        # 使用Heaviside函数计算权重更新
        H_pos = np.where(delta_t > 0, 1, 0)  # 突触后神经元后放电
        H_neg = np.where(delta_t < 0, 1, 0)  # 突触前神经元后放电
        # 对于delta_t == 0的情况
        no_change = np.where(delta_t == 0, 0, 1)  # 当delta_t == 0时不更新权重

        # 应用STDP公式并乘以活跃连接矩阵，以确保只更新活跃连接的权重
        dw = (self.A_P * np.exp(-np.abs(delta_t) / self.tau_P) * H_pos -
              self.A_D * np.exp(-np.abs(delta_t) / self.tau_D) * H_neg) * active_connections * no_change

        # 权重变化
        self.w +=  self.w * dw  #self.w

        # 确保权重在最小值和最大值之间
        self.w = np.clip(self.w, self.w_min, self.w_max)

    def _stdp_dA_dt(self):
        dApre_dt = -self.Apre / self.tau_P
        dApost_dt = -self.Apost / self.tau_D
        return dApre_dt, dApost_dt

    def _STDP2(self):
        # 迭代计算Apre,Apost和g
        self.method(self._stdp_dA_dt, self.Apre, self.Apost)

        # 更新Apre
        prespikingPlace = np.where(self.pre.flaglaunch == 1)[0]
        self.Apre[prespikingPlace] += self.A_P

        # 更新Apost
        postspikingPlace = np.where(self.post.flaglaunch == 1)[0]
        self.Apost[postspikingPlace] += self.A_D

        # 更新w (0维度--post，1维度--pre)
        self.w[:, prespikingPlace] += -self.Apost[:, None]
        self.w[postspikingPlace, :] += self.Apre[None, :]

        # 确保权重在最小值和最大值之间
        self.w = np.clip(self.w, self.w_min, self.w_max)  # 限定范围


