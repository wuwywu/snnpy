# encoding: utf-8
# Author    : HuangWF
# Datetime  : 2024/3/13
# User      : WuY
# File      : synapse_ds.py
# ds/dt形式的化学突触模型
# 这个突触形式与离子通道的开关有关

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse


class synbase_ds(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_ds"]
    """
    def __init__(self, pre, post, conn=None, synType="chem_ds", method="euler"):
        super().__init__(pre=pre, post= post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_ds":
            self.syn = self.syn_chem_ds  # ds/dt_化学突触
        self._params()
        self._vars()

    def _params(self):
        # 0维度--post，1维度--pre
        self.w = .1*np.ones((self.post.num, self.pre.num)) # 设定连接权重
        # Alpha_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.Vshp = 5  # 突触是否起作用的膜电位阈值

    def _vars(self):
        self.t = self.post.t
        self.s = 0* np.random.rand(self.pre.num)

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """

        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t    # 这个是非常重要的
        I_post = self.syn()     # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def ds_dt(self, pre_mem=None):
        if pre_mem is None: pre_mem = self.pre.mem
        ds_dt = 2*(1-self.s)/(1+np.exp(-pre_mem/self.Vshp))-self.s
        return ds_dt

    def syn_chem_ds(self, post_mem=None):
        """
        args:
            pre_fire_t: 突触前的放电时间
            post_mem： 突触后的膜电位
        """
        if post_mem is None: post_mem = self.post.mem

        g_syn = self.s * np.expand_dims((self.e - post_mem), axis=1)
        Isyn = (self.w * self.conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre

        #更新化学突触中的s
        self.method(self.ds_dt, self.s)

        return Isyn


#包含时间延迟
class syn_delay_ds(Synapse): 
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_ds"]
    delayer: 延迟器
    """
    def __init__(self, pre, post, conn=None, synType="chem_ds", method="euler", delayer=None):
        super().__init__(pre=pre, post= post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.delayer = delayer
        self.synType = synType
        if self.synType == "electr":
            self.syn = self.syn_electr  # 电突触
        elif self.synType == "chem_ds":
            self.syn = self.syn_chem_ds  # ds/dt_化学突触
        self._params()
        self._vars()

    def _params(self):
        # 0维度--post，1维度--pre
        self.w = .1*np.ones((self.post.num, self.pre.num)) # 设定连接权重
        # Alpha_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.Vshp = 5  # 突触是否起作用的膜电位阈值

    def _vars(self):
        self.t = self.post.t
        self.s = 0* np.random.rand(self.pre.num)

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t    # 这个是非常重要的
        I_post = self.syn()     # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def syn_electr(self, pre_mem=None, post_mem=None):
        """
        args:
            pre_mem: 突触前的膜电位
            post_mem: 突触后的膜电位
        """
        if pre_mem is None: pre_mem = self.pre.mem
        if post_mem is None: post_mem = self.post.mem
        pre_mem = self.delayer(self.pre.mem)    # 存储延迟，并给出延迟的值
        vj_vi = pre_mem-np.expand_dims(post_mem, axis=1)   # pre减post
        Isyn = (self.w*self.conn*vj_vi).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn

    def ds_dt(self, pre_mem=None):
        if pre_mem is None: pre_mem = self.pre.mem
        pre_mem = self.delayer(self.pre.mem)
        ds_dt = 2*(1-self.s)/(1+np.exp(-pre_mem/self.Vshp))-self.s
        return ds_dt
    
    def syn_chem_ds(self, post_mem=None):
        """
        args:
            pre_fire_t: 突触前的放电时间
            post_mem: 突触后的膜电位
        """
        if post_mem is None: post_mem = self.post.mem

        g_syn = self.s * np.expand_dims((self.e - post_mem), axis=1)
        Isyn = (self.w * self.conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre

        #更新化学突触中的s
        self.method(self.ds_dt,self.s)

        return Isyn


#包含时间延迟和stdp
class syn_delay_stdp_ds(Synapse):
    def __init__(self, pre, post, conn=None, synType="chem_ds", method="euler", delayer=None):
        super().__init__(pre=pre, post= post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.delayer = delayer
        self.synType = synType
        if self.synType == "electr":
            self.syn = self.syn_electr  # 电突触
        elif self.synType == "chem_ds":
            self.syn = self.syn_chem_ds  # ds/dt_化学突触
        self._params()
        self._vars()

    def _params(self):
        # ds/dt_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.Vshp = 5  # 突触是否起作用的膜电位阈值
        # STDP系数
        # A_D*tau_D>A_P*tau_P: 抑制占主导；A_D*tau_D<A_P*tau_P: 增强占主导
        self.w_min = 0.001      # 设定权重最小值
        self.w_max = 1.0        # 设定权重最大值
        self.A_P = 1  # 可塑性导致突触强度变化（增强）
        self.tau_P = 20.    # 确定一个尖峰跟随另一个尖峰的时间（增强）
        self.A_D = 1.05  # 可塑性导致突触强度变化（抑制）
        self.tau_D = 20.    # 确定一个尖峰跟随另一个尖峰的时间（抑制）
        self.lr = 1e-4    # 学习率

    def _vars(self):
        # 0维度--post，1维度--pre
        self.w = .5 * np.ones((self.post.num, self.pre.num))  # 设定连接权重
        self.t = self.post.t
        self.s = 0* np.random.rand(self.pre.num)

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t    # 这个是非常重要的
        I_post = self.syn()     # 突触后神经元接收的突触电流
        self._STDP()

        self.t += self.dt  # 时间前进

        return I_post
    
    def _STDP(self):
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

    def syn_electr(self, pre_mem=None, post_mem=None):
        """
        args:
            pre_mem: 突触前的膜电位
            post_mem: 突触后的膜电位
        """
        if pre_mem is None: pre_mem = self.pre.mem
        if post_mem is None: post_mem = self.post.mem
        pre_mem = self.delayer(self.pre.mem)    # 存储延迟，并给出延迟的值
        vj_vi = pre_mem-np.expand_dims(post_mem, axis=1)   # pre减post
        Isyn = (self.w*self.conn*vj_vi).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn

    def ds_dt(self, pre_mem=None):
        if pre_mem is None: pre_mem = self.pre.mem
        pre_mem = self.delayer(self.pre.mem)
        ds_dt = 2*(1-self.s)/(1+np.exp(-pre_mem/self.Vshp))-self.s
        return ds_dt
    
    def syn_chem_ds(self, post_mem=None):
        """
        args:
            pre_fire_t: 突触前的放电时间
            post_mem: 突触后的膜电位
        """
        if post_mem is None: post_mem = self.post.mem

        g_syn = self.s * np.expand_dims((self.e - post_mem), axis=1)
        Isyn = (self.w * self.conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre

        # 更新化学突触中的s
        self.method(self.ds_dt,self.s)

        return Isyn

