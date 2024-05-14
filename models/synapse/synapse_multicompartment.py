# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/5/8
# User      : WuY
# File      : synapse_multicompartment.py
# refernce : Bono, J., Clopath, C., 2017. Modeling somatic and dendritic spike mediated plasticity
# at the single neuron and network level. Nat Commun 8, 706. https://doi.org/10.1038/s41467-017-00740-z

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse


class synbase_AMPA_multicompartment:
    """
    pre: 网络前节点
    post: 网络后节点
    conn: 连接矩阵 (post_num, pre_num)
    conn_site : 连接位置 ["prox", "dist"]
    """
    def __init__(self, pre, post, conn=None, conn_site="prox"):
        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.post_N_D = self.post.N_D   # 突触后树突数量
        # 连接矩阵拓展为 (post_num, pre_num) --> (post_num, post_N_D, pre_num)
        self.conn = np.repeat(conn[:, None, :], self.post_N_D, axis=1) 
        self.dt = post.dt               # 计算步长
           
        site_options = ["prox", "dist"]
        if conn_site not in site_options:
            raise ValueError(f"无效选择，conn_site在{site_options}选择")
        self.conn_site = conn_site  # 连接位置

        self._params()
        self._vars()

    def _params(self):
        self.e = 0            # 化学突触的平衡电位 [mV]
        self.tau_AMPA = 2     # 化学突触的时间常数 [ms]
        self.g_max = 60      # 化学突触的最大电导 [nS]

        # 0维度--post，1维度--post_dend，2维度--pre
        self.w = .1 * np.ones((self.post.num, self.post.N_D, self.pre.num))  # 设定连接权重 (post_num, post_N_D, pre_num)

    def _vars(self):
        # 0维度--post，1维度--post_dend，2维度--pre
        self.g = np.zeros((self.post.num, self.post.N_D, self.pre.num))   # 设定突触电导
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t            # 这个是非常重要的
        I_post = self.syn_AMPA()        # 突触后神经元接收的突触电流 (post_num, post_N_D)

        self.t += self.dt  # 时间前进

        return I_post
    
    def syn_AMPA(self):
        pre_fire_launch = self.pre.flaglaunch           # 突触后胞体放电
        if self.conn_site == "prox":
            post_dend_mem = self.post.mem_prox          # 突触后树突膜电位 近端 "prox" (post_num, post_N_D)
            dend_att = 1                                # attenuation: from prox to soma 0.95
        else:
            # 通过将这些隔室的AMPA和NMDA电流乘以2.5倍来模拟远端突触增加的输入电阻
            post_dend_mem = self.post.mem_dist          # 突触后树突膜电位 远端 "dist" (post_num, post_N_D)
            dend_att = 0.4                              # attenuation: from dist to soma 0.4

        Isyn = ((1 / dend_att)*self.g * self.conn * np.expand_dims((self.e - post_dend_mem), axis=2)).sum(axis=2)  # 突触后树突接受到的突触电流(post_num, post_N_D)

        dg_dt = -self.g / self.tau_AMPA     # (post_num, post_N_D, pre_num)
        self.g += dg_dt * self.dt           # (post_num, post_N_D, pre_num)

        # 0维度--post，1维度--post_dend，2维度--pre
        self.g += self.w * self.g_max * pre_fire_launch[None, None, :]     # (post_num, post_N_D, pre_num)

        return Isyn


class synbase_AMPA_multicompartment_delay:
    """
    pre: 网络前节点
    post: 网络后节点
    conn: 连接矩阵 (post_num, pre_num)
    conn_site : 连接位置 ["prox", "dist"]
    delayer: 延迟器
    """
    def __init__(self, pre, post, conn=None, conn_site="prox", delayer=None):
        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.post_N_D = self.post.N_D   # 突触后树突数量
        # 连接矩阵拓展为 (post_num, pre_num) --> (post_num, post_N_D, pre_num)
        self.conn = np.repeat(conn[:, None, :], self.post_N_D, axis=1) 
        self.dt = post.dt               # 计算步长
           
        site_options = ["prox", "dist"]
        if conn_site not in site_options:
            raise ValueError(f"无效选择，conn_site在{site_options}选择")
        self.conn_site = conn_site  # 连接位置
        self.delayer = delayer      # 延时器
        self._params()
        self._vars()

    def _params(self):
        self.e = 0            # 化学突触的平衡电位 [mV]
        self.tau_AMPA = 2     # 化学突触的时间常数 [ms]
        self.g_max = 60      # 化学突触的最大电导 [nS]

        # 0维度--post，1维度--post_dend，2维度--pre
        self.w = .1 * np.ones((self.post.num, self.post.N_D, self.pre.num))  # 设定连接权重 (post_num, post_N_D, pre_num)

    def _vars(self):
        # 0维度--post，1维度--post_dend，2维度--pre
        self.g = np.zeros((self.post.num, self.post.N_D, self.pre.num))   # 设定突触电导
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t            # 这个是非常重要的
        I_post = self.syn_AMPA()        # 突触后神经元接收的突触电流 (post_num, post_N_D)

        self.t += self.dt  # 时间前进

        return I_post
    
    def syn_AMPA(self):
        pre_fire_launch = self.pre.flaglaunch                   # 突触后胞体放电
        pre_fire_launch_delay = self.delayer(pre_fire_launch)   # 存储现在的值，并给出延迟的值
        if self.conn_site == "prox":
            post_dend_mem = self.post.mem_prox          # 突触后树突膜电位 近端 "prox" (post_num, post_N_D)
            dend_att = 1                                # attenuation: from prox to soma 0.95
        else:
            # 通过将这些隔室的AMPA和NMDA电流乘以2.5倍来模拟远端突触增加的输入电阻
            post_dend_mem = self.post.mem_dist          # 突触后树突膜电位 远端 "dist" (post_num, post_N_D)
            dend_att = 0.4                              # attenuation: from dist to soma 0.4

        Isyn = ((1 / dend_att)*self.g * self.conn * np.expand_dims((self.e - post_dend_mem), axis=2)).sum(axis=2)  # 突触后树突接受到的突触电流(post_num, post_N_D)

        dg_dt = -self.g / self.tau_AMPA     # (post_num, post_N_D, pre_num)
        self.g += dg_dt * self.dt           # (post_num, post_N_D, pre_num)

        # 0维度--post，1维度--post_dend，2维度--pre
        self.g += self.w * self.g_max * pre_fire_launch_delay[None, None, :]     # (post_num, post_N_D, pre_num)

        return Isyn
    

# 为了在网络中允许更高的速率而不导致无界的活动，我们通过以下方式在胞体处近似一个抑制性电流
class synbase_inh_multicompartment:
    """
    pre: 网络前节点
    post: 网络后节点
    conn: 连接矩阵 (post_num, pre_num)
    conn_site : 连接位置 ["prox", "dist"]
    delayer: 延迟器
    """
    def __init__(self, pre, post, conn=None, delayer=None):
        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.conn = conn                # 连接矩阵
        self.dt = post.dt               # 计算步长

        self.delayer = delayer      # 延时器
        self._params()
        self._vars()

    def _params(self):
        self.e = -70            # 化学突触的平衡电位 [mV]
        self.tau_inhib = 0.02     # 化学突触的时间常数 [ms]
        self.tau_rise = 0.02       # 时间常数 [ms]

        self.A_inhib = 125        # 化学突触的尺度常数 [nS]

    def _vars(self):
        # 0维度--post, 1维度--pre
        self.g_in = np.zeros((self.post.num, self.pre.num))   # 设定突触电导
        self.E_in = np.zeros((self.post.num, self.pre.num))   # 
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t            # 这个是非常重要的
        I_post = self.syn_inh()        # 突触后神经元接收的突触电流 (post_num,)

        self.t += self.dt  # 时间前进

        return I_post

    def syn_inh(self):
        pre_fire_launch = self.pre.flaglaunch                   # 突触后胞体放电
        post_mem = self.post.mem_soma

        Isyn = (self.A_inhib * self.g_in  * self.conn * np.expand_dims((self.e - post_mem), axis=1)).sum(axis=1)  # 0维度--post，1维度--pre

        E_in_new = self.E_in - self.E_in * self.dt / self.tau_inhib
        self.g_in = self.g_in + (-self.g_in + self.E_in) * self.dt / self.tau_rise

        self.E_in += E_in_new + pre_fire_launch[None, :]

        return Isyn
