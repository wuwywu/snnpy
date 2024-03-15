# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/07
# User      : WuY
# File      : synSTDP.py
# 包含STDP的突触模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt

class synSTDP:
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_Alpha"]
    """
    def __init__(self, pre, post, conn=None, synType="chem_Alpha"):
        self.pre = pre      # 突触前神经元
        self.post = post    # 突触后神经元
        self.conn = conn    # 连接矩阵
        self.dt = post.dt   # 计算步长
        # 选着突触类型
        self.synType = synType
        if self.synType == "electr":
            self.syn = self.syn_electr      # 电突触
        elif self.synType == "chem_Alpha":
            self.syn = self.syn_chem_Alpha  # Alpha_化学突触
        self._fparams()
        self._fvars()

    def _fparams(self):
        # Alpha_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.tau_syn = 2  # 化学突触的时间常数
        # STDP系数
        # A_D*tau_D>A_P*tau_P: 抑制占主导；A_D*tau_D<A_P*tau_P: 增强占主导
        self.w_min = 0.001      # 设定权重最小值
        self.w_max = 1.0        # 设定权重最大值
        self.A_P = 1        # 可塑性导致突触强度变化（增强）
        self.tau_P = 20.    # 确定一个尖峰跟随另一个尖峰的时间（增强）
        self.A_D = 1.2      # 可塑性导致突触强度变化（抑制）
        self.tau_D = 20.    # 确定一个尖峰跟随另一个尖峰的时间（抑制）
        self.lr = 1e-4      # 学习率

    def _fvars(self):
        # 0维度--post，1维度--pre
        self.w = .5 * np.ones((self.post.num, self.pre.num))  # 设定连接权重
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt

        return:
            I_post: 突触后神经元接受到的突触电流
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

        # 应用学习率
        self.w += self.lr * dw

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
        vj_vi = pre_mem-np.expand_dims(post_mem, axis=1)   # pre减post
        Isyn = (self.w*self.conn*vj_vi).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn

    def syn_chem_Alpha(self, pre_fire_t=None, post_mem=None):
        """
        args:
            pre_fire_t: 突触前的放电时间
            post_mem： 突触后的膜电位
        """
        if pre_fire_t is None: pre_fire_t = self.pre.firingTime
        if post_mem is None: post_mem = self.post.mem
        alpha = (self.t - pre_fire_t) / self.tau_syn * np.exp((pre_fire_t-self.t)/self.tau_syn)
        g_syn = alpha * np.expand_dims((self.e - post_mem), axis=1)
        Isyn = (self.w * self.conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn
