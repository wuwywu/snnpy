# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/29
# User      : WuY
# File      : synapse_exp.py
# 这个文件中使用了一个指数衰减的化学突触模型，延迟，STDP的另一中写法
# STDP的写法：分别保存前后神经元的脉冲发放所引起的突触权重变化的迹(trace)

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse


class synbase_exp(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_exp"]
    """
    def __init__(self, pre, post, conn=None, synType="chem_exp", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_exp":
            self.syn = self.syn_chem_exp  # 化学突触
        self._params()
        self._vars()

    def _params(self):
        # 指数衰减模型(化学突触)的系数
        self.e = 0          # 化学突触的平衡电位
        self.tau_syn = 5    # 化学突触的时间常数

    def _vars(self):
        # 0维度--post，1维度--pre
        self.g = np.random.random((self.post.num,  self.pre.num))   # 设定突触电导
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的
        I_post = self.syn()  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def _dg_dt(self):
        # 只有一个变量的时候，返回必须加“ , ”
        dg_dt = -self.g / self.tau_syn
        return dg_dt,

    def syn_chem_exp(self, pre_fire_launch=None, post_mem=None):
        """
        args:
            pre_fire_launch: 突触前的放电开启标志
            post_mem: 突触后的膜电位
        """
        if pre_fire_launch is None: pre_fire_launch = self.pre.flaglaunch
        if post_mem is None: post_mem = self.post.mem

        # 迭代计算g
        self.method(self._dg_dt, self.g)
        Isyn = (self.g * self.conn * np.expand_dims((self.e - post_mem), axis=1)).sum(axis=1)  # 0维度--post，1维度--pre

        # 突触前放电增加g (0维度--post，1维度--pre)
        self.g += self.w * pre_fire_launch[None, : ]

        return Isyn


# 包含时间延迟
# 包含时间延迟
class syn_exp_delay(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_dg"]
    delayer: 延迟器
    """
    def __init__(self, pre, post, conn=None, synType="chem_exp", method="euler", delayer=None):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.delayer = delayer
        self.synType = synType
        if self.synType == "electr":
            self.syn = self.syn_electr  # 电突触
        elif self.synType == "chem_exp":
            self.syn = self.syn_chem_exp  # 化学突触
        self._params()
        self._vars()

    def _params(self):
        # STDP2_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.tau_syn = 5  # 化学突触的时间常数

    def _vars(self):
        # 0维度--post，1维度--pre
        self.g = np.random.random((self.post.num, self.pre.num))  # 设定突触电导
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的
        I_post = self.syn()  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def dg_dt(self):
        # 只有一个变量的时候，返回必须加“ , ”
        dg_dt = -self.g / self.tau_syn
        return dg_dt,

    def syn_electr(self, pre_mem=None, post_mem=None):
        """
        args:
            pre_mem: 突触前的膜电位
            post_mem: 突触后的膜电位
        """
        if pre_mem is None: pre_mem = self.pre.mem
        if post_mem is None: post_mem = self.post.mem
        pre_mem = self.delayer(self.pre.mem)  # 存储延迟，并给出延迟的值
        vj_vi = pre_mem - np.expand_dims(post_mem, axis=1)  # pre减post
        Isyn = (self.g * self.conn * vj_vi).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn

    def syn_chem_exp(self, pre_fire_launch=None, post_mem=None):
        """
        args:
            pre_fire_launch: 突触前的放电开启标志
            post_mem: 突触后的膜电位
        """
        if pre_fire_launch is None: pre_fire_launch = self.pre.flaglaunch
        if post_mem is None: post_mem = self.post.mem

        # 迭代计算g
        self.method(self.dg_dt, self.g)
        g = self.delayer(self.g.flatten()).reshape(self.g.shape)    # 存储延迟，并给出延迟的值
        Isyn = (g * self.conn * np.expand_dims((self.e - post_mem), axis=1)).sum(axis=1)  # 0维度--post，1维度--pre

        # 突触前放电增加g (0维度--post，1维度--pre)
        self.g += self.w * pre_fire_launch[None, :]

        return Isyn


