# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/5/7
# User      : WuY
# File      : synapse_NMDA.py
# 这个文件收集 NMDA(一种兴奋性突触结构) 的各种写法

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse


# ============================ 1 ============================
# references: Bono, J., Clopath, C., 2017. Modeling somatic and dendritic spike mediated plasticity
# at the single neuron and network level. Nat Commun 8, 706. https://doi.org/10.1038/s41467-017-00740-z
# 这个模型使用的是指数衰减模型作为突触电导的更新方式
class synbase_NMDA1(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_exp"]
    method: 计算非线性微分方程的方法，（"eluer", "rk4"）
    """
    def __init__(self, pre, post, conn=None, synType="chem_exp", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_exp":
            self.syn = self.syn_chem_NMDA1  # 化学突触
        self._params()
        self._vars()

    def _params(self):
        # 指数衰减模型(化学突触)的系数
        self.e = 0            # 化学突触的平衡电位 [mV]
        self.tau_NMDA = 50    # 化学突触的时间常数 [ms]
        self.g_max = 50       # 化学突触的最大电导 [nS]

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
        self.t = self.post.t    # 这个是非常重要的
        I_post = self.syn()     # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def _dg_dt(self):
        # 只有一个变量的时候，返回必须加“ , ”
        dg_dt = -self.g / self.tau_NMDA
        return dg_dt,

    def syn_chem_NMDA1(self, pre_fire_launch=None, post_mem=None):
        """
        args:
            pre_fire_launch: 突触前的放电开启标志
            post_mem: 突触后的膜电位
        """
        if pre_fire_launch is None: pre_fire_launch = self.pre.flaglaunch
        if post_mem is None: post_mem = self.post.mem

        # 计算镁离子的阻塞
        B_Mg = 1 / (1 + np.exp(-0.065*post_mem)/3.57)

        # 迭代计算g
        self.method(self._dg_dt, self.g)
        Isyn = (self.g * B_Mg * self.conn * np.expand_dims((self.e - post_mem), axis=1)).sum(axis=1)  # 0维度--post，1维度--pre

        # 突触前放电增加g (0维度--post，1维度--pre)
        self.g += self.w * self.g_max * pre_fire_launch[None, :]

        return Isyn





