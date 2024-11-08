# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/10/28
# User      : WuY
# File      : synapse_sigmoidal.py
# ref: D. Somers and N. Kopell, Rapid synchronization through fast threshold modulation. Biol. Cybernet. 68, 393 (1993).

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from base_Mod import Synapse

class syn_sigmoidal(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_sigmoidal"]
    method: 计算非线性微分方程的方法，（"eluer", "rk4"）
    """
    def __init__(self, pre, post, conn=None, synType="chem_sigmoidal", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_sigmoidal":
            self.syn = self.syn_chem_sigmoidal  # 化学突触
        self._params()
        self._vars()

    def _params(self):
        self.e = 0          # 化学突触的平衡电位
        self.theta = 0      # 放电阈值
        self.epsi = 7       # 放电下滑斜率

    def _vars(self):
        # 0维度--post，1维度--pre
        pass

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

    def syn_chem_sigmoidal(self, pre_mem=None, post_mem=None):
        """
        args:
            pre_mem: 突触前的膜电位
            post_mem: 突触后的膜电位
        """
        if pre_mem is None: pre_mem = self.pre.mem
        if post_mem is None: post_mem = self.post.mem

        # the sigmoidal function (a limiting version is the Heaviside function)
        s = 1 / (1 + np.exp(-self.epsi*(pre_mem - self.theta))) 

        Isyn = (self.w * self.conn * (self.e - post_mem)[:, None] * s[None, :]).sum(1) # 0维度--post，1维度--pre

        return Isyn


class syn_sigmoidal_delay(Synapse):
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_sigmoidal"]
    method: 计算非线性微分方程的方法，（"eluer", "rk4"）
    """
    def __init__(self, pre, post, conn=None, synType="chem_sigmoidal", method="euler", delayer=None):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_sigmoidal":
            self.syn = self.syn_chem_sigmoidal  # 化学突触
        self.delayer = delayer
        self._params()
        self._vars()

    def _params(self):
        self.e = 0          # 化学突触的平衡电位
        self.theta = 0      # 放电阈值
        self.epsi = 7       # 放电下滑斜率

    def _vars(self):
        # 0维度--post，1维度--pre
        pass

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
        Isyn = (self.w * self.conn * vj_vi).sum(axis=1)  # 0维度--post，1维度--pre
        return Isyn

    def syn_chem_sigmoidal(self, pre_mem=None, post_mem=None):
        """
        args:
            pre_mem: 突触前的膜电位
            post_mem: 突触后的膜电位
        """
        if pre_mem is None: pre_mem = self.pre.mem
        if post_mem is None: post_mem = self.post.mem

        # the sigmoidal function (a limiting version is the Heaviside function)

        pre_mem = self.delayer(self.pre.mem)  # 存储延迟，并给出延迟的值
        s = 1 / (1 + np.exp(-self.epsi*(pre_mem - self.theta))) 

        Isyn = (self.w * self.conn * (self.e - post_mem)[:, None] * s[None, :]).sum(1) # 0维度--post，1维度--pre

        return Isyn

