# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/25
# User      : WuY
# File      : synapse.py
# 基础的突触模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt

class synbase:
    """
    pre: 突触前神经元
    post: 突触后神经元
    conn: 连接矩阵
    synType: 突触类型["electr", "chem_Alpha"]
    """
    def __init__(self, pre, post, conn=None, synType="electr"):
        self.pre = pre      # 突触前神经元
        self.post = post    # 突触后神经元
        self.iself = (self.pre is self.post)  # 判断是否只有一个子网络（突触前和和突触后神经元是一个）
        self.conn = conn    # 连接矩阵
        # 选着突触类型
        self.synType = synType
        if self.synType == "electr":
            self.syn = self.syn_electr  # 电突触
        elif self.synType == "chem_Alpha":
            self.syn = self.syn_chem_Alpha  # Alpha_化学突触
        self._fparams()
        self._fvars()

    def _fparams(self):
        # 0维度--post，1维度--pre
        self.w = .1*np.ones((self.post.num, self.pre.num)) # 设定连接权重
        # Alpha_化学突触的系数
        self.e = 0  # 化学突触的平衡电位
        self.tau_syn = 2  # 化学突触的时间常数

    def _fvars(self):
        self.t = self.post.t

    def __call__(self):
        I = self.syn()
        if not self.iself:  # 如果是两个网络则更新pre和post，否者只更新post
            self.pre()  # 突触前神经元
        self.post(I)  # 突触后神经元
        self.t = self.post.t

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
