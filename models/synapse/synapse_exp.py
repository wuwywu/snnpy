# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/29
# User      : WuY
# File      : synapse_exp.py
# 这个文件中使用了一个指数衰减的化学突触模型，延迟，STDP的另一种写法
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
    method: 计算非线性微分方程的方法，（"eluer", "rk4"）
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


# 包含STDP的另一种写法
class syn_exp_stdp(Synapse):
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
        self.e = 0  # 化学突触的平衡电位
        self.tau_syn = 5  # 化学突触的时间常数
        # STDP系数
        # A_D*tau_D>A_P*tau_P: 抑制占主导；A_D*tau_D<A_P*tau_P: 增强占主导
        self.A_P = 0.1  # 突触权重最大变化量（增强）
        self.A_D = 0.095  # 突触权重最大变化量（抑制）
        self.tau_P = 20  # 突触权重变化量随时间差变化的时间常数（增强）
        self.tau_D = 20  # 突触权重变化量随时间差变化的时间常数（抑制）
        self.wmin = 0.  # 设定权重最小值
        self.wmax = 1.  # 设定权重最大值

    def _vars(self):
        # 0维度--post，1维度--pre
        self.Apre = np.zeros(self.pre.num)  # 突触前神经元突触权重变化的迹
        self.Apost = np.zeros(self.post.num)  # 突触后神经元突触权重变化的迹
        self.w = .5 * np.ones((self.post.num, self.pre.num))  # 设定连接权重
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

        # STDP规则改变权重self.w
        self._STDP()

        I_post = self.syn()  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def _stdp_dA_dt(self):
        dApre_dt = -self.Apre / self.tau_P
        dApost_dt = -self.Apost / self.tau_D
        return dApre_dt, dApost_dt

    def _STDP(self):
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
        self.w = np.clip(self.w, self.wmin, self.wmax)  # 限定范围

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
        self.g += self.w * pre_fire_launch[None, :]

        return Isyn


# 包含小世界网络的homeostatic structural plasticity (HSP)的一种写法
class syn_exp_HSP(Synapse):
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
        self.e = 0  # 化学突触的平衡电位
        self.tau_syn = 5  # 化学突触的时间常数
        # HSP的参数
        self.F = .25    # 控制HSP的改变频率
        self.p = 1e-6   # 创建小世界网络是的短边概率

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

        # 用HSP规则改变网络结构self.conn
        self.HSP_smallWorld()

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
        self.g += self.w * pre_fire_launch[None, :]

        return Isyn

    def HSP_smallWorld(self):
        """
        时变小世界网络(HSP规则)
        改动频率，泊松过程中的速率
        这个规则是改变网络结构（连接矩阵self.conn）的规则
        """
        # 动态神经元网络，按照一定的规则在时间循环中改动连接矩阵 --> cojMat
        N = self.post.num
        dt = self.dt
        self.K = self.conn.sum() / N
        # 远处的断边概率
        p_y = (1 - self.p) * self.F * dt
        # 近处的断边概率
        p_j = self.p * self.F * dt

        index = np.arange(N).reshape(1, -1)  # 创建索引并变成数组(一行一列)而不是数列
        dist = np.abs((index - index.T))
        dist = np.where(dist > N / 2, N - dist, dist)  # 神经元的距离矩阵
        dist_y = self.conn * np.random.rand(N, N) * np.where(dist > self.K, 1, 0)  # 远处位置概率矩阵
        dist_j = self.conn * np.random.rand(N, N) * np.where((dist <= self.K) & (dist > 0), 1, 0)  # 近处位置概率矩阵

        # index = np.arange(N)    # 变为一维取值
        for i in range(N):
            dist_y1 = dist_y[i]
            dist_j1 = dist_j[i]
            place_y = np.where((dist_y1 <= p_y) & (dist_y1 > 0))[0]  # 远处断边的位置
            place_j = np.where((dist_j1 <= p_j) & (dist_j1 > 0))[0]  # 近处断边的位置

            if place_y.size != 0:
                near = index[0][np.where((dist[i] <= self.K) & (dist[i] > 0) & (self.conn[i] == 0))]
                if near.size != 0 and near.size >= place_y.size:
                    replace_y = np.random.choice(near, size=len(place_y), replace=False)  # 远处重连的位置(近端)
                    # print(i, place_y, replace_y)
                    self.conn[i, place_y] = 0  # 断边
                    self.conn[i, replace_y] = 1  # 重连

            if place_j.size != 0:
                far = index[0][np.where((dist[i] > self.K) & (self.conn[i] == 0))]
                if far.size != 0 and far.size >= place_j.size:
                    replace_j = np.random.choice(far, size=len(place_j), replace=False)  # 近处重连的位置(远端)
                    # print(i, place_j, replace_j)
                    self.conn[i, place_j] = 0  # 断边
                    self.conn[i, replace_j] = 1  # 重连

