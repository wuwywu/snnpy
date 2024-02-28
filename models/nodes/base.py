# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/24
# User      : WuY
# File      : base.py
# 将各种用于网络`节点`的基础功能集合到这里

import os
import sys
import copy
import numpy as np

class Neurons:
    """
    N: 创建神经元的数量
    method ： 计算非线性微分方程的方法，（"eluer", "rk4"）
    dt ： 计算步长
    神经元的膜电位都写为：mem
    运行时间：t; 时间步长：dt
    神经元数量：num
    """
    def __init__(self, N, method="eluer", dt=0.01):
        self.num = N  # 神经元数量
        if method == "eluer":   self.method = self._eluer
        if method == "rk4":   self.method = self._rk4
        self.dt = dt
        self._fparams()
        self._fvars()

    def _fparams(self):
        self.th_up = 0  # 放电阈值
        self.th_down = -10  # 放电阈下值

    def _fvars(self):
        self.t = 0  # 运行时间
        # 模型放电变量
        self.flag = np.zeros(self.num, dtype=int)           # 模型放电标志
        self.flaglaunch = np.zeros(self.num, dtype=int)     # 模型开始放电标志
        self.firingTime = np.zeros(self.num)                # 记录放电时间

    def _eluer(self, models, I, *args):
        """
        使用 euler 算法计算非线性微分方程
        arg:
            models: 神经元模型函数，输入一个外部激励(所有激励合在一起)，返回所有dvars_dt
            I: 外部激励，所有激励合在一起
            *args： 输入所有变量，与dvars_dt一一对应
        """
        vars = list(args)      # 所有的变量
        dvars_dt = models(I)   # 所有变量的的微分方程
        lens = len(dvars_dt)   # 变量的数量
        for i in range(lens):  # 变量更新
            vars[i] += dvars_dt[i] * self.dt

    def _rk4(self, models, I, *args):
        """
        使用 fourth-order Runge-Kutta(rk4) 算法计算非线性微分方程
        arg:
            models: 神经元模型函数，输入一个外部激励(所有激励合在一起)，返回所有dvars_dt
            I: 外部激励，所有激励合在一起
            *args： 输入所有变量，与dvars_dt一一对应
        """
        vars = list(args)     # 所有的变量
        original_vars = copy.deepcopy(vars) # 原始状态备份
        lens = len(vars)      # 变量的数量
        dt = self.dt          # 时间步长
        # 计算k1
        k1 = models(I)
        # 计算k2
        for i in range(lens):
            vars[i] += original_vars[i] + 0.5*dt*k1[i] - vars[i]
        k2 = models(I)
        # 计算k3
        for i in range(lens):
            vars[i] += original_vars[i] + 0.5*dt*k2[i] - vars[i]
        k3 = models(I)
        # 计算k4
        for i in range(lens):
            vars[i] += original_vars[i] + dt*k3[i] - vars[i]
        k4 = models(I)

        # 最终更新vars
        for i in range(lens):
            vars[i] += original_vars[i] + dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 - vars[i]

    def __call__(self):
        self.t += self.dt  # 时间前进

    def _spikes_eval(self, mem):
        """
        在非人工神经元中，计算神经元的spiking
        """
        # -------------------- 放电开始 --------------------
        self.flaglaunch[:] = 0  # 重置放电开启标志
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))  # 放电开始的位置
        self.flag[firing_StartPlace] = 1  # 放电标志改为放电
        self.flaglaunch[firing_StartPlace] = 1  # 放电开启标志
        self.firingTime[firing_StartPlace] = self.t  # 记录放电时间
        #  -------------------- 放电结束 -------------------
        firing_endPlace = np.where((mem < self.th_down) & (self.flag == 1))  # 放电结束的位置
        self.flag[firing_endPlace] = 0  # 放电标志改为放电

