# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/3/09
# User      : WuY
# File      : couple.py
# 给出一个自定义耦合的模板

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import matplotlib.pyplot as plt


class couple:
    """
    pre: 网络前节点
    post: 网络后节点
    conn: 连接矩阵
    """
    def __init__(self, pre, post, conn=None):
        self.pre = pre      # 网络前节点
        self.post = post    # 网络后节点
        self.conn = conn    # 连接矩阵
        self.dt = post.dt  # 计算步长
        self._fparams()
        self._fvars()

    def _fparams(self):
        # 0维度--post，1维度--pre
        self.w = .1*np.ones((self.post.num, self.pre.num)) # 设定连接权重

    def _fvars(self):
        self.t = self.post.t

    def __call__(self):
        """
        开头和结尾更新时间（重要）
        self.t = self.post.t
        self.t += self.dt
        """
        # 保证couple不管何时创建都能与后节点有相同的时间
        self.t = self.post.t    # 这个是非常重要的

        I_post = self.model_couple()     # 网络后节点接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

    def model_couple(self):
        """
        耦合方式
        return:
            后神经元接受的值
        """
        pass
