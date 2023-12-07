# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/4
# User      : WuY
# File      : learningrule.py
# 用于复杂网络的算法集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
使用方式(DLS)：
    1、创建类：
        N -- 输入变量的数量
        local -- 需要学习同步节点的位置索引
        alpha -- 算法的学习率，
    2、forward类(直接使用)：
        w -- 输入需要学习的权重，shape(output, input)
        input -- 输入值，与权重相乘的值，shape(output, input),一个输出对应一个输入序列
        error -- 输出值与对比值的差值(w*input-(traget-input0))
"""
from base.learningrule.rls import RLS_complex as DLS   # # dynamic learning of synchronization (DLS) algorithm
