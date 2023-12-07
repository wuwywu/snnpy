# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/7
# User      : WuY
# File      : connection.py
# 用于复杂网络的连接矩阵集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from base.connection.createConn import alltoall                 # 全连接
from base.connection.createConn import small_world              # 双向小世界
from base.connection.createConn import small_world_single       # 单向小世界
from base.connection.createConn import scale_network            # 无标度网络
from base.connection.createConn import ERnet                    # ER随机网络
