# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : base_module.py
# 将各种用于神经网络的`各个基础模块`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class BaseModule(nn.Module):
    """
    SNN抽象类, 所有的SNN都要继承这个类, 以实现一些基础方法
    :param time_window: 仿真时间窗口
    """
    def __init__(self, time_window=5):
        super().__init__()
        self.time_window = time_window
    
    def forward(self):
        pass

    def reset(self):
        """
        重置所有神经元的膜电位(神经元的类中写入n_reset方法)
        :return:
        """
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()


