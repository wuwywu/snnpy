# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : nodes.py
# 将各种用于神经网络的`节点`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from surrogate import *

class BaseNode(nn.Module):
    """
    神经元模型的基类
    Args:
        :params
        threshold: 神经元发放脉冲需要达到的阈值(神经元的参数)
        requires_thres_grad: 阈值求梯度
        v_reset: 静息电位
        decay: 膜电位衰减项
        mem_detach: 是否将上一时刻的膜电位在计算图中截断
    """
    def __init__(self,
                 threshold=.5,
                 requires_thres_grad=False,
                 v_reset=0.,
                 mem_detach=False,
                 decay=0.2,
                 ):
        super().__init__()
        self.mem = None
        self.spike = None
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.v_reset = v_reset
        self.decay = decay     # decay constants
        self.mem_detach = mem_detach

    def forward(self, x):
        if self.mem_detach and hasattr(self.mem, 'detach'):
            self.mem = self.mem.detach()
            self.spike = self.spike.detach()
        self.integral(x)
        self.calc_spike()
        return self.spike
        
    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """
        pass

    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲，并reset
        :return: None
        """
        pass
    
    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = None
        self.spike = None

        
# ============================================================================
# 用于SNN的node

class LIFNode(BaseNode):
    def __init__(self, threshold=.5, decay=0.2, act_fun=SpikeAct):
        super().__init__(threshold=threshold, decay=decay)
        self.act_fun = act_fun(alpha=0.5, requires_grad=False)

    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, device=inputs.device)
            self.spike = torch.zeros_like(inputs, device=inputs.device)
        self.mem = self.decay * self.mem
        self.mem += inputs

    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲，并reset
        :return: None
        """
        self.spike = self.act_fun(self.mem-self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class LIFbackEI(BaseNode):
    """
    BackEINode with self feedback connection and excitatory and inhibitory neurons
    Reference：https://www.sciencedirect.com/science/article/pii/S0893608022002520
    Args:
        :params
        in_channel: 反馈调节和EI输出，在该层中卷积核的数量
        threshold: LIF的重置阈值
        decay: LIF的衰减因子
        if_ei: EI输出开关
        if_back: 反馈调节开关
        cfg_backei: EI和反馈调节的卷积核大小(2*cfg_backei+1), 与周围神经元关联范围
        act_fun: LIF的激活函数
        th_fun: EI的符号函数
    """
    def __init__(self, in_channel=None, threshold=.5, decay=0.2, 
                 if_ei=False, if_back=False, cfg_backei=2,
                act_fun=SpikeAct, th_fun=EIAct):
        super().__init__(threshold=threshold, decay=decay)
        self.act_fun = act_fun(alpha=0.5)   # LIF的激活函数
        self.th_fun = th_fun()              # ei的符号函数
        self.if_ei = if_ei
        self.if_back = if_back
        if self.if_ei:
            self.ei = nn.Conv2d(in_channel, in_channel, kernel_size=2*cfg_backei+1, stride=1, padding=cfg_backei)
        if self.if_back:
            self.back = nn.Conv2d(in_channel, in_channel, kernel_size=2*cfg_backei+1, stride=1, padding=cfg_backei)

    def integral(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, device=inputs.device)
            self.spike = torch.zeros_like(inputs, device=inputs.device)
        self.mem = self.decay * self.mem
        if self.if_back:
            self.mem += F.sigmoid(self.back(self.spike)) * inputs
        else:
            self.mem += inputs
    
    def calc_spike(self):
        if self.if_ei:
            ei_gate = self.th_fun(self.ei(self.mem))
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike)
            self.spike = ei_gate * self.spike
        else:
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike.detach())


class LIFSTDP(BaseNode):
    """
    用于执行STDP运算时使用的节点 decay的方式是膜电位乘以decay并直接加上输入电流
    reference : https://doi.org/10.1016/j.neunet.2023.06.019
    args:
        :params
        threshold: 神经元发放脉冲需要达到的阈值(神经元的参数)
        decay: LIF的衰减因子
        act_fun: LIF的激活函数
    """
    def __init__(self, threshold=.5, decay=0.2, act_fun=SpikeActSTDP):
        super().__init__(threshold=threshold, decay=decay)
        self.act_fun = act_fun(alpha=.5, requires_grad=False)   # 激活函数

    def integral(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, device=inputs.device)
            self.spike = torch.zeros_like(inputs, device=inputs.device)
        self.mem = self.decay * self.mem + inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)  # SpikeActSTDP : approximation firing function, LIF的阈值 threshold
        self.mem = self.mem * (1 - self.spike.detach())


if __name__ == "__main__":
    snn = LIFSTDP()
    # snn = LIFbackEI()
    print(snn(torch.tensor([10, 0.1])))
    print(snn.mem)
    print(snn.spike)
