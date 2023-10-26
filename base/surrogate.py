# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : surrogate.py
# 将各种用于神经网络方向传播的`替代函数`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SurrogateFunctionBase(nn.Module):
    """
    Surrogate Function 的基类
    :param alpha: 为一些能够调控函数形状的代理函数提供参数.
    :param requires_grad: 参数 ``alpha`` 是否需要计算梯度, 默认为 ``False``
    """
    def __init__(self, alpha, requires_grad=False):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)
        
    @staticmethod
    def act_fun(x, alpha):
        """
        :param x: 膜电位的输入
        :param alpha: 控制代理梯度形状的变量, 可以为 ``NoneType``
        :return: 激发之后的spike, 取值为 ``[0, 1]``
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: 膜电位输入
        :return: 激发之后的spike
        """
        return self.act_fun(x, self.alpha)
    

class spike_act(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return input.gt(0).float()     # LIF的阈值

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input) < alpha
        hu = hu.float() / (2 * alpha)
        return grad_input * hu, None
    
class SpikeAct(SurrogateFunctionBase):
    def __init__(self, alpha=0.5, requires_grad=False):
        super().__init__(alpha, requires_grad)
    
    @staticmethod
    def act_fun(x, alpha):
        return spike_act.apply(x, alpha)


class ei_act(torch.autograd.Function):
    """ 
        定义EI网络中的符号函数-1/1，
        并根据论文公式对符号函数进行梯度的近似。
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()

class EIAct(SurrogateFunctionBase):
    def __init__(self, alpha=0.5, requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return ei_act.apply(x)
    

class spike_act_stdp(torch.autograd.Function):
    # 定义应用STDP的LIF中的激活函数
    @staticmethod
    def forward(ctx, inputs):
        outputs = input.gt(0).float()
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        return inputs * grad_output

class SpikeActSTDP(SurrogateFunctionBase):
    def __init__(self, alpha=0.5, requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return spike_act_stdp.apply(x)

