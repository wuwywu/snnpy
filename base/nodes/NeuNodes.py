# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : NeuNodes.py
# 将各种用于神经网络的`节点`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
        v_reset: 重设电位
        dt: 积分步长
        decay: 膜电位衰减项
        mem_detach: 是否将上一时刻的膜电位在计算图中截断
    """
    def __init__(self,
                 threshold=.5,
                 requires_thres_grad=False,
                 v_reset=0.,
                 dt=1.,
                 mem_detach=False,
                 decay=0.2,
                 ):
        super().__init__()
        self.mem = None
        self.spike = None
        self.threshold = nn.Parameter(torch.tensor(threshold),
                                      requires_grad=requires_thres_grad)
        self.dt = dt
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
        重置维度
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = None
        self.spike = None

    def i_reset(self):
        """
        输入的维度一致
        在需要频繁重置时,开辟内存的消耗太大
        :return: None
        """
        if self.mem is not None:
            self.mem.fill_(self.v_reset)
            self.spike.fill_(0)

        
# ============================================================================
# 用于SNN的node

class IFNode(BaseNode):
    """
        Ding 添加的第一个神经元模型
        qianmingphy@ccnu
    """
    def __init__(self, threshold=.5, act_fun=SpikeAct):
        super().__init__(threshold=threshold)
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
        self.mem += inputs # 在IF模型中，去掉衰减

    def calc_spike(self):
        """
            通过当前的mem计算是否发放脉冲，并reset
            :return: None
        """
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


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
        mem_detach: 是否将上一时刻的膜电位在计算图中截断
    """
    def __init__(self, threshold=.5, decay=0.2, mem_detach=True, act_fun=SpikeActSTDP):
        super().__init__(threshold=threshold, decay=decay, mem_detach=mem_detach)
        self.act_fun = act_fun(alpha=.5, requires_grad=False)   # 激活函数

    def integral(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, device=inputs.device)
            self.spike = torch.zeros_like(inputs, device=inputs.device)
        self.mem = self.decay * self.mem + inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)  # SpikeActSTDP : approximation firing function, LIF的阈值 threshold
        self.mem = self.mem * (1 - self.spike.detach())


class LIFei(BaseNode):
    """
    这个神经元可是兴奋神经元或抑制性神经元，接收到的突触，也可以是兴奋性输入或抑制性输入。
    reference: doi: 10.3389/fncom.2015.00099

    math:
    $$\tau \frac { d V } { d t } = ( E _ { rest } - V )
    + g _ { e } ( E _ { e c c } - V )
    + g _ { i } ( E _ { i n h } - V )$$

    args:
        threshold: 神经元发放脉冲需要达到的阈值(神经元的参数)
        v_reset: 重设电位
        dt: 积分步长
        mem_detach: 是否将上一时刻的膜电位在计算图中截断
        Erest: 静息电位
        tau: 膜电位时间常数, 用于控制膜电位衰减
        refrac: 持续时间
        mode_refrac: 选择不应期模式["hard", "soft"]
        act_fun: 激活函数
    """
    def __init__(self, threshold=-52., v_reset=-65., dt=.5, mem_detach=True,
                 Erest=-65, tau=100., refrac=5., mode_refrac="hard", act_fun=SpikeAct):
        super().__init__(threshold=threshold, v_reset=v_reset, dt=dt, mem_detach=mem_detach)
        self.Erest = Erest
        self.tau = tau
        self.refrac = refrac
        self.mode_refrac = mode_refrac
        self.act_fun = act_fun(alpha=0.5, requires_grad=False)
        self.timer = None  # 记录放电后的间隔时间

    def integral(self, inputs):
        """
        args:
            inputs: 在外部将突触电流集合一起后输入
        """
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, device=inputs.device)+self.v_reset
            self.spike = torch.zeros_like(inputs, device=inputs.device)
            # 初始时没有不应期
            if self.refrac > self.dt:
                self.timer = torch.zeros_like(inputs, device=inputs.device)+self.refrac+1.
        self.mem += (self.Erest-self.mem+inputs)/self.tau*self.dt

    def calc_spike(self):
        if self.refrac > self.dt:
            self.spike = self.act_fun(self.mem - self.threshold)
            if self.mode_refrac == "soft":
                self.spike[self.timer<=self.refrac] = 0

            self.mem = self.mem * (1 - self.spike.detach()) + self.v_reset * self.spike.detach()
            if self.mode_refrac == "hard":
                self.mem[self.timer<=self.refrac] = self.v_reset
            if self.mode_refrac not in ["hard", "soft"]:
                print("输入错误，不应期模式(mode_refrac): soft, hard, 运行结果不包含不应期")

            # 更新记录时间
            self.timer[self.spike>0] = 0
            self.timer += self.dt
        else:
            # SpikeActSTDP : approximation firing function, LIF的阈值 threshold
            self.spike = self.act_fun(self.mem - self.threshold)
            self.mem = self.mem*(1 - self.spike.detach())+v_reset*self.spike.detach()

    def n_reset(self):
        """
        重置维度
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = None
        self.spike = None
        self.timer = None  # 记录放电后的间隔时间

    def i_reset(self):
        """
        输入的维度一致
        在需要频繁重置时,开辟内存的消耗太大
        :return: None
        """
        if self.mem is not None:
            self.mem.fill_(self.v_reset)
            self.spike.fill_(0)
            if self.refrac > self.dt:
                self.timer.fill_(0)  # 记录放电后的间隔时间


class HHnode(BaseNode):
    """
    Hodgkin–Huxley (HH)模型
    reference: https://doi.org/10.1113/jphysiol.1952.sp004764
    I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
    args:
        threshold: 放电阈值
        dt: 积分步长
        T: 环境温度
    """
    def __init__(self, threshold=20., dt=.01, T=6.3):
        super().__init__(threshold=threshold, dt=dt)
        self.phi = 3**((T-6.3)/10)  # 温度因子
        self.g_Na = 120
        self.g_K = 36
        self.g_l = .3
        self.V_Na = 50.
        self.V_K = -77.
        self.V_l = -54.4
        self.C = 1.
        self.mem_p = None
        self.m = None
        self.n = None
        self.h = None

    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """
        if self.mem is None:
            self.mem = -70.68*torch.ones_like(inputs, device=inputs.device)
            self.mem_p = self.mem.clone()
            self.spike = torch.zeros_like(inputs, device=inputs.device)
            self.m = 0.05*torch.ones_like(inputs, device=inputs.device)
            self.n = 0.31*torch.ones_like(inputs, device=inputs.device)
            self.h = 0.59*torch.ones_like(inputs, device=inputs.device)

        alpha_n = (0.01 * self.mem + .55) / (1 - torch.exp(-5.5 - 0.1 * self.mem))
        beta_n = 0.125 * torch.exp((-self.mem - 65) / 80.0)
        alpha_m = (4 + 0.1 * self.mem) / (1 - torch.exp(-4. - 0.1 * self.mem))
        beta_m = 4.0 * torch.exp((-self.mem - 65) / 18.0)
        alpha_h = 0.07 * torch.exp((-self.mem-65) / 20.0)
        beta_h = 1 / (1 + torch.exp(-3.5 - 0.1 * self.mem))

        self.n += self.dt*(alpha_n * (1 - self.n) - beta_n * self.n)*self.phi
        self.m += self.dt*(alpha_m * (1 - self.m) - beta_m * self.m)*self.phi
        self.h += self.dt*(alpha_h * (1 - self.h) - beta_h * self.h)*self.phi

        I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        I_L = self.g_l * (self.mem - self.V_l)

        self.mem_p = self.mem.clone()
        self.mem +=  self.dt * (inputs - I_Na - I_K - I_L) / self.C

    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲
        :return: None
        """
        self.spike = (self.threshold > self.mem_p).float() * (self.mem > self.threshold).float()

    def n_reset(self):
        """
        重置维度
        神经元重置，重置神经元所有的状态
        :return: None
        """
        self.mem = None
        self.mem_p = None
        self.spike = None
        self.m = None
        self.n = None
        self.h = None


if __name__ == "__main__":
    snn = LIFSTDP()
    # snn = LIFbackEI()
    print(snn(torch.tensor([10, 0.1])))
    print(snn.mem)
    print(snn.spike)
