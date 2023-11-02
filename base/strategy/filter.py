# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/2
# User      : WuY
# File      : filter.py
# description : 对膜电位或者电流进行滤波，减少信息的损失
# 将各种用于神经网络的`滤波器`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ASFilter(nn.Module):
    """
    适应性突触滤波器(Adaptive synaptic filter, ASF)
    reference: https://doi.org/10.1016/j.neunet.2023.06.019
    equ:
        \left. \begin{array}  {l}  {  \delta_{asf}  ( i ^ { ( t ) } ) = \frac { u_{thresh}^{( t )} } { 1 + e ^ { (\sigma_t  ) } } }
        \\ {  \quad \sigma_t = - \alpha_{asf} \frac {  i ^ { ( t ) }  } { u_{thresh}^{( t )} } + \beta_{asf}  } \end{array} \right.
    description:
        ASF 通过非线性函数调整电流，使电流更有可能集中在阈值或静息电位附近。
        将电流集中到阈值将导致对神经元的更多竞争。更多的竞争将有助于避免神经元的主导地位。
        而电流接近静息电位将减少低电流强度神经元发射时产生的噪音。
    """
    def __init__(self, alpha_asf=8., beta_asf=3.2):
        """
        args:
            alpha_asf: 系数, 控制滤波器的功能
            beta_asf: 系数, 控制滤波器的功能
        """
        super().__init__()
        self.alpha_asf = alpha_asf
        self.beta_asf = beta_asf

    def forward(self, current, thre):
        """
        args:
            current: 卷积或全连接后的电流(B,C,H,W)/(B,C)
            thre: 阈值
        return：
            current_ASF: 滤波后的电流
        """
        current = current.clamp(min=0)  # 文章中并没写明裁剪电流(需要调试,影响收敛速度)
        current_ASF = thre / (1 + torch.exp(-(current - self.beta_asf * thre / self.alpha_asf) * (self.alpha_asf / thre)))
        return current_ASF


if __name__=="__main__":
    ASF = ASFilter()
    current = torch.tensor(0.1)
    print(ASF(current, torch.tensor(0.5)))