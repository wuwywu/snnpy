# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/20
# User      : WuY
# File      : stdp_recogDigit2015.py
# paper     : Yiting Dong, Unsupervised learning of digit recognition using spike-timing-dependent plasticity
# doi       : 10.3389/fncom.2015.00099
# 描述       : 这篇文章比较经典，很多的好文章都引用了这篇
"""
网络结构：
    input(泊松编码) --> e --> i( --> e) (e --> voting)
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(r"../")
sys.path.append(r"../../")
# print(sys.path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from base.nodes import LIFei
from base.utils.utils import setup_seed
from base.encoder.encoder import Poisson
from datasets.datasets import mnist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 固定随机种子
setup_seed(0)

datasetPath = r"E:\snnpy\datasets\MNIST"
# 设置参数
dt = .5                 # 积分步长
runTime = 0.35*1000     # 350 ms
restTime = 0.15*1000    # 150 ms
node_e = {
    "thresh": -52.,     # 阈值
    "v_reset": -65.,    # 膜电位重设值
    "Erest": -65.,      # 静息电位
    "refrac": 5.,       # 不应期时间
    "tau": 100.,        # 膜电位时间常数, 用于控制膜电位衰减
}   # 兴奋性神经元参数
node_i = {
    "thresh": -40.,     # 阈值
    "v_reset": -45.,    # 膜电位重设值
    "Erest": -60.,      # 静息电位
    "refrac": 2.,       # 不应期时间
    "tau": 10.,         # 膜电位时间常数, 用于控制膜电位衰减
}   # 抑制性神经元参数
weight = {
    'ee_input': 0.1,
    "ei": 10.4,
    'ie': 17.0
}   # 三个突触权重的大小


nInput = 784    # 输入节点数
nE = 400        # 兴奋性神经元数
nI = nE         # 抑制性神经元数

class create_weight:
    """
    创建权重, ei权重，ie权重都是固定的
    ine权重（input-->e）: 包含了stdp
    """
    @staticmethod
    def create_ei():
        """e-->i 的权重一对一，并且是固定的"""
        assert nI == nE, "兴奋性神经元与抑制性神经元数量不一致"
        return torch.eye(nE)*weight["ei"]

    @staticmethod
    def create_ie():
        """
        i-->e 的权重，除了与对应位置的e神经元不连接外，与其他神经元都连接
        Each of the inhibitory neurons is connected
        to all excitatory ones, except for the one from which it receives
        a connection.
        """
        assert nI == nE, "兴奋性神经元与抑制性神经元数量不一致"
        return (torch.ones((nE, nI))-torch.eye(nE))*weight['ie']

    @staticmethod
    def create_ine():
        """
        从输入神经元到兴奋神经元的所有突触都是通过 STDP 学习的。
        """
        return (torch.rand((nE, nInput)) + 0.01)*weight['ee_input']


class Synapses_ei(nn.Module):
    def __init__(self, pre, conn, post):
        super().__init__()
        self.pre = pre
        self.conn = conn
        self.post = post

    def forward(self, I):
        """
        args:
            I: 输入进来的电流
        return:
            突触前输入到突触后神经元的突触电流
        """
        pass


class Synapses_ie(nn.Module):
    def __init__(self, pre, conn, post):
        super().__init__()
        self.pre = pre
        self.conn = conn
        self.post = post


class Synapses_stdp_ine(nn.Module):
    def __init__(self, conn, post):
        super().__init__()
        self.conn = conn
        self.post = post


class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1
        self.lif_e = LIFei(threshold=node_e["thresh"], v_reset=node_e["v_reset"], dt=dt,
                Erest=node_e["Erest"], tau=node_e["tau"], refrac=node_e["refrac"])
        self.lif_i = LIFei(threshold=node_i["thresh"], v_reset=node_i["v_reset"], dt=dt,
                Erest=node_i["Erest"], tau=node_i["tau"], refrac=node_i["refrac"])
        self.fc_ine = nn.Linear(nInput, nE, bias=False)
        self.fc_ei = nn.Linear(nE, nI, bias=False)
        self.fc_ie = nn.Linear(nI, nE, bias=False)
        self.init_weight()  #

    def forward(self, x):

        pass

    def init_weight(self):
        self.fc_ine.weight.data = create_weight.create_ine()
        self.fc_ei.weight.data = create_weight.create_ei()
        self.fc_ie.weight.data = create_weight.create_ie()



if __name__ == "__main__":
    model = MNISTnet().to(device)
    print(model.fc_ei.weight.data)

    # transform = transforms.Compose([transforms.ToTensor()])
    # train_iter = mnist(train=True, batch_size=1, download=True,
    #                    data_path=datasetPath, transforms_IN=transform)  # transforms_IN=transform
    # test_iter = mnist(train=False, batch_size=1, download=True,
    #                   data_path=datasetPath, transforms_IN=transform)
    #
    # for i, (images, labels) in enumerate(train_iter):
    #     images = images*255/4   #0-63.75Hz
    #     for t in range(int(runTime/dt)):
    #         x = Poisson(images, dt=dt)
    #         print(x.sum())







