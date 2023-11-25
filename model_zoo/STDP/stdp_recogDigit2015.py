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
from tqdm import tqdm

from base.nodes import LIFei
from base.connection.synapses import synchem
from base.connection.layers import VotingLayer
from base.utils.utils import setup_seed
from base.encoder.encoder import Poisson
from datasets.datasets import mnist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 固定随机种子
setup_seed(0)

datasetPath = r"E:\snnpy\datasets\MNIST"
# 设置参数
dt = .5                 # 积分步长
runTime = 0.35*1000     # 350 ms
restTime = 0.15*1000    # 150 ms
node_e = {
    "thresh": -52.,     # 阈值
    "offset": 20.,      # 阈值偏移量
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
    'ee_input': 0.3,
    "ei": 10.4,
    'ie': 17.0
}   # 三个突触权重的大小
synapse = {
    "tau_e": 1.,
    "tau_i": 2.,
    "E_ei": 0.,
    "E_ie": -100.,
    "E_ee": 0.,
    "E_ii": -85.,
}   # 突触的参数

# stdp的参数
tc_pre_ee = 20
tc_post_1_ee = 20
tc_post_2_ee = 40
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0

nInput = 784    # 输入节点数
nE = 400        # 兴奋性神经元数
nI = nE         # 抑制性神经元数

def getargs():
    parser = argparse.ArgumentParser(description="STDP框架研究2015年")

    parser.add_argument('--batch', type=int, default=1, help='批次大小')

    args = parser.parse_args()
    return args

args = getargs()

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


class Synapses_stdp_ine(nn.Module):
    """
    这个模块从输入到兴奋性神经元，并且连接具有突触可塑性 input-->e
    """
    def __init__(self, conn, post, E=0, tau=1, dt=.1):
        super().__init__()
        self.conn = conn
        self.post = post
        self.g = None  # 电导
        self.E = E  # 平衡电位，兴奋突触/抑制突触
        self.tau = tau  # 时间常数
        self.dt = dt  # 积分步长

    def forward(self, x):
        """
        输入图片泊松编码后的值，与兴奋性神经元连接，连接有stdp的作用
        args:
            x: 泊松编码后的 (1, 784)
        return:
            I: 输出到突触后的突触电流值
        """
        dg = self.conn(x)   # 电导的增量
        with torch.no_grad():
            if self.g is None:
                self.g = torch.zeros_like(dg, device=dg.device)
                I = torch.zeros_like(dg, device=dg.device)
            else:
                I = self.g * (self.E - self.post.mem)
            self.updata_g(dg)

        return I, dg

    def updata_g(self, dg):
        """
        使用突触前计算出来的spike,更新电导
        arg:
            dg: 电导的更新
        """
        self.g += (dg-self.g)*self.dt/self.tau


    def n_reset(self):
        self.g = None


class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 20.
        self.lif_e = LIFei(threshold=node_e["thresh"]-node_e["offset"]+self.theta , v_reset=node_e["v_reset"], dt=dt,
                Erest=node_e["Erest"], tau=node_e["tau"], refrac=node_e["refrac"])
        self.lif_i = LIFei(threshold=node_i["thresh"], v_reset=node_i["v_reset"], dt=dt,
                Erest=node_i["Erest"], tau=node_i["tau"], refrac=node_i["refrac"])
        self.fc_ine = nn.Linear(nInput, nE, bias=False)
        self.fc_ei = nn.Linear(nE, nI, bias=False)
        self.fc_ie = nn.Linear(nI, nE, bias=False)
        self.init_weight()  # 初始化权重
        self.create_syn()   # 创建突触
        self.Iie = 0        # 抑制性电流
        self.trace_pre = None   # 突触前的trace
        self.trace_post1 = None  # 突触后的trace1
        self.trace_post2 = None  # 突触后的trace2
        self.dw = 0

        self.voting = VotingLayer(label_shape=10)

    def forward(self, x):
        x = x.clone().detach()  # 突触前的峰

        if self.training:   # 训练状态
            self.normweight()
            Iine, dg = self.syn_ine(x)         # input-->e 兴奋性电流
            with torch.no_grad():
                # recurrent: e-->i-->e
                Iei, spike_e = self.syn_ei(Iine+self.Iie) # e-->i 兴奋性突触
                self.Iie, spike_i = self.syn_ie(Iei) # i-->e 抑制性突触
                self.getthresh(spike_e)

                # x: 突触前峰；spike_e: 突触后峰
                trace_pre = self.cal_trace_pre(x)
                trace_post1, trace_post2 = self.cal_trace_post(spike_e)

                # post(trace)-pre(spike)
                dw_post_pre = torch.autograd.grad(outputs=dg, inputs=self.fc_ine.weight,
                                          retain_graph=True, grad_outputs=trace_post1)[0]
                # pre(trace)-post(soike)
                x.data += trace_pre - x.data  # x变为trace(求导得出的值)
                dw_pre_post = torch.autograd.grad(outputs=dg, inputs=self.fc_ine.weight,
                                                  grad_outputs=spike_e*trace_post2)[0]

                self.dw = nu_ee_pre*dw_post_pre+nu_ee_post*dw_pre_post
                self.updateweight()
        else:   # 测试状态
            with torch.no_grad():
                Iine, dg = self.syn_ine(x)  # input-->e 兴奋性电流
                # recurrent: e-->i-->e
                Iei, spike_e = self.syn_ei(Iine + self.Iie)  # e-->i 兴奋性突触
                self.Iie, spike_i = self.syn_ie(Iei)  # i-->e 抑制性突触

        return spike_e

    def cal_trace_pre(self, x):
        """
        计算突触前的trace
        args:
            x : 突触前峰
        return:
            trace_pre: pre-post
        """
        if self.trace_pre is None:
            self.trace_pre = x.clone().detach()
        else:
            self.trace_pre += -self.trace_pre/tc_pre_ee*dt
            self.trace_pre[x>0.9] = 1
        return self.trace_pre.detach()

    def cal_trace_post(self, s):
        """
        计算突触后的trace
        args:
            s : 突触后峰
        return:
            trace_post1/2: post-pre
        """
        if self.trace_post1 is None:
            self.trace_post1 = torch.zeros_like(s)
            self.trace_post2 = torch.zeros_like(s)
        else:
            self.trace_post1 += -self.trace_post1/tc_post_1_ee*dt
            self.trace_post2 += -self.trace_post2/tc_post_2_ee*dt
        trace_post1 = self.trace_post1.clone().detach()
        trace_post2 = self.trace_post2.clone().detach()
        self.trace_post1[s>0.9] = 1
        self.trace_post2[s>0.9] = 1
        return trace_post1, trace_post2

    def init_weight(self):
        self.fc_ine.weight.data = create_weight.create_ine()
        self.fc_ei.weight.data = create_weight.create_ei()
        self.fc_ie.weight.data = create_weight.create_ie()

    def create_syn(self):
        self.syn_ei = synchem(pre=self.lif_e, conn=self.fc_ei, post=self.lif_i,
                        E=synapse["E_ei"], tau=synapse["tau_e"], dt=dt)     # e-->i 兴奋性突触
        self.syn_ie = synchem(pre=self.lif_i, conn=self.fc_ie, post=self.lif_e,
                        E=synapse["E_ie"], tau=synapse["tau_i"], dt=dt)     # i-->e 抑制性突触
        self.syn_ine = Synapses_stdp_ine(conn=self.fc_ine, post=self.lif_e,
                        E=synapse["E_ee"], tau=synapse["tau_e"], dt=dt)     # input-->e 兴奋性突触

    def getthresh(self, spike_e):
        """
        兴奋性神经元在放电后会使得阈值变大
        长时间不放点，会使得阈值越来越小，会使得神经元放电
        args:
            spike_e: 兴奋性神经元放电
        """
        self.theta += -self.theta/1e7*dt
        self.theta += 0.05*spike_e
        self.lif_e.threshold.data = node_e["thresh"]-node_e["offset"]+self.theta

    def updateweight(self):
        self.fc_ine.weight.data += self.dw
        self.fc_ine.weight.data = torch. \
            clamp(self.fc_ine.weight.data, min=0, max=wmax_ee)

    def normweight(self):
        self.fc_ine.weight.data /= self.fc_ine.weight.data.sum(1, True) / 78.4

    def n_reset(self):
        self.lif_e.n_reset()
        self.lif_i.n_reset()
        self.syn_ei.n_reset()
        self.syn_ie.n_reset()
        self.syn_ine.n_reset()
        self.Iie = 0
        self.trace_pre = None   # 突触前的trace
        self.trace_post1 = None  # 突触后的trace1
        self.trace_post2 = None  # 突触后的trace2
        self.dw = 0



if __name__ == "__main__":
    model = MNISTnet().to(device)
    # print(model.fc_ei.weight.data)

    transform = transforms.Compose([transforms.ToTensor()])
    train_iter = mnist(train=True, batch_size=args.batch, download=True,
                       data_path=datasetPath, transforms_IN=transform)  # transforms_IN=transform
    test_iter = mnist(train=False, batch_size=args.batch, download=True,
                      data_path=datasetPath, transforms_IN=transform)
    #
    input_intensity = 2
    start_input_intensity = input_intensity
    update_interval = 1000 # 更新投票层的频率
    plt.ion()
    for epoch in range(3):
        # 存储全部的spiking和标签
        spikefull = None
        labelfull = None
        model.train()
        for i, (images_train, labels) in enumerate(tqdm(train_iter)):
            current_spike_count = 0
            while current_spike_count<5:
                model.n_reset()
                images = images_train*255*input_intensity/8   #0-63.75Hz
                labels = labels.to(device)
                spike_e = 0
                # model.n_reset()
                for t in range(int(runTime/dt)):    #
                    x = Poisson(images, dt=dt).flatten(start_dim=1).to(device)
                    spike_e += model(x)

                current_spike_count = spike_e.sum()
                input_intensity += 1
                # if current_spike_count<5:
                #     print(current_spike_count)

                # images_rest = torch.zeros_like(images).flatten(start_dim=1).to(device)
                # for t in range(int(restTime/dt)):    #
                #     model(images_rest)

            input_intensity = start_input_intensity

            # 拼接批次维
            if spikefull is None:
                spikefull = spike_e
                labelfull = labels
            else:
                spikefull = torch.cat([spikefull, spike_e], 0)  # (B,C)
                labelfull = torch.cat([labelfull, labels], 0)  # (B,)
            if i % update_interval == 0 and i > 0:
                model.voting.assign_votes(spikefull, labelfull)  # 投票
                result = model.voting(spikefull)
                acc = (result == labelfull).float().mean()
                print("训练：", epoch, acc, i)
                spikefull = None
                labelfull = None

            # 动态权重可视化
            if i % 100 == 0 and i > 0:
                weight = model.fc_ine.weight.data.cpu().numpy()
                weight1 = weight[0].reshape(28,28)
                weight2 = weight[200].reshape(28, 28)
                weight3 = weight[50].reshape(28, 28)
                weight4 = weight[150].reshape(28, 28)
                weight12 = np.concatenate((weight1, weight2), axis=1)
                weight34 = np.concatenate((weight3, weight4), axis=1)
                weight = np.concatenate((weight12, weight34), axis=0)
                plt.clf()
                plt.imshow(weight, interpolation="nearest") # , vmin=0, vmax=wmax_ee
                plt.title(f"epoch={epoch}, i={i}")
                plt.colorbar()
                plt.pause(0.0000000000000000001)


        # ================== 测试 ==================
        model.eval()
        # 存储全部的spiking和标签
        spikefull = None
        labelfull = None
        for i, (images_test, labels) in enumerate(tqdm(test_iter)):  # train_iter test_iter
            model.n_reset()
            images = images_test * 255 * input_intensity / 8  # 0-63.75Hz
            labels = labels.to(device)
            spike_e = 0
            with torch.no_grad():
                for t in range(int(runTime/dt)):
                    x = Poisson(images, dt=dt).flatten(start_dim=1).to(device)
                    spike_e += model(x)

                # images_rest = torch.zeros_like(images).flatten(start_dim=1).to(device)
                # for t in range(int(restTime / dt)):  #
                #     model(images_rest)

                # 拼接批次维
                if spikefull is None:
                    spikefull = spike_e
                    labelfull = labels
                else:
                    spikefull = torch.cat([spikefull, spike_e], 0)  # (B,C)
                    labelfull = torch.cat([labelfull, labels], 0)  # (B,)

        result = model.voting(spikefull)
        acc = (result == labelfull).float().mean()
        print("测试：", epoch, acc, i)

    plt.ioff()
    plt.show()

