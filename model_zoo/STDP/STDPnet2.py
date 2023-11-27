# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/26
# User      : WuY
# File      : STDPnet.py
# paper     : Yiting Dong, An unsupervised STDP-based spiking neural network inspired by biologically plausible learning rules and connections
# doi       : https://doi.org/10.1016/j.neunet.2023.06.019
# 这个代码按照开源代码修改，与文中有非常大的出入
"""
文章中的网络结构
    卷积层(neuron) --> 2x2最大池化层 --> 尖峰归一化层 --> 全连接层(neuron)
    convolutional layer --> 2*2 max pooling layer --> spiking
normalization layer -->  fully connected layer
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(r"../")
sys.path.append(r"../../")
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse

from base.nodes.nodes import LIFSTDP
from base.connection.layers import VotingLayer, WTALayer, LateralInhibition
from datasets.datasets import mnist
from base.utils.utils import setup_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 固定随机种子
setup_seed(0)

parser = argparse.ArgumentParser(description="STDP框架研究")

parser.add_argument('--batch', type=int, default=200, help='批次大小')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
parser.add_argument('--epoch', type=int, default=300, help='学习周期')
parser.add_argument('--time_window', type=int, default=100, help='LIF时间窗口')

args = parser.parse_args()

# STDP的卷积层+neuron
class STDPConv(nn.Module):
    """
    STDP更新权重的卷积层
    网络结构:
        1、卷积; 2、LIF(spiking neuron);
    网络中的构造：
        1、赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
        2、适应性阈值平衡(Adaptive threshold balance, ATB)
        3、适应性突触滤波器(Adaptive synaptic filter, ASF) --> 全连接层中没有
        ASF根据阈值计算，所以必须跟ATB一起进行
    args:
        :params
        in_planes: 卷积层输入特征
        out_planes: 卷积层输出特征
        kernel_size: 卷积核大小
        stride: 卷积的步长
        padding: 卷积的填充
        groups: 批次分组
        decay: LIF的衰减因子
        decay_trace: STDP计算trace时的衰减因子
        offset: 计算梯度时与trace相减的偏置
        inh: 侧抑制的抑制率(mode="threshold", 自适应性阈值)
    """
    def __init__(self, in_planes, out_planes,
                 kernel_size, stride, padding, groups,
                 decay=torch.exp(-1.0 / torch.tensor(100.0)), decay_trace=0.99, offset=0.3,
                 inh=1.625):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.lif = LIFSTDP(decay=decay, mem_detach=True)
        self.WTA = WTALayer(k=1)    # 赢者通吃
        self.normweight(False)
        # 侧抑制
        self.lateralinh = LateralInhibition(self.lif, inh, mode="threshold")
        # STDP参数
        self.trace = None
        self.decay_trace = decay_trace
        self.offset = offset

        self.dw = 0. # STDP的改变的权重变化量（/batch*T）

    def forward(self, x, time_window=10):
        """
        args:
            x: 输入脉冲(B, C, H, W)
            time_window: 时间窗口
        return:
            :spikes: 是脉冲 (B,C,H,W)
        """
        x = x.clone().detach()  # 突触前的峰
        i = self.conv(x)  # 输入电流(经过卷积后)
        thre_max = self.getthresh(i)  # 自适应性阈值
        i_ASF = self.ASFilter(i, thre_max)  # 通过适应阈值对电流进行滤波

        spikes = 0
        for T in range(time_window):
            s = self.mem_update(i_ASF)  # 输出脉冲
            spikes += s
        return spikes

    def STDP(self, current, i_ASF):
        """
        利用STDP获得权重的变化量
        所有的结构都会在这个过程中利用
        args:
            current : 通过卷积后的电流
            :i_ASF : [B,C,H,W] -- 通过卷积和滤波器之后的电流(若包含时间就将其降维到B中)
        return:
            :s是脉冲 (B,C,H,W)
            :dw更新量 (out_planes,in_planes,H,W)
        """
        with torch.no_grad():
            s = self.mem_update(i_ASF)  # 输出脉冲
            # if self.training:   # 是否训练
            #     trace = self.cal_trace(x)  # 通过突触前峰更新trace
            #     x.data += trace - x.data - self.offset  # x变为trace(求导得出的值)
        if self.training:   # 是否训练
            dw = torch.autograd.grad(outputs=i_ASF, inputs=self.conv.weight, grad_outputs=s, retain_graph=True)[0]
            # print(x.size(0))
        else:
            dw = 0.
        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        x : [B,C,W,H] -- 突触前峰
        """
        if self.trace is None:
            self.trace = nn.Parameter(x.clone().detach(), requires_grad=False)
        else:
            self.trace *= self.decay_trace
            self.trace += x
        return self.trace.detach()

    def mem_update(self, x):
        """
        LIF的更新:(经过赢着通吃)
        赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
        args:
            x: 通过卷积核后的输入电流
        return:
            spiking: 输出的脉冲0/1
        """
        x = self.lif(x)  # 通过LIF后的脉冲
        if x.max() > 0:  # 判断有没有脉冲产生
            x = self.WTA(x)      # 赢者通吃(winner take all)
            self.lateralinh(x)   # 抑制不放电神经元的膜电位大小
        return x

    def getthresh(self, current):
        """
        适应性阈值平衡(Adaptive threshold balance, ATB)
        args:
            current: 卷积后的电流(B,C,H,W)
        retuen:
            维度B上的最大电流，阈值（ATB 确保不会因电流过大而丢失信息。）
            (文章中是维度B上的，而程序中是维度C上的，需要调试)
        """
        # thre_max = current.max(1, True)[0].max(2, True)[0].max(3, True)[0]+0.0001
        thre_max = current.max(0, True)[0].max(2, True)[0].max(3, True)[0]+0.0001
        self.lif.threshold.data = thre_max # 更改LIF的阈值
        return thre_max

    def ASFilter(self, current, thre):
        """
        适应性突触滤波器(Adaptive synaptic filter, ASF)
        args:
            current: 卷积后的电流(B,C,H,W)
            thre: 适应性阈值平衡机制调节调节后的阈值
        return：
            current_ASF: 滤波后的电流
        """
        current = current.clamp(min=0)      # 文章中并没写明裁剪电流(需要调试,影响收敛速度)
        current_ASF = thre / (1 + torch.exp(-(current - 4 * thre / 10) * (8 / thre)))
        return current_ASF

    def normgrad(self, force=True):
        """
        将STDP带来的权重变化量放入权重梯度中，然后使用优化器更新权重
        """
        if force:
            min = self.conv.weight.grad.data.min(1, True)[0].min(2, True)[0].min(3, True)[0]
            max = self.conv.weight.grad.data.min(1, True)[0].max(2, True)[0].max(3, True)[0]
            self.conv.weight.grad.data -= min
            tmp = self.offset * max
        else:
            tmp = self.offset * self.spike.mean(0, True).mean(2, True).mean(3, True).permute(1, 0, 2, 3)
        self.conv.weight.grad.data -= tmp
        self.conv.weight.grad.data = -self.conv.weight.grad.data

    def normweight(self, clip=False):
        """
        权重在更新后标准化，防止它们发散或移动
        self.conv.weight --> (N,C,H,W)
        args:
            clip: 是否裁剪权重
        """
        if clip:
            self.conv.weight.data = torch. \
                clamp(self.conv.weight.data, min=-3, max=1.0)
        else:
            N, C, H, W = self.conv.weight.data.shape

            avg = self.conv.weight.data.mean(1, True).mean(2, True).mean(3, True)   # 每个批次的均值不一样
            # 将除了批次维度的其他所有维度全部集中在第2维上，然后可以求出批次上的标准差
            tmp = self.conv.weight.data.reshape(N, 1, -1, 1)
            self.conv.weight.data -= avg
            self.conv.weight.data /= tmp.std(2, unbiased=False, keepdim=True)   # 不使用无偏标准差

    def reset(self):
        """
        重置: 1、LIF的膜电位和spiking; 2、STDP的trace; 3、梯度;
        """
        self.lif.n_reset()
        self.trace = None
        self.dw = 0    # 将权重变化量清0


plus = 0.002  # 控制增长率的系数(线性层适应性阈值平衡),文中给的0.001
class STDPLinear(nn.Module):
    """
        STDP更新权重的全连接层
        网络结构:
            1、全连接层; 2、LIF(spiking neuron);
        网络中的构造：
            1、赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
            2、适应性阈值平衡(Adaptive threshold balance, ATB)
        args:
            :params
            in_planes: 全连接层输入神经元数
            out_planes: 全连接层输出神经元数
            decay: LIF的衰减因子
            decay_trace: STDP计算trace时的衰减因子
            offset: 计算梯度时与trace相减的偏置
            inh: 侧抑制的抑制率(mode="max", 自适应性阈值)
        """
    def __init__(self, in_planes, out_planes,
                 decay=0.99, decay_trace=0.99, offset=0.3,
                 inh=1.625):
        super().__init__()
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.lif = LIFSTDP(decay=decay, mem_detach=True)    # 类似于IF
        self.WTA = WTALayer(k=1)  # 赢者通吃(全连接，每一批所有神经元只有一个k=1放电)
        self.normweight(False)
        # 侧抑制
        self.lateralinh = LateralInhibition(self.lif, inh, mode="max")  # 维度为(N,C)所以使用max就可以了
        # STDP参数
        self.trace = None
        self.decay_trace = decay_trace
        self.offset = offset

        self.dw = 0.  # STDP的改变的权重变化量（/batch*T）
        # 适应性阈值平衡参数
        self.thre_init = True
        # self.threshold = torch.ones(out_planes) *10

    def forward(self, x, time_window=10):
        """
        args:
           x: 输入脉冲(B, C)
           time_window: 时间窗口
        return:
           :spikes: 是脉冲 (B,C)
        """
        x = x.clone().detach()  # 突触前的峰
        i = self.linear(x)  # 输入电流(经过卷积后)
        self.current = i.detach()

        spikes = 0
        for T in range(time_window):
            s = self.mem_update(i)  # 输出脉冲
            spikes += s
        return spikes

    def STDP(self, i):
        """
        利用STDP获得权重的变化量
        所有的结构都会在这个过程中利用
        args:
            :i : [B,C] -- 通过线性层后的电流(若包含时间就将其降维到B中)
        return:
            i 经过全连接后的电流
            :s是脉冲 (B,C)
            :dw更新量 (out_planes,in_planes)
        """
        with torch.no_grad():
            s = self.mem_update(i)  # 输出脉冲
            # if self.training:  # 是否训练
            #     trace = self.cal_trace(x)  # 通过突触前峰更新trace
            #     x.data += trace - x.data - self.offset  # x变为trace(求导得出的值)
        if self.training:  # 是否训练
            dw = torch.autograd.grad(outputs=i, inputs=self.linear.weight, grad_outputs=s, retain_graph=True)[0]
            # print(x.size(0))
            # dw /= x.size(0)  # 批次维度在求导时相加，除去
            # print(self.dw)
        else:
            dw = 0.
        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        x : [B,C] -- 突触前峰
        """
        if self.trace is None:
            self.trace = nn.Parameter(x.clone().detach(), requires_grad=False)
        else:
            self.trace *= self.decay_trace
            self.trace += x
        return self.trace.detach()

    def mem_update(self, x):
        """
        LIF的更新:(经过赢着通吃)
        赢者通吃+侧抑制(winner take all+ Adaptive lateral inhibitory connection)
        args:
            x: 通过全连接后的输入电流 （B,C）
        return:
            spiking: 输出的脉冲0/1
        """
        if self.thre_init:
            # self.lif.threshold.data = torch.ones(x.shape[1], device=x.device) *10
            self.lif.threshold.data = (x.max(0)[0].detach()*3).to(device)
            self.thre_init = False
        xori = x
        x = self.lif(x)  # 通过LIF后的脉冲
        if x.max() > 0:  # 判断有没有脉冲产生
            x = self.WTA(x)  # 赢者通吃(winner take all)
            self.lateralinh(x, xori)  # 抑制不放电神经元的膜电位大小
        return x

    def getthresh(self, s_post, plus=plus):
        """
        适应性阈值平衡(Adaptive threshold balance, ATB)
        全连接在测试的时候似乎不用阈值平衡
        args:
            current: 全连接后的电流(B,C)
            s_post: 通过LIF后，神经元发放脉冲(B,C)
        retuen:
            None
        """
        self.lif.threshold.data += (plus * self.current * s_post).sum(0)
        tmp = self.lif.threshold.max() - 350
        if tmp > 0:
            self.lif.threshold.data -= tmp

    def normgrad(self, force=True):
        """
        将STDP带来的权重变化量放入权重梯度中，然后使用优化器更新权重
        """
        self.linear.weight.grad.data = -self.linear.weight.grad.data

    def normweight(self, clip=False):
        """
        权重在更新后标准化，防止它们发散或移动
        self.conv.weight --> (out_planes,in_planes)
        args:
            clip: 是否裁剪权重
        """
        if clip:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
        else:
            # 文章似乎没有提及裁剪权重的情况(需要调试).似乎对收敛速影响特别大
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
            # 用平均值的效果很差, 代码中似乎实现的是除以最大值(/0.1, /0.01学习效果会越来越差)
            # self.linear.weight.data /= self.linear.weight.data.mean(0, keepdims=True) / 0.01 # .mean(1, keepdims=True)
            self.linear.weight.data /= self.linear.weight.data.max(1, True)[0] / 0.1

    def reset(self):
        """
        重置: 1、LIF的膜电位和spiking; 2、STDP的trace; 3、梯度; 4、适应性阈值平衡参数
        """
        self.lif.n_reset()
        self.trace = None
        self.dw = 0.    # 将权重变化量清0
        # self.thre_init = True     # 4、适应性阈值平衡参数(这个影响非常大)


class STDPMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x, time_window):  # [batch,c,h,w]
        x = self.pool(x)
        return x


class STDPFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x, time_window):  # [batch,c,h,w]
        return self.flatten(x)


class Normliaze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, time_window):    # [batch,c,h,w]
        max = x.max(1, True)[0].max(2, True)[0].max(3, True)[0]
        x /= max+0.000001
        return x


inh = 25  # 调节抑制程度的系数(全连接层)，文中似乎没有说明全连接层的问题
inh2=1.625
channel = 12    # 卷积层的输出特征图数
neuron = 6400   # 投票层的神经元数量
class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            STDPConv(in_planes=1, out_planes=channel, kernel_size=3,    # decay=0.2,
                     stride=1, padding=1, groups=1,
                     decay_trace=0.99, offset=0.3, inh=1.625),
            STDPMaxPool(2, 2, 0),
            Normliaze(),
            # STDPConv(12, 48, 3, 1, 1, 1, inh=inh2),
            # STDPMaxPool(2, 2, 0, static=True),
            # Normliaze(),

            STDPFlatten(start_dim=1),
            STDPLinear(196* channel, neuron, inh=inh, decay_trace=0.1, offset=0.0)   # 5--169, 3--196 , decay_trace=0.0, offset=0.0
        ])

        self.voting = VotingLayer(label_shape=10, alpha=.5) # alpha 指数衰减

    def forward(self, x, inlayer, outlayer, time_window, train_layer=None):
        if train_layer is None:
            for i in range(inlayer, outlayer + 1):
                x = self.net[i](x, time_window)
        else:
            for i in range(inlayer, outlayer + 1):
                if i==train_layer:
                    self.net[i].train()
                else:
                    self.net[i].eval()
                x = self.net[i](x, time_window)

        return x

    def normgrad(self, layer):
        # 只有卷积和全连接层有
        self.net[layer].normgrad()

    def normweight(self, layer, clip=False):
        # 只有卷积和全连接层有
        self.net[layer].normweight(clip)

    def reset(self, layer):
        if isinstance(layer, list):
            for i in layer:
                self.net[i].reset()
        else:
            self.net[layer].reset()


if __name__ == "__main__":
    # 创建网络
    model = MNISTnet().to(device)
    # print(model.net)

    # 数据集
    # transform = transforms.Compose([transforms.Resize((28, 28)),
    #                                 transforms.Grayscale(num_output_channels=1),
    #                                 transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     #   transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    train_iter = mnist(train=True, batch_size=args.batch, transforms_IN=transform)   # transforms_IN=transform
    test_iter = mnist(train=False, batch_size=args.batch, transforms_IN=transform)

    conv_lin_params = [0, 3]  # 卷积和的参数所在的位置(告诉优化器要学习的参数在哪)
    conv_lin_list = [index for index, i in enumerate(model.net) if
                     isinstance(i, (STDPConv, STDPLinear))]  # 卷积和线性层在列表的位置
    # print(conv_lin_list)
    # print(list(model.parameters())[3:4])

    # 创建优化器
    # print(list(model.named_parameters())[conv_lin_params[0]:conv_lin_params[0] + 1])
    optimizer_conv = torch.optim.SGD(list(model.parameters())[conv_lin_params[0]:conv_lin_params[0] + 1], lr=.1)
    # optimizer_conv = torch.optim.Adam(list(model.parameters())[conv_lin_params[0]:conv_lin_params[0] + 1], lr=0.1)
    lr = args.lr
    # print(list(model.named_parameters())[conv_lin_params[1]:conv_lin_params[1] + 1])
    optimizer_lin = torch.optim.SGD(list(model.parameters())[conv_lin_params[1]:conv_lin_params[1] + 1], lr=lr)
    # optimizer_lin = torch.optim.Adam(list(model.parameters())[conv_lin_params[1]:conv_lin_params[1] + 1], lr=lr)
    optimizer = [optimizer_conv, optimizer_lin]

    time_window_conv = 100  # 时间窗口(文章中用的300)
    time_window_lin = 200

    # 创建编码器 2、泊松编码
    # encoder_conv = encoder(schemes=2, time_window=time_window_conv)
    # encoder_lin = encoder(schemes=2, time_window=time_window_lin)\

    # ================== 预训练(卷积层) ==================
    model.train()
    for layer in range(len(conv_lin_list) - 1):  # 遍历所有卷积层
        for epoch in range(3):
            for i, (images, labels) in enumerate(train_iter):
                images = images.float().to(device)
                labels = labels.to(device)
                # images = encoder_conv(images)   # [..., t]
                spikes = model(images, 0, conv_lin_list[layer], time_window_conv)
                optimizer[layer].zero_grad()
                spikes.sum().backward(torch.tensor(1 / (spikes.shape[0] * spikes.shape[2] * spikes.shape[3])))
                model.normgrad(conv_lin_list[layer])
                # print(model.net[conv_lin_list[layer]].conv.weight.grad)
                optimizer[layer].step()
                model.normweight(conv_lin_list[layer], clip=False)
                model.reset(conv_lin_list)  # 重置网络中的卷积层和全连接层
            print("layer", layer, "epoch", epoch, 'Done')

    best = 0
    for epoch in range(args.epoch):
        model.train()
        # ================== 训练(卷积层) ==================
        """
        # 卷积层（一层，可能有两层）
        for layer in range(len(conv_lin_list) - 1):  # 遍历所有卷积层
            # for epoch in range(3):
            for i, (images, labels) in enumerate(train_iter):
                model.reset(conv_lin_list)  # 重置网络中的卷积层和全连接层
                images = images.float().to(device)
                labels = labels.to(device)
                # images = encoder_conv(images)   # [..., t]
                fireRate = 0
                for t in range(time_window_conv):
                    spikes = model(images, 0, conv_lin_list[layer], time_window_conv, train_layer=conv_lin_list[layer])
                    fireRate += spikes
                optimizer[layer].zero_grad()
                model.normgrad(conv_lin_list[layer])
                optimizer[layer].step()
                model.normweight(conv_lin_list[layer], clip=False)
                # print("layer", layer, "epoch", epoch, 'Done')
         """

        # ================== 训练(线性层) ==================
        # 线性层
        layer = len(conv_lin_list) - 1  # 线性层的位置（就最后一层）
        # model.train()
        # 存储全部的spiking和标签
        spikefull = None
        labelfull = None
        for i, (images, labels) in enumerate(train_iter):
            images = images.float().to(device)
            labels = labels.to(device)
            # images = encoder_lin(images)  # [..., t]
            spikes = model(images, 0, conv_lin_list[layer], time_window_lin)    # [B1,C]

            # 拼接批次维
            if spikefull is None:
                spikefull = spikes
                labelfull = labels
            else:
                spikefull = torch.cat([spikefull, spikes], 0)  # (B,C)
                labelfull = torch.cat([labelfull, labels], 0)   #  (B,)

            optimizer[layer].zero_grad()
            spikes.sum().backward()
            model.normgrad(conv_lin_list[layer])
            optimizer[layer].step()
            model.net[conv_lin_list[layer]].getthresh(spikes.detach())
            model.normweight(conv_lin_list[layer], clip=False)
            model.reset(conv_lin_list)  # 重置网络中的卷积层和全连接层

        model.voting.assign_votes(spikefull, labelfull) # 投票
        result = model.voting(spikefull)
        acc = (result == labelfull).float().mean()
        print("训练：", epoch, acc, 'channel', channel, "n", neuron)

        # 减少学习率
        # lr_scheduler(optimizer[layer], epoch, init_lr=lr, lr_decay_epoch=20)

        # ================== 测试 ==================
        model.eval()
        # 存储全部的spiking和标签
        spikefull = None
        labelfull = None
        layer = len(conv_lin_list) - 1  # 线性层的位置（就最后一层）
        for i, (images, labels) in enumerate(test_iter):# train_iter test_iter
            images = images.float().to(device)
            labels = labels.to(device)
            # images = encoder_lin(images)  # [..., t]
            with torch.no_grad():
                spikes = model(images, 0, conv_lin_list[layer], time_window_lin)  # [B1,C]
                # 拼接批次维
                if spikefull is None:
                    spikefull = spikes
                    labelfull = labels
                else:
                    spikefull = torch.cat([spikefull, spikes], 0)  # (B,C)
                    labelfull = torch.cat([labelfull, labels], 0)  # (B,)

                model.reset(conv_lin_list)  # 重置网络中的卷积层和全连接层

        result = model.voting(spikefull)
        acc = (result == labelfull).float().mean()
        print("测试：", epoch, acc, 'channel', channel, "n", neuron)
        # print(torch.where(fireRate>0))

        if best < acc:
            best = acc
        print("test", acc, "best", best)

