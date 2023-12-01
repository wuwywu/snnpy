# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : base_SNNmnist.py
# 使用基础LIF构建一个基础的SNN

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(r"../")
import torch
import torch.nn as nn

from nodes import LIFNode
from connection import addTLayer, FRLayer
from encoder import encoder
from datasets.datasets import mnist
from utils import lr_scheduler, toOneHot, calc_correct_total
from base_module import BaseModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 手写数据集
class MNISTNet(BaseModule):  # Example net for MNIST
    def __init__(self, time_window=5, encodelayer_way=1):
        super().__init__(time_window=time_window)
        conv1 = nn.Conv2d(1, 15, 5, 1, 2)   #  , bias=False
        LIF1 = LIFNode()
        pool1 = nn.AvgPool2d(2)
        conv2 = nn.Conv2d(15, 40, 5, 1, 2)  # , bias=False
        LIF2 = LIFNode()
        pool2 = nn.AvgPool2d(2)
        flatten = nn.Flatten()
        fc1 = nn.Linear(7 * 7 * 40, 300)
        LIF3 = LIFNode()
        fc2 = nn.Linear(300, 10)
        LIF4 = LIFNode()
        self.FiringRate = FRLayer(if_T=True, if_grad=True, time_window=time_window)

        self.feature1 = nn.Sequential(
            addTLayer(conv1, time_window=self.time_window),
            addTLayer(LIF1, time_window=self.time_window),
            addTLayer(pool1, time_window=self.time_window),

            addTLayer(conv2, time_window=self.time_window),
            addTLayer(LIF2, time_window=self.time_window),
            addTLayer(pool2, time_window=self.time_window),

            addTLayer(flatten, time_window=self.time_window),

            addTLayer(fc1, time_window=self.time_window),
            addTLayer(LIF3, time_window=self.time_window),
            addTLayer(fc2, time_window=self.time_window),
            addTLayer(LIF4, time_window=self.time_window),
        )
        # 1、直流编码;2、泊松编码
        self.encodelayer = encoder(schemes=encodelayer_way, time_window=time_window) 

    def forward(self, x):
        # x --> (N, C, H, W)
        self.reset()
        x = self.encodelayer(x)
        # x --> (N, C, H, W, T)
        x = self.feature1(x)
        out = self.FiringRate(x)
        return out


if __name__ == "__main__":
    batch_size = 100        # 批次大小
    learning_rate = 0.001   # 学习率
    # 固定随机种子
    seed = 12
    torch.manual_seed(seed)

    def train(model, criterion, optimizer, epoch, train_loader):
        model.train()   # 开启训练
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            outputs = model(images)
            # labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1).to(device)
            labels_ = toOneHot(labels).to(device)
            loss = criterion(outputs, labels_)
            # 更新参数空间
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 可视化
            if (i+1)%100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
    
    def test(model, criterion, epoch, test_loader):
        model.eval()                # 开启评估
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.float().to(device)
                outputs = model(images)
                # labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1).to(device)
                # labels_ = toOneHot(labels).to(device)
                # loss = criterion(outputs, labels_)
                total1, correct1 = calc_correct_total(outputs, labels)
                total += total1
                correct += correct1

        acc = 100. * correct / total
        print(acc)
                 
    data_path = r".\MNIST"
    # 数据集
    train_loader = mnist(train=True, batch_size=batch_size, download=True,
          data_path = data_path, if_transforms=True)
    test_loader = mnist(train=False, batch_size=batch_size, download=True,
          data_path = data_path, if_transforms=True)
    # 模型
    snn = MNISTNet() # 创建模型
    snn.to(device)
    criterion = nn.MSELoss() # loss函数
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)    # 优化器
    # print(list(snn.state_dict().keys()))

    num_epochs = 10
    for epoch in range(num_epochs):
        train(snn, criterion, optimizer, epoch, train_loader)
        test(snn, criterion, epoch, test_loader)
        lr_scheduler(optimizer, epoch, learning_rate, 40)

