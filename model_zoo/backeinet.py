import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(r"../")
import torch
import torch.nn as nn
import argparse

from base.nodes.NeuNodes import LIFbackEI
from base.connection.layers import addTLayer, FRLayer
from base.encoder.encoder import encoder
from datasets.datasets import mnist, fashion_MNIST, cifar10
from base.utils.utils_snn import lr_scheduler, toOneHot, calc_correct_total
from base.utils.utils import setup_seed
from base_module import BaseModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 固定随机种子
setup_seed(110)

parser = argparse.ArgumentParser(description="输入研究变化参数")

parser.add_argument('--batch', type=int, default=100, help='批次大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--epoch', type=int, default=100, help='学习周期')
parser.add_argument('--time_window', type=int, default=10, help='LIF时间窗口')
parser.add_argument('--dataset', type=str, default="fashion", choices=["mnist", "fashion", "cifar"], help='数据集类型')

args = parser.parse_args()

class MNISTNet(BaseModule):
    def __init__(self, time_window=5, encodelayer_way=1, if_back=True, if_ei=True, data="mnist"):
        super().__init__(time_window=time_window)
        if data == 'mnist':
            cfg_conv = ((1, 15, 5, 1, 0), (15, 40, 5, 1, 0))
            cfg_fc = (300, 10)
            cfg_kernel = (24, 8, 4)
            cfg_backei = 2
        if data == 'fashion':
            cfg_conv = ((1, 32, 5, 1, 2), (32, 64, 5, 1, 2))
            cfg_fc = (1024, 10)
            cfg_kernel = (28, 14, 7)
            cfg_backei = 1

        conv1 = nn.Conv2d(cfg_conv[0][0], cfg_conv[0][1], cfg_conv[0][2], cfg_conv[0][3], cfg_conv[0][4])   #  , bias=False
        LIF1 = LIFbackEI(cfg_conv[0][1], if_ei=if_ei, if_back=if_back, cfg_backei=cfg_backei)
        pool1 = nn.AvgPool2d(2)
        conv2 = nn.Conv2d(cfg_conv[1][0], cfg_conv[1][1], cfg_conv[1][2], cfg_conv[1][3], cfg_conv[1][4])  # , bias=False
        LIF2 = LIFbackEI(cfg_conv[1][1], if_ei=if_ei, if_back=if_back, cfg_backei=cfg_backei)
        pool2 = nn.AvgPool2d(2)
        flatten = nn.Flatten()
        fc1 = nn.Linear(cfg_kernel[2] * cfg_kernel[2] * cfg_conv[1][1], cfg_fc[0])
        LIF3 = LIFbackEI(if_ei=False,if_back=False)
        fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        LIF4 = LIFbackEI(if_ei=False, if_back=False)
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


class CIFARNet(BaseModule):
    def __init__(self, time_window=5, encodelayer_way=1, if_back=True, if_ei=True):
        super().__init__(time_window=time_window)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            LIFbackEI(in_channel=128, if_back=if_back, if_ei=if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            LIFbackEI(in_channel=256, if_back=if_back, if_ei=if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            LIFbackEI(in_channel=512, if_back=if_back, if_ei=if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1024),
            LIFbackEI(if_back=False, if_ei=False),
            nn.Dropout(0.5),

            nn.Linear(1024, 10),
            LIFbackEI(if_back=False, if_ei=False)
        )
        self.encodelayer = encoder(schemes=encodelayer_way, time_window=time_window)
        self.FiringRate = FRLayer(if_T=False, if_grad=True, time_window=time_window)

    def forward(self, inputs):
        # x --> (N, C, H, W)
        self.reset()
        inputs = self.encodelayer(inputs) # (N, C, H, W, T)
    
        for t in range(self.time_window):
            x = inputs[..., t]
            x = self.feature(x)
            fr = self.FiringRate(x)
        return fr   


if __name__ == "__main__":
    batch_size = args.batch        # 批次大小
    learning_rate = args.lr   # 学习率

    def train(model, criterion, optimizer, epoch, train_loader):
        model.train()   # 开启训练
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            outputs = model(images)
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
                # labels_ = toOneHot(labels).to(device)
                # loss = criterion(outputs, labels_)
                total1, correct1 = calc_correct_total(outputs, labels)
                total += total1
                correct += correct1

        acc = 100. * correct / total
        print(acc)
                 
    # data_path = r".\MNIST"
    # 数据集
    if args.dataset=='mnist':
        train_loader = mnist(train=True, batch_size=batch_size, download=True, if_transforms=True)
        test_loader = mnist(train=False, batch_size=batch_size, download=True, if_transforms=True)
        # 模型
        snn = MNISTNet(time_window=args.time_window, encodelayer_way=1, if_back=True, if_ei=True, data=args.dataset) # 创建模型
    if args.dataset=="fashion":
        train_loader = fashion_MNIST(train=True, batch_size=batch_size, num_workers=8, if_transforms=True, transforms_IN=None)
        test_loader = fashion_MNIST(train=False, batch_size=batch_size, num_workers=8, if_transforms=True, transforms_IN=None)
        # 模型
        snn = MNISTNet(time_window=args.time_window, encodelayer_way=1, if_back=True, if_ei=True, data=args.dataset) # 创建模型
    if args.dataset=="cifar":
        train_loader = cifar10(train=True, batch_size=batch_size, download=True, if_transforms=True, transforms_IN=None)
        test_loader = cifar10(train=False, batch_size=batch_size, download=True, if_transforms=True, transforms_IN=None)
        snn = CIFARNet(time_window=args.time_window, encodelayer_way=1, if_back=True, if_ei=True) # 创建模型
    
    snn.to(device)
    criterion = nn.MSELoss() # loss函数
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)    # 优化器
    # print(list(snn.state_dict().keys()))

    num_epochs = args.epoch
    for epoch in range(num_epochs):
        train(snn, criterion, optimizer, epoch, train_loader)
        test(snn, criterion, epoch, test_loader)
        lr_scheduler(optimizer, epoch, learning_rate, 40)