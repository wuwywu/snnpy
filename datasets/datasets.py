# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : datasets.py
# 将各种用于神经网络的`各个数据集`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# ===================================== cifat-10 ===================================== 
def cifar10(train=True, batch_size=100, download=False, if_transforms=True, transforms_IN=None):
    data_path = r".\cifar-10"
    # =============== 训练集 ===============   
    if train:   
        if transforms_IN is None: 
            if if_transforms:
                # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
                transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else: transform_train = None
        else:   transform_train = transforms_IN

        train_dataset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=download, transform=transform_train)
        # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # print(train_dataset.class_to_idx)   # 打印出类型索引对应表
        ## 先去掉transform
        # print(train_dataset[5][1])
        # plt.imshow(train_dataset[5][0])
        # plt.show()
        return train_loader
    # =============== 测试集 ===============
    else:
        if transforms_IN is None: 
            if if_transforms:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else: transform_test = None
        else:   transform_test = transforms_IN
        
        test_dataset = torchvision.datasets.CIFAR10(root= data_path, train=False, download=download, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader
    
# ===================================== cifat-10 ===================================== 


# ===================================== MNIST ===================================== 
def mnist(train=True, batch_size=100, download=False, if_transforms=True, transforms_IN=None):
    data_path = r".\MNIST"
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    # =============== 训练集 ===============
    if train:
        if transforms_IN is None:
            if if_transforms:
                # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
                transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                ])
            else: transform_train = None
        else:   transform_train = transforms_IN

        # 基于torch.utils.data.Dataset写的一个可迭代数据集
        train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=download, transform=transform_train)
        # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return train_loader
    # =============== 测试集 ===============
    else:
        if transforms_IN is None: 
            if if_transforms:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                #   transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                ])
            else: transform_test = None
        else:   transform_test = transforms_IN

        test_dataset = torchvision.datasets.MNIST(root= data_path, train=False, download=download, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return test_loader

# ===================================== MNIST ===================================== 


if __name__=="__main__":
    cifar10(train=True, download=True, if_transforms=True) 

