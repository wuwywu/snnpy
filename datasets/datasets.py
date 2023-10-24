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
def cifar10(train=True, batch_size=100, download=True, if_transforms=True, transforms_IN=None):
    data_path = r".\cifar-10"
    classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
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
        # print(train_dataset.class_to_idx)
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
    

# ===================================== MNIST ===================================== 
def mnist(train=True, batch_size=100, download=True, if_transforms=True, transforms_IN=None):
    data_path = r".\MNIST"
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    classes = ["0 - zero", "1 - one", "2 - two", "3 - three", "4 - four", "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine",
    ]
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


# ===================================== fashion_MNIST ===================================== 
def fashion_MNIST(train=True, batch_size=100, num_workers=8, if_transforms=True, transforms_IN=None):
    """
    获取fashion MNIST数据
    http://arxiv.org/abs/1708.07747
    :param train: 训练状态，True-->训练数据集，False-->测试数据集
    :param batch_size: 批次大小
    :param num_workers: 读取数据线程数
    :param if_transforms: 是否使用预处理
    :param transforms_IN: 自定义预处理
    :return: train_loader, test_loader
    """
    data_path = r".\fashion_MNIST"
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    if train:
        if transforms_IN is None:
            if if_transforms:
                # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
                transform_train = transforms.Compose([
                        transforms.RandomCrop(28, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor()])
            else: transform_train = None
        else:   transform_train = transforms_IN

        # 基于torch.utils.data.Dataset写的一个可迭代数据集
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform_train)
        # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        return train_loader
    # =============== 测试集 ===============
    else:
        if transforms_IN is None: 
            if if_transforms:
                transform_test = transforms.Compose([transforms.ToTensor()])
            else: transform_test = None
        else:   transform_test = transforms_IN

        test_dataset = torchvision.datasets.MNIST(root= data_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return test_loader


if __name__=="__main__":
    transform_test = transforms.Compose([transforms.ToTensor()])
    # cifar10(train=True, download=True, if_transforms=True) 
    FM = fashion_MNIST(train=True, batch_size=100, if_transforms=False, transforms_IN=transform_test)
    FM_iter = iter(FM)
    input, labels = next(FM_iter)
    i = 2
    plt.imshow(input[i].permute(1,2,0))
    print(labels[i])
    plt.show()


