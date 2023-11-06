# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : utils.py
# 将各种用于神经网络的`一些工具`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中\
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr*(0.1**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 将标签转化为one-hot
def toOneHot(labels, num_class=10):
    batch_size = labels.view(-1).shape[0]
    labels_ = torch.zeros(batch_size, num_class).scatter_(1, labels.view(-1, 1), 1)
    return labels_

# 使用输出和标签值(非one-hot)得到，总数和正确数
def calc_correct_total(outputs, labels):
    """
    args:
        :param outputs: 网络输出值
        :param labels: 标签值
    return:
        :param total: 标签总数
        :param correct: 输出正确数
    """
    _, predicted = outputs.cpu().max(1) # max()输出(值，索引)
    labels = labels.view(-1)
    total = float(labels.size(0)) # 输出标签总数
    correct = float(predicted.eq(labels).sum().item()) # 输出正确数
    return total, correct


def setup_seed(seed):
    """
    为CPU，GPU，所有GPU，numpy，python设置随机数种子，并禁止hash随机化
    :param seed: 随机种子值
    :return:
    """
    torch.manual_seed(seed)             # 给cpu设置随机种子
    torch.cuda.manual_seed(seed)        # 给当前gpu设置随机种子
    torch.cuda.manual_seed_all(seed)    # 给所有gpu设置随机种子
    np.random.seed(seed)                # 给numpy设置随机种子
    random.seed(seed)                   # 给自带随机函数设置随机种子

    # 该标记可启用 cudnn 自动调整器，它能找到使用的最佳算法
    # 针对特定配置。(该模式适用于输入尺寸没有变化的情况）
    torch.backends.cudnn.benchmark = False  # 禁止自动优化配置带来的随机
    # 该标志只允许使用与基准不同的确定性 cudnn 算法。
    torch.backends.cudnn.deterministic = True   # 获得确定性（牺牲最佳性能）

    os.environ['PYTHONHASHSEED'] = str(seed)    # 禁止hash随机化


class Checkpoint:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch


if __name__=="__main__":
    setup_seed(4)


