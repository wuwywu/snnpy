# encoding: utf-8
# Author    : HuangWF
# Datetime  : 2024/3/11
# User      : HuangWF
# File      : Multi_layer_connect.py
# 多层网络间的连接方式：随机连接、全连接、一对一连接


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 创建随机连接矩阵
def random_connect(n_pre, n_post, p):
    random_matrix = np.random.rand(n_post, n_pre)
    connection_matrix = (random_matrix < p).astype(int)
    return connection_matrix


# 创建一对一连接矩阵
def one_to_one_connect(n):
    return  np.eye(n, dtype=int)


# 创建一对多最近邻连接矩阵
def one_to_many_connect(n, k, bound=None):
    '''
    n:神经元个数
    k:最近邻连接个数
    bound:是否考虑周期边界
    '''
    one_to_many_matrix = np.zeros((n, n), dtype=int)

    # 不考虑周期边界
    if bound is None:
        for i in range(n):
            for delta in range(-k//2, k//2 + 1):
                neighbor = i + delta
                # 确保索引在合理范围内
                if 0 <= neighbor < n:
                    one_to_many_matrix[i, neighbor] = 1
                    
    # 考虑周期边界              
    if bound is not None:
        for i in range(n):
            one_to_many_matrix[i, i] = 1  # 自身连接
            for j in range(1, k//2 + 1):
                # 计算前后最近邻的索引，使用模运算确保索引循环
                left_neighbor = (i - j) % n
                right_neighbor = (i + j) % n
                
                one_to_many_matrix[i, left_neighbor] = 1
                one_to_many_matrix[i, right_neighbor] = 1

    return  one_to_many_matrix
