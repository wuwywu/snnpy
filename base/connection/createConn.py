# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/6
# User      : WuY
# File      : createConn.py
# 将各种用于网络的`连接矩阵`集合到这里
import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def alltoall(in_num=10, out_num=None, selfconn=False):
    """
    全连接all-to-all
    创建一个全 1 矩阵(控制是否自突触)
    args:
        in_num:     输入节点数
        out_num:    输出节点数
        selfconn:   是否自连接(对角线上为1/0)
    return:
        matrix:     连接矩阵shape(out_num, in_num)
    """
    if out_num is None:
        out_num = in_num
    matrix = torch.ones((out_num, in_num))
    if not selfconn:
        assert in_num == out_num, "输入和输出维度不一致"
        # 将对角线上的元素设置为 0
        matrix.fill_diagonal_(0)

    return matrix


# ======================== 小世界网络 ========================
def small_world(n=10, k=2, p=0.1):
    """
    创建一个小世界网络的连接矩阵(双向)
    reference: D.J. Watts, S.H. Strogatz, Collective dynamics of small-world networks, Nature 393, 440-442 (1998).
    args:
        n: 节点数
        k: 每个节点的平均度（为偶数，设定为奇数这变成向下的偶数）
        p: 每个节点断边重连的概率
    """
    # 创建一个环形网络
    matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            matrix[i, (i + j) % n] = 1
            matrix[i, (i - j) % n] = 1

    # 重连部分连接
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if torch.rand(1).item() < p:
                new_neighbor = torch.randint(0, n, (1,)).item()
                while new_neighbor == i or matrix[i, new_neighbor] == 1:
                    new_neighbor = torch.randint(0, n, (1,)).item()
                matrix[i, (i + j) % n] = 0
                matrix[(i + j) % n, i] = 0
                matrix[i, new_neighbor] = 1
                matrix[new_neighbor, i] = 1

    return matrix

def small_world_single(n=10, k=2, p=0.1):
    """
    创建一个单向连接的小世界网络
    reference: D.J. Watts, S.H. Strogatz, Collective dynamics of small-world networks, Nature 393, 440-442 (1998).
    args:
        n (int): 网络中节点的数量
        k (int): 每个节点的出度(总度为2k，出度+入度)
        p (float): 重连概率
    Returns:
        matrix: 表示连接关系的矩阵
    """

    # 创建一个 n x n 的零矩阵
    matrix = torch.zeros((n, n))

    # 将每个节点的出度设置为 k
    out_degree = k

    # 对每个节点进行连接
    for i in range(n):
        # 计算节点 i 右边的 k//2 个节点的索引
        right_neighbors = [(i + j + 1) % n for j in range(out_degree)]
        # 将这些节点与节点 i 进行连接
        matrix[i, right_neighbors] = 1

    # 随机重连节点
    for i in range(n):
        for j in range(out_degree):
            if torch.rand(1).item() < p:
                # 随机选择一个节点进行重连
                new_neighbor = torch.randint(0, n, (1,)).item()
                while new_neighbor == i or matrix[i, new_neighbor] == 1:
                    new_neighbor = torch.randint(0, n, (1,)).item()

                # 断开原来的连接，建立新的连接
                matrix[i, (i + j + 1) % n] = 0
                matrix[i, new_neighbor] = 1

    return matrix


# ======================== 无标度网络 ========================
def scale_network(Num=100, N_init=5, add=2):
    """
    创建一个无标度网络
    reference: A.-L. Barabási, R. Albert, Emergence of scaling in random networks, Science 286(5439), 509-512 (1999).
    args:
        Num: 网络总的节点数;
        N_init: 网络的初始节点个数;
        add: 新点与旧网络连边的数目
    """
    # Num, N_init, add = int(Num), int(N_init), int(N_init)
    matrix = torch.zeros((Num, Num))
    degree = torch.zeros(Num)
    # 开始全连接
    matrix[:N_init, :N_init] = 1
    matrix[:N_init, :N_init] = matrix[:N_init, :N_init] - torch.eye(N_init)

    Js = N_init
    while  Js<Num:
        # 计算新节点与已有节点的连接概率
        prob1 = torch.sum(matrix[:Js, :Js], axis=1) / matrix[:Js, :Js].sum()
        # 根据概率连接新节点
        targets = torch.multinomial(prob1, add, replacement=False)
        matrix[Js, targets] = 1
        matrix[targets, Js] = 1

        Js += 1

    return matrix


def ERnet(num_nodes, edge_probability):
    """
    生成不对称的ER随机网络
    reference: P. Erdős,  A. Rényi, On random graphs I, Publicationes Mathematicae (Debrecen), 6, 290-297 (1959).
    Args:
        num_nodes (int): 网络中的节点数
        edge_probability (float): 从节点i到节点j存在连接的概率（不一定对称）
    Returns:
        matrix (Tensor): 不对称ER随机网络的邻接矩阵，1表示存在连接，0表示无连接
    """
    matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < edge_probability:
                matrix[i, j] = 1

    return matrix


if __name__ == "__main__":
    Num = int(100)
    N_init = 5
    add = 2
    scale_network(Num, N_init, add)
