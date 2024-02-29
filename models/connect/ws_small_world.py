# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/29
# User      : WuY
# File      : small-world.py
# 小世界网络模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个小世界网络的连接矩阵(双向)
def create_sw(n, k, p):
    # 创建一个环形网络
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            matrix[i, (i + j) % n] = 1
            matrix[i, (i - j) % n] = 1

    # 重连部分连接
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if np.random.rand() < p:
                # 找到一个合适的新邻居，确保不是自己，也不是当前的邻居
                available = [x for x in range(n) if x != i and matrix[i, x] == 0]
                if available:  # 确保还有可用的节点进行重连
                    new_neighbor = np.random.choice(available)
                    matrix[i, (i + j) % n] = 0
                    matrix[(i + j) % n, i] = 0
                    matrix[i, new_neighbor] = 1
                    matrix[new_neighbor, i] = 1

    return matrix


# 创建一个单向连接的小世界网络
def create_Disw(n, k, p):
    """
    创建一个单向连接的小世界网络

    Parameters:
        n (int): 网络中节点的数量
        k (int): 每个节点的出度
        p (float): 重连概率

    Returns:
        numpy.ndarray: 表示连接关系的矩阵
    """
    # 创建一个 n x n 的零矩阵
    matrix = np.zeros((n, n), dtype=int)

    # 对每个节点进行连接
    for i in range(n):
        for j in range(1, k + 1):
            matrix[i, (i + j) % n] = 1

    # 随机重连节点
    for i in range(n):
        for j in range(1, k + 1):
            if np.random.rand() < p:
                # 获取当前已连接的邻居
                current_neighbors = np.where(matrix[i] == 1)[0]
                # 生成可用于重连的非邻居列表
                available = [x for x in range(n) if x not in current_neighbors and x != i]
                if available:  # 如果有可用的非邻居
                    # 随机选择一个新邻居
                    new_neighbor = np.random.choice(available)
                    # 断开原来的连接，选择第一个连接断开可能不够精确，这里应该是断开一个随机的已有连接
                    old_neighbor = (i + j) % n
                    if matrix[i, old_neighbor] == 1:  # 确保这是一个有效的连接
                        matrix[i, old_neighbor] = 0
                    # 建立新的连接
                    matrix[i, new_neighbor] = 1

    return matrix


# ============= 使用 networkx 创建网络，并计算其性质 =============
# 创建一个小世界网络的连接矩阵(双向)
class Small_World:
    def __init__(self, n, k, p):
        """
        初始化小世界网络模型。

        Parameters:
            n (int): 网络中节点的数量。
            k (int): 每个节点的近邻数（假设k为偶数）。
            p (float): 重连概率。
        """
        self.n = n
        self.k = k
        self.p = p
        # self.create_sw_network()

    def create_sw_network(self):
        # 创建小世界网络
        self.G = nx.watts_strogatz_graph(self.n, self.k, self.p)
        # 将网络转换为邻接矩阵，并返回其NumPy数组
        adj_matrix = nx.to_numpy_array(self.G, dtype=int)
        return adj_matrix

    def degree_distribution(self, plot=False):
        """
        计算网络的度分布，并可选地绘制度分布图。

        Parameters:
            plot (bool): 是否绘制度分布图的开关。

        Returns:
            degree_distribution (dict): 键为度，值为对应该度的节点比例。
        """
        # 计算所有节点的度
        degrees = [degree for node, degree in self.G.degree()]
        # 获取唯一度值并计算每个度的频率
        unique_degrees = set(degrees)
        degree_distribution = {degree: degrees.count(degree) / float(self.n) for degree in unique_degrees}

        if plot:
            # 绘制度分布图
            degrees, counts = zip(*degree_distribution.items())
            plt.figure(figsize=(8, 5))
            plt.bar(degrees, counts, color='b')
            plt.title("Degree Distribution")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.yscale('log')  # 对于大型网络或度分布范围广的网络，使用对数尺度可能更合适
            plt.show()

        return degree_distribution

    def average_path_length(self):
        """
        计算网络的平均路径长度。

        Returns:
            L (float): 网络的平均路径长度。
        """
        L = nx.average_shortest_path_length(self.G)
        return L

    def clustering_coefficient(self):
        """
        计算网络的平均聚类系数。

        Returns:
            C (float): 网络的平均聚类系数。
        """
        C = nx.average_clustering(self.G)
        return C

    def average_degree(self):
        """
        计算网络的平均度。

        Returns:
            avg_degree (float): 网络的平均度。
        """
        total_degrees = sum(deg for n, deg in self.G.degree())
        avg_degree = total_degrees / float(self.n)
        return avg_degree

    def draw_network(self, node_size_multiplier=100):
        """
        绘制网络图，节点大小根据度来调整。

        Parameters:
            node_size_multiplier (int): 节点大小的乘数因子，用于调整节点的显示大小。
        """
        # 计算每个节点的度
        degrees = dict(self.G.degree())
        # 将度映射到大小
        node_sizes = [degrees[node] * node_size_multiplier for node in self.G.nodes()]

        plt.figure(figsize=(12, 12))
        # 使用圆形布局，并根据节点的度来调整节点的大小
        nx.draw_circular(self.G, node_size=node_sizes, with_labels=True, font_size=8)
        plt.title("Small World Network")
        plt.show()


# 创建一个单向连接的小世界网络
class DiSmall_World:
    def __init__(self, n, k, p):
        """
        初始化有向小世界网络模型。

        Parameters:
            n (int): 网络中节点的数量。
            k (int): 每个节点的出度（总连接数是k的两倍，因为考虑入度和出度）。
            p (float): 重连概率。
        """
        self.n = n
        self.k = k
        self.p = p
        # self.create_disw_network()

    def create_disw_network(self):
        """
        创建并返回有向小世界网络。
        """
        self.G = nx.DiGraph()
        # 添加环形局部连接
        for i in range(self.n):
            for j in range(1, self.k + 1):
                self.G.add_edge(i, (i + j) % self.n)
        # 随机重连节点
        for i in range(self.n):
            for j in range(1, self.k + 1):
                if np.random.rand() < self.p:
                    while True:
                        new_target = np.random.randint(self.n)
                        # 避免自环和重复的边
                        if new_target != i and not self.G.has_edge(i, new_target):
                            # 移除原有的边
                            self.G.remove_edge(i, (i + j) % self.n)
                            # 添加新的边
                            self.G.add_edge(i, new_target)
                            break

        adj_matrix = nx.to_numpy_array(self.G, dtype=int)

        return adj_matrix

    def degree_distribution(self, plot=False, degree_type='both'):
        """
        计算并可选地绘制网络的度分布（入度、出度或总度）。

        Parameters:
            plot (bool): 是否绘制度分布图的开关。
            degree_type (str): 'in'表示入度，'out'表示出度，'both'表示总度。
        """
        if degree_type == 'in':
            degrees = [deg for node, deg in self.G.in_degree()]
        elif degree_type == 'out':
            degrees = [deg for node, deg in self.G.out_degree()]
        else:  # both
            degrees = [deg for node, deg in self.G.degree()]

        # 获取唯一度值并计算每个度的频率
        unique_degrees = set(degrees)
        degree_distribution = {degree: degrees.count(degree) / float(self.n) for degree in unique_degrees}

        if plot:
            degrees, counts = zip(*sorted(degree_distribution.items()))
            plt.figure(figsize=(8, 5))
            plt.bar(degrees, counts, color='b')
            plt.title(f"{degree_type.capitalize()} Degree Distribution")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.yscale('log')
            plt.show()

        return degree_distribution

    def average_path_length(self):
        """
        计算网络最大强连通分量的平均路径长度。

        Returns:
            L (float): 网络最大强连通分量的平均路径长度。
        """
        # 找到最大强连通分量
        largest_cc = max(nx.strongly_connected_components(self.G), key=len)
        subgraph = self.G.subgraph(largest_cc).copy()
        # 计算最大强连通分量的平均路径长度
        L = nx.average_shortest_path_length(subgraph)
        return L

    def clustering_coefficient(self):
        """
        计算网络的平均聚类系数。

        Returns:
            C (float): 网络的平均聚类系数。
        """
        C = nx.average_clustering(self.G)
        return C

    def average_degree(self, degree_type='both'):
        """
        计算网络的平均度，可以选择计算入度、出度或总度。

        Parameters:
            degree_type (str): 'in'表示入度，'out'表示出度，'both'表示总度（默认）。

        Returns:
            avg_degree (float): 指定类型的网络平均度。
        """
        if degree_type == 'in':
            # 计算入度的平均值
            total_degrees = sum(deg for n, deg in self.G.in_degree())
        elif degree_type == 'out':
            # 计算出度的平均值
            total_degrees = sum(deg for n, deg in self.G.out_degree())
        else:
            # 计算总度（入度+出度）的平均值
            total_degrees = sum(deg for n, deg in self.G.degree())

        avg_degree = total_degrees / float(self.n)
        return avg_degree

    def draw_network(self, node_size_multiplier=100, arrow_size=20):
        """
        绘制网络图，节点大小根据度来调整，并可调整箭头大小。

        Parameters:
            node_size_multiplier (int): 节点大小的乘数因子，用于调整节点的显示大小。
            arrow_size (int): 箭头的大小。
        """
        # 计算每个节点的度
        degrees = dict(self.G.degree())
        # 将度映射到大小
        node_sizes = [degrees[node] * node_size_multiplier for node in self.G.nodes()]

        # 使用圆形布局
        pos = nx.circular_layout(self.G)

        plt.figure(figsize=(12, 12))
        # 绘制网络，包括节点和边，调整箭头大小
        nx.draw_networkx(self.G, pos, node_size=node_sizes, with_labels=True,
                         arrows=True, arrowsize=arrow_size, font_size=8)

        plt.title("Small World Network")
        plt.axis('off')  # 关闭坐标轴
        plt.show()


if __name__ == "__main__":
    Num = int(100)
    # conn = create_sw(Num, 4, 0.5)
    conn = create_Disw(Num, 3, 0.4)
    print(conn)
