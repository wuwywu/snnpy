# encoding: utf-8
# Author    : HuangWF
# Datetime  : 2024/3/11
# User      : HuangWF
# File      : ER_random.py
# Erdős-Rényi（E-R）模型


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# 随机网络，G(n, p)模型  概率p连接一条边, 单向
def create_ER_p(n, p):
    """
    n-网络的总节点数
    p-网络中节点间连接的概率
    """
    random_matrix = np.random.rand(n, n)
    connection_matrix = (random_matrix < p).astype(int)
    np.fill_diagonal(connection_matrix, 0)  #不考虑自连接
    return connection_matrix


# 随机网络，G(n, M)模型  网络固定有M条边，单向
def create_ER_M(n, M):
    """
    n-网络的总节点数
    M-网络的总边数
    """
    # 确保M不大于n*(n-1)/2，即无向图的最大边数（不含自环）
    max_edges = n * (n - 1) // 2   
    if M > max_edges:
        raise ValueError(f"M should be less than or equal to {max_edges} for n={n}")
    
    # 初始化一个全零矩阵
    connection_matrix = np.zeros((n, n), dtype=int)
    # 生成所有可能的边，不包括自环
    possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    # 随机选择M条边
    chosen_edges = np.random.choice(len(possible_edges), size=M, replace=False)
    
    # 在connection_matrix中标记选中的边
    for edge_index in chosen_edges:
        i, j = possible_edges[edge_index]
        connection_matrix[i, j] = 1
    
    return connection_matrix


# ============= 使用 networkx 创建网络，并计算其性质 =============
# 创建一个ER随机网络的连接矩阵(双向)
class Erdos_Renyi:
    """
    n-网络的总节点数
    p-网络中节点间连接的概率
    M-网络的总边数

    注意：p, M 只选择一个
    """
    def __init__(self, n, p=None, M=None):
        self.n = n
        self.p = p
        self.M = M

    def creat_ER_network(self):
        if self.M is not None:
            self.G =nx.gnm_random_graph(self.n, self.M, directed=False)
        elif self.p is not None:
            self.G =nx.erdos_renyi_graph(self.n, self.p, directed=False)

        if not nx.is_connected(self.G):
            print("网络不是连通的")

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

    def draw_network(self, node_size_multiplier=10):
        """
        绘制网络图，节点大小和颜色根据度来调整，以更好地体现ER随机图网络的特点。

        Parameters:
            node_size_multiplier (int): 节点大小的乘数因子，用于调整节点的显示大小。
        """
        # 计算每个节点的度
        degrees = dict(self.G.degree())
        # 将度映射到大小
        node_sizes = [degrees[node] * node_size_multiplier for node in self.G.nodes()]
        # 将度映射到颜色，使用颜色映射
        max_degree = max(degrees.values())
        node_colors = [degrees[node] / max_degree for node in self.G.nodes()]

        # 使用Spring布局，这种布局能较好地体现随机图的特性
        # pos = nx.spring_layout(self.G)
        pos = nx.kamada_kawai_layout(self.G)

        plt.figure(figsize=(8, 8))
        # 绘制网络，调整节点的大小和颜色
        nx.draw(self.G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis,
                with_labels=False, edge_color='grey')
        plt.title("Erdős-Rényi Random Graph")
        plt.show()


# 创建一个ER随机网络的连接矩阵(单向)
class DiErdos_Renyi:
    """
    n-网络的总节点数
    p-网络中节点间连接的概率
    M-网络的总边数

    注意：p, M 只选择一个
    """
    def __init__(self, n, p=None, M=None):
        self.n = n
        self.p = p
        self.M = M

    def creat_ER_network(self):
        if self.M is not None:
            self.G =nx.gnm_random_graph(self.n, self.M, directed=True)
        elif self.p is not None:
            self.G =nx.erdos_renyi_graph(self.n, self.p, directed=True)

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

    def draw_network(self, node_size_multiplier=10, arrow_size=10):
        """
        绘制网络图，节点大小和颜色根据度来调整，以更好地体现ER随机图网络的特点。

        Parameters:
            node_size_multiplier (int): 节点大小的乘数因子，用于调整节点的显示大小。
            arrow_size (int): 箭头的大小。
        """
        # 计算每个节点的度
        degrees = dict(self.G.degree())
        # 将度映射到大小
        node_sizes = [degrees[node] * node_size_multiplier for node in self.G.nodes()]
        # 将度映射到颜色，使用颜色映射
        max_degree = max(degrees.values())
        node_colors = [degrees[node] / max_degree for node in self.G.nodes()]

        # 使用Spring布局，这种布局能较好地体现随机图的特性
        # pos = nx.spring_layout(self.G)
        pos = nx.kamada_kawai_layout(self.G)

        plt.figure(figsize=(8, 8))
        # 绘制网络，调整节点的大小和颜色
        nx.draw(self.G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis,
                with_labels=False, edge_color='grey', arrows=True, arrowsize=arrow_size)
        plt.title("DiErdős-Rényi Random Graph")
        plt.show()
