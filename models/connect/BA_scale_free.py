# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/29
# User      : WuY
# File      : BA_scale_free.py
# barabasi_albert无标度网络模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 创建一个BA无标度网络网络
def create_sf(n=100, n_init=5, n_add=2):
    """
    n-网络的总节点数
    n_init-网络的初始节点数
    n_add-新节点与旧网络连接的数目
    """
    matrix = np.zeros((n, n))  # 初始化邻接矩阵
    # 开始全连接初始网络
    matrix[:n_init, :n_init] = 1
    matrix[:n_init, :n_init] -= np.eye(n_init)  # 移除自连接

    Js = n_init
    while Js < n:
        # 计算新节点与已有节点的连接概率
        prob1 = np.sum(matrix[:Js, :Js], axis=1) / np.sum(matrix[:Js, :Js])
        # 根据概率选择连接的目标节点
        targets = np.random.choice(Js, size=n_add, replace=False, p=prob1)
        matrix[Js, targets] = 1
        matrix[targets, Js] = 1

        Js += 1

    return matrix


# ============= 使用 networkx 创建网络，并计算其性质 =============
# 创建一个BA无标度网络网络
class scale_free:
    """
    n-网络的总节点数
    n_init-网络的初始节点数
    n_add-新节点与旧网络连接的数目
    """
    def __init__(self, n=100, n_init=5, n_add=2):
        self.n = n
        self.n_init = n_init
        self.n_add = n_add

    def create_sf_network(self):
        # 创建一个完全图作为初始图
        initial_graph = nx.complete_graph(self.n_init)

        # 使用barabasi_albert_graph方法，从初始图开始生成无标度网络
        self.G = nx.barabasi_albert_graph(self.n, self.n_add, initial_graph=initial_graph)

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

    def draw_network(self, with_labels=False):
        # 获取网络中每个节点的度
        degrees = dict(self.G.degree())

        # 调整节点大小的系数，使得度大的节点更加突出
        node_sizes = [v * 8 for v in degrees.values()]  # 增大系数以改变节点大小

        # 使用度数决定节点颜色的深浅，同时选用鲜明的颜色映射以突出高度节点
        max_degree = max(degrees.values())
        node_colors = [(degree / max_degree) for degree in degrees.values()]

        # 使用spring布局
        # pos = nx.spring_layout(self.G, k=0.05, iterations=30, seed=42)
        pos = nx.kamada_kawai_layout(self.G)
        # 绘制网络图
        plt.figure(figsize=(8, 8))

        # 绘制具有弧度的连线
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, edge_color='grey',
                           width=.6, arrows=False)

        # 绘制节点，使用viridis颜色映射并根据节点度数调整颜色和大小
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis)

        # 如果需要，添加节点标签
        if with_labels:
            nx.draw_networkx_labels(self.G, pos, font_size=8, font_color='black')

        plt.title("Scale-Free Network")
        plt.axis('off')
        plt.show()

    def fit_power_law_distribution(self, plot=False):
        # 计算度分布
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        degree_count = np.array(np.unique(degrees, return_counts=True)).T
        k = degree_count[:, 0]  # Degree
        Pk = degree_count[:, 1] / sum(degree_count[:, 1])  # Degree distribution

        # 定义幂律函数
        def power_law(k, A, gamma):
            return A * np.power(k, -gamma)

        # 拟合幂律分布参数
        params, cov = curve_fit(power_law, k, Pk, maxfev=2000)

        # 输出拟合结果
        print(f"拟合结果: A = {params[0]:.4f}, γ = {params[1]:.4f}")
        print(f"协方差矩阵: {cov}")

        if plot:
            # 绘制原始度分布和拟合的幂律分布
            plt.figure(figsize=(8, 5))
            plt.scatter(k, Pk, color='red', label='Original Data')
            plt.plot(k, power_law(k, *params), label=f'Fitted Power Law (γ = {params[1]:.2f})')
            plt.xlabel('Degree (k)')
            plt.ylabel('P(k)')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.title('Degree Distribution with Power Law Fit')
            plt.show()

        # 返回拟合的参数和协方差矩阵
        return params, cov


if __name__ == "__main__":
    Num = int(100)
    # conn = create_sw(Num, 4, 0.5)
    conn = create_sf(Num, 5, 2)
    print(conn.sum()/Num)

