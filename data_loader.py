"""
数据加载模块
功能：加载Cora数据集并提供可视化功能
"""

import os
import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class CoraDataLoader:
    """Cora数据集加载器"""
    
    def __init__(self, root='./data'):
        """
        初始化数据加载器
        Args:
            root: 数据存储路径
        """
        print("=" * 50)
        print("正在加载Cora数据集...")
        
        # 加载Cora数据集（会自动下载）
        self.dataset = Planetoid(root=root, name='Cora')
        self.data = self.dataset[0]  # 获取图数据
        
        print(f"Dataset loaded successfully!")
        self.print_dataset_info()
        
    def print_dataset_info(self):
        """打印数据集的基本信息"""
        print("=" * 50)
        print("Cora Dataset Information:")
        print(f"  - 节点数量: {self.data.num_nodes}")
        print(f"  - 边数量: {self.data.num_edges}")
        print(f"  - 节点特征维度: {self.data.num_node_features}")
        print(f"  - 类别数量: {self.dataset.num_classes}")
        print(f"  - 训练集节点数: {self.data.train_mask.sum().item()}")
        print(f"  - 验证集节点数: {self.data.val_mask.sum().item()}")
        print(f"  - 测试集节点数: {self.data.test_mask.sum().item()}")
        print(f"  - 图是否有孤立节点: {'是' if self.data.has_isolated_nodes() else '否'}")
        print(f"  - 图是否有自环: {'是' if self.data.has_self_loops() else '否'}")
        print("=" * 50)
        
    def visualize_graph(self, num_nodes=500, save_path='outputs/images/graph_structure.png'):
        """
        可视化图结构（只显示部分节点，避免过于拥挤）
        Args:
            num_nodes: 要显示的节点数量
            save_path: 保存图片的路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\n正在生成图结构可视化（显示前{num_nodes}个节点）...")
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 先添加所有节点（确保节点ID连续）
        G.add_nodes_from(range(num_nodes))
        
        # 获取边的列表
        edge_index = self.data.edge_index.numpy()
        
        # 只添加前num_nodes个节点相关的边
        edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < num_nodes and dst < num_nodes:
                edges.append((src, dst))
        
        G.add_edges_from(edges)
        
        # 获取节点标签（类别）
        labels = self.data.y.numpy()[:num_nodes]
        
        # 设置图的大小
        plt.figure(figsize=(12, 10))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制节点，根据类别着色
        nx.draw_networkx_nodes(G, pos, 
                              node_color=labels,
                              node_size=50,
                              cmap='tab10',
                              alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        
        plt.title(f'Cora数据集图结构可视化（前{num_nodes}个节点）', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图结构已保存到: {save_path}")
        plt.close()
        
    def visualize_label_distribution(self, save_path='outputs/images/label_distribution.png'):
        """
        可视化标签分布
        Args:
            save_path: 保存图片的路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在生成标签分布图...")
        
        labels = self.data.y.numpy()
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts, color='steelblue', alpha=0.8)
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('节点数量', fontsize=12)
        plt.title('Cora数据集标签分布', fontsize=14)
        plt.xticks(unique_labels)
        plt.grid(axis='y', alpha=0.3)
        
        # 在柱子上显示数值
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            plt.text(label, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"标签分布图已保存到: {save_path}")
        plt.close()
        
    def get_data(self):
        """返回数据对象"""
        return self.data, self.dataset.num_classes


if __name__ == '__main__':
    # 测试数据加载器
    loader = CoraDataLoader()
    
    # 可视化图结构
    loader.visualize_graph(num_nodes=300)
    
    # 可视化标签分布
    loader.visualize_label_distribution()
    
    print("\n数据加载模块测试完成！")

