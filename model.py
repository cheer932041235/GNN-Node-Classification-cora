"""
GCN模型定义
功能：实现图卷积网络（Graph Convolutional Network）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    图卷积网络模型
    
    架构说明：
    - 输入层 -> GCN层1 -> ReLU -> Dropout -> GCN层2 -> 输出
    
    核心思想：
    每个节点通过聚合邻居节点的特征来更新自己的表示
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        """
        初始化GCN模型
        
        Args:
            num_features: 输入特征维度（Cora中是1433）
            num_classes: 输出类别数（Cora中是7）
            hidden_dim: 隐藏层维度
            dropout: Dropout比率，用于防止过拟合
        """
        super(GCN, self).__init__()
        
        # 第一层图卷积：输入特征 -> 隐藏层
        self.conv1 = GCNConv(num_features, hidden_dim)
        
        # 第二层图卷积：隐藏层 -> 输出类别
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
        self.dropout = dropout
        
        print("=" * 50)
        print("GCN模型结构：")
        print(f"  输入维度: {num_features}")
        print(f"  隐藏层维度: {hidden_dim}")
        print(f"  输出类别数: {num_classes}")
        print(f"  Dropout率: {dropout}")
        print("=" * 50)
        
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyG的Data对象，包含：
                - x: 节点特征矩阵 [num_nodes, num_features]
                - edge_index: 边索引 [2, num_edges]
                
        Returns:
            输出: 每个节点属于各个类别的logits [num_nodes, num_classes]
        """
        x, edge_index = data.x, data.edge_index
        
        # 第一层GCN + ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Dropout层（训练时随机丢弃部分神经元）
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        # 返回对数概率（用于交叉熵损失）
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, data):
        """
        获取节点的嵌入表示（用于可视化）
        
        Args:
            data: PyG的Data对象
            
        Returns:
            节点嵌入向量 [num_nodes, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # 只通过第一层GCN获取嵌入
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        return x


class DeepGCN(nn.Module):
    """
    更深的GCN模型（3层）
    适合想要尝试更复杂模型的场景
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        super(DeepGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        
        self.dropout = dropout
        
        print("=" * 50)
        print("Deep GCN模型结构（3层）：")
        print(f"  输入维度: {num_features}")
        print(f"  隐藏层维度: {hidden_dim} -> {hidden_dim}")
        print(f"  输出类别数: {num_classes}")
        print(f"  Dropout率: {dropout}")
        print("=" * 50)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # 测试模型
    print("测试GCN模型...")
    
    # 创建一个简单的模型
    model = GCN(num_features=1433, num_classes=7, hidden_dim=64)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {total_params:,}")
    
    print("\n模型定义测试完成！")

