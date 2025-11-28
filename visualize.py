"""
可视化模块
功能：可视化节点嵌入、预测结果等
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

from data_loader import CoraDataLoader
from model import GCN

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, model, data):
        """
        初始化可视化器
        
        Args:
            model: 训练好的GCN模型
            data: 图数据
        """
        self.model = model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.model.eval()
        
    def get_embeddings(self):
        """获取节点嵌入"""
        with torch.no_grad():
            embeddings = self.model.get_embeddings(self.data)
        return embeddings.cpu().numpy()
    
    def visualize_embeddings_tsne(self, save_path='outputs/images/embeddings_tsne.png'):
        """
        使用t-SNE降维可视化节点嵌入
        
        Args:
            save_path: 保存路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在使用t-SNE降维可视化节点嵌入...")
        print("(这可能需要一些时间...)")
        
        # 获取嵌入
        embeddings = self.get_embeddings()
        labels = self.data.y.cpu().numpy()
        
        # t-SNE降维到2维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        
        # 为每个类别使用不同的颜色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], 
                       embeddings_2d[mask, 1],
                       c=[color],
                       label=f'类别 {label}',
                       alpha=0.6,
                       s=20)
        
        plt.title('节点嵌入t-SNE可视化', fontsize=16)
        plt.xlabel('维度 1', fontsize=12)
        plt.ylabel('维度 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE可视化已保存到: {save_path}")
        plt.close()
        
    def visualize_embeddings_pca(self, save_path='outputs/images/embeddings_pca.png'):
        """
        使用PCA降维可视化节点嵌入
        
        Args:
            save_path: 保存路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在使用PCA降维可视化节点嵌入...")
        
        # 获取嵌入
        embeddings = self.get_embeddings()
        labels = self.data.y.cpu().numpy()
        
        # PCA降维到2维
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], 
                       embeddings_2d[mask, 1],
                       c=[color],
                       label=f'类别 {label}',
                       alpha=0.6,
                       s=20)
        
        plt.title(f'节点嵌入PCA可视化\n(解释方差: {pca.explained_variance_ratio_.sum():.2%})', 
                 fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA可视化已保存到: {save_path}")
        plt.close()
        
    def visualize_predictions(self, save_path='outputs/images/predictions.png'):
        """
        可视化预测结果（对比真实标签）
        
        Args:
            save_path: 保存路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在生成预测结果可视化...")
        
        # 获取预测
        with torch.no_grad():
            out = self.model(self.data)
            pred = out.max(dim=1)[1].cpu().numpy()
        
        true_labels = self.data.y.cpu().numpy()
        
        # 使用PCA降维
        embeddings = self.get_embeddings()
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 真实标签
        for label in np.unique(true_labels):
            mask = true_labels == label
            axes[0].scatter(embeddings_2d[mask, 0], 
                          embeddings_2d[mask, 1],
                          label=f'类别 {label}',
                          alpha=0.6,
                          s=20)
        axes[0].set_title('真实标签', fontsize=14)
        axes[0].legend()
        
        # 2. 预测标签
        for label in np.unique(pred):
            mask = pred == label
            axes[1].scatter(embeddings_2d[mask, 0], 
                          embeddings_2d[mask, 1],
                          label=f'类别 {label}',
                          alpha=0.6,
                          s=20)
        axes[1].set_title('预测标签', fontsize=14)
        axes[1].legend()
        
        # 3. 预测错误的节点
        correct = pred == true_labels
        axes[2].scatter(embeddings_2d[correct, 0], 
                      embeddings_2d[correct, 1],
                      c='green',
                      label='预测正确',
                      alpha=0.3,
                      s=20)
        axes[2].scatter(embeddings_2d[~correct, 0], 
                      embeddings_2d[~correct, 1],
                      c='red',
                      label='预测错误',
                      alpha=0.8,
                      s=30,
                      marker='x')
        axes[2].set_title(f'预测准确率: {correct.sum() / len(correct):.2%}', fontsize=14)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果可视化已保存到: {save_path}")
        plt.close()
        
    def plot_confusion_matrix(self, save_path='outputs/images/confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 保存路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在生成混淆矩阵...")
        
        from sklearn.metrics import confusion_matrix
        
        # 获取预测
        with torch.no_grad():
            out = self.model(self.data)
            pred = out.max(dim=1)[1].cpu().numpy()
        
        true_labels = self.data.y.cpu().numpy()
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred)
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=range(cm.shape[1]),
                   yticklabels=range(cm.shape[0]))
        plt.title('混淆矩阵（归一化）', fontsize=14)
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        plt.close()


def main():
    """主函数"""
    print("=" * 50)
    print("开始可视化分析...")
    print("=" * 50)
    
    # 1. 加载数据
    loader = CoraDataLoader()
    data, num_classes = loader.get_data()
    
    # 2. 加载训练好的模型
    model = GCN(
        num_features=data.num_node_features,
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.5
    )
    
    try:
        model.load_state_dict(torch.load('outputs/models/best_model.pth'))
        print("已加载outputs/models/best_model.pth")
    except:
        try:
            model.load_state_dict(torch.load('outputs/models/final_model.pth'))
            print("已加载outputs/models/final_model.pth")
        except:
            print("未找到训练好的模型！请先运行train.py训练模型。")
            return
    
    # 3. 创建可视化器
    visualizer = Visualizer(model, data)
    
    # 4. 生成各种可视化
    visualizer.visualize_embeddings_pca()
    visualizer.visualize_embeddings_tsne()  # 这个会比较慢
    visualizer.visualize_predictions()
    visualizer.plot_confusion_matrix()
    
    print("\n" + "=" * 50)
    print("所有可视化完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()

