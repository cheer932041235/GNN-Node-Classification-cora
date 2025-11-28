"""
训练脚本
功能：训练GCN模型进行节点分类
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data_loader import CoraDataLoader
from model import GCN

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer:
    """GCN训练器"""
    
    def __init__(self, model, data, lr=0.01, weight_decay=5e-4):
        """
        初始化训练器
        
        Args:
            model: GCN模型
            data: 图数据
            lr: 学习率
            weight_decay: 权重衰减（L2正则化）
        """
        self.model = model
        self.data = data
        
        # 使用Adam优化器
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 检查是否有GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 将模型和数据移到设备上
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # 记录训练历史
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.test_accs = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()  # 设置为训练模式
        self.optimizer.zero_grad()  # 清空梯度
        
        # 前向传播
        out = self.model(self.data)
        
        # 只在训练集上计算损失
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.optimizer.step()
        
        # 计算训练集准确率
        pred = out[self.data.train_mask].max(dim=1)[1]
        acc = pred.eq(self.data.y[self.data.train_mask]).sum().item() / self.data.train_mask.sum().item()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, mask):
        """
        在指定的数据集上评估模型
        
        Args:
            mask: 数据掩码（train_mask / val_mask / test_mask）
            
        Returns:
            准确率
        """
        self.model.eval()  # 设置为评估模式
        
        out = self.model(self.data)
        pred = out[mask].max(dim=1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        
        return acc
    
    def train(self, epochs=200, early_stopping=True, patience=20):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            early_stopping: 是否使用早停
            patience: 早停的耐心值（验证集准确率多少轮不提升就停止）
        """
        print("=" * 50)
        print("开始训练...")
        print("=" * 50)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="训练进度"):
            # 训练一个epoch
            loss, train_acc = self.train_epoch()
            
            # 在验证集和测试集上评估
            val_acc = self.evaluate(self.data.val_mask)
            test_acc = self.evaluate(self.data.test_mask)
            
            # 记录历史
            self.train_losses.append(loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.test_accs.append(test_acc)
            
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Loss: {loss:.4f}")
                print(f"  训练集准确率: {train_acc:.4f}")
                print(f"  验证集准确率: {val_acc:.4f}")
                print(f"  测试集准确率: {test_acc:.4f}")
            
            # 早停检查
            if early_stopping:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # 保存最佳模型（确保目录存在）
                    os.makedirs('outputs/models', exist_ok=True)
                    torch.save(self.model.state_dict(), 'outputs/models/best_model.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"\n早停触发！验证集准确率已经{patience}轮没有提升。")
                    print(f"最佳验证集准确率: {best_val_acc:.4f}")
                    # 加载最佳模型
                    self.model.load_state_dict(torch.load('outputs/models/best_model.pth'))
                    break
        
        # 最终评估
        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        
        final_train_acc = self.evaluate(self.data.train_mask)
        final_val_acc = self.evaluate(self.data.val_mask)
        final_test_acc = self.evaluate(self.data.test_mask)
        
        print(f"最终训练集准确率: {final_train_acc:.4f}")
        print(f"最终验证集准确率: {final_val_acc:.4f}")
        print(f"最终测试集准确率: {final_test_acc:.4f}")
        print("=" * 50)
        
    def plot_training_history(self, save_path='outputs/images/training_history.png'):
        """绘制训练历史曲线"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("\n正在生成训练历史曲线...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='训练损失', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('损失', fontsize=12)
        axes[0].set_title('训练损失曲线', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(self.train_accs, label='训练集', linewidth=2)
        axes[1].plot(self.val_accs, label='验证集', linewidth=2)
        axes[1].plot(self.test_accs, label='测试集', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('准确率', fontsize=12)
        axes[1].set_title('准确率曲线', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史曲线已保存到: {save_path}")
        plt.close()
    
    def get_predictions(self):
        """获取模型的预测结果"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            pred = out.max(dim=1)[1]
        return pred.cpu().numpy()


def main():
    """主函数"""
    # 1. 加载数据
    loader = CoraDataLoader()
    data, num_classes = loader.get_data()
    
    # 2. 创建模型
    model = GCN(
        num_features=data.num_node_features,
        num_classes=num_classes,
        hidden_dim=64,  # 隐藏层维度
        dropout=0.5     # Dropout率
    )
    
    # 3. 创建训练器
    trainer = Trainer(
        model=model,
        data=data,
        lr=0.01,           # 学习率
        weight_decay=5e-4  # 权重衰减
    )
    
    # 4. 训练模型
    trainer.train(
        epochs=200,         # 训练200轮
        early_stopping=True, # 使用早停
        patience=20         # 20轮不提升就停止
    )
    
    # 5. 绘制训练历史
    trainer.plot_training_history()
    
    # 6. 保存最终模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/final_model.pth')
    print("\n模型已保存到: outputs/models/final_model.pth")


if __name__ == '__main__':
    main()

